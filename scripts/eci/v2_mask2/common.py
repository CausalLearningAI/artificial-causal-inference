"""
Shared frame / background / keypoint access for the ECI v2 thin-part mask test (scripts/eci/v2_mask2/).

Sources (each gives canonical encoder-input frames x (B, 3, S, S) uint8 on the GPU, the frozen cue's pixels, the
per-video median background and, for the public benchmarks only, keypoints used ONLY to score):
    lab      mice v1 / ants v2 / ants v3 via src/eci/v2/pipeline.py Series (frozen v2 decode, rotation, dark cue).
    calms21  CalMS21 task 1 videos (1024 x 570 top view; resident black, intruder WHITE), padded to a 1024 x 1024 canvas
             as the earlier CalMS21 encode (227 rows above), keypoints (x, y) native px, 7 parts per mouse.
    mabe22   MABe22 mouse triplet clips (224 x 224, black mice), keypoints 512-px scale (y, x), 12 parts per mouse,
             video frame = label index + 1.
The frozen ('dark') cue on the benchmarks = the lab mice rule: grey < 60 AND (pix_bg - grey) > 40, pix_bg = per-pixel
0.98 quantile of the grey levels of the background frames (as scripts/eci/fg_background.py).
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from src.eci.v2.mask import grey_of  # noqa: E402

FFMPEG = str(Path.home() / '.conda/envs/ffmpeg611/bin/ffmpeg')
OUT = Path(os.environ.get('V2M2_OUT') or REPO / 'results/vision/eci_v2_mask2')
BEE = Path(os.environ.get('V2M2_BEE') or '/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_mask2')
CFG = {d: json.loads((REPO / f'configs/eci_v2/{d}.json').read_text()) for d in ('mice', 'ants')}
CAL = REPO / 'dataset/benchmark/calms21'
MAB = REPO / 'dataset/benchmark/mabe22'
CAL_PARTS = ['nose', 'ear_l', 'ear_r', 'neck', 'hip_l', 'hip_r', 'tail_base']
MAB_PARTS = ['nose', 'ear_l', 'ear_r', 'neck', 'paw_fl', 'paw_fr', 'center', 'paw_hl', 'paw_hr', 'tail_base',
             'tail_mid', 'tail_tip']  # MABe22 mouse order (tail_base = 9 checked semantically in eci_prep.py)
N_BG = 48
log = lambda s: print(time.strftime('%H:%M:%S'), s, flush=True)  # noqa: E731


def series_cfg(dom, name):
    return next(s for s in CFG[dom]['series'] if s['name'] == name)


def dark_cue(x, pix_bg, dark_abs=60, dark_rel=40):
    g = grey_of(x).float()
    return (g < dark_abs) & ((pix_bg[None] - g) > dark_rel)


def q98(frames):
    """(N, 3, H, W) uint8 -> (H, W) float per-pixel 0.98 quantile of grey (higher interpolation as torch.quantile)."""
    g = grey_of(frames).float()
    return torch.quantile(g.flatten(1).T.contiguous(), 0.98, dim=1).view(g.shape[1:]) if g.numel() < 2**24 else \
        torch.cat([torch.quantile(g[:, y0:y0 + 64].flatten(1).T.contiguous(), 0.98, dim=1).view(-1, g.shape[2])
                   for y0 in range(0, g.shape[1], 64)])


# ------------------------------------------------------------------------------------------------------------ CalMS21
def cal_names(split):
    d = np.load(CAL / f'derived/task1_{split}.npz', allow_pickle=True)
    return list(d['names']), d['lens']


class CalKP:
    def __init__(self, split):
        d = np.load(CAL / f'derived/task1_{split}.npz', allow_pickle=True)
        self.names, self.lens = list(d['names']), d['lens']
        self.st = np.r_[0, np.cumsum(self.lens)]
        self.kp = d['keypoints']  # (T, mouse, xy, 7)

    def get(self, name, i):
        """-> (2 mice, 7, 2) (x, y) on the 1024 x 1024 canvas."""
        j = self.names.index(name)
        k = self.kp[self.st[j] + i].transpose(0, 2, 1).astype(np.float32).copy()
        k[..., 1] += 227
        return k


def cal_decode(name, split, idx, threads=4):
    """frames idx (sorted ints) of a CalMS21 task-1 video -> (n, 3, 1024, 1024) uint8 (padded canvas, ImageNet-mean
    pad colour), one accurate seek per frame (30 fps)."""
    path = CAL / f'videos/task1_videos_mp4/{split}/{name}.mp4'
    out = []
    for i in idx:
        cmd = [FFMPEG, '-nostdin', '-v', 'error', '-threads', str(threads)]
        if i > 0:
            cmd += ['-ss', f'{(i - 0.5) / 30:.6f}']
        cmd += ['-i', str(path), '-frames:v', '1', '-vf', 'pad=1024:1024:0:227:color=0x7C7468', '-f', 'rawvideo',
                '-pix_fmt', 'rgb24', '-']
        p = subprocess.run(cmd, capture_output=True)
        a = np.frombuffer(p.stdout, np.uint8)
        assert a.size == 1024 * 1024 * 3, (name, i, a.size, p.stderr[-300:])
        out.append(a.reshape(1024, 1024, 3))
    return np.stack(out).transpose(0, 3, 1, 2)


# ------------------------------------------------------------------------------------------------------------ MABe22
def mab_decode(set_, clip):
    p = subprocess.run([FFMPEG, '-nostdin', '-v', 'error', '-threads', '1', '-i', str(MAB / f'videos/{set_}/{clip}.avi'),
                        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], check=True, capture_output=True)
    a = np.frombuffer(p.stdout, np.uint8)
    n = a.size // (224 * 224 * 3)
    return a.reshape(n, 224, 224, 3).transpose(0, 3, 1, 2)


def mab_keypoints(set_):
    f = {'mouse_userTrain': 'mouse_user_train.npy', 'mouse_submission': 'mouse_submission_keypoints.npy'}[set_]
    return np.load(MAB / 'raw' / f, allow_pickle=True).item()['sequences']


def mab_kp(seq, i, scale=1.0):
    """label index i -> (3 mice, 12, 2) (x, y) in 224 px x scale; missing (0, 0) -> NaN."""
    k = seq['keypoints'][i].astype(np.float32)  # (3, 12, 2) (y, x) 512 px
    miss = (k == 0).all(-1)
    k = k[..., ::-1] * 224 / 512 * scale
    k[miss] = np.nan
    return k.copy()
