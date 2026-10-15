"""
Shared constants and helpers for the CalMS21 ECI benchmark (scripts/benchmark/calms21/*.py).

Data (READ-ONLY, BeeGFS): dataset/benchmark/calms21 -> /mnt/beegfs/locatgrp/shared/rcadei/benchmark/calms21
    videos/task1_videos_mp4/{train,test}/<seq>.mp4      1024 x 570 top view, 2 mice (resident black, intruder white)
    annotations/task{1,2,3}_*/calms21_task*_{train,test}.json   MARS keypoints (T, mouse, xy, 7 parts), scores, labels
    derived/task1_{train,test}.npz                       compact copy written by inventory.py (keypoints, labels)
Outputs: results/vision/benchmark_calms21/ (gitignored). Token stores: /mnt/beegfs/locatgrp/shared/rcadei/eci_tok/calms21
Frame convention: video frame of label / keypoint index i = i + OFFSET (measured by inventory.py, plan/offset.json).
"""
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
ROOT = Path('/mnt/beegfs/locatgrp/shared/rcadei/benchmark/calms21')
VID = ROOT / 'videos' / 'task1_videos_mp4'
ANN = ROOT / 'annotations'
DER = ROOT / 'derived'
OUT = Path(os.environ.get('CALMS_OUT') or REPO / 'results/vision/benchmark_calms21')
PLAN = OUT / 'plan'
TOK = Path(os.environ.get('CALMS_TOK') or '/mnt/beegfs/locatgrp/shared/rcadei/eci_tok/calms21')
FFMPEG = str(Path.home() / '.conda/envs/ffmpeg611/bin/ffmpeg')
FFPROBE = str(Path.home() / '.conda/envs/ffmpeg611/bin/ffprobe')
W, H = 1024, 570
CANVAS = 1024                     # frames padded to a 1024 x 1024 square, no resize, no crop
PADY = (CANVAS - H) // 2          # 227 rows above, 227 below
PADV = (124, 116, 104)            # pad colour = ImageNet mean (normalises to ~0)
TASK1 = ['attack', 'investigation', 'mount', 'other']
PARTS = ['nose', 'ear_l', 'ear_r', 'neck', 'hip_l', 'hip_r', 'tail_base']
log = lambda s: print(time.strftime('%H:%M:%S'), s, flush=True)  # noqa: E731


def stage_dir():
    d = Path(os.environ.get('LOCAL_DIR') or '')
    if not str(d).startswith('/localhome') and not str(d).startswith(os.environ.get('TMPDIR') or '/nonexistent'):
        raise RuntimeError(f'LOCAL_DIR must be under /localhome or $TMPDIR, got {d!r}')
    d.mkdir(parents=True, exist_ok=True)
    return d


def video_path(split, seq):
    return VID / split / f'{seq}.mp4'


def decode_frames(path, want, size=None, gray=False, threads=2, max_frames=None):
    """Stream-decode a video ONCE (ffmpeg pipe) and keep the frames whose index is in `want` (iterable of ints; None =
    all frames). size = (w, h) to rescale in ffmpeg (bicubic), else native. -> (dict index -> uint8 array, n decoded)."""
    w, h = size or (W, H)
    c = 1 if gray else 3
    cmd = [FFMPEG, '-nostdin', '-v', 'error', '-threads', str(threads), '-i', str(path)]
    if max_frames:
        cmd += ['-frames:v', str(max_frames)]
    if size:
        cmd += ['-vf', f'scale={w}:{h}:flags=bicubic']
    cmd += ['-f', 'rawvideo', '-pix_fmt', 'gray' if gray else 'rgb24', '-']
    nb = w * h * c
    keep = None if want is None else set(int(x) for x in want)
    last = None if keep is None else max(keep, default=-1)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=nb * 4)
    out, n = {}, 0
    try:
        while True:
            if last is not None and n > last:
                break
            b = p.stdout.read(nb)
            if len(b) < nb:
                break
            if keep is None or n in keep:
                a = np.frombuffer(b, np.uint8)
                out[n] = a.reshape(h, w) if gray else a.reshape(h, w, 3)
            n += 1
    finally:
        p.stdout.close()
        p.kill()
        p.wait()
    return out, n


def offset():
    return int(json.loads((PLAN / 'offset.json').read_text())['offset'])
