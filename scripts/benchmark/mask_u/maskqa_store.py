"""
Mask-U test (rules: scripts/benchmark/eci_mask_u.sh) - mask quality of BOTH arms on every eval frame of the token stores
(CPU job; reads the packed per-frame masks _masks.npz of encode_mabe.py / encode_calms.py). Keypoints only measure.
    mabe   18,496 submission eval frames, IPW-weighted (eval_frames.parquet w); keypoints 512 px, (y, x), label-indexed;
           all-(0, 0) keypoints = missing; animals mouse0..2 (keypoint slots, no identity meaning)
    calms  10,000 test eval frames, IPW-weighted; MARS keypoints score > 0.5; resident (black, mouse 0), intruder (white,
           mouse 1)
Per mask: keypoint recall per animal (fraction of the animal's valid keypoints inside a kept cell, weighted mean over
frames), fraction of frames where the animal is missed entirely, kept cells per frame (and fraction of the 32 x 32
grid; CalMS21 also of the video cells), hull precision (fraction of kept cells within 1 cell of some animal's keypoint
convex hull), frames with 0 kept cells. Output: OUT/maskqa/<bench>_store.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
from common import OUT, TOK, hull_cells, log, recall_cells  # noqa: E402


def load_masks(bench, set_='eval'):
    js = sorted((TOK / bench).glob(f'{set_}_*of*.json'))
    bases = sorted(j.stem for j in js if 'smoke' not in j.name)
    Fm = pd.concat([pd.read_parquet(TOK / bench / f'{b}_frames.parquet') for b in bases]).reset_index(drop=True)
    M = {k: np.concatenate([np.unpackbits(np.load(TOK / bench / f'{b}_masks.npz')[k], axis=1)[:, :1024].astype(bool)
                            for b in bases]) for k in ('cur', 'U')}
    return Fm, M


def main():
    bench = sys.argv[1]
    if bench == 'mabe':
        PLAN = REPO / 'results/vision/benchmark_mabe22/eci/plan'
        E = pd.read_parquet(PLAN / 'eval_frames.parquet')
        Fm, M = load_masks('mabe')
        key = 'clip'
        sk = np.load(REPO / 'dataset/benchmark/mabe22/raw/mouse_submission_keypoints.npy', allow_pickle=True).item()['sequences']
        kp = np.stack([np.asarray(sk[c]['keypoints'][i], np.float32) for c, i in zip(Fm['clip'], Fm.i)]) * 224 / 512
        valid = ~(kp == 0).all(-1)
        y, x = np.clip(kp[..., 0], 0, 223.9), np.clip(kp[..., 1], 0, 223.9)
        cells = (y // 7).astype(int) * 32 + (x // 7).astype(int)
        xyc = np.stack([x / 7, y / 7], -1)
        animals = ['mouse0', 'mouse1', 'mouse2']
        ncells = 1024
    else:
        PLAN = REPO / 'results/vision/benchmark_calms21/plan'
        E = pd.read_parquet(PLAN / 'eval_frames.parquet')
        Fm, M = load_masks('calms')
        key = 'video'
        Z = np.load('/mnt/beegfs/locatgrp/shared/rcadei/benchmark/calms21/derived/task1_test.npz')
        names = list(Z['names'])
        st = np.r_[0, np.cumsum(Z['lens'])]
        rows = np.array([st[names.index(v)] + i for v, i in zip(Fm.video, Fm.i)])
        k_, sc = Z['keypoints'][rows], Z['scores'][rows]
        PADY = 227
        x, y = k_[:, :, 0, :], k_[:, :, 1, :] + PADY
        cells = np.clip((y // 32).astype(int), 0, 31) * 32 + np.clip((x // 32).astype(int), 0, 31)
        valid = sc > 0.5
        xyc = np.stack([x / 32, y / 32], -1)
        animals = ['resident_black', 'intruder_white']
        r = np.arange(32)
        ncells = int((((r * 32 + 31) >= PADY) & (r * 32 < PADY + 570)).sum() * 32)
    idx = pd.MultiIndex.from_frame(Fm[[key, 'i']]).get_indexer(pd.MultiIndex.from_frame(E[[key, 'i']]))
    assert (idx >= 0).all() and len(idx) == len(Fm)
    w = np.empty(len(Fm))
    w[idx] = E.w.values
    hull = hull_cells(xyc, valid)
    res = {'frames': int(len(Fm)), 'weights': 'IPW (eval_frames.parquet w)'}
    for nm, m in M.items():
        rec, miss = recall_cells(m, cells, valid)
        kept = m.sum(1)
        ok = kept > 0
        prec = (m & hull).sum(1) / np.maximum(kept, 1)
        out = {'kept_per_frame_w': float(np.average(kept, weights=w)), 'kept_frac_grid_w': float(np.average(kept, weights=w) / 1024),
               'kept_frac_video_cells_w': float(np.average(kept, weights=w) / ncells), 'frames_0_kept': int((~ok).sum()),
               'hull_precision_w': float(np.average(prec[ok], weights=w[ok])), 'recall_per_animal_w': {},
               'missed_frac_per_animal_w': {}}
        for j, a in enumerate(animals):
            f = np.isfinite(rec[:, j])
            out['recall_per_animal_w'][a] = float(np.average(rec[f, j], weights=w[f]))
            out['missed_frac_per_animal_w'][a] = float(np.average(miss[f, j], weights=w[f]))
        f = valid.any(2).any(1)
        allrec = np.nansum(np.where(np.isfinite(rec), rec * valid.sum(2), 0), 1) / np.maximum(valid.sum((1, 2)), 1)
        out['recall_all_kp_w'] = float(np.average(allrec[f], weights=w[f]))
        out['frames_with_a_missed_animal_w'] = float(np.average(miss.any(1)[f], weights=w[f]))
        if bench == 'mabe' and 'huddles' in E:
            hud = np.zeros(len(Fm), bool)
            hud[idx] = E.huddles.values
            g = f & hud
            out['recall_all_kp_huddles_w'] = float(np.average(allrec[g], weights=w[g]))
        res[nm] = out
        log(f'{bench} {nm}: ' + json.dumps(out))
    inter = (M['cur'] & M['U']).sum(1)
    res['overlap'] = {'cur_cells_inside_U_w': float(np.average(inter / np.maximum(M['cur'].sum(1), 1), weights=w)),
                      'U_cells_inside_cur_w': float(np.average(inter / np.maximum(M['U'].sum(1), 1), weights=w))}
    (OUT / 'maskqa').mkdir(parents=True, exist_ok=True)
    (OUT / 'maskqa' / f'{bench}_store.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
