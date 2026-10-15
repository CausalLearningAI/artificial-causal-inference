"""
MABe22 mouse ECI evaluation: mask QA against the pose keypoints (CPU job). The keypoints are used ONLY to measure the
mask, never as features. Keypoints: 512 px scale, (y, x) order, label-frame indexed (scale / order / offset found by
eci_prep.py). For every eval frame: the fraction of the 3 x 12 keypoints that fall in a kept patch (448 grid: patch =
7 x 7 px of the 224 frame; 224 grid: 14 x 14 px), the fraction of mice with NO keypoint in a kept patch, and the
fraction of frames where some mouse is missed; split by huddles label too.
Output: OUT/masks/mask_qa.json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eci_common import OUT, PLAN, RAW, TOK, log  # noqa: E402
from eci_train import shards  # noqa: E402


def main():
    E = pd.read_parquet(PLAN / 'eval_frames.parquet')
    sk = np.load(RAW / 'mouse_submission_keypoints.npy', allow_pickle=True).item()['sequences']
    res = {}
    for name, G, px in (('pos448', 32, 7), ('pos224', 16, 14)):
        col = 'n448' if G == 32 else 'n224'
        pos = np.concatenate([np.load(TOK / f'{k}_{name}.npy') for k in shards('eval')])
        Fm = pd.concat([pd.read_parquet(TOK / f'{k}_frames.parquet') for k in shards('eval')]).reset_index(drop=True)
        st = np.r_[0, np.cumsum(Fm[col].values)]
        hit, miss_m, miss_f = [], [], []
        for f in range(len(E)):
            kp = sk[E['clip'][f]]['keypoints'][E.i[f]].astype(np.float32) * 224 / 512  # (3, 12, 2) (y, x)
            y, x = np.clip(kp[..., 0], 0, 223), np.clip(kp[..., 1], 0, 223)
            cell = (y // px).astype(int) * G + (x // px).astype(int)
            kept = np.zeros(G * G, bool)
            kept[pos[st[f]:st[f + 1]]] = True
            valid = ~(kp == 0).all(-1)  # (3, 12): a keypoint at (0, 0) is missing
            vm = valid.any(1)  # mice with any keypoint
            if not vm.any():
                hit.append(np.nan); miss_m.append(np.nan); miss_f.append(np.nan)
                continue
            h = kept[cell]  # (3, 12)
            hit.append(h[valid].mean())
            mm = ~(h & valid).any(1)
            miss_m.append(mm[vm].mean())
            miss_f.append(float(mm[vm].any()))
        hit, miss_m, miss_f = map(np.array, (hit, miss_m, miss_f))
        ok = np.isfinite(hit)
        hit, miss_m, miss_f, w, Fm, E2 = hit[ok], miss_m[ok], miss_f[ok], E.w.values[ok], Fm[ok], E[ok]
        hud = E2.huddles.values.astype(bool)
        res[name] = {'kp_recall_w': float(np.average(hit, weights=w)),
                     'mouse_missed_frac_w': float(np.average(miss_m, weights=w)),
                     'frames_with_a_missed_mouse_w': float(np.average(miss_f, weights=w)),
                     'kp_recall_huddles_w': float(np.average(hit[hud], weights=w[hud])),
                     'frames_with_a_missed_mouse_huddles_w': float(np.average(miss_f[hud], weights=w[hud])),
                     'frames_scored': int(ok.sum()), 'note': 'all-(0,0) keypoints = missing, excluded',
                     'kept_per_frame_w': float(np.average(Fm[col].values, weights=w)),
                     'kept_frac_of_grid_w': float(np.average(Fm[col].values, weights=w) / (G * G))}
        log(f'{name}: {res[name]}')
    (OUT / 'masks').mkdir(parents=True, exist_ok=True)
    (OUT / 'masks' / (os.environ.get('MASKQA_NAME') or 'mask_qa.json')).write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
