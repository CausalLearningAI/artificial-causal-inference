"""
MABe22 ECI round 3, E3 reference (CPU job; keypoints used only as a reference, never as features): mouse body length in
patches for each input size S. Keypoints: mouse_submission_keypoints.npy, 512-px scale, (y, x) order (eci_summary),
assumed MABe22 order nose 0, neck 3, centre back 6, tail base 9 (the pose reference identified 0 and 9 label-free; the
mean step lengths 0-3 / 3-6 / 6-9 are printed as a check). Per eval frame (plan/eval_frames.parquet) and mouse: chain
length nose -> neck -> centre back -> tail base (512 px); patches = length x S / 512 / 14 (224-px source upscaled to S,
14-px patches); median / IQR over frames x mice (mice with any keypoint at 0 skipped). Output OUT/r3/bodylen.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eci_common import OUT, PLAN, RAW, log  # noqa: E402

E = pd.read_parquet(PLAN / 'eval_frames.parquet')
sk = np.load(RAW / 'mouse_submission_keypoints.npy', allow_pickle=True).item()['sequences']
L, NT, steps, skipped = [], [], [], 0
for c, i in zip(E['clip'].values, E.i.values):
    k = sk[c]['keypoints'][i].astype(np.float32)  # (3, 12, 2)
    for m in range(3):
        p = k[m, [0, 3, 6, 9]]
        if (k[m] == 0).all(1).any():
            skipped += 1
            continue
        st = np.linalg.norm(np.diff(p, axis=0), axis=1)
        steps.append(st)
        L.append(st.sum())
        NT.append(np.linalg.norm(p[0] - p[3]))
L, NT, steps = np.array(L), np.array(NT), np.array(steps)
out = {'n_mouse_frames': int(len(L)), 'skipped_zero_keypoints': skipped,
       'mean_steps_px512_nose_neck_back_tailbase': steps.mean(0).round(2).tolist(),
       'chain_px512_median': float(np.median(L)), 'chain_px512_iqr': np.percentile(L, [25, 75]).round(2).tolist(),
       'nose_tailbase_straight_px512_median': float(np.median(NT))}
for S in (224, 336, 448, 672):
    f = S / 512 / 14
    out[f'patches_per_mouse_length_{S}'] = {'median': float(np.median(L) * f),
                                            'iqr': (np.percentile(L, [25, 75]) * f).round(2).tolist(),
                                            'straight_median': float(np.median(NT) * f)}
(OUT / 'r3').mkdir(parents=True, exist_ok=True)
(OUT / 'r3' / 'bodylen.json').write_text(json.dumps(out, indent=1))
log(json.dumps(out))
