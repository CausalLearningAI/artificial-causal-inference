"""CalMS21 - cost per arm (addendum 1 (3) of eci_res.sh) and patches per body length (addendum 1 (1)); login safe.
encoder tokens / frame = full-frame tokens the encoder processes per sampled frame (G^2 + prefix per image, x2 images
(t and t-60) for the Delta arms; V-JEPA: 8 G^2 per 16-frame clip, one clip); SAE tokens / frame = mean kept tokens
after the mask (eval + train frames, pass A cells x 4 on the 64 grid); frames / GPU-s = encoder inputs per GPU-second
from the encode JSONs, divided by the inputs per frame (2 for Delta arms). -> OUT/res/cost.md"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calms_common import OUT, TOK  # noqa: E402

ARMS = {'d2_448': (14, 448, 1, 2), 'd2_896': (14, 896, 1, 2), 'd3_512_L10': (16, 512, 5, 2), 'd3_1024_L10': (16, 1024, 5, 2),
        'd3l_512_L20': (16, 512, 5, 2), 'd3l_1024_L20': (16, 1024, 5, 2), 'd2l_448_L20': (14, 448, 1, 2),
        'd2l_896_L20': (14, 896, 1, 2), 'vj_512': (16, 512, 0, 1), 'vj_1024': (16, 1024, 0, 1),
        'd3_1024lo_L10': (16, 1024, 5, 2), 'd3l_1024lo_L20': (16, 1024, 5, 2), 'd2_896lo': (14, 896, 1, 2),
        'd3_512': (16, 512, 5, 2), 'd3_1024': (16, 1024, 5, 2), 'd2l_448': (14, 448, 1, 2)}
BL = json.loads((OUT / 'bodylen' / 'bodylen.json').read_text())
L = BL['body_len_native_px_median']
n32 = np.concatenate([pd.read_parquet(f).n32.values for f in sorted(TOK.glob('*of6_frames.parquet'))])
rate = {}
for f in sorted((OUT / 'encode_json').glob('*.json')):
    if 'smoke' in f.name or f.name.startswith('p2_'):
        continue
    for c, v in json.loads(f.read_text())['per_config'].items():
        r = rate.setdefault(c, [0.0, 0])
        r[0] += v['gpu_s']
        r[1] += v['inputs']
rows = ['# CalMS21 cost per arm and patches per body length', '',
        f"Body length (label-free, resident, native px): {L:.1f} (IQR {BL['body_len_iqr']}); keypoint nose-tail base "
        f"{BL['kp_nose_tail_base_px_median_same_frames']:.1f}. Mean kept 32-grid cells per frame {n32.mean():.1f}.", '',
        '| arm | grid | input px | patch px | patches / body length | encoder tokens / frame (full frame) | '
        'SAE tokens / frame (after mask) | frames / GPU-s (encode, L40S/A40 mix) |', '|---|---|---|---|---|---|---|---|']
for c, (ps, S, npre, k) in ARMS.items():
    G = S // ps
    vj = c.startswith('vj')
    enc_tok = (8 * G * G) if vj else k * (G * G + npre)
    sae_tok = n32.mean() * (4 if G == 64 else 1)
    fr = rate[c][1] / rate[c][0] / k if c in rate and rate[c][0] else float('nan')
    ppb = L * S / 1024 / ps
    rows.append(f'| {c} | {G} | {S} | {ps} | {ppb:.2f}{" (source detail ~40 px / body)" if "lo" in c.split("_")[1] else ""} | '
                f'{enc_tok:,} | {sae_tok:.0f} | {fr:.1f} |')
(OUT / 'res').mkdir(parents=True, exist_ok=True)
(OUT / 'res' / 'cost.md').write_text('\n'.join(rows) + '\n')
print('\n'.join(rows))
