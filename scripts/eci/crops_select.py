"""
Crop prototype step 0: choose the 20 training videos (label-free) and calibrate the blob-area /
pair-distance thresholds of src/eci/crops.py from the dark-blob statistics of those videos.

Training videos: not in the 8 held-out validation pools; one video per (line, genotype, phase)
cell (3 x 2 x 3 = 18), each from a different pool, plus 2 more from further pools, alternating sex.
Calibration (no labels): 60 frames per training video, blobs with thresholds unset.
    n_mice      the most common blob count among frames (with blobs >= min_area)
    single      A1 = median blob area in frames whose blob count is n_mice (all mice apart -> each
                blob is mostly one mouse); single range = [0.5 A1, 1.5 A1] (two mice in contact
                have ~2 A1 minus overlap; the 5th-95th percentile range was measured too wide:
                its top reached ~2 A1)
    pair_d      0.25 x median major-axis length of those single blobs (~ one head length)

Output: dataset/mice/v1/eci/crops/config.json (train videos, params, calibration stats)
Usage: python scripts/eci/crops_select.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.eci.crops import DEFAULTS, dark_mask, find_blobs  # noqa: E402
from src.eci.foreground import obs_rows  # noqa: E402
from PIL import Image  # noqa: E402

OUT = REPO / 'dataset/mice/v1/eci/crops'
BG = REPO / 'dataset/mice/v1/eci/fg448/background'


def main():
    val_pools = json.loads((REPO / 'dataset/mice/v1/eci/sae/matryoshka_btk_1024_k16_ep20_s0/metrics.json').read_text())['val_pools']
    exp = pd.read_csv(REPO / 'data/mice/v1/experiment.csv')
    exp = exp[~exp.pool.isin(val_pools)]
    rng = np.random.default_rng(0)
    used, pick = set(), []
    cells = [(l, g, ph) for l in sorted(exp.line.unique()) for g in sorted(exp.genotype.unique()) for ph in 'HOP']
    for k, (l, g, ph) in enumerate(cells):
        c = exp[(exp.line == l) & (exp.genotype == g) & (exp.phase == ph) & ~exp.pool.isin(used)]
        sx = 'mf'[k % 2]
        c = c[c.sex == sx] if (c.sex == sx).any() else c
        r = c.iloc[rng.integers(len(c))]
        used.add(r.pool); pick.append(r.observation_id)
    rest = exp[~exp.pool.isin(used)]
    for k in range(2):
        r = rest.iloc[rng.integers(len(rest))]
        rest = rest[rest.pool != r.pool]
        used.add(r.pool); pick.append(r.observation_id)
    sel = exp.set_index('observation_id').loc[pick]
    print(sel[['pool', 'line', 'genotype', 'sex', 'phase', 'odor']].to_string())

    ann = REPO / 'dataset/mice/v1/annotations.csv'
    fp = pd.read_csv(ann, usecols=['frame_path'])['frame_path'].values
    ranges = obs_rows(ann)
    p = dict(DEFAULTS)
    recs = []
    for o in pick:
        pix_bg = np.load(BG / f'{o}.npz')['pix_bg']
        lo, hi = ranges[o]
        for r in np.linspace(lo, hi - 1, 60).astype(int):
            grey = np.asarray(Image.open(REPO / 'dataset' / fp[r]).convert('L'))
            bl = find_blobs(dark_mask(grey, pix_bg, p), p)
            for b in bl:
                recs.append((o, r, len(bl), b['area'], b['major'], b['minor']))
    df = pd.DataFrame(recs, columns=['obs', 'row', 'n', 'area', 'major', 'minor'])
    per_frame = df.groupby('row').n.first()
    n_mice = int(per_frame.mode().iloc[0])
    s = df[df.n == n_mice]
    q = np.percentile(s.area, [5, 25, 50, 75, 95])
    single_lo, single_hi = float(0.5 * q[2]), float(1.5 * q[2])
    pair_d = float(0.25 * s.major.median())
    p.update(single_lo=single_lo, single_hi=single_hi, pair_d=pair_d)
    stats = {'blob_count_hist': per_frame.value_counts().sort_index().to_dict(), 'n_mice_mode': n_mice,
             'single_area_pct_5_25_50_75_95': q.tolist(), 'single_major_median': float(s.major.median()),
             'single_minor_median': float(s.minor.median()), 'all_area_pct_1_50_99': np.percentile(df.area, [1, 50, 99]).tolist()}
    h, e = np.histogram(df.area / q[2], bins=np.arange(0, 5.01, 0.25))
    stats['area_hist_in_units_of_A1'] = {f'{a:.2f}': int(c) for a, c in zip(e[:-1], h)}
    stats['blob_count_hist'] = {int(k): int(v) for k, v in stats['blob_count_hist'].items()}
    print(json.dumps(stats, indent=1)); print(p)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'config.json').write_text(json.dumps({'train_videos': pick, 'val_pools': val_pools, 'params': p,
                                                 'calibration': stats}, indent=1))


if __name__ == '__main__':
    main()
