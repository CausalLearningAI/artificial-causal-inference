"""Is the encode bottleneck the HuggingFace row lookup rather than JPEG decoding?

Observed: encoding 444k frames at 224px on an H100 ran at 317 frames/s -- ~50 ms per frame
per worker, while decoding a 512x512 JPEG costs ~2-3 ms. So ~95% of the time is spent
somewhere other than the decode. Prime suspect is `hf_dataset[i]['image']`: a random-row
lookup into the arrow-backed dataset. An earlier benchmark missed this because it read
CONTIGUOUS indices; production reads `all_needed`, a sparse subset spanning the dataset.

annotations.csv already carries a `frame_path` column pointing at the on-disk JPEG
(relative to dataset/), so the HF indirection can be bypassed entirely.

Compares, on the SAME sparse access pattern production uses:
  A) hf_dataset[i]['image']          -- current path
  B) PIL.Image.open(frame_path)      -- direct file read
Each on a DISJOINT index set, so neither warms the OS page cache for the other.

CPU-only -- safe to run while GPU jobs are training.

Usage: python scripts/mice_behavior/bench_frame_read.py --n 1500
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
import grid_search_frame as gsf
from src.dataset.get_dataset import load_dataset

p = argparse.ArgumentParser()
p.add_argument('--n', type=int, default=1500, help='frames to time per method')
p.add_argument('--input-size', type=int, default=224)
args = p.parse_args()

DATASET_ROOT = Path('dataset')
ann = pd.read_csv(gsf.DATASET_DIR / 'mice' / 'v1' / 'annotations.csv', usecols=['frame_path'])
n_total = len(ann)
print(f'{n_total:,} frames in annotations.csv', flush=True)

# Sparse, sorted, spread across the whole dataset -- mimics `all_needed`. Two disjoint sets.
rng = np.random.default_rng(0)
pick = np.sort(rng.choice(n_total, size=2 * args.n, replace=False))
idx_hf, idx_direct = np.sort(pick[0::2]), np.sort(pick[1::2])

print('\nA) HuggingFace ds[i][\'image\'] (current path)...', flush=True)
hf = load_dataset(subject='mice', version='v1', dataset_root=str(gsf.DATASET_DIR), frame_type='full')
t0 = time.perf_counter()
for i in idx_hf:
    im = hf[int(i)]['image']
    im.load()
    im = im.resize((args.input_size, args.input_size))
dt_hf = time.perf_counter() - t0

print('B) direct PIL open of frame_path...', flush=True)
paths = ann.frame_path.values
t0 = time.perf_counter()
for i in idx_direct:
    with Image.open(DATASET_ROOT / paths[int(i)]) as im:
        im.load()
        im = im.resize((args.input_size, args.input_size))
dt_direct = time.perf_counter() - t0

fps_hf, fps_direct = args.n / dt_hf, args.n / dt_direct
print(f'\n{"method":<34} {"ms/frame":>10} {"frames/s":>10}')
print(f'{"A) hf_dataset[i][image]":<34} {1000*dt_hf/args.n:>10.1f} {fps_hf:>10.1f}')
print(f'{"B) PIL.open(frame_path)":<34} {1000*dt_direct/args.n:>10.1f} {fps_direct:>10.1f}')
print(f'\nsingle-threaded speedup from bypassing HF: {fps_direct/fps_hf:.1f}x')
print(f'extrapolated to 16 workers: HF ~{16*fps_hf:.0f} fps | direct ~{16*fps_direct:.0f} fps')
print(f'(production encode currently runs at ~317 fps at 224px on H100)')
print(f'\n444,000-frame encode: HF ~{444000/(16*fps_hf)/60:.0f} min | direct ~{444000/(16*fps_direct)/60:.0f} min')
