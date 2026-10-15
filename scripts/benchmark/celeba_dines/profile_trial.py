"""Timing profile of DINES on one trial (CPU job): forward-set size, number of conditioning sets, seconds per set.
  python profile_trial.py <tau> <N> <seed> <delta> [max_subsets]"""
import cProfile
import pstats
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))
from eci.dines import dines  # noqa: E402

BEEGFS = Path('/mnt/beegfs/locatgrp/shared/rcadei/dines_celeba')
tau, N, seed, delta = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
max_subsets = int(sys.argv[5]) if len(sys.argv) > 5 else 4096
codes = np.load(BEEGFS / 'codes_f32.npy', mmap_mode='r')
arr = np.load(BEEGFS / 'original_trials.npz')[f'{tau}_{N}_{seed}']
Z, T = np.asarray(codes[arr[:, 0]], dtype=np.float64), arr[:, 1].astype(np.int64)
pr = cProfile.Profile()
t0 = time.time()
pr.enable()
r = dines(Z, T, delta=delta, max_subsets=max_subsets, verbose=True)
pr.disable()
print(f'total {time.time() - t0:.1f}s forward {r["elapsed_forward_s"]:.1f}s terminal {r["elapsed_terminal_s"]:.1f}s '
      f'|S~| {len(r["forward_set"])} sets {len(r["sets"])} complete {r["terminal_complete"]} selected {[j + 1 for j in r["selected"]]}')
print('forward', [j + 1 for j in r['forward_set']])
pstats.Stats(pr).sort_stats('cumulative').print_stats(18)
