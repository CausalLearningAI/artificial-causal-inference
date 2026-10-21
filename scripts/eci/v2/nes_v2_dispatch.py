"""
Run another dispatcher / runner (argv[1] = its path, the rest passed through) with an in-memory patch of
src/eci/contrasts.py (no shared file is edited):
  V2_ZERO_THR=1   pooled_thresholds() returns 0 for every quantile, so the bout / latency runners call a frame 'above'
                  iff its code > 0. Used for 0 / 1 codes sets (the per-neuron MAP binary '<sae>_map'): bouts and latency
                  then come from the SAME binary (a pooled q-quantile of a 0 / 1 signal would be 1 for any neuron active
                  on > 5% of frames, and 'above' is strict). The setting keys keep their q labels (q0.90 / 0.95 / 0.99 are
                  identical here).
"""
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(os.environ.get('NES_CODE_ROOT') or Path(__file__).resolve().parents[3])
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts/eci'))
import numpy as np  # noqa: E402
from eci import contrasts as C  # noqa: E402

if os.environ.get('V2_ZERO_THR') == '1':
    _orig = C.pooled_thresholds

    def _zero(codes_path, qs, n_sample=2_000_000, seed=0, col_chunk=64):
        m = np.load(codes_path, mmap_mode='r').shape[1]
        n = np.load(codes_path, mmap_mode='r').shape[0]
        return np.zeros((len(qs), m)), min(n, n_sample)
    C.pooled_thresholds = _zero
    print('[nes_v2_dispatch] V2_ZERO_THR=1: thresholds = 0 (above = code > 0)', flush=True)
target = sys.argv[1]
sys.argv = [target] + sys.argv[2:]
runpy.run_path(target, run_name='__main__')
