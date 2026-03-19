#!/usr/bin/env python
"""Optimize HSV color bounds for ant tracking configs.

Runs all versions in parallel using multiprocessing.
Each version: samples frames, grid-searches (H, S, V) lower bounds
using 2D survival functions (no contour finding in inner loop).

Usage:
    python src/tracking/optimize_bounds.py              # all versions
    python src/tracking/optimize_bounds.py v2 v3        # specific versions
"""

import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.tracking.calibration import optimize_color_bounds, load_config


def run(version: str) -> dict:
    return optimize_color_bounds(version, n_frames_per_obs=10, max_obs=50,
                                 verbose=True)


if __name__ == "__main__":
    versions = sys.argv[1:] or ["v1", "v2", "v3", "v4", "v5"]

    print(f"Optimizing color bounds for: {', '.join(versions)}")
    t0 = time.time()

    # Use 'spawn' context — 'fork' deadlocks with OpenCV's internal threads
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(versions), mp_context=ctx) as pool:
        all_results = dict(zip(versions, pool.map(run, versions)))

    elapsed = time.time() - t0

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  Results  ({elapsed:.1f}s)")
    print(f"{'='*72}")
    for v in versions:
        cfg = load_config(v)
        print(f"\n  {v}:")
        for color in ["blue", "yellow"]:
            r = all_results[v][color]
            print(
                f"    {color:7s}  lb={r['lb']}  ub={r['ub']}"
                f"  score={r['score']:.3f}"
            )
            print(
                f"             prec={r['segmentation_precision']:.3f}"
                f"  no_det={r['no_detection_rate']:.3f}"
                f"  over_det={r['over_detection_rate']:.3f}"
            )
    print()
