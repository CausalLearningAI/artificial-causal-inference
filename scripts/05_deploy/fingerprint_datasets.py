#!/usr/bin/env python3
"""
Fingerprint the PPCI datasets exactly as the deployed model sees them.

For each requested version, loads PPCIDataset.from_disk with the deployed
config (results/ppci/ants/performances/final/config.json: encoder, token,
frame_type, dist_mode) and DS_KWARGS, applies the context window the same way
deploy_model.py does, and records a sha256 of every array/field plus row
counts. Run it before and after a code or data change: equal JSONs mean the
model gets byte-identical inputs.

Fields hashed (raw X before the context window, X after it):
  X_raw, X, Y (NaN-aware: NaN mask + values with NaN set to 0), T, E, W,
  W_cols, obs_ids, frame_idx, annotated_mask, train_mask, val_mask,
  has_annotations, n_rows, n_obs, X shape.

Usage (CPU job, ~64 GB):
  python scripts/05_deploy/fingerprint_datasets.py \
      --out results/ppci/ants/regress_30min/fingerprint_baseline.json
  python scripts/05_deploy/fingerprint_datasets.py --versions v5 --full-versions v5 --out /tmp/fp.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from ppci.dataset import PPCIDataset          # noqa: E402
from ppci.hparam_search import DS_KWARGS      # noqa: E402

DEPLOY_CONFIG = ROOT / "results" / "ppci" / "ants" / "performances" / "final" / "config.json"


def _sha(a) -> str:
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode() + str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def _sha_str(values) -> str:
    return hashlib.sha256("\n".join(str(v) for v in values).encode()).hexdigest()


def fingerprint(ds: PPCIDataset, X_raw_sha: str) -> dict:
    Y = ds.Y.float().numpy()
    nan = np.isnan(Y)
    return {
        "n_rows":          int(len(ds.X)),
        "n_obs":           int(len(np.unique(ds.obs_ids))),
        "X_shape":         list(ds.X.shape),
        "X_raw":           X_raw_sha,
        "X":               _sha(ds.X),
        "Y_nan_mask":      _sha(nan),
        "Y_values":        _sha(np.where(nan, 0.0, Y).astype(np.float32)),
        "n_Y_nan":         int(nan.sum()),
        "T":               _sha_str(ds.T),
        "E":               _sha(ds.E),
        "W":               _sha(ds.W),
        "W_cols":          list(ds.W_cols),
        "obs_ids":         _sha_str(ds.obs_ids),
        "frame_idx":       _sha(ds.frame_idx),
        "annotated_mask":  _sha(ds.annotated_mask),
        "n_annotated":     int(ds.annotated_mask.sum()),
        "train_mask":      _sha(ds.train_mask),
        "val_mask":        _sha(ds.val_mask),
        "has_annotations": str(ds.has_annotations),
    }


def run_one(version: str, encoder: str, token: str, frame_type: str, dist_mode: str,
            k: int, mode: str) -> dict:
    t0 = time.time()
    ds = PPCIDataset.from_disk(
        "ants", version, encoder, token,
        frame_type=frame_type, dist_mode=dist_mode,
        n_val_videos=0, **DS_KWARGS,
    )
    X_raw_sha = _sha(ds.X)
    if k > 0:
        ds.apply_context_window(k, mode=mode)
    fp = fingerprint(ds, X_raw_sha)
    fp["load"] = dict(version=version, encoder=encoder, token=token, frame_type=frame_type,
                      dist_mode=dist_mode, context_window=k, context_mode=mode)
    print(f"  [{version} {frame_type}/{encoder}/{token} dist={dist_mode} k={k}] "
          f"rows={fp['n_rows']:,}  obs={fp['n_obs']}  annotated={fp['n_annotated']:,}  "
          f"X={fp['X'][:12]}  ({time.time() - t0:.0f}s)", flush=True)
    del ds
    return fp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--versions", nargs="+", default=["v3", "v4", "v5"],
                        help="Versions loaded with the deployed (POV) config")
    parser.add_argument("--full-versions", nargs="+", default=["v5"],
                        help="Versions also loaded full-frame, deployed encoder/token, no distances")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    with open(DEPLOY_CONFIG) as f:
        cfg = json.load(f)
    encoder, token = cfg["encoder"], cfg.get("token", "class")
    k, mode = int(cfg.get("context_window", 0)), cfg.get("context_mode", "mean")
    frame_type, dist_mode = cfg.get("frame_type", "full"), cfg.get("dist_mode", "none")

    out = {"deploy_config": str(DEPLOY_CONFIG.relative_to(ROOT)), "ds_kwargs": DS_KWARGS,
           "datasets": {}}
    for v in args.versions:
        out["datasets"][f"{v}/{frame_type}/{dist_mode}"] = run_one(
            v, encoder, token, frame_type, dist_mode, k, mode)
    for v in args.full_versions:
        out["datasets"][f"{v}/full/none"] = run_one(v, encoder, token, "full", "none", k, mode)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
