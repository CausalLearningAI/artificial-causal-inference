#!/usr/bin/env python3
"""
Generate per-observation annotation CSV files from the final deployed POV model.

For each observation in the target version, writes a CSV with:
  frame_id    — frame index at original FPS (default 30)
  time_sec    — time in seconds (frame_id / original_fps)
  Y2F_prob    — model probability for yellow-to-focal grooming
  B2F_prob    — model probability for blue-to-focal grooming
  Y2F         — Y2F_prob binarized at threshold (default 0.5)
  B2F         — B2F_prob binarized at threshold (default 0.5)

Frames that were downsampled for the ML pipeline are upsampled back to
original FPS by repeating the prediction for each original-fps frame that
falls within the corresponding ML-fps interval.

Usage:
  python scripts/05_deploy/generate_annotations.py               # v5, all obs
  python scripts/05_deploy/generate_annotations.py --version v4
  python scripts/05_deploy/generate_annotations.py --obs 5_17_7
  python scripts/05_deploy/generate_annotations.py --threshold 0.4
  python scripts/05_deploy/generate_annotations.py --out-dir /custom/output
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from ppci.dataset import PPCIDataset          # noqa: E402
from ppci.hparam_search import (              # noqa: E402
    RESULTS_DIR, DS_KWARGS,
    BASE_CFG,
    _build_finetune_cfg,
)
from ppci.train import build_model            # noqa: E402


# ── constants ────────────────────────────────────────────────────────────────

ORIGINAL_FPS = 30   # native camera frame rate
ML_FPS       = 5    # target_fps from configs/data/ants.yaml


# ── helpers ──────────────────────────────────────────────────────────────────

def load_model(deploy_dir: Path, device: torch.device):
    """Load the deployed model from results dir."""
    model_path  = deploy_dir / "model.pt"
    config_path = deploy_dir / "config.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    with open(config_path) as f:
        cfg_dict = json.load(f)

    encoder   = cfg_dict["encoder"]
    token     = cfg_dict.get("token", "class")
    k         = int(cfg_dict.get("context_window", BASE_CFG.training.context_window))
    mode      = cfg_dict.get("context_mode", BASE_CFG.training.context_mode)
    dist_mode = cfg_dict.get("dist_mode", "none")

    # Build dataset just to get input shape for model init
    ref_version = cfg_dict["finetune_versions"][-1]
    ds_ref = PPCIDataset.from_disk(
        "ants", ref_version, encoder, token,
        frame_type="pov", dist_mode=dist_mode,
        n_val_videos=0, **DS_KWARGS,
    )
    if k > 0:
        ds_ref.apply_context_window(k, mode=mode)

    finetune_cfg = _build_finetune_cfg(cfg_dict)
    model = build_model(ds_ref, finetune_cfg)
    del ds_ref

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    return model, cfg_dict


def _expand_to_original_fps(
    ml_frame_idx: np.ndarray,
    y2f_prob: np.ndarray,
    b2f_prob: np.ndarray,
    original_fps: int,
    ml_fps: int,
    n_total_original_frames: int | None = None,
) -> pd.DataFrame:
    """
    Expand ML-fps predictions back to original_fps by repeating each prediction
    for every original-fps frame that maps to the same ML-fps frame.

    ML frame at index `i` corresponds to original frames in the interval:
      [i * stride, (i+1) * stride)   where stride = original_fps / ml_fps

    If `n_total_original_frames` is provided, we also fill any tail frames
    (beyond the last ML frame) with the last prediction.
    """
    stride = original_fps // ml_fps  # e.g. 6 if 30 / 5

    rows = []
    for i, (ml_fi, p_y, p_b) in enumerate(zip(ml_frame_idx, y2f_prob, b2f_prob)):
        start = int(ml_fi) * stride
        # End: either start of next ML frame or end of video
        if i + 1 < len(ml_frame_idx):
            end = int(ml_frame_idx[i + 1]) * stride
        else:
            end = start + stride  # one stride's worth for the last ML frame
        for orig_fi in range(start, end):
            rows.append((orig_fi, p_y, p_b))

    # Optionally cover any remaining tail frames
    if n_total_original_frames is not None and rows:
        last_orig = rows[-1][0]
        if last_orig + 1 < n_total_original_frames:
            last_py, last_pb = rows[-1][1], rows[-1][2]
            for orig_fi in range(last_orig + 1, n_total_original_frames):
                rows.append((orig_fi, last_py, last_pb))

    df = pd.DataFrame(rows, columns=["frame_id", "Y2F_prob", "B2F_prob"])
    return df


def generate_annotations_for_version(
    version: str,
    deploy_dir: Path,
    out_dir: Path,
    threshold: float = 0.5,
    obs_filter: str | None = None,
    original_fps: int = ORIGINAL_FPS,
    ml_fps: int = ML_FPS,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {deploy_dir} ...")
    model, cfg_dict = load_model(deploy_dir, device)

    encoder   = cfg_dict["encoder"]
    token     = cfg_dict.get("token", "class")
    k         = int(cfg_dict.get("context_window", BASE_CFG.training.context_window))
    mode      = cfg_dict.get("context_mode", BASE_CFG.training.context_mode)
    dist_mode = cfg_dict.get("dist_mode", "none")

    print(f"Loading dataset ants/{version} (pov, dist_mode={dist_mode}) ...")
    ds = PPCIDataset.from_disk(
        "ants", version, encoder, token,
        frame_type="pov", dist_mode=dist_mode,
        n_val_videos=0, **DS_KWARGS,
    )
    if k > 0:
        ds.apply_context_window(k, mode=mode)

    print("Running inference ...")
    ds.add_predictions(model, device)

    # Build frame-level DataFrame
    Yhat = ds.Yhat.numpy()  # (N, 2) for [Y2F, B2F]
    y2f_prob = Yhat[:, 0]
    b2f_prob = Yhat[:, 1]
    obs_ids   = ds.obs_ids
    frame_idx = ds.frame_idx.numpy()

    out_dir.mkdir(parents=True, exist_ok=True)

    unique_obs = np.unique(obs_ids)
    if obs_filter is not None:
        if obs_filter not in unique_obs:
            raise ValueError(f"Observation '{obs_filter}' not found in {version}. "
                             f"Available: {sorted(unique_obs)}")
        unique_obs = np.array([obs_filter])

    print(f"Writing CSVs to {out_dir} ({len(unique_obs)} observations) ...")
    for obs in unique_obs:
        mask = obs_ids == obs
        ml_fi   = frame_idx[mask]
        p_y     = y2f_prob[mask]
        p_b     = b2f_prob[mask]

        # Sort by ML frame index
        sort_ord = np.argsort(ml_fi)
        ml_fi, p_y, p_b = ml_fi[sort_ord], p_y[sort_ord], p_b[sort_ord]

        df = _expand_to_original_fps(ml_fi, p_y, p_b, original_fps, ml_fps)
        df["time_sec"] = df["frame_id"] / original_fps
        df["Y2F"] = (df["Y2F_prob"] >= threshold).astype(int)
        df["B2F"] = (df["B2F_prob"] >= threshold).astype(int)

        # Final column order
        df = df[["frame_id", "time_sec", "Y2F_prob", "B2F_prob", "Y2F", "B2F"]]
        df.to_csv(out_dir / f"{obs}.csv", index=False, float_format="%.6f")

    print(f"Done. {len(unique_obs)} CSVs written to {out_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate per-observation annotation CSVs from the deployed POV model.",
    )
    parser.add_argument("--version", default="v5",
                        help="Dataset version to annotate (default: v5)")
    parser.add_argument("--deploy-dir", type=Path, default=None,
                        help="Path to deployed model dir (default: results/ppci/ants/hparam/deploy/final)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output directory for CSVs (default: results/ppci/ants/annotations/{version}/)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Binarization threshold for Y2F/B2F (default: 0.5)")
    parser.add_argument("--obs", default=None, metavar="OBS_ID",
                        help="Process only this observation (default: all)")
    parser.add_argument("--original-fps", type=int, default=ORIGINAL_FPS,
                        help=f"Original camera FPS (default: {ORIGINAL_FPS})")
    parser.add_argument("--ml-fps", type=int, default=ML_FPS,
                        help=f"ML pipeline FPS (default: {ML_FPS})")
    args = parser.parse_args()

    deploy_dir = args.deploy_dir or (RESULTS_DIR / "deploy" / "final")
    out_dir    = args.out_dir    or (ROOT / "results" / "ppci" / "ants" / "annotations" / args.version)

    generate_annotations_for_version(
        version=args.version,
        deploy_dir=deploy_dir,
        out_dir=out_dir,
        threshold=args.threshold,
        obs_filter=args.obs,
        original_fps=args.original_fps,
        ml_fps=args.ml_fps,
    )


if __name__ == "__main__":
    main()
