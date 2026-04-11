#!/usr/bin/env python3
"""
Evaluate tracking quality by comparing proximity labels (B2F, Y2F) from the
tracker against human annotations.

Idea: when an annotator marks grooming (Y_Y2F=1 or Y_B2F=1), the yellow/blue
ant must be close to the focal ant.  So the tracker's proximity flag should
also be 1.  Measuring *recall* of the annotation-positive frames tells us how
often the tracker agrees with the human on "ants are interacting".

Output: prints per-observation and aggregate recall for each annotated version.

Usage:
    python src/tracking/evaluate_tracking.py --config-name ants/v3
    python src/tracking/evaluate_tracking.py --config-name ants/v3 obs_id=a4
"""

import sys
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def evaluate_observation(
    tracking_path: Path,
    annotations: pd.DataFrame,
    obs_id: str,
) -> dict | None:
    """Compare tracking proximity labels with annotation labels for one observation.

    Returns dict with per-signal recall and counts, or None if no data.
    """
    tracking = pd.read_csv(tracking_path)
    ann = annotations[annotations["observation_id"].astype(str) == str(obs_id)].copy()

    if ann.empty:
        return None

    merged = pd.merge(
        ann[["frame_idx", "Y_Y2F", "Y_B2F"]],
        tracking[["frame_idx", "Y2F", "B2F"]],
        on="frame_idx",
        how="inner",
    )

    if merged.empty:
        return None

    result = {"obs_id": obs_id, "n_frames": len(merged)}

    for ann_col, track_col, label in [
        ("Y_Y2F", "Y2F", "Y2F"),
        ("Y_B2F", "B2F", "B2F"),
    ]:
        positives = merged[ann_col] == 1
        n_pos = positives.sum()
        if n_pos > 0:
            tp = (merged.loc[positives, track_col] == 1).sum()
            recall = tp / n_pos
        else:
            tp = 0
            recall = np.nan

        negatives = merged[ann_col] == 0
        n_neg = negatives.sum()
        if n_neg > 0:
            fp = (merged.loc[negatives, track_col] == 1).sum()
            precision_denom = tp + fp
            precision = tp / precision_denom if precision_denom > 0 else np.nan
        else:
            fp = 0
            precision = np.nan

        result[f"{label}_n_pos"] = int(n_pos)
        result[f"{label}_tp"] = int(tp)
        result[f"{label}_recall"] = recall
        result[f"{label}_fp"] = int(fp)
        result[f"{label}_precision"] = precision

    return result


@hydra.main(
    version_base=None,
    config_path="../../configs/tracking",
    config_name="ants/v3",
)
def main(cfg: DictConfig) -> None:
    subject = cfg.subject
    version = cfg.version
    obs_id_filter = str(OmegaConf.select(cfg, "obs_id", default="") or "")

    tracking_dir = PROJECT_ROOT / f"dataset/{subject}/{version}/tracking"
    annotations_path = PROJECT_ROOT / f"dataset/{subject}/{version}/annotations.csv"

    if not tracking_dir.exists():
        print(f"[ERROR] Tracking directory not found: {tracking_dir}")
        return

    if not annotations_path.exists():
        print(f"Evaluating tracking for {subject}/{version}")
        print(f"  max_dist_activity = {cfg.max_dist_activity}px")
        print("  [SKIP] annotations not found; outcome-based metrics disabled.")
        return

    annotations = pd.read_csv(annotations_path)
    required_cols = {"observation_id", "frame_idx", "Y_Y2F", "Y_B2F"}
    missing_cols = required_cols - set(annotations.columns)
    if missing_cols:
        miss = ", ".join(sorted(missing_cols))
        print(f"Evaluating tracking for {subject}/{version}")
        print(f"  max_dist_activity = {cfg.max_dist_activity}px")
        print(f"  [SKIP] annotations missing required columns: {miss}")
        print("  outcome-based metrics disabled.")
        return
    tracking_files = sorted(tracking_dir.glob("*.csv"))

    if obs_id_filter:
        tracking_files = [f for f in tracking_files if f.stem == obs_id_filter]
        if not tracking_files:
            print(f"[ERROR] obs_id='{obs_id_filter}' not found in {tracking_dir}")
            return

    print(f"Evaluating tracking for {subject}/{version}  "
          f"({len(tracking_files)} observations)")
    print(f"  max_dist_activity = {cfg.max_dist_activity}px\n")

    results = []
    for tf in tracking_files:
        obs_id = tf.stem
        r = evaluate_observation(tf, annotations, obs_id)
        if r is not None:
            results.append(r)

    if not results:
        print("No observations with both tracking and annotations found.")
        return

    df = pd.DataFrame(results)

    # --- Per-observation table ---
    print(f"{'obs_id':>12s}  {'Y2F_rec':>7s}  {'B2F_rec':>7s}  "
          f"{'Y2F_pre':>7s}  {'B2F_pre':>7s}  "
          f"{'Y2F+':>5s}  {'B2F+':>5s}  {'frames':>6s}")
    print("-" * 75)
    for _, row in df.iterrows():
        y_rec = f"{row['Y2F_recall']:.2f}" if not np.isnan(row["Y2F_recall"]) else "  n/a"
        b_rec = f"{row['B2F_recall']:.2f}" if not np.isnan(row["B2F_recall"]) else "  n/a"
        y_pre = f"{row['Y2F_precision']:.2f}" if not np.isnan(row["Y2F_precision"]) else "  n/a"
        b_pre = f"{row['B2F_precision']:.2f}" if not np.isnan(row["B2F_precision"]) else "  n/a"
        print(f"{row['obs_id']:>12s}  {y_rec:>7s}  {b_rec:>7s}  "
              f"{y_pre:>7s}  {b_pre:>7s}  "
              f"{row['Y2F_n_pos']:5d}  {row['B2F_n_pos']:5d}  {row['n_frames']:6d}")

    # --- Aggregate (micro-averaged: pool all frames) ---
    print("-" * 75)
    for label in ["Y2F", "B2F"]:
        total_pos = df[f"{label}_n_pos"].sum()
        total_tp = df[f"{label}_tp"].sum()
        total_fp = df[f"{label}_fp"].sum()
        micro_recall = total_tp / total_pos if total_pos > 0 else float("nan")
        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else float("nan")
        macro_recall = df[f"{label}_recall"].dropna().mean()
        macro_precision = df[f"{label}_precision"].dropna().mean()
        n_obs_with_pos = df[f"{label}_n_pos"].gt(0).sum()
        print(f"  {label}  micro-recall={micro_recall:.3f}  micro-precision={micro_precision:.3f}  "
              f"(TP={total_tp}/{total_pos})  "
              f"macro-recall={macro_recall:.3f}  macro-precision={macro_precision:.3f}  "
              f"({n_obs_with_pos} obs with positives)")

    # --- Save results ---
    output_path = PROJECT_ROOT / f"results/tracking/{subject}/{version}/tracking_eval.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved per-observation results to {output_path}")


if __name__ == "__main__":
    main()
