#!/usr/bin/env python3
"""Compute tracking quality metrics for the summary script.

Joins tracking CSVs with annotations to compute:
  - blue_det / yellow_det: color mark detection rate
  - suff_B2F / suff_Y2F: of annotated grooming frames, % where tracker agrees
  - blob distribution (0/1/2/3)

Usage:
    python src/tracking/tracking_summary.py ants v3
    python src/tracking/tracking_summary.py ants v4
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main(subject: str, version: str):
    base = Path(__file__).resolve().parents[2]
    track_dir = base / "dataset" / subject / version / "tracking"
    ann_path = base / "dataset" / subject / version / "annotations.csv"

    if not track_dir.exists():
        print("NOT_FOUND")
        return

    csvs = sorted(track_dir.glob("*.csv"))
    if not csvs:
        print("EMPTY")
        return

    # Load all tracking CSVs
    dfs = []
    for csv in csvs:
        obs_id = csv.stem
        df = pd.read_csv(csv)
        df["observation_id"] = obs_id
        dfs.append(df)
    track = pd.concat(dfs, ignore_index=True)

    n_videos = len(csvs)
    n_frames = len(track)

    # Detect blob column name
    blob_col = "n_blobs" if "n_blobs" in track.columns else "n_ants_detected"

    # Color detection rates
    raw_blue_det = track["raw_blue_x"].notna().mean() * 100
    raw_yellow_det = track["raw_yellow_x"].notna().mean() * 100

    # Blob distribution
    blob_counts = track[blob_col].value_counts(normalize=True).sort_index() * 100
    b0 = blob_counts.get(0, 0.0)
    b1 = blob_counts.get(1, 0.0)
    b2 = blob_counts.get(2, 0.0)
    b3 = blob_counts.get(3, 0.0)

    # Sufficiency: join with annotations
    suff_b2f = "N/A"
    suff_y2f = "N/A"
    if ann_path.exists():
        ann = pd.read_csv(ann_path)
        required_cols = {"observation_id", "frame_idx", "Y_B2F", "Y_Y2F"}
        if required_cols.issubset(set(ann.columns)):
            ann = ann[["observation_id", "frame_idx", "Y_B2F", "Y_Y2F"]]
            merged = track.merge(ann, on=["observation_id", "frame_idx"], how="inner")

            if len(merged) > 0:
                # suff_B2F: among frames where annotation says B2F grooming, % tracker agrees
                ann_b2f = merged["Y_B2F"] == 1
                if ann_b2f.sum() > 0:
                    suff_b2f = f"{merged.loc[ann_b2f, 'B2F'].mean() * 100:.0f}%"

                ann_y2f = merged["Y_Y2F"] == 1
                if ann_y2f.sum() > 0:
                    suff_y2f = f"{merged.loc[ann_y2f, 'Y2F'].mean() * 100:.0f}%"

    # Print compact output for the summary script to parse
    print(f"videos={n_videos} frames={n_frames}")
    print(f"blue_det={raw_blue_det:.0f}% yellow_det={raw_yellow_det:.0f}%")
    print(f"suff_B2F={suff_b2f} suff_Y2F={suff_y2f}")
    print(f"blobs: 0={b0:.1f}% 1={b1:.1f}% 2={b2:.1f}% 3={b3:.1f}%")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <subject> <version>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
