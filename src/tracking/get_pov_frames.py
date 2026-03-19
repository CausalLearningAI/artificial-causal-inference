#!/usr/bin/env python3
"""
Extract POV (point-of-view) crop frames from full frames using tracking data.

For each observation, reads the tracking CSV and the pre-extracted full JPG
frames, then saves a centred square crop around each ant for every frame.

Blue/yellow POVs are centred on the persistent **mark position**
(mark_blue_x/y, mark_yellow_x/y) from the tracking CSV.  These are never
NaN — they track the raw colour detection when visible and stay leashed to
the body centroid otherwise.

Output:
    dataset/{subject}/{version}/frames/pov/blue/{obs_id}/frame_*.jpg
    dataset/{subject}/{version}/frames/pov/yellow/{obs_id}/frame_*.jpg

The crop size is 2*pov_radius x 2*pov_radius pixels (matches AntTracker.crop_pov).
Black padding is added when the crop extends beyond frame boundaries.

Usage:
    # All observations
    python src/tracking/get_pov_frames.py --config-name ants/v3

    # Single observation (QC)
    python src/tracking/get_pov_frames.py --config-name ants/v3 "obs_id='3_10_1'"

    # Re-generate existing outputs
    python src/tracking/get_pov_frames.py --config-name ants/v3 overwrite=true
"""

import sys
import time
from pathlib import Path

import cv2
import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tracking.tracker import AntTracker


def _fmt(seconds: float) -> str:
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


_IDENTITIES = ("blue", "yellow")


def extract_pov_frames(
    obs_id: str,
    tracking_csv: Path,
    frames_dir: Path,
    pov_base: Path,
    pov_radius: int,
    overwrite: bool = False,
) -> tuple[int, int]:
    """
    Extract POV crops for one observation (one crop per identity per frame).

    Returns (n_saved, n_skipped).
    """
    # Create per-identity output dirs up front (color-first layout)
    color_dirs = {}
    for color in _IDENTITIES:
        d = pov_base / color / obs_id
        d.mkdir(parents=True, exist_ok=True)
        color_dirs[color] = d

    df = pd.read_csv(tracking_csv)
    frame_files = sorted(frames_dir.glob("frame_*.jpg"))
    if not frame_files:
        return 0, 0

    n = min(len(frame_files), len(df))
    saved, skipped = 0, 0

    for fi in range(n):
        fname = frame_files[fi].name
        # Skip if all requested identity crops already exist
        if not overwrite and all(
            (color_dirs[c] / fname).exists() for c in _IDENTITIES
        ):
            skipped += 1
            continue

        frame_bgr = cv2.imread(str(frame_files[fi]))
        if frame_bgr is None:
            continue

        row = df.iloc[fi]

        # Blue/yellow: persistent mark positions (never NaN).
        _POS_COLS = {
            "blue":   ("mark_blue_x",   "mark_blue_y"),
            "yellow": ("mark_yellow_x", "mark_yellow_y"),
        }
        for color in _IDENTITIES:
            col_x, col_y = _POS_COLS[color]
            cx = float(row[col_x])
            cy = float(row[col_y])

            crop = AntTracker.crop_pov(frame_bgr, cx, cy, pov_radius)
            cv2.imwrite(str(color_dirs[color] / fname), crop)

        saved += 1

    return saved, skipped


@hydra.main(version_base=None, config_path="../../configs/tracking", config_name="ants/v3")
def main(cfg: DictConfig) -> None:
    subject    = cfg.subject
    version    = cfg.version
    obs_id_arg = str(OmegaConf.select(cfg, "obs_id", default="") or "")
    overwrite  = bool(OmegaConf.select(cfg, "overwrite", default=False))
    pov_radius = int(cfg.pov_radius)

    tracking_dir = PROJECT_ROOT / f"dataset/{subject}/{version}/tracking"
    frames_base  = PROJECT_ROOT / f"dataset/{subject}/{version}/frames/full"
    pov_base     = PROJECT_ROOT / f"dataset/{subject}/{version}/frames/pov"

    if not tracking_dir.exists():
        print(f"[ERROR] Tracking directory not found: {tracking_dir}")
        print("  Run get_tracking.py first.")
        return

    csv_files = sorted(tracking_dir.glob("*.csv"))
    if not csv_files:
        print(f"[ERROR] No tracking CSVs in {tracking_dir}")
        return

    if obs_id_arg:
        csv_files = [f for f in csv_files if f.stem == obs_id_arg]
        if not csv_files:
            print(f"[ERROR] obs_id='{obs_id_arg}' not found in {tracking_dir}")
            return

    print(f"Extracting POV frames: {subject}/{version}")
    print(f"  pov_radius: {pov_radius}px  →  {2*pov_radius}×{2*pov_radius} crops")
    print(f"  {len(csv_files)} observations  →  {pov_base}")
    print(f"  overwrite: {overwrite}")

    total_saved, total_skipped = 0, 0
    start = time.time()

    for idx, csv_path in enumerate(csv_files, 1):
        obs_id = csv_path.stem
        frames_dir = frames_base / obs_id

        if not frames_dir.exists():
            print(f"  [SKIP] No full frames: {obs_id}")
            continue

        saved, skipped = extract_pov_frames(
            obs_id, csv_path, frames_dir, pov_base, pov_radius, overwrite
        )
        total_saved   += saved
        total_skipped += skipped

        if idx % 20 == 0 or idx == len(csv_files):
            elapsed = time.time() - start
            eta = elapsed / idx * (len(csv_files) - idx)
            print(f"  [{idx:3d}/{len(csv_files)}] {idx/len(csv_files)*100:5.1f}%  "
                  f"elapsed={_fmt(elapsed)}  eta={_fmt(eta)}")

    elapsed = time.time() - start
    print(f"\nDone: {total_saved} saved, {total_skipped} skipped  ({_fmt(elapsed)})")
    print(f"  Output: {pov_base}/")


if __name__ == "__main__":
    main()
