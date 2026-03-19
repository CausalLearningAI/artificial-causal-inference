#!/usr/bin/env python3
"""
Run ant tracking on all standardized observation videos and save per-video
tracking CSVs.

HSV color bounds are read directly from the tracking config (blue_marking_lb /
blue_marking_ub / yellow_marking_lb / yellow_marking_ub).  Use the notebook
notebooks/tracking/ants_config_check.ipynb to estimate and tune bounds, then
save them to the config before running this script.

Output: dataset/{subject}/{version}/tracking/{obs_id}.csv
Columns: frame_idx, blue_x, blue_y, yellow_x, yellow_y,
         focal_x, focal_y, mark_blue_x, mark_blue_y,
         mark_yellow_x, mark_yellow_y, B2F, Y2F, n_blobs

Usage:
    # All videos for a version
    python src/tracking/get_tracking.py --config-name ants/v3
    python src/tracking/get_tracking.py --config-name ants/v1

    # Single video (for quick QC before running everything)
    python src/tracking/get_tracking.py --config-name ants/v3 obs_id=a4 overwrite=true

    # Override any algorithm param on the fly
    python src/tracking/get_tracking.py --config-name ants/v3 max_dist_activity=60
"""

import sys
import time
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tracking.calibration import get_background
from src.tracking.tracker import AntTracker


def build_tracker(cfg: DictConfig) -> AntTracker:
    return AntTracker(
        blue_lower=np.array(list(cfg.blue_marking_lb),    dtype=np.uint8),
        blue_upper=np.array(list(cfg.blue_marking_ub),    dtype=np.uint8),
        yellow_lower=np.array(list(cfg.yellow_marking_lb), dtype=np.uint8),
        yellow_upper=np.array(list(cfg.yellow_marking_ub), dtype=np.uint8),
        quantile=float(cfg.quantile),
        n_background_frames=int(cfg.n_background_frames),
        max_dist_activity=int(cfg.max_dist_activity),
        max_step=int(cfg.max_step),
        min_area_color=int(cfg.min_area_color),
        max_area_color=int(cfg.get("max_area_color", 400)),
        min_area_body=int(cfg.min_area_body),
        max_area_body=int(cfg.max_area_body),
    )


def _fmt(seconds: float) -> str:
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


@hydra.main(version_base=None, config_path="../../configs/tracking", config_name="ants/v3")
def main(cfg: DictConfig) -> None:
    subject = cfg.subject
    version = cfg.version
    # str() cast: OmegaConf may parse numeric-looking IDs (e.g. "3_10_1" → 3101)
    obs_id_filter = str(OmegaConf.select(cfg, "obs_id", default="") or "")
    overwrite     = bool(OmegaConf.select(cfg, "overwrite", default=False))

    observations_dir = PROJECT_ROOT / f"data/{subject}/{version}/observations/full"
    output_dir = PROJECT_ROOT / f"dataset/{subject}/{version}/tracking"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not observations_dir.exists():
        print(f"[ERROR] Observations directory not found: {observations_dir}")
        return

    all_video_files = sorted(
        list(observations_dir.glob("*.mkv")) + list(observations_dir.glob("*.mp4"))
    )
    if not all_video_files:
        print(f"[ERROR] No video files found in {observations_dir}")
        return

    # Filter to a single observation if obs_id is set (useful for QC)
    if obs_id_filter:
        video_files = [v for v in all_video_files if v.stem == obs_id_filter]
        if not video_files:
            print(f"[ERROR] obs_id='{obs_id_filter}' not found in {observations_dir}")
            return
        print(f"Tracking 1 video  ({subject}/{version}/{obs_id_filter})  [test mode]")
    else:
        video_files = all_video_files
        print(f"Tracking {len(video_files)} videos  ({subject}/{version})")

    print(f"  quantile={cfg.quantile}  proximity={cfg.max_dist_activity}px  "
          f"pov_radius={cfg.pov_radius}px")
    print(f"  blue  {list(cfg.blue_marking_lb)} → {list(cfg.blue_marking_ub)}")
    print(f"  yellow {list(cfg.yellow_marking_lb)} → {list(cfg.yellow_marking_ub)}")

    tracker = build_tracker(cfg)
    print(f"  Output → {output_dir}")

    ok, skipped, failed = 0, 0, 0
    start = time.time()

    for idx, video_path in enumerate(video_files, 1):
        obs_id = video_path.stem
        csv_path = output_dir / f"{obs_id}.csv"

        if csv_path.exists() and not overwrite:
            skipped += 1
            continue

        try:
            bg = get_background(version, obs_id)
            df = tracker.track_video(video_path, background=bg)
            df.to_csv(csv_path, index=False)
            ok += 1
        except Exception as e:
            print(f"  [FAIL] {obs_id}: {e}")
            failed += 1

        if idx % 10 == 0 or idx == len(video_files):
            elapsed = time.time() - start
            eta = elapsed / idx * (len(video_files) - idx)
            print(f"  [{idx:3d}/{len(video_files)}] {idx/len(video_files)*100:5.1f}%  "
                  f"elapsed={_fmt(elapsed)}  eta={_fmt(eta)}")

    print(f"\nDone: {ok} tracked, {skipped} skipped, {failed} failed  "
          f"({_fmt(time.time() - start)})")


if __name__ == "__main__":
    main()
