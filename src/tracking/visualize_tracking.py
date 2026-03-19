#!/usr/bin/env python3
"""
Generate tracking demo videos (and optional PNG snapshots) for quality inspection.

PRIMARY OUTPUT  — annotated .mp4 for each sampled observation:
    results/{subject}/{version}/tracking_viz/{obs_id}.mp4

SECONDARY OUTPUT — side-by-side PNG grid of evenly-spaced snapshots:
    results/{subject}/{version}/tracking_viz/{obs_id}_grid.png

The MP4 video is the main diagnostic tool:
  - Watch it frame by frame to spot identity swaps, missed ants, drift.
  - Color coding: BLUE circle = blue nestmate, YELLOW circle = yellow nestmate,
    GREEN circle = focal ant.  Dashed ring = POV crop radius.
  - Top-left text shows frame index and n_ants_detected.

Usage:
    # 5 random experiments, full video (default)
    python src/tracking/visualize_tracking.py ants/v3

    # All experiments
    python src/tracking/visualize_tracking.py ants/v3 n_sample=-1

    # Specific experiment
    python src/tracking/visualize_tracking.py ants/v3 obs_id=a4

    # Short 60-second clip (faster review for long experiments)
    python src/tracking/visualize_tracking.py ants/v3 max_seconds=60

    # Full videos + PNG grids for all experiments
    python src/tracking/visualize_tracking.py ants/v3 n_sample=-1 save_grid=true

Viewing the videos:
    # From terminal (requires ffplay):
    ffplay results/ants/v3/tracking_viz/a4.mp4

    # Or open in any video player (VLC, mpv, QuickTime, etc.)
    # In VSCode: install "Video Player" extension, or just open a terminal.
"""

import random
import subprocess
import sys
from pathlib import Path
from typing import Optional

import cv2
import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tracking.tracker import AntTracker

# Target FPS for demo videos (should match pipeline target_fps)
_DEFAULT_FPS = 5.0


def _available_ffmpeg_encoders() -> set[str]:
    try:
        out = subprocess.check_output(
            ["ffmpeg", "-hide_banner", "-encoders"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return set()

    encoders: set[str] = set()
    for line in out.splitlines():
        parts = line.strip().split()
        if len(parts) >= 2 and parts[0].startswith("V"):
            encoders.add(parts[1])
    return encoders


def _codec_candidates() -> list[str]:
    available = _available_ffmpeg_encoders()
    preferred = ["libx264", "libopenh264", "mpeg4"]
    if not available:
        return preferred
    return [c for c in preferred if c in available] or preferred


def _annotate_frame(
    frame_bgr: np.ndarray, row: pd.Series, pov_radius: int
) -> np.ndarray:
    import math

    def _raw(key: str):
        v = row.get(key, float("nan"))
        return None if (v is None or (isinstance(v, float) and math.isnan(v))) else float(v)

    raw_bx, raw_by = _raw("raw_blue_x"),   _raw("raw_blue_y")
    raw_yx, raw_yy = _raw("raw_yellow_x"), _raw("raw_yellow_y")

    return AntTracker.draw_tracking(
        frame_bgr,
        blue_pos=(row["blue_x"],   row["blue_y"]),
        yellow_pos=(row["yellow_x"], row["yellow_y"]),
        focal_pos=(row["focal_x"],  row["focal_y"]),
        pov_radius=pov_radius,
        frame_idx=int(row["frame_idx"]),
        n_detected=int(row.get("n_blobs", row.get("n_ants_detected", 3))),
        raw_blue_pos=(raw_bx, raw_by)     if raw_bx is not None else None,
        raw_yellow_pos=(raw_yx, raw_yy)   if raw_yx is not None else None,
    )


def write_demo_video(
    obs_id: str,
    frame_files: list[Path],
    tracking_df: pd.DataFrame,
    output_path: Path,
    pov_radius: int,
    max_frames: int,
    fps: float,
) -> None:
    """
    Write annotated H.264 MP4 for up to max_frames frames (-1 = all).

    Frames are piped directly to ffmpeg (libx264, yuv420p) so the output
    plays in VSCode, browsers, and all standard video players without
    needing a codec plugin.
    """
    if not frame_files:
        return
    sample = cv2.imread(str(frame_files[0]))
    if sample is None:
        return
    h, w = sample.shape[:2]

    n = min(len(frame_files), len(tracking_df))
    if max_frames > 0:
        n = min(n, max_frames)

    last_err = ""
    last_ret = -1
    tried: list[str] = []

    for codec in _codec_candidates():
        tried.append(codec)
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{w}x{h}",
            "-pix_fmt", "bgr24",
            "-r", str(fps),
            "-i", "pipe:0",
            "-vcodec", codec,
            "-pix_fmt", "yuv420p",   # broadest player compatibility
            str(output_path),
        ]
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        broken_pipe = False
        try:
            for fi in range(n):
                frame_bgr = cv2.imread(str(frame_files[fi]))
                if frame_bgr is None:
                    continue
                proc.stdin.write(_annotate_frame(frame_bgr, tracking_df.iloc[fi], pov_radius).tobytes())
        except BrokenPipeError:
            broken_pipe = True
        finally:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()

        ret = proc.wait()
        err = ""
        if proc.stderr:
            err = proc.stderr.read().decode("utf-8", errors="ignore")

        if ret == 0 and not broken_pipe:
            if codec != "libx264":
                print(f"  [demo] using ffmpeg codec fallback: {codec}")
            return

        last_ret = ret
        last_err = err

        # If encoder is missing, try next candidate; otherwise stop immediately.
        if "Unknown encoder" in err or "Encoder not found" in err:
            continue
        break

    tail = "\n".join((last_err or "").splitlines()[-20:])
    raise RuntimeError(
        f"ffmpeg failed while writing demo video ({output_path}). "
        f"returncode={last_ret}. tried_codecs={tried}.\n{tail}"
    )


def write_snapshot_grid(
    obs_id: str,
    frame_files: list[Path],
    tracking_df: pd.DataFrame,
    output_path: Path,
    pov_radius: int,
    n_snapshots: int,
) -> None:
    """Write a side-by-side PNG of evenly-spaced annotated frames."""
    n = min(len(frame_files), len(tracking_df))
    if n == 0:
        return
    indices = np.linspace(0, n - 1, n_snapshots, dtype=int)
    panels = []
    for fi in indices:
        frame_bgr = cv2.imread(str(frame_files[fi]))
        if frame_bgr is None:
            continue
        panels.append(_annotate_frame(frame_bgr, tracking_df.iloc[fi], pov_radius))
    if not panels:
        return
    grid = np.concatenate(panels, axis=1)
    # Label bar
    bar = np.zeros((24, grid.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, obs_id, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (220, 220, 220), 1, cv2.LINE_AA)
    cv2.imwrite(str(output_path), np.concatenate([bar, grid], axis=0))


@hydra.main(version_base=None, config_path="../../configs/tracking", config_name="ants/v3")
def main(cfg: DictConfig) -> None:
    subject = cfg.subject
    version = cfg.version

    frames_base  = PROJECT_ROOT / f"dataset/{subject}/{version}/frames/full"
    tracking_dir = PROJECT_ROOT / f"dataset/{subject}/{version}/tracking"
    output_dir   = PROJECT_ROOT / f"results/{subject}/{version}/tracking_viz"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extra CLI params with defaults
    n_sample     = int(OmegaConf.select(cfg, "n_sample",     default=5))
    max_seconds  = float(OmegaConf.select(cfg, "max_seconds", default=-1.0))   # -1 = full video
    save_grid    = bool(OmegaConf.select(cfg, "save_grid",    default=False))
    n_snapshots  = int(OmegaConf.select(cfg, "n_snapshots",  default=7))
    obs_id_arg   = str(OmegaConf.select(cfg, "obs_id",       default=""))
    seed         = int(OmegaConf.select(cfg, "seed",         default=42))
    fps          = float(OmegaConf.select(cfg, "fps",         default=_DEFAULT_FPS))
    skip_existing = bool(OmegaConf.select(cfg, "skip_existing", default=False))
    pov_radius   = int(cfg.pov_radius)

    max_frames = int(max_seconds * fps) if max_seconds > 0 else -1

    if not tracking_dir.exists():
        print(f"[ERROR] Tracking directory not found: {tracking_dir}")
        print("  Run get_tracking.py first.")
        return

    csv_files = sorted(tracking_dir.glob("*.csv"))
    if not csv_files:
        print(f"[ERROR] No tracking CSVs in {tracking_dir}")
        return

    if obs_id_arg:
        obs_ids = [obs_id_arg]
    else:
        all_ids = [f.stem for f in csv_files]
        if n_sample < 0 or n_sample >= len(all_ids):
            obs_ids = all_ids
        else:
            random.seed(seed)
            obs_ids = sorted(random.sample(all_ids, n_sample))

    clip_label = f"{max_seconds:.0f}s clip" if max_frames > 0 else "full video"
    print(f"Visualizing {len(obs_ids)} observations ({clip_label}) → {output_dir}")

    for obs_id in obs_ids:
        csv_path = tracking_dir / f"{obs_id}.csv"
        if not csv_path.exists():
            print(f"  [SKIP] No CSV: {obs_id}")
            continue
        tracking_df = pd.read_csv(csv_path)

        frame_files = sorted((frames_base / obs_id).glob("frame_*.jpg"))
        if not frame_files:
            print(f"  [SKIP] No frames: {obs_id}")
            continue

        n = min(len(frame_files), len(tracking_df))
        print(f"  {obs_id}  ({n} frames)", end="", flush=True)

        # Primary: demo video
        video_path = output_dir / f"{obs_id}.mp4"
        if skip_existing and video_path.exists():
            print("  [SKIP existing demo]", end="")
            if save_grid:
                grid_path = output_dir / f"{obs_id}_grid.png"
                if grid_path.exists():
                    print("  [SKIP existing grid]", end="")
            print()
            continue

        try:
            write_demo_video(obs_id, frame_files, tracking_df, video_path,
                             pov_radius, max_frames, fps)
            print(f"  → {video_path.name}", end="")
        except Exception as exc:
            print(f"  [FAIL demo: {exc}]", end="")
            print()
            continue

        # Secondary: PNG grid (opt-in)
        if save_grid:
            grid_path = output_dir / f"{obs_id}_grid.png"
            write_snapshot_grid(obs_id, frame_files, tracking_df, grid_path,
                                pov_radius, n_snapshots)
            print(f"  + {grid_path.name}", end="")

        print()

    print(f"\nTo view:")
    print(f"  ffplay {output_dir}/<obs_id>.mp4")
    print(f"  # or open in VLC / any video player")
    print(f"\nWhat to check in the video:")
    print(f"  - BLUE circle (filled)  = matched body of blue-marked ant")
    print(f"  - BLUE cross (+)        = raw HSV dot detection (shown when fired)")
    print(f"  - YELLOW circle/cross   = same for yellow/orange-marked ant")
    print(f"  - GREY circle           = focal (unmarked) ant")
    print(f"  - Ring = POV crop boundary (radius={pov_radius}px)")
    print(f"  - 'n_ants=3' throughout (drops to 1-2 when ants clump)")
    print(f"  - No persistent identity swaps between blue/yellow")
    print(f"  - Cross should be on or very near the circle (dot on body)")
    print(f"  - Frames with NO cross = dot not detected that frame (anchor held)")


if __name__ == "__main__":
    main()
