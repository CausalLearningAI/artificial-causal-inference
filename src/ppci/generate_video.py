#!/usr/bin/env python3
"""
Generate an annotated video for a given observation.

Reads the source video (original FPS) and the annotation CSV produced by
generate_annotations.py, then overlays a colored border:
  - Yellow border  → Y2F == 1  (only)
  - Blue border    → B2F == 1  (only)
  - Split border   → Y2F == 1 AND B2F == 1  (yellow top/bottom, blue left/right)
  - No border      → both 0

Uses ffmpeg with a drawbox filter expression — one pass, no audio.

Requires: ffmpeg (available as CLI command).

Usage:
  python src/ppci/generate_video.py --obs 5_17_7
  python src/ppci/generate_video.py --obs 5_17_7 --version v5
  python src/ppci/generate_video.py --obs 5_17_7 --border-width 20
  python src/ppci/generate_video.py --obs 5_17_7 --out /custom/output/5_17_7.mp4
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# ── constants ────────────────────────────────────────────────────────────────

BORDER_WIDTH = 15   # pixels


# ── helpers ──────────────────────────────────────────────────────────────────

def _find_source_video(version: str, obs_id: str) -> Path:
    """Locate the source video file for this observation."""
    source_dir = ROOT / "data" / "ants" / version / "observations" / "source"
    for ext in (".mkv", ".mp4", ".avi", ".mov"):
        p = source_dir / f"{obs_id}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Source video not found for obs '{obs_id}' in {source_dir}. "
        f"Tried extensions: .mkv .mp4 .avi .mov"
    )


def _load_valid_interval(version: str, obs_id: str) -> tuple[int, int]:
    """Return (start_frame, end_frame) at original FPS from experiment.csv."""
    exp_csv = ROOT / "data" / "ants" / version / "experiment.csv"
    df = pd.read_csv(exp_csv)
    row = df[df["observation_id"] == obs_id]
    if row.empty:
        raise ValueError(f"Observation '{obs_id}' not found in {exp_csv}")
    return int(row.iloc[0]["start_frame"]), int(row.iloc[0]["end_frame"])


def _build_drawbox_filter(ann: pd.DataFrame, bw: int, frame_offset: int = 0) -> str:
    """
    Build an ffmpeg drawbox filter string that colors borders per-frame.

    For each frame:
      - Y2F=1, B2F=0  → yellow rectangle
      - B2F=1, Y2F=0  → blue rectangle
      - both=1         → yellow top/bottom + blue left/right (4 boxes)
      - both=0         → no box

    ffmpeg drawbox uses: x=X:y=Y:w=W:h=H:color=C:t=T:enable='between(n,F,F)'
    We group consecutive runs of identical (y2f, b2f) to minimize filter length.
    """
    if ann.empty:
        return "null"

    # Build run-length encoding over (Y2F, B2F) pairs
    frames = ann[["Y2F", "B2F"]].values  # (N, 2) int
    runs = []  # (start_frame, end_frame, y2f, b2f)
    i = 0
    while i < len(frames):
        y2f, b2f = int(frames[i, 0]), int(frames[i, 1])
        j = i + 1
        while j < len(frames) and int(frames[j, 0]) == y2f and int(frames[j, 1]) == b2f:
            j += 1
        runs.append((ann.index[i], ann.index[j - 1], y2f, b2f))
        i = j

    boxes = []
    # ffmpeg frame numbering starts at 0 within the trimmed clip,
    # so subtract frame_offset (= start_frame) from absolute frame IDs.
    for (f_start, f_end, y2f, b2f) in runs:
        if y2f == 0 and b2f == 0:
            continue
        enable = f"between(n,{f_start - frame_offset},{f_end - frame_offset})"
        if y2f == 1 and b2f == 0:
            # Full yellow border
            boxes.append(
                f"drawbox=x=0:y=0:w=iw:h=ih:color=yellow@1.0:t={bw}:enable='{enable}'"
            )
        elif b2f == 1 and y2f == 0:
            # Full blue border
            boxes.append(
                f"drawbox=x=0:y=0:w=iw:h=ih:color=blue@1.0:t={bw}:enable='{enable}'"
            )
        else:
            # Both: yellow top strip, yellow bottom strip, blue left strip, blue right strip
            boxes.append(
                f"drawbox=x=0:y=0:w=iw:h={bw}:color=yellow@1.0:t=fill:enable='{enable}'"
            )
            boxes.append(
                f"drawbox=x=0:y=ih-{bw}:w=iw:h={bw}:color=yellow@1.0:t=fill:enable='{enable}'"
            )
            boxes.append(
                f"drawbox=x=0:y=0:w={bw}:h=ih:color=blue@1.0:t=fill:enable='{enable}'"
            )
            boxes.append(
                f"drawbox=x=iw-{bw}:y=0:w={bw}:h=ih:color=blue@1.0:t=fill:enable='{enable}'"
            )

    if not boxes:
        return "null"
    return ",".join(boxes)


def generate_video(
    obs_id: str,
    version: str,
    annotations_dir: Path,
    out_path: Path,
    border_width: int = BORDER_WIDTH,
) -> None:
    # ── load annotations ──────────────────────────────────────────────────────
    ann_path = annotations_dir / f"{obs_id}.csv"
    if not ann_path.exists():
        raise FileNotFoundError(
            f"Annotations not found: {ann_path}\n"
            f"Run generate_annotations.py first."
        )
    ann = pd.read_csv(ann_path)
    ann = ann.set_index("frame_id")

    # ── locate source video ───────────────────────────────────────────────────
    src_video = _find_source_video(version, obs_id)
    print(f"Source video : {src_video}")
    print(f"Annotations  : {ann_path}  ({len(ann)} frames)")

    # ── read valid interval from experiment.csv ───────────────────────────────
    start_frame, end_frame = _load_valid_interval(version, obs_id)
    original_fps = 30  # from data config
    start_sec = start_frame / original_fps
    end_sec   = end_frame   / original_fps
    print(f"Valid interval: frames {start_frame}–{end_frame} "
          f"({start_sec:.1f}s – {end_sec:.1f}s)")

    # Restrict annotations to the valid interval
    ann = ann[(ann.index >= start_frame) & (ann.index < end_frame)]

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── build ffmpeg drawbox filter ───────────────────────────────────────────
    # frame_offset shifts absolute frame IDs to ffmpeg's 0-based counter within
    # the trimmed clip.
    vf = _build_drawbox_filter(ann, border_width, frame_offset=start_frame)
    n_active = int(((ann["Y2F"] == 1) | (ann["B2F"] == 1)).sum())
    print(f"Active border frames: {n_active} / {len(ann)}")

    # ── run ffmpeg ────────────────────────────────────────────────────────────
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_sec),
        "-to", str(end_sec),
        "-i", str(src_video),
        "-vf", vf,
        "-c:v", "libopenh264",
        "-b:v", "4M",
        "-an",                    # no audio
        str(out_path),
    ]
    print(f"Running ffmpeg ...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] ffmpeg failed:\n{result.stderr[-2000:]}", file=sys.stderr)
        sys.exit(1)

    print(f"Output video : {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate annotation-overlay video for one observation.",
    )
    parser.add_argument("--obs", required=True, metavar="OBS_ID",
                        help="Observation ID, e.g. '5_17_7'")
    parser.add_argument("--version", default="v5",
                        help="Dataset version (default: v5)")
    parser.add_argument("--annotations-dir", type=Path, default=None,
                        help="Directory with per-obs annotation CSVs "
                             "(default: results/ppci/ants/annotations/{version}/)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output video path "
                             "(default: results/ppci/ants/annotated_videos/{version}/{obs_id}.mp4)")
    parser.add_argument("--border-width", type=int, default=BORDER_WIDTH,
                        help=f"Border thickness in pixels (default: {BORDER_WIDTH})")
    args = parser.parse_args()

    annotations_dir = (
        args.annotations_dir
        or ROOT / "results" / "ppci" / "ants" / "annotations" / args.version
    )
    out_path = (
        args.out
        or ROOT / "results" / "ppci" / "ants" / "annotated_videos" / args.version / f"{args.obs}.mp4"
    )

    generate_video(
        obs_id=args.obs,
        version=args.version,
        annotations_dir=annotations_dir,
        out_path=out_path,
        border_width=args.border_width,
    )


if __name__ == "__main__":
    main()
