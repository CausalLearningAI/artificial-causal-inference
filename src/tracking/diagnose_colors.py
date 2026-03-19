#!/usr/bin/env python3
"""
Color-dot diagnostic for ant tracking.

Runs the full color estimation pipeline on one video and produces:

  results/{subject}/{version}/color_diag/{obs_id}_masks.png
      Grid of sample frames showing raw HSV masks and detected blobs.
      Blue channel left, yellow channel right; rows = sample frames.
      Green contour = accepted blob.  Red = rejected (too small/large).

  results/{subject}/{version}/color_diag/{obs_id}_hsv_hist.png
      HSV distributions (H, S, V) of the ACCEPTED dot pixels across all
      sample frames.  Dashed vertical lines = estimated bounds.
      Shows whether the bounds are tight, loose, or biased.

  results/{subject}/{version}/color_diag/{obs_id}_detection_rate.png
      Per-frame detection flag (blue / yellow detected: 1 or 0).
      Helps identify periods where detection is lost.

  stdout summary
      Estimated bounds, detection counts, morphological closing effect.

Usage:
    python src/tracking/diagnose_colors.py --config-name ants/v3 "obs_id='3_10_1'"
    python src/tracking/diagnose_colors.py --config-name ants/v1 "obs_id='a1'" +n_sample=60
"""

import sys
from pathlib import Path

import cv2
import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tracking.tracker import (
    AntTracker,
    _MAX_DOT_AREA_FRACTION,
    _SEED_RANGES,
    _sample_color_pixels,
    estimate_color_bounds,
)

# ── helpers ───────────────────────────────────────────────────────────────────

def _color_mask_on_frame(
    frame_bgr: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    min_area: int,
    color_bgr: tuple,
) -> np.ndarray:
    """
    Return a copy of frame with:
      - color mask region darkened to 30 % brightness
      - accepted blobs outlined in green
      - rejected blobs (too small or too large) outlined in red
    """
    vis = frame_bgr.copy()
    hsv  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    # Apply same morphological closing used in tracker
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Darken non-mask pixels to 40 %
    non_mask = (mask == 0)
    vis[non_mask] = (vis[non_mask] * 0.35).astype(np.uint8)
    # Tint mask pixels with the dot color
    vis[mask > 0] = (vis[mask > 0] * 0.5 + np.array(color_bgr) * 0.5).astype(np.uint8)

    frame_area = frame_bgr.shape[0] * frame_bgr.shape[1]
    max_area   = frame_area * _MAX_DOT_AREA_FRACTION
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    accepted, rejected = [], []
    for cnt in contours:
        a = cv2.contourArea(cnt)
        if a >= min_area and a <= max_area:
            accepted.append(cnt)
        elif a > 0:
            rejected.append(cnt)

    cv2.drawContours(vis, accepted, -1, (0, 220, 0),  2)
    cv2.drawContours(vis, rejected, -1, (0, 0, 220),  1)

    return vis, len(accepted) > 0


def _collect_dot_pixels(
    cap: cv2.VideoCapture,
    lower: np.ndarray,
    upper: np.ndarray,
    n_frames: int,
    min_area: int,
) -> tuple:
    """
    Sample n_frames evenly, collect HSV pixels of accepted dot blobs.
    Returns (all_pixels ndarray shape (N,3), per_frame_detected list[bool]).
    """
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, frame_count - 1, n_frames, dtype=int)
    frame_area = None
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    all_pixels  = []
    per_detected = []

    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, frame = cap.read()
        if not ret:
            per_detected.append(False)
            continue
        if frame_area is None:
            frame_area = frame.shape[0] * frame.shape[1]
        max_area = frame_area * _MAX_DOT_AREA_FRACTION

        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        found = False
        for cnt in contours:
            a = cv2.contourArea(cnt)
            if a >= min_area and a <= max_area:
                bm = np.zeros(mask.shape, dtype=np.uint8)
                cv2.drawContours(bm, [cnt], -1, 255, -1)
                pixels = hsv[bm == 255]
                if len(pixels):
                    all_pixels.append(pixels)
                    found = True
        per_detected.append(found)

    px = np.concatenate(all_pixels, axis=0) if all_pixels else np.empty((0, 3))
    return px, per_detected


# ── main diagnostic ───────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="../../configs/tracking", config_name="ants/v3")
def main(cfg: DictConfig) -> None:
    subject  = cfg.subject
    version  = cfg.version
    obs_id   = str(OmegaConf.select(cfg, "obs_id", default=""))
    n_sample = int(OmegaConf.select(cfg, "n_sample", default=40))
    n_grid   = int(OmegaConf.select(cfg, "n_grid",   default=8))
    min_area_color = int(cfg.min_area_color)

    obs_dir = PROJECT_ROOT / f"data/{subject}/{version}/observations/full"
    if not obs_id:
        videos = sorted(obs_dir.glob("*.mkv")) + sorted(obs_dir.glob("*.mp4"))
        if not videos:
            print(f"[ERROR] No videos in {obs_dir}"); return
        video_path = videos[0]
        obs_id = video_path.stem
    else:
        video_path = obs_dir / f"{obs_id}.mkv"
        if not video_path.exists():
            video_path = obs_dir / f"{obs_id}.mp4"
    if not video_path.exists():
        print(f"[ERROR] Video not found: {obs_id}"); return

    out_dir = PROJECT_ROOT / f"results/{subject}/{version}/color_diag"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDiagnosing color detection: {subject}/{version}/{obs_id}")
    print(f"  video: {video_path}")
    print(f"  n_sample frames: {n_sample}  |  grid rows: {n_grid}")
    print(f"  min_area_color: {min_area_color}")

    # ── Step 1: estimate bounds (with optional fixed overrides from yaml) ────
    print("\n── Estimating color bounds ──")
    bounds = estimate_color_bounds(video_path)

    # Apply per-version fixed overrides (e.g. v1 yellow wraparound)
    for key, attr in [
        ("blue_lower",   "fixed_blue_lower"),
        ("blue_upper",   "fixed_blue_upper"),
        ("yellow_lower", "fixed_yellow_lower"),
        ("yellow_upper", "fixed_yellow_upper"),
    ]:
        val = OmegaConf.select(cfg, attr, default=None)
        if val is not None:
            bounds[key] = np.array(list(val), dtype=np.uint8)
            print(f"  [{attr} override applied: {bounds[key]}]")

    for color in ("blue", "yellow"):
        lo = bounds[f"{color}_lower"]
        hi = bounds[f"{color}_upper"]
        seed_lo = _SEED_RANGES[color]["lower"]
        seed_hi = _SEED_RANGES[color]["upper"]
        print(f"  {color:6s}  estimated lower={lo}  upper={hi}")
        print(f"         seed    lower={seed_lo}  upper={seed_hi}")
        if np.array_equal(lo, seed_lo) and np.array_equal(hi, seed_hi):
            print(f"  *** WARNING: {color} fell back to seed range — no dots found during estimation!")

    # ── Step 2: collect pixel distributions ──────────────────────────────────
    print("\n── Collecting HSV pixel distributions ──")
    cap = cv2.VideoCapture(str(video_path))

    blue_px,   blue_det   = _collect_dot_pixels(
        cap, bounds["blue_lower"],   bounds["blue_upper"],   n_sample, min_area_color)
    yellow_px, yellow_det = _collect_dot_pixels(
        cap, bounds["yellow_lower"], bounds["yellow_upper"], n_sample, min_area_color)

    n_b = sum(blue_det)
    n_y = sum(yellow_det)
    print(f"  blue   detected in {n_b}/{n_sample} frames  ({100*n_b/n_sample:.0f}%)")
    print(f"  yellow detected in {n_y}/{n_sample} frames  ({100*n_y/n_sample:.0f}%)")
    if len(blue_px):
        print(f"  blue   pixel stats  H={blue_px[:,0].mean():.0f}±{blue_px[:,0].std():.0f}"
              f"  S={blue_px[:,1].mean():.0f}±{blue_px[:,1].std():.0f}"
              f"  V={blue_px[:,2].mean():.0f}±{blue_px[:,2].std():.0f}")
    if len(yellow_px):
        print(f"  yellow pixel stats  H={yellow_px[:,0].mean():.0f}±{yellow_px[:,0].std():.0f}"
              f"  S={yellow_px[:,1].mean():.0f}±{yellow_px[:,1].std():.0f}"
              f"  V={yellow_px[:,2].mean():.0f}±{yellow_px[:,2].std():.0f}")

    # ── Step 3: HSV distribution plot ────────────────────────────────────────
    print("\n── Generating HSV histogram plot ──")
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    fig.suptitle(f"HSV dot-pixel distributions — {subject}/{version}/{obs_id}", fontsize=12)

    channel_names  = ["Hue (0–179)", "Saturation (0–255)", "Value (0–255)"]
    channel_ranges = [(0, 179), (0, 255), (0, 255)]
    bins_per_ch    = [90, 64, 64]

    for row, (color, pixels, lo, hi) in enumerate([
        ("blue",   blue_px,   bounds["blue_lower"],   bounds["blue_upper"]),
        ("yellow", yellow_px, bounds["yellow_lower"], bounds["yellow_upper"]),
    ]):
        c_rgb = (0.3, 0.5, 0.9) if color == "blue" else (0.9, 0.75, 0.1)
        for col, (ch, cname, crange, nbins) in enumerate(
                zip(range(3), channel_names, channel_ranges, bins_per_ch)):
            ax = axes[row, col]
            ax.set_title(f"{color}  –  {cname}", fontsize=9)
            if len(pixels) > 0:
                ax.hist(pixels[:, ch], bins=nbins, range=crange,
                        color=c_rgb, alpha=0.75, density=True)
                ax.axvline(lo[ch], color="navy" if color=="blue" else "goldenrod",
                           ls="--", lw=1.5, label=f"lower={lo[ch]}")
                ax.axvline(hi[ch], color="navy" if color=="blue" else "goldenrod",
                           ls="-",  lw=1.5, label=f"upper={hi[ch]}")
                ax.legend(fontsize=7)
            else:
                ax.text(0.5, 0.5, "no pixels detected",
                        ha="center", va="center", transform=ax.transAxes,
                        color="red", fontsize=10)
            ax.set_xlim(crange)
            ax.set_yticks([])

    plt.tight_layout()
    hist_path = out_dir / f"{obs_id}_hsv_hist.png"
    plt.savefig(hist_path, dpi=130)
    plt.close()
    print(f"  saved → {hist_path}")

    # ── Step 4: detection-rate timeline ──────────────────────────────────────
    print("\n── Generating detection-rate timeline ──")
    fig, ax = plt.subplots(figsize=(14, 2.5))
    xs = np.arange(n_sample)
    ax.bar(xs, [1 if d else 0 for d in blue_det],
           color=(0.3, 0.5, 0.9), alpha=0.7, label="blue detected")
    ax.bar(xs, [-1 if d else 0 for d in yellow_det],
           color=(0.9, 0.75, 0.1), alpha=0.7, label="yellow detected")
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_xlim(0, n_sample)
    ax.set_ylim(-1.5, 1.5)
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(["yellow\ndetected", "", "blue\ndetected"])
    ax.set_xlabel("Sample frame index")
    ax.set_title(f"Per-frame dot detection — {subject}/{version}/{obs_id}  "
                 f"(blue {100*n_b/n_sample:.0f}%  yellow {100*n_y/n_sample:.0f}%)")
    plt.tight_layout()
    rate_path = out_dir / f"{obs_id}_detection_rate.png"
    plt.savefig(rate_path, dpi=130)
    plt.close()
    print(f"  saved → {rate_path}")

    # ── Step 5: sample frame mask grid ───────────────────────────────────────
    print("\n── Generating mask grid ──")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    grid_indices = np.linspace(0, frame_count - 1, n_grid, dtype=int)

    C_BLUE   = (180, 60,  60)
    C_YELLOW = (40,  190, 190)

    panels = []
    for fi in grid_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ret, frame = cap.read()
        if not ret:
            continue

        vis_b, det_b = _color_mask_on_frame(
            frame, bounds["blue_lower"],   bounds["blue_upper"],   min_area_color, C_BLUE)
        vis_y, det_y = _color_mask_on_frame(
            frame, bounds["yellow_lower"], bounds["yellow_upper"], min_area_color, C_YELLOW)

        h, w = frame.shape[:2]
        sep = np.full((h, 4, 3), 40, dtype=np.uint8)
        row_img = np.concatenate([vis_b, sep, vis_y], axis=1)

        bar = np.zeros((20, row_img.shape[1], 3), dtype=np.uint8)
        label = (f"frame {fi}   "
                 f"blue={'OK' if det_b else '--'}   "
                 f"yellow={'OK' if det_y else '--'}")
        cv2.putText(bar, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (220, 220, 220), 1, cv2.LINE_AA)
        panels.append(np.concatenate([bar, row_img], axis=0))

    if panels:
        grid = np.concatenate(panels, axis=0)
        masks_path = out_dir / f"{obs_id}_masks.png"
        cv2.imwrite(str(masks_path), grid)
        print(f"  saved → {masks_path}")
        print(f"  (left column = blue mask, right column = yellow mask)")
        print(f"   green contour = accepted blob, red contour = rejected blob)")

    cap.release()

    print(f"\n── Summary ──")
    print(f"  Blue   detection rate : {100*n_b/n_sample:.0f}%")
    print(f"  Yellow detection rate : {100*n_y/n_sample:.0f}%")
    if n_b < n_sample * 0.5:
        print("  *** blue detection is weak (<50%) — check hsv_hist.png")
        print("      possible fixes: widen bounds in yaml, or lower min_area_color")
    if n_y < n_sample * 0.5:
        print("  *** yellow detection is weak (<50%) — check hsv_hist.png")
    print(f"\n  All outputs in: {out_dir}/")
    print(f"  Recommended workflow:")
    print(f"    1. Open {obs_id}_masks.png — verify green contours land on the dots")
    print(f"    2. Open {obs_id}_hsv_hist.png — check dashed bounds capture the histogram peak")
    print(f"    3. Open {obs_id}_detection_rate.png — check detection is consistent over time")
    print(f"    4. If bounds look off, override in configs/tracking/{subject}/{version}.yaml")


if __name__ == "__main__":
    main()
