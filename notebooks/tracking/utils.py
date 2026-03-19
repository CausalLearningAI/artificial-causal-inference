"""
Notebook utilities for tracking configuration analysis.

Used by notebooks/tracking/ants_config.ipynb.

All general-purpose tracking functions live in src/tracking/:
  - detection.py   — per-frame detection (ant mask, body blobs, colour filter, identity)
  - calibration.py — colour-bound optimisation, config I/O, path helpers
  - tracker.py     — temporal tracking (AntTracker)

This file re-exports what the notebook needs and adds visualisation-only code.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Re-export detection functions
from src.tracking.detection import (  # noqa: E402
    compute_ant_mask,
    find_body_blobs,
    filter_color_pixels,
    find_mark_blob,
    _find_best_color_blob,
    match_identities,
)

# Re-export calibration / config / path helpers
from src.tracking.calibration import (  # noqa: E402
    frames_dir,
    config_path,
    list_obs_ids,
    load_config,
    save_config,
    get_bounds_from_config,
    load_frame,
    get_background,
    optimize_color_bounds,
)


# ── Backward-compatible wrappers ─────────────────────────────────────────────

def filter_blue_pixels(
    hsv: np.ndarray,
    ant_mask: np.ndarray,
    h_range: Tuple[int, int] = (100, 125),
    min_s: int = 80,
    min_v: int = 80,
    min_blob_area: int = 10,
    max_blob_area: int = 100,
) -> np.ndarray:
    """Blue marking detection (delegates to filter_color_pixels)."""
    return filter_color_pixels(
        hsv, ant_mask,
        h_range=h_range, min_s=min_s, min_v=min_v,
        min_blob_area=min_blob_area, max_blob_area=max_blob_area,
    )


def filter_yellow_pixels(
    hsv: np.ndarray,
    ant_mask: np.ndarray,
    h_range: Tuple[int, int] = (0, 20),
    min_s: int = 80,
    min_v: int = 80,
    min_blob_area: int = 10,
    max_blob_area: int = 100,
) -> np.ndarray:
    """Yellow marking detection (delegates to filter_color_pixels)."""
    return filter_color_pixels(
        hsv, ant_mask,
        h_range=h_range, min_s=min_s, min_v=min_v,
        min_blob_area=min_blob_area, max_blob_area=max_blob_area,
    )



# ══════════════════════════════════════════════════════════════════════════════
# Visualization — modular pipeline
# ══════════════════════════════════════════════════════════════════════════════
#
# Detection functions (compute_ant_mask, find_body_blobs, filter_color_pixels,
# match_identities) are imported from src/tracking/detection.py above.
# filter_blue_pixels / filter_yellow_pixels are thin wrappers defined earlier.
#
# Panels:
#   [0] scale        — config params at 1:1 pixel scale
#   [1] original     — raw frame
#   [2] no background — frame - background
#   [3] ants         — ant mask + body dots (green=1, orange=2, red=3+)
#   [4] color marking — original + filtered blue/yellow pixels overlaid
#   [5] tracking     — final result: dot per body, POV square, X at mark
#
# ══════════════════════════════════════════════════════════════════════════════

# ── Panel renderers ──────────────────────────────────────────────────────────

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _panel_scale(cfg: dict, shape: Tuple[int, int]) -> np.ndarray:
    """Scale reference at true 1:1 pixel scale (no magnification)."""
    H, W = shape[:2]
    canvas = np.full((H, W, 3), 250, dtype=np.uint8)

    max_step       = cfg["max_step"]
    pov_radius     = cfg["pov_radius"]
    min_area_body  = cfg["min_area_body"]
    max_area_body  = cfg["max_area_body"]
    min_area_color = cfg["min_area_color"]

    cx, cy = W // 2, H // 2

    # max_step circle (dashed gray)
    for ang in range(0, 360, 8):
        a1, a2 = np.deg2rad(ang), np.deg2rad(ang + 4)
        cv2.line(canvas,
                 (int(cx + max_step * np.cos(a1)), int(cy + max_step * np.sin(a1))),
                 (int(cx + max_step * np.cos(a2)), int(cy + max_step * np.sin(a2))),
                 (170, 170, 170), 1)

    # POV square (dashed orange)
    half = pov_radius
    for i in range(0, 2 * half, 8):
        # top edge
        cv2.line(canvas, (cx - half + i, cy - half),
                 (cx - half + min(i + 4, 2 * half), cy - half), (0, 140, 220), 1)
        # bottom
        cv2.line(canvas, (cx - half + i, cy + half),
                 (cx - half + min(i + 4, 2 * half), cy + half), (0, 140, 220), 1)
        # left
        cv2.line(canvas, (cx - half, cy - half + i),
                 (cx - half, cy - half + min(i + 4, 2 * half)), (0, 140, 220), 1)
        # right
        cv2.line(canvas, (cx + half, cy - half + i),
                 (cx + half, cy - half + min(i + 4, 2 * half)), (0, 140, 220), 1)

    # Body size annulus
    r_max = max(3, int(np.sqrt(max_area_body / np.pi)))
    r_min = max(2, int(np.sqrt(min_area_body / np.pi)))
    r_dot = max(2, int(np.sqrt(min_area_color / np.pi)))
    cv2.circle(canvas, (cx, cy), r_max, (180, 230, 180), -1)
    cv2.circle(canvas, (cx, cy), r_min, (90,  200,  90), -1)
    cv2.circle(canvas, (cx, cy), r_max, (0,   150,   0), 1)
    cv2.circle(canvas, (cx, cy), r_min, (0,   110,   0), 1)

    # Min dot
    dot_cx = cx + r_min + r_dot + 6
    cv2.circle(canvas, (dot_cx, cy), r_dot, (50, 120, 200), -1)

    fs, th = 0.35, 1
    _label(canvas, f"max_step={max_step}px", cx, cy - max_step - 8,
           (140, 140, 140), fs, th, center=True)
    _label(canvas, f"pov={pov_radius}px", cx + half + 4, cy - half - 4,
           (0, 140, 220), fs, th)
    _label(canvas, f"body=[{min_area_body},{max_area_body}]px2", cx, cy + r_max + 14,
           (0, 130, 0), fs, th, center=True)
    _label(canvas, f"dot>={min_area_color}px2", dot_cx + r_dot + 4, cy + 4,
           (30, 100, 180), fs, th)

    return canvas


def _label(img, text, x, y, color, fs, th, center=False):
    if center:
        tw = cv2.getTextSize(text, _FONT, fs, th)[0][0]
        x = x - tw // 2
    y = max(14, min(img.shape[0] - 4, y))
    x = max(2, min(img.shape[1] - 4, x))
    cv2.putText(img, text, (x, y), _FONT, fs, color, th, cv2.LINE_AA)


def _panel_noback(frame_nb: np.ndarray) -> np.ndarray:
    """Background-subtracted frame (RGB)."""
    return cv2.cvtColor(frame_nb, cv2.COLOR_BGR2RGB)


def _panel_ants(
    ant_mask: np.ndarray,
    body_blobs: List[Dict],
    shape: Tuple[int, int],
) -> np.ndarray:
    """Ant mask with one dot per body blob, colored by estimated count."""
    H, W = shape[:2]
    vis = np.full((H, W, 3), 30, dtype=np.uint8)
    vis[ant_mask > 0] = [220, 220, 220]

    COUNT_COLORS = {1: (0, 200, 0), 2: (0, 165, 255), 3: (0, 0, 220)}
    COUNT_LABELS = {1: "1", 2: "2", 3: "3"}
    for blob in body_blobs:
        cx, cy = blob["centroid"]
        n = blob["n_agents_in_blob"]
        col = COUNT_COLORS.get(n, (0, 0, 220))
        cv2.circle(vis, (cx, cy), 6, col, -1)
        cv2.circle(vis, (cx, cy), 6, (255, 255, 255), 1)
        cv2.putText(vis, COUNT_LABELS.get(n, "?"), (cx + 9, cy + 5),
                    _FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Legend in bottom-left
    lx, ly = 8, H - 10
    for n, col_bgr in sorted(COUNT_COLORS.items(), reverse=True):
        cv2.circle(vis, (lx, ly), 5, col_bgr, -1)
        cv2.putText(vis, f"{n} ant{'s' if n > 1 else ''}", (lx + 10, ly + 5),
                    _FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        ly -= 16

    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


def _panel_color_single(
    frame_rgb: np.ndarray,
    hsv: np.ndarray,
    det_mask: np.ndarray,
    color: str,
    h_range: Optional[Tuple[int, int]] = None,
    min_s: int = 80,
    min_v: int = 80,
    min_blob_area: int = 10,
    max_blob_area: int = 100,
) -> Tuple[np.ndarray, str]:
    """One color channel: ALL pixels passing HSV filter in green on full frame.

    Every pixel matching (H in range) & (S >= min) & (V >= min) is painted
    green — no ant_mask, no blob filtering, no contour.  Single scattered
    pixels (noise) are dilated so they become visible at display scale.

    Ratio = n_det / n_all where:
      n_det = pixels in the final selected blob (after morph close + identity)
      n_all = total pixels across all valid-sized blobs on full frame
              (color filter → morph close → area filter, no ant_mask restriction)
    """
    vis = frame_rgb.copy()
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    h_lo, h_hi = h_range if h_range is not None else ((100, 125) if color == "blue" else (0, 20))

    # Raw HSV filter on full frame
    all_px = (H >= h_lo) & (H <= h_hi) & (S >= min_s) & (V >= min_v)

    # Dilate so single-pixel matches are visible at display resolution
    all_px_vis = all_px.astype(np.uint8) * 255
    kernel_vis = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    all_px_vis = cv2.dilate(all_px_vis, kernel_vis, iterations=1)
    vis[all_px_vis > 0] = (0, 200, 0)

    # Denominator: total pixels in all valid-sized blobs on full frame
    # Steps 1-3: color filter → contours → morph close → area filter
    color_u8 = all_px.astype(np.uint8) * 255
    contours, _ = cv2.findContours(color_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kernel_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    h_img, w_img = hsv.shape[:2]
    n_all = 0
    for cnt in contours:
        blob = np.zeros((h_img, w_img), dtype=np.uint8)
        cv2.drawContours(blob, [cnt], -1, 255, -1)
        blob = cv2.morphologyEx(blob, cv2.MORPH_CLOSE, kernel_morph)
        area = cv2.countNonZero(blob)
        if min_blob_area <= area <= max_blob_area:
            n_all += area

    # Numerator: pixels in the final selected blob
    n_det = int((det_mask > 0).sum()) if (det_mask > 0).any() else 0

    if n_all == 0:
        ratio_str = "N/A"
    else:
        ratio_str = f"{n_det / n_all:.0%}"
    stats_text = f"{color}: {n_det}/{n_all}px ({ratio_str})"

    return vis, stats_text


def _panel_tracking(
    frame_rgb: np.ndarray,
    body_blobs: List[Dict],
    labels: List[str],
    blue_pos: Optional[Tuple[int, int]],
    yellow_pos: Optional[Tuple[int, int]],
    cfg: dict,
) -> np.ndarray:
    """Final tracking: identity dots + POV squares centered on color marks.

    POV is centered on the color mark position (more precise localization,
    especially within merged blobs).  No POV for focal/unknown agents.
    """
    vis = frame_rgb.copy()
    pov_r = cfg["pov_radius"]
    COLORS = {
        "blue":    (60, 130, 255),
        "yellow":  (255, 200, 0),
        "focal":   (180, 180, 180),
        "unknown": (255, 255, 255),
    }

    for blob, label in zip(body_blobs, labels):
        cx, cy = blob["centroid"]
        col = COLORS[label]

        # Body centroid dot
        cv2.circle(vis, (cx, cy), 7, col, -1)
        cv2.circle(vis, (cx, cy), 7, (40, 40, 40), 1)
        _label(vis, label, cx + 10, cy - 10, col, 0.9, 2)

    # POV squares centered on color mark positions (not centroids)
    if blue_pos is not None:
        bx, by = blue_pos
        cv2.drawMarker(vis, blue_pos, COLORS["blue"], cv2.MARKER_TILTED_CROSS, 18, 2)
        pt1 = (bx - pov_r, by - pov_r)
        pt2 = (bx + pov_r, by + pov_r)
        cv2.rectangle(vis, pt1, pt2, COLORS["blue"], 2, cv2.LINE_AA)

    if yellow_pos is not None:
        yx, yy = yellow_pos
        cv2.drawMarker(vis, yellow_pos, COLORS["yellow"], cv2.MARKER_TILTED_CROSS, 18, 2)
        pt1 = (yx - pov_r, yy - pov_r)
        pt2 = (yx + pov_r, yy + pov_r)
        cv2.rectangle(vis, pt1, pt2, COLORS["yellow"], 2, cv2.LINE_AA)

    # Red warning only for colors whose marking was not detected at all
    missing = []
    if blue_pos is None:
        missing.append("blue")
    if yellow_pos is None:
        missing.append("yellow")
    if missing:
        H_img = vis.shape[0]
        warn_text = "MISSING: " + ", ".join(missing)
        _label(vis, warn_text, 4, H_img - 10, (255, 40, 40), 0.9, 2)

    return vis


# ── Per-frame analysis ────────────────────────────────────────────────────────

# Column names for the multi-frame grid (scale is shown once, not per frame).
_PANEL_NAMES = ["original", "no background", "ants", "blue marking", "yellow marking", "tracking"]


def _analyze_frame(
    frame: np.ndarray,
    background: np.ndarray,
    cfg: dict,
) -> Tuple[List[np.ndarray], List[str]]:
    """Run the full pipeline on one frame.

    Returns (panels, stats_texts) where panels is 6 images and stats_texts
    has the blue/yellow detection stats for matplotlib overlay.
    """
    min_area_body = cfg["min_area_body"]
    max_area_body = cfg["max_area_body"]

    # Use full config bounds (H, S, V)
    blue_lb  = cfg.get("blue_marking_lb",   [100, 80, 80])
    blue_ub  = cfg.get("blue_marking_ub",   [125, 255, 255])
    yellow_lb = cfg.get("yellow_marking_lb", [0, 80, 80])
    yellow_ub = cfg.get("yellow_marking_ub", [20, 255, 255])

    blue_h   = (blue_lb[0],   blue_ub[0])
    yellow_h = (yellow_lb[0], yellow_ub[0])
    blue_sv  = (blue_lb[1],   blue_lb[2])
    yellow_sv = (yellow_lb[1], yellow_lb[2])
    min_area_color = cfg.get("min_area_color", 10)
    max_area_color = cfg.get("max_area_color", 100)

    frame_nb    = cv2.absdiff(frame, background)
    ant_mask    = compute_ant_mask(frame_nb, min_area=min_area_body)
    blobs       = find_body_blobs(ant_mask, min_area_body, max_area_body)
    hsv         = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blue_mask   = filter_blue_pixels(
        hsv, ant_mask,
        h_range=blue_h, min_s=blue_sv[0], min_v=blue_sv[1],
        min_blob_area=min_area_color, max_blob_area=max_area_color,
    )
    yellow_mask = filter_yellow_pixels(
        hsv, ant_mask,
        h_range=yellow_h, min_s=yellow_sv[0], min_v=yellow_sv[1],
        min_blob_area=min_area_color, max_blob_area=max_area_color,
    )
    labels, blue_pos, yellow_pos = match_identities(
        blobs, blue_mask, yellow_mask,
    )
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    blue_panel, blue_stats = _panel_color_single(
        frame_rgb, hsv, blue_mask, "blue",
        h_range=blue_h, min_s=blue_sv[0], min_v=blue_sv[1],
        min_blob_area=min_area_color, max_blob_area=max_area_color,
    )
    yellow_panel, yellow_stats = _panel_color_single(
        frame_rgb, hsv, yellow_mask, "yellow",
        h_range=yellow_h, min_s=yellow_sv[0], min_v=yellow_sv[1],
        min_blob_area=min_area_color, max_blob_area=max_area_color,
    )

    panels = [
        frame_rgb,
        _panel_noback(frame_nb),
        _panel_ants(ant_mask, blobs, frame.shape),
        blue_panel,
        yellow_panel,
        _panel_tracking(frame_rgb, blobs, labels, blue_pos, yellow_pos, cfg),
    ]
    return panels, [blue_stats, yellow_stats]


# ── Main plot ────────────────────────────────────────────────────────────────

def plot_frame_analysis(
    frames: "np.ndarray | List[np.ndarray]",
    background: np.ndarray,
    cfg: dict,
    title: str = "",
    frame_labels: Optional[List[str]] = None,
) -> plt.Figure:
    """
    Multi-frame diagnostic grid.

    Parameters
    ----------
    frames : single BGR frame or list of BGR frames
    background : background image (shared across frames)
    cfg : tracking config dict
    title : figure suptitle
    frame_labels : optional row labels (e.g. ["frame 50", "frame 150", ...])

    Layout: N_frames rows x 5 columns.
    Columns = original / no background / ants / color marking / tracking.
    Use plot_scale_overview() separately for scale + version samples.
    """
    if isinstance(frames, np.ndarray) and frames.ndim == 3:
        frames = [frames]
    n_frames = len(frames)

    if frame_labels is None:
        frame_labels = [f"frame {i}" for i in range(n_frames)]

    n_cols = len(_PANEL_NAMES)
    fig, axes = plt.subplots(
        n_frames, n_cols,
        figsize=(n_cols * 2.8, n_frames * 2.8),
        squeeze=False,
    )

    _STATS_COLORS = {"blue": "#3C82FF", "yellow": "#FFC800"}

    for r, frame in enumerate(frames):
        panels, stats_texts = _analyze_frame(frame, background, cfg)
        for c, (img, name) in enumerate(zip(panels, _PANEL_NAMES)):
            ax = axes[r, c]
            ax.imshow(img)
            ax.axis("off")
            if r == 0:
                ax.set_title(name, fontsize=11)

            # Matplotlib text overlay for color marking stats (sharp at any DPI)
            if name == "blue marking":
                ax.text(0.02, 0.97, stats_texts[0], transform=ax.transAxes,
                        fontsize=9, fontweight="bold", color=_STATS_COLORS["blue"],
                        va="top", ha="left",
                        bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5))
            elif name == "yellow marking":
                ax.text(0.02, 0.97, stats_texts[1], transform=ax.transAxes,
                        fontsize=9, fontweight="bold", color=_STATS_COLORS["yellow"],
                        va="top", ha="left",
                        bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5))

        # Row label on the left margin
        axes[r, 0].set_ylabel(frame_labels[r], fontsize=10, color="gray",
                              rotation=0, labelpad=50, va="center")

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    return fig


def plot_scale_overview(
    version_frames: Dict[str, np.ndarray],
    configs: Dict[str, dict],
) -> plt.Figure:
    """Scale reference + one sample frame per version (side by side).

    Parameters
    ----------
    version_frames : {version: BGR frame} for each version to show
    configs : {version: config dict}

    Layout: 1 row.  First column = scale (from first version's config),
    then one column per version showing the original frame.
    """
    versions = list(version_frames.keys())
    n = 1 + len(versions)
    fig, axes = plt.subplots(1, n, figsize=(n * 2.8, 3))

    # Scale from first version
    first_v = versions[0]
    scale_img = _panel_scale(configs[first_v], version_frames[first_v].shape)
    axes[0].imshow(scale_img)
    axes[0].set_title("scale", fontsize=8)
    axes[0].axis("off")

    for i, v in enumerate(versions):
        axes[i + 1].imshow(cv2.cvtColor(version_frames[v], cv2.COLOR_BGR2RGB))
        axes[i + 1].set_title(v, fontsize=8)
        axes[i + 1].axis("off")

    plt.tight_layout()
    return fig


# ── Color palette visualization ───────────────────────────────────────────────


def plot_color_palette(
    configs: Dict[str, dict],
    title: str = "Color filter bounds per version",
) -> plt.Figure:
    """Show the admitted HSV color palette as H x S grids at min/max V.

    Layout: rows = versions, 4 columns = blue@V_min, blue@V_max,
    yellow@V_min, yellow@V_max.  Accepted region at full color, rest dimmed.
    """
    versions = list(configs.keys())
    n_versions = len(versions)
    n_cols = 4
    fig, axes = plt.subplots(
        n_versions, n_cols,
        figsize=(3.0 * n_cols, 2.4 * n_versions + 0.4),
        squeeze=False,
    )

    n_h, n_s = 180, 64

    for r, v in enumerate(versions):
        cfg = configs[v]
        for ci, color in enumerate(["blue", "yellow"]):
            lb = cfg.get(f"{color}_marking_lb", [0, 80, 80])
            ub = cfg.get(f"{color}_marking_ub", [179, 255, 255])
            h_lo, h_hi = int(lb[0]), int(ub[0])
            s_lo, s_hi = int(lb[1]), int(ub[1])
            v_lo, v_hi = int(lb[2]), int(ub[2])

            for vi, v_val in enumerate([v_lo, v_hi]):
                col_idx = ci * 2 + vi
                ax = axes[r, col_idx]

                # Build H (x) x S (y) grid at this V
                hh = np.arange(n_h).reshape(1, n_h).repeat(n_s, axis=0)
                ss = np.linspace(0, 255, n_s, dtype=int).reshape(n_s, 1).repeat(n_h, axis=1)
                ss = ss[::-1]  # high S at top
                vv = np.full_like(hh, v_val)
                hsv_grid = np.stack([hh, ss, vv], axis=-1).astype(np.uint8)
                rgb_grid = cv2.cvtColor(hsv_grid, cv2.COLOR_HSV2RGB)

                # Accepted mask: H + S within bounds
                if h_lo <= h_hi:
                    h_ok = (hh >= h_lo) & (hh <= h_hi)
                else:
                    h_ok = (hh >= h_lo) | (hh <= h_hi)
                s_ok = (ss >= s_lo) & (ss <= s_hi)
                accepted = h_ok & s_ok

                # Dim rejected pixels
                rgb_grid[~accepted] = (
                    rgb_grid[~accepted].astype(np.int16) * 2 // 10
                ).clip(0, 255).astype(np.uint8)

                ax.imshow(rgb_grid, aspect="auto",
                          extent=[0, 179, 0, 255])
                ax.set_xlim(0, 179)
                ax.set_ylim(0, 255)

                # Axes: always show H and S ticks
                ax.set_xlabel("H", fontsize=8)
                ax.set_ylabel("S", fontsize=8)

                # Row label: version on leftmost column
                if col_idx == 0:
                    ax.set_ylabel(f"{v}\nS", fontsize=9)

                # Subplot title: color + V label
                v_label = r"$V_{\min}$" if vi == 0 else r"$V_{\max}$"
                ax.set_title(f"{color}  {v_label}={v_val}", fontsize=9)

                # Dashed rectangle for accepted H+S region
                rect_w = (h_hi - h_lo) if h_lo <= h_hi else (180 - h_lo + h_hi)
                rect = mpatches.Rectangle(
                    (h_lo, s_lo), rect_w, s_hi - s_lo,
                    linewidth=1.5, edgecolor="white",
                    facecolor="none", linestyle="--",
                )
                ax.add_patch(rect)

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


