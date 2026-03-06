"""Utility functions for treatment comparison animation generation."""

import os
import cv2
import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def sample_obs_with_meta(df, treatment, n, seed=42):
    """Sample n observations for a treatment, returning (obs_id, batch, position) tuples."""
    random.seed(seed)
    sub = df.loc[df["T"] == treatment].drop_duplicates("observation_id")
    obs_list = sub["observation_id"].tolist()
    sampled_ids = random.sample(obs_list, min(n, len(obs_list)))
    sub_indexed = sub.set_index("observation_id")
    result = []
    for oid in sampled_ids:
        row = sub_indexed.loc[oid]
        batch = str(row["W_batch"]) if "W_batch" in row.index else ""
        pos = str(row["W_position"]) if "W_position" in row.index else ""
        result.append((oid, batch, pos))
    return result


def build_grooming_lookup(df, obs_ids):
    """Build {obs_id: set(frame_idx)} for frames with any grooming activity."""
    outcome_cols = [c for c in df.columns if c.startswith("Y_") and c not in ["Y_BOL", "Y_FOL", "Y_YOL"]]
    lookup = {}
    for obs_id in obs_ids:
        obs_df = df[df["observation_id"] == obs_id]
        active = obs_df[outcome_cols].any(axis=1)
        lookup[obs_id] = set(obs_df.loc[active, "frame_idx"].values)
    return lookup


def detect_dish_circle(img_path):
    """Detect the round dish in a frame using HoughCircles. Returns (cx, cy, r) or None."""
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)
    h, w = gray.shape
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=h // 2,
        param1=50, param2=30, minRadius=h // 4, maxRadius=h // 2,
    )
    if circles is not None:
        c = circles[0][0]
        return int(c[0]), int(c[1]), int(c[2])
    return None


def crop_circle_pil(img, cx, cy, r, bg_color=(0, 0, 0)):
    """Crop a PIL image to a square around the circle, mask outside to bg_color."""
    left, top = max(cx - r, 0), max(cy - r, 0)
    right, bottom = min(cx + r, img.width), min(cy + r, img.height)
    cropped = img.crop((left, top, right, bottom))

    w, h = cropped.size
    mask = Image.new("L", (w, h), 0)
    mask_draw = ImageDraw.Draw(mask)
    mcx, mcy = cx - left, cy - top
    mask_draw.ellipse([mcx - r, mcy - r, mcx + r, mcy + r], fill=255)

    bg = Image.new("RGB", (w, h), bg_color)
    bg.paste(cropped, mask=mask)
    return bg


def load_clip(frames_dir, obs_id, start, n_frames, size, crop_circle=False):
    """Load n_frames consecutive frames, resized to size.

    Returns (list_of_PIL_images, list_of_frame_indices).
    """
    clip, frame_indices = [], []
    circle = None

    for i in range(start, start + n_frames + 100):
        path = frames_dir / obs_id / f"frame_{i:06d}.jpg"
        if not path.exists():
            if i - start > 100:
                break
            continue

        if crop_circle and circle is None:
            circle = detect_dish_circle(path)

        img = Image.open(path).convert("RGB")
        if crop_circle and circle is not None:
            img = crop_circle_pil(img, *circle)
        img = img.resize(size, Image.Resampling.LANCZOS)
        clip.append(img)
        frame_indices.append(i)

        if len(clip) >= n_frames:
            break
    return clip, frame_indices


def _get_fonts():
    """Load title and small fonts."""
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    try:
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 10)
    except OSError:
        font_small = ImageFont.load_default()
    return font, font_small


def _draw_label(draw, x, y, text, font):
    """Draw text with a solid black background bar."""
    bbox = draw.textbbox((x, y), text, font=font)
    draw.rectangle([bbox[0] - 2, bbox[1] - 1, bbox[2] + 2, bbox[3] + 1], fill=(0, 0, 0))
    draw.text((x, y), text, fill=(255, 255, 0), font=font)


def _draw_green_border(draw, x, y, w, h, width=2):
    """Draw a green rectangle border."""
    for i in range(width):
        draw.rectangle([x + i, y + i, x + w - 1 - i, y + h - 1 - i], outline=(0, 255, 0))


def _build_frames(
    mode,
    clips_t1, clips_t2,
    indices_t1, indices_t2,
    meta_t1, meta_t2,
    obs_t1, obs_t2,
    grooming_lookup,
    treatment_1, treatment_2,
    thumb_size=(100, 100),
    grid_rows=5, grid_cols=4,
    gap=2, margin=10, separator=40, border_w=2,
):
    """Render all animation frames as a list of numpy arrays."""
    fw, fh = thumb_size
    n_per_side = grid_rows * grid_cols
    font, font_small = _get_fonts()

    show_titles = (mode == "diagnostic")
    header_h = 40 if show_titles else 10

    grid_w = grid_cols * fw + (grid_cols - 1) * gap
    grid_h = grid_rows * fh + (grid_rows - 1) * gap
    canvas_w = 2 * margin + 2 * grid_w + separator
    canvas_h = 2 * margin + header_h + grid_h

    left_x0 = margin
    right_x0 = margin + grid_w + separator
    grid_y0 = margin + header_h

    bg_color = (0, 0, 0) if mode == "final" else (255, 255, 255)
    title_color = (0, 0, 0)  # only used in diagnostic

    min_len = min(
        min((len(c) for c in clips_t1), default=1),
        min((len(c) for c in clips_t2), default=1),
    )

    frames = []
    for t in range(min_len):
        canvas = Image.new("RGB", (canvas_w, canvas_h), color=bg_color)
        draw = ImageDraw.Draw(canvas)

        if show_titles:
            draw.text((left_x0 + grid_w // 2, margin), f"Treatment {treatment_1}",
                      fill=title_color, font=font, anchor="mt")
            draw.text((right_x0 + grid_w // 2, margin), f"Treatment {treatment_2}",
                      fill=title_color, font=font, anchor="mt")

        for idx in range(n_per_side):
            row, col = divmod(idx, grid_cols)
            cx = col * (fw + gap)
            cy = grid_y0 + row * (fh + gap)

            # Left grid (treatment 1)
            if idx < len(clips_t1) and t < len(clips_t1[idx]):
                canvas.paste(clips_t1[idx][t], (left_x0 + cx, cy))
                fi = indices_t1[idx][t] if t < len(indices_t1[idx]) else None
                if fi is not None and fi in grooming_lookup.get(obs_t1[idx], set()):
                    _draw_green_border(draw, left_x0 + cx, cy, fw, fh, border_w)
                if mode == "diagnostic":
                    _, batch, pos = meta_t1[idx]
                    _draw_label(draw, left_x0 + cx + 2, cy + 2, f"b={batch} p={pos}", font_small)

            # Right grid (treatment 2)
            if idx < len(clips_t2) and t < len(clips_t2[idx]):
                canvas.paste(clips_t2[idx][t], (right_x0 + cx, cy))
                fi = indices_t2[idx][t] if t < len(indices_t2[idx]) else None
                if fi is not None and fi in grooming_lookup.get(obs_t2[idx], set()):
                    _draw_green_border(draw, right_x0 + cx, cy, fw, fh, border_w)
                if mode == "diagnostic":
                    _, batch, pos = meta_t2[idx]
                    _draw_label(draw, right_x0 + cx + 2, cy + 2, f"b={batch} p={pos}", font_small)

        frames.append(np.array(canvas))

    return frames


def generate_comparison(
    df,
    frames_dir,
    treatment_1, treatment_2,
    output_path,
    mode="final",
    grid_rows=5, grid_cols=4,
    start_frame=1000,
    duration=10,
    fps=5,
    thumb_size=(400, 400),
    show_activity=True,
):
    """End-to-end: sample, load clips, render, and save a treatment comparison animation.

    Args:
        df: DataFrame with columns observation_id, T, frame_idx, Y_*, W_batch, W_position
        frames_dir: Path to extracted frames (e.g. dataset/ants/v3/frames/full/)
        treatment_1, treatment_2: treatment codes to compare
        output_path: where to save (.png for APNG recommended)
        mode: "diagnostic" or "final"
        duration: total animation duration in seconds
        fps: playback framerate
        show_activity: if True, draw green border on frames with grooming activity
    """
    n_per_side = grid_rows * grid_cols
    n_frames = duration * fps

    # Sample
    meta_t1 = sample_obs_with_meta(df, treatment_1, n_per_side)
    meta_t2 = sample_obs_with_meta(df, treatment_2, n_per_side)
    obs_t1 = [m[0] for m in meta_t1]
    obs_t2 = [m[0] for m in meta_t2]

    # Grooming lookup
    grooming_lookup = build_grooming_lookup(df, obs_t1 + obs_t2) if show_activity else {}

    # Load clips (parallel I/O)
    crop_circle = (mode == "final")
    all_obs = obs_t1 + obs_t2
    with ThreadPoolExecutor() as pool:
        results = list(pool.map(
            lambda obs: load_clip(frames_dir, obs, start_frame, n_frames, thumb_size, crop_circle),
            all_obs,
        ))
    n1 = len(obs_t1)
    clips_t1, indices_t1 = zip(*results[:n1])
    clips_t2, indices_t2 = zip(*results[n1:])

    # Render
    frames = _build_frames(
        mode=mode,
        clips_t1=list(clips_t1), clips_t2=list(clips_t2),
        indices_t1=list(indices_t1), indices_t2=list(indices_t2),
        meta_t1=meta_t1, meta_t2=meta_t2,
        obs_t1=obs_t1, obs_t2=obs_t2,
        grooming_lookup=grooming_lookup,
        treatment_1=treatment_1, treatment_2=treatment_2,
        thumb_size=thumb_size, grid_rows=grid_rows, grid_cols=grid_cols,
    )

    # Save as animated PNG (APNG) — full 24-bit color, unlike GIF's 256-color limit
    output_path = Path(output_path)
    os.makedirs(output_path.parent, exist_ok=True)

    duration_ms = int(1000 / fps)
    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(
        str(output_path),
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
    )

    print(f"Saved: {output_path} ({len(frames) / fps:.1f}s at {fps} fps)")
    return output_path
