"""Step 2: headless replacement for AMADEUS's segmentation GUI (gui/gui_segmentation.py).

    python segment.py <video_id> [--analysis PATH] [--session DIR] [--workers N] [--preview-only]

AMADEUS normally needs a person to tune foreground segmentation in a Tk GUI. That GUI
writes three files into <session>/segmentation/, which the headless pipeline
(main/batch.py) then reads:
  <stem>_list_of_blobs_gui.pickle   SimpleNamespace(blobs_in_video=[[SimpleNamespace(contour, is_outlier)]
                                    per frame], source_frame_count, background/training frame ranges)
  background.png                    used to paste isolated animals into synthetic crowd images
  segmentation_gui_config.json      used by identity correction to re-segment crops
This script writes the same three files with the same pixel pipeline
(main/segmentation_core.py mirrors the GUI's CPU path) and the GUI's default frame
ranges. What it replaces by rule instead of by hand:
  * ROI: ants = Hough circle of the dish (per video); mice = bounding box of the bright
    bedding (per video). Same rule for every video of a setup.
  * background: median of 100 evenly spaced frames, but with dark (animal) pixels masked
    out first, so animals that sit still for a long time do not get baked into the
    background (the GUI uses a plain median).
  * single-animal area bounds ("absolute" mode in the GUI): from 200 sampled frames,
    take the frames whose blob count equals the number of animals, median blob area m,
    bounds [AREA_LO * m, AREA_HI * m]. Blobs outside are flagged is_outlier (= crossing).
Thresholds are per setup (SEG below), chosen by looking at frames.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from pilot import AMADEUS_DIR, SETUPS, VIDEOS, out_dir

sys.path.insert(0, str(AMADEUS_DIR / "main"))
from segmentation_core import SegConfig, compute_foreground_mask, build_static_roi_mask  # noqa: E402

# GUI defaults (gui_segmentation.py): 100 background samples over the whole video,
# training range = whole video, but only frames 0..19999 for videos with >= 20000 frames,
# training_frame_interval 5, 200 analysis samples.
BG_SAMPLES = 100
TRAIN_MAX_FRAME = 19999
TRAIN_INTERVAL = 5
AREA_SAMPLES = 200
AREA_LO, AREA_HI = 0.5, 1.5

SEG = {
    # Ants: dark body on light sand. Hybrid mode = dark AND different from background, so the
    # static dark marker / shadow on the dish rim is ignored.
    "ants": dict(segmentation_mode="dark_region_and_background_diff", dark_threshold=110, diff_threshold=35,
                 threshold=110, bright_threshold=200, blur_ksize=5, open_iter=1, close_iter=1, fill_holes=True,
                 invert_mask=False, min_area=60, region_expand_px=0, region_expand_merge_only=False,
                 roi="dish_circle", roi_margin_px=15, bg_mask_threshold=110, bg_mask_dilate=15),
    # Mice: black mice with bright shave marks on white bedding. A plain dark threshold cuts the
    # shave marks out of the body and splits mice into fragments, so use dark (<=165, includes the
    # grey marks and excludes the white bedding) AND different from the (mouse-free) background.
    # open_iter=5 strips the tails (~8 px wide at 1032 px), which otherwise chain mice together.
    "mice": dict(segmentation_mode="dark_region_and_background_diff", threshold=165, dark_threshold=165,
                 diff_threshold=40, bright_threshold=200, blur_ksize=5, open_iter=5, close_iter=2, fill_holes=True,
                 invert_mask=False, min_area=600, region_expand_px=0, region_expand_merge_only=False,
                 roi="bedding_rect", roi_margin_px=30, bg_mask_threshold=70, bg_mask_dilate=41),
}


def read_frames(path: str, indices: list[int]) -> dict[int, np.ndarray]:
    cap = cv2.VideoCapture(path)
    out = {}
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok:
            out[int(i)] = fr
    cap.release()
    return out


def masked_median_background(frames: list[np.ndarray], dark_thr: int, dilate: int) -> np.ndarray:
    stack = np.stack(frames).astype(np.float32)               # (n, h, w, 3)
    dark = []
    k = np.ones((dilate, dilate), np.uint8)
    for fr in frames:
        g = cv2.GaussianBlur(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        m = (g < dark_thr).astype(np.uint8)
        dark.append(cv2.dilate(m, k) > 0)
    dark = np.stack(dark)                                      # (n, h, w)
    plain = np.median(stack, axis=0)
    stack[dark] = np.nan
    with np.errstate(all="ignore"):
        masked = np.nanmedian(stack, axis=0)
    masked = np.where(np.isnan(masked), plain, masked)         # pixel dark in every sample: static object
    return np.clip(masked, 0, 255).astype(np.uint8)


def find_roi(species: str, bg: np.ndarray, margin: int) -> dict:
    h, w = bg.shape[:2]
    g = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    if species == "ants":
        gm = cv2.medianBlur(g, 7)
        cs = cv2.HoughCircles(gm, cv2.HOUGH_GRADIENT, dp=2, minDist=500, param1=60, param2=40,
                              minRadius=int(0.36 * min(h, w)), maxRadius=int(0.51 * min(h, w)))
        if cs is None:
            raise RuntimeError("dish circle not found")
        x, y, r = cs[0][0]
        return {"enabled": True, "reverse": False, "shape": "circle", "x": int(round(x)), "y": int(round(y)),
                "w": int(round(r)) + margin, "h": 0, "frame_start": 0, "frame_end": -1}
    # mice: bright bedding region, largest component, its bounding box
    gb = cv2.GaussianBlur(g, (21, 21), 0)
    m = (gb > 150).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, bw, bh = (int(v) for v in stats[i, :4])
    x0, y0 = max(0, x - margin), max(0, y - margin)
    x1, y1 = min(w, x + bw + margin), min(h, y + bh + margin)
    return {"enabled": True, "reverse": False, "shape": "rectangle", "x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0,
            "frame_start": 0, "frame_end": -1}


def seg_config(s: dict, bg: np.ndarray) -> SegConfig:
    return SegConfig(mode=s["segmentation_mode"], ksize=s["blur_ksize"], threshold=s["threshold"],
                     dark_threshold=s["dark_threshold"], bright_threshold=s["bright_threshold"],
                     diff_threshold=s["diff_threshold"], open_iter=s["open_iter"], close_iter=s["close_iter"],
                     fill_holes=s["fill_holes"], invert_mask=s["invert_mask"], expand_px=s["region_expand_px"],
                     expand_merge_only=s["region_expand_merge_only"], background_bgr=bg)


def blobs_of(frame: np.ndarray, cfg: SegConfig, roi_mask: np.ndarray, min_area: float) -> list[np.ndarray]:
    mask = compute_foreground_mask(frame, cfg, roi_mask)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c.astype(np.int32) for c in cnts if abs(cv2.contourArea(c)) >= min_area]


_W = {}


def _init_worker(path, s, bg, roi_mask):
    _W.update(path=path, cfg=seg_config(s, bg), roi=roi_mask, min_area=float(s["min_area"]))


def _segment_chunk(args):
    a, b = args
    cap = cv2.VideoCapture(_W["path"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, a)
    out = {}
    for fid in range(a, b):
        ok, fr = cap.read()
        if not ok:
            break
        out[fid] = blobs_of(fr, _W["cfg"], _W["roi"], _W["min_area"])
    cap.release()
    return out


def draw_preview(frame, contours, flags, path):
    im = frame.copy()
    for c, o in zip(contours, flags):
        cv2.drawContours(im, [c], -1, (0, 0, 255) if o else (0, 255, 0), 2)
    cv2.imwrite(str(path), im)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video_id")
    ap.add_argument("--analysis", default=None)
    ap.add_argument("--session", default=None)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    ap.add_argument("--preview-only", action="store_true", help="only background, ROI, area bounds and previews")
    args = ap.parse_args()

    vid = args.video_id
    species = VIDEOS[vid]["species"]
    n_animals = SETUPS[species]["num_objects"]
    s = dict(SEG[species])
    od = out_dir(vid)
    analysis = args.analysis or str(od / "work" / f"{vid}.mp4")
    session = Path(args.session or od / "session")
    seg_dir = session / "segmentation"
    seg_dir.mkdir(parents=True, exist_ok=True)
    (seg_dir / "preview").mkdir(exist_ok=True)
    t0 = time.time()

    cap = cv2.VideoCapture(analysis)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    train_end_resolved = total - 1 if total < TRAIN_MAX_FRAME + 1 else TRAIN_MAX_FRAME
    train_end_cfg = -1 if train_end_resolved == total - 1 else train_end_resolved

    # background (masked median)
    bg_idx = np.linspace(0, total - 1, min(BG_SAMPLES, total), dtype=int).tolist()
    frames = read_frames(analysis, bg_idx)
    bg = masked_median_background(list(frames.values()), s["bg_mask_threshold"], s["bg_mask_dilate"])
    cv2.imwrite(str(seg_dir / "background.png"), bg)
    del frames

    roi = find_roi(species, bg, s["roi_margin_px"])
    roi_mask = build_static_roi_mask([roi], bg.shape)
    cfg = seg_config(s, bg)

    # single-animal area bounds
    a_idx = np.linspace(0, train_end_resolved, min(AREA_SAMPLES, train_end_resolved + 1), dtype=int).tolist()
    sample = read_frames(analysis, a_idx)
    counts, single_areas, all_areas = [], [], []
    sample_blobs = {}
    for fid, fr in sample.items():
        cs = blobs_of(fr, cfg, roi_mask, s["min_area"])
        sample_blobs[fid] = cs
        areas = [abs(cv2.contourArea(c)) for c in cs]
        counts.append(len(cs))
        all_areas += areas
        if len(cs) == n_animals:
            single_areas += areas
    n_frames_all_sep = sum(c == n_animals for c in counts)
    if n_frames_all_sep >= 10:
        m = float(np.median(single_areas))
        lo, hi = AREA_LO * m, AREA_HI * m
        bounds_rule = f"median area of blobs in frames with exactly {n_animals} blobs, x[{AREA_LO},{AREA_HI}]"
    else:  # GUI default: IQR rule over all blobs
        q1, med, q3 = np.percentile(all_areas, [25, 50, 75])
        m = float(med)
        lo, hi = min(med, q1 - 1.5 * (q3 - q1)), max(med, q3 + 1.5 * (q3 - q1))
        bounds_rule = "fallback GUI IQR rule (too few frames with all animals separate)"
    hist = {int(k): int(v) for k, v in zip(*np.unique(counts, return_counts=True))}
    summary = dict(video_id=vid, analysis=analysis, total_frames=total, roi=roi, settings=s,
                   area_single_median=m, area_bounds=[lo, hi], area_bounds_rule=bounds_rule,
                   sample_blob_count_hist=hist, n_sample_frames=len(sample),
                   n_sample_frames_all_separate=int(n_frames_all_sep),
                   training_frame_range=[0, train_end_resolved], training_frame_interval=TRAIN_INTERVAL)
    print(json.dumps({k: summary[k] for k in ("roi", "area_single_median", "area_bounds", "sample_blob_count_hist")}))

    # previews: 12 sampled frames with contours (green = single animal, red = outlier/crossing)
    for fid in a_idx[:: max(1, len(a_idx) // 12)][:12]:
        cs = sample_blobs.get(fid, [])
        flags = [not (lo <= abs(cv2.contourArea(c)) <= hi) for c in cs]
        im = sample[fid].copy()
        if roi["shape"] == "circle":
            cv2.circle(im, (roi["x"], roi["y"]), roi["w"], (255, 0, 0), 2)
        else:
            cv2.rectangle(im, (roi["x"], roi["y"]), (roi["x"] + roi["w"], roi["y"] + roi["h"]), (255, 0, 0), 2)
        draw_preview(im, cs, flags, seg_dir / "preview" / f"frame_{fid:06d}.jpg")
    del sample, sample_blobs

    if args.preview_only:
        (seg_dir / "headless_segmentation_summary.json").write_text(json.dumps(summary, indent=2))
        return

    # full pass over the training range, every frame (as the GUI's Processing does)
    chunk = 500
    tasks = [(a, min(a + chunk, train_end_resolved + 1)) for a in range(0, train_end_resolved + 1, chunk)]
    per_frame: dict[int, list[np.ndarray]] = {}
    with ProcessPoolExecutor(max_workers=max(1, args.workers), initializer=_init_worker,
                             initargs=(analysis, s, bg, roi_mask)) as ex:
        for res in ex.map(_segment_chunk, tasks):
            per_frame.update(res)
    blobs_in_video = []
    n_single = n_out = 0
    for fid in range(total):
        fb = []
        for c in per_frame.get(fid, []):
            out = not (lo <= abs(cv2.contourArea(c)) <= hi)
            n_out += out
            n_single += (not out)
            fb.append(SimpleNamespace(contour=c, is_outlier=bool(out)))
        blobs_in_video.append(fb)
    stem = Path(analysis).stem
    pkl = seg_dir / f"{stem}_list_of_blobs_gui.pickle"
    with open(pkl, "wb") as f:
        pickle.dump(SimpleNamespace(blobs_in_video=blobs_in_video, source_frame_count=total,
                                    background_frame_start=0, background_frame_end=total - 1,
                                    training_frame_start=0, training_frame_end=train_end_cfg,
                                    training_frame_interval=TRAIN_INTERVAL), f)

    # GUI-compatible config (identity_correction._load_tracking_seg_config reads 'settings')
    settings = {k: v for k, v in s.items() if k not in ("roi", "roi_margin_px", "bg_mask_threshold", "bg_mask_dilate")}
    settings.update(video_path=analysis, bg_method="median", background_frame_start=0, background_frame_end=-1,
                    training_frame_start=0, training_frame_end=train_end_cfg,
                    training_frame_interval=TRAIN_INTERVAL, area_outlier_method="absolute",
                    area_absolute_min=lo, area_absolute_max=hi, area_iqr_min=1.5, area_iqr_max=1.5,
                    analysis_sample_count=AREA_SAMPLES, additional_outlier_enabled=False,
                    single_blob_smoothing_enabled=False, single_blob_smoothing_level=3,
                    result_import_enabled=False, roi_sets=[roi], additional_outlier_sets=[])
    cfg_json = {"version": 4, "app": "AMADEUS segmentation (headless, scripts/tracking_pilot/amadeus/segment.py)",
                "settings": settings, "video_conversion": {},
                "hidden_state": {"sampled_frame_indices": a_idx, "analysis_iqr_stats": {},
                                 "analysis_bounds": {"area": [lo, hi]}, "analysis_signature": "headless",
                                 "analysis_is_stale": False, "frame_count": total, "current_frame": 0}}
    (seg_dir / "segmentation_gui_config.json").write_text(json.dumps(cfg_json, indent=2))
    summary.update(n_blobs_single=int(n_single), n_blobs_outlier=int(n_out), pickle=str(pkl),
                   elapsed_sec=time.time() - t0)
    (seg_dir / "headless_segmentation_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {pkl}: {n_single} single-animal blobs, {n_out} outlier blobs, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
