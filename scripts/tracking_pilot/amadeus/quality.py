"""Step 6b: raw-quality numbers and a sanity-check image grid for one video.

    python quality.py <video_id> [--session DIR] [--conf 0.5]

Reads tracks.parquet / detections.parquet / meta.json written by normalize.py and writes
quality.json and quality_grid.png next to them.

Definitions (N = number of animals: 3 ants, 4 mice; all over the frames of the analysed window):
  detector count  = number of raw YOLO boxes in a frame with conf >= --conf (default 0.25, the
                    Ultralytics default prediction threshold; fixed before looking at full-run results),
                    after the detector's own class-agnostic NMS (IoU 0.8). Also reported at 0.1 (what the
                    AMADEUS tracker ingests), 0.5 and 0.7.
  all_present     = frames in which the final tracks contain all N track_ids.
  filled          = a (frame, track) row whose box is not a real detection (detected == False).
  merge event     = a maximal run of consecutive frames with detector count < N.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from pilot import SETUPS, VIDEOS, out_dir

COLORS = [(0, 0, 255), (0, 200, 0), (255, 0, 0), (0, 200, 255), (255, 0, 255), (255, 255, 0)]


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, length) of maximal runs of True."""
    m = np.concatenate([[False], mask.astype(bool), [False]])
    d = np.diff(m.astype(int))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return list(zip(starts.tolist(), (ends - starts).tolist()))


def det_counts(det: pd.DataFrame, frames: np.ndarray, conf: float) -> np.ndarray:
    c = det[det["conf"] >= conf].groupby("frame_src").size()
    return c.reindex(frames, fill_value=0).to_numpy()


def count_hist(counts: np.ndarray, n: int) -> dict:
    total = len(counts)
    h = {str(k): float(np.mean(counts == k)) for k in range(n + 2)}
    h[f">={n + 2}"] = float(np.mean(counts >= n + 2))
    h["n_frames"] = int(total)
    return h


def obb_corners(cx, cy, w, h, heading_rad):
    """Box corners from centre/size/heading; with NaN heading use the raw AMADEUS row instead."""
    ux, uy = np.cos(heading_rad), np.sin(heading_rad)
    vx, vy = -uy, ux
    pts = [(cx + sx * w / 2 * ux + sy * h / 2 * vx, cy + sx * w / 2 * uy + sy * h / 2 * vy)
           for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
    return np.array(pts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video_id")
    ap.add_argument("--session", default=None)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    vid = args.video_id
    n = SETUPS[VIDEOS[vid]["species"]]["num_objects"]
    od = out_dir(vid)
    dest = od if (args.session is None or Path(args.session).name == "session") else od / Path(args.session).name
    meta = json.loads((dest / "meta.json").read_text())
    tracks = pd.read_parquet(dest / "tracks.parquet")
    det = pd.read_parquet(dest / "detections.parquet")
    start, n_frames = meta["window"]["start_frame"], meta["window"]["n_frames"]
    fps = meta["source_ffprobe"]["fps"]
    frames = np.arange(start, start + n_frames)

    counts = det_counts(det, frames, args.conf)
    detector = {"conf_threshold": args.conf,
                "frac_exactly_N": float(np.mean(counts == n)),
                "count_distribution": count_hist(counts, n),
                "frac_exactly_N_at_conf": {str(c): float(np.mean(det_counts(det, frames, c) == n))
                                           for c in (0.1, 0.25, 0.5, 0.7)}}
    under = counts < n
    ep = runs(under)
    durs = np.array([l for _, l in ep], dtype=float) / fps
    merges = {"frac_frames_below_N": float(under.mean()), "n_episodes": len(ep),
              "duration_sec_median": float(np.median(durs)) if len(durs) else 0.0,
              "duration_sec_p90": float(np.percentile(durs, 90)) if len(durs) else 0.0,
              "duration_sec_max": float(durs.max()) if len(durs) else 0.0,
              "total_sec_below_N": float(durs.sum())}

    per_variant = {}
    for variant, tv in tracks.groupby("variant"):
        per_frame = tv.groupby("frame_src").agg(n_tracks=("track_id", "nunique"), n_det=("detected", "sum"))
        per_frame = per_frame.reindex(frames, fill_value=0)
        allp = per_frame["n_tracks"] == n
        per_variant[variant] = {
            "frac_frames_all_N_present": float(allp.mean()),
            "frac_frames_all_N_present_and_all_detected": float((allp & (per_frame["n_det"] == n)).mean()),
            "frac_of_all_present_frames_with_any_filled": float((allp & (per_frame["n_det"] < n)).sum()
                                                                / max(1, allp.sum())),
            "frac_track_rows_detected": float(tv["detected"].mean()),
            "frac_track_rows_detected_during_merge_frames": float(
                tv[tv["frame_src"].isin(frames[under])]["detected"].mean()) if under.any() else None,
            "n_distinct_track_ids": int(tv["track_id"].nunique()),
            "track_rows_by_obb_source": {str(k): int(v) for k, v in tv["obb_source"].value_counts().items()},
            "frac_track_rows_position_spike_replaced": float(((tv["obb_source"] == 1) & (tv["obb_corrected"] == 1)).mean()),
            "frac_track_rows_gap_filled": float((tv["obb_source"] != 1).mean()),
            "frac_rows_heading_nan": float(tv["heading"].isna().mean()),
        }

    rt = meta["runtimes_sec"]
    quality = {"video_id": vid, "N": n, "n_frames": int(n_frames), "fps": fps, "detector": detector,
               "merge_events": merges, "tracks": per_variant,
               "runtime_sec": {k: rt.get(k) for k in ("train", "detect", "track", "refinement_with_idcorr",
                                                       "refinement_without_idcorr")} | {"wall": rt.get("wall")},
               "gpu": meta.get("gpu")}
    (dest / "quality.json").write_text(json.dumps(quality, indent=2))
    print(json.dumps(quality, indent=1))

    # sanity grid: 12 random frames with detector count < N; final idcorr tracks (or noidcorr if missing)
    variant = "idcorr" if "idcorr" in per_variant else "noidcorr"
    tv = tracks[tracks["variant"] == variant]
    cand = frames[under]
    rng = np.random.default_rng(args.seed)
    pick = np.sort(rng.choice(cand, size=min(12, len(cand)), replace=False)) if len(cand) else []
    sx, sy = meta["scale_xy"]
    cap = cv2.VideoCapture(meta["analysis_path"])
    tiles = []
    for fs in pick:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fs - start))
        ok, im = cap.read()
        if not ok:
            continue
        for r in det[(det["frame_src"] == fs) & (det["conf"] >= args.conf)].itertuples():
            p = np.array([[getattr(r, f"x{i}") * sx, getattr(r, f"y{i}") * sy] for i in range(4)], np.int32)
            cv2.polylines(im, [p], True, (255, 255, 255), 1)
        for r in tv[tv["frame_src"] == fs].itertuples():
            col = COLORS[r.track_id % len(COLORS)]
            hd = r.heading if np.isfinite(r.heading) else (r.axis_angle if np.isfinite(r.axis_angle) else 0.0)
            p = (obb_corners(r.cx, r.cy, r.w, r.h, hd) * [sx, sy]).astype(np.int32)
            cv2.polylines(im, [p], True, col, 2 if r.detected else 1, lineType=cv2.LINE_AA)
            c = (int(r.cx * sx), int(r.cy * sy))
            if np.isfinite(r.heading):
                tip = (int(c[0] + 0.6 * r.w * sx * np.cos(r.heading)), int(c[1] + 0.6 * r.w * sy * np.sin(r.heading)))
                cv2.arrowedLine(im, c, tip, col, 2, tipLength=0.3)
            cv2.putText(im, f"{r.track_id}{'' if r.detected else '*'}", (c[0] + 8, c[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
        k = int(counts[fs - start])
        cv2.putText(im, f"frame_src {fs}  det>={args.conf}: {k}/{n}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 0, 0), 3)
        cv2.putText(im, f"frame_src {fs}  det>={args.conf}: {k}/{n}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 255, 255), 1)
        tiles.append(cv2.resize(im, (600, 600)))
    cap.release()
    if tiles:
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
        cv2.imwrite(str(dest / "quality_grid.png"), grid)
        print(f"wrote {dest / 'quality_grid.png'} ({variant}; thick = detected, thin + '*' = filled; "
              f"white = raw detector boxes)")


if __name__ == "__main__":
    main()
