"""Step 6a: convert AMADEUS outputs to the pilot contract (tracks.parquet + meta.json).

    python normalize.py <video_id> [--session DIR]

Reads, from <session>/main/<model>/tracking/<dataset>/<run>/<video>/:
  buffer/buffers_id_resolved.pkl  (variant "idcorr": refinement WITH contrastive identity correction)
  buffer/buffers_filled.pkl       (variant "noidcorr": refinement without it)
  buffer/all_blobs.pkl            (raw detector output, before tracking)
and the user-facing CSV <session>/results/*{,_id_resolved}.csv (frame, cx_i, cy_i, w_i, h_i, heading_i).

Writes into results/tracking_pilot/{species}/{video_id}/amadeus/:
  tracks.parquet      long format, both variants stacked (column `variant`)
  detections.parquet  raw detector boxes (conf >= 0.1, AMADEUS's own threshold) in source coordinates
  raw/                copies of the AMADEUS CSVs, config(s), logs, timing
  meta.json

Coordinate / index conventions written to meta.json["conventions"].
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from pilot import AMADEUS_DIR, SETUPS, VIDEOS, out_dir

OBB_SOURCE_DETECTED = 1.0          # main/multi_staged_association.py
VARIANTS = {"idcorr": "id_resolved", "noidcorr": "filled"}

CONVENTIONS = {
    "frame_src": "0-based frame index in the SOURCE video (decode order). frame_src = window.start_frame + "
                 "frame_analysis; the analysis copy has the same fps, so no resampling.",
    "t_sec": "frame_src / source fps",
    "cx_cy": "OBB centre in SOURCE pixels, x to the right, y down, origin at the top-left corner of the frame "
             "(analysis pixels divided by the scale factor).",
    "w_h": "OBB side lengths in SOURCE pixels: w = long side (body length axis), h = short side "
           "(AMADEUS normalize_long_side_rect).",
    "heading": "radians in (-pi, pi], image coordinates: heading = atan2(dy, dx) of the vector pointing from the "
               "body centre towards the HEAD, x right / y down. So 0 = facing right (+x), +pi/2 = facing down "
               "(+y, towards the bottom of the image), -pi/2 = facing up. Angle grows clockwise on screen. "
               "Derived from AMADEUS heading_deg (0 = up, clockwise, unit vector (sin, -cos)) as "
               "radians(heading_deg - 90). 'Head' is what AMADEUS learned as front from the direction of motion "
               "(self-supervised); it can be flipped 180 deg for stretches of time. NaN when AMADEUS gives no "
               "direction.",
    "detected": "True iff the box is the detector's own box for this frame: AMADEUS provenance obb_source == 1 "
                "(OBB_SOURCE_DETECTED) AND obb_corrected != 1. False = synthesised or replaced by the "
                "tracker/refinement: interpolated gap fills (obb_source 3: KF / overlap / pre / post fills), "
                "pre/post-correction copies (4/5), or a detection whose box refinement replaced by an "
                "interpolated box because it was judged a position spike (obb_source 1 but obb_corrected 1, "
                "assign_type 25 = ATYPE_POS_FIX). Raw codes kept in columns obb_source, obb_corrected, "
                "assign_type (main/assign_types.py).",
    "axis_angle": "orientation of the OBB long side (body axis, no head/tail sign), radians in (-pi/2, pi/2], "
                  "same image convention as heading (isotropic scaling, so unchanged by the scale factor). "
                  "Use it when heading is NaN.",
    "conf": "YOLO confidence of the detection AMADEUS associated with this (frame, track) (score_buf). Present "
            "also for position-spike rows (the detection existed, its box was replaced); NaN for gap fills.",
    "track_id": "AMADEUS anonymous id 0..N-1 (fixed-population mode: exactly N ids by construction).",
}


def find_tracking_dir(session: Path) -> Path:
    cands = [Path(p) for p in glob.glob(str(session / "main" / "*" / "tracking" / "*" / "*" / "*"))
             if os.path.isdir(p) and os.path.isdir(os.path.join(p, "buffer"))]
    if len(cands) != 1:
        raise SystemExit(f"expected exactly one tracking output dir, found {cands}")
    return cands[0]


def heading_to_rad(deg: np.ndarray) -> np.ndarray:
    rad = np.radians(np.asarray(deg, dtype=float) - 90.0)
    return np.arctan2(np.sin(rad), np.cos(rad))


def per_tid_frame(buffers: dict, name: str) -> dict:
    out = {}
    for tid, per in (buffers.get(name) or {}).items():
        for f, v in per.items():
            try:
                out[(int(tid), int(f))] = float(v[0] if isinstance(v, (tuple, list, np.ndarray)) else v)
            except (TypeError, ValueError, IndexError):
                pass
    return out


def axis_angles(buffers: dict) -> dict:
    """Long-axis orientation of each stored OBB, radians in (-pi/2, pi/2] (image coords, x right, y down)."""
    out = {}
    for tid, per in (buffers.get("obb_buf") or {}).items():
        for f, v in per.items():
            a = np.asarray(v, dtype=float)
            if a.size != 8 or not np.isfinite(a).all():
                continue
            pts = np.stack([a[:4], a[4:]], axis=1)
            e1, e2 = pts[1] - pts[0], pts[2] - pts[1]
            e = e1 if np.hypot(*e1) >= np.hypot(*e2) else e2
            ang = math.atan2(e[1], e[0])
            if ang <= -math.pi / 2:
                ang += math.pi
            elif ang > math.pi / 2:
                ang -= math.pi
            out[(int(tid), int(f))] = ang
    return out


def load_variant(tdir: Path, session: Path, family: str, n: int):
    pkl = tdir / "buffer" / f"buffers_{family}.pkl"
    suffix = "_id_resolved" if family == "id_resolved" else ""
    csvs = [p for p in glob.glob(str(session / "results" / f"*{suffix}.csv"))
            if (family == "id_resolved") == p.endswith("_id_resolved.csv")]
    if not pkl.exists() or len(csvs) != 1:
        return None
    wide = pd.read_csv(csvs[0])
    with open(pkl, "rb") as f:
        buffers = pickle.load(f)
    src = per_tid_frame(buffers, "obb_source_buf")
    score = per_tid_frame(buffers, "score_buf")
    atype = per_tid_frame(buffers, "assign_type_buf")
    corrected = per_tid_frame(buffers, "obb_corrected_buf")
    axis = axis_angles(buffers)
    rows = []
    for tid in range(n):
        sub = wide[["frame", f"cx{tid}", f"cy{tid}", f"w{tid}", f"h{tid}", f"heading{tid}"]].copy()
        sub.columns = ["frame_analysis", "cx", "cy", "w", "h", "heading_amadeus_deg"]
        sub = sub[np.isfinite(sub["cx"]) & np.isfinite(sub["cy"])]
        sub["track_id"] = tid
        keys = list(zip([tid] * len(sub), sub["frame_analysis"].astype(int)))
        sub["obb_source"] = [src.get(k, np.nan) for k in keys]
        sub["conf"] = [score.get(k, np.nan) for k in keys]
        sub["assign_type"] = [atype.get(k, np.nan) for k in keys]
        sub["obb_corrected"] = [corrected.get(k, np.nan) for k in keys]
        sub["axis_angle"] = [axis.get(k, np.nan) for k in keys]
        rows.append(sub)
    df = pd.concat(rows, ignore_index=True)
    df["detected"] = (df["obb_source"] == OBB_SOURCE_DETECTED) & (df["obb_corrected"] != 1.0)
    return df, csvs[0], str(pkl)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video_id")
    ap.add_argument("--session", default=None)
    args = ap.parse_args()
    vid = args.video_id
    species = VIDEOS[vid]["species"]
    n = SETUPS[species]["num_objects"]
    od = out_dir(vid)
    session = Path(args.session or od / "session")
    cfg = yaml.safe_load(open(session / "config.yaml"))
    analysis = cfg["TRACKING_VIDEO_PATH"]
    if not os.path.isabs(analysis):
        analysis = os.path.normpath(os.path.join(cfg["SESSION_PATH"], analysis))
    stem = Path(analysis).stem
    prep = json.loads((od / f"prep{stem[len(vid):]}.json").read_text())   # work/<vid><suffix>.mp4 -> prep<suffix>.json
    start = prep["window"]["start_frame"]
    sx, sy = prep["scale_xy"]
    fps = prep["source"]["fps"]
    dest = od if session.name == "session" else od / session.name   # debug sessions do not overwrite
    dest.mkdir(parents=True, exist_ok=True)

    tdir = find_tracking_dir(session)
    parts, provenance = [], {}
    for variant, family in VARIANTS.items():
        res = load_variant(tdir, session, family, n)
        if res is None:
            provenance[variant] = "MISSING"
            continue
        df, csv_path, pkl_path = res
        df["variant"] = variant
        provenance[variant] = dict(csv=csv_path, buffers=pkl_path)
        parts.append(df)
    if not parts:
        raise SystemExit("no AMADEUS result found")
    tracks = pd.concat(parts, ignore_index=True)
    tracks["frame_src"] = (start + tracks["frame_analysis"]).astype(np.int64)
    tracks["t_sec"] = tracks["frame_src"] / fps
    tracks["cx"] = tracks["cx"] / sx
    tracks["cy"] = tracks["cy"] / sy
    tracks["w"] = tracks["w"] / sx          # sx == sy for every pilot video (isotropic scaling)
    tracks["h"] = tracks["h"] / sy
    tracks["heading"] = heading_to_rad(tracks["heading_amadeus_deg"])
    tracks["track_id"] = tracks["track_id"].astype(np.int64)
    cols = ["frame_src", "t_sec", "track_id", "cx", "cy", "w", "h", "heading", "detected", "conf", "variant",
            "axis_angle", "frame_analysis", "heading_amadeus_deg", "obb_source", "obb_corrected", "assign_type"]
    tracks = tracks[cols].sort_values(["variant", "frame_src", "track_id"]).reset_index(drop=True)
    tracks.to_parquet(dest / "tracks.parquet", index=False)

    # raw detector output in source coordinates
    det = pd.read_pickle(tdir / "buffer" / "all_blobs.pkl")
    pts = np.stack([det[[f"x{i}" for i in range(4)]].to_numpy(float), det[[f"y{i}" for i in range(4)]].to_numpy(float)],
                   axis=-1)
    ctr = pts.mean(axis=1)
    e1 = np.linalg.norm(pts[:, 1] - pts[:, 0], axis=1)
    e2 = np.linalg.norm(pts[:, 2] - pts[:, 1], axis=1)
    detections = pd.DataFrame({
        "frame_src": (start + det["frame"].astype(int)).astype(np.int64), "frame_analysis": det["frame"].astype(int),
        "cx": ctr[:, 0] / sx, "cy": ctr[:, 1] / sy, "w": np.maximum(e1, e2) / sx, "h": np.minimum(e1, e2) / sx,
        "conf": det["score"].astype(float), "direction_class_deg": det["direction"].astype(float),
        **{f"x{i}": pts[:, i, 0] / sx for i in range(4)}, **{f"y{i}": pts[:, i, 1] / sy for i in range(4)},
    })
    detections.to_parquet(dest / "detections.parquet", index=False)

    # raw copies
    raw = dest / "raw"
    raw.mkdir(exist_ok=True)
    for p in glob.glob(str(session / "results" / "*.csv")):
        shutil.copy2(p, raw)
    for name in ("config.yaml", "log.txt", "time.csv", "gpu.txt", "wall_times.json"):
        if (session / name).exists():
            shutil.copy2(session / name, raw / name)
    for name in ("config.yaml", "log.txt", "time.csv"):
        if (session / "noidcorr_pass" / name).exists():
            shutil.copy2(session / "noidcorr_pass" / name, raw / f"noidcorr_pass_{name}")
    for name in ("segmentation_gui_config.json", "headless_segmentation_summary.json", "background.png"):
        if (session / "segmentation" / name).exists():
            shutil.copy2(session / "segmentation" / name, raw / f"segmentation_{name}")
    for p in glob.glob(str(tdir / "log" / "*.json")) + glob.glob(str(tdir / "*.csv")):
        shutil.copy2(p, raw / f"tracking_{Path(p).name}")

    # runtimes per AMADEUS stage (time.csv of both passes)
    def stage_times(p: Path) -> dict:
        if not p.exists():
            return {}
        t = pd.read_csv(p)
        return {r.stage: float(r.elapsed_seconds) for r in t.itertuples() if r.status in ("success", "failed")}

    st = stage_times(session / "time.csv")
    st2 = stage_times(session / "noidcorr_pass" / "time.csv")
    git_commit = (AMADEUS_DIR / ".git" / "HEAD").read_text().strip()
    gpu = (session / "gpu.txt").read_text().strip() if (session / "gpu.txt").exists() else None
    seg_summary = json.loads((session / "segmentation" / "headless_segmentation_summary.json").read_text())
    wall = json.loads((session / "wall_times.json").read_text()) if (session / "wall_times.json").exists() else {}
    meta = dict(
        video_id=vid, species=species, num_animals=n, source_path=prep["source_path"], source_ffprobe=prep["source"],
        window=prep["window"], analysis_path=prep["analysis_path"], analysis_ffprobe=prep["analysis"],
        scale_factor=prep["scale"], scale_xy=prep["scale_xy"], fps_used=prep["analysis"]["fps"],
        frame_mapping=prep["frame_mapping"], frame_alignment_checks=prep["alignment_checks"],
        amadeus=dict(repo="https://github.com/jpmyrmecol/AMADEUS", tag="v1.1.6", commit=git_commit,
                     install=str(AMADEUS_DIR)),
        settings=dict(num_objects=n, overlap="super_heavy (GUI: Very heavy) -> EMBEDDING.ENABLE",
                      backward_movement="yes", detector="trained per video (YOLO11n-OBB, fine-tuned from yolo11n-obb.pt)",
                      train_img_size=cfg["TRAIN_IMG_SIZE"], num_images=cfg["NUM_IMAGES"],
                      epochs=cfg["training"]["EPOCHS"], analysis=cfg["analysis"], embedding=cfg["EMBEDDING"],
                      auto_params={k: cfg.get(k) for k in ("MIN_OVERLAP", "MAX_OVERLAP", "DIR_MIN_SEC", "FREE_SCALE",
                                                           "LOCALIZED", "CLUSTER_FRAMES")},
                      segmentation=seg_summary),
        runtimes_sec=dict(amadeus_stages_main_pass=st, amadeus_stages_noidcorr_pass=st2, wall=wall,
                          train=st.get("obb_detector_training"), detect=st.get("obb_detection"),
                          track=st.get("multi_staged_association"),
                          refinement_with_idcorr=st.get("refinement"), refinement_without_idcorr=st2.get("refinement")),
        gpu=gpu, variants=provenance, conventions=CONVENTIONS,
        n_rows={v: int((tracks["variant"] == v).sum()) for v in tracks["variant"].unique()},
    )
    (dest / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"wrote {dest / 'tracks.parquet'} ({len(tracks)} rows), variants={list(provenance)}")


if __name__ == "__main__":
    main()
