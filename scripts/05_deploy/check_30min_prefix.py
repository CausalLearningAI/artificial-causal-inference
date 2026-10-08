#!/usr/bin/env python3
"""
Pre-rebuild checks for extending an ants version from 10 to 30 minutes.

For a few observations, re-standardizes the 30-min source video with
standardize.py's exact ffmpeg call (FFmpeg of the current env), then measures
how far the first 10 minutes drift from the existing 10-min artifacts, and how
tracking quality over minutes 10-30 compares with minutes 0-10.

Per observation (all heavy files live in the job's /localhome staging dir):
  1. video   — framemd5 equality and mean |pixel diff| of the first N_OLD frames
               (new 30-min video vs old observations/full video)
  2. jpgs    — get_frames.extract_frames on the new video; md5 / mean |diff| vs
               the old frames/full JPGs on a subset of frames
  3. tracker — AntTracker on the new video twice:
                 bg=old : the cached backgrounds/q85 npy (10-min JPGs), what
                          get_tracking.py does today when the cache exists
                 bg=new : q85 of 100 frames spread over the new 30-min JPGs
                          (calibration.get_background recipe, as v6 had)
               rows < N_OLD vs the old tracking CSV (max distances), seam
               continuity at N_OLD-1 -> N_OLD for the splice old[:N_OLD] + new[N_OLD:],
               quality metrics per segment (0-10 vs 10-30)
  4. sheet   — PNG of blue/yellow POV crops around the seam from the splice
Reference (no staging needed, small CSVs read once):
  - the same tracking-quality metrics on v6 and vA (processed in one pass)
  - per-video positive rate / mean probability, 0-10 vs 10-30, of the deployed
    model's annotations of v6 and vA (results/ppci/ants/annotations/)

Usage (CPU job):
  python scripts/05_deploy/check_30min_prefix.py --obs 5_1_5 5_12_7 5_24_3 5_8_3 5_19_5
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.standardize import process_video          # noqa: E402
from src.dataset.get_frames import extract_frames        # noqa: E402
from src.tracking.get_tracking import build_tracker      # noqa: E402
from src.tracking.tracker import AntTracker              # noqa: E402

SOURCE_FPS = 30
ML_FPS = 5
IDS = ("blue", "yellow", "focal")


def _sh(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


def framemd5(video: Path, n: int) -> list[str]:
    out = _sh(["ffmpeg", "-v", "error", "-i", str(video), "-map", "0:v", "-frames:v", str(n),
               "-f", "framemd5", "-"])
    return [ln.split(",")[-1].strip() for ln in out.splitlines() if ln and not ln.startswith("#")]


def video_diff(a: Path, b: Path, n: int) -> np.ndarray:
    """Per-frame mean |a - b| (BGR, uint8 scale) over the first n frames."""
    ca, cb = cv2.VideoCapture(str(a)), cv2.VideoCapture(str(b))
    out = []
    for _ in range(n):
        ra, fa = ca.read()
        rb, fb = cb.read()
        if not (ra and rb):
            break
        out.append(float(np.abs(fa.astype(np.int16) - fb.astype(np.int16)).mean()))
    ca.release(); cb.release()
    return np.array(out)


def n_frames(video: Path) -> int:
    out = _sh(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
               "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(video)])
    return int(out.strip())


def background_from_jpgs(frames_dir: Path, quantile: float, n: int = 100) -> np.ndarray:
    """Same recipe as src/tracking/calibration.get_background, without its cache."""
    files = sorted(frames_dir.glob("frame_*.jpg"))
    idx = np.linspace(0, len(files) - 1, min(n, len(files)), dtype=int)
    stack = np.array([cv2.imread(str(files[i])) for i in idx])
    return np.quantile(stack, quantile, axis=0).astype(np.uint8)


def _xy(df: pd.DataFrame, kind: str, ident: str) -> np.ndarray:
    pre = "mark_" if kind == "mark" else ""
    if kind == "mark" and ident == "focal":
        pre = ""
    return df[[f"{pre}{ident}_x", f"{pre}{ident}_y"]].to_numpy(dtype=float)


def compare_rows(old: pd.DataFrame, new: pd.DataFrame, lo: int, hi: int) -> dict:
    """Max distance between old and new positions on frames [lo, hi)."""
    o, n = old.iloc[lo:hi], new.iloc[lo:hi]
    res = {}
    for kind in ("centroid", "mark"):
        for ident in IDS:
            if kind == "mark" and ident == "focal":
                continue
            d = np.linalg.norm(_xy(o, kind, ident) - _xy(n, kind, ident), axis=1)
            res[f"{kind}_{ident}_max"] = float(d.max())
            res[f"{kind}_{ident}_n_gt1px"] = int((d > 1).sum())
    return res


def seam(old: pd.DataFrame, new: pd.DataFrame, n_old: int) -> dict:
    """Continuity of the splice old[:n_old] + new[n_old:] at frame n_old.

    For each identity: jump old[n_old-1] -> new[n_old], the new run's own jump
    new[n_old-1] -> new[n_old], and the percentile of the splice jump among
    all frame-to-frame jumps of the new run. swap_<a>_<b>: True when new[n_old]
    of identity a lies closer to old[n_old-1] of identity b than of a.
    """
    res = {}
    for kind in ("centroid", "mark"):
        ids = IDS if kind == "centroid" else ("blue", "yellow")
        for ident in ids:
            p_old = _xy(old, kind, ident)[n_old - 1]
            p_new_prev = _xy(new, kind, ident)[n_old - 1]
            p_new = _xy(new, kind, ident)[n_old]
            jumps = np.linalg.norm(np.diff(_xy(new, kind, ident), axis=0), axis=1)
            j = float(np.linalg.norm(p_new - p_old))
            res[f"{kind}_{ident}_splice_jump"] = j
            res[f"{kind}_{ident}_own_jump"] = float(np.linalg.norm(p_new - p_new_prev))
            res[f"{kind}_{ident}_splice_jump_pct"] = float((jumps < j).mean() * 100)
        for a in ids:
            for b in ids:
                if a == b:
                    continue
                p_new = _xy(new, kind, a)[n_old]
                d_same = np.linalg.norm(p_new - _xy(old, kind, a)[n_old - 1])
                d_other = np.linalg.norm(p_new - _xy(old, kind, b)[n_old - 1])
                res[f"{kind}_swap_{a}_{b}"] = bool(d_other < d_same)
    return res


def quality(df: pd.DataFrame, lo: int, hi: int, radius: int, size: int = 512) -> dict:
    """Tracking-quality metrics on frames [lo, hi)."""
    d = df.iloc[lo:hi]
    n = len(d)
    if n < 2:
        return {}
    res = {"n_frames": n}
    for c in ("blue", "yellow"):
        res[f"dot_{c}_rate"] = float(d[f"raw_{c}_x"].notna().mean())
    nb = d["n_blobs"].to_numpy()
    res["n_blobs_0_rate"] = float((nb == 0).mean())
    res["n_blobs_3_rate"] = float((nb == 3).mean())
    for ident in IDS:
        xy = _xy(d, "centroid", ident)
        step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        res[f"{ident}_frozen_rate"] = float((step == 0).mean())
        res[f"{ident}_jump_gt50_per1k"] = float((step > 50).sum() * 1000 / (n - 1))
        res[f"{ident}_step_p99"] = float(np.percentile(step, 99))
    for c in ("blue", "yellow"):
        xy = _xy(d, "mark", c)
        step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        res[f"mark_{c}_jump_gt50_per1k"] = float((step > 50).sum() * 1000 / (n - 1))
        # crop padding: fraction of the 2r x 2r POV crop that falls outside the frame
        x, y = np.round(xy[:, 0]), np.round(xy[:, 1])
        w = np.clip(np.minimum(x + radius, size) - np.maximum(x - radius, 0), 0, None)
        h = np.clip(np.minimum(y + radius, size) - np.maximum(y - radius, 0), 0, None)
        pad = 1 - (w * h) / (2 * radius) ** 2
        res[f"crop_{c}_pad_gt50_rate"] = float((pad > 0.5).mean())
    return res


def _median_dict(rows: list[dict]) -> dict:
    df = pd.DataFrame(rows)
    return {k: float(np.median(df[k])) for k in df.columns}


def reference_tracking(version: str, n_old: int, radius: int) -> dict:
    tdir = ROOT / "dataset" / "ants" / version / "tracking"
    seg = {"0-10": [], "10-30": []}
    for p in sorted(tdir.glob("*.csv")):
        df = pd.read_csv(p)
        seg["0-10"].append(quality(df, 0, n_old, radius))
        seg["10-30"].append(quality(df, n_old, len(df), radius))
    return {k: _median_dict(v) | {"n_videos": len(v)} for k, v in seg.items()}


def reference_predictions(version: str, split_frame: int) -> dict:
    """Per-video 0-10 vs 10-30 positive rate and mean probability (30 fps CSVs)."""
    adir = ROOT / "results" / "ppci" / "ants" / "annotations" / version
    rows = []
    for p in sorted(adir.glob("*.csv")):
        df = pd.read_csv(p)
        early, late = df[df.frame_id < split_frame], df[df.frame_id >= split_frame]
        r = {"obs": p.stem, "n_rows": len(df)}
        for o in ("Y2F", "B2F"):
            for name, part in (("early", early), ("late", late)):
                r[f"{o}_pos_{name}"] = float(part[o].mean())
                r[f"{o}_prob_{name}"] = float(part[f"{o}_prob"].mean())
        rows.append(r)
    df = pd.DataFrame(rows)
    out = {"n_videos": len(df), "n_rows_unique": sorted(df.n_rows.unique().tolist())}
    for o in ("Y2F", "B2F"):
        for m in ("pos", "prob"):
            e, l = df[f"{o}_{m}_early"], df[f"{o}_{m}_late"]
            out[f"{o}_{m}_early_mean"] = float(e.mean())
            out[f"{o}_{m}_late_mean"] = float(l.mean())
            out[f"{o}_{m}_late_over_early_pooled"] = float(l.mean() / e.mean()) if e.mean() > 0 else None
            out[f"{o}_{m}_late_minus_early_median"] = float((l - e).median())
    return out


def contact_sheet(rows: list[tuple[str, list[np.ndarray]]], out: Path, frames: list[int]) -> None:
    tiles = []
    for label, crops in rows:
        strip = np.concatenate(crops, axis=1)
        bar = np.full((18, strip.shape[1], 3), 255, np.uint8)
        cv2.putText(bar, label, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        tiles += [bar, strip]
    head = np.full((18, tiles[0].shape[1], 3), 255, np.uint8)
    w = tiles[1].shape[1] // len(frames)
    for i, f in enumerate(frames):
        cv2.putText(head, f"f{f}{' seam' if i == len(frames) // 2 else ''}", (i * w + 4, 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200) if i >= len(frames) // 2 else (0, 0, 0),
                    1, cv2.LINE_AA)
    cv2.imwrite(str(out), np.concatenate([head] + tiles, axis=0))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--obs", nargs="+", required=True)
    ap.add_argument("--version", default="v5")
    ap.add_argument("--minutes", type=int, default=30)
    ap.add_argument("--old-minutes", type=int, default=10)
    ap.add_argument("--jpg-step", type=int, default=10, help="Compare every k-th old JPG (+ the last 10)")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results" / "ppci" / "ants" / "regress_30min")
    ap.add_argument("--refs", nargs="*", default=["v6", "vA"])
    args = ap.parse_args()

    stage = Path(os.environ.get("STAGE_DIR") or
                 f"/localhome/{os.environ['USER']}/{os.environ.get('SLURM_JOB_ID', 'local')}")
    stage.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    trk_out = args.out_dir / "tracking_test"
    trk_out.mkdir(exist_ok=True)

    v = args.version
    n_old = args.old_minutes * 60 * ML_FPS
    data_cfg = OmegaConf.load(ROOT / "configs" / "data" / "ants.yaml")
    std_cfg = OmegaConf.create({"data": OmegaConf.to_container(data_cfg), "overwrite": {"videos": True}})
    tcfg = OmegaConf.load(ROOT / "configs" / "tracking" / "ants" / f"{v}.yaml")
    tracker = build_tracker(tcfg)
    radius = int(tcfg.pov_radius)
    exp = pd.read_csv(ROOT / "data" / "ants" / v / "experiment.csv").set_index("observation_id")
    ffv = _sh(["ffmpeg", "-version"]).splitlines()[0]

    report = {"ffmpeg": ffv, "n_old_frames": n_old, "obs": {}}
    sheet_rows = []
    sheet_frames = list(range(n_old - 4, n_old + 4))

    for obs in args.obs:
        t0 = time.time()
        r = {}
        wd = stage / obs
        wd.mkdir(exist_ok=True)
        # ── stage inputs (one copy each) ──────────────────────────────────────
        src = wd / "source.mkv"
        shutil.copy(ROOT / "data" / "ants" / v / "observations" / "source" / f"{obs}.mkv", src)
        old_vid = wd / "old_full.mkv"
        shutil.copy(ROOT / "data" / "ants" / v / "observations" / "full" / f"{obs}.mkv", old_vid)
        old_trk = pd.read_csv(ROOT / "dataset" / "ants" / v / "tracking" / f"{obs}.csv")
        old_bg = np.load(ROOT / "dataset" / "ants" / v / "backgrounds" / "q85" / f"{obs}.npy")
        jpg_idx = sorted(set(range(0, n_old, args.jpg_step)) | set(range(n_old - 10, n_old)))
        old_jpg = wd / "old_jpg"
        old_jpg.mkdir(exist_ok=True)
        old_frames_dir = ROOT / "dataset" / "ants" / v / "frames" / "full" / obs
        for i in jpg_idx:
            shutil.copy(old_frames_dir / f"frame_{i:06d}.jpg", old_jpg / f"frame_{i:06d}.jpg")

        # ── 1. re-standardize at 30 min, exact standardize.py call ────────────
        start = int(exp.loc[obs, "start_frame"])
        end = start + args.minutes * 60 * SOURCE_FPS
        new_vid = wd / "new_full.mkv"
        ok = process_video(src, new_vid, start, end, std_cfg)
        if not ok:
            raise RuntimeError(f"{obs}: standardize failed")
        r["new_video_frames"] = n_frames(new_vid)
        r["old_video_frames"] = n_frames(old_vid)
        md_old, md_new = framemd5(old_vid, n_old), framemd5(new_vid, n_old)
        eq = np.array([a == b for a, b in zip(md_old, md_new)])
        r["framemd5_equal"] = int(eq.sum())
        r["framemd5_compared"] = int(len(eq))
        r["framemd5_first_mismatch"] = int(np.argmin(eq)) if not eq.all() else None
        vd = video_diff(old_vid, new_vid, n_old)
        r["video_mad_mean"] = float(vd.mean())
        r["video_mad_max"] = float(vd.max())
        r["video_mad_last10_mean"] = float(vd[-10:].mean())

        # ── 2. JPGs with get_frames' exact call ───────────────────────────────
        new_jpg = wd / "new_jpg"
        extract_frames(new_vid, new_jpg, overwrite=True, fps=data_cfg.target_fps,
                       frame_format=data_cfg.frame_format)
        r["new_jpg_count"] = len(list(new_jpg.glob("frame_*.jpg")))
        md_eq, mads = 0, []
        for i in jpg_idx:
            a, b = old_jpg / f"frame_{i:06d}.jpg", new_jpg / f"frame_{i:06d}.jpg"
            md_eq += hashlib.md5(a.read_bytes()).digest() == hashlib.md5(b.read_bytes()).digest()
            ia, ib = cv2.imread(str(a)).astype(np.int16), cv2.imread(str(b)).astype(np.int16)
            mads.append(float(np.abs(ia - ib).mean()))
        r["jpg_compared"] = len(jpg_idx)
        r["jpg_md5_equal"] = int(md_eq)
        r["jpg_mad_mean"] = float(np.mean(mads))
        r["jpg_mad_max"] = float(np.max(mads))

        # ── 3. tracking, two backgrounds ──────────────────────────────────────
        new_bg = background_from_jpgs(new_jpg, float(tcfg.quantile))
        r["bg_old_vs_new_mad"] = float(np.abs(old_bg.astype(np.int16) - new_bg.astype(np.int16)).mean())
        runs = {}
        for name, bg in (("bg_old", old_bg), ("bg_new", new_bg)):
            df = tracker.track_video(new_vid, background=bg)
            df.to_csv(trk_out / f"{obs}_{name}.csv", index=False)
            runs[name] = df
            rr = {
                "n_rows": len(df),
                "prefix_all": compare_rows(old_trk, df, 0, n_old),
                "prefix_last10": compare_rows(old_trk, df, n_old - 10, n_old),
                "seam": seam(old_trk, df, n_old),
                "quality_0_10": quality(df, 0, n_old, radius),
                "quality_10_30": quality(df, n_old, len(df), radius),
            }
            r[name] = rr
        r["quality_old_0_10"] = quality(old_trk, 0, n_old, radius)

        # ── 4. seam contact sheet (splice with bg_old: old rows/JPGs < n_old) ─
        for name in ("bg_old", "bg_new"):
            df = runs[name]
            for c in ("blue", "yellow"):
                crops = []
                for f in sheet_frames:
                    if f < n_old:
                        img, row = cv2.imread(str(old_jpg / f"frame_{f:06d}.jpg")), old_trk.iloc[f]
                    else:
                        img, row = cv2.imread(str(new_jpg / f"frame_{f:06d}.jpg")), df.iloc[f]
                    crops.append(AntTracker.crop_pov(img, row[f"mark_{c}_x"], row[f"mark_{c}_y"], radius))
                sheet_rows.append((f"{obs} {c} (new rows: {name})", crops))

        report["obs"][obs] = r
        print(f"[{obs}] frames new={r['new_video_frames']} old={r['old_video_frames']}  "
              f"framemd5 {r['framemd5_equal']}/{r['framemd5_compared']}  "
              f"video MAD {r['video_mad_mean']:.3f} (max {r['video_mad_max']:.3f})  "
              f"jpg md5 {r['jpg_md5_equal']}/{r['jpg_compared']} MAD {r['jpg_mad_mean']:.3f}  "
              f"({time.time() - t0:.0f}s)", flush=True)
        for name in ("bg_old", "bg_new"):
            pa, pl, sm = r[name]["prefix_all"], r[name]["prefix_last10"], r[name]["seam"]
            print(f"    {name}: prefix max mark blue/yellow {pa['mark_blue_max']:.1f}/"
                  f"{pa['mark_yellow_max']:.1f}px (frames>1px {pa['mark_blue_n_gt1px']}/"
                  f"{pa['mark_yellow_n_gt1px']}); last10 {pl['mark_blue_max']:.1f}/"
                  f"{pl['mark_yellow_max']:.1f}px; splice jump mark blue/yellow "
                  f"{sm['mark_blue_splice_jump']:.1f}/{sm['mark_yellow_splice_jump']:.1f}px",
                  flush=True)
        shutil.rmtree(wd)

    contact_sheet(sheet_rows, args.out_dir / "seam_contact_sheet_pre.png", sheet_frames)

    report["reference_tracking"] = {ref: reference_tracking(ref, n_old, radius) for ref in args.refs}
    report["reference_tracking"][f"{v}_old_10min"] = {"0-10": reference_tracking(v, n_old, radius)["0-10"]}
    split = args.old_minutes * 60 * SOURCE_FPS
    report["reference_predictions"] = {ref: reference_predictions(ref, split) for ref in args.refs}

    out = args.out_dir / "prefix_check.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
