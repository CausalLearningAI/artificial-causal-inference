#!/usr/bin/env python3
"""
Per-observation preparation of a NEW ants version (native 30-min clips), one array task per
video, staged on node-local disk. It makes the same calls as the serial stage scripts, so the
outputs are the ones those scripts would write:

  frames  standardize (src/data/standardize.process_video) -> data/ants/<v>/observations/full/;
          JPGs at target_fps (src/dataset/get_frames.extract_frames) -> dataset/ants/<v>/frames/full/<stem>/
  track   tracking of the standardized video with the cached background
          (get_tracking.build_tracker + calibration.get_background) -> dataset/ants/<v>/tracking/<obs>.csv;
          POV crops on the persistent mark positions (AntTracker.crop_pov, as get_pov_frames.py)
          -> dataset/ants/<v>/frames/pov/{blue,yellow}/<stem>/

`track` without `frames` in the same task re-extracts the JPGs locally from the standardized
video and checks them byte for byte against the NFS JPGs on a sample, so the crops are cut from
the same pixels as frames/full without reading 9000 files back from NFS.

Resume-friendly: `frames` skips an observation whose video and complete JPG directory exist;
`track` skips one whose tracking CSV and complete POV directories exist (unless --overwrite).

Usage (inside a Slurm array; see job_prepare_obs.sh):
    python scripts/00_data/prepare_obs.py --version v4_infested --idx 0 --steps frames,track
Per-task logs: results/ppci/ants/<version>/prepare_log/<obs>.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

COLS = {"blue": ("mark_blue_x", "mark_blue_y"), "yellow": ("mark_yellow_x", "mark_yellow_y")}


def _stage(name: str) -> Path:
    """Node-local staging dir: /localhome/$USER/<name>, else $TMPDIR."""
    for base in (Path("/localhome") / os.environ["USER"], Path(os.environ.get("TMPDIR", "/tmp")) / os.environ["USER"]):
        try:
            base.mkdir(parents=True, exist_ok=True)
            return base / name
        except OSError:
            continue
    raise RuntimeError("no writable node-local staging directory")


def _n_jpg(d: Path) -> int:
    return len(list(d.glob("frame_*.jpg"))) if d.is_dir() else 0


def _copy_dir(src: Path, dst: Path) -> int:
    """Copy every JPG of src into dst (write to .tmp, then rename, so a killed job leaves no truncated JPG)."""
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(src.glob("frame_*.jpg")):
        tmp = dst / (p.name + ".tmp")
        shutil.copyfile(p, tmp)
        os.replace(tmp, dst / p.name)
        n += 1
    return n


def _md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--version", required=True)
    ap.add_argument("--idx", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--obs", default=None, help="observation id (overrides --idx)")
    ap.add_argument("--steps", default="frames,track")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--n-md5", type=int, default=60, help="JPGs compared byte for byte when re-extracting")
    a = ap.parse_args()
    steps = set(a.steps.split(","))

    from src.data.standardize import process_video
    from src.dataset.get_frames import extract_frames

    v = a.version
    exp = pd.read_csv(ROOT / "data" / "ants" / v / "experiment.csv")
    exp = exp[exp["valid"] == 1].reset_index(drop=True)
    row = exp.set_index("observation_id").loc[a.obs] if a.obs else exp.iloc[a.idx]
    obs = a.obs or str(row["observation_id"])
    data_cfg = OmegaConf.load(ROOT / "configs" / "data" / "ants.yaml")
    std_cfg = OmegaConf.create({"data": OmegaConf.to_container(data_cfg), "overwrite": {"videos": True}})
    fps = data_cfg.target_fps
    n_expected = int(row["end_frame"] - row["start_frame"]) // (30 // fps)
    stem = Path(row["observation_file"]).stem

    vid_nfs = ROOT / "data" / "ants" / v / "observations" / "full" / row["observation_file"]
    fdir = ROOT / "dataset" / "ants" / v / "frames" / "full" / stem
    trk = ROOT / "dataset" / "ants" / v / "tracking" / f"{obs}.csv"
    pdirs = {c: ROOT / "dataset" / "ants" / v / "frames" / "pov" / c / stem for c in COLS}
    logdir = ROOT / "results" / "ppci" / "ants" / v / "prepare_log"
    logdir.mkdir(parents=True, exist_ok=True)
    logp = logdir / f"{obs}.json"
    log = json.loads(logp.read_text()) if logp.exists() else {}
    log.pop("error", None)
    log.update({"obs": obs, "n_expected": n_expected, "node": os.uname().nodename,
                "ffmpeg": os.popen("ffmpeg -version").readline().strip()})

    stage = _stage(f"{os.environ.get('SLURM_JOB_ID', 'local')}_{obs}")
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    jpg = stage / "frames"
    vid = stage / row["observation_file"]
    t0 = time.time()
    try:
        # ── frames ────────────────────────────────────────────────────────────
        if "frames" in steps:
            if not a.overwrite and vid_nfs.exists() and _n_jpg(fdir) == n_expected:
                log["frames"] = "skipped (complete)"
            else:
                src = stage / ("source" + Path(row["observation_file"]).suffix)
                shutil.copyfile(ROOT / "data" / "ants" / v / "observations" / "source" / row["observation_file"], src)
                if not process_video(src, vid, int(row["start_frame"]), int(row["end_frame"]), std_cfg):
                    raise RuntimeError("standardize failed")
                src.unlink()
                vid_nfs.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(vid, str(vid_nfs) + ".tmp")
                os.replace(str(vid_nfs) + ".tmp", vid_nfs)
                extract_frames(vid, jpg, overwrite=True, fps=fps, frame_format=data_cfg.frame_format)
                n = _n_jpg(jpg)
                if n != n_expected:
                    raise RuntimeError(f"{n} JPGs, expected {n_expected}")
                if fdir.exists():
                    shutil.rmtree(fdir)
                log["frames_copied"] = _copy_dir(jpg, fdir)
                log["frames_seconds"] = round(time.time() - t0, 1)

        # ── tracking + POV crops ─────────────────────────────────────────────
        if "track" in steps:
            if not a.overwrite and trk.exists() and all(_n_jpg(d) == n_expected for d in pdirs.values()):
                log["track"] = "skipped (complete)"
            else:
                t1 = time.time()
                from src.tracking.calibration import get_background
                from src.tracking.get_tracking import build_tracker
                from src.tracking.tracker import AntTracker
                tcfg = OmegaConf.load(ROOT / "configs" / "tracking" / "ants" / f"{v}.yaml")
                radius = int(tcfg.pov_radius)
                if _n_jpg(fdir) != n_expected:
                    raise RuntimeError(f"{fdir}: {_n_jpg(fdir)} JPGs, expected {n_expected}; run the frames step")
                if not vid.exists():
                    shutil.copyfile(vid_nfs, vid)
                if _n_jpg(jpg) != n_expected:
                    extract_frames(vid, jpg, overwrite=True, fps=fps, frame_format=data_cfg.frame_format)
                    if _n_jpg(jpg) != n_expected:
                        raise RuntimeError(f"re-extraction gave {_n_jpg(jpg)} JPGs")
                    idx = np.linspace(0, n_expected - 1, a.n_md5).round().astype(int)
                    bad = [int(i) for i in idx
                           if _md5(jpg / f"frame_{i:06d}.jpg") != _md5(fdir / f"frame_{i:06d}.jpg")]
                    log["reextract_md5_checked"], log["reextract_md5_mismatch"] = len(idx), bad
                    if bad:
                        raise RuntimeError(f"local re-extraction differs from frames/full at {bad[:5]}")
                tracker = build_tracker(tcfg)
                df = tracker.track_video(vid, background=get_background(v, obs))
                trk.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(str(trk) + ".tmp", index=False)
                os.replace(str(trk) + ".tmp", trk)
                df = pd.read_csv(trk)            # crops read the CSV, as get_pov_frames.py
                if len(df) != n_expected:
                    raise RuntimeError(f"{len(df)} tracking rows, expected {n_expected}")
                log["n_tracking_rows"] = len(df)
                crops = stage / "pov"
                for c in COLS:
                    (crops / c).mkdir(parents=True, exist_ok=True)
                for fi in range(n_expected):
                    name = f"frame_{fi:06d}.jpg"
                    img = cv2.imread(str(jpg / name))
                    if img is None:
                        raise RuntimeError(f"unreadable {name}")
                    r = df.iloc[fi]
                    for c, (cx, cy) in COLS.items():
                        cv2.imwrite(str(crops / c / name), AntTracker.crop_pov(img, float(r[cx]), float(r[cy]), radius))
                for c, d in pdirs.items():
                    if d.exists():
                        shutil.rmtree(d)
                    log[f"pov_{c}_copied"] = _copy_dir(crops / c, d)
                log["track_seconds"] = round(time.time() - t1, 1)
        log["seconds"] = round(time.time() - t0, 1)
        log["ok"] = True
    except Exception as e:
        log["ok"], log["error"] = False, repr(e)
        raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)
        logp.write_text(json.dumps(log, indent=1))
        print(json.dumps(log), flush=True)


if __name__ == "__main__":
    main()
