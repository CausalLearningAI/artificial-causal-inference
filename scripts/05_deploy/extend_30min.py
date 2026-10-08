#!/usr/bin/env python3
"""
Extend an ants version from its 10-min clips to 30 min, keeping minutes 0-10 bit-identical.

Option B: every artifact of the annotated window (frames 0..N_OLD-1 at 5 fps) is kept
from the 10-min build; only frames N_OLD.. are new. The 10-min artifacts that get
regenerated whole (videos, tracking CSVs, annotations.csv, HF dataset, POV embedding
caches, deployed annotation CSVs) are first moved to archive/10min/ants/<version>/,
mirroring their repo-relative paths (same filesystem, so a rename). frames/full and
frames/pov are extended in place: files of frames < N_OLD are never opened for writing.

Subcommands (run in order; the heavy ones inside Slurm jobs):
  archive            move the 10-min artifacts to the archive (refuses to overwrite)
  obs-list           valid observation ids (experiment.csv order) -> <work>/<version>_valid_obs.txt
  extend-obs --idx   one observation (array task), staged in /localhome:
                       standardize at the new end_frame (standardize.process_video), copy the
                       video to observations/full; extract JPGs (get_frames.extract_frames) and
                       copy frames >= N_OLD to frames/full; track the whole new video with the
                       cached background (get_tracking's tracker + calibration.get_background)
                       -> <work>/tracking_newrun/<obs>.csv; POV crops of frames >= N_OLD from the
                       new-run rows (identical to the spliced rows there) -> frames/pov; for
                       annotated observations also the new-pipeline crops of frames < N_OLD
                       -> <work>/check5_crops/<obs>.tar (never into the live dirs)
  verify-prefix      framemd5 of the new vs archived video and new-run vs archived tracking, frames < N_OLD
  splice-tracking    per observation: archived rows < N_OLD (text, verbatim) + new-run rows
                       >= N_OLD -> dataset/.../tracking/<obs>.csv; seam check at N_OLD
  splice-embeddings  rows with frame_idx < N_OLD of a freshly extracted POV cache are replaced
                       by the archived rows (drift reported first); rewrites .npy/.pt/dataset/

<work> = results/ppci/ants/regress_30min.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tarfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

WORK = ROOT / "results" / "ppci" / "ants" / "regress_30min"
ARCHIVE = ROOT / "archive" / "10min" / "ants"
IDS = ("blue", "yellow", "focal")


def archived_paths(v: str) -> list[str]:
    """Repo-relative paths of the 10-min artifacts that are rebuilt whole."""
    return [
        f"data/ants/{v}/observations/full",
        f"data/ants/{v}/observations/metadata.json",
        f"dataset/ants/{v}/tracking",
        f"dataset/ants/{v}/annotations.csv",
        f"dataset/ants/{v}/hf/full",
        f"dataset/ants/{v}/embeddings/pov/blue/dinov2/class",
        f"dataset/ants/{v}/embeddings/pov/yellow/dinov2/class",
        f"results/ppci/ants/annotations/{v}",
    ]


def _arch(v: str, rel: str) -> Path:
    return ARCHIVE / v / rel


def experiment(v: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "ants" / v / "experiment.csv")


def n_old_frames(row, ratio: float) -> int:
    """ML frames inside the annotated window: source frame start + i*ratio < annotation_end_frame."""
    i = np.arange(100_000)
    return int(((int(row["start_frame"]) + i * ratio).astype(int) < int(row["annotation_end_frame"])).sum())


def _fmt(s: float) -> str:
    return time.strftime("%H:%M:%S", time.gmtime(s))


# ── archive ──────────────────────────────────────────────────────────────────

def cmd_archive(a) -> None:
    rels = a.paths or archived_paths(a.version)
    for rel in rels:
        src, dst = ROOT / rel, _arch(a.version, rel)
        if not src.exists():
            print(f"  [missing] {rel}")
            continue
        if dst.exists():
            sys.exit(f"[ERROR] {dst} exists; not overwriting the archive")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if os.stat(src).st_dev != os.stat(dst.parent).st_dev:
            sys.exit(f"[ERROR] {rel}: archive is on another filesystem; refusing to copy")
        os.rename(src, dst)
        print(f"  moved {rel} -> {dst.relative_to(ROOT)}")


def cmd_obs_list(a) -> None:
    exp = experiment(a.version)
    obs = exp.loc[exp["valid"] == 1, "observation_id"].astype(str).tolist()
    out = WORK / f"{a.version}_valid_obs.txt"
    out.write_text("\n".join(obs) + "\n")
    print(f"{len(obs)} valid observations -> {out.relative_to(ROOT)}")


# ── extend one observation ───────────────────────────────────────────────────

def _local_stage(name: str) -> Path:
    """Node-local staging dir: /localhome/$USER when the node has it, else $TMPDIR or /tmp."""
    for base in (Path("/localhome") / os.environ["USER"], Path(os.environ.get("TMPDIR", "/tmp")) / os.environ["USER"]):
        try:
            base.mkdir(parents=True, exist_ok=True)
            return base / name
        except OSError:
            continue
    raise RuntimeError("no writable node-local staging directory")


def _check_prefix_dir(d: Path, n_old: int) -> int:
    """frames/<...>/<obs> must hold exactly frames 0..n_old-1 (plus, on a re-run, frames >= n_old)."""
    names = sorted(p.name for p in d.glob("frame_*.jpg"))
    idx = [int(n[6:12]) for n in names]
    pre = [i for i in idx if i < n_old]
    if pre != list(range(n_old)):
        raise RuntimeError(f"{d}: expected frames 0..{n_old - 1}, found {len(pre)} below {n_old}")
    return len(idx) - n_old


def _copy_frames(src_dir: Path, dst_dir: Path, lo: int, hi: int) -> int:
    for i in range(lo, hi):
        shutil.copyfile(src_dir / f"frame_{i:06d}.jpg", dst_dir / f"frame_{i:06d}.jpg")
    return hi - lo


def cmd_extend_obs(a) -> None:
    from src.data.standardize import process_video
    from src.dataset.get_frames import extract_frames
    from src.tracking.calibration import get_background
    from src.tracking.get_tracking import build_tracker
    from src.tracking.tracker import AntTracker

    v = a.version
    obs = (WORK / f"{v}_valid_obs.txt").read_text().split()[a.idx]
    exp = experiment(v).set_index("observation_id").loc[obs]
    data_cfg = OmegaConf.load(ROOT / "configs" / "data" / "ants.yaml")
    std_cfg = OmegaConf.create({"data": OmegaConf.to_container(data_cfg), "overwrite": {"videos": False}})
    tcfg = OmegaConf.load(ROOT / "configs" / "tracking" / "ants" / f"{v}.yaml")
    radius = int(tcfg.pov_radius)
    n_old = n_old_frames(exp, 30 / data_cfg.target_fps)
    n_new = int(exp["end_frame"] - exp["start_frame"]) // (30 // data_cfg.target_fps)

    stage = _local_stage(f"{os.environ.get('SLURM_JOB_ID', 'local')}_{a.idx}")
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    log = {"obs": obs, "n_old": n_old, "n_new_expected": n_new,
           "ffmpeg": os.popen("ffmpeg -version").readline().strip()}
    t0 = time.time()
    try:
        # 1. standardize (exact standardize.py call) -> observations/full
        stem = Path(exp["observation_file"]).stem
        src = stage / "source.mkv"
        shutil.copyfile(ROOT / "data" / "ants" / v / "observations" / "source" / exp["observation_file"], src)
        vid = stage / exp["observation_file"]
        if not process_video(src, vid, int(exp["start_frame"]), int(exp["end_frame"]), std_cfg):
            raise RuntimeError("standardize failed")
        out_vid = ROOT / "data" / "ants" / v / "observations" / "full" / exp["observation_file"]
        out_vid.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(vid, str(out_vid) + ".tmp")
        os.replace(str(out_vid) + ".tmp", out_vid)
        src.unlink()

        # 2. JPGs (exact get_frames call); copy frames >= n_old only
        jpg = stage / "frames"
        extract_frames(vid, jpg, overwrite=True, fps=data_cfg.target_fps, frame_format=data_cfg.frame_format)
        n_jpg = len(list(jpg.glob("frame_*.jpg")))
        log["n_jpg"] = n_jpg
        if n_jpg != n_new:
            raise RuntimeError(f"{n_jpg} JPGs, expected {n_new}")
        fdir = ROOT / "dataset" / "ants" / v / "frames" / "full" / stem
        log["frames_full_preexisting_new"] = _check_prefix_dir(fdir, n_old)
        log["frames_full_copied"] = _copy_frames(jpg, fdir, n_old, n_jpg)

        # 3. tracking of the whole new video, cached background (as get_tracking.py)
        tracker = build_tracker(tcfg)
        df = tracker.track_video(vid, background=get_background(v, obs))
        trk = WORK / "tracking_newrun" / f"{obs}.csv"
        trk.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(trk, index=False)
        df = pd.read_csv(trk)                      # crops read the CSV, as get_pov_frames.py
        log["n_tracking_rows"] = len(df)
        if len(df) != n_jpg:
            raise RuntimeError(f"{len(df)} tracking rows, {n_jpg} JPGs")

        # 4. POV crops (AntTracker.crop_pov on the JPG, mark positions, as get_pov_frames.py)
        cols = {"blue": ("mark_blue_x", "mark_blue_y"), "yellow": ("mark_yellow_x", "mark_yellow_y")}
        annotated = isinstance(exp["annotation_file"], str) and exp["annotation_file"] != ""
        crops = stage / "pov"
        for c in cols:
            (crops / c).mkdir(parents=True)
        lo = 0 if annotated else n_old
        for fi in range(lo, n_jpg):
            img = cv2.imread(str(jpg / f"frame_{fi:06d}.jpg"))
            row = df.iloc[fi]
            for c, (cx, cy) in cols.items():
                cv2.imwrite(str(crops / c / f"frame_{fi:06d}.jpg"),
                            AntTracker.crop_pov(img, float(row[cx]), float(row[cy]), radius))
        for c in cols:
            pdir = ROOT / "dataset" / "ants" / v / "frames" / "pov" / c / stem
            log[f"pov_{c}_preexisting_new"] = _check_prefix_dir(pdir, n_old)
            log[f"pov_{c}_copied"] = _copy_frames(crops / c, pdir, n_old, n_jpg)
        if annotated:
            tar = WORK / "check5_crops" / f"{obs}.tar"
            tar.parent.mkdir(parents=True, exist_ok=True)
            with tarfile.open(str(tar) + ".tmp", "w") as tf:
                for c in cols:
                    for fi in range(n_old):
                        tf.add(crops / c / f"frame_{fi:06d}.jpg", arcname=f"{c}/frame_{fi:06d}.jpg")
            os.replace(str(tar) + ".tmp", tar)
            log["check5_crops"] = 2 * n_old
        log["seconds"] = round(time.time() - t0, 1)
        log["ok"] = True
    finally:
        shutil.rmtree(stage, ignore_errors=True)
        (WORK / "extend_log").mkdir(exist_ok=True)
        (WORK / "extend_log" / f"{obs}.json").write_text(json.dumps(log, indent=1))
    print(json.dumps(log))


# ── tracking splice + seam check ─────────────────────────────────────────────

def _xy(df: pd.DataFrame, kind: str, ident: str) -> np.ndarray:
    pre = "mark_" if kind == "mark" and ident != "focal" else ""
    return df[[f"{pre}{ident}_x", f"{pre}{ident}_y"]].to_numpy(dtype=float)


def seam_row(old: pd.DataFrame, new: pd.DataFrame, n: int, margin: float) -> dict:
    """Agreement of the two runs at the seam.

    dist_<kind>_<id>: |old[n-1] - new[n-1]| (the new run's state just before the splice).
    swap_<a>_<b>: assigning new[n] to old[n-1] with a and b exchanged is cheaper than
    the identity-preserving assignment by more than `margin` px.
    """
    r = {}
    for kind, ids in (("centroid", IDS), ("mark", ("blue", "yellow"))):
        for i in ids:
            r[f"dist_{kind}_{i}"] = float(np.linalg.norm(_xy(old, kind, i)[n - 1] - _xy(new, kind, i)[n - 1]))
            r[f"jump_{kind}_{i}"] = float(np.linalg.norm(_xy(new, kind, i)[n] - _xy(old, kind, i)[n - 1]))
        for x in range(len(ids)):
            for y in range(x + 1, len(ids)):
                a, b = ids[x], ids[y]
                oa, ob = _xy(old, kind, a)[n - 1], _xy(old, kind, b)[n - 1]
                na, nb = _xy(new, kind, a)[n], _xy(new, kind, b)[n]
                same = np.linalg.norm(na - oa) + np.linalg.norm(nb - ob)
                swapped = np.linalg.norm(na - ob) + np.linalg.norm(nb - oa)
                r[f"swap_{kind}_{a}_{b}"] = bool(swapped + margin < same)
    return r


def cmd_splice_tracking(a) -> None:
    v = a.version
    exp = experiment(v).set_index("observation_id")
    obs_list = (WORK / f"{v}_valid_obs.txt").read_text().split()
    out_dir = ROOT / "dataset" / "ants" / v / "tracking"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, flagged = [], []
    for obs in obs_list:
        n = n_old_frames(exp.loc[obs], 6.0)
        old_p = _arch(v, f"dataset/ants/{v}/tracking") / f"{obs}.csv"
        new_p = WORK / "tracking_newrun" / f"{obs}.csv"
        old_l = old_p.read_text().split("\n")
        new_l = new_p.read_text().split("\n")
        if old_l[0] != new_l[0]:
            raise RuntimeError(f"{obs}: tracking header differs")
        if old_l[-1] != "" or new_l[-1] != "" or len(old_l) - 2 != n:
            raise RuntimeError(f"{obs}: archived CSV has {len(old_l) - 2} rows, expected {n}")
        old, new = pd.read_csv(old_p), pd.read_csv(new_p)
        r = {"obs": obs, "n_new_rows": len(new)} | seam_row(old, new, n, a.swap_margin)
        dmax = max(v_ for k, v_ in r.items() if k.startswith("dist_"))
        r["dist_max"] = dmax
        r["flag"] = bool(dmax > a.flag_px or any(v_ for k, v_ in r.items() if k.startswith("swap_")))
        rows.append(r)
        if r["flag"]:
            flagged.append(obs)
        if a.dry_run:
            continue
        spliced = "\n".join(old_l[: n + 1] + new_l[n + 1:])
        tmp = out_dir / f"{obs}.csv.tmp"
        tmp.write_text(spliced)
        chk = tmp.read_text().split("\n")
        assert chk[: n + 1] == old_l[: n + 1] and chk[n + 1:] == new_l[n + 1:]
        os.replace(tmp, out_dir / f"{obs}.csv")
    df = pd.DataFrame(rows)
    df.to_csv(WORK / "seam_check.csv", index=False)
    summ = {"n_videos": len(df), "n_flagged": len(flagged), "flagged": flagged,
            "flag_px": a.flag_px, "swap_margin_px": a.swap_margin,
            "dist_max_quantiles": {q: float(df.dist_max.quantile(q)) for q in (0.5, 0.9, 0.99, 1.0)},
            "n_any_swap": int(df[[c for c in df if c.startswith("swap_")]].any(axis=1).sum()),
            "spliced": not a.dry_run}
    (WORK / "seam_check.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


# ── prefix verification ──────────────────────────────────────────────────────

def _framemd5(video: Path, n: int) -> list[str]:
    import subprocess
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-map", "0:v", "-frames:v", str(n),
                          "-f", "framemd5", "-"], capture_output=True, text=True, check=True).stdout
    return [ln.split(",")[-1].strip() for ln in out.splitlines() if ln and not ln.startswith("#")]


def cmd_verify_prefix(a) -> None:
    """New 30-min video vs archived 10-min video (framemd5 of frames < N_OLD), and the new
    tracking run vs the archived tracking rows < N_OLD (exact-row agreement, max distance)."""
    v = a.version
    exp = experiment(v).set_index("observation_id")
    rows = []
    for obs in (WORK / f"{v}_valid_obs.txt").read_text().split():
        e = exp.loc[obs]
        n = n_old_frames(e, 6.0)
        f = e["observation_file"]
        new_md5 = _framemd5(ROOT / "data" / "ants" / v / "observations" / "full" / f, n)
        old_md5 = _framemd5(_arch(v, f"data/ants/{v}/observations/full") / f, n)
        old = pd.read_csv(_arch(v, f"dataset/ants/{v}/tracking") / f"{obs}.csv")
        new = pd.read_csv(WORK / "tracking_newrun" / f"{obs}.csv").iloc[:n]
        cols = [c for c in old.columns if c != "frame_idx"]
        same = ((old[cols] == new[cols]) | (old[cols].isna() & new[cols].isna())).all(axis=1)
        d = max(float(np.linalg.norm(_xy(old, k, i) - _xy(new, k, i), axis=1).max())
                for k, ids in (("centroid", IDS), ("mark", ("blue", "yellow"))) for i in ids)
        rows.append({"obs": obs, "framemd5_equal": sum(x == y for x, y in zip(old_md5, new_md5)),
                     "framemd5_n": min(len(old_md5), len(new_md5)), "tracking_rows_identical": int(same.sum()),
                     "tracking_rows": n, "tracking_max_dist": d})
    df = pd.DataFrame(rows)
    df.to_csv(WORK / "prefix_verify.csv", index=False)
    summ = {"n_videos": len(df),
            "videos_framemd5_all_equal": int((df.framemd5_equal == df.framemd5_n).sum()),
            "frames_framemd5_equal": int(df.framemd5_equal.sum()), "frames_compared": int(df.framemd5_n.sum()),
            "videos_tracking_prefix_identical": int((df.tracking_rows_identical == df.tracking_rows).sum()),
            "tracking_rows_identical": int(df.tracking_rows_identical.sum()),
            "tracking_rows": int(df.tracking_rows.sum()),
            "tracking_max_dist_max": float(df.tracking_max_dist.max())}
    (WORK / "prefix_verify.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


def cmd_verify_annotations(a) -> None:
    """New annotations.csv: in-window rows must equal the archived table (every archived column),
    rows past the window NaN outcomes and eval_window False."""
    v = a.version
    new = pd.read_csv(ROOT / "dataset" / "ants" / v / "annotations.csv")
    old = pd.read_csv(_arch(v, f"dataset/ants/{v}/annotations.csv"))
    ycols = [c for c in new if c.startswith("Y_")]
    win = new[new["eval_window"]].reset_index(drop=True)
    out = new[~new["eval_window"]]
    same = win[old.columns].equals(old)
    if not same:
        diff = (win[old.columns] != old) & ~(win[old.columns].isna() & old.isna())
        print("differing cells per column:", diff.sum()[diff.sum() > 0].to_dict())
    rep = {"rows": len(new), "obs": int(new.observation_id.nunique()),
           "frames_per_obs": sorted(new.groupby("observation_id").size().unique().tolist()),
           "eval_window_rows": len(win), "window_equals_archived": bool(same),
           "past_window_rows": len(out), "past_window_Y_all_nan": bool(out[ycols].isna().all().all()),
           "past_window_min_frame": int(out.frame_idx.min()) if len(out) else None,
           "annotated_rows": int(new[ycols].notna().all(axis=1).sum())}
    (WORK / "annotations_verify.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    if not same:
        raise SystemExit("in-window rows differ from the archived annotations.csv")


# ── embedding splice ─────────────────────────────────────────────────────────

def cmd_splice_embeddings(a) -> None:
    import torch
    from datasets import Dataset

    v, ident = a.version, a.identity
    rel = f"dataset/ants/{v}/embeddings/pov/{ident}/dinov2/class"
    live, arch = ROOT / rel, _arch(v, rel)
    old = torch.load(arch / "embeddings.pt", weights_only=True)
    new = torch.load(live / "embeddings.pt", weights_only=True)
    old_keys = pd.read_csv(_arch(v, f"dataset/ants/{v}/annotations.csv"), usecols=["observation_id", "frame_idx"])
    new_keys = pd.read_parquet(live / "row_keys.parquet")
    if len(old_keys) != len(old) or len(new_keys) != len(new):
        raise RuntimeError("key tables do not match the caches")
    pos = new_keys.reset_index().rename(columns={"index": "row"})
    m = old_keys.merge(pos, on=["observation_id", "frame_idx"], how="left", validate="one_to_one")
    if m["row"].isna().any():
        raise RuntimeError(f"{int(m['row'].isna().sum())} archived rows missing from the new cache")
    rows = torch.from_numpy(m["row"].to_numpy(dtype=np.int64))
    ann = pd.read_csv(ROOT / "dataset" / "ants" / v / "annotations.csv", usecols=["eval_window"])["eval_window"]
    if not np.array_equal(np.sort(rows.numpy()), np.flatnonzero(ann.to_numpy(bool))):
        raise RuntimeError("archived rows are not exactly the eval_window rows of the new table")

    d = (new[rows] - old).abs()
    cos = torch.nn.functional.cosine_similarity(new[rows], old, dim=1)
    rep = {"identity": ident, "n_new": len(new), "n_prefix": len(rows),
           "prefix_max_abs": float(d.max()), "prefix_mean_abs": float(d.mean()),
           "prefix_rows_bit_identical": int((d.amax(dim=1) == 0).sum()),
           "prefix_cos_min": float(cos.min()), "prefix_cos_mean": float(cos.mean()),
           "old_abs_mean": float(old.abs().mean())}
    print(json.dumps(rep))
    # re-embedded old crops of the annotated observations, kept for check 5
    ctl = WORK / "check5" / f"reembed_old_crops_{ident}.pt"
    ctl.parent.mkdir(parents=True, exist_ok=True)
    keep = m["observation_id"].isin(a.control_obs or []).to_numpy()
    torch.save({"keys": m.loc[keep, ["observation_id", "frame_idx"]].reset_index(drop=True),
                "emb": new[rows[torch.from_numpy(keep)]].clone()}, ctl)

    if not a.dry_run:
        new[rows] = old
        n, dim = new.shape
        arr = new.numpy()
        mm = np.memmap(live / "embeddings.npy.tmp", dtype="float32", mode="w+", shape=(n, dim))
        mm[:] = arr
        mm.flush()
        del mm
        torch.save(new, live / "embeddings.pt.tmp")
        col = "embedding_dinov2_class"

        def _gen():
            for s in range(0, n, 50_000):
                for r_ in arr[s:s + 50_000]:
                    yield {col: r_}
        ds = Dataset.from_generator(_gen)
        ds.set_format(type="torch", columns=[col])
        shutil.rmtree(live / "dataset.tmp", ignore_errors=True)
        ds.save_to_disk(str(live / "dataset.tmp"))
        os.replace(live / "embeddings.npy.tmp", live / "embeddings.npy")
        os.replace(live / "embeddings.pt.tmp", live / "embeddings.pt")
        shutil.rmtree(live / "dataset")
        os.replace(live / "dataset.tmp", live / "dataset")
        chk = torch.load(live / "embeddings.pt", weights_only=True)
        assert torch.equal(chk[rows], old)
        rep["spliced"] = True
    (WORK / f"embedding_splice_{ident}.json").write_text(json.dumps(rep, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--version", default="v5")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("archive")
    s.add_argument("paths", nargs="*", help="Repo-relative paths (default: archived_paths)")
    sub.add_parser("obs-list")
    s = sub.add_parser("extend-obs")
    s.add_argument("--idx", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    sub.add_parser("verify-prefix")
    sub.add_parser("verify-annotations")
    s = sub.add_parser("splice-tracking")
    s.add_argument("--flag-px", type=float, default=20.0)
    s.add_argument("--swap-margin", type=float, default=5.0)
    s.add_argument("--dry-run", action="store_true")
    s = sub.add_parser("splice-embeddings")
    s.add_argument("--identity", required=True, choices=["blue", "yellow"])
    s.add_argument("--control-obs", nargs="*")
    s.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    {"archive": cmd_archive, "obs-list": cmd_obs_list, "extend-obs": cmd_extend_obs,
     "verify-prefix": cmd_verify_prefix, "verify-annotations": cmd_verify_annotations, "splice-tracking": cmd_splice_tracking, "splice-embeddings": cmd_splice_embeddings}[a.cmd](a)


if __name__ == "__main__":
    main()
