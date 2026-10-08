#!/usr/bin/env python3
"""
Check 5 of the 30-min extension: does the new processing cost accuracy on labelled frames?

For the annotated observations, minutes 0-10 were re-processed by the new pipeline
(extend_30min.py extend-obs: new video, new JPGs, new tracking run, new POV crops ->
<work>/check5_crops/<obs>.tar). This job embeds those crops exactly like
get_embeddings.py (DINOv2 class token, fp16, batch 192), recomputes the tracking
distances from the new run, and evaluates the deployed model on the windowed v5 dataset
with the annotated observations' rows replaced:

  A  deployed inputs (archived embeddings + archived tracking)      -> must equal metrics.json v5
  B  old crops re-embedded by the 30-min extraction (embedding re-run only)
  C  new pipeline: new crops' embeddings + new-run distances
  C_emb   new embeddings, archived distances
  C_dist  archived embeddings, new-run distances

Embeddings are saved under <work>/check5/ (not the live cache).
When the new pipeline reproduces the 10-min frames and tracking bit for bit (ffmpeg 6.1.1 path),
the new crops equal the archived ones: --crops-identical checks that on every --crop-step-th frame
of the tars against the live crops and then evaluates only A and B (B = the only difference).
Usage (GPU job): python scripts/05_deploy/check5_reprocess.py [--crops-identical]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from ppci.dataset import PPCIDataset                     # noqa: E402
from ppci.generate_annotations import load_model         # noqa: E402
from ppci.hparam_search import DS_KWARGS                 # noqa: E402
from ppci.train import compute_metrics                   # noqa: E402

WORK = ROOT / "results" / "ppci" / "ants" / "regress_30min"
DEPLOY = ROOT / "results" / "ppci" / "ants" / "performances" / "final"
N_OLD, D = 3000, 768


def embed(stage: Path, obs_list: list[str], ident: str) -> torch.Tensor:
    from datasets import Dataset, Image as HFImage
    from src.embedding.get_embeddings import EmbeddingExtractor
    paths = [str(stage / o / ident / f"frame_{i:06d}.jpg") for o in obs_list for i in range(N_OLD)]
    ds = Dataset.from_dict({"image": paths}).cast_column("image", HFImage())
    ex = EmbeddingExtractor(encoder="dinov2", device="cuda", batch_size=192, num_workers=8, token="class")
    mm = ex.extract_batch_to_file(ds, stage / f"emb_{ident}.npy", num_samples=len(ds))
    return torch.from_numpy(np.array(mm, dtype=np.float32))


def dists(csv: Path) -> np.ndarray:
    """[B2F, Y2F, B2Y] of rows < N_OLD, as ppci.dataset._load_tracking_distances."""
    t = pd.read_csv(csv).iloc[:N_OLD]
    def d(a, b):
        x = np.sqrt((t[f"{a}_x"] - t[f"{b}_x"]) ** 2 + (t[f"{a}_y"] - t[f"{b}_y"]) ** 2).to_numpy(np.float64)
        return np.nan_to_num(x, nan=0.0).astype(np.float32)
    return np.stack([d("blue", "focal"), d("yellow", "focal"), d("blue", "yellow")], axis=1)


def crops_identical(obs_list: list[str], step: int) -> dict:
    """md5 of tar members (new-pipeline crops) vs the live crops (archived 10-min crops)."""
    n = same = 0
    for o in obs_list:
        with tarfile.open(WORK / "check5_crops" / f"{o}.tar") as tf:
            for ident in ("blue", "yellow"):
                for i in range(0, N_OLD, step):
                    a = tf.extractfile(f"{ident}/frame_{i:06d}.jpg").read()
                    b = (ROOT / "dataset" / "ants" / "v5" / "frames" / "pov" / ident / o / f"frame_{i:06d}.jpg").read_bytes()
                    n += 1
                    same += hashlib.md5(a).digest() == hashlib.md5(b).digest()
    return {"crops_compared": n, "crops_md5_equal": same}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops-identical", action="store_true")
    ap.add_argument("--crop-step", type=int, default=10)
    args = ap.parse_args()
    stage = Path(os.environ.get("STAGE_DIR", f"/tmp/{os.environ['USER']}/check5"))
    stage.mkdir(parents=True, exist_ok=True)
    out = WORK / "check5"
    out.mkdir(exist_ok=True)
    obs_list = sorted(p.stem for p in (WORK / "check5_crops").glob("*.tar"))
    crop_check = None
    if args.crops_identical:
        crop_check = crops_identical(obs_list, args.crop_step)
        print(json.dumps(crop_check), flush=True)
        if crop_check["crops_md5_equal"] != crop_check["crops_compared"]:
            raise SystemExit("new-pipeline crops differ from the live crops: run without --crops-identical")
    else:
        for o in obs_list:
            with tarfile.open(WORK / "check5_crops" / f"{o}.tar") as tf:
                tf.extractall(stage / o)
    new_emb = {}
    for ident in ("blue", "yellow") if not args.crops_identical else ():
        f = out / f"new_pipeline_{ident}.pt"
        if f.exists():
            new_emb[ident] = torch.load(f, weights_only=True)
        else:
            new_emb[ident] = embed(stage, obs_list, ident)
            torch.save(new_emb[ident], f)
    for ident, e in new_emb.items():                         # as from_disk: NaN rows -> 0
        nan = torch.isnan(e).any(dim=1)
        print(f"new-pipeline {ident}: {int(nan.sum())} NaN rows", flush=True)
        e[nan] = 0.0
    keys = pd.DataFrame({"observation_id": np.repeat(obs_list, N_OLD), "frame_idx": np.tile(np.arange(N_OLD), len(obs_list))})
    new_d = np.concatenate([dists(WORK / "tracking_newrun" / f"{o}.csv") for o in obs_list])

    device = torch.device("cuda")
    model, cfg = load_model(DEPLOY, device)
    k, mode = int(cfg["context_window"]), cfg["context_mode"]
    ds = PPCIDataset.from_disk("ants", "v5", cfg["encoder"], cfg["token"], frame_type="pov",
                               dist_mode=cfg["dist_mode"], n_val_videos=0, **DS_KWARGS)
    X0 = ds.X.clone()
    pos = pd.DataFrame({"observation_id": ds.obs_ids, "frame_idx": ds.frame_idx.numpy(), "row": np.arange(len(ds))})
    rows = torch.from_numpy(keys.merge(pos, how="left", on=["observation_id", "frame_idx"])["row"].to_numpy(np.int64))

    ctl = {i: torch.load(WORK / "check5" / f"reembed_old_crops_{i}.pt", weights_only=False) for i in ("blue", "yellow")}
    ctl_rows = {}
    for i, c in ctl.items():
        m = c["keys"].reset_index().merge(keys.reset_index().rename(columns={"index": "k"}),
                                          on=["observation_id", "frame_idx"])
        e = torch.empty(len(keys), D)
        e[torch.from_numpy(m["k"].to_numpy())] = c["emb"][torch.from_numpy(m["index"].to_numpy())]
        assert len(m) == len(keys)
        ctl_rows[i] = e

    nd = torch.from_numpy(new_d)
    blue_d, yellow_d = nd[:, [0, 2]], nd[:, [1, 2]]          # [B2F, B2Y], [Y2F, B2Y]

    def variant(blue=None, yellow=None, dist=False):
        X = X0.clone()
        sub = X[rows]
        if blue is not None:
            sub[:, :D] = blue
        if yellow is not None:
            sub[:, D + 2:2 * D + 2] = yellow
        if dist:
            sub[:, D:D + 2] = blue_d
            sub[:, 2 * D + 2:] = yellow_d
        X[rows] = sub
        return X

    variants = {
        "A_deployed": variant(),
        "B_reembed_old_crops": variant(ctl_rows["blue"], ctl_rows["yellow"]),
    }
    if new_emb:
        variants |= {
            "C_new_pipeline": variant(new_emb["blue"], new_emb["yellow"], dist=True),
            "C_emb_only": variant(new_emb["blue"], new_emb["yellow"]),
        }
    variants["C_dist_only"] = variant(dist=True)
    res, probs = {}, {}
    ann = ~torch.isnan(ds.Y).any(dim=1)
    for name, X in variants.items():
        ds.X, ds._context_window_applied = X, 0
        ds.apply_context_window(k, mode=mode)
        m = compute_metrics(model, ds.X, ds.Y, device)
        model.eval()
        with torch.no_grad():
            probs[name] = torch.cat([model.probs(ds.X[ann][s:s + 4096].to(device)).cpu()
                                     for s in range(0, int(ann.sum()), 4096)])
        res[name] = {k_: float(m[k_]) for k_ in ("acc", "bacc", "recall", "precision")}
        res[name]["input_rows_changed"] = int((X != X0).any(dim=1).sum())
        print(name, json.dumps(res[name]), flush=True)
    base = probs["A_deployed"]
    for name, p in probs.items():
        res[name]["mean_abs_prob_diff_vs_A"] = float((p - base).abs().mean())
        res[name]["binary_agreement_vs_A"] = float(((p >= 0.5) == (base >= 0.5)).float().mean())
        res[name]["positive_rate"] = [float(x) for x in (p >= 0.5).float().mean(dim=0)]
    with open(DEPLOY / "metrics.json") as f:
        res["deployed_metrics_v5"] = json.load(f)["v5"]
    res["n_obs"], res["n_annotated_frames"] = len(obs_list), int(ann.sum())
    res["crop_check"] = crop_check
    (out / "check5.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
