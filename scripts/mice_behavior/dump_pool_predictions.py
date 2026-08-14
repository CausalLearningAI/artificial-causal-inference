#!/usr/bin/env python3
"""Dump per-pool mean predicted behaviour rate for ALL 72 mice v1 pools, for PPI++.

This is the missing input to `ppi_report.py`. PPI needs f evaluated on the 48 UNANNOTATED
pools (which no existing script touches -- every eval path enumerates observations from
`pair_labels.parquet`, which only covers the 144 annotated ones) and, on the 24 annotated
pools, predictions that are OUT-OF-FOLD.

Cross-fitting is not optional. An in-sample f shrinks the rectifier (Y - lam*f) toward
zero, which understates the correction and produces an overconfident interval -- the exact
failure PPI exists to prevent. So a labeled pool's prediction is only ever taken from a run
that held that pool out (`val_pools` in the run's config.json). Pass several runs whose
val_pools tile the 24 annotated pools:

    python scripts/mice_behavior/dump_pool_predictions.py \
        --runs fold1 fold2 fold3 fold4 fold5 fold6 \
        --frame-stride 10 --out results/vision/mice/frame/_figures/pool_preds.csv

Any annotated pool not held out by SOME run is reported and dropped, rather than silently
contributing a contaminated prediction. Unannotated pools were never trained on by any run,
so their predictions are averaged over all runs.

MEMORY: at 448px a cached frame is 1024 patches x 768 dims x fp16 = 1.5 MB, so the whole
dataset cannot be held the way `eval_downstream_obs.py` holds a 4-pool val split. This
script streams ONE OBSERVATION AT A TIME: encode that observation's needed frames, run the
head, accumulate the mean, free. Peak cache is ~6.6 GiB of CPU RAM (the longest
observation: 9000 frames -> 4498 unique context frames at --frame-stride 10).

COST: ~1.3M encoder forwards at --frame-stride 10 over all 432 observations, per run. Only
correctness was verified here (on a GTX 1080 Ti); the full pass has NOT been benchmarked at
scale, so time it on a couple of observations with --limit-obs before committing a SLURM
walltime. Rate estimates are means over hundreds of frames per observation, so striding
costs very little precision relative to the between-pool spread that actually limits the
contrast -- raise --frame-stride if the full pass is too slow.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from src.mice_behavior.model import MouseFrameClassifier      # noqa: E402
from src.mice_behavior.head_cfg import get_head_cfg           # noqa: E402
from src.mice_behavior.pools import load_obs_to_pool_map      # noqa: E402
from src.dataset.get_dataset import load_dataset              # noqa: E402
from train_patchgrid_online import _ImageDataset              # noqa: E402

MODEL_ID = "facebook/dinov2-base"
EMB_DIM = 768
PATCH_SIZE = 14
FRAME_DIR = ROOT / "results" / "vision" / "mice" / "frame"


# -- observation index ranges ------------------------------------------------

def observation_bounds() -> dict:
    """{observation_id: (global_start, global_end)} in annotations.csv row order.

    Row order IS the global index the encoding path uses, and the file covers all 432
    observations (the 288 unannotated ones carry NaN labels, which is irrelevant here --
    we only need their frames).
    """
    ann = pd.read_csv(ROOT / "dataset" / "mice" / "v1" / "annotations.csv",
                      usecols=["observation_id"])
    df = pd.DataFrame({"oid": ann.observation_id, "i": np.arange(len(ann))})
    return {oid: (int(g.i.iloc[0]), int(g.i.iloc[-1]) + 1)
            for oid, g in df.groupby("oid", sort=False)}


def build_anchors(start: int, end: int, frame_stride: int, context_k: int, stride: int):
    """Anchor global indices plus their context grid, clipped to the observation.

    Context positions outside the observation are clamped and flagged in `pad`, matching
    FrameBatchData's convention -- context must never cross an observation boundary, or a
    bout from a different video leaks into the window.
    """
    anchors = np.arange(start, end, frame_stride, dtype=np.int64)
    offsets = np.arange(-context_k, context_k + 1, dtype=np.int64) * stride
    grid = anchors[:, None] + offsets[None, :]
    pad = (grid < start) | (grid >= end)
    grid = np.clip(grid, start, end - 1)
    return anchors, grid, pad, offsets


# -- run loading -------------------------------------------------------------

def load_run(name: str, device):
    run_dir = FRAME_DIR / name
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"no config.json in {run_dir}")
    cfg = json.loads(cfg_path.read_text())
    head_cfg = dict(get_head_cfg())
    head_cfg.update(cfg.get("cfg", {}) or {})

    input_size = int(cfg.get("input_size", 224))
    n_patches = (input_size // PATCH_SIZE) ** 2

    from transformers import AutoImageProcessor, AutoModel
    processor = AutoImageProcessor.from_pretrained(MODEL_ID, use_fast=True)
    encoder = AutoModel.from_pretrained(MODEL_ID)
    enc_path = run_dir / "best_encoder.pt"
    if enc_path.exists():
        # Since 80aa0fe the checkpoint holds ONLY the unfrozen blocks (e.g. layers 10-11 +
        # final layernorm), not the whole backbone -- so overlay it on the pretrained
        # weights with strict=False rather than replacing the state dict outright.
        partial = torch.load(enc_path, map_location="cpu", weights_only=True)
        missing, unexpected = encoder.load_state_dict(partial, strict=False)
        if unexpected:
            raise RuntimeError(f"[{name}] unexpected keys in best_encoder.pt: {unexpected[:5]}")
        tuned = sorted({k.split(".")[2] for k in partial if k.startswith("encoder.layer.")})
        print(f"  [{name}] fine-tuned encoder: overlaid {len(partial)} tensors "
              f"(blocks {','.join(tuned)}) onto pretrained {MODEL_ID}")
    encoder = encoder.to(device).eval().requires_grad_(False)

    model = MouseFrameClassifier(
        emb_dim=EMB_DIM, n_heads=head_cfg["n_heads"], hidden_dim=head_cfg["hidden_dim"],
        use_patch_grid=True, dropout=head_cfg.get("dropout", 0.4),
        cross_attn_dim=cfg.get("cross_attn_dim"), patch_pool_dim=cfg.get("patch_pool_dim"),
    ).to(device)
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device,
                                     weights_only=True))
    model.eval()

    return {
        "name": name, "cfg": cfg, "encoder": encoder, "processor": processor,
        "model": model, "input_size": input_size, "n_patches": n_patches,
        "context_k": int(cfg.get("context_k", 2)), "stride": int(cfg.get("stride", 1)),
        "val_pools": set(cfg.get("val_pools", [])),
    }


# -- inference ---------------------------------------------------------------

@torch.no_grad()
def predict_observation(run, hf_dataset, bounds, oid, frame_stride, device,
                        batch_size=64, workers=12):
    """Mean predicted [nt, nn] probability over one observation's anchor frames."""
    start, end = bounds[oid]
    anchors, grid, pad, offsets = build_anchors(start, end, frame_stride,
                                                run["context_k"], run["stride"])
    need = np.unique(grid)
    loader = DataLoader(
        _ImageDataset(hf_dataset, need, run["processor"], input_size=run["input_size"]),
        batch_size=96, num_workers=workers, pin_memory=(device.type == "cuda"),
        shuffle=False, prefetch_factor=4,
    )
    cache = torch.empty((len(need), run["n_patches"], EMB_DIM), dtype=torch.float16)
    cur = 0
    for pixel_values in loader:
        pixel_values = pixel_values.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=device.type == "cuda"):
            out = run["encoder"](pixel_values=pixel_values)
        tok = out.last_hidden_state[:, 1:].half().cpu()
        cache[cur:cur + tok.shape[0]] = tok
        cur += tok.shape[0]

    probs = []
    for b0 in range(0, len(anchors), batch_size):
        sl = slice(b0, min(b0 + batch_size, len(anchors)))
        abs_idx, mask = grid[sl], pad[sl]
        B, T = abs_idx.shape
        pos = np.searchsorted(need, abs_idx)
        ctx = cache[torch.from_numpy(pos.ravel())].reshape(B, T, run["n_patches"], EMB_DIM)
        ctx = ctx.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=device.type == "cuda"):
            logits = run["model"](
                ctx.float(),
                offsets=torch.from_numpy(np.broadcast_to(offsets, (B, T)).copy()).to(device),
                key_padding_mask=torch.from_numpy(mask).to(device),
            )
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
    del cache
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(probs).mean(axis=0)


# -- main --------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True,
                    help="run directory names under results/vision/mice/frame/")
    ap.add_argument("--frame-stride", type=int, default=10,
                    help="use every Nth frame as an anchor (default 10)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit-obs", type=int, default=None, help="smoke-test on N observations")
    ap.add_argument("--out", type=Path, default=FRAME_DIR / "_figures" / "pool_preds.csv")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp = pd.read_csv(ROOT / "data" / "mice" / "v1" / "experiment.csv")
    exp["labeled"] = exp.annotator.notna()
    obs_to_pool = load_obs_to_pool_map(str(ROOT / "data"))
    pool_geno = dict(zip(exp.pool, exp.genotype))
    labeled_pools = set(exp.loc[exp.labeled, "pool"])

    bounds = observation_bounds()
    obs_ids = [o for o in exp.observation_id if o in bounds]
    if args.limit_obs:
        obs_ids = obs_ids[:args.limit_obs]

    hf_dataset = load_dataset(subject="mice", version="v1",
                              dataset_root=str(ROOT / "dataset"), frame_type="full")

    # Which run may predict which pool: labeled pools only from a run that held them out.
    runs = [load_run(n, device) for n in args.runs]
    held_out = {}
    for r in runs:
        for p in r["val_pools"]:
            held_out.setdefault(p, []).append(r["name"])
    missing = sorted(labeled_pools - set(held_out))
    if missing:
        print(f"\n[WARNING] {len(missing)} annotated pool(s) held out by NO run and therefore "
              f"dropped from the labeled set (in-sample predictions are not usable for PPI):"
              f"\n  {missing}\n  -> supply k-fold runs whose val_pools tile all 24 pools.\n")

    rows, t0 = [], time.time()
    for i, oid in enumerate(obs_ids):
        pool = obs_to_pool[oid]
        is_lab = pool in labeled_pools
        usable = [r for r in runs if (not is_lab) or (pool in r["val_pools"])]
        if not usable:
            continue
        preds = [predict_observation(r, hf_dataset, bounds, oid, args.frame_stride,
                                     device, args.batch_size, args.workers)
                 for r in usable]
        p = np.mean(preds, axis=0)
        rows.append({"observation_id": oid, "pool": pool, "genotype": pool_geno[pool],
                     "labeled": is_lab, "f_nt": float(p[0]), "f_nn": float(p[1]),
                     "source_runs": "|".join(r["name"] for r in usable)})
        if (i + 1) % 10 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(obs_ids)} obs  {el/60:.1f} min elapsed  "
                  f"~{el/(i+1)*(len(obs_ids)-i-1)/60:.0f} min left", flush=True)

    obs_df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    obs_path = args.out.with_name(args.out.stem + "_obs.csv")
    obs_df.to_csv(obs_path, index=False)

    pool_df = (obs_df.groupby(["pool", "genotype", "labeled"])[["f_nt", "f_nn"]]
               .mean().reset_index())
    pool_df.to_csv(args.out, index=False)
    print(f"\n  {len(obs_df)} observations -> {len(pool_df)} pools")
    print(f"  labeled pools: {int(pool_df.labeled.sum())}   "
          f"unlabeled: {int((~pool_df.labeled).sum())}")
    print(f"  per-observation: {obs_path}")
    print(f"  per-pool:        {args.out}")
    print(f"\n  next:  python scripts/mice_behavior/ppi_report.py --predictions {args.out}")


if __name__ == "__main__":
    main()
