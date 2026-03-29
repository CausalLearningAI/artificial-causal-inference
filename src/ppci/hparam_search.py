#!/usr/bin/env python3
"""
Hyperparameter search for PPCI ants deployment (Exp 5).

Pipeline per config:
  Early stages (backbone / context / arch):
      1. Pretrain MLP on v1  (fresh weights)
      2. Continue pretrain on v2  (warm-start, optimizer reset)
      3. Finetune on v3  (warm-start, lower lr, 10 fixed val videos)
      4. Evaluate on v4  (test set)

  Late stages (finetune / augmentation):
      — Load the saved pretrained_v1v2.pt from the best arch run  ← NO re-pretrain
      3. Finetune on v3  (only finetuning hyperparams vary)
      4. Evaluate on v4

Stages (run in order):
  backbone_context ~20 runs  (encoder × context_window × context_mode) — joint, unbiased
  arch             ~19 runs  (hidden_dim × hidden_layers × dropout)
  finetune         ~18 runs  (method × finetune_lr × weight_decay)  — loads saved pretrain
  augmentation      ~9 runs  (noise_std × mixup_alpha)               — loads saved pretrain

  Note: backbone and context are searched jointly to avoid the bias that would
  arise from fixing context_window while comparing encoders (or vice versa).

Usage:
  # Check how many array jobs a stage needs:
  python src/ppci/hparam_search.py --stage backbone_context --list

  # Submit a stage as a SLURM job array:
  STAGE=backbone_context sbatch --array=0-24  scripts/03_train/hparam.sh
  STAGE=arch             sbatch --array=0-25  scripts/03_train/hparam.sh
  STAGE=finetune         sbatch --array=0-19  scripts/03_train/hparam.sh
  STAGE=augmentation     sbatch --array=0-9   scripts/03_train/hparam.sh

  # Dry run (print config, skip training):
  python src/ppci/hparam_search.py --stage backbone --job-idx 0 --dry-run

  # Print results table for all stages (or one):
  python src/ppci/hparam_search.py --print-results
  python src/ppci/hparam_search.py --print-results --stage backbone

Saving structure:
  results/ppci/ants/hparam/
  ├── backbone/
  │   ├── summary.csv                  # one row per completed run
  │   └── {run_id}/
  │       ├── config.json
  │       ├── pretrained_v1v2.pt       # MLP state dict after v1+v2 pretrain
  │       ├── finetuned_v3.pt          # MLP state dict after v3 finetune
  │       └── metrics.json             # val_bacc_v3, test_bacc_v4, …
  ├── context/  (same layout)
  ├── arch/     (same layout)  ← best run's pretrained_v1v2.pt reused by finetune stage
  ├── finetune/ (no pretrained_v1v2.pt — loads from arch/best/)
  └── augmentation/
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))          # for "from src.dataset…" inside PPCIDataset.from_disk
sys.path.insert(0, str(ROOT / "src"))  # for "from ppci.dataset import …"

from ppci.dataset import PPCIDataset  # noqa: E402
from ppci.model import MLP            # noqa: E402
from ppci.train import build_model, compute_metrics, train  # noqa: E402

RESULTS_DIR     = ROOT / "results" / "ppci" / "ants" / "hparam"
RESULTS_DIR_POV = ROOT / "results" / "ppci" / "ants" / "hparam_pov"
DATASET_ROOT    = ROOT / "dataset" / "ants"

# ── shared dataset kwargs ─────────────────────────────────────────────────────
DS_KWARGS = dict(
    outcome_cols=["Y_Y2F", "Y_B2F"],
    task="multilabel",
    env_cols=["W_batch", "W_nestbox"],
    env_include_treatment=False,
    seed=0,
)

# POV uses the same outcome_cols and task as full — the from_disk POV branch
# does horizontal concatenation of blue/yellow embeddings, same 2-output target.

V3_N_VAL        = 10   # fixed val videos for v3 (same across ALL runs via seed=0)
PRETRAIN_EPOCHS = 30   # v1 and v2; model saved = best by val/bacc (or train/bacc if no val)
FINETUNE_EPOCHS = 20   # v3; model saved = best by val/bacc on fixed 10 val videos

# Stages that skip v1/v2 pretraining and instead load a saved pretrained model
FINETUNE_ONLY_STAGES = {"finetune", "augmentation"}

# ── base config ───────────────────────────────────────────────────────────────
BASE_CFG = OmegaConf.create({
    "outcome": {"columns": ["Y_Y2F", "Y_B2F"], "task": "multilabel"},
    "mlp": {
        "hidden_dim":      512,
        "hidden_layers":   1,
        "dropout":         0.3,
        "context_head_dim": 64,
    },
    "training": {
        "method":              "ERM",
        "batch_size":          256,
        "num_epochs":          PRETRAIN_EPOCHS,
        "lr":                  1e-4,
        "weight_decay":        0.001,
        "seed":                0,
        "pos_weight":          False,
        "calibrate_temperature": True,
        "frame_stride":        10,
        "ic_weight":           1.0,
        "augment_noise_std":   0.0,
        "augment_mixup_alpha": 0.0,
        "context_window":      4,
        "context_mode":        "mean",
        "device":              "cuda",
    },
    "wandb": {"mode": "disabled"},
})


# ── embedding availability ────────────────────────────────────────────────────
def _embeddings_available(encoder: str, token: str, versions: list[str]) -> bool:
    for v in versions:
        path = DATASET_ROOT / v / "embeddings" / "full" / encoder / token / "embeddings.npy"
        if not path.exists():
            return False
    return True


def _pov_embeddings_available(encoder: str, token: str, versions: list[str]) -> bool:
    for v in versions:
        for identity in ("blue", "yellow"):
            path = DATASET_ROOT / v / "embeddings" / "pov" / identity / encoder / token
            if not (path / "embeddings.npy").exists() and not (path / "embeddings.pt").exists():
                return False
    return True


# ── run-id ────────────────────────────────────────────────────────────────────
def _fmt(v: Any) -> str:
    if isinstance(v, float):
        if v == 0.0:
            return "0"
        if v < 0.01:
            return f"{v:.0e}"
        return f"{v:.4f}".rstrip("0").rstrip(".")
    return str(v)


_ID_KEYS = [
    "encoder", "token",
    "context_window", "context_mode",
    "hidden_dim", "hidden_layers", "dropout", "siamese",
    "dist_mode",
    "method", "finetune_lr", "weight_decay",
    "augment_noise_std", "augment_mixup_alpha",
]


def _run_id(cfg_dict: dict) -> str:
    return "__".join(f"{k}={_fmt(cfg_dict[k])}" for k in _ID_KEYS if k in cfg_dict)


# ── summary CSV ───────────────────────────────────────────────────────────────
def _load_best(stage_dir: Path) -> dict:
    """Return the config+metrics row with the highest val_bacc_v3."""
    summary = stage_dir / "summary.csv"
    if not summary.exists():
        raise FileNotFoundError(
            f"No summary at {summary}\nRun stage '{stage_dir.name}' first."
        )
    rows: list[dict] = []
    with open(summary, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    if not rows:
        raise ValueError(f"Empty summary: {summary}")
    best = max(rows, key=lambda r: float(r["val_bacc_v3"]))
    parsed: dict = {}
    for k, v in best.items():
        try:
            f_ = float(v)
            parsed[k] = int(f_) if f_ == int(f_) else f_
        except (ValueError, TypeError):
            parsed[k] = v
    return parsed


def _warn_training_health(diag: dict | None, phase: str) -> None:
    """Print warnings if training health checks fail."""
    if not diag:
        return
    issues = []
    if not diag.get("loss_ok", True):
        issues.append("NaN/Inf loss — LR is likely too high")
    elif diag.get("lr_too_high", False):
        issues.append(
            f"Loss increased in first 5 epochs (improvement={diag['loss_improvement']:.1%}) "
            "— LR may be too high"
        )
    if diag.get("lr_too_low", False) and not diag.get("lr_too_high", False):
        issues.append(
            f"Loss barely decreased (improvement={diag['loss_improvement']:.1%}) "
            "— LR may be too low"
        )
    if not diag.get("converged", True):
        issues.append(
            f"Not converged: best_epoch={diag['best_epoch']}/{diag['n_epochs']} "
            "— consider increasing num_epochs"
        )
    gap = diag.get("overfit_gap")
    if gap is not None and gap > 0.15:
        tr  = diag.get("train_bacc_at_best", "?")
        val = diag.get("val_bacc_at_best",   "?")
        issues.append(
            f"Overfitting: train_bacc={tr:.3f}  val_bacc={val:.3f}  gap={gap:.3f} "
            "— try more dropout / weight decay"
        )
    if issues:
        print(f"\n  !! [{phase}] training health warnings:")
        for issue in issues:
            print(f"       - {issue}")
    else:
        tr  = diag.get("train_bacc_at_best")
        val = diag.get("val_bacc_at_best")
        converged  = diag.get("converged", True)
        improvement = diag.get("loss_improvement", 0)
        status = (
            f"converged=True  best_epoch={diag['best_epoch']}/{diag['n_epochs']}"
            f"  loss_drop={improvement:.0%}"
        )
        if tr is not None:
            status += f"  train_bacc={tr:.3f}"
        if val is not None:
            status += f"  val_bacc={val:.3f}"
        if not converged:
            status += "  [not converged]"
        print(f"  ✓ [{phase}] {status}")


def _append_to_summary(stage_dir: Path, row: dict) -> None:
    path = stage_dir / "summary.csv"
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _best_pretrained_path(results_dir: Path) -> Path:
    """Return path to pretrained_v1v2.pt from the best arch run."""
    best = _load_best(results_dir / "arch")
    path = results_dir / "arch" / best["run_id"] / "pretrained_v1v2.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Pretrained model not found: {path}\n"
            "Run stage 'arch' first and make sure it completed successfully."
        )
    return path


def _best_pretrained_path_pov(results_dir: Path) -> Path:
    """Return path to pretrained_v2.pt from the best POV arch run."""
    best = _load_best(results_dir / "arch")
    path = results_dir / "arch" / best["run_id"] / "pretrained_v2.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Pretrained POV model not found: {path}\n"
            "Run stage 'arch' with --frame-type pov first."
        )
    return path


# ── stage config generators ───────────────────────────────────────────────────

_ENCODERS = [
    ("dinov2",  "class"),
    ("dinov2",  "mean"),
    ("dinov3",  "class"),
    ("dinov3",  "mean"),
]

# Context options: k=0 (no context), k=2/4 mean-pool, k=2/4 concat+attention.
_CONTEXT_OPTIONS = (
    [(k, "mean")   for k in [0, 2, 4]] +
    [(k, "concat") for k in [2, 4]]
)


def _backbone_context_configs(results_dir: Path) -> list[dict]:
    """Joint encoder × context × dist_mode search.

    All (encoder, context_window, context_mode, dist_mode) combos are run
    together so that encoder ranking is not biased by a fixed context setting.
    """
    configs = []
    for encoder, token in _ENCODERS:
        if not _embeddings_available(encoder, token, ["v1", "v2", "v3", "v4"]):
            print(f"[SKIP] {encoder}/{token}: embeddings missing for v1–v4")
            continue
        for k, mode in _CONTEXT_OPTIONS:
            for dist_mode in ["none", "late"]:
                configs.append({
                    "encoder":        encoder,
                    "token":          token,
                    "context_window": k,
                    "context_mode":   mode,
                    "dist_mode":      dist_mode,
                })
    return configs


def _pov_backbone_context_configs(results_dir: Path) -> list[dict]:
    """POV variant: only encoders with pov embeddings on v2/v3/v4 (no v1)."""
    configs = []
    for encoder, token in _ENCODERS:
        if not _pov_embeddings_available(encoder, token, ["v2", "v3", "v4"]):
            print(f"[SKIP] {encoder}/{token}: pov embeddings missing for v2–v4")
            continue
        for k, mode in _CONTEXT_OPTIONS:
            for dist_mode in ["none", "late"]:
                configs.append({
                    "encoder":        encoder,
                    "token":          token,
                    "context_window": k,
                    "context_mode":   mode,
                    "dist_mode":      dist_mode,
                })
    return configs


def _arch_configs(results_dir: Path) -> list[dict]:
    best = _load_best(results_dir / "backbone_context")
    base = {
        "encoder":        best["encoder"],
        "token":          best["token"],
        "context_window": int(best["context_window"]),
        "context_mode":   best["context_mode"],
        "dist_mode":      best.get("dist_mode", "none"),
    }
    configs = []
    for hidden_dim in [256, 512, 1024]:
        for hidden_layers in [0, 1, 2]:
            for dropout in [0.0, 0.2, 0.4]:
                # Linear probe (layers=0): hidden_dim and dropout are irrelevant.
                # Keep only one representative combo to avoid duplicate runs.
                if hidden_layers == 0 and (hidden_dim != 512 or dropout != 0.0):
                    continue
                configs.append({**base,
                                 "hidden_dim":    hidden_dim,
                                 "hidden_layers": hidden_layers,
                                 "dropout":       dropout})
    return configs


def _pov_arch_configs(results_dir: Path) -> list[dict]:
    """POV variant of arch search: adds siamese={True, False} to the grid."""
    best = _load_best(results_dir / "backbone_context")
    base = {
        "encoder":        best["encoder"],
        "token":          best["token"],
        "context_window": int(best["context_window"]),
        "context_mode":   best["context_mode"],
        "dist_mode":      best.get("dist_mode", "none"),
    }
    configs = []
    for hidden_dim in [256, 512, 1024]:
        for hidden_layers in [0, 1, 2]:
            for dropout in [0.0, 0.2, 0.4]:
                for siamese in [True, False]:
                    if hidden_layers == 0 and (hidden_dim != 512 or dropout != 0.0):
                        continue
                    configs.append({**base,
                                     "hidden_dim":    hidden_dim,
                                     "hidden_layers": hidden_layers,
                                     "dropout":       dropout,
                                     "siamese":       siamese})
    return configs


def _finetune_configs(results_dir: Path) -> list[dict]:
    """method × finetune_lr × weight_decay.

    Pretrain settings are fixed (best arch run's pretrained_v1v2.pt is reused).
    No pretrain_lr sweep — that would require retraining on v1+v2 for each combo,
    defeating the purpose of caching the pretrained model.
    """
    best_bk   = _load_best(results_dir / "backbone_context")
    best_arch = _load_best(results_dir / "arch")
    base = {
        "encoder":        best_bk["encoder"],
        "token":          best_bk["token"],
        "context_window": int(best_bk["context_window"]),
        "context_mode":   best_bk["context_mode"],
        "dist_mode":      best_bk.get("dist_mode", "none"),
        "hidden_dim":     int(best_arch["hidden_dim"]),
        "hidden_layers":  int(best_arch["hidden_layers"]),
        "dropout":        float(best_arch["dropout"]),
        "siamese":        str(best_arch.get("siamese", "True")).lower() in ("true", "1"),
    }
    configs = []
    for method in ["ERM", "DERM"]:
        for finetune_lr in [1e-5, 5e-5, 1e-4]:
            for weight_decay in [0.0, 0.001, 0.01]:
                configs.append({**base,
                                 "method":      method,
                                 "finetune_lr": finetune_lr,
                                 "weight_decay": weight_decay})
    return configs


def _augmentation_configs(results_dir: Path) -> list[dict]:
    """augment_noise_std × augment_mixup_alpha; inherits all previous best settings."""
    best_bk   = _load_best(results_dir / "backbone_context")
    best_arch = _load_best(results_dir / "arch")
    best_ft   = _load_best(results_dir / "finetune")
    base = {
        "encoder":        best_bk["encoder"],
        "token":          best_bk["token"],
        "context_window": int(best_bk["context_window"]),
        "context_mode":   best_bk["context_mode"],
        "dist_mode":      best_bk.get("dist_mode", "none"),
        "hidden_dim":     int(best_arch["hidden_dim"]),
        "hidden_layers":  int(best_arch["hidden_layers"]),
        "dropout":        float(best_arch["dropout"]),
        "siamese":        str(best_arch.get("siamese", "True")).lower() in ("true", "1"),
        "method":         best_ft["method"],
        "finetune_lr":    float(best_ft["finetune_lr"]),
        "weight_decay":   float(best_ft["weight_decay"]),
    }
    configs = []
    for noise_std in [0.0, 0.01, 0.03]:
        for mixup_alpha in [0.0, 0.2, 0.4]:
            configs.append({**base,
                             "augment_noise_std":   noise_std,
                             "augment_mixup_alpha": mixup_alpha})
    return configs


STAGE_FNS: dict[str, Any] = {
    "backbone_context": _backbone_context_configs,
    "arch":             _arch_configs,
    "finetune":         _finetune_configs,
    "augmentation":     _augmentation_configs,
}


# ── config builder ────────────────────────────────────────────────────────────
def _build_finetune_cfg(cfg_dict: dict) -> DictConfig:
    """Build the OmegaConf config for the v3 finetune step."""
    return OmegaConf.merge(BASE_CFG, OmegaConf.create({
        "mlp": {
            "hidden_dim":    int(cfg_dict.get("hidden_dim",    BASE_CFG.mlp.hidden_dim)),
            "hidden_layers": int(cfg_dict.get("hidden_layers", BASE_CFG.mlp.hidden_layers)),
            "dropout":       float(cfg_dict.get("dropout",     BASE_CFG.mlp.dropout)),
            "siamese":       bool(cfg_dict.get("siamese",      True)),
        },
        "training": {
            "method":              cfg_dict.get("method",             BASE_CFG.training.method),
            "lr":                  float(cfg_dict.get("finetune_lr",  BASE_CFG.training.lr)),
            "weight_decay":        float(cfg_dict.get("weight_decay", BASE_CFG.training.weight_decay)),
            "num_epochs":          FINETUNE_EPOCHS,
            "context_window":      int(cfg_dict.get("context_window", BASE_CFG.training.context_window)),
            "context_mode":        cfg_dict.get("context_mode",       BASE_CFG.training.context_mode),
            "augment_noise_std":   float(cfg_dict.get("augment_noise_std",   BASE_CFG.training.augment_noise_std)),
            "augment_mixup_alpha": float(cfg_dict.get("augment_mixup_alpha", BASE_CFG.training.augment_mixup_alpha)),
        },
    }))


def _build_pretrain_cfg(cfg_dict: dict) -> DictConfig:
    """Build the OmegaConf config for the v1/v2 pretrain steps (fixed lr=1e-4)."""
    return OmegaConf.merge(BASE_CFG, OmegaConf.create({
        "mlp": {
            "hidden_dim":    int(cfg_dict.get("hidden_dim",    BASE_CFG.mlp.hidden_dim)),
            "hidden_layers": int(cfg_dict.get("hidden_layers", BASE_CFG.mlp.hidden_layers)),
            "dropout":       float(cfg_dict.get("dropout",     BASE_CFG.mlp.dropout)),
            "siamese":       bool(cfg_dict.get("siamese",      True)),
        },
        "training": {
            "method":         "ERM",   # always ERM for pretraining
            "lr":             1e-4,    # fixed pretrain lr
            "weight_decay":   0.001,   # fixed pretrain weight decay
            "num_epochs":     PRETRAIN_EPOCHS,
            "context_window": int(cfg_dict.get("context_window", BASE_CFG.training.context_window)),
            "context_mode":   cfg_dict.get("context_mode", BASE_CFG.training.context_mode),
        },
    }))


# ── main runner ───────────────────────────────────────────────────────────────
def run_one(
    cfg_dict: dict,
    stage: str,
    results_dir: Path,
    dry_run: bool = False,
    frame_type: str = "full",
) -> dict:
    """Run the pipeline for one hyperparameter config.

    frame_type="full":
      Early stages: pretrain v1 → v2 → finetune v3 → eval v4.
      Late stages:  load saved pretrained_v1v2.pt → finetune v3 → eval v4.

    frame_type="pov":
      Early stages: pretrain v2 only → finetune v3 → eval v4  (v1 has no POV).
      Late stages:  load saved pretrained_v2.pt → finetune v3 → eval v4.
      Embeddings are POV crops augmented with [dist_to_focal, dist_to_other].
      Blue→B2F and yellow→Y2F are merged into a single "grooming" outcome.
    """
    run_id    = _run_id(cfg_dict)
    stage_dir = results_dir / stage
    out_dir   = stage_dir / run_id

    # ── skip if already done ──────────────────────────────────────────────────
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists():
        print(f"[SKIP] {run_id}  (already done)")
        with open(metrics_path) as f:
            metrics = json.load(f)
        # Ensure this run appears in the summary (may be missing after cleanup)
        summary_path = stage_dir / "summary.csv"
        if not summary_path.exists() or run_id not in summary_path.read_text():
            summary_row = {"run_id": run_id, **cfg_dict, **metrics}
            summary_row.pop("train_diagnostics", None)
            _append_to_summary(stage_dir, summary_row)
        return metrics

    encoder = cfg_dict["encoder"]
    token   = cfg_dict.get("token", "class")
    k       = int(cfg_dict.get("context_window", BASE_CFG.training.context_window))
    mode    = cfg_dict.get("context_mode", BASE_CFG.training.context_mode)

    # ── check embedding availability ──────────────────────────────────────────
    if frame_type == "pov":
        if not _pov_embeddings_available(encoder, token, ["v2", "v3", "v4"]):
            print(f"[SKIP] {encoder}/{token}  (pov embeddings not ready for v2–v4)")
            return {}
    else:
        if not _embeddings_available(encoder, token, ["v1", "v2", "v3", "v4"]):
            print(f"[SKIP] {encoder}/{token}  (embeddings not ready for all v1–v4)")
            return {}

    # ── print header ──────────────────────────────────────────────────────────
    finetune_only = stage in FINETUNE_ONLY_STAGES
    finetune_diag: dict | None = None
    dist_mode = cfg_dict.get("dist_mode", "none")
    print(f"\n{'='*72}")
    print(f"[RUN]  stage={stage}  frame_type={frame_type}  "
          f"{'(finetune-only — loads saved pretrain)' if finetune_only else '(full pretrain+finetune)'}")
    print(f"       id={run_id}")
    print(f"       encoder={encoder}/{token}  k={k}  mode={mode}  dist_mode={dist_mode}")
    print(f"       arch: dim={cfg_dict.get('hidden_dim', 512)}"
          f"  layers={cfg_dict.get('hidden_layers', 1)}"
          f"  dropout={cfg_dict.get('dropout', 0.3)}"
          f"  siamese={cfg_dict.get('siamese', True)}")
    print(f"       finetune: method={cfg_dict.get('method', 'ERM')}"
          f"  lr={cfg_dict.get('finetune_lr', 1e-4):.0e}"
          f"  wd={cfg_dict.get('weight_decay', 0.001)}"
          f"  noise={cfg_dict.get('augment_noise_std', 0.0)}"
          f"  mixup={cfg_dict.get('augment_mixup_alpha', 0.0)}")

    if dry_run:
        print("  [DRY RUN — skipping training]")
        return {}

    finetune_cfg = _build_finetune_cfg(cfg_dict)
    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_kwargs = DS_KWARGS

    def _load_and_apply(version: str, n_val: int) -> PPCIDataset:
        """Load one dataset version and immediately apply context window."""
        ds = PPCIDataset.from_disk("ants", version, encoder, token,
                                   frame_type=frame_type,
                                   dist_mode=dist_mode,
                                   n_val_videos=n_val, **ds_kwargs)
        if k > 0:
            ds.apply_context_window(k, mode=mode)
        return ds

    # ══════════════════════════════════════════════════════════════════════════
    # BRANCH A — early stages: pretrain then finetune
    # ══════════════════════════════════════════════════════════════════════════
    if not finetune_only:
        pretrain_cfg = _build_pretrain_cfg(cfg_dict)

        if frame_type == "pov":
            # v1 has no POV embeddings — pretrain on v2 only
            print("\n[1] Loading v2 (POV, pretrain) ...")
            ds_v2 = _load_and_apply("v2", n_val=5)
            print("\n[2] Pretraining on v2 ...")
            model = build_model(ds_v2, pretrain_cfg)
            model = train(ds_v2, model, pretrain_cfg)
            print(f"  → best_epoch={getattr(model, 'best_epoch', '?')}")
            _warn_training_health(getattr(model, "training_diagnostics", None), "pretrain v2")
            torch.save(model.state_dict(), out_dir / "pretrained_v2.pt")
            print(f"  → saved: pretrained_v2.pt")
            del ds_v2
        else:
            print("\n[1] Loading v1 ...")
            ds_v1 = _load_and_apply("v1", n_val=5)

            print("\n[2] Pretraining on v1 ...")
            model = build_model(ds_v1, pretrain_cfg)
            model = train(ds_v1, model, pretrain_cfg)
            print(f"  → best_epoch={getattr(model, 'best_epoch', '?')}")
            _warn_training_health(getattr(model, "training_diagnostics", None), "pretrain v1")
            del ds_v1

            print("\n[2 cont] Loading v2, continuing pretrain (warm-start from v1) ...")
            ds_v2 = _load_and_apply("v2", n_val=5)
            model = train(ds_v2, model, pretrain_cfg)
            print(f"  → best_epoch={getattr(model, 'best_epoch', '?')}")
            _warn_training_health(getattr(model, "training_diagnostics", None), "pretrain v2")
            torch.save(model.state_dict(), out_dir / "pretrained_v1v2.pt")
            print(f"  → saved: pretrained_v1v2.pt")
            del ds_v2

    # ══════════════════════════════════════════════════════════════════════════
    # BRANCH B — late stages: load saved pretrained model weights
    # ══════════════════════════════════════════════════════════════════════════
    else:
        if frame_type == "pov":
            pretrained_path = _best_pretrained_path_pov(results_dir)
        else:
            pretrained_path = _best_pretrained_path(results_dir)
        print(f"\n[2] Loading pretrained model from:\n    {pretrained_path}")
        ds_tmp = _load_and_apply("v3", n_val=0)
        model  = build_model(ds_tmp, finetune_cfg)
        del ds_tmp
        model.load_state_dict(torch.load(pretrained_path, map_location=device))
        model = model.to(device)

    # ── finetune on v3 (both branches) ───────────────────────────────────────
    print("\n[3] Loading v3, finetuning ...")
    ds_v3 = _load_and_apply("v3", n_val=V3_N_VAL)
    model = train(ds_v3, model, finetune_cfg)
    finetune_diag = getattr(model, "training_diagnostics", None)
    torch.save(model.state_dict(), out_dir / "finetuned_v3.pt")
    print(f"  → best_epoch={getattr(model, 'best_epoch', '?')}  saved: finetuned_v3.pt")
    _warn_training_health(finetune_diag, "finetune v3")

    val_metrics = compute_metrics(model, ds_v3.X_val, ds_v3.Y_val, device)
    val_bacc    = float(val_metrics.get("bacc", 0.0))
    val_acc     = float(val_metrics.get("acc",  0.0))
    print(f"  → val_bacc_v3={val_bacc:.4f}  val_acc_v3={val_acc:.4f}")
    del ds_v3

    # ── evaluate on v4 (test) ─────────────────────────────────────────────────
    print("\n[4] Loading v4, evaluating (test) ...")
    ds_v4        = _load_and_apply("v4", n_val=0)
    test_metrics = compute_metrics(model, ds_v4.X, ds_v4.Y, device)
    test_bacc    = float(test_metrics.get("bacc", 0.0))
    test_acc     = float(test_metrics.get("acc",  0.0))
    print(f"  → test_bacc_v4={test_bacc:.4f}  test_acc_v4={test_acc:.4f}")
    del ds_v4

    # ── save results ──────────────────────────────────────────────────────────
    metrics = {
        "val_bacc_v3":    val_bacc,
        "test_bacc_v4":   test_bacc,
        "val_acc_v3":     val_acc,
        "test_acc_v4":    test_acc,
        "val_recall_v3":  float(val_metrics.get("recall",  0.0)),
        "test_recall_v4": float(test_metrics.get("recall", 0.0)),
        "best_epoch":     int(getattr(model, "best_epoch", -1)),
        "train_diagnostics": finetune_diag or {},
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg_dict, f, indent=2)

    summary_row = {"run_id": run_id, **cfg_dict, **metrics}
    summary_row.pop("train_diagnostics", None)  # dict breaks CSV parsing
    _append_to_summary(stage_dir, summary_row)

    print(f"\n  ✓ Done  val_bacc_v3={val_bacc:.4f}  test_bacc_v4={test_bacc:.4f}")
    print(f"  Results: {out_dir}")
    return metrics


# ── results printer ───────────────────────────────────────────────────────────
_METRIC_COLS = ["val_bacc_v3", "test_bacc_v4", "val_acc_v3", "test_acc_v4"]
_CFG_COLS_PER_STAGE = {
    "backbone_context": ["encoder", "token", "context_window", "context_mode", "dist_mode"],
    "arch":             ["hidden_dim", "hidden_layers", "dropout", "siamese"],
    "finetune":         ["method", "finetune_lr", "weight_decay"],
    "augmentation":     ["augment_noise_std", "augment_mixup_alpha"],
}


def _find_failed_configs(stage: str, results_dir: Path) -> list[dict]:
    """Return configs that were expected but whose metrics.json is missing.

    Only feasible for backbone_context (no dependency on prior stage output).
    Returns [] for other stages or if expected configs can't be determined.
    """
    if stage != "backbone_context":
        return []
    try:
        is_pov = "pov" in results_dir.name
        gen = _pov_backbone_context_configs if is_pov else _backbone_context_configs
        expected = gen(results_dir)
    except Exception:
        return []
    failed = []
    stage_dir = results_dir / stage
    for cfg in expected:
        run_id = _run_id(cfg)
        if not (stage_dir / run_id / "metrics.json").exists():
            failed.append(cfg)
    return failed


def print_results(results_dir: Path, stage: str | None = None) -> None:
    """Print a formatted leaderboard for one or all stages."""
    stages = [stage] if stage else list(STAGE_FNS.keys())

    for s in stages:
        summary_path = results_dir / s / "summary.csv"
        if not summary_path.exists():
            print(f"\n=== {s} — no results yet ===")
            continue

        with open(summary_path, newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            print(f"\n=== {s} — empty summary ===")
            continue

        def _safe_float(v, default=0.0):
            try:
                return float(v)
            except (ValueError, TypeError):
                return default

        # Sort by test_bacc_v4 descending
        rows.sort(key=lambda r: _safe_float(r.get("test_bacc_v4")), reverse=True)

        cfg_cols = _CFG_COLS_PER_STAGE.get(s, [])
        display_cols = cfg_cols + _METRIC_COLS

        # Compute column widths
        widths = {c: max(len(c), max(len(str(r.get(c, ""))) for r in rows))
                  for c in display_cols}
        widths["rank"] = max(len("rank"), len(str(len(rows))))

        sep   = "  "
        hdr   = sep.join(f"{'rank':>{widths['rank']}}") + sep
        hdr  += sep.join(f"{c:>{widths[c]}}" for c in display_cols)
        divider = "-" * len(hdr)

        # Find best val and test rows
        best_val_idx  = max(range(len(rows)), key=lambda i: _safe_float(rows[i].get("val_bacc_v3")))
        best_test_idx = 0  # already sorted by test_bacc_v4

        # Check for missing configs (pending or failed)
        missing = _find_failed_configs(s, results_dir)
        stage_dir = results_dir / s
        n_failed = sum(1 for cfg in missing
                       if (stage_dir / _run_id(cfg)).exists())
        n_pending = len(missing) - n_failed

        print(f"\n{'='*len(hdr)}")
        n_complete = len(rows)
        status_parts = [f"{n_complete} completed"]
        if n_pending:
            status_parts.append(f"{n_pending} pending")
        if n_failed:
            status_parts.append(f"{n_failed} FAILED")
        print(f"  {s.upper()} ({', '.join(status_parts)})")
        print(f"{'='*len(hdr)}")
        print(hdr)
        print(divider)

        for i, row in enumerate(rows):
            rank_str = f"{i+1:>{widths['rank']}}"
            parts = []
            for c in display_cols:
                val = row.get(c, "")
                if c in _METRIC_COLS:
                    f = _safe_float(val, None)
                    if f is not None:
                        parts.append(f"{f:.4f}".rjust(widths[c]))
                        continue
                parts.append(str(val).rjust(widths[c]))
            line = rank_str + sep + sep.join(parts)
            marker = ""
            if i == best_test_idx:
                marker += " ← best test"
            if i == best_val_idx and best_val_idx != best_test_idx:
                marker += " ← best val"
            print(line + marker)

        # Print best config summary
        best = rows[0]
        print(divider)
        print(f"  Best (test): " + "  ".join(
            f"{c}={best.get(c, '?')}" for c in cfg_cols
        ))
        best_val = rows[best_val_idx]
        if best_val_idx != 0:
            print(f"  Best (val):  " + "  ".join(
                f"{c}={best_val.get(c, '?')}" for c in cfg_cols
            ))

        # Report failed configs (started but no metrics.json — crashed)
        if n_failed:
            print(f"\n  !! {n_failed} config(s) failed (started but no metrics.json):")
            for cfg in missing:
                if (stage_dir / _run_id(cfg)).exists():
                    desc = f"encoder={cfg['encoder']}/{cfg['token']}"
                    desc += f"  k={cfg['context_window']}  mode={cfg['context_mode']}"
                    print(f"       - {desc}")


# ── CLI ────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="PPCI ants hyperparameter search"
    )
    parser.add_argument("--stage", choices=list(STAGE_FNS),
                        help="Stage to run or filter (required except with --print-results)")
    parser.add_argument("--job-idx", type=int, default=None,
                        help="SLURM array task index (0-based); omit with --list")
    parser.add_argument("--list", action="store_true",
                        help="Print all configs for this stage and exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print config but skip training")
    parser.add_argument("--print-results", action="store_true",
                        help="Print leaderboard table(s) and exit")
    parser.add_argument("--frame-type", choices=["full", "pov"], default="full",
                        help="'full' (default) or 'pov' — selects embedding view and pipeline")
    parser.add_argument("--results-dir", type=Path, default=None,
                        help="Root results directory (default: hparam/ or hparam_pov/)")
    args = parser.parse_args()

    # Default results dir depends on frame_type
    if args.results_dir is None:
        args.results_dir = RESULTS_DIR_POV if args.frame_type == "pov" else RESULTS_DIR

    # For POV, use POV-specific config generators
    stage_fns = dict(STAGE_FNS)
    if args.frame_type == "pov":
        stage_fns["backbone_context"] = _pov_backbone_context_configs
        stage_fns["arch"] = _pov_arch_configs

    if args.print_results:
        print_results(args.results_dir, args.stage)
        return

    if args.stage is None:
        parser.error("--stage is required (or use --print-results without --stage)")

    try:
        configs = stage_fns[args.stage](args.results_dir)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    if not configs:
        print(f"[WARN] No runnable configs for stage '{args.stage}' "
              "(embeddings may not be extracted yet).")
        sys.exit(0)

    if args.list:
        print(f"Stage '{args.stage}' (frame_type={args.frame_type}): {len(configs)} configs  "
              f"→ use --array=0-{len(configs)-1} in sbatch")
        for i, c in enumerate(configs):
            print(f"  [{i:3d}] {c}")
        sys.exit(0)

    if args.job_idx is None:
        parser.error("--job-idx is required (or use --list / --print-results)")

    if args.job_idx >= len(configs):
        print(f"[NOOP] job-idx={args.job_idx} >= n_configs={len(configs)}")
        sys.exit(0)

    run_one(configs[args.job_idx], args.stage, args.results_dir,
            dry_run=args.dry_run, frame_type=args.frame_type)


if __name__ == "__main__":
    main()
