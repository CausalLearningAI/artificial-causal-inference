#!/usr/bin/env python3
"""
Deploy the best PPCI ants model: train on all available data, evaluate on v5.

This script is run AFTER the hparam search (src/ppci/hparam_search.py) is complete.
Best hyperparameters are loaded automatically from the latest completed search stage.

Training pipeline
-----------------
  1. Pretrain on v1           (warm start from scratch)
  2. Pretrain on v2           (warm-start from v1)
  3. Finetune on v3 + v4      (combined — both are high-quality, share treatments)
  4. Evaluate on v5           (partially-annotated; skipped if embeddings not ready)

Compared to the hparam search pipeline:
  - n_val=0 everywhere: every annotated frame is used for training
  - No hyperparameter decisions: everything loaded from search results
  - v3+v4 are finetuned JOINTLY (better quality, shared treatment space)
  - auto_fix (LR retry, epoch extension) stays ON for best-effort training

Usage:
  python src/ppci/deploy_model.py                # train + eval
  python src/ppci/deploy_model.py --dry-run      # print config, skip training
  python src/ppci/deploy_model.py --eval-only    # skip training, run v5 eval on saved model
  python src/ppci/deploy_model.py --skip-if-exists  # no-op if model.pt already saved

--skip-if-exists vs --eval-only:
  --skip-if-exists: if model.pt exists, do nothing (training AND eval skipped)
  --eval-only:      skip training (model.pt must exist), always run v5 evaluation
                    → use this when v5 embeddings arrive after the model was trained

Saving:
  results/ppci/ants/hparam/deploy/
  ├── config.json          # best hparams (written automatically from search results)
  ├── model.pt             # final model state dict
  └── metrics.json         # v5 test metrics (if v5 is available)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from ppci.dataset import PPCIDataset        # noqa: E402
from ppci.hparam_search import (            # noqa: E402
    RESULTS_DIR, DS_KWARGS,
    BASE_CFG,
    _embeddings_available, _load_best,
    _build_pretrain_cfg, _build_finetune_cfg,
    _warn_training_health,
)
from ppci.train import build_model, compute_metrics, train  # noqa: E402
from ppci.visualize import plot_summary, plot_comparison_versions  # noqa: E402


# ── load best hparams from the latest completed stage ────────────────────────

def load_best_config(results_dir: Path) -> tuple[dict, str]:
    """Return (cfg_dict, stage_name) from the most complete search stage.

    Falls back through stages from most to least complete so the deploy always
    uses the richest available configuration.
    """
    for stage in ["augmentation", "finetune", "arch", "backbone_context"]:
        try:
            cfg = _load_best(results_dir / stage)
            return cfg, stage
        except (FileNotFoundError, ValueError):
            continue
    raise RuntimeError(
        f"No hparam search results found under {results_dir}.\n"
        "Run `bash scripts/03_train/hparam_full.sh` first."
    )


# ── per-version evaluation and plotting ──────────────────────────────────────

def _eval_version(
    version: str,
    model,
    encoder: str,
    token: str,
    k: int,
    mode: str,
    device: torch.device,
    out_dir: Path,
    outcomes: list[str],
    treatment_labels: dict | None = None,
) -> dict | None:
    """Run inference on one version, save plot_summary, return classification metrics.

    Returns a metrics dict (acc, bacc, recall, precision) if annotations exist,
    or None if the version has no annotations (predictions are still plotted).
    Returns {} (empty) if embeddings are not available.
    """
    if not _embeddings_available(encoder, token, [version]):
        print(f"\n[{version}] Embeddings not available — skipping.")
        return {}

    print(f"\n[{version}] Loading for evaluation ...")
    ds = PPCIDataset.from_disk("ants", version, encoder, token,
                               n_val_videos=0, **DS_KWARGS)
    if k > 0:
        ds.apply_context_window(k, mode=mode)

    ds.add_predictions(model, device)
    obs_df = ds.obs_level()

    has_annotations = bool(ds.has_annotations)
    n_annotated = int((ds.Y.abs().sum(dim=-1) > 0).sum()) if has_annotations else 0

    # Save plot_summary (annotations=True shows ATE+PP-ATE+PO, False shows PP-ATE+PO only)
    plot_path = str(out_dir / f"plot_summary_{version}.png")
    plot_summary(
        obs_df,
        outcomes=outcomes,
        annotations=has_annotations,
        treatment_labels=treatment_labels,
        save=True,
        save_path=plot_path,
    )
    print(f"  ✓ plot_summary saved: {plot_path}  (annotations={has_annotations})")

    if not has_annotations or n_annotated == 0:
        print(f"  [no annotations] Predictions plotted; classification metrics skipped.")
        return None  # None = no annotations (distinct from {} = embeddings missing)

    # Classification metrics on annotated frames only
    m = compute_metrics(model, ds.X, ds.Y, device)
    print(f"  → bacc={m.get('bacc', 0):.4f}  acc={m.get('acc', 0):.4f}"
          f"  recall={m.get('recall', 0):.4f}  precision={m.get('precision', 0):.4f}")
    del ds
    return {
        "acc":       float(m.get("acc",       0.0)),
        "bacc":      float(m.get("bacc",      0.0)),
        "recall":    float(m.get("recall",    0.0)),
        "precision": float(m.get("precision", 0.0)),
    }


# ── multi-version evaluation loop ────────────────────────────────────────────

def _run_evaluation(model, cfg_dict, encoder, token, k, mode, device, out_dir: Path) -> None:
    """Evaluate on v1–v5, save per-version plot_summary and a cross-version comparison."""
    outcomes = [c.replace("Y_", "") for c in DS_KWARGS.get("outcome_cols", ["Y_Y2F", "Y_B2F"])]
    versions = ["v1", "v2", "v3", "v4", "v5"]
    version_metrics: dict = {}

    for version in versions:
        m = _eval_version(version, model, encoder, token, k, mode, device,
                          out_dir, outcomes)
        version_metrics[version] = m  # None = no annotations, {} = no embeddings

    # Save per-version metrics to JSON (skip versions with no embeddings)
    metrics_out = {
        v: m for v, m in version_metrics.items() if m is not None and m != {}
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"\n  ✓ Metrics saved: {out_dir / 'metrics.json'}")

    # Plot comparison across versions (only versions with embeddings appear on x-axis)
    available = {v: m for v, m in version_metrics.items() if m != {}}
    if available:
        comp_path = str(out_dir / "plot_comparison_versions.png")
        plot_comparison_versions(available, save=True, save_path=comp_path)
        print(f"  ✓ plot_comparison_versions saved: {comp_path}")


# ── main deployment run ───────────────────────────────────────────────────────

def deploy(
    results_dir: Path,
    dry_run: bool = False,
    skip_if_exists: bool = False,
    eval_only: bool = False,
) -> None:
    out_dir    = results_dir / "deploy"
    model_path = out_dir / "model.pt"

    # ── load best config (always — needed for eval-only too) ─────────────────
    cfg_dict, from_stage = load_best_config(results_dir)
    encoder = cfg_dict["encoder"]
    token   = cfg_dict.get("token", "class")
    k       = int(cfg_dict.get("context_window", BASE_CFG.training.context_window))
    mode    = cfg_dict.get("context_mode", BASE_CFG.training.context_mode)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pretrain_cfg = _build_pretrain_cfg(cfg_dict)
    finetune_cfg = _build_finetune_cfg(cfg_dict)

    print("=" * 72)
    print("[DEPLOY] PPCI ants — pretrain v1+v2, finetune v3+v4 jointly, eval v5")
    print(f"  hparams from   : {from_stage} stage  (loaded automatically)")
    print(f"  encoder        : {encoder}/{token}  k={k}  mode={mode}")
    print(f"  arch           : dim={cfg_dict.get('hidden_dim', 512)}"
          f"  layers={cfg_dict.get('hidden_layers', 1)}"
          f"  dropout={cfg_dict.get('dropout', 0.3)}")
    print(f"  finetune       : method={cfg_dict.get('method', 'ERM')}"
          f"  lr={cfg_dict.get('finetune_lr', 1e-4):.0e}"
          f"  wd={cfg_dict.get('weight_decay', 0.001)}")
    print(f"  n_val          : 0 everywhere (all frames used for training)")
    print(f"  output         : {out_dir}")
    print("=" * 72)

    # ── eval-only: load saved model and evaluate all versions ────────────────
    if eval_only:
        if not model_path.exists():
            print(f"[ERROR] --eval-only requires a saved model at {model_path}",
                  file=sys.stderr)
            sys.exit(1)
        print(f"\n[eval-only] Loading model from {model_path}")
        # Build model structure from a temporary dataset, then load weights
        ds_tmp = PPCIDataset.from_disk("ants", "v4", encoder, token,
                                       n_val_videos=0, **DS_KWARGS)
        if k > 0:
            ds_tmp.apply_context_window(k, mode=mode)
        model = build_model(ds_tmp, finetune_cfg)
        del ds_tmp
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        out_dir.mkdir(parents=True, exist_ok=True)
        _run_evaluation(model, cfg_dict, encoder, token, k, mode, device, out_dir)
        return

    # ── skip-if-exists ───────────────────────────────────────────────────────
    if skip_if_exists and model_path.exists():
        print(f"[SKIP] model already deployed: {model_path}")
        return

    if dry_run:
        print("[DRY RUN — skipping training]")
        return

    # ── check embeddings ─────────────────────────────────────────────────────
    if not _embeddings_available(encoder, token, ["v1", "v2", "v3", "v4"]):
        print(f"[ERROR] Embeddings missing for {encoder}/{token} (v1–v4).", file=sys.stderr)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    def _load(version: str) -> PPCIDataset:
        # n_val_videos=0: every annotated frame used for training (no held-out val)
        ds = PPCIDataset.from_disk("ants", version, encoder, token,
                                   n_val_videos=0, **DS_KWARGS)
        if k > 0:
            ds.apply_context_window(k, mode=mode)
        return ds

    # ── pretrain: v1 → v2 (warm-start chain) ─────────────────────────────────
    model = None
    for version in ["v1", "v2"]:
        print(f"\n[pretrain] Loading {version} ...")
        ds = _load(version)
        if model is None:
            model = build_model(ds, pretrain_cfg)
        model = train(ds, model, pretrain_cfg)
        print(f"  → best_epoch={getattr(model, 'best_epoch', '?')}")
        _warn_training_health(getattr(model, "training_diagnostics", None), f"pretrain {version}")
        del ds

    # ── finetune: v3 + v4 jointly ────────────────────────────────────────────
    # v3 and v4 are higher-quality recordings that share treatment types;
    # combining them gives more treatment diversity for causal methods.
    print("\n[finetune] Loading v3 + v4 (combined) ...")
    ds_v3v4 = PPCIDataset.concat([_load("v3"), _load("v4")])
    model = train(ds_v3v4, model, finetune_cfg)
    print(f"  → best_epoch={getattr(model, 'best_epoch', '?')}")
    _warn_training_health(getattr(model, "training_diagnostics", None), "finetune v3+v4")
    del ds_v3v4

    # ── save model + config ───────────────────────────────────────────────────
    torch.save(model.state_dict(), model_path)
    with open(out_dir / "config.json", "w") as f:
        json.dump({**cfg_dict, "hparams_from_stage": from_stage}, f, indent=2)
    print(f"\n  ✓ Model saved: {model_path}")

    # ── evaluate and plot all versions ────────────────────────────────────────
    _run_evaluation(model, cfg_dict, encoder, token, k, mode, device, out_dir)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deploy best PPCI model: pretrain v1+v2, finetune v3+v4, eval v5"
    )
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR,
                        help=f"Hparam search results root (default: {RESULTS_DIR})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print config but skip training")
    parser.add_argument("--skip-if-exists", action="store_true",
                        help="No-op if model.pt already exists (training AND eval skipped)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training; load saved model.pt and run v5 evaluation. "
                             "Use this when v5 embeddings arrive after training.")
    args = parser.parse_args()

    if args.eval_only and args.skip_if_exists:
        parser.error("--eval-only and --skip-if-exists are mutually exclusive")

    deploy(args.results_dir,
           dry_run=args.dry_run,
           skip_if_exists=args.skip_if_exists,
           eval_only=args.eval_only)


if __name__ == "__main__":
    main()
