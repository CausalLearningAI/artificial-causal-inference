#!/usr/bin/env python3
"""
Deploy the best PPCI ants model across multiple training configurations.

This script is run AFTER the hparam search (src/ppci/hparam_search.py) is complete.
Best hyperparameters are loaded automatically from the latest completed search stage.

Each configuration defines:
  pretrain_versions — warm-start chain run sequentially (v1 → v2 → …)
  finetune_versions — finetuned jointly in one combined step
  test_versions     — held-out evaluation (embeddings may be absent; skipped if so)

All other versions that appear in pretrain+finetune are evaluated in-sample.

Saving (one subfolder per config):
  results/ppci/ants/hparam/deploy/{config_name}/
  ├── config.json                          # best hparams + deploy config metadata
  ├── model.pt                             # final model state dict
  ├── metrics.json                         # per-version classification metrics
  ├── plot_summary_{version}.png           # ATE / PP-ATE / PO per version
  ├── plot_comparison_versions.png         # versions on x-axis, metrics as bars
  └── plot_comparison_versions_by_metric.png  # metrics on x-axis, versions as bars

Usage:
  python src/ppci/deploy_model.py                           # run all configs
  python src/ppci/deploy_model.py --config v4_holdout       # run one config by name
  python src/ppci/deploy_model.py --dry-run                 # print configs, skip training
  python src/ppci/deploy_model.py --eval-only               # re-eval all (model.pt must exist)
  python src/ppci/deploy_model.py --eval-only --config v4_holdout  # re-eval one config
  python src/ppci/deploy_model.py --skip-if-exists          # skip configs already trained

--skip-if-exists: if model.pt already exists for a config, skip that config entirely.
--eval-only:      skip training; load saved model.pt and re-run evaluation + plots.
                  Useful when new embeddings (e.g. v5) arrive after training.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
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
from ppci.visualize import (                # noqa: E402
    plot_summary,
    plot_comparison_versions,
    plot_comparison_versions_by_metric,
)


# ── deployment configurations ─────────────────────────────────────────────────

@dataclass
class DeployConfig:
    """One named deployment configuration.

    Attributes:
        name:              Subfolder name under deploy/ (must be a valid dir name).
        pretrain_versions: Versions used in the sequential warm-start pretrain chain.
        finetune_versions: Versions finetuned jointly in a single combined step.
        test_versions:     Held-out versions (embeddings optional; skipped if missing).
        description:       Human-readable summary printed at the start of the run.
    """
    name:              str
    pretrain_versions: list[str]
    finetune_versions: list[str]
    test_versions:     list[str]
    description:       str = field(default="")

    @property
    def train_versions(self) -> list[str]:
        """All versions used for training (pretrain + finetune), in order."""
        seen, out = set(), []
        for v in self.pretrain_versions + self.finetune_versions:
            if v not in seen:
                out.append(v)
                seen.add(v)
        return out

    @property
    def all_versions(self) -> list[str]:
        """train_versions followed by test_versions (deduped)."""
        seen, out = set(), []
        for v in self.train_versions + self.test_versions:
            if v not in seen:
                out.append(v)
                seen.add(v)
        return out


DEPLOY_CONFIGS: list[DeployConfig] = [
    DeployConfig(
        name="validate",
        pretrain_versions=["v1", "v2"],
        finetune_versions=["v3"],
        test_versions=["v4"],
        description="pretrain v1→v2 · finetune v3 · test v4  (held-out validation)",
    ),
    DeployConfig(
        name="final",
        pretrain_versions=["v1", "v2"],
        finetune_versions=["v3", "v4"],
        test_versions=["v5"],
        description="pretrain v1→v2 · finetune v3+v4 · test v5  (final model)",
    ),
]

# Convenience lookup by name
_CONFIGS_BY_NAME: dict[str, DeployConfig] = {c.name: c for c in DEPLOY_CONFIGS}


# ── load best hparams from the latest completed stage ────────────────────────

def load_best_config(results_dir: Path) -> tuple[dict, str]:
    """Return (cfg_dict, stage_name) from the most complete search stage."""
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

    Returns:
        dict  — metrics (acc, bacc, recall, precision) if annotations exist.
        None  — version has no annotations (predictions still plotted).
        {}    — embeddings unavailable (version skipped entirely).
    """
    if not _embeddings_available(encoder, token, [version]):
        print(f"\n[{version}] Embeddings not available — skipping.")
        return {}

    print(f"\n[{version}] Loading for evaluation ...")
    ds = PPCIDataset.from_disk("ants", version, encoder, token,
                               n_val_videos=0, **DS_KWARGS)

    raw_mean    = float(ds.X.mean())
    raw_std     = float(ds.X.std())
    n_zero_rows = int((ds.X.abs().sum(dim=1) == 0).sum())
    zero_pct    = 100.0 * n_zero_rows / len(ds.X)
    print(f"  embedding stats: mean={raw_mean:.3f}  std={raw_std:.3f}  "
          f"zero_rows={n_zero_rows:,} ({zero_pct:.1f}%)")
    if n_zero_rows > 0:
        print(f"  [ERROR] {version} embeddings contain {n_zero_rows:,} all-zero rows "
              f"({zero_pct:.1f}% of {len(ds.X):,}). The extraction job was likely "
              f"killed before completion. Re-run: "
              f"python src/embedding/get_embeddings.py encoder={encoder} token={token} "
              f"experiment=ants/{version} overwrite.embeddings=true")
    elif raw_std < 1.0:
        print(f"  [WARNING] {version} embeddings look corrupted "
              f"(std={raw_std:.3f} — expected >> 1). "
              f"Re-run embedding extraction before trusting these predictions.")

    if k > 0:
        ds.apply_context_window(k, mode=mode)

    ds.add_predictions(model, device)
    obs_df = ds.obs_level()

    has_annotations = bool(ds.has_annotations)
    n_annotated = int((ds.Y.abs().sum(dim=-1) > 0).sum()) if has_annotations else 0

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
        return None

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


def _run_evaluation(
    model,
    cfg: DeployConfig,
    encoder: str,
    token: str,
    k: int,
    mode: str,
    device: torch.device,
    out_dir: Path,
) -> None:
    """Evaluate all versions for a config, save per-version summaries and comparison plots."""
    outcomes = [c.replace("Y_", "") for c in DS_KWARGS.get("outcome_cols", ["Y_Y2F", "Y_B2F"])]
    version_metrics: dict = {}

    for version in cfg.all_versions:
        m = _eval_version(version, model, encoder, token, k, mode, device, out_dir, outcomes)
        version_metrics[version] = m  # None = no annotations, {} = no embeddings

    # Save per-version metrics to JSON (skip versions with no embeddings)
    metrics_out = {v: m for v, m in version_metrics.items() if m is not None and m != {}}
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"\n  ✓ Metrics saved: {out_dir / 'metrics.json'}")

    # Comparison plots (only versions with embeddings appear)
    available = {v: m for v, m in version_metrics.items() if m != {}}
    if not available:
        return

    comp_path = str(out_dir / "plot_comparison_versions.png")
    plot_comparison_versions(
        available,
        train_versions=cfg.train_versions,
        pretrain_versions=cfg.pretrain_versions,
        test_versions=cfg.test_versions,
        save=True,
        save_path=comp_path,
    )
    print(f"  ✓ plot_comparison_versions saved: {comp_path}")

    comp_by_metric_path = str(out_dir / "plot_comparison_versions_by_metric.png")
    plot_comparison_versions_by_metric(
        available,
        train_versions=cfg.train_versions,
        pretrain_versions=cfg.pretrain_versions,
        test_versions=cfg.test_versions,
        save=True,
        save_path=comp_by_metric_path,
    )
    print(f"  ✓ plot_comparison_versions_by_metric saved: {comp_by_metric_path}")


# ── single-config deployment ──────────────────────────────────────────────────

def deploy_one(
    cfg: DeployConfig,
    results_dir: Path,
    dry_run: bool = False,
    skip_if_exists: bool = False,
    eval_only: bool = False,
) -> None:
    """Train (or load) and evaluate one DeployConfig."""
    out_dir    = results_dir / "deploy" / cfg.name
    model_path = out_dir / "model.pt"

    cfg_dict, from_stage = load_best_config(results_dir)
    encoder = cfg_dict["encoder"]
    token   = cfg_dict.get("token", "class")
    k       = int(cfg_dict.get("context_window", BASE_CFG.training.context_window))
    mode    = cfg_dict.get("context_mode", BASE_CFG.training.context_mode)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pretrain_cfg = _build_pretrain_cfg(cfg_dict)
    finetune_cfg = _build_finetune_cfg(cfg_dict)

    print("=" * 72)
    print(f"[DEPLOY] {cfg.name} — {cfg.description}")
    print(f"  pretrain       : {' → '.join(cfg.pretrain_versions)}")
    print(f"  finetune       : {' + '.join(cfg.finetune_versions)}  (joint)")
    print(f"  test           : {', '.join(cfg.test_versions)}")
    print(f"  hparams from   : {from_stage} stage")
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

    # ── eval-only ─────────────────────────────────────────────────────────────
    if eval_only:
        if not model_path.exists():
            print(f"[ERROR] --eval-only requires a saved model at {model_path}",
                  file=sys.stderr)
            sys.exit(1)
        print(f"\n[eval-only] Loading model from {model_path}")
        # Use the last finetune version to infer model structure
        ref_version = cfg.finetune_versions[-1]
        ds_tmp = PPCIDataset.from_disk("ants", ref_version, encoder, token,
                                       n_val_videos=0, **DS_KWARGS)
        if k > 0:
            ds_tmp.apply_context_window(k, mode=mode)
        model = build_model(ds_tmp, finetune_cfg)
        del ds_tmp
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        out_dir.mkdir(parents=True, exist_ok=True)
        _run_evaluation(model, cfg, encoder, token, k, mode, device, out_dir)
        return

    # ── skip-if-exists ────────────────────────────────────────────────────────
    if skip_if_exists and model_path.exists():
        print(f"[SKIP] model already deployed: {model_path}")
        return

    if dry_run:
        print("[DRY RUN — skipping training]")
        return

    # ── check embeddings ──────────────────────────────────────────────────────
    required = cfg.pretrain_versions + cfg.finetune_versions
    if not _embeddings_available(encoder, token, required):
        print(f"[ERROR] Embeddings missing for {encoder}/{token} "
              f"({', '.join(required)}).", file=sys.stderr)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    def _load(version: str) -> PPCIDataset:
        ds = PPCIDataset.from_disk("ants", version, encoder, token,
                                   n_val_videos=0, **DS_KWARGS)
        if k > 0:
            ds.apply_context_window(k, mode=mode)
        return ds

    # ── pretrain: sequential warm-start chain ─────────────────────────────────
    model = None
    for version in cfg.pretrain_versions:
        print(f"\n[pretrain] Loading {version} ...")
        ds = _load(version)
        if model is None:
            model = build_model(ds, pretrain_cfg)
        model = train(ds, model, pretrain_cfg)
        print(f"  → best_epoch={getattr(model, 'best_epoch', '?')}")
        _warn_training_health(getattr(model, "training_diagnostics", None),
                              f"pretrain {version}")
        del ds

    # ── finetune: joint training on all finetune versions ─────────────────────
    ft_label = " + ".join(cfg.finetune_versions)
    print(f"\n[finetune] Loading {ft_label} (combined) ...")
    ds_ft = PPCIDataset.concat([_load(v) for v in cfg.finetune_versions])
    model = train(ds_ft, model, finetune_cfg)
    print(f"  → best_epoch={getattr(model, 'best_epoch', '?')}")
    _warn_training_health(getattr(model, "training_diagnostics", None),
                          f"finetune {ft_label}")
    del ds_ft

    # ── save model + config ───────────────────────────────────────────────────
    torch.save(model.state_dict(), model_path)
    with open(out_dir / "config.json", "w") as f:
        json.dump({
            **cfg_dict,
            "hparams_from_stage":  from_stage,
            "deploy_config":       cfg.name,
            "pretrain_versions":   cfg.pretrain_versions,
            "finetune_versions":   cfg.finetune_versions,
            "test_versions":       cfg.test_versions,
        }, f, indent=2)
    print(f"\n  ✓ Model saved: {model_path}")

    # ── evaluate and plot all versions ────────────────────────────────────────
    _run_evaluation(model, cfg, encoder, token, k, mode, device, out_dir)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    config_names = list(_CONFIGS_BY_NAME.keys())

    parser = argparse.ArgumentParser(
        description="Deploy best PPCI model for one or all named configurations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(
            [f"Available configs:"]
            + [f"  {c.name:20s} — {c.description}" for c in DEPLOY_CONFIGS]
        ),
    )
    parser.add_argument(
        "--config",
        choices=config_names,
        default=None,
        metavar="NAME",
        help=f"Run only this config (choices: {', '.join(config_names)}). "
             "Default: run all configs sequentially.",
    )
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR,
                        help=f"Hparam search results root (default: {RESULTS_DIR})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print config info but skip training")
    parser.add_argument("--skip-if-exists", action="store_true",
                        help="No-op for any config whose model.pt already exists")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training; load saved model.pt and re-run evaluation. "
                             "Use this when new embeddings arrive after training.")
    args = parser.parse_args()

    if args.eval_only and args.skip_if_exists:
        parser.error("--eval-only and --skip-if-exists are mutually exclusive")

    configs_to_run = (
        [_CONFIGS_BY_NAME[args.config]] if args.config else DEPLOY_CONFIGS
    )

    for cfg in configs_to_run:
        deploy_one(
            cfg,
            args.results_dir,
            dry_run=args.dry_run,
            skip_if_exists=args.skip_if_exists,
            eval_only=args.eval_only,
        )


if __name__ == "__main__":
    main()
