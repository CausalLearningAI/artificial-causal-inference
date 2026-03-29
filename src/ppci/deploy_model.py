#!/usr/bin/env python3
"""
Deploy the best PPCI ants model across multiple training configurations.

Run AFTER the hparam search is complete. Best hyperparameters are loaded
automatically from the latest completed search stage.

Each configuration defines:
  pretrain_versions — warm-start chain run sequentially (v1 → v2 → …)
  finetune_versions — finetuned jointly in one combined step
  test_versions     — held-out evaluation (skipped if embeddings absent)

Usage:
  python src/ppci/deploy_model.py                                # full, all configs
  python src/ppci/deploy_model.py --frame-type pov               # pov, all configs
  python src/ppci/deploy_model.py --config validate              # one config only
  python src/ppci/deploy_model.py --eval-only                    # re-eval + plots
  python src/ppci/deploy_model.py --dry-run                      # preview, no training
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
    RESULTS_DIR, RESULTS_DIR_POV, DS_KWARGS,
    BASE_CFG,
    _embeddings_available, _pov_embeddings_available, _load_best,
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
    """One named deployment configuration."""
    name:              str
    pretrain_versions: list[str]
    finetune_versions: list[str]
    test_versions:     list[str]
    description:       str = field(default="")

    @property
    def train_versions(self) -> list[str]:
        seen, out = set(), []
        for v in self.pretrain_versions + self.finetune_versions:
            if v not in seen:
                out.append(v)
                seen.add(v)
        return out

    @property
    def all_versions(self) -> list[str]:
        seen, out = set(), []
        for v in self.train_versions + self.test_versions:
            if v not in seen:
                out.append(v)
                seen.add(v)
        return out


DEPLOY_CONFIGS: dict[str, list[DeployConfig]] = {
    "full": [
        DeployConfig(
            name="validate",
            pretrain_versions=["v1", "v2"],
            finetune_versions=["v3"],
            test_versions=["v4"],
            description="pretrain v1→v2 · finetune v3 · test v4",
        ),
        DeployConfig(
            name="final",
            pretrain_versions=["v1", "v2"],
            finetune_versions=["v3", "v4"],
            test_versions=["v5"],
            description="pretrain v1→v2 · finetune v3+v4 · test v5",
        ),
    ],
    "pov": [
        DeployConfig(
            name="validate",
            pretrain_versions=["v2"],
            finetune_versions=["v3"],
            test_versions=["v4"],
            description="pretrain v2 · finetune v3 · test v4",
        ),
        DeployConfig(
            name="final",
            pretrain_versions=["v2"],
            finetune_versions=["v3", "v4"],
            test_versions=["v5"],
            description="pretrain v2 · finetune v3+v4 · test v5",
        ),
    ],
}


# ── load best hparams from the latest completed stage ────────────────────────

def load_best_config(results_dir: Path) -> tuple[dict, str]:
    """Return (cfg_dict, stage_name) from the most complete search stage."""
    for stage in ["training", "augmentation", "finetune", "arch", "backbone_context"]:
        try:
            cfg = _load_best(results_dir / stage)
            return cfg, stage
        except (FileNotFoundError, ValueError):
            continue
    raise RuntimeError(
        f"No hparam search results found under {results_dir}.\n"
        "Run the hparam search first."
    )


# ── helpers ──────────────────────────────────────────────────────────────────

def _check_embeddings(
    encoder: str, token: str, versions: list[str], frame_type: str,
) -> bool:
    if frame_type == "pov":
        return _pov_embeddings_available(encoder, token, versions)
    return _embeddings_available(encoder, token, versions)


def _load_dataset(
    version: str,
    encoder: str, token: str,
    k: int, mode: str,
    frame_type: str, dist_mode: str,
    n_val: int = 0,
) -> PPCIDataset:
    ds = PPCIDataset.from_disk(
        "ants", version, encoder, token,
        frame_type=frame_type, dist_mode=dist_mode,
        n_val_videos=n_val, **DS_KWARGS,
    )
    if k > 0:
        ds.apply_context_window(k, mode=mode)
    return ds


# ── per-version evaluation ──────────────────────────────────────────────────

def _eval_version(
    version: str,
    model,
    encoder: str, token: str,
    k: int, mode: str,
    frame_type: str, dist_mode: str,
    device: torch.device,
    out_dir: Path,
    outcomes: list[str],
    treatment_labels: dict | None = None,
) -> dict | None:
    """Evaluate one version. Returns metrics dict, None (no annotations), or {} (no embeddings)."""
    if not _check_embeddings(encoder, token, [version], frame_type):
        print(f"\n[{version}] Embeddings not available — skipping.")
        return {}

    print(f"\n[{version}] Loading for evaluation ...")
    ds = _load_dataset(version, encoder, token, k, mode, frame_type, dist_mode)

    raw_mean    = float(ds.X.mean())
    raw_std     = float(ds.X.std())
    n_zero_rows = int((ds.X.abs().sum(dim=1) == 0).sum())
    zero_pct    = 100.0 * n_zero_rows / len(ds.X)
    print(f"  embedding stats: mean={raw_mean:.3f}  std={raw_std:.3f}  "
          f"zero_rows={n_zero_rows:,} ({zero_pct:.1f}%)")
    if n_zero_rows > 0:
        print(f"  [ERROR] {version} has {n_zero_rows:,} all-zero rows ({zero_pct:.1f}%).")
    elif raw_std < 1.0:
        print(f"  [WARNING] {version} embeddings look corrupted (std={raw_std:.3f}).")

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
    print(f"  ✓ plot_summary saved: {plot_path}")

    if not has_annotations or n_annotated == 0:
        print(f"  [no annotations] Classification metrics skipped.")
        return None

    m = compute_metrics(model, ds.X, ds.Y, device)
    print(f"  → bacc={m.get('bacc', 0):.4f}  acc={m.get('acc', 0):.4f}"
          f"  recall={m.get('recall', 0):.4f}  precision={m.get('precision', 0):.4f}")
    del ds
    return {k: float(m.get(k, 0.0)) for k in ("acc", "bacc", "recall", "precision")}


def _run_evaluation(
    model,
    cfg: DeployConfig,
    encoder: str, token: str,
    k: int, mode: str,
    frame_type: str, dist_mode: str,
    device: torch.device,
    out_dir: Path,
) -> None:
    """Evaluate all versions, save metrics and comparison plots."""
    outcomes = [c.replace("Y_", "") for c in DS_KWARGS.get("outcome_cols", ["Y_Y2F", "Y_B2F"])]
    version_metrics: dict = {}

    for version in cfg.all_versions:
        m = _eval_version(
            version, model, encoder, token, k, mode,
            frame_type, dist_mode, device, out_dir, outcomes,
        )
        version_metrics[version] = m

    # Save per-version metrics
    metrics_out = {v: m for v, m in version_metrics.items() if m is not None and m != {}}
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"\n  ✓ Metrics saved: {out_dir / 'metrics.json'}")

    # Comparison plots
    available = {v: m for v, m in version_metrics.items() if m != {}}
    if not available:
        return

    for plot_fn, suffix in [
        (plot_comparison_versions, "plot_comparison_versions.png"),
        (plot_comparison_versions_by_metric, "plot_comparison_versions_by_metric.png"),
    ]:
        path = str(out_dir / suffix)
        plot_fn(
            available,
            train_versions=cfg.train_versions,
            pretrain_versions=cfg.pretrain_versions,
            test_versions=cfg.test_versions,
            save=True,
            save_path=path,
        )
        print(f"  ✓ {suffix} saved: {path}")


# ── single-config deployment ──────────────────────────────────────────────────

def deploy_one(
    cfg: DeployConfig,
    results_dir: Path,
    frame_type: str = "full",
    dry_run: bool = False,
    skip_if_exists: bool = False,
    eval_only: bool = False,
) -> None:
    """Train (or load) and evaluate one DeployConfig."""
    out_dir    = results_dir / "deploy" / cfg.name
    model_path = out_dir / "model.pt"

    cfg_dict, from_stage = load_best_config(results_dir)
    encoder   = cfg_dict["encoder"]
    token     = cfg_dict.get("token", "class")
    k         = int(cfg_dict.get("context_window", BASE_CFG.training.context_window))
    mode      = cfg_dict.get("context_mode", BASE_CFG.training.context_mode)
    dist_mode = cfg_dict.get("dist_mode", "none")
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pretrain_cfg = _build_pretrain_cfg(cfg_dict)
    finetune_cfg = _build_finetune_cfg(cfg_dict)

    print("=" * 72)
    print(f"[DEPLOY] {cfg.name} — {cfg.description}")
    print(f"  frame_type     : {frame_type}")
    print(f"  pretrain       : {' → '.join(cfg.pretrain_versions)}")
    print(f"  finetune       : {' + '.join(cfg.finetune_versions)}  (joint)")
    print(f"  test           : {', '.join(cfg.test_versions)}")
    print(f"  hparams from   : {from_stage} stage")
    print(f"  encoder        : {encoder}/{token}  k={k}  mode={mode}  dist_mode={dist_mode}")
    print(f"  arch           : dim={cfg_dict.get('hidden_dim', 512)}"
          f"  layers={cfg_dict.get('hidden_layers', 1)}"
          f"  dropout={cfg_dict.get('dropout', 0.3)}"
          f"  siamese={cfg_dict.get('siamese', True)}")
    print(f"  finetune       : method={cfg_dict.get('method', 'ERM')}"
          f"  lr={cfg_dict.get('finetune_lr', 1e-4):.0e}"
          f"  wd={cfg_dict.get('weight_decay', 0.001)}")
    print(f"  output         : {out_dir}")
    print("=" * 72)

    load_kw = dict(
        encoder=encoder, token=token, k=k, mode=mode,
        frame_type=frame_type, dist_mode=dist_mode,
    )

    # ── eval-only ─────────────────────────────────────────────────────────────
    if eval_only:
        if not model_path.exists():
            print(f"[ERROR] --eval-only requires {model_path}", file=sys.stderr)
            sys.exit(1)
        print(f"\n[eval-only] Loading model from {model_path}")
        ref_version = cfg.finetune_versions[-1]
        ds_tmp = _load_dataset(ref_version, **load_kw)
        model = build_model(ds_tmp, finetune_cfg)
        del ds_tmp
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        out_dir.mkdir(parents=True, exist_ok=True)
        _run_evaluation(model, cfg, **load_kw, device=device, out_dir=out_dir)
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
    if not _check_embeddings(encoder, token, required, frame_type):
        print(f"[ERROR] Embeddings missing for {encoder}/{token} "
              f"({', '.join(required)}).", file=sys.stderr)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── pretrain: sequential warm-start chain ─────────────────────────────────
    model = None
    for version in cfg.pretrain_versions:
        print(f"\n[pretrain] Loading {version} ...")
        ds = _load_dataset(version, **load_kw)
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
    ds_ft = PPCIDataset.concat([_load_dataset(v, **load_kw) for v in cfg.finetune_versions])
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
            "frame_type":          frame_type,
            "hparams_from_stage":  from_stage,
            "deploy_config":       cfg.name,
            "pretrain_versions":   cfg.pretrain_versions,
            "finetune_versions":   cfg.finetune_versions,
            "test_versions":       cfg.test_versions,
        }, f, indent=2)
    print(f"\n  ✓ Model saved: {model_path}")

    # ── evaluate and plot all versions ────────────────────────────────────────
    _run_evaluation(model, cfg, **load_kw, device=device, out_dir=out_dir)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deploy best PPCI model for one or all named configurations.",
    )
    parser.add_argument("--frame-type", choices=["full", "pov"], default="full",
                        help="Embedding view: 'full' (default) or 'pov'")
    parser.add_argument("--config", default=None, metavar="NAME",
                        help="Run only this config (validate / final). Default: all.")
    parser.add_argument("--results-dir", type=Path, default=None,
                        help="Hparam search results root (default: auto from frame-type)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-if-exists", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    if args.eval_only and args.skip_if_exists:
        parser.error("--eval-only and --skip-if-exists are mutually exclusive")

    results_dir = args.results_dir or (
        RESULTS_DIR_POV if args.frame_type == "pov" else RESULTS_DIR
    )

    configs = DEPLOY_CONFIGS[args.frame_type]
    configs_by_name = {c.name: c for c in configs}

    if args.config:
        if args.config not in configs_by_name:
            parser.error(f"Unknown config '{args.config}'. "
                         f"Choices: {', '.join(configs_by_name)}")
        configs = [configs_by_name[args.config]]

    for cfg in configs:
        deploy_one(
            cfg,
            results_dir,
            frame_type=args.frame_type,
            dry_run=args.dry_run,
            skip_if_exists=args.skip_if_exists,
            eval_only=args.eval_only,
        )


if __name__ == "__main__":
    main()
