"""
PPCI — Prediction-Powered Causal Inference

Public API
----------
PPCIDataset       — data container (embeddings + HF dataset + env/split logic)
MLP               — dual-head MLP (outcome head + auxiliary treatment head)
build_model       — convenience factory: MLP from PPCIDataset + config
train             — training entry point: ERM | DERM | vREx | IRM + W&B logging
compute_metrics   — accuracy / balanced accuracy
compute_ate_bias  — EAD-style ATE bias for causal evaluation

Quick-start
-----------
    from src.ppci import PPCIDataset, build_model, train
    from src.dataset.get_dataset import load_dataset
    from src.embedding.get_embeddings import load_embeddings_from_disk
    from omegaconf import OmegaConf

    # 1. Load data
    hf = load_dataset("ants", "v3", from_disk=True)
    emb = load_embeddings_from_disk("ants", "v3", "siglip", "class")

    # 2. Build dataset
    cfg = OmegaConf.load("configs/ppci/ants/config.yaml")
    ds = PPCIDataset(
        hf, emb,
        outcome_cols=list(cfg.outcome.columns),
        task=cfg.outcome.task,
        env_cols=list(cfg.causal.confounders),
        env_include_treatment=True,
        n_val_videos=cfg.validation.n_videos,
    )

    # 3. Train
    model = build_model(ds, cfg)
    best = train(ds, model, cfg)
"""

from .dataset import PPCIDataset, compute_Y
from .model import MLP
from .train import build_model, train, compute_metrics, compute_teb_average
from .evaluate import aggregate_probs, compute_ate, compute_teb, compute_teb_all_pairs
from .visualize import (
    plot_outcome_distribution, plot_outcome_distribution_ants,
    plot_ate_matrix, plot_po_barplot, plot_summary,
    plot_comparison,
)

__all__ = [
    "PPCIDataset",
    "compute_Y",
    "MLP",
    "build_model",
    "train",
    "compute_metrics",
    "compute_teb_average",
    "aggregate_probs",
    "compute_ate",
    "compute_teb",
    "compute_teb_all_pairs",
    "plot_outcome_distribution",
    "plot_outcome_distribution_ants",
    "plot_ate_matrix",
    "plot_po_barplot",
    "plot_summary",
    "plot_comparison",
]
