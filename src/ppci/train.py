"""PPCI training routines.

Supported methods
-----------------
ERM   — Empirical Risk Minimization (standard)
DERM  — Deconfounded ERM: per-sample reweighting by Var(Y|E) / P(Y,E)
vREx  — Variance Risk Extrapolation: mean_e(L^e) + λ · Var_e(L^e)
IRM   — Invariant Risk Minimisation: sum_e(L^e) + λ · sum_e(||∇_{w=1} L^e||²)

Usage
-----
    model = build_model(dataset, cfg)
    best_model = train(dataset, model, cfg)

    # Multi-dataset training
    merged = PPCIDataset.concat([ds_v3, ds_v4])
    best_model = train(merged, model, cfg)

W&B
---
If cfg.wandb.mode != "disabled", training progress and causal metrics are
logged via wandb.  Requires `pip install wandb`.
"""

from __future__ import annotations

import warnings
from copy import deepcopy
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from .dataset import PPCIDataset
from .model import MLP, SiameseMLP
from .evaluate import compute_teb


# ---------------------------------------------------------------------------
# Post-hoc temperature calibration
# ---------------------------------------------------------------------------

def _calibrate_temperature(
    model: MLP,
    X_val: torch.Tensor,
    Y_val: torch.Tensor,
    device: torch.device,
) -> float:
    """Find scalar temperature T minimising BCE(sigmoid(logit/T), Y) on the val set.

    Corrects systematic probability bias (over/under-estimation of APO) without
    retraining.  Sets model.temperature in-place and returns the value.

    T > 1 → softer predictions (down-scales logits, pushes probabilities toward
             0.5); fixes models that are overconfident.
    T < 1 → sharper predictions; fixes models that are underconfident, which is
             the typical cause of APO overestimation under class imbalance.
    """
    from scipy.optimize import minimize_scalar

    model.eval()
    batch_size = 4096
    logit_chunks = []
    with torch.no_grad():
        for start in range(0, len(X_val), batch_size):
            logit_chunks.append(model(X_val[start : start + batch_size].to(device)).cpu().float())
    logits = torch.cat(logit_chunks, dim=0)

    Y = Y_val.float()
    bce = nn.BCEWithLogitsLoss()

    def nll(log_T: float) -> float:
        return float(bce(logits / np.exp(log_T), Y))

    # Search in [exp(-3), exp(3)] ≈ [0.05, 20]
    res = minimize_scalar(nll, bounds=(-3.0, 3.0), method="bounded")
    T = float(np.exp(res.x))
    model.temperature = T
    return T


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def _make_loss_fn(
    Y_train: torch.Tensor,
    device: torch.device,
    reduction: str = "mean",
    use_pos_weight: bool = False,
) -> nn.Module:
    """BCEWithLogitsLoss, optionally with class-frequency pos_weight.

    Y_train shape: (N,) for single outcome or (N, k) for multiple outcomes.
    use_pos_weight=True weights the positive class by neg/pos to counter
    class imbalance, but tends to inflate predicted probabilities (high recall,
    low precision) and cause APO overestimation.  Disabled by default.
    """
    if use_pos_weight:
        Y = Y_train.float()
        pos = Y.sum(0).clamp(min=1.0)
        neg = (1 - Y).sum(0).clamp(min=1.0)
        pos_weight = (neg / pos).to(device)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction=reduction)
    return nn.BCEWithLogitsLoss(reduction=reduction)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _is_binary(Y_col: torch.Tensor) -> bool:
    """True if all values in a 1-D tensor are in {0, 1}."""
    u = Y_col.unique()
    return bool(u.numel() <= 2 and u.ge(0.0).all() and u.le(1.0).all()
                and ((u == 0) | (u == 1)).all())


def _col_metrics(yt: torch.Tensor, yh_bin: torch.Tensor, yh_prob: torch.Tensor) -> Dict[str, float]:
    """Per-column metrics, switching on binary vs continuous."""
    if _is_binary(yt):
        bacc, rec, pre, _ = _binary_stats(yt, yh_bin)
        return {
            "acc": float((yh_bin == yt).float().mean()),
            "bacc": bacc,
            "recall": rec,
            "precision": pre,
        }
    diff = yh_prob - yt
    return {
        "mse": float((diff ** 2).mean()),
        "mae": float(diff.abs().mean()),
    }


def compute_metrics(
    model: MLP,
    X: torch.Tensor,
    Y: torch.Tensor,
    device: torch.device,
    batch_size: int = 4096,
) -> Dict[str, float]:
    """Compute metrics per outcome column, then average.

    Binary columns  → acc, bacc, recall, precision.
    Continuous cols → mse, mae.
    Uses batched GPU inference to avoid OOM with large embeddings.
    """
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            chunks.append(model.probs(X[start : start + batch_size].to(device)).cpu())
    yh_prob = torch.cat(chunks, dim=0)
    yh_bin  = yh_prob.round()

    Y = Y.float()
    if Y.dim() == 1:
        return _col_metrics(Y, yh_bin, yh_prob)

    # Multi-column: aggregate per metric key (binary cols share one set, continuous another)
    col_results = [_col_metrics(Y[:, k], yh_bin[:, k], yh_prob[:, k]) for k in range(Y.shape[1])]
    all_keys = {k for r in col_results for k in r}
    result: Dict[str, float] = {}
    for key in all_keys:
        vals = [r[key] for r in col_results if key in r]
        result[key] = float(np.mean(vals))
    # Per-column breakdown
    for k, r in enumerate(col_results):
        for key, val in r.items():
            result[f"{key}_{k}"] = val
    return result


def _binary_stats(Y: torch.Tensor, Y_hat: torch.Tensor) -> tuple[float, float, float, float]:
    """Returns (bacc, recall, precision, specificity)."""
    tp = float(((Y == 1) & (Y_hat == 1)).sum())
    fn = float(((Y == 1) & (Y_hat != 1)).sum())
    fp = float(((Y != 1) & (Y_hat == 1)).sum())
    tn = float(((Y != 1) & (Y_hat != 1)).sum())
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    pre = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    return (rec + spe) / 2, rec, pre, spe


def _binary_bacc(Y: torch.Tensor, Y_hat: torch.Tensor) -> float:
    return _binary_stats(Y, Y_hat)[0]


# ---------------------------------------------------------------------------
# TEB helper (for W&B causal metrics)
# ---------------------------------------------------------------------------

def compute_teb_average(
    model: MLP,
    dataset: PPCIDataset,
    device: torch.device,
    task: Optional[str] = None,
) -> Dict[str, float]:
    """Observation-level TEB (Treatment Effect Bias) averaged over all treatment pairs.

    Enumerates every unique pair of treatment groups present in the dataset,
    computes EAD-based TEB for each (observation-level, mean aggregation),
    and averages all metrics across pairs.  This gives a single unambiguous
    scalar per outcome for W&B tracking regardless of how many treatment arms
    are present.

    Returns {} if fewer than two treatment groups are present.
    """
    eval_task = task if task in ("or", "sum") else None

    t_vals = sorted(dataset.T.unique().tolist())
    pairs = list(combinations(t_vals, 2))
    if not pairs:
        return {}

    accumulated: Dict[str, List[float]] = {}
    for t0, t1 in pairs:
        pair = compute_teb(
            model, dataset, device,
            T_control=t0,
            T_treatment=t1,
            eval_task=eval_task,
        )
        for k, v in pair.items():
            accumulated.setdefault(k, []).append(v)

    return {
        k: float(np.mean(np.abs(vs)) if k.endswith("/bias") else np.mean(vs))
        for k, vs in accumulated.items()
    }


# ---------------------------------------------------------------------------
# Embedding augmentation
# ---------------------------------------------------------------------------

def _augment_batch(
    X: torch.Tensor,
    Y: torch.Tensor,
    noise_std: float = 0.0,
    mixup_alpha: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply embedding-space augmentations to a batch.

    Since embeddings are pre-extracted (not raw images), geometric transforms
    like rotation/mirroring are approximated in feature space:

    - noise_std > 0: isotropic Gaussian noise on each embedding.
    - mixup_alpha > 0: Mixup (Zhang et al. 2018) — convex interpolation of two
      randomly-paired samples using a Beta(α, α) mixing coefficient.
      alpha=0.2 is a common starting point.
    """
    if noise_std > 0.0:
        X = X + torch.randn_like(X) * noise_std

    if mixup_alpha > 0.0:
        lam = float(np.random.beta(mixup_alpha, mixup_alpha))
        perm = torch.randperm(X.size(0), device=X.device)
        X = lam * X + (1.0 - lam) * X[perm]
        Y = lam * Y + (1.0 - lam) * Y[perm]

    return X, Y


# ---------------------------------------------------------------------------
# IRM penalty
# ---------------------------------------------------------------------------

def _irm_penalty(
    logits: torch.Tensor,
    Y: torch.Tensor,
    loss_fn_nored: nn.Module,
) -> torch.Tensor:
    """IRM gradient penalty for one environment.

    Computes ||∇_{w=1} L(w · f(x), y)||²  where w is a dummy scalar.
    """
    scale = torch.ones(1, dtype=logits.dtype, device=logits.device, requires_grad=True)
    loss = loss_fn_nored(logits * scale, Y).mean()
    (grad,) = torch.autograd.grad(loss, [scale], create_graph=True)
    return grad.pow(2).sum()


# ---------------------------------------------------------------------------
# Build model helper
# ---------------------------------------------------------------------------

def build_model(dataset: PPCIDataset, cfg: DictConfig) -> MLP:
    """Instantiate an MLP from a PPCIDataset and config.

    Applies temporal context window to the dataset (mutates dataset.X in-place)
    before reading the input dimension.  When ``context_mode='concat'`` the MLP
    receives a :class:`TemporalAttention` front-end that learns to aggregate the
    window rather than relying on a large first layer.

    One output neuron per outcome column; aggregation (or/sum) is eval-only.
    """
    context_window   = int(cfg.training.get("context_window", 0))
    context_mode     = str(cfg.training.get("context_mode", "concat"))
    if getattr(dataset, "_context_window_applied", 0) == 0:
        dataset.apply_context_window(context_window, mode=context_mode)

    context_size     = getattr(dataset, "_context_size", 1)
    context_head_dim = int(cfg.mlp.get("context_head_dim", 64))

    shared_kwargs = dict(
        input_dim=dataset.X.shape[1],
        hidden_dim=cfg.mlp.hidden_dim,
        hidden_layers=cfg.mlp.hidden_layers,
        dropout=cfg.mlp.get("dropout", 0.0),
        context_size=context_size,
        context_head_dim=context_head_dim,
    )

    if getattr(dataset, "frame_type", "full") == "pov":
        return SiameseMLP(
            **shared_kwargs,
            siamese=bool(cfg.mlp.get("siamese", True)),
            n_dist=getattr(dataset, "n_dist", 0),
        )

    return MLP(
        n_outcomes=len(dataset.outcome_cols),
        n_dist=getattr(dataset, "n_dist", 0),
        **shared_kwargs,
    )


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def train(
    dataset: PPCIDataset,
    model: MLP,
    cfg: DictConfig,
    test_dataset: Optional[PPCIDataset] = None,
) -> MLP:
    """Train an MLP on a PPCIDataset.

    When ``cfg.training.auto_fix`` is True (default), automatically:
    - Retries with LR ÷ 3 (or × 3) if training diverges or barely learns (up to 2 retries).
    - Extends training by 50 % more epochs from the best checkpoint if not yet converged.

    This ensures every config gets a fair best-effort comparison in the hparam search.

    Returns:
        Best model (by val balanced accuracy, or last epoch if no val set).
    """
    auto_fix = bool(cfg.training.get("auto_fix", True))
    method   = cfg.training.method
    valid_methods = {"ERM", "DERM", "vREx", "IRM"}
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}, got '{method}'")

    device = torch.device(
        cfg.training.get("device", "cuda") if torch.cuda.is_available() else "cpu"
    )
    model = model.to(device)
    run   = _init_wandb(cfg, method)

    # Save initial weights for LR-retry resets
    initial_weights = deepcopy(model.state_dict()) if auto_fix else None
    working_cfg     = cfg
    best: Optional[MLP] = None

    # ── LR retry loop (original + up to 2 retries) ────────────────────────────
    for attempt in range(3):
        if attempt > 0 and initial_weights is not None:
            model.load_state_dict(deepcopy(initial_weights))
            model = model.to(device)

        _fn = _train_flat if method in ("ERM", "DERM") else _train_env_aware
        best = _fn(dataset, model, working_cfg, method, device, run)
        diag = getattr(best, "training_diagnostics", {})

        if not auto_fix or attempt == 2:
            break

        orig_lr = float(working_cfg.training.lr)
        if not diag.get("loss_ok", True) or diag.get("lr_too_high", False):
            new_lr = orig_lr / 3.0
            print(f"  [auto-fix] LR too high ({orig_lr:.1e}) → retry with lr={new_lr:.1e}")
        elif diag.get("lr_too_low", False):
            new_lr = orig_lr * 3.0
            print(f"  [auto-fix] LR too low ({orig_lr:.1e}) → retry with lr={new_lr:.1e}")
        else:
            break  # training is healthy — no retry needed

        # Make a fresh mutable copy so the caller's cfg is never mutated
        working_cfg = OmegaConf.create(OmegaConf.to_container(working_cfg, resolve=True))
        working_cfg.training.lr = new_lr

    # ── Epoch extension if not converged ──────────────────────────────────────
    diag = getattr(best, "training_diagnostics", {})
    if auto_fix and not diag.get("converged", True):
        orig_n = int(working_cfg.training.num_epochs)
        extra  = orig_n // 2
        print(f"  [auto-fix] not converged (best_epoch={diag.get('best_epoch')}/{orig_n})"
              f" → +{extra} epochs (warm start from best model)")
        ext_cfg = OmegaConf.create(OmegaConf.to_container(working_cfg, resolve=True))
        ext_cfg.training.num_epochs = extra
        _fn = _train_flat if method in ("ERM", "DERM") else _train_env_aware
        extended = _fn(dataset, best, ext_cfg, method, device, run=None)
        ext_val  = getattr(extended, "training_diagnostics", {}).get("val_bacc_at_best") or 0.0
        orig_val = diag.get("val_bacc_at_best") or 0.0
        if ext_val >= orig_val:
            best = extended
            print(f"  [auto-fix] extended model kept: val_bacc {orig_val:.4f} → {ext_val:.4f}")
        else:
            print(f"  [auto-fix] original kept: val_bacc {orig_val:.4f} (extended {ext_val:.4f})")

    # ── Finalize ──────────────────────────────────────────────────────────────
    if test_dataset is not None and run is not None:
        _log_test_metrics(best, test_dataset, working_cfg, device, run)
    if run is not None:
        run.finish()

    return best


# ---------------------------------------------------------------------------
# Training diagnostics
# ---------------------------------------------------------------------------

def _compute_train_diagnostics(
    history: List[Dict[str, float]],
    best_epoch: int,
    num_epochs: int,
) -> Dict:
    """Summarise training health from the per-epoch history.

    Checks
    ------
    converged      best_epoch < 90% of num_epochs (still improving at end = needs more epochs)
    loss_ok        no NaN / Inf in loss
    lr_too_high    loss at epoch 5 > 1.5× loss at epoch 1  (diverging early)
    lr_too_low     relative loss drop over all epochs < 5%
    overfit_gap    train_bacc - val_bacc at best epoch  (>0.15 = overfitting)
    """
    if not history:
        return {}

    losses   = [h.get("train/loss", float("nan")) for h in history]
    tr_baccs = [h.get("train/bacc", float("nan")) for h in history]
    va_baccs = [h.get("val/bacc",   float("nan")) for h in history]
    has_val  = any(np.isfinite(v) for v in va_baccs)

    loss_ok    = all(np.isfinite(l) for l in losses)
    finite_losses = [l for l in losses if np.isfinite(l)]

    lr_too_high = False
    if len(finite_losses) >= 5:
        lr_too_high = finite_losses[4] > finite_losses[0] * 1.5

    loss_improvement = 0.0
    if finite_losses and finite_losses[0] > 0:
        loss_improvement = (finite_losses[0] - finite_losses[-1]) / finite_losses[0]
    lr_too_low = loss_improvement < 0.05

    best_idx          = best_epoch - 1
    train_at_best     = tr_baccs[best_idx] if 0 <= best_idx < len(tr_baccs) else float("nan")
    val_at_best       = va_baccs[best_idx] if (has_val and 0 <= best_idx < len(va_baccs)) else float("nan")
    overfit_gap       = (train_at_best - val_at_best) if (np.isfinite(train_at_best) and np.isfinite(val_at_best)) else float("nan")

    return {
        "n_epochs":           num_epochs,
        "best_epoch":         best_epoch,
        "converged":          best_epoch < 0.9 * num_epochs,
        "loss_ok":            loss_ok,
        "lr_too_high":        lr_too_high,
        "lr_too_low":         lr_too_low,
        "loss_improvement":   round(loss_improvement, 4),
        "train_bacc_at_best": round(train_at_best, 4) if np.isfinite(train_at_best) else None,
        "val_bacc_at_best":   round(val_at_best,   4) if np.isfinite(val_at_best)   else None,
        "overfit_gap":        round(overfit_gap,   4) if np.isfinite(overfit_gap)   else None,
    }


# ---------------------------------------------------------------------------
# ERM / DERM (flat dataloader)
# ---------------------------------------------------------------------------

def _train_flat(
    dataset: PPCIDataset,
    model: MLP,
    cfg: DictConfig,
    method: str,
    device: torch.device,
    run,
) -> MLP:
    batch_size = cfg.training.batch_size
    num_epochs = cfg.training.num_epochs
    lr = cfg.training.lr
    has_val = int(dataset.val_mask.sum()) > 0

    use_pos_weight = bool(cfg.training.get("pos_weight", False))
    frame_stride = int(cfg.training.get("frame_stride", 1))
    noise_std = float(cfg.training.get("augment_noise_std", 0.0))
    mixup_alpha = float(cfg.training.get("augment_mixup_alpha", 0.0))
    loss_fn = _make_loss_fn(dataset.Y_train, device, use_pos_weight=use_pos_weight)
    loss_fn_nored = _make_loss_fn(dataset.Y_train, device, reduction="none", use_pos_weight=use_pos_weight)

    weights = dataset.compute_derm_weights() if method == "DERM" else None
    loader = dataset.get_train_loader(batch_size, weights=weights, frame_stride=frame_stride)
    weight_decay = float(cfg.training.get("weight_decay", 0.0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Keep eval tensors on CPU; compute_metrics uses batched GPU inference
    X_train_dev = dataset.X_train
    X_val_dev = dataset.X_val if has_val else None
    # Subsample up to 10k frames for cheap per-epoch train metrics
    _n_tr = len(dataset.X_train)
    _rng = np.random.default_rng(0)
    train_idx = torch.from_numpy(
        _rng.choice(_n_tr, size=min(10_000, _n_tr), replace=False)
    )

    best_val_bacc = -1.0
    best_model = deepcopy(model)
    history: List[Dict[str, float]] = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in loader:
            if weights is not None:
                X_b, Y_b, E_b, w_b = [t.to(device) for t in batch]
            else:
                X_b, Y_b, E_b = [t.to(device) for t in batch]
                w_b = None

            Y_b = Y_b.float()
            X_b, Y_b = _augment_batch(X_b, Y_b, noise_std=noise_std, mixup_alpha=mixup_alpha)
            optimizer.zero_grad()

            logits = model(X_b)   # (N, k) or (N,) — matches Y_b shape

            if w_b is not None:
                # DERM: per-sample weighted loss
                loss_per_sample = loss_fn_nored(logits, Y_b)
                if loss_per_sample.dim() > 1:
                    loss_per_sample = loss_per_sample.mean(dim=-1)
                loss = (loss_per_sample * w_b).mean()
            else:
                loss = loss_fn(logits, Y_b)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        metrics = _epoch_metrics(model, dataset, device, epoch_loss / n_batches, has_val,
                                 X_train_dev, X_val_dev, train_idx, cfg)
        history.append(metrics)
        if run is not None:
            _log_metrics(run, metrics, cfg, dataset, model, device, epoch)
        _print_epoch(epoch, metrics)

        val_bacc = metrics.get("val/bacc", metrics.get("train/bacc", 0.0))
        if val_bacc >= best_val_bacc:
            best_val_bacc = val_bacc
            best_model = deepcopy(model)
            best_model.best_epoch = epoch

    if has_val and X_val_dev is not None and cfg.training.get("calibrate_temperature", True):
        T = _calibrate_temperature(best_model, X_val_dev, dataset.Y_val, device)
        print(f"  temperature calibration → T={T:.4f}")

    best_model.training_diagnostics = _compute_train_diagnostics(
        history, getattr(best_model, "best_epoch", num_epochs), num_epochs
    )
    return best_model


# ---------------------------------------------------------------------------
# vREx / IRM (per-environment dataloaders)
# ---------------------------------------------------------------------------

def _train_env_aware(
    dataset: PPCIDataset,
    model: MLP,
    cfg: DictConfig,
    method: str,
    device: torch.device,
    run,
) -> MLP:
    batch_size = cfg.training.batch_size
    num_epochs = cfg.training.num_epochs
    lr = cfg.training.lr
    ic_weight = cfg.training.get("ic_weight", 1.0)
    has_val = int(dataset.val_mask.sum()) > 0

    use_pos_weight = bool(cfg.training.get("pos_weight", False))
    frame_stride = int(cfg.training.get("frame_stride", 1))
    noise_std = float(cfg.training.get("augment_noise_std", 0.0))
    mixup_alpha = float(cfg.training.get("augment_mixup_alpha", 0.0))
    env_loaders = dataset.get_env_train_loaders(batch_size, frame_stride=frame_stride)
    n_envs = len(env_loaders)

    if n_envs < 2:
        warnings.warn(
            f"[{method}] Only {n_envs} training environment found. "
            "Environment-aware training degenerates to ERM.",
            stacklevel=2,
        )

    loss_fn = _make_loss_fn(dataset.Y_train, device, use_pos_weight=use_pos_weight)
    loss_fn_nored = _make_loss_fn(dataset.Y_train, device, reduction="none", use_pos_weight=use_pos_weight)
    weight_decay = float(cfg.training.get("weight_decay", 0.0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Keep eval tensors on CPU; compute_metrics uses batched GPU inference
    X_train_dev = dataset.X_train
    X_val_dev = dataset.X_val if has_val else None
    _n_tr = len(dataset.X_train)
    _rng = np.random.default_rng(0)
    train_idx = torch.from_numpy(
        _rng.choice(_n_tr, size=min(10_000, _n_tr), replace=False)
    )

    best_val_bacc = -1.0
    best_model = deepcopy(model)
    history: List[Dict[str, float]] = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0

        # Iterate over the minimum number of batches across all environments
        env_iters = [iter(loader) for loader in env_loaders]
        n_steps = min(len(loader) for loader in env_loaders)

        for _ in range(n_steps):
            optimizer.zero_grad()

            # --- collect one batch per environment ---
            env_batches: List[Tuple[torch.Tensor, torch.Tensor]] = []
            for env_iter in env_iters:
                X_b, Y_b = [t.to(device) for t in next(env_iter)]
                X_b, Y_b = _augment_batch(X_b, Y_b.float(), noise_std=noise_std, mixup_alpha=mixup_alpha)
                env_batches.append((X_b, Y_b))

            # --- compute per-environment losses (and IRM penalties if needed) ---
            env_losses: List[torch.Tensor] = []
            irm_penalties: List[torch.Tensor] = []

            for X_b, Y_b in env_batches:
                logits = model(X_b)   # (N, k) or (N,)
                env_loss = loss_fn(logits, Y_b)
                env_losses.append(env_loss)

                if method == "IRM":
                    # Penalty: ||∇_{w=1} L(w·logits, Y)||²
                    irm_penalties.append(
                        _irm_penalty(logits.detach().requires_grad_(True), Y_b, loss_fn_nored)
                    )

            stacked = torch.stack(env_losses)

            if method == "vREx":
                # vREx: mean + λ · Var across environments
                loss = stacked.mean() + ic_weight * stacked.var()
            else:
                # IRM: sum + λ · sum of gradient penalties
                penalty = torch.stack(irm_penalties).sum()
                loss = stacked.sum() + ic_weight * penalty

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        n_steps = max(n_steps, 1)
        metrics = _epoch_metrics(model, dataset, device, epoch_loss / n_steps, has_val,
                                 X_train_dev, X_val_dev, train_idx, cfg)
        history.append(metrics)
        if run is not None:
            _log_metrics(run, metrics, cfg, dataset, model, device, epoch)
        _print_epoch(epoch, metrics)

        val_bacc = metrics.get("val/bacc", metrics.get("train/bacc", 0.0))
        if val_bacc >= best_val_bacc:
            best_val_bacc = val_bacc
            best_model = deepcopy(model)
            best_model.best_epoch = epoch

    if has_val and X_val_dev is not None and cfg.training.get("calibrate_temperature", True):
        T = _calibrate_temperature(best_model, X_val_dev, dataset.Y_val, device)
        print(f"  temperature calibration → T={T:.4f}")

    best_model.training_diagnostics = _compute_train_diagnostics(
        history, getattr(best_model, "best_epoch", num_epochs), num_epochs
    )
    return best_model


# ---------------------------------------------------------------------------
# Epoch metrics + logging helpers
# ---------------------------------------------------------------------------

def _epoch_metrics(
    model: MLP,
    dataset: PPCIDataset,
    device: torch.device,
    train_loss: float,
    has_val: bool,
    X_train_dev: torch.Tensor,
    X_val_dev: Optional[torch.Tensor],
    train_idx: torch.Tensor,
    cfg: Optional[DictConfig] = None,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {"train/loss": train_loss}
    tr = compute_metrics(model, X_train_dev[train_idx], dataset.Y_train[train_idx], device)
    for key, val in tr.items():
        if not key.rsplit("_", 1)[-1].isdigit():  # skip per-column keys (acc_0, mse_1, …)
            metrics[f"train/{key}"] = val
    if has_val and X_val_dev is not None:
        va = compute_metrics(model, X_val_dev, dataset.Y_val, device)
        for key, val in va.items():
            if "_" not in key.lstrip("_"):
                metrics[f"val/{key}"] = val
    if cfg is not None and cfg.get("log_teb", False):
        eval_task = cfg.get("outcome", {}).get("task", None)
        teb = compute_teb_average(model, dataset, device, task=eval_task)
        for k, v in teb.items():
            metrics[f"causal/{k}"] = v
    return metrics


def _log_metrics(
    run,
    metrics: Dict[str, float],
    cfg: DictConfig,
    dataset: PPCIDataset,
    model: MLP,
    device: torch.device,
    epoch: int,
):
    """Log metrics + causal bias to W&B."""
    payload = dict(metrics, epoch=epoch)

    # Compute TEB if not already in metrics (i.e. log_teb=False skipped it above)
    if not any(k.startswith("causal/") for k in metrics):
        eval_task = cfg.get("outcome", {}).get("task", None)
        ate_metrics = compute_teb_average(model, dataset, device, task=eval_task)
        for k, v in ate_metrics.items():
            payload[f"causal/{k}"] = v

    run.log(payload)


def _log_test_metrics(
    model: MLP,
    test_dataset: PPCIDataset,
    cfg: DictConfig,
    device: torch.device,
    run,
):
    m = compute_metrics(model, test_dataset.X, test_dataset.Y, device)
    eval_task = cfg.get("outcome", {}).get("task", None)
    ate = compute_teb_average(model, test_dataset, device, task=eval_task)
    payload = {f"test/{k}": v for k, v in m.items()}
    payload.update({f"test/causal/{k}": v for k, v in ate.items()})
    run.log(payload)


def _print_epoch(epoch: int, metrics: Dict[str, float]):
    parts = [f"Epoch {epoch:3d}"]
    # Print loss first, then train metrics, then val metrics (stable order)
    _ORDER = ("loss", "acc", "bacc", "recall", "precision", "mse", "mae")
    for prefix in ("train", "val"):
        for stat in _ORDER:
            k = f"{prefix}/{stat}"
            if k in metrics:
                label = k.replace("/", "_")
                suffix = "~" if prefix == "train" and stat != "loss" else ""
                parts.append(f"{label}={metrics[k]:.3f}{suffix}")
    for k in sorted(metrics):
        if k.startswith("causal/"):
            label = k[len("causal/"):]
            parts.append(f"teb_{label}={metrics[k]:.3f}")
    print("  ".join(parts))


# ---------------------------------------------------------------------------
# W&B init
# ---------------------------------------------------------------------------

def _init_wandb(cfg: DictConfig, method: str):
    """Initialise a W&B run, or return None if disabled / unavailable."""
    wb_cfg = cfg.get("wandb", {})
    mode = wb_cfg.get("mode", "disabled")
    if mode == "disabled":
        return None
    try:
        import wandb
    except ImportError:
        warnings.warn("[PPCI] wandb not installed. Skipping W&B logging.", stacklevel=3)
        return None

    run_name = wb_cfg.get("run_name", None) or method
    run = wandb.init(
        project=wb_cfg.get("project", "ppci"),
        entity=wb_cfg.get("entity", None),
        group=wb_cfg.get("group", None),
        tags=list(wb_cfg.get("tags", [])),
        name=run_name,
        mode=mode,
        config={
            "method": method,
            "mlp": dict(cfg.mlp),
            "training": dict(cfg.training),
            "causal": dict(cfg.get("causal", {})),
        },
    )
    return run
