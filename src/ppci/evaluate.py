"""Observation-level causal evaluation for PPCI.

The PPCI workflow:
  1. Frame-level predictions (N_frames, n_outcomes) from the MLP
  2. Aggregate to observation level by averaging frames within each video
  3. Estimate ATE using classical / doubly-robust / tree-based estimators
  4. Bias = ATE(Y_hat) - ATE(Y_true)

Usage
-----
    from src.ppci.evaluate import aggregate_to_observations, compute_teb

    obs_df = aggregate_to_observations(model, dataset, device)

    # Full evaluation: multiple outcomes × methods
    bias = compute_teb(
        model, dataset, device,
        T_control=0, T_treatment=1,
        methods=["ead", "aipw"],
        eval_task="or",      # also compute for the 'or' aggregation
    )

Available ATE methods
---------------------
    ead         — naive difference in means  E[Y|T=1] - E[Y|T=0]
    aipw        — AIPW (XGBoost outcome + Logistic propensity)
    slearner    — S-Learner          (econml)
    tlearner    — T-Learner          (econml)
    xlearner    — X-Learner          (econml)
    drlearner   — DR-Learner         (econml)
    causalforest— CausalForestDML    (econml)
"""

from __future__ import annotations

import warnings
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import ttest_1samp

from .dataset import PPCIDataset
from .model import MLP


# ---------------------------------------------------------------------------
# Aggregation helper (used here and re-exported for external callers)
# ---------------------------------------------------------------------------

def aggregate_probs(probs: torch.Tensor, task) -> torch.Tensor:
    """Aggregate per-outcome probabilities for evaluation.

    Args:
        probs: (N, k) or (N,) probability tensor from model.probs().
        task:  None / "multilabel" → identity (return as-is)
               "or"  → P(any positive) = 1 - prod(1 - p_k)
               "sum" → expected count  = sum(p_k)

    Returns:
        (N, k) for multilabel/None, or (N,) scalar for or/sum.
    """
    if probs.dim() == 1 or task in (None, "multilabel"):
        return probs
    if task == "or":
        return 1.0 - (1.0 - probs).prod(dim=-1)
    if task == "sum":
        return probs.sum(dim=-1)
    raise ValueError(f"Unknown aggregation task '{task}'. Use: or, sum, multilabel, None")


# ---------------------------------------------------------------------------
# Step 1: aggregate frame-level data to observation level
# ---------------------------------------------------------------------------

def aggregate_to_observations(
    model: MLP,
    dataset: PPCIDataset,
    device: torch.device,
    agg: str = "mean",
    eval_task: Optional[str] = None,
) -> pd.DataFrame:
    """Average frame-level predictions and ground truth to one row per video.

    Args:
        model:     Trained MLP.
        dataset:   PPCIDataset (uses ALL frames, not just train/val split).
        device:    Torch device.
        agg:       How to aggregate frames within each observation: "mean" | "sum".
        eval_task: Optional cross-outcome aggregation for an extra pair of columns:
                   "or"  → P(any positive)   / any(Y>0)
                   "sum" → expected count     / total count

    Returns:
        DataFrame with one row per observation_id and columns:
          T             — treatment class index (constant within observation)
          Y_{label}     — true outcome, aggregated over frames
          Yhat_{label}  — predicted outcome, aggregated over frames
          (+ Y_{task}/Yhat_{task} when eval_task is set)
          W_*           — covariates (first value within each observation)
    """
    model.eval()
    with torch.no_grad():
        probs = model.probs(dataset.X.to(device)).cpu()   # (N, k) or (N,)

    Y = dataset.Y.float()

    # Build per-frame dict
    rows: dict = {
        "observation_id": dataset.obs_ids,
        "T": dataset.T,
    }

    # Per-column true Y and Y_hat
    if probs.dim() == 1:
        labels = [dataset.outcome_cols[0].replace("Y_", "")]
        rows[f"Y_{labels[0]}"]    = Y.numpy()
        rows[f"Yhat_{labels[0]}"] = probs.numpy()
    else:
        labels = [col.replace("Y_", "") for col in dataset.outcome_cols]
        for k, lbl in enumerate(labels):
            rows[f"Y_{lbl}"]    = Y[:, k].numpy()
            rows[f"Yhat_{lbl}"] = probs[:, k].numpy()

    # Auto-add mean across all outcome columns for multilabel (frame-level, before groupby)
    if probs.dim() == 2 and probs.shape[1] > 1:
        rows["Y_avg"]    = Y.mean(dim=-1).numpy()
        rows["Yhat_avg"] = probs.mean(dim=-1).numpy()

    # Optional cross-outcome aggregation
    if eval_task in ("or", "sum") and probs.dim() == 2:
        agg_probs_col = aggregate_probs(probs, eval_task).numpy()
        agg_Y_col     = aggregate_probs(Y, eval_task).numpy()
        rows[f"Y_{eval_task}"]    = agg_Y_col
        rows[f"Yhat_{eval_task}"] = agg_probs_col

    # Covariates
    if dataset.W.shape[1] > 0:
        for i, col in enumerate(dataset.W_cols):
            rows[col] = dataset.W[:, i].numpy()

    df_frames = pd.DataFrame(rows)

    # Aggregate Y/Yhat columns over frames
    agg_fn   = "mean" if agg == "mean" else "sum"
    y_cols   = [c for c in df_frames.columns if c.startswith("Y_") or c.startswith("Yhat_")]
    w_cols   = [c for c in df_frames.columns if c.startswith("W_")]

    obs_y    = df_frames.groupby("observation_id")[y_cols].agg(agg_fn).reset_index()
    obs_meta = df_frames.groupby("observation_id")[["T"] + w_cols].first().reset_index()

    result = obs_y.merge(obs_meta, on="observation_id")
    result["_has_annotations"] = dataset.has_annotations
    return result


# ---------------------------------------------------------------------------
# Step 2: ATE estimation at observation level
# ---------------------------------------------------------------------------

def compute_ate(
    obs_df: pd.DataFrame,
    Y_col: str,
    T_control: int = 0,
    T_treatment: int = 1,
    W_cols: Optional[List[str]] = None,
    method: str = "ead",
) -> Tuple[float, float, float]:
    """Estimate ATE from observation-level data.

    Binary contrast: rows with T == T_control or T == T_treatment are used.

    Args:
        obs_df:      DataFrame (one row per observation) with columns T, Y_col, W_*.
        Y_col:       Outcome column (can be true Y or predicted Yhat).
        T_control:   Control treatment class index.
        T_treatment: Treatment class index.
        W_cols:      Covariate column names (optional; used by AIPW/econml).
        method:      Estimator: ead | aipw | slearner | tlearner | xlearner |
                     drlearner | causalforest

    Returns:
        (ate, std, p_value)
    """
    df = obs_df[obs_df["T"].isin([T_control, T_treatment])].copy()
    if len(df) == 0:
        return float("nan"), float("nan"), float("nan")

    df["_T"] = (df["T"] == T_treatment).astype(int)
    Y = df[Y_col].values.astype(float)
    T = df["_T"].values.astype(int)

    if W_cols:
        available = [c for c in W_cols if c in df.columns]
        W = df[available].values.astype(float) if available else np.ones((len(Y), 1))
    else:
        W = np.ones((len(Y), 1))

    dispatch = {
        "ead":         _ate_ead,
        "aipw":        _ate_aipw,
        "slearner":    lambda y, t, w: _ate_econml(y, t, w, "slearner"),
        "tlearner":    lambda y, t, w: _ate_econml(y, t, w, "tlearner"),
        "xlearner":    lambda y, t, w: _ate_econml(y, t, w, "xlearner"),
        "drlearner":   lambda y, t, w: _ate_econml(y, t, w, "drlearner"),
        "causalforest": lambda y, t, w: _ate_econml(y, t, w, "causalforest"),
    }
    if method not in dispatch:
        raise ValueError(f"Unknown method '{method}'. Choose from: {list(dispatch)}")
    return dispatch[method](Y, T, W)


# ---------------------------------------------------------------------------
# Step 3: TEB (Treatment Effect Bias) evaluation
# ---------------------------------------------------------------------------

def compute_teb(
    model: MLP,
    dataset: PPCIDataset,
    device: torch.device,
    T_control: int = 0,
    T_treatment: int = 1,
    methods: List[str] = ["ead"],
    agg: str = "mean",
    eval_task: Optional[str] = None,
) -> Dict[str, float]:
    """Observation-level TEB for all outcome columns and all estimators.

    For each (outcome, method) pair, computes ATE on both true Y and predicted
    Yhat, reporting:
      {label}/{method}/ate_true   — ATE estimated from ground-truth labels
      {label}/{method}/ate_pred   — ATE estimated from model predictions
      {label}/{method}/bias       — ate_pred − ate_true
      {label}/{method}/ate_true_std
      {label}/{method}/ate_pred_std

    Args:
        model, dataset, device: standard arguments
        T_control / T_treatment: treatment class indices to compare
        methods:    list of estimator names (see compute_ate for options)
        agg:        frame aggregation: "mean" | "sum"
        eval_task:  optional cross-outcome aggregation: "or" | "sum" | None

    Returns:
        Flat dict of all metrics.
    """
    obs_df = aggregate_to_observations(model, dataset, device, agg=agg, eval_task=eval_task)

    # Build pairs (label, Y_col, Yhat_col)
    y_true_cols = [c for c in obs_df.columns
                   if c.startswith("Y_") and not c.startswith("Yhat_")]
    w_cols = [c for c in obs_df.columns if c.startswith("W_")]

    result: Dict[str, float] = {}
    for y_col in y_true_cols:
        label    = y_col[len("Y_"):]          # "Y2F", "B2F", "or", etc.
        yhat_col = f"Yhat_{label}"
        if yhat_col not in obs_df.columns:
            continue

        for method in methods:
            try:
                ate_t, std_t, _ = compute_ate(
                    obs_df, y_col, T_control, T_treatment, w_cols or None, method
                )
                ate_p, std_p, _ = compute_ate(
                    obs_df, yhat_col, T_control, T_treatment, w_cols or None, method
                )
            except Exception as exc:
                warnings.warn(f"[compute_teb] {label}/{method} failed: {exc}")
                continue

            pfx = f"{label}/{method}"
            result[f"{pfx}/ate_true"]     = ate_t
            result[f"{pfx}/ate_true_std"] = std_t
            result[f"{pfx}/ate_pred"]     = ate_p
            result[f"{pfx}/ate_pred_std"] = std_p
            result[f"{pfx}/bias"]         = ate_p - ate_t

    return result


# ---------------------------------------------------------------------------
# All-pairs TEB evaluation
# ---------------------------------------------------------------------------

def compute_teb_all_pairs(
    model: MLP,
    dataset: PPCIDataset,
    device: torch.device,
    methods: List[str] = ["ead"],
    agg: str = "mean",
    eval_task: Optional[str] = None,
) -> Tuple[Dict[str, Dict], pd.DataFrame]:
    """Compute TEB for every unique treatment pair and report per-pair + average.

    Args:
        model, dataset, device: standard arguments.
        methods:   list of estimator names.
        agg:       frame aggregation: "mean" | "sum".
        eval_task: optional cross-outcome aggregation: "or" | "sum" | None.

    Returns:
        (results, summary_df) where:
          results     — dict keyed by pair label "T{t0}_vs_T{t1}" → flat metrics dict
                        plus key "avg" → averaged metrics across all pairs
          summary_df  — tidy DataFrame with columns:
                        pair | outcome | method | ate_true | ate_pred | bias
    """
    obs_df = aggregate_to_observations(model, dataset, device, agg=agg, eval_task=eval_task)

    t_vals = sorted(obs_df["T"].unique().tolist())
    pairs = list(combinations(t_vals, 2))
    if not pairs:
        return {}, pd.DataFrame()

    y_true_cols = [c for c in obs_df.columns
                   if c.startswith("Y_") and not c.startswith("Yhat_")]
    w_cols = [c for c in obs_df.columns if c.startswith("W_")]

    results: Dict[str, Dict] = {}
    rows = []

    for t0, t1 in pairs:
        pair_key = f"T{t0}_vs_T{t1}"
        pair_metrics: Dict[str, float] = {}

        for y_col in y_true_cols:
            label = y_col[len("Y_"):]
            yhat_col = f"Yhat_{label}"
            if yhat_col not in obs_df.columns:
                continue

            for method in methods:
                try:
                    ate_t, std_t, _ = compute_ate(obs_df, y_col,   t0, t1, w_cols or None, method)
                    ate_p, std_p, _ = compute_ate(obs_df, yhat_col, t0, t1, w_cols or None, method)
                except Exception as exc:
                    warnings.warn(f"[compute_teb_all_pairs] {pair_key}/{label}/{method}: {exc}")
                    continue

                pfx = f"{label}/{method}"
                pair_metrics[f"{pfx}/ate_true"]     = ate_t
                pair_metrics[f"{pfx}/ate_true_std"] = std_t
                pair_metrics[f"{pfx}/ate_pred"]     = ate_p
                pair_metrics[f"{pfx}/ate_pred_std"] = std_p
                pair_metrics[f"{pfx}/bias"]         = ate_p - ate_t

                rows.append({
                    "pair":     pair_key,
                    "outcome":  label,
                    "method":   method,
                    "ate_true": ate_t,
                    "ate_pred": ate_p,
                    "bias":     ate_p - ate_t,
                })

        results[pair_key] = pair_metrics

    # Average across pairs for each (outcome, method) metric
    if results:
        all_keys = set(k for m in results.values() for k in m)
        avg: Dict[str, float] = {}
        for k in all_keys:
            vals = [m[k] for m in results.values() if k in m]
            avg[k] = float(np.mean(np.abs(vals)) if k.endswith("/bias") else np.mean(vals))
        results["avg"] = avg

    summary_df = pd.DataFrame(rows) if rows else pd.DataFrame()
    return results, summary_df


# ---------------------------------------------------------------------------
# Private ATE estimators
# ---------------------------------------------------------------------------

def _ate_ead(Y: np.ndarray, T: np.ndarray, W: np.ndarray) -> Tuple[float, float, float]:
    """Naive difference in means."""
    y1, y0 = Y[T == 1], Y[T == 0]
    if len(y1) == 0 or len(y0) == 0:
        return float("nan"), float("nan"), float("nan")
    ate = float(y1.mean() - y0.mean())
    std = float(np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0)))
    # One-sided t-test: H0: ATE ≤ 0
    try:
        ite = np.concatenate([y1 - y0.mean(), -(y0 - y1.mean())])
        pval = float(ttest_1samp(ite, 0, alternative="greater").pvalue)
    except Exception:
        pval = float("nan")
    return ate, std, pval


def _ate_aipw(Y: np.ndarray, T: np.ndarray, W: np.ndarray) -> Tuple[float, float, float]:
    """AIPW with XGBoost outcome model and Logistic propensity.

    Doubly robust: consistent if either the outcome or propensity model is correct.
    """
    try:
        from xgboost import XGBRegressor
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        warnings.warn("[AIPW] xgboost or scikit-learn not installed. Falling back to EAD.")
        return _ate_ead(Y, T, W)

    N = len(Y)

    # Propensity score P(T=1 | W)
    if len(np.unique(T)) < 2:
        return float("nan"), float("nan"), float("nan")
    try:
        prop_model = LogisticRegression(max_iter=200, C=1.0)
        prop_model.fit(W, T)
        ps = prop_model.predict_proba(W)[:, 1].clip(0.05, 0.95)  # clip for stability
    except Exception:
        ps = np.full(N, T.mean())

    # Outcome model E[Y | W, T]
    WT = np.column_stack([W, T])
    out_model = XGBRegressor(n_estimators=50, max_depth=3, verbosity=0, random_state=0)
    out_model.fit(WT, Y)
    mu0 = out_model.predict(np.column_stack([W, np.zeros(N)]))
    mu1 = out_model.predict(np.column_stack([W, np.ones(N)]))

    # AIPW pseudo-outcomes
    ite = (mu1 - mu0
           + T * (Y - mu1) / ps
           - (1 - T) * (Y - mu0) / (1 - ps))

    ate = float(ite.mean())
    std = float(np.sqrt(ite.var(ddof=1) / N))
    try:
        pval = float(ttest_1samp(ite, 0, alternative="greater").pvalue)
    except Exception:
        pval = float("nan")
    return ate, std, pval


def _ate_econml(
    Y: np.ndarray, T: np.ndarray, W: np.ndarray, method: str
) -> Tuple[float, float, float]:
    """ATE via econml meta-learners or causal forest."""
    try:
        from econml.metalearners import SLearner, TLearner, XLearner
        from econml.dr import DRLearner
        from econml.dml import CausalForestDML
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import GradientBoostingRegressor
    except ImportError as e:
        warnings.warn(f"[econml] {e}. Falling back to EAD.")
        return _ate_ead(Y, T, W)

    base = GradientBoostingRegressor(n_estimators=50, max_depth=2, random_state=0)

    if method == "slearner":
        m = SLearner(overall_model=base)
    elif method == "tlearner":
        m = TLearner(models=base)
    elif method == "xlearner":
        m = XLearner(models=base)
    elif method == "drlearner":
        m = DRLearner(random_state=0)
    elif method == "causalforest":
        m = CausalForestDML(discrete_treatment=True, random_state=0)
    else:
        raise ValueError(f"Unknown econml method: {method}")

    try:
        m.fit(Y=Y, T=T, X=W)
        ite = m.effect(W)
        ate = float(np.mean(ite))
        std = float(np.sqrt(np.var(ite, ddof=1) / len(ite)))
        try:
            pval = float(ttest_1samp(ite, 0, alternative="greater").pvalue)
        except Exception:
            pval = float("nan")
        return ate, std, pval
    except Exception as exc:
        warnings.warn(f"[econml/{method}] estimation failed: {exc}. Falling back to EAD.")
        return _ate_ead(Y, T, W)
