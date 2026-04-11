"""Visualization utilities for PPCI causal evaluation.

plot_outcome_distribution      — Bar plot of mean outcome per treatment (one subplot per outcome).
plot_outcome_distribution_ants — Grouped bar plot for ants: Y2F (yellow) and B2F (blue) per treatment.

plot_ate_matrix   — ATE matrix (ground-truth outcomes, use_pred=False)
                    or PP-ATE matrix (predicted outcomes, use_pred=True)
                    entry (i,j) = E[Y|T=tᵢ] - E[Y|T=tⱼ]   (lower triangle only)
                    cells coloured solid red/blue by significance (Welch t-test, alpha=0.05),
                    gray if not significant.

plot_po_barplot   — Average potential outcomes per treatment with ±SE error bars.
                    Shows ground-truth (green) and predicted (violet) side-by-side.

plot_summary      — Convenience wrapper: one row per outcome.
                    annotations=True  → ATE | PP-ATE | PO barplot (3 cols)
                    annotations=False → PP-ATE | PO barplot pred-only (2 cols)

plot_error_examples — Grid of N worst prediction errors with frame images.
                      One row per error: frame image, true Y, predicted Yhat, obs context.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from scipy.stats import sem as _sem, t as _t
from scipy.stats import ttest_ind as _ttest


def _ci95(values: np.ndarray) -> float:
    """95% CI half-width via t-distribution: t_{n-1, 0.025} * SEM."""
    n = len(values)
    if n < 2:
        return float("nan")
    return float(_t.ppf(0.975, df=n - 1) * _sem(values))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_COLOR_POS = "#cc3333"   # significant positive
_COLOR_NEG = "#3366cc"   # significant negative
_COLOR_NS  = "#e8e8e8"   # not significant
_COLOR_MASKED = "#cccccc"  # diagonal / upper triangle


def _fmt_pval(p: float) -> str:
    if p < 0.0001:
        return "(p<.0001)"
    if p < 0.001:
        return "(p<.001)"
    if p < 0.01:
        return "(p<.01)"
    return f"(p={p:.2f})"


# ---------------------------------------------------------------------------
# ATE / PP-ATE matrix
# ---------------------------------------------------------------------------

def plot_ate_matrix(
    obs_df: pd.DataFrame,
    outcome: str,
    use_pred: bool = False,
    ax: Optional[matplotlib.axes.Axes] = None,
    title: Optional[str] = None,
    treatment_labels: Optional[Dict[int, str]] = None,
    alpha: float = 0.05,
    fmt: str = ".3f",
) -> matplotlib.axes.Axes:
    """Plot the ATE matrix (ground truth) or PP-ATE matrix (model predictions).

    Lower triangle only (upper is anti-symmetric redundancy).
    Each cell is coloured solid red (significant positive effect) or blue
    (significant negative effect) via a Welch two-sample t-test at the given
    alpha level; non-significant cells are gray.
    Cell annotations show: ATE value, ±SE, and p-value.

    Args:
        obs_df:           Observation-level DataFrame (from aggregate_to_observations).
        outcome:          Outcome label, e.g. "Y2F".
        use_pred:         If True, uses Ŷ (PP-ATE); otherwise uses Y (ATE).
        ax:               Matplotlib Axes to draw on (created if None).
        title:            Axes title (default: "ATE — {outcome}" or "PP-ATE — …").
        treatment_labels: Optional mapping {treatment_value → display_name}.
        alpha:            Significance threshold for t-test (default 0.05).
        fmt:              Number format for ATE annotation.

    Returns:
        The matplotlib Axes.
    """
    col = f"{'Yhat' if use_pred else 'Y'}_{outcome}"
    if col not in obs_df.columns:
        raise ValueError(f"Column {col!r} not in obs_df.")

    t_vals = sorted(obs_df["T"].unique().tolist())
    n      = len(t_vals)
    xlbls  = [str(treatment_labels.get(t, t)) if treatment_labels else str(t) for t in t_vals]

    grp_vals = {t: obs_df.loc[obs_df["T"] == t, col].dropna().values for t in t_vals}
    grp_mean = {t: float(grp_vals[t].mean()) for t in t_vals}
    grp_sem  = {t: float(_sem(grp_vals[t]))  for t in t_vals}

    if ax is None:
        _, ax = plt.subplots(figsize=(max(5, n * 0.9), max(4, n * 0.8)))

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_aspect("equal")

    fontsize = max(8, 12 - n)

    for i in range(n):
        for j in range(n):
            if i <= j:
                # diagonal + upper triangle: masked
                ax.add_patch(mpatches.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    facecolor=_COLOR_MASKED, edgecolor="white", linewidth=0.5, zorder=1,
                ))
                continue

            ate  = grp_mean[t_vals[i]] - grp_mean[t_vals[j]]
            se   = np.sqrt(grp_sem[t_vals[i]] ** 2 + grp_sem[t_vals[j]] ** 2)
            _, p = _ttest(grp_vals[t_vals[i]], grp_vals[t_vals[j]], equal_var=False)

            sig = p < alpha
            if sig and ate > 0:
                bg, tc = _COLOR_POS, "white"
            elif sig and ate < 0:
                bg, tc = _COLOR_NEG, "white"
            else:
                bg, tc = _COLOR_NS, "black"

            ax.add_patch(mpatches.Rectangle(
                (j - 0.5, i - 0.5), 1, 1,
                facecolor=bg, edgecolor="white", linewidth=0.5, zorder=1,
            ))
            # ATE value
            ax.text(j, i - 0.18, f"{ate:{fmt}}", ha="center", va="center",
                    fontsize=fontsize, color=tc, fontweight="bold", zorder=2)
            # ±SE
            ax.text(j, i + 0.12, f"±{se:{fmt}}", ha="center", va="center",
                    fontsize=max(fontsize - 2, 6), color=tc, zorder=2)
            # p-value
            ax.text(j, i + 0.35, _fmt_pval(p), ha="center", va="center",
                    fontsize=max(fontsize - 2, 6), color=tc, style="italic", zorder=2)

    ax.set_xticks(range(n));  ax.set_xticklabels(xlbls, fontsize=11)
    ax.set_yticks(range(n));  ax.set_yticklabels(xlbls, fontsize=11)
    ax.set_xlabel("Control", fontsize=12)
    ax.set_ylabel("Treatment", fontsize=12)
    kind = "PP-ATE" if use_pred else "ATE"
    ax.set_title(title or f"{kind} — {outcome}", fontsize=13)

    # Legend patches (proxy artists — must NOT be added via add_patch, which calls get_path())
    legend_handles = [
        mpatches.Patch(color=_COLOR_POS, label=rf"H1: $\tau > 0$"),
        mpatches.Patch(color=_COLOR_NEG, label=rf"H1: $\tau < 0$"),
        mpatches.Patch(color=_COLOR_NS,  label=rf"H0: $\tau = 0$"),
    ]
    ax.legend(handles=legend_handles, fontsize=7, loc="upper right",
              framealpha=0.8, handlelength=1)

    return ax


# ---------------------------------------------------------------------------
# Potential outcome bar plot
# ---------------------------------------------------------------------------

def plot_po_barplot(
    obs_df: pd.DataFrame,
    outcome: str,
    ax: Optional[matplotlib.axes.Axes] = None,
    title: Optional[str] = None,
    treatment_labels: Optional[Dict[int, str]] = None,
    fmt: str = ".3f",
    pred_only: bool = False,
) -> matplotlib.axes.Axes:
    """Bar plot of average potential outcomes per treatment with 95% CI error bars.

    Shows E[Y | T=t] (green, ground truth) and E[Ŷ | T=t] (violet, predicted)
    side-by-side for each treatment.  Error bars are 95% CIs via t_{n-1} distribution.
    When ground-truth columns are absent only the predicted bars are drawn.

    Args:
        obs_df:           Observation-level DataFrame (from aggregate_to_observations).
        outcome:          Outcome label, e.g. "Y2F".
        ax:               Matplotlib Axes to draw on (created if None).
        title:            Axes title.
        treatment_labels: Optional mapping {treatment_value → display_name}.
        fmt:              Unused (kept for API consistency).

    Returns:
        The matplotlib Axes.
    """
    y_col    = f"Y_{outcome}"
    yhat_col = f"Yhat_{outcome}"
    if yhat_col not in obs_df.columns:
        raise ValueError(f"Missing column {yhat_col!r} in obs_df.")

    has_gt = (y_col in obs_df.columns) and not pred_only

    t_vals = sorted(obs_df["T"].unique().tolist())
    n      = len(t_vals)
    xlbls  = [str(treatment_labels.get(t, t)) if treatment_labels else str(t) for t in t_vals]

    pred_means = np.array([obs_df.loc[obs_df["T"] == t, yhat_col].mean() for t in t_vals])
    pred_sems  = np.array([_ci95(obs_df.loc[obs_df["T"] == t, yhat_col].values) for t in t_vals])

    if ax is None:
        _, ax = plt.subplots(figsize=(max(5, n * 0.7), 4))

    x = np.arange(n)

    if has_gt:
        def _gt_vals(t):
            v = obs_df.loc[obs_df["T"] == t, y_col].values
            return v[~np.isnan(v)]
        gt_means = np.array([_gt_vals(t).mean() for t in t_vals])
        gt_sems  = np.array([_ci95(_gt_vals(t)) for t in t_vals])
        width = 0.38
        gt_bars   = ax.bar(x - width / 2, gt_means, width, yerr=gt_sems, capsize=3,
                           color="#44aa77", alpha=0.85, label="Ground truth")
        pred_bars = ax.bar(x + width / 2, pred_means, width, yerr=pred_sems, capsize=3,
                           color="#9944bb", alpha=0.85, label="Predicted")
        for bar, val in zip(gt_bars, gt_means):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7, color="#333333")
        for bar, val in zip(pred_bars, pred_means):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7, color="#333333")
    else:
        pred_bars = ax.bar(x, pred_means, yerr=pred_sems, capsize=3,
                           color="#9944bb", alpha=0.85, label="Predicted")
        for bar, val in zip(pred_bars, pred_means):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels(xlbls, fontsize=11)
    ax.set_xlabel("Treatment", fontsize=12)
    ylabel = "% predicted activity/total time" if pred_only else "% activity/total time"
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title or f"Potential outcomes — {outcome}", fontsize=13)
    ax.legend(fontsize=9)

    return ax


# ---------------------------------------------------------------------------
# Summary figure
# ---------------------------------------------------------------------------

def plot_summary(
    obs_df: pd.DataFrame,
    outcomes: list,
    annotations: Optional[bool] = None,
    treatment_labels: Optional[Dict[int, str]] = None,
    alpha: float = 0.05,
    save: bool = False,
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """Generate a summary figure for all outcomes.

    Args:
        obs_df:           Observation-level DataFrame (from aggregate_to_observations).
        outcomes:         List of outcome labels, e.g. ["Y2F", "B2F"].
        annotations:      If True, plot 3 columns: ATE matrix | PP-ATE matrix | PO barplot.
                          If False, plot 2 columns: PP-ATE matrix | PO barplot (pred only,
                          no significance legend).
                          If None (default), inferred from obs_df._has_annotations.
        treatment_labels: Optional mapping {treatment_value → display_name}.
        alpha:            Significance threshold (passed to plot_ate_matrix).
        save:             If True, save the figure instead of showing it.
        save_path:        File path to save (required when save=True).

    Returns:
        The matplotlib Figure.
    """
    if annotations is None:
        annotations = bool(obs_df["_has_annotations"].iloc[0])

    all_outcomes = list(outcomes)
    if "Y_avg" in obs_df.columns:
        all_outcomes = all_outcomes + ["avg"]

    n_out = len(all_outcomes)
    n_cols = 3 if annotations else 2
    fig, axes = plt.subplots(n_out, n_cols, figsize=(7 * n_cols, 6 * n_out), squeeze=False)

    for row, outcome in enumerate(all_outcomes):
        col_idx = 0
        if annotations:
            plot_ate_matrix(obs_df, outcome, use_pred=False,
                            ax=axes[row, col_idx],
                            treatment_labels=treatment_labels, alpha=alpha)
            col_idx += 1

        plot_ate_matrix(obs_df, outcome, use_pred=True,
                        ax=axes[row, col_idx],
                        treatment_labels=treatment_labels, alpha=alpha)
        col_idx += 1

        plot_po_barplot(obs_df, outcome,
                        ax=axes[row, col_idx],
                        treatment_labels=treatment_labels,
                        pred_only=not annotations)
        col_idx += 1

    plt.tight_layout()

    if save:
        if save_path is None:
            raise ValueError("save_path must be provided when save=True.")
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path)

    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Dataset-level outcome distribution
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Method comparison plot
# ---------------------------------------------------------------------------

_OUTCOME_PALETTE = {
    "Y2F": "#f0c040",   # yellow
    "B2F": "#4477aa",   # blue
}
_FALLBACK_BIAS_COLOR = "#aaaaaa"   # grey for avg / unknown

_METRIC_COLORS  = ["#4477aa", "#cc5533", "#44aa77"]
_METRIC_KEYS    = ["acc", "precision", "recall"]
_METRIC_LABELS  = ["Accuracy", "Precision", "Recall"]


def plot_comparison(results: dict) -> matplotlib.figure.Figure:
    """Side-by-side comparison of training methods.

    Left subplot  — Classification metrics (acc, precision, recall) per method,
                    three grouped bars per method, y-axis in [0, 1].
    Right subplot — avg(|TEB|) per outcome, averaged over all treatment pairs,
                    with 95% CI error bars. Colors: Y2F=yellow, B2F=blue, others=grey.

    Designed to be called twice for validation and test evaluation:

    * **Validation**: pass val-split metrics for the left plot and full-experiment
      ``summary_df`` (from :func:`compute_teb_all_pairs`) for the right plot.
    * **Test**: pass test-split metrics and test-split ``summary_df`` for both.

    Args:
        results: dict mapping method_name → {
                    ``acc``:        float,
                    ``precision``:  float,
                    ``recall``:     float,
                    ``summary_df``: pd.DataFrame from ``compute_teb_all_pairs``
                                    (columns: pair | outcome | method | bias)
                 }

    Returns:
        matplotlib Figure (not shown; call ``plt.show()`` or ``fig.show()``).
    """
    methods = list(results.keys())
    n = len(methods)
    x = np.arange(n)

    # Gather outcome labels from the first non-empty summary_df
    outcomes: list = []
    for v in results.values():
        df = v.get("summary_df")
        if df is not None and not df.empty and "outcome" in df.columns:
            outcomes = sorted(df["outcome"].unique().tolist())
            break

    def _bias_color(out: str) -> str:
        return _OUTCOME_PALETTE.get(out, _FALLBACK_BIAS_COLOR)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # ── Left: acc / precision / recall ───────────────────────────────────────
    n_m    = len(_METRIC_KEYS)
    bar_w  = 0.8 / n_m
    off_l  = (np.arange(n_m) - (n_m - 1) / 2) * bar_w

    for i, (key, label, clr) in enumerate(zip(_METRIC_KEYS, _METRIC_LABELS, _METRIC_COLORS)):
        vals = [results[m].get(key, float("nan")) for m in methods]
        axes[0].bar(x + off_l[i], vals, bar_w, color=clr, alpha=0.85, label=label)

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods)
    axes[0].set_ylim(0, 1)
    axes[0].legend()

    # ── Right: avg(|TEB|) per outcome with 95% CI ────────────────────────────
    n_out   = max(len(outcomes), 1)
    bar_w2  = 0.8 / n_out
    off_r   = (np.arange(n_out) - (n_out - 1) / 2) * bar_w2

    for j, outcome in enumerate(outcomes):
        means, cis = [], []
        for method in methods:
            df = results[method].get("summary_df")
            if df is None or df.empty:
                means.append(float("nan"))
                cis.append(0.0)
                continue
            vals = df[df["outcome"] == outcome]["bias"].abs().values
            if len(vals) == 0:
                means.append(float("nan"))
                cis.append(0.0)
            elif len(vals) == 1:
                means.append(float(vals[0]))
                cis.append(0.0)
            else:
                means.append(float(vals.mean()))
                cis.append(float(1.96 * _sem(vals)))

        axes[1].bar(x + off_r[j], means, bar_w2, yerr=cis, capsize=3,
                    color=_bias_color(outcome), alpha=0.85, label=outcome)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods)
    axes[1].set_ylabel("avg(|TEB|)")
    if outcomes:
        axes[1].legend(fontsize=8)

    plt.tight_layout()
    return fig


def plot_comparison_versions(
    version_metrics: dict,
    train_versions: Optional[list] = None,
    pretrain_versions: Optional[list] = None,
    test_versions: Optional[list] = None,
    save: bool = False,
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """Classification metrics across experiment versions.

    x-axis: experiment version (e.g. v1, v2, v3, v4, v5)
    y-axis: acc / bacc / recall / precision per version

    Versions without annotations (unannotated v5) are greyed out — bars show
    NaN so they appear blank, with a "no labels" annotation on the bar.
    Versions in train_versions get a suffix and a light yellow background to
    signal that metrics are in-sample; pretrain_versions (a subset of
    train_versions) are labelled "(pretrain)" while the remaining training
    versions are labelled "(finetune)".

    Args:
        version_metrics:   dict mapping version_name → {acc, bacc, recall, precision}
                           or None/empty dict for versions without annotations.
        train_versions:    List of version names that were part of training data
                           (their metrics are in-sample and shown with a warning
                           tint). Pass None to skip this annotation.
        pretrain_versions: Subset of train_versions used for pretraining; shown
                           with "(pretrain)" suffix vs "(finetune)" for the rest.
        test_versions:     Held-out test versions; shown with "(test)" suffix.
        save:              If True, save the figure to save_path.
        save_path:         Required when save=True.

    Returns:
        matplotlib Figure.
    """
    train_versions    = set(train_versions or [])
    pretrain_versions = set(pretrain_versions or [])
    test_versions     = set(test_versions or [])
    versions = list(version_metrics.keys())
    n = len(versions)
    x = np.arange(n)

    metric_keys   = ["bacc", "acc", "recall", "precision"]
    metric_labels = ["Balanced Acc", "Accuracy", "Recall", "Precision"]
    metric_colors = ["#222266", "#4477aa", "#44aa77", "#cc5533"]

    n_m   = len(metric_keys)
    bar_w = 0.8 / n_m
    offs  = (np.arange(n_m) - (n_m - 1) / 2) * bar_w

    fig, ax = plt.subplots(figsize=(max(8, 1.8 * n), 5))

    # Shade training versions before drawing bars (zorder=0)
    for xi, v in enumerate(versions):
        if v in train_versions:
            ax.axvspan(xi - 0.45, xi + 0.45, color="#fff8e1", alpha=0.8, zorder=0)

    for i, (key, label, clr) in enumerate(zip(metric_keys, metric_labels, metric_colors)):
        vals = []
        for v in versions:
            m = version_metrics[v]
            vals.append(float(m[key]) if m and key in m else float("nan"))
        bars = ax.bar(x + offs[i], vals, bar_w, color=clr, alpha=0.85, label=label)

        for bar, val in zip(bars, vals):
            if np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, 0.02,
                        "–", ha="center", va="bottom", fontsize=8, color="#888888")
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=7, color="#333333")

    # Shade versions without annotations
    for xi, v in enumerate(versions):
        if not version_metrics[v]:
            ax.axvspan(xi - 0.45, xi + 0.45, color="#eeeeee", alpha=0.6, zorder=0)
            ax.text(xi, -0.07, "no labels", ha="center", va="top",
                    fontsize=7, color="#888888", style="italic",
                    transform=ax.get_xaxis_transform())

    ax.set_xticks(x)
    # Differentiate pretrain / finetune / test on x-tick labels
    def _xtick(v: str) -> str:
        if v in pretrain_versions:
            return f"{v}\n(pretrain)"
        if v in train_versions:
            return f"{v}\n(finetune)"
        if v in test_versions:
            return f"{v}\n(test)"
        return v
    ax.set_xticklabels([_xtick(v) for v in versions], fontsize=11)
    ax.set_xlabel("Experiment version", fontsize=12)
    ax.set_ylabel("Metric", fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.set_title("Model performance across experiment versions", fontsize=13)
    # Legend: metric colors only, placed in lower-right corner
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles, fontsize=9, loc="lower right",
              framealpha=0.9, handlelength=1.2)
    plt.tight_layout()

    if save:
        if save_path is None:
            raise ValueError("save_path must be provided when save=True.")
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.close(fig)
    return fig


def plot_comparison_versions_by_metric(
    version_metrics: dict,
    train_versions: Optional[list] = None,
    pretrain_versions: Optional[list] = None,
    test_versions: Optional[list] = None,
    save: bool = False,
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """Same data as plot_comparison_versions but transposed layout.

    x-axis: metric (Balanced Acc, Accuracy, Recall, Precision)
    legend: experiment version

    This layout makes it easy to compare versions head-to-head on each metric.

    Args:
        version_metrics:   dict mapping version_name → {acc, bacc, recall, precision}
                           or None/empty dict for versions without annotations.
        train_versions:    List of training versions (labelled in legend).
        pretrain_versions: Subset of train_versions used for pretraining.
        test_versions:     Held-out test versions (labelled in legend).
        save:              If True, save to save_path.
        save_path:         Required when save=True.

    Returns:
        matplotlib Figure.
    """
    train_versions    = set(train_versions or [])
    pretrain_versions = set(pretrain_versions or [])
    test_versions     = set(test_versions or [])

    # Only versions that have metric dicts (skip None / {})
    versions = [v for v, m in version_metrics.items() if m]
    n_v = len(versions)

    metric_keys   = ["bacc", "acc", "recall", "precision"]
    metric_labels = ["Balanced Acc", "Accuracy", "Recall", "Precision"]
    n_m = len(metric_keys)

    x = np.arange(n_m)
    bar_w = 0.8 / max(n_v, 1)
    offs  = (np.arange(n_v) - (n_v - 1) / 2) * bar_w

    # Per-version colors: use tab10
    _tab10 = plt.cm.tab10
    version_colors = {v: _tab10(i / 10) for i, v in enumerate(versions)}

    fig, ax = plt.subplots(figsize=(max(7, 1.8 * n_m), 5))

    for i, version in enumerate(versions):
        m = version_metrics[version]
        vals = [float(m[k]) if m and k in m else float("nan") for k in metric_keys]

        if version in pretrain_versions:
            vlabel = f"{version} (pretrain)"
        elif version in train_versions:
            vlabel = f"{version} (finetune)"
        elif version in test_versions:
            vlabel = f"{version} (test)"
        else:
            vlabel = version

        alpha = 0.65 if version in train_versions else 0.90
        bars = ax.bar(x + offs[i], vals, bar_w,
                      color=version_colors[version], alpha=alpha, label=vlabel)

        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                        f"{val:.2f}", ha="center", va="bottom",
                        fontsize=7, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Metric value", fontsize=12)
    ax.set_title("Model performance across experiment versions", fontsize=13)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.9, handlelength=1.2)
    plt.tight_layout()

    if save:
        if save_path is None:
            raise ValueError("save_path must be provided when save=True.")
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.close(fig)
    return fig


def plot_outcome_distribution(ds, treatment_labels=None, save=False, results_dir="./results"):
    """Bar plot of mean outcome per treatment, one subplot per outcome column.

    Error bars are SEM across observations (observation_id), not across frames,
    so they reflect between-individual variability rather than sample-size artefacts.

    Args:
        ds:               PPCIDataset object (uses full Y, T, obs_ids).
        treatment_labels: Optional dict {treatment_value -> display_name}.
        save:             If True, saves to results_dir instead of showing.
        results_dir:      Directory for saved figure.
    """
    Y = ds.Y
    T = ds.T
    obs_ids = ds.obs_ids
    outcome_cols = ds.outcome_cols

    if Y.dim() == 1:
        Y = Y.unsqueeze(1)

    Y = Y.float().numpy()
    T = T.numpy() if hasattr(T, "numpy") else T

    t_vals = sorted(np.unique(T).tolist())
    xlbls = [str(treatment_labels.get(t, t)) if treatment_labels else str(t) for t in t_vals]
    n_outcomes = len(outcome_cols)

    fig, axs = plt.subplots(1, n_outcomes, figsize=(5 * n_outcomes, 4))
    if n_outcomes == 1:
        axs = [axs]

    for k, (ax, col) in enumerate(zip(axs, outcome_cols)):
        y_k = Y[:, k]
        means, sems = [], []
        for t in t_vals:
            mask = T == t
            obs_means = np.array([
                y_k[(obs_ids == oid) & mask].mean()
                for oid in np.unique(obs_ids[mask])
            ])
            means.append(obs_means.mean())
            sems.append(_sem(obs_means))
        means, sems = np.array(means), np.array(sems)
        x = np.arange(len(t_vals))
        ax.bar(x, means, yerr=sems, capsize=4, color="#4477aa", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(xlbls)
        ax.set_xlabel("Treatment")
        ax.set_ylabel("Mean outcome")
        ax.set_title(col.replace("Y_", ""))

    plt.tight_layout()

    if save:
        os.makedirs(results_dir, exist_ok=True)
        fig.savefig(os.path.join(results_dir, "outcome_distribution.png"))
    else:
        plt.show()


def plot_outcome_distribution_ants(ds, treatment_labels=None, title=None, save=False, results_dir="./results"):
    """Grouped bar plot for ants: Y2F (yellow) and B2F (blue) per treatment.

    Each treatment has two bars with SEM error bars across observations (observation_id).

    Args:
        ds:               PPCIDataset object (uses full Y, T, obs_ids).
        treatment_labels: Optional dict {treatment_value -> display_name}.
        save:             If True, saves to results_dir instead of showing.
        results_dir:      Directory for saved figure.
    """
    Y = ds.Y
    T = ds.T
    obs_ids = ds.obs_ids
    outcome_cols = ds.outcome_cols

    if Y.dim() == 1:
        Y = Y.unsqueeze(1)

    Y = Y.float().numpy()
    T = T.numpy() if hasattr(T, "numpy") else T

    y2f_idx = next((i for i, c in enumerate(outcome_cols) if "Y2F" in c), None)
    b2f_idx = next((i for i, c in enumerate(outcome_cols) if "B2F" in c), None)

    if y2f_idx is None or b2f_idx is None:
        raise ValueError("Dataset must have both Y_Y2F and Y_B2F outcome columns.")

    t_vals = sorted(np.unique(T).tolist())
    xlbls = [str(treatment_labels.get(t, t)) if treatment_labels else str(t) for t in t_vals]

    y2f_means, y2f_sems = [], []
    b2f_means, b2f_sems = [], []

    for t in t_vals:
        mask = T == t
        unique_obs = np.unique(obs_ids[mask])
        for idx, (means_list, sems_list) in zip(
            [y2f_idx, b2f_idx],
            [(y2f_means, y2f_sems), (b2f_means, b2f_sems)]
        ):
            obs_means = np.array([Y[(obs_ids == oid) & mask, idx].mean() for oid in unique_obs])
            means_list.append(obs_means.mean())
            sems_list.append(_sem(obs_means))

    y2f_means, y2f_sems = np.array(y2f_means), np.array(y2f_sems)
    b2f_means, b2f_sems = np.array(b2f_means), np.array(b2f_sems)

    x = np.arange(len(t_vals))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(5, 1.5 * len(t_vals)), 4))
    ax.bar(x - width / 2, y2f_means, width, yerr=y2f_sems, capsize=4,
           color="#f0c040", alpha=0.9, label="Y2F grooming")
    ax.bar(x + width / 2, b2f_means, width, yerr=b2f_sems, capsize=4,
           color="#4477aa", alpha=0.9, label="B2F grooming")

    ax.set_xticks(x)
    ax.set_xticklabels(xlbls)
    ax.set_xlabel("Treatment")
    ax.set_ylabel("% activity/total time")
    resolved_title = title if title is not None else getattr(ds, "name", None)
    if resolved_title:
        ax.set_title(resolved_title)
    ax.legend()
    plt.tight_layout()

    if save:
        os.makedirs(results_dir, exist_ok=True)
        fig.savefig(os.path.join(results_dir, "outcome_distribution_ants.png"))
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Error examples
# ---------------------------------------------------------------------------

def plot_error_examples(
    dataset,
    outcome_cols: list[str],
    n: int = 20,
    dataset_root: str = "./dataset",
    seed: int = 0,
    save: bool = False,
    save_path: Optional[str] = None,
    frame_type: str = "full",
    treatment_filter=None,
    outcome_error_filter: Optional[str] = None,
) -> None:
    """Plot a grid of N prediction error examples with frame images.

    Rows are sampled from annotated frames where the rounded prediction
    disagrees with the true label on at least one outcome.  Each row shows:
      - The frame image (full view, and for pov: yellow + blue POV crops)
      - True Y and predicted Yhat per outcome column (colour-coded)
      - Contextual info: observation_id, frame_idx, treatment

    Args:
        dataset:              PPCIDataset with add_predictions() already called.
        outcome_cols:         List of outcome column names used (e.g. ["Y_Y2F", "Y_B2F"]).
        n:                    Number of error examples to show (default 20).
        dataset_root:         Root of the dataset directory (default "./dataset").
        seed:                 RNG seed for sampling when errors > n (default 0).
        save:                 If True, save to save_path instead of showing.
        save_path:            File path for the saved figure.
        frame_type:           "full" (default) or "pov". When "pov", also shows yellow and
                              blue POV crop images alongside the full frame.
        treatment_filter:     Optional treatment value (or list of values) to restrict
                              errors to frames with that treatment (e.g. "B" or 2).
        outcome_error_filter: Optional outcome column name (e.g. "Y_Y2F") to restrict
                              errors to frames where that specific outcome is wrong.
    """
    from pathlib import Path
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Pillow is required for plot_error_examples: pip install Pillow")

    if dataset.Yhat is None:
        raise ValueError("Call dataset.add_predictions(model, device) before plot_error_examples().")

    Y    = dataset.Y.float()       # (N,) or (N, k)
    Yhat = dataset.Yhat.float()    # same shape
    ann  = dataset.annotated_mask  # (N,) bool

    # Identify error frames among annotated ones
    if Y.dim() == 1:
        wrong = (Yhat.round() != Y) & ann
    else:
        if outcome_error_filter is not None and outcome_error_filter in outcome_cols:
            oc_idx = outcome_cols.index(outcome_error_filter)
            wrong = (Yhat[:, oc_idx].round() != Y[:, oc_idx]) & ann
        else:
            wrong = ((Yhat.round() != Y).any(dim=1)) & ann

    # Apply treatment filter
    if treatment_filter is not None:
        import torch as _torch
        T_tensor = dataset.T
        if not isinstance(treatment_filter, (list, tuple)):
            treatment_filter = [treatment_filter]
        t_mask = _torch.zeros(len(T_tensor), dtype=_torch.bool)
        for tv in treatment_filter:
            if isinstance(T_tensor[0], str):
                t_mask |= _torch.tensor([t == tv for t in T_tensor], dtype=_torch.bool)
            else:
                t_mask |= (T_tensor == tv)
        wrong = wrong & t_mask

    error_indices = wrong.nonzero(as_tuple=True)[0]
    if len(error_indices) == 0:
        print("[plot_error_examples] No prediction errors found in annotated frames.")
        return

    # Sample up to n
    rng = np.random.default_rng(seed)
    if len(error_indices) > n:
        chosen = rng.choice(len(error_indices), size=n, replace=False)
        error_indices = error_indices[chosen]
    n_show = len(error_indices)

    labels = [c.replace("Y_", "") for c in outcome_cols]
    n_outcomes = len(labels)
    pov_colors = ["yellow", "blue"] if frame_type == "pov" else []

    # Layout: one row per error, cols = [full_image, (pov_yellow, pov_blue,) label_col_0, …]
    n_img_cols = 1 + len(pov_colors)
    n_cols = n_img_cols + n_outcomes
    col_w  = [2.5] * n_img_cols + [1.2] * n_outcomes
    fig, axes = plt.subplots(
        n_show, n_cols,
        figsize=(sum(col_w), n_show * 2.2),
        gridspec_kw={"width_ratios": col_w},
    )
    if n_show == 1:
        axes = axes[np.newaxis, :]   # keep 2-D

    filter_desc = ""
    if treatment_filter is not None:
        tv = treatment_filter if isinstance(treatment_filter, (list, tuple)) else [treatment_filter]
        filter_desc += f"  T∈{{{','.join(str(x) for x in tv)}}}"
    if outcome_error_filter is not None:
        filter_desc += f"  err on {outcome_error_filter.replace('Y_', '')}"
    fig.suptitle(
        f"Prediction errors  ({n_show} of {int(wrong.sum())} annotated errors shown){filter_desc}",
        fontsize=13, y=1.002,
    )

    obs_ids    = dataset.obs_ids
    frame_idxs = dataset.frame_idx.numpy()
    T_vals     = dataset.T

    for row, idx in enumerate(error_indices.tolist()):
        obs_id    = obs_ids[idx]
        frame_idx = int(frame_idxs[idx])
        treatment = T_vals[idx]

        # ── frame image ────────────────────────────────────────────────────
        # Reconstruct path from obs_id and frame_idx
        obs_file_stem = obs_id  # e.g. "5_1_5"

        def _load_frame_image(glob_pattern: str):
            candidates = list(Path(dataset_root).glob(glob_pattern))
            if candidates:
                try:
                    return np.array(Image.open(candidates[0]).convert("RGB"))
                except Exception:
                    pass
            return None

        def _show_image(ax, img, title):
            if img is not None:
                ax.imshow(img)
            else:
                ax.set_facecolor("#dddddd")
                ax.text(0.5, 0.5, "no image", ha="center", va="center",
                        transform=ax.transAxes, fontsize=7)
            ax.axis("off")
            if title:
                ax.set_title(title, fontsize=7, pad=2)

        full_img = _load_frame_image(
            f"*/*/frames/full/{obs_file_stem}/frame_{frame_idx:06d}.jpg"
        )
        _show_image(axes[row, 0], full_img,
                    f"{obs_id}  f={frame_idx}  T={treatment}")

        # ── POV images (yellow, blue) ───────────────────────────────────────
        for pov_col_offset, color in enumerate(pov_colors, start=1):
            pov_img = _load_frame_image(
                f"*/*/frames/pov/{color}/{obs_file_stem}/frame_{frame_idx:06d}.jpg"
            )
            title = color if row == 0 else None
            _show_image(axes[row, pov_col_offset], pov_img, title)

        # ── per-outcome label panels ────────────────────────────────────────
        y_row    = Y[idx]    if Y.dim() == 1    else Y[idx]
        yhat_row = Yhat[idx] if Yhat.dim() == 1 else Yhat[idx]

        if Y.dim() == 1:
            y_vals    = [float(y_row)]
            yhat_vals = [float(yhat_row)]
        else:
            y_vals    = y_row.tolist()
            yhat_vals = yhat_row.tolist()

        for col, (label, y_true, y_pred) in enumerate(zip(labels, y_vals, yhat_vals)):
            ax = axes[row, n_img_cols + col]
            correct = round(y_pred) == round(y_true)
            bg = "#d4edda" if correct else "#f8d7da"   # green / red
            ax.set_facecolor(bg)
            ax.text(0.5, 0.65,
                    f"Y={int(y_true)}",
                    ha="center", va="center", fontsize=10, fontweight="bold",
                    transform=ax.transAxes)
            ax.text(0.5, 0.30,
                    f"Ŷ={y_pred:.2f}",
                    ha="center", va="center", fontsize=9,
                    color="#555555", transform=ax.transAxes)
            ax.axis("off")
            if row == 0:
                ax.set_title(label, fontsize=9, pad=3)

    plt.tight_layout()
    if save and save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
