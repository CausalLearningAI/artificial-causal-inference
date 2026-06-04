#!/usr/bin/env python3
"""
Plot precision, recall, accuracy, and balanced accuracy per treatment group
on the test set, comparing the 'final' and 'validate' models side by side.

  - final   model: test on v5
  - validate model: test on v4
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from ppci.dataset import PPCIDataset
from ppci.hparam_search import DS_KWARGS
from ppci.train import build_model
from ppci.hparam_search import BASE_CFG, _build_finetune_cfg

DEPLOY_DIR = ROOT / "results" / "ppci" / "ants" / "performances"

METRICS = ["acc", "bacc", "fpr", "fnr"]
METRIC_NAMES = {
    "acc":  "Accuracy",
    "bacc": "Balanced Acc",
    "fpr":  "FP Rate",
    "fnr":  "FN Rate",
}


def _binary_stats(yt: torch.Tensor, yh: torch.Tensor):
    """Return (bacc, fpr, fnr) for binary tensors."""
    tp = float(((yt == 1) & (yh == 1)).sum())
    tn = float(((yt == 0) & (yh == 0)).sum())
    fp = float(((yt == 0) & (yh == 1)).sum())
    fn = float(((yt == 1) & (yh == 0)).sum())
    fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    fnr = fn / (fn + tp) if (fn + tp) > 0 else float("nan")
    tpr = 1 - fnr if not np.isnan(fnr) else float("nan")
    tnr = 1 - fpr if not np.isnan(fpr) else float("nan")
    bacc = (tpr + tnr) / 2 if not (np.isnan(tpr) or np.isnan(tnr)) else float("nan")
    return bacc, fpr, fnr


def compute_per_treatment_metrics(
    model,
    ds: PPCIDataset,
    device: torch.device,
    batch_size: int = 4096,
) -> dict[str, dict[str, float]]:
    """Compute acc/bacc/recall/precision per treatment group (averaged over outcome columns)."""
    X = ds.X.float()
    Y = ds.Y.float()
    T = ds.T  # numpy str array, shape (N,)

    # Filter annotated frames
    if Y.dim() == 1:
        ann = ~torch.isnan(Y)
    else:
        ann = ~torch.isnan(Y).any(dim=1)
    X = X[ann]; Y = Y[ann]; T = T[ann.numpy()]

    # Run batched inference
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            chunks.append(model.probs(X[start:start + batch_size].to(device)).cpu())
    yh_prob = torch.cat(chunks, dim=0)
    yh_bin = yh_prob.round()

    unique_treatments = sorted(set(T.tolist()))
    result: dict[str, dict[str, float]] = {}

    for t_val in unique_treatments:
        mask = torch.from_numpy(T == t_val)
        Yt = Y[mask]
        Yht_bin = yh_bin[mask]

        if len(Yt) == 0:
            continue

        # Average over outcome columns
        n_cols = Yt.shape[1] if Yt.dim() > 1 else 1
        acc_vals, bacc_vals, fpr_vals, fnr_vals = [], [], [], []

        for k in range(n_cols):
            yt_k = Yt[:, k] if Yt.dim() > 1 else Yt
            yh_k = Yht_bin[:, k] if Yht_bin.dim() > 1 else Yht_bin

            acc = float((yt_k == yh_k).float().mean())
            bacc, fpr, fnr = _binary_stats(yt_k, yh_k)
            acc_vals.append(acc); bacc_vals.append(bacc)
            fpr_vals.append(fpr); fnr_vals.append(fnr)

        result[t_val] = {
            "acc":  float(np.nanmean(acc_vals)),
            "bacc": float(np.nanmean(bacc_vals)),
            "fpr":  float(np.nanmean(fpr_vals)),
            "fnr":  float(np.nanmean(fnr_vals)),
        }

    return result


def load_model_and_eval(deploy_name: str, test_version: str, device: torch.device):
    """Load a deployed model, evaluate on test_version, return per-treatment metrics."""
    out_dir = DEPLOY_DIR / deploy_name
    cfg_path = out_dir / "config.json"
    model_path = out_dir / "model.pt"

    with open(cfg_path) as f:
        cfg_dict = json.load(f)

    encoder   = cfg_dict["encoder"]
    token     = cfg_dict.get("token", "class")
    k         = int(cfg_dict.get("context_window", 0))
    mode      = cfg_dict.get("context_mode", "mean")
    dist_mode = cfg_dict.get("dist_mode", "none")
    frame_type = cfg_dict.get("frame_type", "pov")

    print(f"\n[{deploy_name}] Loading {test_version} ({frame_type}, {encoder}/{token}, k={k}) ...")
    ds = PPCIDataset.from_disk(
        "ants", test_version, encoder, token,
        frame_type=frame_type, dist_mode=dist_mode,
        n_val_videos=0, **DS_KWARGS,
    )
    if k > 0:
        ds.apply_context_window(k, mode=mode)

    # Build model with right architecture
    try:
        finetune_cfg = _build_finetune_cfg(cfg_dict)
    except Exception:
        finetune_cfg = BASE_CFG

    model = build_model(ds, finetune_cfg)
    state = torch.load(model_path, map_location=device, weights_only=True)
    # Remap trunk ↔ featurizer if needed
    model_top = set(k_.split(".")[0] for k_ in model.state_dict())
    ckpt_top  = set(k_.split(".")[0] for k_ in state)
    if "featurizer" in model_top and "featurizer" not in ckpt_top and "trunk" in ckpt_top:
        state = {("featurizer" + k_[len("trunk"):] if k_.startswith("trunk.") else k_): v
                 for k_, v in state.items()}
    elif "trunk" in model_top and "trunk" not in ckpt_top and "featurizer" in ckpt_top:
        state = {("trunk" + k_[len("featurizer"):] if k_.startswith("featurizer.") else k_): v
                 for k_, v in state.items()}
    model.load_state_dict(state)
    model = model.to(device)

    metrics = compute_per_treatment_metrics(model, ds, device)
    print(f"  → treatments: {list(metrics.keys())}")
    for t, m in metrics.items():
        print(f"  T={t}: acc={m['acc']:.3f}  bacc={m['bacc']:.3f}  "
              f"fpr={m['fpr']:.3f}  fnr={m['fnr']:.3f}")
    return metrics


def _plot_one(ax, metrics: dict, title: str, cmap) -> None:
    """Grouped bar chart for one model: x=treatment, groups=metric."""
    all_treatments = sorted(metrics.keys())
    treatment_names = all_treatments  # use original codes

    n_metrics = len(METRICS)
    bar_width = 0.18
    treatment_gap = 0.3
    colors = [cmap(0.4 + 0.45 * j / (n_metrics - 1)) for j in range(n_metrics)]

    x_tick_centers = []
    x_cursor = 0.0
    for ti, (t_val, t_name) in enumerate(zip(all_treatments, treatment_names)):
        group_start = x_cursor
        for ji, metric in enumerate(METRICS):
            val = metrics.get(t_val, {}).get(metric, float("nan"))
            bar_x = x_cursor + ji * bar_width
            ax.bar(bar_x, val, bar_width * 0.9, color=colors[ji],
                   label=METRIC_NAMES[metric] if ti == 0 else "")
            if not np.isnan(val):
                ax.text(bar_x, val + 0.005, f"{val:.2f}", ha="center", va="bottom",
                        fontsize=7, rotation=90)
        group_end = x_cursor + n_metrics * bar_width
        x_tick_centers.append((group_start + group_end) / 2)
        x_cursor = group_end + treatment_gap

    ax.set_xticks(x_tick_centers)
    ax.set_xticklabels(treatment_names, fontsize=11)
    ax.set_ylabel("Metric value", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    return ax.get_legend_handles_labels()


def plot(final_metrics, validate_metrics, save_path: str) -> None:
    """Two-subplot figure: one per model, x=treatment, grouped bars=metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Model performance by treatment — test set", fontsize=14, y=1.02)

    handles_v, labels_v = _plot_one(axes[0], validate_metrics, "Validate model (test on v4)", plt.cm.Blues)
    handles_f, labels_f = _plot_one(axes[1], final_metrics,    "Final model (test on v5)",    plt.cm.Oranges)

    # One legend per subplot, horizontal row below each
    axes[0].legend(handles_v, labels_v, loc="upper center", ncol=len(METRICS),
                   fontsize=9, bbox_to_anchor=(0.5, -0.12), frameon=True)
    axes[1].legend(handles_f, labels_f, loc="upper center", ncol=len(METRICS),
                   fontsize=9, bbox_to_anchor=(0.5, -0.12), frameon=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n✓ Plot saved: {save_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    final_metrics    = load_model_and_eval("final",    "v5", device)
    validate_metrics = load_model_and_eval("validate", "v4", device)

    out_path = str(DEPLOY_DIR / "plot_performance_by_treatment.png")
    plot(final_metrics, validate_metrics, save_path=out_path)


if __name__ == "__main__":
    main()
