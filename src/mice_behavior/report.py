"""Shared report generation for mouse behavior classifier variants (CLS, patch-grid).

Produces a 3-row figure:
    Row 0: train/val loss curve, val macro PR-AUC over epochs
    Row 1: per-ordered-pair ROC / PR
    Row 2: collapsed per-frame ROC / PR (did behavior X happen anywhere in this frame)
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

LABEL_NAMES = ['none', 'nt', 'nn']


def collect_val_predictions(model, val_loader, dev):
    all_probs, all_labels = [], []
    with torch.no_grad():
        for ctx, offsets, a1, a2, labels, mask in val_loader:
            logits = model(
                ctx.to(dev), a1.to(dev), a2.to(dev),
                offsets=offsets.to(dev), key_padding_mask=mask.to(dev),
            )
            all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def generate_report(probs, labels, history, title, out_dir: Path):
    out_dir = Path(out_dir)
    n_pairs = 12
    n_frames_val = len(labels) // n_pairs
    probs_r = probs.reshape(n_frames_val, n_pairs, 3)
    labels_r = labels.reshape(n_frames_val, n_pairs)

    fig, axes = plt.subplots(3, 2, figsize=(12, 15))
    save_data = {}

    best_idx = int(np.argmax(history['macro_pr_auc']))
    best_epoch = history['eval_epoch'][best_idx]

    axes[0, 0].plot(history['epoch'], history['train_loss'], label='train loss')
    axes[0, 0].plot(history['eval_epoch'], history['val_loss'], label='val loss')
    axes[0, 0].axvline(best_epoch, color='tab:green', ls='--', alpha=0.6, label=f'best epoch ({best_epoch})')
    axes[0, 0].set_xlabel('epoch'); axes[0, 0].set_ylabel('loss')
    axes[0, 0].set_title('Train / val loss'); axes[0, 0].legend()

    axes[0, 1].plot(history['eval_epoch'], history['macro_pr_auc'], color='tab:green')
    axes[0, 1].axvline(best_epoch, color='tab:green', ls='--', alpha=0.6)
    axes[0, 1].scatter([best_epoch], [history['macro_pr_auc'][best_idx]], color='tab:red', zorder=5,
                        label=f"best={history['macro_pr_auc'][best_idx]:.3f} @ epoch {best_epoch}")
    axes[0, 1].set_xlabel('epoch'); axes[0, 1].set_ylabel('macro PR-AUC')
    axes[0, 1].set_title('Val macro PR-AUC over training'); axes[0, 1].legend()

    for c, name in enumerate(LABEL_NAMES):
        y_true = (labels == c).astype(int)
        y_score = probs[:, c]
        fpr, tpr, _ = roc_curve(y_true, y_score)
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        roc_auc = roc_auc_score(y_true, y_score)
        pr_auc = average_precision_score(y_true, y_score)
        axes[1, 0].plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.3f})')
        axes[1, 1].plot(rec, prec, label=f'{name} (AP={pr_auc:.3f})')
        save_data[f'pair_{name}_fpr'] = fpr
        save_data[f'pair_{name}_tpr'] = tpr
        save_data[f'pair_{name}_prec'] = prec
        save_data[f'pair_{name}_rec'] = rec
        save_data[f'pair_{name}_roc_auc'] = roc_auc
        save_data[f'pair_{name}_pr_auc'] = pr_auc
    axes[1, 0].plot([0, 1], [0, 1], 'k--', alpha=0.3)
    axes[1, 0].set_xlabel('FPR'); axes[1, 0].set_ylabel('TPR'); axes[1, 0].set_title('Per-pair ROC'); axes[1, 0].legend()
    axes[1, 1].set_xlabel('Recall'); axes[1, 1].set_ylabel('Precision'); axes[1, 1].set_title('Per-pair PR'); axes[1, 1].legend()

    for c, name in [(1, 'nt'), (2, 'nn')]:
        frame_true = (labels_r == c).any(axis=1).astype(int)
        frame_score = probs_r[:, :, c].max(axis=1)
        fpr, tpr, _ = roc_curve(frame_true, frame_score)
        prec, rec, _ = precision_recall_curve(frame_true, frame_score)
        roc_auc = roc_auc_score(frame_true, frame_score)
        pr_auc = average_precision_score(frame_true, frame_score)
        axes[2, 0].plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.3f})')
        axes[2, 1].plot(rec, prec, label=f'{name} (AP={pr_auc:.3f})')
        save_data[f'frame_{name}_fpr'] = fpr
        save_data[f'frame_{name}_tpr'] = tpr
        save_data[f'frame_{name}_prec'] = prec
        save_data[f'frame_{name}_rec'] = rec
        save_data[f'frame_{name}_roc_auc'] = roc_auc
        save_data[f'frame_{name}_pr_auc'] = pr_auc
    axes[2, 0].plot([0, 1], [0, 1], 'k--', alpha=0.3)
    axes[2, 0].set_xlabel('FPR'); axes[2, 0].set_ylabel('TPR'); axes[2, 0].set_title('Collapsed per-frame ROC'); axes[2, 0].legend()
    axes[2, 1].set_xlabel('Recall'); axes[2, 1].set_ylabel('Precision'); axes[2, 1].set_title('Collapsed per-frame PR'); axes[2, 1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_dir / 'report.png', dpi=150)
    np.savez(out_dir / 'roc_pr_data.npz', **save_data)
