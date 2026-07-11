"""Shared report generation for mouse behavior classifier variants (CLS, patch-grid).

Produces a 3-row figure:
    Row 0: train/val loss curve; val macro PR-AUC (nt/nn only) over epochs, per couple and
        per frame
    Row 1: behaviors per mice couples — ROC / PR per ordered pair
    Row 2: aggregated behaviors per frame — ROC / PR (did behavior X happen anywhere in this
        frame, for nt/nn; did NO pair interact anywhere in this frame, for none)
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


def collect_val_predictions_fast(model, val_data, dev, batch_size=1024):
    """Same output as collect_val_predictions, but gathers batches from a
    FastBatchData instance (vectorized numpy fancy-indexing) instead of a
    per-sample Dataset/DataLoader — the per-sample path is the same
    __getitem__-per-call bottleneck already eliminated from training; this
    closes the same gap for the final report's full-val-set evaluation.
    Indices are walked in order (0..N-1) so the (frame, 12 pairs) grouping
    generate_report relies on for the collapsed-per-frame view is preserved."""
    model.eval()
    all_probs, all_labels = [], []
    idx_all = np.arange(len(val_data))
    with torch.no_grad():
        for b0 in range(0, len(idx_all), batch_size):
            batch_idx = idx_all[b0:b0 + batch_size]
            ctx, offs, a1, a2, lbl, mask = val_data.get_batch(batch_idx)
            ctx, offs, a1, a2, mask = (
                ctx.to(dev, non_blocking=True).float(), offs.to(dev, non_blocking=True),
                a1.to(dev, non_blocking=True), a2.to(dev, non_blocking=True), mask.to(dev, non_blocking=True),
            )
            logits = model(ctx, a1, a2, offsets=offs, key_padding_mask=mask)
            all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            all_labels.append(lbl.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def cfg_str(cfg: dict) -> str:
    return (f"heads={cfg['n_heads']}, context_k={cfg['context_k']}, hidden={cfg['hidden_dim']}, "
            f"neg_ratio={cfg['neg_ratio']}, loss={cfg['loss_type']}")


def generate_report(probs, labels, history, variant_name: str, cfg: dict, out_dir: Path):
    out_dir = Path(out_dir)
    n_pairs = 12
    n_frames_val = len(labels) // n_pairs
    probs_r = probs.reshape(n_frames_val, n_pairs, 3)
    labels_r = labels.reshape(n_frames_val, n_pairs)

    fig, axes = plt.subplots(3, 2, figsize=(12, 15))
    save_data = {}

    best_idx = int(np.argmax(history['pair_macro_pr_auc']))
    best_epoch = history['eval_epoch'][best_idx]

    # Dummy/random baselines — what a classifier with zero discriminative power would score,
    # respecting only the true class proportions (no covariate information at all).
    # Loss: the plotted val_loss is train_fast()'s CLASS-WEIGHTED cross-entropy (rare classes
    # get ~10-20x weight), not plain unweighted CE — so the baseline must apply the same
    # weighting (reconstructed the same way train_fast() computes it, from neg_ratio and this
    # eval set's label counts) or the two aren't on a comparable scale. This is the loss of a
    # model that always outputs the true class-prior probabilities.
    # PR-AUC: for an uninformative/random-score classifier, AP converges to the positive
    # class's prevalence — so the random baseline is just the (macro) base rate, computed the
    # same two ways as the real metrics (per couple / per frame).
    label_counts = np.bincount(labels, minlength=3)
    p = label_counts / label_counts.sum()
    n_pos_total = max(int((labels > 0).sum()), 1)
    n1, n2 = max(int(label_counts[1]), 1), max(int(label_counts[2]), 1)
    n0 = max(cfg['neg_ratio'] * n_pos_total, 1)
    sampled_counts = np.array([n0, n1, n2], dtype=np.float64)
    class_weights = sampled_counts.sum() / (3 * sampled_counts)
    per_class_loss = -np.log(np.clip(p, 1e-12, None))
    random_loss = float(np.sum(label_counts * class_weights * per_class_loss) / np.sum(label_counts * class_weights))
    pair_random = float(np.mean([(labels == c).mean() for c in (1, 2)]))
    frame_random = float(np.mean([(labels_r == c).any(axis=1).mean() for c in (1, 2)]))

    axes[0, 0].plot(history['epoch'], history['train_loss'], label='train loss')
    axes[0, 0].plot(history['eval_epoch'], history['val_loss'], label='val loss')
    axes[0, 0].axhline(random_loss, color='gray', ls=':', alpha=0.7, label='random (class-prior) baseline')
    axes[0, 0].axvline(best_epoch, color='tab:green', ls='--', alpha=0.6)
    axes[0, 0].text(best_epoch, axes[0, 0].get_ylim()[1], f'best epoch ({best_epoch})',
                     color='tab:green', rotation=90, va='top', ha='right', fontsize=8)
    axes[0, 0].set_xlabel('epoch'); axes[0, 0].set_ylabel('loss')
    axes[0, 0].set_title('Loss'); axes[0, 0].legend()

    # macro PR-AUC (nt/nn only, 'none' excluded — see row 1/2 note below) over training, both
    # ways: "couples" = per-ordered-pair (row 1), "per frame" = collapsed-per-frame (row 2).
    # These are the exact same quantities train_fast() computes and logs each eval epoch.
    axes[0, 1].plot(history['eval_epoch'], history['pair_macro_pr_auc'], color='tab:blue', label='per couple (nt/nn)')
    axes[0, 1].plot(history['eval_epoch'], history['frame_macro_pr_auc'], color='tab:orange', label='per frame (nt/nn)')
    axes[0, 1].axhline(pair_random, color='tab:blue', ls=':', alpha=0.7, label='per couple, random baseline')
    axes[0, 1].axhline(frame_random, color='tab:orange', ls=':', alpha=0.7, label='per frame, random baseline')
    axes[0, 1].axvline(best_epoch, color='tab:green', ls='--', alpha=0.6)
    axes[0, 1].text(best_epoch, axes[0, 1].get_ylim()[1], f'best epoch ({best_epoch})',
                     color='tab:green', rotation=90, va='top', ha='right', fontsize=8)
    axes[0, 1].set_xlabel('epoch'); axes[0, 1].set_ylabel('macro PR-AUC')
    axes[0, 1].set_title('PR-AUC'); axes[0, 1].legend(fontsize=8)

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
    axes[1, 0].set_xlabel('FPR'); axes[1, 0].set_ylabel('TPR')
    axes[1, 0].set_title('Behaviors per mice couples (ROC)'); axes[1, 0].legend()
    axes[1, 1].set_xlabel('Recall'); axes[1, 1].set_ylabel('Precision')
    axes[1, 1].set_title('Behaviors per mice couples (PR)'); axes[1, 1].legend()

    # 'none' is aggregated the opposite way from nt/nn: a frame only counts as a true
    # "no interaction" frame if ALL 12 pairs are none (any single interacting pair means
    # something happened in that frame), so we take the min confidence across pairs rather
    # than the max. Using .any()/.max() for 'none' too would be trivially ~100% positive
    # (almost every individual pair already is 'none'), which is what row 1 already covers.
    for c, name in [(0, 'none'), (1, 'nt'), (2, 'nn')]:
        if c == 0:
            frame_true = (labels_r == c).all(axis=1).astype(int)
            frame_score = probs_r[:, :, c].min(axis=1)
        else:
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
    axes[2, 0].set_xlabel('FPR'); axes[2, 0].set_ylabel('TPR')
    axes[2, 0].set_title('Aggregated behaviors per frame (ROC)'); axes[2, 0].legend()
    axes[2, 1].set_xlabel('Recall'); axes[2, 1].set_ylabel('Precision')
    axes[2, 1].set_title('Aggregated behaviors per frame (PR)'); axes[2, 1].legend()

    fig.suptitle(f'{variant_name} — {cfg_str(cfg)}')
    fig.tight_layout()
    fig.savefig(out_dir / 'report.png', dpi=150)
    np.savez(out_dir / 'roc_pr_data.npz', **save_data)
