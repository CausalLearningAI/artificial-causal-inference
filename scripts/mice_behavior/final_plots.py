"""
Produce the final 2x2 ROC/PR plot for the best mice-behavior classifier found
by grid_search.py.

Row 1: per-ordered-pair classification — ROC curve (left) / PR curve (right),
       one curve per class (none/nt/nn), on held-out val samples.
Row 2: collapsed "did behavior X happen anywhere in this frame" task — ROC
       curve (left) / PR curve (right), one curve per behavior (nt, nn),
       using max score over the 12 ordered pairs vs frame-level ground truth.

Saves: results/mice_behavior/final_roc_pr.png
       results/mice_behavior/final_roc_pr_data.npz (raw arrays for re-plotting)

Usage:
    python scripts/mice_behavior/final_plots.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.mice_behavior.dataset import MouseOPairDataset, collate_fn
from src.mice_behavior.model import MouseOPairClassifier
from src.mice_behavior.pools import load_obs_to_pool_map

RESULTS_DIR = Path('./results/mice_behavior')
DATASET_DIR = Path('./dataset')
LABEL_NAMES = ['none', 'nt', 'nn']


def main():
    with open(RESULTS_DIR / 'best_config.json') as f:
        info = json.load(f)
    cfg, val_pools = info['cfg'], set(info['val_pools'])
    encoder, token, emb_dim = info['encoder'], info['token'], info['emb_dim']

    pair_labels_path = DATASET_DIR / 'mice' / 'v1' / 'pair_labels.parquet'
    annotations_csv = DATASET_DIR / 'mice' / 'v1' / 'annotations.csv'
    embeddings_path = DATASET_DIR / 'mice' / 'v1' / 'embeddings' / 'full' / encoder / token / 'embeddings.npy'

    obs_to_pool = load_obs_to_pool_map(Path('./data'))
    all_obs = pd.read_parquet(pair_labels_path)['observation_id'].unique().tolist()
    val_obs = [o for o in all_obs if obs_to_pool[o] in val_pools]

    val_ds = MouseOPairDataset(
        str(annotations_csv), str(pair_labels_path), str(embeddings_path),
        obs_ids=val_obs, context_k=cfg['context_k'], emb_dim=emb_dim,
    )

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MouseOPairClassifier(
        emb_dim=emb_dim, n_heads=cfg['n_heads'], hidden_dim=cfg['hidden_dim'],
    ).to(dev)
    model.load_state_dict(torch.load(RESULTS_DIR / 'best_model.pt', map_location=dev))
    model.eval()

    loader = torch.utils.data.DataLoader(
        val_ds, batch_size=512, shuffle=False, num_workers=0, collate_fn=collate_fn
    )

    all_probs, all_labels = [], []
    with torch.no_grad():
        for ctx, offsets, a1, a2, labels, mask in loader:
            logits = model(ctx.to(dev), a1.to(dev), a2.to(dev), offsets=offsets.to(dev), key_padding_mask=mask.to(dev))
            all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            all_labels.append(labels.numpy())
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)

    n_opairs = 12
    n_frames_val = len(labels) // n_opairs
    assert len(labels) % n_opairs == 0
    probs_r = probs.reshape(n_frames_val, n_opairs, 3)
    labels_r = labels.reshape(n_frames_val, n_opairs)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    save_data = {}

    # --- Row 1: per-pair ---
    for c, name in enumerate(LABEL_NAMES):
        y_true = (labels == c).astype(int)
        y_score = probs[:, c]
        fpr, tpr, _ = roc_curve(y_true, y_score)
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        roc_auc = roc_auc_score(y_true, y_score)
        pr_auc = average_precision_score(y_true, y_score)
        axes[0, 0].plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.3f})')
        axes[0, 1].plot(rec, prec, label=f'{name} (AP={pr_auc:.3f})')
        save_data[f'pair_{name}_fpr'] = fpr
        save_data[f'pair_{name}_tpr'] = tpr
        save_data[f'pair_{name}_prec'] = prec
        save_data[f'pair_{name}_rec'] = rec
        save_data[f'pair_{name}_roc_auc'] = roc_auc
        save_data[f'pair_{name}_pr_auc'] = pr_auc

    axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.3)
    axes[0, 0].set_xlabel('False Positive Rate'); axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title('Per-pair ROC'); axes[0, 0].legend()
    axes[0, 1].set_xlabel('Recall'); axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title('Per-pair PR'); axes[0, 1].legend()

    # --- Row 2: collapsed per-frame ---
    pos_score = probs_r[:, :, 1] + probs_r[:, :, 2]
    for c, name in [(1, 'nt'), (2, 'nn')]:
        frame_true = (labels_r == c).any(axis=1).astype(int)
        frame_score = probs_r[:, :, c].max(axis=1)
        fpr, tpr, _ = roc_curve(frame_true, frame_score)
        prec, rec, _ = precision_recall_curve(frame_true, frame_score)
        roc_auc = roc_auc_score(frame_true, frame_score)
        pr_auc = average_precision_score(frame_true, frame_score)
        axes[1, 0].plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.3f})')
        axes[1, 1].plot(rec, prec, label=f'{name} (AP={pr_auc:.3f})')
        save_data[f'frame_{name}_fpr'] = fpr
        save_data[f'frame_{name}_tpr'] = tpr
        save_data[f'frame_{name}_prec'] = prec
        save_data[f'frame_{name}_rec'] = rec
        save_data[f'frame_{name}_roc_auc'] = roc_auc
        save_data[f'frame_{name}_pr_auc'] = pr_auc

    axes[1, 0].plot([0, 1], [0, 1], 'k--', alpha=0.3)
    axes[1, 0].set_xlabel('False Positive Rate'); axes[1, 0].set_ylabel('True Positive Rate')
    axes[1, 0].set_title('Collapsed per-frame ROC ("did X happen anywhere")'); axes[1, 0].legend()
    axes[1, 1].set_xlabel('Recall'); axes[1, 1].set_ylabel('Precision')
    axes[1, 1].set_title('Collapsed per-frame PR'); axes[1, 1].legend()

    fig.suptitle(f'Mouse behavior classifier — best config: {cfg}')
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'final_roc_pr.png', dpi=150)
    np.savez(RESULTS_DIR / 'final_roc_pr_data.npz', **save_data)
    print(f'Saved {RESULTS_DIR / "final_roc_pr.png"} and final_roc_pr_data.npz')


if __name__ == '__main__':
    main()
