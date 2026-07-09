"""
Full training run of the patch-grid (attention-pooled 4x4 DINOv2 tokens)
variant of the mouse behavior classifier, on the standard 80/20 pool split,
with per-epoch validation (eval_every=1) so a full loss curve is available.

Produces results/mice_behavior/patchgrid_report.png:
    Row 0: train/val loss curve (left), val macro PR-AUC over epochs (right)
    Row 1: per-ordered-pair ROC (left) / PR (right)
    Row 2: collapsed per-frame ROC (left) / PR (right)
Also saves patchgrid_history.json and patchgrid_roc_pr_data.npz.

Usage:
    python scripts/mice_behavior/train_patchgrid_final.py
"""
import json
import random
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
from src.mice_behavior.build_pair_labels import build_pair_labels
from src.mice_behavior.dataset import MousePairDatasetPatchGrid, collate_fn
from src.mice_behavior.model import MouseBehaviorClassifier
from src.mice_behavior.pools import load_obs_to_pool_map
from src.mice_behavior.train import train_fast

DATA_DIR = Path('./data')
DATASET_DIR = Path('./dataset')
RESULTS_DIR = Path('./results/mice_behavior')
SEED = 42
ENCODER, TOKEN = 'dinov2', 'class_l-2'
PATCH_GRID_DIR = DATASET_DIR / 'mice' / 'v1' / 'embeddings' / 'full' / 'dinov2' / 'patch_grid4'
LABEL_NAMES = ['none', 'nt', 'nn']

# Reasonable defaults consistent with the CLS-only baseline discussed this session.
CFG = dict(n_heads=1, context_k=2, hidden_dim=256, neg_ratio=10, loss_type='ce')
N_EPOCHS = 100


def main():
    pair_labels_path = build_pair_labels(DATA_DIR, DATASET_DIR, overwrite=False)
    annotations_csv = DATASET_DIR / 'mice' / 'v1' / 'annotations.csv'
    cls_embeddings_path = DATASET_DIR / 'mice' / 'v1' / 'embeddings' / 'full' / ENCODER / TOKEN / 'embeddings.npy'
    n_frames = sum(1 for _ in open(annotations_csv)) - 1
    emb_dim = cls_embeddings_path.stat().st_size // (4 * n_frames)

    obs_to_pool = load_obs_to_pool_map(DATA_DIR)
    all_obs = pd.read_parquet(pair_labels_path)['observation_id'].unique().tolist()
    pools = sorted({obs_to_pool[o] for o in all_obs})

    rng = random.Random(SEED)
    shuffled = pools[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * 0.2))
    val_pool_set = set(shuffled[:n_val])
    train_obs = [o for o in all_obs if obs_to_pool[o] not in val_pool_set]
    val_obs = [o for o in all_obs if obs_to_pool[o] in val_pool_set]
    print(f'Split: {len(train_obs)} train obs / {len(val_obs)} val obs ({len(pools)-n_val}/{n_val} pools)')

    out_dir = RESULTS_DIR / 'patchgrid_final'
    print(f'Training patch-grid model (fast vectorized path): {CFG}, {N_EPOCHS} epochs, eval every epoch...')
    result = train_fast(
        annotations_csv=str(annotations_csv),
        pair_labels_parquet=str(pair_labels_path),
        embeddings_path=str(cls_embeddings_path),
        output_dir=str(out_dir),
        train_obs_ids=train_obs,
        val_obs_ids=val_obs,
        context_k=CFG['context_k'],
        emb_dim=emb_dim,
        n_heads=CFG['n_heads'],
        hidden_dim=CFG['hidden_dim'],
        n_epochs=N_EPOCHS,
        neg_ratio=CFG['neg_ratio'],
        loss_type=CFG['loss_type'],
        lr=3e-4,  # CLS is stable at the 1e-3 default; the extra PatchAttnPool stage in this
                  # variant repeatedly collapsed into a dead (ln(3)-loss) state at 1e-3 even
                  # with grad clipping down to max_norm=0.5 — lower lr specifically here.
        device='cuda',
        seed=SEED,
        verbose=True,
        use_patch_grid=True,
        patch_embeddings_path=str(PATCH_GRID_DIR / 'embeddings.npy'),
        patch_global_idx_path=str(PATCH_GRID_DIR / 'global_idx.npy'),
        n_patches=16,
        eval_every=1,
    )
    history = result['history']
    print(f"Best macro PR-AUC: {result['best_pr_auc']:.4f}  {result['best_per_class']}")

    with open(RESULTS_DIR / 'patchgrid_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    with open(RESULTS_DIR / 'patchgrid_config.json', 'w') as f:
        json.dump({'cfg': CFG, 'val_pools': sorted(val_pool_set), 'n_epochs': N_EPOCHS,
                   'best_pr_auc': result['best_pr_auc'], 'best_per_class': result['best_per_class'],
                   'emb_dim': emb_dim}, f, indent=2)

    # --- Re-run best checkpoint on val to get full probs/labels for ROC/PR plots ---
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MouseBehaviorClassifier(
        emb_dim=emb_dim, n_heads=CFG['n_heads'], hidden_dim=CFG['hidden_dim'], use_patch_grid=True,
    ).to(dev)
    model.load_state_dict(torch.load(out_dir / 'best_model.pt', map_location=dev))
    model.eval()

    val_ds = MousePairDatasetPatchGrid(
        str(annotations_csv), str(pair_labels_path), cls_embeddings_path=str(cls_embeddings_path),
        embeddings_path=str(PATCH_GRID_DIR / 'embeddings.npy'), global_idx_path=str(PATCH_GRID_DIR / 'global_idx.npy'),
        obs_ids=val_obs, context_k=CFG['context_k'], emb_dim=emb_dim, n_patches=16,
    )
    loader = torch.utils.data.DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=0, collate_fn=collate_fn)

    all_probs, all_labels = [], []
    with torch.no_grad():
        for ctx, offsets, a1, a2, labels, mask in loader:
            logits = model(ctx.to(dev), a1.to(dev), a2.to(dev), offsets=offsets.to(dev), key_padding_mask=mask.to(dev))
            all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            all_labels.append(labels.numpy())
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)

    n_pairs = 12
    n_frames_val = len(labels) // n_pairs
    probs_r = probs.reshape(n_frames_val, n_pairs, 3)
    labels_r = labels.reshape(n_frames_val, n_pairs)

    fig, axes = plt.subplots(3, 2, figsize=(12, 15))
    save_data = {}

    # Row 0: loss curve + val macro PR-AUC
    axes[0, 0].plot(history['epoch'], history['train_loss'], label='train loss')
    axes[0, 0].plot(history['eval_epoch'], history['val_loss'], label='val loss')
    axes[0, 0].set_xlabel('epoch'); axes[0, 0].set_ylabel('loss')
    axes[0, 0].set_title('Train / val loss'); axes[0, 0].legend()

    axes[0, 1].plot(history['eval_epoch'], history['macro_pr_auc'], color='tab:green')
    axes[0, 1].set_xlabel('epoch'); axes[0, 1].set_ylabel('macro PR-AUC')
    axes[0, 1].set_title('Val macro PR-AUC over training')

    # Row 1: per-pair ROC/PR
    for c, name in enumerate(LABEL_NAMES):
        y_true = (labels == c).astype(int)
        y_score = probs[:, c]
        fpr, tpr, _ = roc_curve(y_true, y_score)
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        roc_auc = roc_auc_score(y_true, y_score)
        pr_auc = average_precision_score(y_true, y_score)
        axes[1, 0].plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.3f})')
        axes[1, 1].plot(rec, prec, label=f'{name} (AP={pr_auc:.3f})')
        save_data[f'pair_{name}_fpr'] = fpr; save_data[f'pair_{name}_tpr'] = tpr
        save_data[f'pair_{name}_prec'] = prec; save_data[f'pair_{name}_rec'] = rec
        save_data[f'pair_{name}_roc_auc'] = roc_auc; save_data[f'pair_{name}_pr_auc'] = pr_auc
    axes[1, 0].plot([0, 1], [0, 1], 'k--', alpha=0.3)
    axes[1, 0].set_xlabel('FPR'); axes[1, 0].set_ylabel('TPR'); axes[1, 0].set_title('Per-pair ROC'); axes[1, 0].legend()
    axes[1, 1].set_xlabel('Recall'); axes[1, 1].set_ylabel('Precision'); axes[1, 1].set_title('Per-pair PR'); axes[1, 1].legend()

    # Row 2: collapsed per-frame ROC/PR
    for c, name in [(1, 'nt'), (2, 'nn')]:
        frame_true = (labels_r == c).any(axis=1).astype(int)
        frame_score = probs_r[:, :, c].max(axis=1)
        fpr, tpr, _ = roc_curve(frame_true, frame_score)
        prec, rec, _ = precision_recall_curve(frame_true, frame_score)
        roc_auc = roc_auc_score(frame_true, frame_score)
        pr_auc = average_precision_score(frame_true, frame_score)
        axes[2, 0].plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.3f})')
        axes[2, 1].plot(rec, prec, label=f'{name} (AP={pr_auc:.3f})')
        save_data[f'frame_{name}_fpr'] = fpr; save_data[f'frame_{name}_tpr'] = tpr
        save_data[f'frame_{name}_prec'] = prec; save_data[f'frame_{name}_rec'] = rec
        save_data[f'frame_{name}_roc_auc'] = roc_auc; save_data[f'frame_{name}_pr_auc'] = pr_auc
    axes[2, 0].plot([0, 1], [0, 1], 'k--', alpha=0.3)
    axes[2, 0].set_xlabel('FPR'); axes[2, 0].set_ylabel('TPR'); axes[2, 0].set_title('Collapsed per-frame ROC'); axes[2, 0].legend()
    axes[2, 1].set_xlabel('Recall'); axes[2, 1].set_ylabel('Precision'); axes[2, 1].set_title('Collapsed per-frame PR'); axes[2, 1].legend()

    fig.suptitle(f'Patch-grid (attention-pooled) mouse behavior classifier — {CFG}')
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'patchgrid_report.png', dpi=150)
    np.savez(RESULTS_DIR / 'patchgrid_roc_pr_data.npz', **save_data)
    print(f'Saved {RESULTS_DIR / "patchgrid_report.png"}')


if __name__ == '__main__':
    main()
