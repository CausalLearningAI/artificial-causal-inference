"""
Threshold-free diagnostic for the mouse pairwise behavior classifier.

1. Per-class ROC-AUC / PR-AUC (average precision) on val — signal vs. no-signal,
   independent of the argmax threshold used during training.
2. Per-frame pair-prediction variance — checks whether the 12 ordered-pair
   predictions for the same frame are actually distinguishable from each other,
   which they can only be if the query mechanism extracts pair-specific info.
3. Collapsed "any interaction in this frame" task — max score over the 12 pairs
   vs. frame-level any-positive label. If this scores much better than the
   per-pair task, the model has learned "something is happening" but not "to whom."
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.mice_behavior.dataset import MousePairDataset, collate_fn
from src.mice_behavior.model import MouseBehaviorClassifier
from src.mice_behavior.pools import load_obs_to_pool_map

import random

DATA_DIR = Path('./data')
DATASET_DIR = Path('./dataset')
RESULTS_DIR = Path('./results/mice_behavior')
ENCODER, TOKEN = 'dinov2', 'class_l-2'
SEED = 42
VAL_FRAC = 0.2
CONTEXT_K = 2
N_HEADS = 1
HIDDEN_DIM = 256

LABEL_NAMES = ['none', 'nt', 'nn']


def main():
    pair_labels_path = DATASET_DIR / 'mice' / 'v1' / 'pair_labels.parquet'
    annotations_csv = DATASET_DIR / 'mice' / 'v1' / 'annotations.csv'
    embeddings_path = (
        DATASET_DIR / 'mice' / 'v1' / 'embeddings' / 'full' / ENCODER / TOKEN / 'embeddings.npy'
    )

    obs_to_pool = load_obs_to_pool_map(DATA_DIR)
    all_obs = pd.read_parquet(pair_labels_path)['observation_id'].unique().tolist()
    pools = sorted({obs_to_pool[o] for o in all_obs})
    rng = random.Random(SEED)
    rng.shuffle(pools)
    n_val = max(1, int(len(pools) * VAL_FRAC))
    val_pool_set = set(pools[:n_val])
    val_obs = [o for o in all_obs if obs_to_pool[o] in val_pool_set]
    print(f'val pools: {sorted(val_pool_set)}')

    n_frames = sum(1 for _ in open(annotations_csv)) - 1
    emb_dim = embeddings_path.stat().st_size // (4 * n_frames)

    print('Building val dataset (same split as training)...')
    val_ds = MousePairDataset(
        str(annotations_csv), str(pair_labels_path), str(embeddings_path),
        obs_ids=val_obs, context_k=CONTEXT_K, emb_dim=emb_dim,
    )

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MouseBehaviorClassifier(emb_dim=emb_dim, n_heads=N_HEADS, hidden_dim=HIDDEN_DIM).to(dev)
    model.load_state_dict(torch.load(RESULTS_DIR / 'best_model.pt', map_location=dev))
    model.eval()

    loader = torch.utils.data.DataLoader(
        val_ds, batch_size=512, shuffle=False, num_workers=0, collate_fn=collate_fn
    )

    all_probs, all_labels = [], []
    with torch.no_grad():
        for ctx, offsets, a1, a2, labels, mask in loader:
            logits = model(ctx.to(dev), a1.to(dev), a2.to(dev), offsets=offsets.to(dev), key_padding_mask=mask.to(dev))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())
    probs = np.concatenate(all_probs)   # (N, 3)
    labels = np.concatenate(all_labels)  # (N,)

    print(f'\nval samples: {len(labels):,}')
    for c, name in enumerate(LABEL_NAMES):
        y_true = (labels == c).astype(int)
        prevalence = y_true.mean()
        roc = roc_auc_score(y_true, probs[:, c])
        pr = average_precision_score(y_true, probs[:, c])
        print(f'  class={name:5s}  prevalence={prevalence:.5f}  ROC-AUC={roc:.4f}  PR-AUC={pr:.4f}  '
              f'(PR-AUC/prevalence={pr/prevalence:.2f}x)')

    # --- Per-frame pair-prediction variance check ---
    # samples are ordered: for each annotated frame, 12 consecutive rows (one per ordered pair)
    n_pairs = 12
    n_frames_val = len(labels) // n_pairs
    assert len(labels) % n_pairs == 0, 'unexpected ordering, cannot reshape'
    probs_r = probs.reshape(n_frames_val, n_pairs, 3)
    labels_r = labels.reshape(n_frames_val, n_pairs)

    # std across the 12 pairs of P(nt)+P(nn) score, per frame, averaged over frames
    pos_score = probs_r[:, :, 1] + probs_r[:, :, 2]  # (n_frames, 12)
    within_frame_std = pos_score.std(axis=1).mean()
    overall_std = pos_score.std()
    print(f'\nwithin-frame std of P(positive) across the 12 pairs: {within_frame_std:.5f}')
    print(f'overall std of P(positive) across all samples:        {overall_std:.5f}')
    print(f'  (ratio within/overall = {within_frame_std/overall_std:.3f}; '
          f'near 1.0 => pairs are NOT distinguished, variance is purely frame-driven)')

    # --- Collapsed "any interaction in this frame" task ---
    frame_any_true = (labels_r > 0).any(axis=1).astype(int)
    frame_max_score = pos_score.max(axis=1)
    roc_any = roc_auc_score(frame_any_true, frame_max_score)
    pr_any = average_precision_score(frame_any_true, frame_max_score)
    print(f'\ncollapsed "any interaction in frame" task (n_frames={n_frames_val:,}, '
          f'prevalence={frame_any_true.mean():.4f}):')
    print(f'  ROC-AUC={roc_any:.4f}  PR-AUC={pr_any:.4f}')

    # --- Per-behavior-type collapsed (identity-free) task: is nt/nn happening anywhere in frame ---
    for c, name in [(1, 'nt'), (2, 'nn')]:
        frame_true_c = (labels_r == c).any(axis=1).astype(int)
        frame_score_c = probs_r[:, :, c].max(axis=1)
        roc_c = roc_auc_score(frame_true_c, frame_score_c)
        pr_c = average_precision_score(frame_true_c, frame_score_c)
        prev_c = frame_true_c.mean()
        print(f'collapsed "{name} happening in frame" task (prevalence={prev_c:.4f}): '
              f'ROC-AUC={roc_c:.4f}  PR-AUC={pr_c:.4f}  (enrichment={pr_c/prev_c:.2f}x)')


if __name__ == '__main__':
    main()
