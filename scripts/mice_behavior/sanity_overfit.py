"""
One-off Karpathy-style sanity check: can the model memorize pair-level labels
on data it's directly trained on (train_obs_ids == val_obs_ids, no
regularization, no early stopping)? If pair-level macro PR-AUC can't be driven
well above the ~0.01-0.02 seen on held-out data even here, that's strong
evidence the ceiling is an information/architecture limit (missing per-mouse
identity features), not a code bug. Delete after use.

Usage:
    python scripts/mice_behavior/sanity_overfit.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.mice_behavior.build_pair_labels import build_pair_labels
from src.mice_behavior.pools import load_obs_to_pool_map
from src.mice_behavior.train import train

DATA_DIR, DATASET_DIR = Path('./data'), Path('./dataset')
pair_labels_path = build_pair_labels(DATA_DIR, DATASET_DIR, overwrite=False)
annotations_csv = DATASET_DIR / 'mice/v1/annotations.csv'
cls_embeddings_path = DATASET_DIR / 'mice/v1/embeddings/full/dinov2/class_l-2/embeddings.npy'
n_frames = sum(1 for _ in open(annotations_csv)) - 1
emb_dim = cls_embeddings_path.stat().st_size // (4 * n_frames)

obs_to_pool = load_obs_to_pool_map(DATA_DIR)
all_obs = pd.read_parquet(pair_labels_path)['observation_id'].unique().tolist()

result = train(
    annotations_csv=str(annotations_csv), pair_labels_parquet=str(pair_labels_path),
    embeddings_path=str(cls_embeddings_path), output_dir='./results/mice_behavior/search/tmp/sanity_overfit',
    train_obs_ids=all_obs, val_obs_ids=all_obs, context_k=2, emb_dim=emb_dim,
    n_heads=8, hidden_dim=256, n_epochs=60, neg_ratio=10, loss_type='ce',
    device='cuda', seed=42, verbose=True, eval_every=5, early_stop_patience=None,
    dropout=0.0, weight_decay=0.0,
)
print('SANITY RESULT best_pr_auc (train==val, pair-level, should be near memorization-high if pipeline is sound):', result['best_pr_auc'])
print(result['best_per_class'])
