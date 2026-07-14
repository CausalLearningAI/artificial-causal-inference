"""
Full training run of the per-frame behavior classifier — no mouse-identity
conditioning, one sample per annotated frame (not per ordered pair), multi-label
target [has_nt, has_nn]. Same 80/20 pool split and encoder as train_cls_final.py,
for a direct comparison against the pairwise model.

Writes results/mice_behavior/frame/{best_model.pt, history.json, config.json}.

Usage:
    python scripts/mice_behavior/train_frame_final.py
"""
import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.mice_behavior.build_pair_labels import build_pair_labels
from src.mice_behavior.pools import load_obs_to_pool_map
from src.mice_behavior.train import train_frame

DATA_DIR = Path('./data')
DATASET_DIR = Path('./dataset')
RESULTS_DIR = Path('./results/mice_behavior')
SEED = 42
ENCODER, TOKEN = 'dinov2', 'class_l-2'

# n_heads/context_k/hidden_dim match train_cls_final.py's CFG for comparability.
CFG = dict(n_heads=8, context_k=2, hidden_dim=256, neg_ratio=10)
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

    out_dir = RESULTS_DIR / 'frame'
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Training per-frame model: {CFG}, {N_EPOCHS} epochs, eval every epoch...')
    result = train_frame(
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
        device='cuda',
        seed=SEED,
        verbose=True,
        use_patch_grid=False,
        eval_every=1,
    )
    history = result['history']
    print(f"Best macro AP: {result['best_ap']:.4f}  {result['best_per_label']}")

    with open(out_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    with open(out_dir / 'config.json', 'w') as f:
        json.dump({'cfg': CFG, 'val_pools': sorted(val_pool_set), 'n_epochs': N_EPOCHS,
                   'best_ap': result['best_ap'], 'best_per_label': result['best_per_label'],
                   'emb_dim': emb_dim}, f, indent=2)


if __name__ == '__main__':
    main()
