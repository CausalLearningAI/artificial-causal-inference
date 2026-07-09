#!/bin/bash
#SBATCH --job-name=mice_pg_sanity
#SBATCH --output=logs/mice_pg_sanity_%j.out
#SBATCH --error=logs/mice_pg_sanity_%j.err
#SBATCH --time=00:20:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

module load conda
conda activate crl
export PYTHONUNBUFFERED=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
mkdir -p logs

python -u -c "
import sys, pandas as pd, time
sys.path.insert(0, '.')
from src.mice_behavior.train import train
from src.mice_behavior.pools import load_obs_to_pool_map

pair_labels_path = 'dataset/mice/v1/pair_labels.parquet'
annotations_csv = 'dataset/mice/v1/annotations.csv'
cls_embeddings_path = 'dataset/mice/v1/embeddings/full/dinov2/class_l-2/embeddings.npy'
pg_embeddings_path = 'dataset/mice/v1/embeddings/full/dinov2/patch_grid4/embeddings.npy'
pg_global_idx_path = 'dataset/mice/v1/embeddings/full/dinov2/patch_grid4/global_idx.npy'

obs_to_pool = load_obs_to_pool_map('./data')
all_obs = pd.read_parquet(pair_labels_path)['observation_id'].unique().tolist()
pools = sorted({obs_to_pool[o] for o in all_obs})
train_pools = set(pools[:2])
val_pools = set(pools[2:3])
train_obs = [o for o in all_obs if obs_to_pool[o] in train_pools]
val_obs = [o for o in all_obs if obs_to_pool[o] in val_pools]
print('obs', len(train_obs), len(val_obs), flush=True)

t0 = time.time()
result = train(
    annotations_csv=annotations_csv, pair_labels_parquet=pair_labels_path,
    embeddings_path=cls_embeddings_path, output_dir='/tmp/mice_sanity_patchgrid_slurm',
    train_obs_ids=train_obs, val_obs_ids=val_obs,
    context_k=2, emb_dim=768, n_heads=1, hidden_dim=32,
    n_epochs=2, neg_ratio=3, device='cuda', seed=42,
    loss_type='ce', verbose=True,
    use_patch_grid=True, patch_embeddings_path=pg_embeddings_path,
    patch_global_idx_path=pg_global_idx_path, n_patches=16,
)
print('best_pr_auc', result['best_pr_auc'], result['best_per_class'], flush=True)
print('history', result['history'], flush=True)
print('PATCH-GRID SANITY CHECK PASSED', time.time()-t0, flush=True)
"
