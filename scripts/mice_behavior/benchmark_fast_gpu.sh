#!/bin/bash
#SBATCH --job-name=mice_fast_gpu_bench
#SBATCH --output=logs/mice_fast_gpu_bench_%j.out
#SBATCH --error=logs/mice_fast_gpu_bench_%j.err
#SBATCH --time=00:20:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:A40:1

module load conda
conda activate crl
export PYTHONUNBUFFERED=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
mkdir -p logs

python -u -c "
import sys, pandas as pd, time
sys.path.insert(0, '.')
from src.mice_behavior.train import train_fast
from src.mice_behavior.pools import load_obs_to_pool_map
import random

pair_labels_path = 'dataset/mice/v1/pair_labels.parquet'
annotations_csv = 'dataset/mice/v1/annotations.csv'
cls_embeddings_path = 'dataset/mice/v1/embeddings/full/dinov2/class_l-2/embeddings.npy'
pg_embeddings_path = 'dataset/mice/v1/embeddings/full/dinov2/patch_grid4/embeddings.npy'
pg_global_idx_path = 'dataset/mice/v1/embeddings/full/dinov2/patch_grid4/global_idx.npy'

obs_to_pool = load_obs_to_pool_map('./data')
all_obs = pd.read_parquet(pair_labels_path)['observation_id'].unique().tolist()
pools = sorted({obs_to_pool[o] for o in all_obs})
rng = random.Random(42)
shuffled = pools[:]; rng.shuffle(shuffled)
n_val = max(1, int(len(shuffled)*0.2))
val_pool_set = set(shuffled[:n_val])
train_obs = [o for o in all_obs if obs_to_pool[o] not in val_pool_set]
val_obs = [o for o in all_obs if obs_to_pool[o] in val_pool_set]

print('=== patch-grid GPU-resident fast benchmark (5 epochs) ===', flush=True)
t0 = time.time()
result = train_fast(
    annotations_csv=annotations_csv, pair_labels_parquet=pair_labels_path,
    embeddings_path=cls_embeddings_path, output_dir='/tmp/mice_fast_gpu_bench_pg',
    train_obs_ids=train_obs, val_obs_ids=val_obs,
    context_k=2, emb_dim=768, n_heads=1, hidden_dim=256, n_epochs=5,
    neg_ratio=10, device='cuda', seed=42, loss_type='ce', verbose=True, eval_every=1,
    use_patch_grid=True, patch_embeddings_path=pg_embeddings_path,
    patch_global_idx_path=pg_global_idx_path, n_patches=16, lr=3e-4,
)
print('patch-grid GPU-resident fast: total', time.time()-t0, 's for 5 epochs', flush=True)
print('BENCHMARK DONE', flush=True)
"
