#!/bin/bash
#
# Merge (pairs kept at >= 1% of frames) + verify (64 random frames recomputed from scratch, per-shard codes_max
# alignment) + delete shards for scripts/eci/spatial_encode_all.sh. Writes codes/<SAE>_pairs, _zones, _blobs.
#
#SBATCH --job-name=eci_spatial_merge
#SBATCH --output=logs/eci_spatial_merge_%j.out
#SBATCH --error=logs/eci_spatial_merge_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

module load conda
conda activate crl

export PYTHONUNBUFFERED=1
set -euo pipefail

cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
python -u scripts/eci/spatial_encode_all.py --domain "${DOMAIN:-mice}" --sae "${SAE:?set SAE}" --n-shards "${N_SHARDS:-24}" \
    --merge --verify --delete-shards
