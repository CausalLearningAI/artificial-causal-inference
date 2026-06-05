#!/bin/bash
#
# Train mouse pairwise behavior classifier (cross-attention over temporal frame embeddings).
#
# Usage:
#   sbatch scripts/mice_behavior/train.sh
#
#SBATCH --job-name=mice_behavior
#SBATCH --output=logs/mice_behavior_%j.out
#SBATCH --error=logs/mice_behavior_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:H100:1

module load conda
conda activate crl

export PYTHONUNBUFFERED=1
set -euo pipefail

cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference

mkdir -p logs

python -u scripts/mice_behavior/run_train.py \
    --encoder dinov2 \
    --token class_l-2 \
    --context-k 2 \
    --n-heads 8 \
    --hidden-dim 256 \
    --epochs 30 \
    --batch-size 512 \
    --lr 1e-3 \
    --val-frac 0.2 \
    --device cuda \
    --seed 42 \
    --overwrite-labels
