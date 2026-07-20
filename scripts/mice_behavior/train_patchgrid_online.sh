#!/bin/bash
#
# "Online" DINOv2 patch-grid training (all 256 raw patch tokens, no pooling,
# encoded once via a frozen DINOv2 forward pass, no permanent cache). See
# train_patchgrid_online.py docstring.
#
# Usage:
#   sbatch scripts/mice_behavior/train_patchgrid_online.sh
#
#SBATCH --job-name=mice_patchgrid_online
#SBATCH --output=logs/mice_patchgrid_online_%j.out
#SBATCH --error=logs/mice_patchgrid_online_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --gres=gpu:H100:1

module load conda
conda activate crl

export PYTHONUNBUFFERED=1
set -euo pipefail

cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference

mkdir -p logs

python -u scripts/mice_behavior/train_patchgrid_online.py \
    --max-train-frames 200000 \
    --n-epochs 30 \
    --patience 10 \
    --batch-size 256 \
    --encode-batch-size 256 \
    --num-workers 16
