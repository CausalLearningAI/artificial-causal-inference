#!/bin/bash
#
# Extract coarse DINOv2/DINOv3 patch-grid embeddings for the annotated mice v1 pools only.
#
# Usage:
#   sbatch scripts/mice_behavior/extract_patch_embeddings.sh
#   ENCODER=dinov3 sbatch scripts/mice_behavior/extract_patch_embeddings.sh
#
#SBATCH --job-name=mice_patch_embed
#SBATCH --output=logs/mice_patch_embed_%j.out
#SBATCH --error=logs/mice_patch_embed_%j.err
#SBATCH --time=07:00:00
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

mkdir -p logs

ENCODER=${ENCODER:-dinov2}

python -u scripts/mice_behavior/extract_patch_embeddings.py \
    --encoder "${ENCODER}" \
    --grid-size 4 \
    --batch-size 32 \
    --num-workers 8 \
    --device cuda
