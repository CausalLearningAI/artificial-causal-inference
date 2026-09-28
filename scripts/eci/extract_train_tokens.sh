#!/bin/bash
#
# ECI step 1: final-layer DINOv2-base patch tokens + CLS for a stratified
# training subset of mice v1 (64 frames per observation, 432 observations).
#
# Usage:
#   sbatch scripts/eci/extract_train_tokens.sh
#   EXTRA_ARGS="--overwrite" sbatch scripts/eci/extract_train_tokens.sh
#
#SBATCH --job-name=eci_train_tokens
#SBATCH --output=logs/eci_train_tokens_%j.out
#SBATCH --error=logs/eci_train_tokens_%j.err
#SBATCH --time=03:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1

module load conda
conda activate crl

export PYTHONUNBUFFERED=1
set -euo pipefail

cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
mkdir -p logs

python -u scripts/eci/extract_train_tokens.py \
    --resolution 224 \
    --n-per-obs 64 \
    --seed 0 \
    --batch-size 128 \
    --num-workers 8 \
    --device cuda \
    ${EXTRA_ARGS:-}
