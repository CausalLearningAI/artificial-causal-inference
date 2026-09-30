#!/bin/bash
#
# Foreground pipeline step 2: foreground-only DINOv2 patch tokens (448, no crop) of the 1 fps
# frames of all 432 videos -> dataset/mice/v1/eci/train_tokens/dinov2_base_l-1_fg448_fps1/
# (16 tasks x 27 videos, ~80M tokens, ~125 GB). Needs scripts/eci/fg_background.sh first.
# Finished shards are skipped (resubmit to resume).
#
# Usage: sbatch scripts/eci/fg_extract_train.sh
#   mask v3 + motion (token 2 frames earlier), ~115 GB:
#   EXTRA_ARGS="--rule v3 --motion-delta 2 --batch-size 64 --out-dir dataset/mice/v1/eci/train_tokens/dinov2_base_l-1_fgv3_fps1_d2" \
#       sbatch scripts/eci/fg_extract_train.sh
#   ants full frame (no mask, rule all, 25% of the 1024 patches per 1 fps frame, 39.3M tokens, 54 GB, ~12 min):
#   DOMAIN=ants EXTRA_ARGS="--rule all --patch-frac 0.25 --out-dir dataset/ants/eci/train_tokens/dinov2_base_l-1_antsfull448_fps1_pf25" \
#       sbatch --export=ALL --partition=gpu100 scripts/eci/fg_extract_train.sh
#
#SBATCH --job-name=eci_fg_tok
#SBATCH --output=logs/eci_fg_tok_%A_%a.out
#SBATCH --error=logs/eci_fg_tok_%A_%a.err
#SBATCH --array=0-15
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --gres=gpu:1

module load conda
conda activate crl

export PYTHONUNBUFFERED=1
set -euo pipefail

cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
mkdir -p logs

python -u scripts/eci/fg_extract_train.py --domain "${DOMAIN:-mice}" --task ${SLURM_ARRAY_TASK_ID} --n-tasks 16 --num-workers 22 ${EXTRA_ARGS:-}
