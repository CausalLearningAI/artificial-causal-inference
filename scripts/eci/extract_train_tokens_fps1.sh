#!/bin/bash
#
# ECI step 1 (v2): final-layer DINOv2-base patch tokens + CLS for EVERY 5th frame
# (1 frame per second) of all 432 mice v1 observations = 518,400 frames, ~204 GB fp16.
# Frames are stored in a seeded random order, split into 16 shards (array tasks).
# Output: dataset/mice/v1/eci/train_tokens/dinov2_base_l-1_fps1/
#
# Usage:
#   jid=$(sbatch --parsable scripts/eci/extract_train_tokens_fps1.sh)
#   # after all 16 tasks finish (CPU, ~1 min): concatenated metadata + checks
#   python scripts/eci/extract_train_tokens.py --stride 5 --n-shards 16 --finalize
#
#SBATCH --job-name=eci_tok_fps1
#SBATCH --output=logs/eci_tok_fps1_%A_%a.out
#SBATCH --error=logs/eci_tok_fps1_%A_%a.err
#SBATCH --array=0-15
#SBATCH --time=03:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --gres=gpu:1

module load conda
conda activate crl

export PYTHONUNBUFFERED=1
set -euo pipefail

cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
mkdir -p logs

python -u scripts/eci/extract_train_tokens.py \
    --stride 5 \
    --n-shards 16 \
    --shard ${SLURM_ARRAY_TASK_ID} \
    --resolution 224 \
    --seed 0 \
    --batch-size 128 \
    --num-workers 22 \
    --device cuda \
    ${EXTRA_ARGS:-}
