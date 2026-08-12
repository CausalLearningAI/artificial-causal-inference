#!/bin/bash
#
# "Online" DINOv2 patch-grid training (all 256 raw patch tokens, no pooling,
# encoded once via a frozen DINOv2 forward pass, no permanent cache). See
# train_patchgrid_online.py docstring.
#
# Usage (env vars override defaults; TAG/CROSS_ATTN_DIM/PATCH_DROPOUT/PATCH_NOISE_STD/
# FRAME_DROPOUT let you fire off several regularization variants without clobbering
# each other's results dir):
#   sbatch scripts/mice_behavior/train_patchgrid_online.sh
#   TAG=reg_b CROSS_ATTN_DIM=128 PATCH_DROPOUT=0.2 FRAME_DROPOUT=0.25 sbatch scripts/mice_behavior/train_patchgrid_online.sh
#
#SBATCH --job-name=mice_patchgrid_online
#SBATCH --output=logs/mice_patchgrid_online_%j.out
#SBATCH --error=logs/mice_patchgrid_online_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --gres=gpu:1080ti:1

module load conda
conda activate crl

export PYTHONUNBUFFERED=1
set -euo pipefail

cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference

mkdir -p logs

TAG=${TAG:-}
CROSS_ATTN_DIM=${CROSS_ATTN_DIM:-192}
PATCH_DROPOUT=${PATCH_DROPOUT:-0.0}
PATCH_NOISE_STD=${PATCH_NOISE_STD:-0.0}
FRAME_DROPOUT=${FRAME_DROPOUT:-0.0}
MAX_TRAIN_FRAMES=${MAX_TRAIN_FRAMES:-200000}
N_EPOCHS=${N_EPOCHS:-20}
PATIENCE=${PATIENCE:-8}
VAL_MONITOR_SIZE=${VAL_MONITOR_SIZE:-50000}
# frame reads are NFS-LATENCY-bound (~100ms/frame, decode is only ~2-3ms of it), so
# workers should exceed the CPU count -- they sit blocked on I/O, not computing.
NUM_WORKERS=${NUM_WORKERS:-16}
# separate from NUM_WORKERS: the training gather reads from the RAM token cache and
# holds workers*prefetch batches in flight, so a large value here OOMs (see py help).
GATHER_WORKERS=${GATHER_WORKERS:-6}
CONTEXT_K_ARGS=""
if [ -n "${CONTEXT_K:-}" ]; then
    CONTEXT_K_ARGS="--context-k ${CONTEXT_K}"
fi
LAYERNORM_ARGS=""
if [ "${USE_LAYERNORM:-0}" = "1" ]; then
    LAYERNORM_ARGS="--use-layernorm"
fi
LOSS=${LOSS:-bce}
FOCAL_GAMMA=${FOCAL_GAMMA:-2.0}
INPUT_SIZE_ARGS=""
if [ -n "${INPUT_SIZE:-}" ]; then
    INPUT_SIZE_ARGS="--input-size ${INPUT_SIZE}"
fi
LR_SCHEDULE=${LR_SCHEDULE:-none}
PATCH_POOL_DIM=${PATCH_POOL_DIM:-0}
LR_DECAY_EPOCHS_ARGS=""
if [ -n "${LR_DECAY_EPOCHS:-}" ]; then
    LR_DECAY_EPOCHS_ARGS="--lr-decay-epochs ${LR_DECAY_EPOCHS}"
fi
NEG_RATIO=${NEG_RATIO:-10}
BATCH_SIZE=${BATCH_SIZE:-512}

python -u scripts/mice_behavior/train_patchgrid_online.py \
    --max-train-frames "${MAX_TRAIN_FRAMES}" \
    --n-epochs "${N_EPOCHS}" \
    --patience "${PATIENCE}" \
    --batch-size "${BATCH_SIZE}" \
    --encode-batch-size 256 \
    --num-workers "${NUM_WORKERS}" \
    --gather-workers "${GATHER_WORKERS}" \
    --tag "${TAG}" \
    --cross-attn-dim "${CROSS_ATTN_DIM}" \
    --patch-pool-dim "${PATCH_POOL_DIM}" \
    --patch-dropout "${PATCH_DROPOUT}" \
    --patch-noise-std "${PATCH_NOISE_STD}" \
    --frame-dropout "${FRAME_DROPOUT}" \
    --neg-ratio "${NEG_RATIO}" \
    --val-monitor-size "${VAL_MONITOR_SIZE}" \
    ${CONTEXT_K_ARGS} \
    --loss "${LOSS}" \
    --focal-gamma "${FOCAL_GAMMA}" \
    ${LAYERNORM_ARGS} \
    ${INPUT_SIZE_ARGS} \
    --lr-schedule "${LR_SCHEDULE}" \
    ${LR_DECAY_EPOCHS_ARGS}
