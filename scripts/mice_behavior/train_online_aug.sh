#!/bin/bash
#
# Online per-batch encoding from a JPEG-bytes RAM cache, with optional D4 augmentation.
# See train_online_aug.py docstring.
#   AUGMENT=d4 CONTEXT_K=2 sbatch scripts/mice_behavior/train_online_aug.sh
#
#SBATCH --job-name=online_aug
#SBATCH --output=logs/online_aug_%j.out
#SBATCH --error=logs/online_aug_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=gpu100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=150G
#SBATCH --gres=gpu:H100:1

module load conda
conda activate crl
export PYTHONUNBUFFERED=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
mkdir -p logs

JPEG_CACHE_ARGS=""
if [ -n "${JPEG_CACHE_FILE:-}" ]; then
    JPEG_CACHE_ARGS="--jpeg-cache-file ${JPEG_CACHE_FILE}"
fi

MOTION_ARGS=""
if [ "${USE_MOTION:-0}" = "1" ]; then
    MOTION_ARGS="--use-motion"
fi

WANDB_ARGS=""
if [ "${WANDB:-0}" = "1" ]; then
    WANDB_ARGS="--wandb"
fi

# only forward these when set, so an unset var keeps the value inherited from best_cfg
OVERRIDE_ARGS=""
for kv in "lr:LR" "weight-decay:WEIGHT_DECAY" "dropout:DROPOUT"; do
    flag="${kv%%:*}"; var="${kv##*:}"; val="${!var:-}"
    [ -n "$val" ] && OVERRIDE_ARGS="${OVERRIDE_ARGS} --${flag} ${val}"
done

python -u scripts/mice_behavior/train_online_aug.py \
    --context-k "${CONTEXT_K:-2}" \
    --augment "${AUGMENT:-d4}" \
    --neg-ratio "${NEG_RATIO:-1}" \
    --max-train-frames "${MAX_TRAIN_FRAMES:-300000}" \
    --input-size "${INPUT_SIZE:-224}" \
    --batch-size "${BATCH_SIZE:-64}" \
    --read-workers "${READ_WORKERS:-32}" \
    --decode-workers "${DECODE_WORKERS:-16}" \
    --val-monitor-size "${VAL_MONITOR_SIZE:-12500}" \
    --lr-decay-epochs "${LR_DECAY_EPOCHS:-6}" \
    --n-epochs "${N_EPOCHS:-20}" \
    --patience "${PATIENCE:-8}" \
    --optimizer "${OPTIMIZER:-adam}" \
    --warmup-epochs "${WARMUP_EPOCHS:-0}" \
    ${OVERRIDE_ARGS} ${MOTION_ARGS} ${WANDB_ARGS} ${JPEG_CACHE_ARGS} \
    --tag "${TAG:-online_aug}"
