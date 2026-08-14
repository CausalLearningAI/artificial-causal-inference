#!/bin/bash
#
# Stage A of the ssl_dapt arm: domain-adaptive SSL on unlabelled mice v1 frames.
# See ssl_dapt.py's docstring for the hypothesis, the objective and the collapse guards.
#
#   SMOKE=1 sbatch scripts/mice_behavior/ssl_dapt.sh        # ~5 min pipeline check
#   sbatch scripts/mice_behavior/ssl_dapt.sh               # the real run
#
#SBATCH --job-name=ssl_dapt
#SBATCH --output=logs/ssl_dapt_%j.out
#SBATCH --error=logs/ssl_dapt_%j.err
#SBATCH --time=10:00:00
# gpu, not gpu100: the A100 partition is where the rest of this ablation runs, so arms stay
# measured on comparable hardware. Override with PARTITION/GRES for an H100.
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
# 120G, not the 180G train_online_aug.sh needs. That number is set by a JPEG cache that grows
# lazily every epoch as fresh negatives pull in new frames; here the frame set is FIXED at
# startup, so the cache reaches its full ~10.3 GiB during the first read and never grows again.
# The cache is also memory-mapped from one file, i.e. clean file-backed pages the kernel can
# reclaim rather than anonymous pages it must OOM-kill on.
#SBATCH --mem=120G
#SBATCH --gres=gpu:A100:1

module load conda
conda activate crl
export PYTHONUNBUFFERED=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
mkdir -p logs

SMOKE_ARGS=""
if [ "${SMOKE:-0}" = "1" ]; then
    SMOKE_ARGS="--smoke"
fi

WANDB_ARGS=""
if [ "${WANDB:-1}" = "1" ]; then
    WANDB_ARGS="--wandb"
fi

python -u scripts/mice_behavior/ssl_dapt.py \
    --tag "${TAG:-ssl_dapt}" \
    --input-size "${INPUT_SIZE:-448}" \
    --frame-stride "${FRAME_STRIDE:-10}" \
    --include-labeled-obs "${INCLUDE_LABELED_OBS:-1}" \
    --unfreeze-blocks "${UNFREEZE_BLOCKS:-2}" \
    --encoder-lr "${ENCODER_LR:-3e-5}" \
    --layerwise-decay "${LAYERWISE_DECAY:-0.65}" \
    --mask-ratio "${MASK_RATIO:-0.5}" \
    --target-layers "${TARGET_LAYERS:-4}" \
    --n-epochs "${N_EPOCHS:-12}" \
    --warmup-epochs "${WARMUP_EPOCHS:-1}" \
    --batch-size "${BATCH_SIZE:-64}" \
    --augment "${AUGMENT:-d4_photo}" \
    --read-workers "${READ_WORKERS:-32}" \
    --decode-workers "${DECODE_WORKERS:-16}" \
    --jpeg-cache-file "${JPEG_CACHE_FILE:-dataset/mice/v1/jpegcache_ssl}" \
    ${WANDB_ARGS} ${SMOKE_ARGS}
