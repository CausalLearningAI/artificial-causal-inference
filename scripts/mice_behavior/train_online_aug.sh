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
# 180G, not 150G: the JPEG cache is populated LAZILY and keeps growing every epoch (each epoch
# resamples negatives, pulling in frames no earlier epoch touched), so peak RSS lands near the
# end of training, not at the start. A 120G allocation OOM-killed four arms of the previous
# sweep mid-run, which is the most expensive possible time to die.
#SBATCH --mem=180G
#SBATCH --gres=gpu:H100:1
#
# Every #SBATCH above can be overridden per-submission on the sbatch command line, which is
# how the A100/gpu-partition sweeps are launched (H100s exist only on gpu100):
#   sbatch --partition=gpu --gres=gpu:A100:1 --time=14:00:00 --mem=120G ... this_script

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

# Deconfounded ERM (ours). A flag, not a value, so it is only forwarded when asked for and
# every prior run stays byte-identical.
DERM_ARGS=""
if [ "${DERM:-0}" = "1" ]; then
    DERM_ARGS="--derm"
fi

WANDB_ARGS=""
if [ "${WANDB:-0}" = "1" ]; then
    WANDB_ARGS="--wandb"
fi

SMOKE_ARGS=""
if [ "${SMOKE:-0}" = "1" ]; then
    SMOKE_ARGS="--smoke"
fi

# Stage B of the ssl_dapt arm: start the encoder from the SSL-adapted checkpoint instead of the
# hub weights. Only forwarded when set, so every other run is byte-identical to before.
INIT_ENCODER_ARGS=""
if [ -n "${INIT_ENCODER:-}" ]; then
    INIT_ENCODER_ARGS="--init-encoder ${INIT_ENCODER}"
fi

# only forward these when set, so an unset var keeps the value inherited from best_cfg
# (SEED unset => gsf.SEED, i.e. every prior run reproduces; PHOTO_STRENGTH unset => 1.0)
OVERRIDE_ARGS=""
for kv in "lr:LR" "weight-decay:WEIGHT_DECAY" "dropout:DROPOUT" "seed:SEED" \
          "photo-strength:PHOTO_STRENGTH"; do
    flag="${kv%%:*}"; var="${kv##*:}"; val="${!var:-}"
    [ -n "$val" ] && OVERRIDE_ARGS="${OVERRIDE_ARGS} --${flag} ${val}"
done

python -u scripts/mice_behavior/train_online_aug.py \
    --context-k "${CONTEXT_K:-2}" \
    --augment "${AUGMENT:-d4}" \
    --neg-ratio "${NEG_RATIO:-1}" \
    --max-train-frames "${MAX_TRAIN_FRAMES:-300000}" \
    --input-size "${INPUT_SIZE:-224}" \
    --pixel-source "${PIXEL_SOURCE:-0}" \
    --val-pools "${VAL_POOLS:-}" \
    --pool-grid "${POOL_GRID:-0}" \
    --cross-attn-dim "${CROSS_ATTN_DIM:-0}" \
    --patch-pool-dim "${PATCH_POOL_DIM:-0}" \
    --n-train-pools "${N_TRAIN_POOLS:-0}" \
    --env-key "${ENV_KEY:-none}" \
    --derm-floor "${DERM_FLOOR:-0.02}" \
    --vrex-beta "${VREX_BETA:-0}" \
    --vrex-warmup-epochs "${VREX_WARMUP_EPOCHS:-5}" \
    --batch-size "${BATCH_SIZE:-64}" \
    --read-workers "${READ_WORKERS:-32}" \
    --decode-workers "${DECODE_WORKERS:-16}" \
    --val-monitor-size "${VAL_MONITOR_SIZE:-12500}" \
    --lr-decay-epochs "${LR_DECAY_EPOCHS:-6}" \
    --n-epochs "${N_EPOCHS:-20}" \
    --patience "${PATIENCE:-8}" \
    --optimizer "${OPTIMIZER:-adam}" \
    --warmup-epochs "${WARMUP_EPOCHS:-0}" \
    --stride "${STRIDE:-1}" \
    --unfreeze-blocks "${UNFREEZE_BLOCKS:-0}" \
    --ft-mode "${FT_MODE:-full}" \
    --encoder-lr "${ENCODER_LR:-1e-5}" \
    --layerwise-decay "${LAYERWISE_DECAY:-0.65}" \
    --patch-selfattn-dim "${PATCH_SELFATTN_DIM:-0}" \
    --pool-queries "${POOL_QUERIES:-1}" \
    ${OVERRIDE_ARGS} ${MOTION_ARGS} ${WANDB_ARGS} ${SMOKE_ARGS} ${JPEG_CACHE_ARGS} \
    ${INIT_ENCODER_ARGS} ${DERM_ARGS} \
    --tag "${TAG:-online_aug}"
