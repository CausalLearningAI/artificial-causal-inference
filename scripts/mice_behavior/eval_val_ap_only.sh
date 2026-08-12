#!/bin/bash
#
# Recover full-val AP for a saved patchgrid256 checkpoint whose training job timed out
# before its own final eval pass. See eval_val_ap_only.py docstring.
#
# Usage:
#   TAG=res504 INPUT_SIZE=504 sbatch scripts/mice_behavior/eval_val_ap_only.sh
#
#SBATCH --job-name=eval_val_ap_only
#SBATCH --output=logs/eval_val_ap_only_%j.out
#SBATCH --error=logs/eval_val_ap_only_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=500G
#SBATCH --gres=gpu:H100:1

module load conda
conda activate crl
export PYTHONUNBUFFERED=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
mkdir -p logs

TAG=${TAG:?must set TAG}
INPUT_SIZE_ARGS=""
if [ -n "${INPUT_SIZE:-}" ]; then
    INPUT_SIZE_ARGS="--input-size ${INPUT_SIZE}"
fi
BLUR_TO_ARGS=""
if [ -n "${BLUR_TO:-}" ]; then
    BLUR_TO_ARGS="--blur-to ${BLUR_TO}"
fi
CROSS_ATTN_DIM_ARGS=""
if [ -n "${CROSS_ATTN_DIM:-}" ]; then
    CROSS_ATTN_DIM_ARGS="--cross-attn-dim ${CROSS_ATTN_DIM}"
fi
PATCH_POOL_DIM_ARGS=""
if [ -n "${PATCH_POOL_DIM:-}" ]; then
    PATCH_POOL_DIM_ARGS="--patch-pool-dim ${PATCH_POOL_DIM}"
fi

python -u scripts/mice_behavior/eval_val_ap_only.py --tag "${TAG}" ${INPUT_SIZE_ARGS} ${BLUR_TO_ARGS} ${CROSS_ATTN_DIM_ARGS} ${PATCH_POOL_DIM_ARGS}
