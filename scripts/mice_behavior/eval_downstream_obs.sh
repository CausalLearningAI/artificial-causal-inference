#!/bin/bash
#
# Phase 0.3: per-observation / per-pool aggregate behavior-rate accuracy for a trained
# checkpoint -- the quantity PPCI actually consumes. See eval_downstream_obs.py docstring.
#
# Usage:
#   TAG=res448 INPUT_SIZE=448 sbatch scripts/mice_behavior/eval_downstream_obs.sh
#
#SBATCH --job-name=eval_downstream_obs
#SBATCH --output=logs/eval_downstream_obs_%j.out
#SBATCH --error=logs/eval_downstream_obs_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=500G
#SBATCH --gres=gpu:1

module load conda
conda activate crl
export PYTHONUNBUFFERED=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
mkdir -p logs

TAG=${TAG:?must set TAG}
EXTRA=""
[ -n "${INPUT_SIZE:-}" ] && EXTRA="$EXTRA --input-size ${INPUT_SIZE}"
[ -n "${CROSS_ATTN_DIM:-}" ] && EXTRA="$EXTRA --cross-attn-dim ${CROSS_ATTN_DIM}"
[ -n "${PATCH_POOL_DIM:-}" ] && EXTRA="$EXTRA --patch-pool-dim ${PATCH_POOL_DIM}"

python -u scripts/mice_behavior/eval_downstream_obs.py --tag "${TAG}" ${EXTRA}
