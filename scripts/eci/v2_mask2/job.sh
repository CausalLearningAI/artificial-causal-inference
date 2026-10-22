#!/bin/bash
#
# ECI v2 thin-part mask test job wrapper. SCRIPT = python file in scripts/eci/v2_mask2/, EXTRA = its arguments (":"
# instead of "," because sbatch --export splits on commas; "@" instead of spaces is NOT needed, spaces are fine).
# Staging in /localhome/$USER/<job>_<task>_v2m2, or $TMPDIR on nodes without /localhome (removed on exit); GPU
# utilisation / memory logged every 60 s ([gpu] lines).
#   sbatch -J v2m2_x -p gpu --gres=gpu:1 --constraint="L40S|A100|A40" --exclude=gpu240,gpu241,gpu242 \
#     --export=ALL,SCRIPT=explore.py,EXTRA="--src mice" scripts/eci/v2_mask2/job.sh
#SBATCH --job-name=v2_mask2
#SBATCH --output=/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference/results/vision/eci_v2_mask2/logs/%x_%j.out
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
module load conda
conda activate crl
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
if [ -d /localhome ] && mkdir -p /localhome/$USER 2>/dev/null; then BASE=/localhome/$USER; else BASE=${TMPDIR:-/tmp}; fi
export LOCAL_DIR=$BASE/${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}_v2m2
mkdir -p $LOCAL_DIR
SMI=""
if nvidia-smi -L >/dev/null 2>&1; then
  nvidia-smi -L
  nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used --format=csv,noheader -l 60 | sed -u 's/^/[gpu] /' &
  SMI=$!
fi
trap 'rm -rf $LOCAL_DIR; [ -n "$SMI" ] && kill $SMI 2>/dev/null || true' EXIT
df -h $BASE | tail -1
echo "host $(hostname) job ${SLURM_JOB_ID:-} script $SCRIPT extra ${EXTRA:-} git $(git rev-parse --short HEAD)"
EXTRA=${EXTRA:-}; EXTRA=${EXTRA//:/,}
python -u scripts/eci/v2_mask2/$SCRIPT ${EXTRA:-}
echo "done"
