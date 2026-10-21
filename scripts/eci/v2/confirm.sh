#!/bin/bash
#
# ECI v2 freeze round job wrapper (scripts/eci/v2/freeze.py). STEP = mtrain | mscore | ...; EXTRA = arguments (lists
# with ":" instead of "," because sbatch --export splits on commas). Staging in /localhome/$USER/<job>_<task>_v2f
# (removed on exit); GPU utilisation / memory logged every 60 s ([gpu] lines).
#   sbatch -J v2_mtrain_F1 -p gpu --gres=gpu:1 --constraint="L40S|A100|A40" --exclude=gpu240,gpu241,gpu242 \
#     --export=ALL,STEP=mtrain,EXTRA="--arms F1 --seeds 6-15" scripts/eci/v2/freeze.sh
#SBATCH --job-name=v2_confirm
#SBATCH --output=/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference/results/vision/eci_v2/logs/%x_%j.out
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
module load conda
conda activate crl
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
export LOCAL_DIR=/localhome/$USER/${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}_v2f
mkdir -p $LOCAL_DIR
SMI=""
if nvidia-smi -L >/dev/null 2>&1; then
  nvidia-smi -L
  nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used --format=csv,noheader -l 60 | sed -u 's/^/[gpu] /' &
  SMI=$!
fi
trap 'rm -rf $LOCAL_DIR; [ -n "$SMI" ] && kill $SMI 2>/dev/null || true' EXIT
df -h /localhome | tail -1
echo "host $(hostname) job ${SLURM_JOB_ID:-} step $STEP extra ${EXTRA:-} git $(git rev-parse --short HEAD)"
EXTRA=${EXTRA:-}; EXTRA=${EXTRA//:/,}
python -u scripts/eci/v2/confirm.py $STEP ${EXTRA:-}
echo "done"
