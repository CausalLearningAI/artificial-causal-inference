#!/bin/bash
#
# NN/NT round 3 job wrapper (scripts/eci/nnnt/r3.py). STEP = geom | train | score | sheets; EXTRA = arguments.
# Staging in /localhome/$USER/<job>_<task>_r3 (removed on exit); GPU utilisation / memory logged every 60 s ([gpu] lines).
#   sbatch -p gpu --gres=gpu:1 --constraint="L40S|A100|A40" --exclude=gpu240,gpu241,gpu242 \
#     --export=ALL,STEP=train,EXTRA="--arms R2:R3" scripts/eci/nnnt/r3.sh      (geom: CPU, -p defaultp)
#SBATCH --job-name=nnnt_r3
#SBATCH --output=/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference/results/vision/eci_nnnt/logs/%x_%A_%a.out
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
module load conda
conda activate crl
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
export LOCAL_DIR=/localhome/$USER/${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}_r3
mkdir -p $LOCAL_DIR
SMI=""
if nvidia-smi -L >/dev/null 2>&1; then
  nvidia-smi -L
  nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used --format=csv,noheader -l 60 | sed -u 's/^/[gpu] /' &
  SMI=$!
fi
trap 'rm -rf $LOCAL_DIR; [ -n "$SMI" ] && kill $SMI 2>/dev/null || true' EXIT
df -h /localhome | tail -1
echo "host $(hostname) job ${SLURM_JOB_ID:-} task ${SLURM_ARRAY_TASK_ID:-} step $STEP extra ${EXTRA:-}"
EXTRA=${EXTRA:-}; EXTRA=${EXTRA//:/,}  # sbatch --export splits on commas: pass lists with ":" (e.g. --seeds 0:1:2)
PY="python -u scripts/eci/nnnt/r3.py"
$PY $STEP ${EXTRA:-}
