#!/bin/bash
# Job wrapper for scripts/eci/v2/v21_store.py. EXTRA = its arguments (":" for ","). Staging in /localhome/$USER/<job>_<task>
# (removed on exit); GPU utilisation + memory logged every 300 s.
#SBATCH --output=/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference/results/vision/eci_v2_0B/store_v21/logs/%x_%A_%a.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
module load conda
conda activate crl
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error PYTHONWARNINGS=ignore
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
if [ -d /localhome ] && mkdir -p /localhome/$USER 2>/dev/null; then BASE=/localhome/$USER; else BASE=${TMPDIR:-/tmp}; fi
export LOCAL_DIR=$BASE/${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}
mkdir -p $LOCAL_DIR
SMI=""
if nvidia-smi -L >/dev/null 2>&1; then
  nvidia-smi -L
  nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used --format=csv,noheader -l 300 | sed -u 's/^/[gpu] /' &
  SMI=$!
fi
trap 'rm -rf $LOCAL_DIR; [ -n "$SMI" ] && kill $SMI 2>/dev/null || true' EXIT
echo "host $(hostname) job ${SLURM_JOB_ID:-} task ${SLURM_ARRAY_TASK_ID:-} extra ${EXTRA:-} git $(git rev-parse --short HEAD 2>/dev/null || echo na)"
EXTRA=${EXTRA:-}; EXTRA=${EXTRA//:/,}
python -u scripts/eci/v2/v21_store.py ${EXTRA}
echo done
