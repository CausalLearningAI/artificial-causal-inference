#!/bin/bash
#
# Job wrapper for scripts/eci/v2/mask4_val.py. EXTRA = its arguments (":" for "," because sbatch --export splits
# on commas). Staging in /localhome/$USER/<job>_<task>_tmp (removed on exit); GPU utilisation logged every 120 s.
#   GPU: sbatch -J m4cap -p gpu --gres=gpu:1 --constraint="L40S|A100|A40" --exclude=gpu240,gpu241,gpu242,gpu278 --export=ALL,EXTRA="capture --dom mice --ser v1" scripts/eci/v2/mask4.sh
#SBATCH --job-name=m4
#SBATCH --output=/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference/results/vision/eci_v2_0B/mask_approval_v2/logs/%x_%j_%a.out
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
module load conda
conda activate crl
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
if [ -d /localhome ] && mkdir -p /localhome/$USER 2>/dev/null; then BASE=/localhome/$USER; else BASE=${TMPDIR:-/tmp}; fi
export LOCAL_DIR=$BASE/${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}_tmp
mkdir -p $LOCAL_DIR
SMI=""
if nvidia-smi -L >/dev/null 2>&1; then
  nvidia-smi -L
  nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used --format=csv,noheader -l 120 | sed -u 's/^/[gpu] /' &
  SMI=$!
fi
trap 'rm -rf $LOCAL_DIR; [ -n "$SMI" ] && kill $SMI 2>/dev/null || true' EXIT
df -h $BASE | tail -1
echo "host $(hostname) job ${SLURM_JOB_ID:-} task ${SLURM_ARRAY_TASK_ID:-} extra ${EXTRA:-} git $(git rev-parse --short HEAD 2>/dev/null || echo na)"
EXTRA=${EXTRA:-}; EXTRA=${EXTRA//:/,}; EXTRA=${EXTRA//@TASK@/${SLURM_ARRAY_TASK_ID:-0}}
python -u scripts/eci/v2/mask4_val.py ${EXTRA:-}
echo "done"
