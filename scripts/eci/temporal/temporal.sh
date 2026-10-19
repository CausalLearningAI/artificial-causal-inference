#!/bin/bash
#
# Temporal-context study job wrapper (scripts/eci/temporal/temporal.py). STEP = plan | vjsmoke | pool | ctx | vj | geom |
# train | codes | score | report | sheets | cost | selftest; EXTRA = more arguments. Array tasks: SLURM_ARRAY_TASK_ID =
# the shard / part. Staging in /localhome/$USER/<job>_<task>_tmp (removed on exit); GPU utilisation / memory logged
# every 60 s into the job log ([gpu] lines, parsed by the 'cost' step).
#   GPU: sbatch -p gpu --gres=gpu:1 --constraint="L40S|A100|A40" --exclude=gpu240,gpu241,gpu242 -J tmp_<step> \
#          --export=ALL,STEP=<step> scripts/eci/temporal/temporal.sh
#   CPU: sbatch -p defaultp -J tmp_<step> --export=ALL,STEP=<step> scripts/eci/temporal/temporal.sh
#SBATCH --job-name=tmp
#SBATCH --output=/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference/results/vision/eci_temporal/logs/%x_%A_%a.out
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
module load conda
conda activate crl
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
export LOCAL_DIR=/localhome/$USER/${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}_tmp
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
PY="python -u scripts/eci/temporal/temporal.py"
if [ "$STEP" = "selftest" ]; then
  # 6 pool frames, one video's first 120 frames of the ctx pass (mask / token checks vs the stored diag set)
  $PY pool --shard 0 --limit 6
  $PY ctx --shard 0 --videos ${VIDEO:-0} --max-frames 120
  exit 0
fi
SH=""
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ] && [ "$STEP" != "train" ]; then SH="--shard $SLURM_ARRAY_TASK_ID"; fi
$PY $STEP $SH ${EXTRA:-}
