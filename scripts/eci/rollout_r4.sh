#!/bin/bash
#
# R4 rollout job wrapper (scripts/eci/rollout_r4.py). STEP = plan | check | prepass | train | encode | merge | summaries
# | align; DOMAIN = mice | ants_v2 | ants_v3 | frogs; EXTRA = more arguments. Staging in /localhome/$USER/<job>_r4
# (removed on exit); GPU utilisation / memory logged every 60 s into the job log. Partition / GPU flags on the command
# line, e.g.
#   GPU: sbatch -p gpu --gres=gpu:1 --constraint="L40S|A100|A40" --exclude=gpu242,gpu150,gpu230,gpu241,gpu240 \
#          --cpus-per-task=24 --mem=96G --time=03:00:00 --array=0-13 --export=ALL,STEP=encode,DOMAIN=mice scripts/eci/rollout_r4.sh
#   CPU: sbatch -p defaultp --cpus-per-task=16 --mem=64G --time=00:30:00 --export=ALL,STEP=plan,DOMAIN=mice scripts/eci/rollout_r4.sh
#   (encode: SLURM_ARRAY_TASK_ID = the shard)
#SBATCH --job-name=r4
#SBATCH --output=results/vision/eci_rollout_r4/logs/%x_%A_%a.out
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
module load conda
conda activate crl
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
export LOCAL_DIR=/localhome/$USER/${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}_r4
mkdir -p $LOCAL_DIR
SMI=""
if nvidia-smi -L >/dev/null 2>&1; then
  nvidia-smi -L
  nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used --format=csv,noheader -l 60 | sed -u 's/^/[gpu] /' &
  SMI=$!
fi
trap 'rm -rf $LOCAL_DIR; [ -n "$SMI" ] && kill $SMI 2>/dev/null || true' EXIT
df -h /localhome | tail -1
echo "host $(hostname) job ${SLURM_JOB_ID:-} task ${SLURM_ARRAY_TASK_ID:-} step $STEP domain ${DOMAIN:-} extra ${EXTRA:-}"
SH=""
if [ "$STEP" = "encode" ] && [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then SH="--shard $SLURM_ARRAY_TASK_ID"; fi
DOM=""; [ -n "${DOMAIN:-}" ] && DOM="--domain $DOMAIN"
python -u scripts/eci/rollout_r4.py $STEP $DOM $SH ${EXTRA:-}
