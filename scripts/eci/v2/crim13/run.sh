#!/bin/bash
#
# ECI v2 on CRIM13 (validation check) - Slurm job wrapper for scripts/eci/v2/crim13/run.py. STEP = encode | train |
# score | report; EXTRA = its arguments (":" for "," because sbatch --export splits on commas). Staging in
# /localhome/$USER/<job>_<task>_v2c (removed on exit); GPU utilisation / memory every 60 s ([gpu] lines).
#   G="-p gpu --gres=gpu:1 --constraint=L40S|A100|A40 --exclude=gpu240,gpu241,gpu242,gpu150,gpu230,eta296,eta300"
#   sbatch -J v2crim_enc0 $G --export=ALL,STEP=encode,EXTRA="--shard 0" scripts/eci/v2/crim13/run.sh
#   sbatch -J v2crim_train $G ... STEP=train ; sbatch -J v2crim_score $G ... STEP=score
#   sbatch -J v2crim_report -p defaultp --export=ALL,STEP=report scripts/eci/v2/crim13/run.sh
#SBATCH --job-name=v2crim
#SBATCH --output=/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference/results/vision/eci_v2/crim13/logs/%x_%j.out
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
module load conda
conda activate crl
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
export LOCAL_DIR=/localhome/$USER/${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}_v2c
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
python -u scripts/eci/v2/crim13/run.py $STEP ${EXTRA:-}
echo "done"
