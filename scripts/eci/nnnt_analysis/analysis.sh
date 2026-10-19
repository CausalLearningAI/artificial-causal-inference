#!/bin/bash
#
# NN/NT analysis job wrapper (scripts/eci/nnnt_analysis/analysis.py). STEP = labels | diag | temporal.
#   sbatch -p defaultp --export=ALL,STEP=labels scripts/eci/nnnt_analysis/analysis.sh
#   sbatch -p gpu --gres=gpu:1 --constraint="A100" --exclude=gpu240,gpu241,gpu242 --export=ALL,STEP=diag \
#     scripts/eci/nnnt_analysis/analysis.sh
# Staging in /localhome/$USER/<job>_na (removed on exit); GPU utilisation / memory logged every 60 s ([gpu] lines).
#SBATCH --job-name=nnnt_analysis
#SBATCH --output=/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference/results/vision/eci_nnnt_analysis/logs/%x_%j.out
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
module load conda
conda activate crl
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
export LOCAL_DIR=/localhome/$USER/${SLURM_JOB_ID:-local}_na
mkdir -p $LOCAL_DIR
SMI=""
if nvidia-smi -L >/dev/null 2>&1; then
  nvidia-smi -L
  nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used --format=csv,noheader -l 60 | sed -u 's/^/[gpu] /' &
  SMI=$!
fi
trap 'rm -rf $LOCAL_DIR; [ -n "$SMI" ] && kill $SMI 2>/dev/null || true' EXIT
df -h /localhome | tail -1
echo "host $(hostname) job ${SLURM_JOB_ID:-} step $STEP"
python -u scripts/eci/nnnt_analysis/analysis.py $STEP
