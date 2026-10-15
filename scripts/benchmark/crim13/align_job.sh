#!/bin/bash
#
# CRIM13 alignment job wrapper: PY = align_plan | align_encode | align_train | align_probe | align_sheets
# (scripts/benchmark/crim13/), EXTRA = its arguments. GPU jobs: -p gpu --exclude=gpu242,gpu150,gpu230,gpu241,gpu240
# (never gpu100); CPU jobs: -p defaultp. Staging in /localhome/$USER/<job> (removed on exit). GPU jobs log nvidia-smi
# utilisation / memory every 60 s to results/vision/benchmark_crim13/align/logs/gpu_<job>.csv.
#SBATCH --job-name=crim_align
#SBATCH --output=results/vision/benchmark_crim13/align/logs/%x_%j.out
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
module load conda
conda activate crl
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1 HF_HUB_OFFLINE=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
export LOCAL_DIR=/localhome/$USER/${SLURM_JOB_ID:-local}
mkdir -p $LOCAL_DIR
SMI=""
if nvidia-smi -L >/dev/null 2>&1; then
  nvidia-smi -L
  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 60 \
    > results/vision/benchmark_crim13/align/logs/gpu_${SLURM_JOB_NAME:-x}_${SLURM_JOB_ID:-local}.csv 2>/dev/null &
  SMI=$!
fi
trap 'rm -rf $LOCAL_DIR; [ -n "$SMI" ] && kill $SMI 2>/dev/null || true' EXIT
echo "host $(hostname) job ${SLURM_JOB_ID:-} py $PY extra ${EXTRA:-} local $LOCAL_DIR"
python -u scripts/benchmark/crim13/$PY.py ${EXTRA:-}
