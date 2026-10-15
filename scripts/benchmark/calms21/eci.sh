#!/bin/bash
#
# CalMS21 ECI job wrapper: PY = eci_plan | eci_encode | eci_train | eci_probe (scripts/benchmark/calms21/), EXTRA = its
# arguments. GPU jobs: -p gpu --exclude=gpu242,gpu150,gpu230,gpu241 (never gpu100); CPU jobs: -p defaultp.
# Staging in /localhome/$USER/<job> (removed on exit). GPU jobs log nvidia-smi utilisation / memory every 60 s to
# results/vision/benchmark_calms21/logs/gpu_<job>.csv. Outputs: results/vision/benchmark_calms21/ (gitignored).
#SBATCH --job-name=calms_eci
#SBATCH --output=results/vision/benchmark_calms21/logs/%x_%j.out
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
module load conda
conda activate crl
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1 HF_HUB_OFFLINE=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
export LOCAL_DIR=/localhome/$USER/${SLURM_JOB_ID:-local}_calms
mkdir -p $LOCAL_DIR
SMI=""
if nvidia-smi -L >/dev/null 2>&1; then
  nvidia-smi -L
  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 60 \
    > results/vision/benchmark_calms21/logs/gpu_${SLURM_JOB_NAME:-x}_${SLURM_JOB_ID:-local}.csv 2>/dev/null &
  SMI=$!
fi
trap 'rm -rf $LOCAL_DIR; [ -n "$SMI" ] && kill $SMI 2>/dev/null || true' EXIT
echo "host $(hostname) job ${SLURM_JOB_ID:-} py $PY extra ${EXTRA:-} local $LOCAL_DIR"
df -h $LOCAL_DIR | tail -1
python -u scripts/benchmark/calms21/$PY.py ${EXTRA:-}
