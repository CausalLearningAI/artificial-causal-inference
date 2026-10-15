#!/bin/bash
#
# Mask-U test job wrapper (rules: scripts/benchmark/eci_mask_u.sh): PY = maskqa | encode_mabe | encode_calms | train |
# lab (scripts/benchmark/mask_u/), EXTRA = its arguments. Staging in /localhome/$USER/<job>_masku (removed on exit);
# GPU utilisation / memory logged every 60 s (nvidia-smi) into the job log.
#SBATCH --job-name=masku
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
module load conda
conda activate crl
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1 HF_HUB_OFFLINE=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
export LOCAL_DIR=/localhome/$USER/${SLURM_JOB_ID:-local}_masku
mkdir -p $LOCAL_DIR
SMI=""
if nvidia-smi -L >/dev/null 2>&1; then
  nvidia-smi -L
  nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used --format=csv,noheader -l 60 | sed -u 's/^/[gpu] /' &
  SMI=$!
fi
trap 'rm -rf $LOCAL_DIR; [ -n "$SMI" ] && kill $SMI 2>/dev/null || true' EXIT
echo "host $(hostname) job ${SLURM_JOB_ID:-} py $PY extra ${EXTRA:-} local $LOCAL_DIR"
python -u scripts/benchmark/mask_u/$PY.py ${EXTRA:-}
