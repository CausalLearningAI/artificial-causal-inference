#!/bin/bash
#
# Encoder-study job wrapper (= eci.sh + GPU utilisation logged every 60 s): PY = eci_enc_encode | eci_enc_probe |
# eci_enc_train (scripts/benchmark/mabe22/), EXTRA = its arguments. Staging in /localhome/$USER/<job>_mabe, removed on
# exit. Submitted by eci_encoders.sh.
#SBATCH --job-name=mabe_enc
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

module load conda
conda activate crl
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1 HF_HUB_OFFLINE=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
export LOCAL_DIR=/localhome/$USER/${SLURM_JOB_ID:-local}_mabe
mkdir -p $LOCAL_DIR
GPULOG=""
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used --format=csv,noheader -l 60 | sed -u 's/^/[gpu] /' &
    GPULOG=$!
fi
trap 'rm -rf $LOCAL_DIR; [ -n "$GPULOG" ] && kill $GPULOG 2>/dev/null || true' EXIT
echo "host $(hostname) job ${SLURM_JOB_ID:-} py $PY extra ${EXTRA:-} local $LOCAL_DIR"
python -u scripts/benchmark/mabe22/$PY.py ${EXTRA:-}
