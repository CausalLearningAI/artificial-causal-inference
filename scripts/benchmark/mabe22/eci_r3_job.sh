#!/bin/bash
#
# MABe22 ECI round 3 job wrapper (from 2026-10-15 ~11:40): eci.sh's environment and staging, plus GPU utilisation /
# memory logged every 60 s to the job log (nvidia-smi in the background). PY / EXTRA as in eci.sh.
#SBATCH --output=results/vision/benchmark_mabe22/eci/r3/logs/%x_%j.out
module load conda
conda activate crl
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1 HF_HUB_OFFLINE=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
export LOCAL_DIR=/localhome/$USER/${SLURM_JOB_ID:-local}_mabe
mkdir -p $LOCAL_DIR
SMI=""
if nvidia-smi -L >/dev/null 2>&1; then
    nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 60 | sed -u 's/^/[gpu] /' &
    SMI=$!
fi
trap '[ -n "$SMI" ] && kill $SMI 2>/dev/null; rm -rf $LOCAL_DIR' EXIT
echo "host $(hostname) job ${SLURM_JOB_ID:-} py $PY extra ${EXTRA:-} local $LOCAL_DIR"
nvidia-smi -L 2>/dev/null || true
python -u scripts/benchmark/mabe22/$PY.py ${EXTRA:-}
