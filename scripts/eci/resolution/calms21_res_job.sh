#!/bin/bash
#
# Job wrapper of the CalMS21 extra-grid resolution test (launcher + rules: calms21_res.sh). MODE = encode | train,
# EXTRA = arguments of scripts/eci/resolution/calms21_res.py <MODE>. Same as scripts/benchmark/calms21/eci.sh, with the
# logs under results/vision/eci_resolution/calms21/logs/: staging in /localhome/$USER/<job>_calms (removed on exit),
# nvidia-smi utilisation / memory every 60 s to logs/gpu_<name>_<job>.csv.
module load conda
conda activate crl
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1 HF_HUB_OFFLINE=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
LOGS=results/vision/eci_resolution/calms21/logs
export LOCAL_DIR=/localhome/$USER/${SLURM_JOB_ID:-local}_calms
mkdir -p $LOCAL_DIR
SMI=""
if nvidia-smi -L >/dev/null 2>&1; then
  nvidia-smi -L
  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 60 \
    > $LOGS/gpu_${SLURM_JOB_NAME:-x}_${SLURM_JOB_ID:-local}.csv 2>/dev/null &
  SMI=$!
fi
trap 'rm -rf $LOCAL_DIR; [ -n "$SMI" ] && kill $SMI 2>/dev/null || true' EXIT
echo "host $(hostname) job ${SLURM_JOB_ID:-} mode $MODE extra ${EXTRA:-} tok ${CALMS_TOK:-default} out ${CALMS_OUT:-default}"
df -h $LOCAL_DIR | tail -1
python -u scripts/eci/resolution/calms21_res.py $MODE ${EXTRA:-}
