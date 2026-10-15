#!/bin/bash
#
# CalMS21 ECI ROLE SWAP job wrapper (rules: eci_swap.sh). PY = a script of scripts/benchmark/calms21/ (no .py), EXTRA =
# its arguments. Sets CALMS_OUT = results/vision/benchmark_calms21/swap and CALMS_TOK = the pairpush store (BeeGFS)
# unless given. Staging in /localhome/$USER/<job>_swap (removed on exit). GPU jobs log nvidia-smi utilisation / memory
# every 60 s to swap/logs/gpu_<job name>_<job id>.csv. GPU: -p gpu (never gpu100), CPU: -p defaultp.
#SBATCH --output=results/vision/benchmark_calms21/swap/logs/%x_%j.out
module load conda
conda activate crl
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1 HF_HUB_OFFLINE=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
export CALMS_OUT=${CALMS_OUT:-results/vision/benchmark_calms21/swap}
export CALMS_TOK=${CALMS_TOK:-/mnt/beegfs/locatgrp/shared/rcadei/eci_tok/pairpush/calms21}
export LOCAL_DIR=/localhome/$USER/${SLURM_JOB_ID:-local}_swap
mkdir -p $LOCAL_DIR $CALMS_OUT/logs
SMI=""
if nvidia-smi -L >/dev/null 2>&1; then
  nvidia-smi -L
  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 60 \
    > $CALMS_OUT/logs/gpu_${SLURM_JOB_NAME:-x}_${SLURM_JOB_ID:-local}.csv 2>/dev/null &
  SMI=$!
fi
trap 'rm -rf $LOCAL_DIR; [ -n "$SMI" ] && kill $SMI 2>/dev/null || true' EXIT
echo "host $(hostname) job ${SLURM_JOB_ID:-} py $PY extra ${EXTRA:-} out $CALMS_OUT tok $CALMS_TOK local $LOCAL_DIR"
df -h $LOCAL_DIR | tail -1
python -u scripts/benchmark/calms21/$PY.py ${EXTRA:-}
