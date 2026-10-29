#!/bin/bash
#
# Job wrapper for scripts/eci/v2/v20_bench_store.py. EXTRA = its arguments (":" for "," because sbatch --export splits
# on commas). Staging in /localhome/$USER/<job>_<task>_tmp (removed on exit); GPU utilisation logged every 120 s.
#   CPU: sbatch -J vbs_size -p defaultp --export=ALL,EXTRA="size" scripts/eci/v2/v20_bench_store.sh
#SBATCH --job-name=vbs
#SBATCH --output=/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference/results/vision/eci_v2_0B/bench_store/logs/%x_%j_%a.out
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
module load conda
conda activate crl
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
if [ -d /localhome ] && mkdir -p /localhome/$USER 2>/dev/null; then BASE=/localhome/$USER; else BASE=${TMPDIR:-/tmp}; fi
export LOCAL_DIR=$BASE/${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}_tmp
mkdir -p $LOCAL_DIR
SMI=""
if nvidia-smi -L >/dev/null 2>&1; then
  nvidia-smi -L
  nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used --format=csv,noheader -l 120 | sed -u 's/^/[gpu] /' &
  SMI=$!
fi
trap 'rm -rf $LOCAL_DIR; [ -n "$SMI" ] && kill $SMI 2>/dev/null || true' EXIT
df -h $BASE | tail -1
echo "host $(hostname) job ${SLURM_JOB_ID:-} task ${SLURM_ARRAY_TASK_ID:-} extra ${EXTRA:-} git $(git rev-parse --short HEAD 2>/dev/null || echo na)"
EXTRA=${EXTRA:-}; EXTRA=${EXTRA//:/,}
python -u scripts/eci/v2/v20_bench_store.py ${EXTRA:-}
echo "done"
