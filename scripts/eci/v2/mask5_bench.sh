#!/bin/bash
# Job wrapper for scripts/eci/v2/mask5_bench.py: CHAIN = "step args;step args" (":" for ",").
#   sbatch -J m5b_cal -p gpu --gres=gpu:1 --constraint="L40S|A100|A40" --exclude=gpu240,gpu241,gpu242,gpu278 \
#          --export=ALL,CHAIN="run --set cal;score --set cal;sheet --set cal" scripts/eci/v2/mask5_bench.sh
#SBATCH --output=/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference/results/vision/eci_v2_0B/mask_approval_v3/logs/%x_%j.out
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
module load conda
conda activate crl
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error PYTHONWARNINGS=ignore
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
if [ -d /localhome ] && mkdir -p /localhome/$USER 2>/dev/null; then BASE=/localhome/$USER; else BASE=${TMPDIR:-/tmp}; fi
export LOCAL_DIR=$BASE/${SLURM_JOB_ID:-local}
mkdir -p $LOCAL_DIR
SMI=""
if nvidia-smi -L >/dev/null 2>&1; then
  nvidia-smi -L
  nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used --format=csv,noheader -l 120 | sed -u 's/^/[gpu] /' &
  SMI=$!
fi
trap 'rm -rf $LOCAL_DIR; [ -n "$SMI" ] && kill $SMI 2>/dev/null || true' EXIT
echo "host $(hostname) job ${SLURM_JOB_ID:-} git $(git rev-parse --short HEAD 2>/dev/null)"
IFS=';' read -ra ST <<< "${CHAIN//:/,}"
for s in "${ST[@]}"; do echo "== $s"; python -u scripts/eci/v2/mask5_bench.py $s || { echo "FAILED: $s"; exit 1; }; done
echo done
