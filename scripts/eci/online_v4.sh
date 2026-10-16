#!/bin/bash
#
# Ants v4 with the online atlas ants model (scripts/eci/online_v4.py). STEP = verify (GPU) | layout | check (CPU).
# Logs: results/vision/eci_online_v4/logs/. GPU utilisation is logged every 60 s to the job log's .gpu sibling.
#   STEP=verify sbatch -p debug_gpu --gres=gpu:1 --exclude=gpu242,gpu150,gpu230,gpu241,gpu240,gpu100 --export=ALL scripts/eci/online_v4.sh
#   STEP=layout sbatch -p defaultp --export=ALL scripts/eci/online_v4.sh
#SBATCH --job-name=online_v4
#SBATCH --output=results/vision/eci_online_v4/logs/%x_%j.out
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
module load conda
conda activate crl
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
export LOCAL_DIR=/localhome/$USER/${SLURM_JOB_ID}
mkdir -p $LOCAL_DIR
GP=""
trap 'rm -rf $LOCAL_DIR; [ -n "$GP" ] && kill $GP 2>/dev/null || true' EXIT
echo "host $(hostname) job $SLURM_JOB_ID step $STEP head $(git rev-parse HEAD 2>/dev/null || echo ?)"
df -h /localhome | tail -1
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  nvidia-smi -L
  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 60 \
    > results/vision/eci_online_v4/logs/online_v4_${SLURM_JOB_ID}.gpu 2>&1 &
  GP=$!
fi
python -u scripts/eci/online_v4.py ${STEP:?}
