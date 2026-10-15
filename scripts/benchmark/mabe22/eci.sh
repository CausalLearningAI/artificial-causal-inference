#!/bin/bash
#
# MABe22 mouse ECI launcher: PY = eci_prep | eci_encode | eci_train | eci_probe | eci_sheets (scripts/benchmark/mabe22/),
# EXTRA = its arguments. GPU jobs: -p gpu --gres=gpu:1 --exclude=gpu242,gpu150,gpu230,gpu241 (never gpu100); CPU jobs:
# -p defaultp --gres=none. Staging in /localhome/$USER/<job> ($TMPDIR when missing), removed on exit.
# Outputs: results/vision/benchmark_mabe22/eci/ (gitignored).
#   PY=eci_prep sbatch -p defaultp --cpus-per-task=16 --mem=96G --time=01:00:00 scripts/benchmark/mabe22/eci.sh
#   PY=eci_encode EXTRA="--set eval --shard 0 --nshard 8" sbatch -p gpu --gres=gpu:1 ... scripts/benchmark/mabe22/eci.sh
#SBATCH --job-name=mabe_eci
#SBATCH --output=results/vision/benchmark_mabe22/eci/logs/%x_%j.out
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

module load conda
conda activate crl
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1 HF_HUB_OFFLINE=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
if [ -d /localhome ] && [ -w /localhome/$USER -o -w /localhome ]; then
    export LOCAL_DIR=/localhome/$USER/${SLURM_JOB_ID:-local}_mabe
else
    export LOCAL_DIR=${TMPDIR:-/tmp}/${SLURM_JOB_ID:-local}_mabe
    export TMPDIR=${TMPDIR:-/tmp}
fi
mkdir -p $LOCAL_DIR
trap 'rm -rf $LOCAL_DIR' EXIT
echo "host $(hostname) job ${SLURM_JOB_ID:-} py $PY extra ${EXTRA:-} local $LOCAL_DIR"
df -h $LOCAL_DIR | tail -1
nvidia-smi -L 2>/dev/null || true
python -u scripts/benchmark/mabe22/$PY.py ${EXTRA:-}
