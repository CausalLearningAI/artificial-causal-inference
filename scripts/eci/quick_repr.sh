#!/bin/bash
#
# quick_repr launcher (scripts/eci/quick_repr.py). EXTRA = args after "run". GPU: 'gpu' partition, never gpu100,
# excluding gpu150 / gpu242 / gpu230 / gpu241. Staging in /localhome/$USER/<job> (removed on exit).
# Outputs: results/vision/eci_quick_repr/ (gitignored).
#
# Usage (from the repo root):
#   EXTRA="--arms c448 W01 W10" sbatch scripts/eci/quick_repr.sh                          # T-W (+ C1 retrained control)
#   EXTRA="--arms L8 L10 --probe-sets L8 L10" sbatch scripts/eci/quick_repr.sh            # T-L
#   EXTRA="--arms reg --probe-sets reg" sbatch scripts/eci/quick_repr.sh                  # T-R
#   QR_OUT=results/vision/eci_quick_repr/smoke EXTRA="--arms c448 W01 L8 --probe-sets L8 --smoke --seeds 0 1" \
#       sbatch --time=00:30:00 scripts/eci/quick_repr.sh
#SBATCH --job-name=qrepr
#SBATCH --output=results/vision/eci_quick_repr/logs/%x_%j.out
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu150,gpu242,gpu230,gpu241
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

module load conda
conda activate crl
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1 HF_HUB_OFFLINE=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
export LOCAL_DIR=/localhome/$USER/${SLURM_JOB_ID:-local}_qr
mkdir -p $LOCAL_DIR
trap 'rm -rf $LOCAL_DIR' EXIT
echo "host $(hostname) job ${SLURM_JOB_ID:-} extra ${EXTRA:-} out ${QR_OUT:-default}"
nvidia-smi -L 2>/dev/null || true
python -u scripts/eci/quick_repr.py run ${EXTRA:-}
