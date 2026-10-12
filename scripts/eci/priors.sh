#!/bin/bash
#
# PRIORS test (scripts/eci/priors.py): INV (D4 invariance) and PRED (1 s future-change prediction) loss terms on the
# C1 SAE, mice + ants. STEP = plan | encode | warmup | train | evaluate | sheets | clean (summary is login safe:
# python scripts/eci/priors.py summary). GPU: 'gpu' partition (never gpu100), excluding gpu242 / gpu150 / gpu230 /
# gpu241. Staging in /localhome/$USER/<job> (removed on exit). Outputs: results/vision/eci_priors/ (gitignored).
#
# Usage (from the repo root; mkdir -p results/vision/eci_priors/logs first):
#   STEP=plan sbatch -p defaultp --gres=none --cpus-per-task=4 --mem=64G --time=01:00:00 scripts/eci/priors.sh
#   STEP=encode EXTRA="--domain mice --workers 14" sbatch --time=01:00:00 scripts/eci/priors.sh
#   STEP=warmup EXTRA="--domain mice" sbatch --time=00:30:00 scripts/eci/priors.sh
#   STEP=train  EXTRA="--domain mice" sbatch --time=01:30:00 scripts/eci/priors.sh
#   STEP=evaluate EXTRA="--domain mice" sbatch --mem=200G --time=02:30:00 scripts/eci/priors.sh
#   STEP=sheets EXTRA="--browse-arm INV --workers 16" sbatch -p defaultp --gres=none --cpus-per-task=16 --mem=64G --time=01:00:00 scripts/eci/priors.sh
#SBATCH --job-name=eci_priors
#SBATCH --output=results/vision/eci_priors/logs/%x_%j.out
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --exclude=gpu242,gpu150,gpu230,gpu241
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G

module load conda
conda activate crl
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
export LOCAL_DIR=/localhome/$USER/${SLURM_JOB_ID:-local}_pr
mkdir -p $LOCAL_DIR
trap 'rm -rf $LOCAL_DIR' EXIT
echo "host $(hostname) job ${SLURM_JOB_ID:-} step $STEP extra ${EXTRA:-}"
df -h /localhome | tail -1
nvidia-smi -L 2>/dev/null || true
python -u scripts/eci/priors.py $STEP ${EXTRA:-}
