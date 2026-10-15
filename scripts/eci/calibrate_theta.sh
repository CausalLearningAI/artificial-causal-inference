#!/bin/bash
#
# calibrate_theta launcher (scripts/eci/calibrate_theta.py). STEP = selftest | run (default run); EXTRA = more args.
# GPU: 'gpu' partition, never gpu100, excluding gpu150 / gpu242 / gpu230 / gpu241. Staging in
# /localhome/$USER/<job> ($TMPDIR when /localhome is missing), removed on exit.
# Outputs: results/vision/eci_calibrate_theta/ (gitignored).
#
# Usage (from the repo root):
#   STEP=selftest sbatch --time=00:15:00 scripts/eci/calibrate_theta.sh
#   sbatch scripts/eci/calibrate_theta.sh
#   python scripts/eci/calibrate_theta.py summary
#SBATCH --job-name=calth
#SBATCH --output=results/vision/eci_calibrate_theta/logs/%x_%j.out
#SBATCH --time=01:30:00
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
if [ -d /localhome ] && [ -w /localhome/$USER -o -w /localhome ]; then
    export LOCAL_DIR=/localhome/$USER/${SLURM_JOB_ID:-local}_ct
else
    export LOCAL_DIR=${TMPDIR:-/tmp}/${SLURM_JOB_ID:-local}_ct
    export TMPDIR=${TMPDIR:-/tmp}
fi
mkdir -p $LOCAL_DIR
trap 'rm -rf $LOCAL_DIR' EXIT
echo "host $(hostname) job ${SLURM_JOB_ID:-} step ${STEP:-run} extra ${EXTRA:-} local $LOCAL_DIR"
nvidia-smi -L 2>/dev/null || true
python -u scripts/eci/calibrate_theta.py ${STEP:-run} ${EXTRA:-}
