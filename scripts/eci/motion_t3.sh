#!/bin/bash
#
# T3 motion channel (scripts/eci/motion_t3.py). STEP = encode | train | evaluate | video | summary.
# GPU jobs on the 'gpu' partition (never gpu100), excluding gpu150 (11 GB) and gpu242 (GPU 0 failed 2026-10-06).
# Everything heavy is staged to /localhome/$USER/$SLURM_JOB_ID (removed on exit).
#
# Usage (OUT defaults to results/vision/eci_t3_motion/<domain>):
#   mkdir -p results/vision/eci_t3_motion/logs
#   neighbour tokens t-1 (array over frame blocks):
#     DOMAIN=mice STEP=encode EXTRA="--offset -1 --n-tasks 4" sbatch --array=0-3 --export=ALL scripts/eci/motion_t3.sh
#   train / evaluate:
#     DOMAIN=mice STEP=train EXTRA="--arms M1 M5" sbatch --export=ALL scripts/eci/motion_t3.sh
#     DOMAIN=mice STEP=evaluate EXTRA="--arms BASE M1 M5" sbatch --export=ALL scripts/eci/motion_t3.sh
#   mice video level (CPU): DOMAIN=mice STEP=video sbatch --export=ALL -p defaultp --gres=none --mem=32G scripts/eci/motion_t3.sh
#   summary (login safe): python scripts/eci/motion_t3.py summary
#SBATCH --job-name=eci_t3mot
#SBATCH --output=results/vision/eci_t3_motion/logs/%x_%A_%a.out
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu150,gpu242
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G

module load conda
conda activate crl
export PYTHONUNBUFFERED=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
export LOCAL_DIR=/localhome/$USER/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}
mkdir -p $LOCAL_DIR
trap 'rm -rf $LOCAL_DIR' EXIT
df -h /localhome | tail -1
nvidia-smi -L 2>/dev/null || true
DOMAIN=${DOMAIN:-mice}
EXTRA=${EXTRA:-}
TASK=""
if [ "${STEP}" = "encode" ]; then TASK="--task ${SLURM_ARRAY_TASK_ID:-0}"; fi
python -u scripts/eci/motion_t3.py ${STEP} --domain $DOMAIN $TASK $EXTRA
