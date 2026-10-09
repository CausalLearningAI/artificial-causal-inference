#!/bin/bash
#
# C1 rollout (scripts/eci/rollout_c1.py). STEP = tokens | train | encode | merge | align; DOMAIN = mice | ants | frogs.
# GPU jobs on the 'gpu' partition (never gpu100), excluding gpu150 (11 GB), gpu242 (GPU 0 failed) and gpu230
# (/localhome nearly full). Inputs are staged to /localhome/$USER/$SLURM_JOB_ID_<task> (removed on exit).
#
#   mkdir -p results/vision/eci_rollout_c1/logs
#   frogs 1 fps motion store (35 videos):  DOMAIN=frogs STEP=tokens sbatch --array=0-34 --export=ALL scripts/eci/rollout_c1.sh
#   deployment SAE:                       DOMAIN=mice STEP=train sbatch --export=ALL scripts/eci/rollout_c1.sh
#   per-frame codes (24 shards):          DOMAIN=mice STEP=encode sbatch --array=0-23 --time=02:30:00 --export=ALL scripts/eci/rollout_c1.sh
#   merge + verify:                       DOMAIN=mice STEP=merge sbatch --time=02:00:00 --export=ALL scripts/eci/rollout_c1.sh
#   label alignment (mice, ants):         DOMAIN=mice STEP=align sbatch --export=ALL scripts/eci/rollout_c1.sh
#   cost estimate (login safe):           python scripts/eci/rollout_c1.py plan
#SBATCH --job-name=eci_c1
#SBATCH --output=results/vision/eci_rollout_c1/logs/%x_%A_%a.out
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu150,gpu242,gpu230
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G

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
if [ "${STEP}" = "tokens" ]; then TASK="--task ${SLURM_ARRAY_TASK_ID:-0}"; fi
if [ "${STEP}" = "encode" ]; then TASK="--shard ${SLURM_ARRAY_TASK_ID:-0}"; fi
python -u scripts/eci/rollout_c1.py ${STEP} --domain $DOMAIN $TASK $EXTRA
