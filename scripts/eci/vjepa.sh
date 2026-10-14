#!/bin/bash
#
# V-JEPA pilot launcher (scripts/eci/vjepa.py). STEP = plan | smoke | encode | probe | train | c1 | eval | sheets |
# summary; EXTRA = extra args. GPU: 'gpu' partition, L40S, never gpu100, excluding gpu150 / gpu242 / gpu230 / gpu241.
# Staging in /localhome/$USER/<job> (removed on exit). Outputs: results/vision/eci_vjepa/ (gitignored).
#
# Usage (from the repo root):
#   STEP=plan   sbatch -p defaultp --gres=none --cpus-per-task=4 --mem=32G --time=00:20:00 scripts/eci/vjepa.sh
#   STEP=smoke  sbatch --time=00:20:00 scripts/eci/vjepa.sh
#   STEP=encode EXTRA="--split eval"  sbatch --array=0-7 --time=01:30:00 scripts/eci/vjepa.sh
#   STEP=encode EXTRA="--split train" sbatch --array=0-3 --time=01:30:00 scripts/eci/vjepa.sh
#   STEP=probe  EXTRA="--seeds 0 1 2" sbatch --time=01:30:00 scripts/eci/vjepa.sh
#   STEP=train  EXTRA="--seeds 0"     sbatch --time=00:40:00 scripts/eci/vjepa.sh
#   STEP=c1     sbatch --time=00:40:00 scripts/eci/vjepa.sh
#   STEP=eval   EXTRA="--seeds 0 --c1" sbatch --time=00:40:00 scripts/eci/vjepa.sh
#   STEP=sheets sbatch --time=00:40:00 scripts/eci/vjepa.sh
#SBATCH --job-name=vjepa
#SBATCH --output=results/vision/eci_vjepa/logs/%x_%A_%a.out
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --exclude=gpu150,gpu242,gpu230,gpu241
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G

module load conda
conda activate crl
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1 HF_HUB_OFFLINE=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
export LOCAL_DIR=/localhome/$USER/${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}_vj
mkdir -p $LOCAL_DIR
trap 'rm -rf $LOCAL_DIR' EXIT
echo "host $(hostname) job ${SLURM_JOB_ID:-} task ${SLURM_ARRAY_TASK_ID:-} step $STEP extra ${EXTRA:-}"
nvidia-smi -L 2>/dev/null || true
python -u scripts/eci/vjepa.py $STEP ${EXTRA:-}
