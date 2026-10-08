#!/bin/bash
#
# Per-frame instance-split GATE test (scripts/eci/split_test_*.py). One script, STEP = select | slots | sam | score,
# DOMAIN = mice | ants. Outputs in results/vision/eci_split_test/<domain>/ (gitignored).
#   select  CPU   eval frames, calibration, B2 start frames, frames.tar          (split_test_select.py)
#   slots   GPU   slot-attention models (A) trained on train-video tokens, masks on the eval frames (split_test_slots.py)
#   sam     GPU   SAM 2.1 per-frame point prompts (B1) and short-range propagation (B2) (split_test_sam.py)
#   native  CPU   mice only: eval frames + B2 windows decoded from the 2064 px source videos at 1024 px (split_test_native.py)
#   score   CPU   automatic rule, pair check, contact sheets, table               (split_test_score.py)
# SAM 2 is NOT installed in the project env: the official repo is cloned at /nfs/scistore19/locatgrp/rcadei/tools/sam2
# (commit 2b90b9f) and put on PYTHONPATH together with tools/sam2_deps (iopath 0.1.10 + portalocker, pip --target);
# checkpoints in tools/sam2_ckpt (SAM 2.1 hiera large / base-plus from dl.fbaipublicfiles.com).
# Inputs are staged to /localhome/$USER/$SLURM_JOB_ID and removed at the end.
#
# Usage:
#   DOMAIN=mice STEP=select sbatch --export=ALL --partition=defaultp --gres=none --time=02:00:00 scripts/eci/split_test.sh
#   DOMAIN=mice STEP=slots  sbatch --export=ALL scripts/eci/split_test.sh
#SBATCH --job-name=split_test
#SBATCH --output=logs/split_test_%j.out
#SBATCH --time=01:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu150,gpu242
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

module load conda
conda activate crl
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
mkdir -p logs
export STAGE=/localhome/$USER/${SLURM_JOB_ID:-local}
mkdir -p $STAGE
trap 'rm -rf $STAGE' EXIT
T=/nfs/scistore19/locatgrp/rcadei/tools
export PYTHONPATH=$T/sam2:$T/sam2_deps:${PYTHONPATH:-}
D=results/vision/eci_split_test/$DOMAIN
echo "host $(hostname) job ${SLURM_JOB_ID:-} step $STEP domain $DOMAIN"
case $STEP in
  select)
    python -u scripts/eci/split_test_select.py --domain $DOMAIN --workers ${SLURM_CPUS_PER_TASK:-8} ;;
  slots)
    nvidia-smi -L
    python -u scripts/eci/split_test_slots.py --domain $DOMAIN ${EXTRA:-} ;;
  sam)
    nvidia-smi -L
    cp $D/frames.tar $STAGE/
    case "${EXTRA:-}" in *--native*) cp $D/native.tar $STAGE/ ;; esac
    python -u scripts/eci/split_test_sam.py --domain $DOMAIN ${EXTRA:-} ;;
  native)
    python -u scripts/eci/split_test_native.py --workers ${SLURM_CPUS_PER_TASK:-12} ;;
  score)
    cp $D/frames.tar $STAGE/
    python -u scripts/eci/split_test_score.py --domain $DOMAIN ${EXTRA:-} ;;
esac
