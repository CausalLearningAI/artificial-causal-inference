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
# Measured 2026-10-08 (results/vision/eci_split_test/<domain>/table.json, visual_check.json; % of held-out frames split
# correctly by the pre-registered automatic rule, contact | non-contact; visual = Claude's count on 50 contact sheets):
#   mice  blob 19.8 | 27.1 (visual 8%)   slots A_k5 47.2 | 42.2 (32%)   A_k6 14.1   A_k5seed 19.1
#         SAM B1npk 74.9 | 73.0 (74%)    B1npk native 1024 px 74.6 | 74.1    B2 67.8 | 50.1 (60%), but 95.1 | 94.2 on the
#         frames with a clean start frame within 10 s (coverage 71.3% | 53.2%; visual 30/31)   AMADEUS boxes (pilot
#         videos, long-term tracking) 78.4 | 88.7. Post-hoc hybrid B2-else-B1npk 85.8 | 77.2 (visual 46/50).
#   ants  blob 5.6 | 57.3 (4%)   slots A_k4 24.7 | 39.2 (26%)   SAM B1n 34.0 | 52.2 (18%)   per-ant zoom B1nz 30.7 | 75.9
#         B2 7.8 | 35.8 (covered contact frames 13.7, coverage 51.8%)   AMADEUS (pilot = train videos) 80.3 | 95.5.
#   Gate (>= 85% contact, automatic and visual): no method passes either domain.
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
