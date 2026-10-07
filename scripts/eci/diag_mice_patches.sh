#!/bin/bash
#
# Patch-token contact diagnostic, mice v1 (scripts/eci/diag_mice_patches.py). STEP = a | b_extract | b_probe | b
# (b = b_extract then b_probe in one job, sharing the local staging dir).
#   a          Diagnostic A, 448 px store tokens, pool-grouped 5-fold CV            -> $OUT/a/
#   b          ~20k-frame subset re-encoded at 448 and 896, same probes            -> $OUT/b/
# GPU jobs on the 'gpu' partition, excluding the 11 GB 2080 Ti node gpu150 and gpu242 (GPU 0 failed 2026-10-06).
# Inputs are read once into RAM (A) or staged to /localhome/$USER/$SLURM_JOB_ID (B); the staging dir is removed.
#
# Usage:
#   OUT=results/vision/eci_repr_diag/mice STEP=a sbatch --export=ALL scripts/eci/diag_mice_patches.sh
#   OUT=results/vision/eci_repr_diag/mice STEP=b sbatch --export=ALL scripts/eci/diag_mice_patches.sh
#   EXTRA="--max-videos 1 --steps 300 --seeds 0" for a smoke test
#SBATCH --job-name=diag_mice_patches
#SBATCH --output=logs/diag_mice_patches_%j.out
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --exclude=gpu150,gpu242
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G

module load conda
conda activate crl
export PYTHONUNBUFFERED=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
mkdir -p logs
OUT=${OUT:-results/vision/eci_repr_diag/mice}
EXTRA=${EXTRA:-}        # extra args for a / b_probe
EXTRA_X=${EXTRA_X:-}    # extra args for b_extract (e.g. --n-subset 600)
STAGE=/localhome/$USER/${SLURM_JOB_ID:-manual}
mkdir -p $STAGE
trap 'rm -rf $STAGE' EXIT
nvidia-smi -L
df -h /localhome | tail -1
P=scripts/eci/diag_mice_patches.py
case ${STEP} in
  a)         python -u $P a --out $OUT $EXTRA ;;
  b_extract) python -u $P b_extract --out $OUT --stage $STAGE $EXTRA ;;
  b_probe)   python -u $P b_probe --out $OUT --stage $STAGE $EXTRA ;;
  b)         python -u $P b_extract --out $OUT --stage $STAGE $EXTRA_X
             python -u $P b_probe --out $OUT --stage $STAGE $EXTRA ;;
esac
