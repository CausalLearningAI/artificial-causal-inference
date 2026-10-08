#!/bin/bash
#
# Explainer figures of the ECI frame representation (current 512/448 vs ideal native/896), ants + mice.
# CPU only (DINOv2-base on ~100 frames per domain + ffmpeg decode of 7 native frames per domain).
# Usage: sbatch scripts/eci/explainer_figs.sh      Output: results/vision/eci_explainer/
#
#SBATCH --job-name=eci_explainer
#SBATCH --output=logs/eci_explainer_%j.out
#SBATCH --error=logs/eci_explainer_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=defaultp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

module load conda
conda activate crl
export PYTHONUNBUFFERED=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
mkdir -p logs
export STAGE=/localhome/$USER/$SLURM_JOB_ID
mkdir -p "$STAGE"
trap 'rm -rf "$STAGE"' EXIT
python -u scripts/eci/explainer_figs.py ${EXTRA_ARGS:-}
