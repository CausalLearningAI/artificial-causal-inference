#!/bin/bash
#
# ECI: compare v1 (64 frames/obs) and v2 (1 fps) SAEs on the same held-out 1 fps tokens:
# FVE / L0 / dead per prefix, rare-moment FVE on annotated nose-nose / nose-body /
# nose-tail frames, best single-latent AUROC of max-pooled codes, decoder stability.
# Output: dataset/mice/v1/eci/diagnostics/sae_compare_v1_vs_fps1.json
#
# Usage:
#   sbatch scripts/eci/compare_sae.sh
#
#SBATCH --job-name=eci_sae_compare
#SBATCH --output=logs/eci_sae_compare_%j.out
#SBATCH --error=logs/eci_sae_compare_%j.err
#SBATCH --time=03:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --gres=gpu:1

module load conda
conda activate crl

export PYTHONUNBUFFERED=1
set -euo pipefail

cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
mkdir -p logs

python -u scripts/eci/compare_sae.py ${EXTRA_ARGS:-}
