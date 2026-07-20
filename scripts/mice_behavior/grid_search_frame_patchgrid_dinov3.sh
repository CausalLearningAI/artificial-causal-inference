#!/bin/bash
#
# Per-frame classifier, patch-grid-only hyperparameter search — DINOv3 instead of
# DINOv2. Own from-scratch search over the same hyperparameter space (not just
# reusing DINOv2's winning config) so the DINOv2-vs-DINOv3 comparison is fair.
#
# Usage:
#   sbatch scripts/mice_behavior/grid_search_frame_patchgrid_dinov3.sh
#
#SBATCH --job-name=mice_grid_search_frame_patchgrid_dinov3
#SBATCH --output=logs/mice_grid_search_frame_patchgrid_dinov3_%j.out
#SBATCH --error=logs/mice_grid_search_frame_patchgrid_dinov3_%j.err
#SBATCH --time=05:00:00
#SBATCH --partition=visualize
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:2080ti:1

module load conda
conda activate crl

export PYTHONUNBUFFERED=1
set -euo pipefail

cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference

mkdir -p logs

python -u scripts/mice_behavior/grid_search_frame.py --variant patchgrid_dinov3
