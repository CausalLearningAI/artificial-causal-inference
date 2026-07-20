#!/bin/bash
#
# Per-frame classifier, patch-grid-only hyperparameter search — DINOv2+DINOv3
# concatenated (each L2-normalized first; the 20x raw-norm mismatch between them
# collapsed an earlier unnormalized comparison to exact-chance predictions).
# Own from-scratch search, same rationale as the DINOv3-only variant.
#
# Higher --mem than the single-encoder searches: the concat loader briefly holds
# the full per-observation span (not just the max_train_frames-bounded sample
# count — that bound only trims which samples get used afterward, not what the
# loader itself materializes) at double the single-encoder width.
#
# Usage:
#   sbatch scripts/mice_behavior/grid_search_frame_patchgrid_concat.sh
#
#SBATCH --job-name=mice_grid_search_frame_patchgrid_concat
#SBATCH --output=logs/mice_grid_search_frame_patchgrid_concat_%j.out
#SBATCH --error=logs/mice_grid_search_frame_patchgrid_concat_%j.err
#SBATCH --time=05:00:00
#SBATCH --partition=visualize
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:2080ti:1

module load conda
conda activate crl

export PYTHONUNBUFFERED=1
set -euo pipefail

cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference

mkdir -p logs

python -u scripts/mice_behavior/grid_search_frame.py --variant patchgrid_concat
