#!/bin/bash
#
# CLS-only hyperparameter search. Run separately from patch-grid's search
# (grid_search_patchgrid.sh) so patch-grid always starts with a clean GPU.
#
# Usage:
#   sbatch scripts/mice_behavior/grid_search_cls.sh
#
#SBATCH --job-name=mice_grid_search_cls
#SBATCH --output=logs/mice_grid_search_cls_%j.out
#SBATCH --error=logs/mice_grid_search_cls_%j.err
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

python -u scripts/mice_behavior/grid_search.py --variant cls
