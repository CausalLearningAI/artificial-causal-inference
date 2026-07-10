#!/bin/bash
#
# Autonomous grid search + final plots for the mouse behavior classifier.
#
# Usage:
#   sbatch scripts/mice_behavior/grid_search.sh
#
#SBATCH --job-name=mice_grid_search
#SBATCH --output=logs/mice_grid_search_%j.out
#SBATCH --error=logs/mice_grid_search_%j.err
#SBATCH --time=10:00:00
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

python -u scripts/mice_behavior/grid_search.py
