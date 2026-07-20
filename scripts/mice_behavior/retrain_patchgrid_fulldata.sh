#!/bin/bash
#
# Retrain the confirmed-best DINOv2 patch-grid config with max_train_frames raised
# to 1M (effectively unbounded). See retrain_patchgrid_fulldata.py docstring.
#
# Usage:
#   sbatch scripts/mice_behavior/retrain_patchgrid_fulldata.sh
#
#SBATCH --job-name=mice_patchgrid_fulldata
#SBATCH --output=logs/mice_patchgrid_fulldata_%j.out
#SBATCH --error=logs/mice_patchgrid_fulldata_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:H100:1

module load conda
conda activate crl

export PYTHONUNBUFFERED=1
set -euo pipefail

cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference

mkdir -p logs

python -u scripts/mice_behavior/retrain_patchgrid_fulldata.py
