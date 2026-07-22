#!/bin/bash
#
# Pool-level k-fold CV for patchgrid256, reusing the already-found best hyperparameters
# (no re-search per fold). See kfold_patchgrid_online.py docstring.
#
# Usage:
#   ENCODER=dinov2 sbatch scripts/mice_behavior/kfold_patchgrid_online.sh
#   ENCODER=dinov3 sbatch scripts/mice_behavior/kfold_patchgrid_online.sh
#
#SBATCH --job-name=kfold_patchgrid256
#SBATCH --output=logs/kfold_patchgrid256_%j.out
#SBATCH --error=logs/kfold_patchgrid256_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=gpu100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=220G
#SBATCH --gres=gpu:H100:1

module load conda
conda activate crl
export PYTHONUNBUFFERED=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
mkdir -p logs

ENCODER=${ENCODER:-dinov2}

python -u scripts/mice_behavior/kfold_patchgrid_online.py \
    --encoder "${ENCODER}" \
    --k 5 \
    --max-train-frames 200000 \
    --n-epochs 20 \
    --patience 7 \
    --batch-size 256 \
    --encode-batch-size 256 \
    --num-workers 16
