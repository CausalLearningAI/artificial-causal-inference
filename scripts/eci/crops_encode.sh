#!/bin/bash
#
# Crop prototype step 1: blobs -> crops -> DINOv2 CLS / mean-patch per crop.
# Array 0-1: train split (20 videos, 1 fps) in 2 tasks; 2-7: eval split (held-out annotated, 5 fps) in 6 tasks.
# Output: dataset/mice/v1/eci/crops/{train,eval}/task_XX.npz
# Usage: python scripts/eci/crops_select.py ; sbatch scripts/eci/crops_encode.sh
#
#SBATCH --job-name=eci_crops_enc
#SBATCH --output=logs/eci_crops_enc_%A_%a.out
#SBATCH --error=logs/eci_crops_enc_%A_%a.err
#SBATCH --array=0-7
#SBATCH --time=04:00:00
#SBATCH --partition=gpu100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

module load conda
conda activate crl
export PYTHONUNBUFFERED=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
mkdir -p logs
i=${SLURM_ARRAY_TASK_ID}
if [ $i -lt 2 ]; then
    python -u scripts/eci/crops_encode.py --split train --task $i --n-tasks 2 --workers 22
else
    python -u scripts/eci/crops_encode.py --split eval --task $((i - 2)) --n-tasks 6 --workers 22
fi
