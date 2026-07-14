#!/bin/bash
#SBATCH --job-name=mice_diag_frame_patchgrid
#SBATCH --output=logs/mice_diag_frame_patchgrid_%j.out
#SBATCH --error=logs/mice_diag_frame_patchgrid_%j.err
#SBATCH --time=01:00:00
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

python -u scripts/mice_behavior/diag_frame_variants.py --mode patchgrid
