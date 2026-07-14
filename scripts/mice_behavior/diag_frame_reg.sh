#!/bin/bash
#SBATCH --job-name=mice_diag_frame_reg
#SBATCH --output=logs/mice_diag_frame_reg_%j.out
#SBATCH --error=logs/mice_diag_frame_reg_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=visualize
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:2080ti:1

module load conda
conda activate crl
export PYTHONUNBUFFERED=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
mkdir -p logs

python -u scripts/mice_behavior/diag_frame_variants.py --mode reg
