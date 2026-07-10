#!/bin/bash
#SBATCH --job-name=mice_sanity_overfit
#SBATCH --output=logs/mice_sanity_overfit_%j.out
#SBATCH --error=logs/mice_sanity_overfit_%j.err
#SBATCH --time=00:20:00
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

python -u scripts/mice_behavior/sanity_overfit.py
