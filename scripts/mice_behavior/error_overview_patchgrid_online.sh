#!/bin/bash
#SBATCH --job-name=error_overview_patchgrid_online
#SBATCH --output=logs/error_overview_patchgrid_online_%j.out
#SBATCH --error=logs/error_overview_patchgrid_online_%j.err
#SBATCH --time=00:20:00
#SBATCH --partition=gpu100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gres=gpu:H100:1

module load conda
conda activate crl
export PYTHONUNBUFFERED=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
mkdir -p logs
python -u scripts/mice_behavior/error_overview_patchgrid_online.py
