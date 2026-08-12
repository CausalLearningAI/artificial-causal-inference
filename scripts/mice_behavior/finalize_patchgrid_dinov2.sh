#!/bin/bash
#SBATCH --job-name=finalize_patchgrid256_dinov2
#SBATCH --output=logs/finalize_patchgrid256_dinov2_%j.out
#SBATCH --error=logs/finalize_patchgrid256_dinov2_%j.err
#SBATCH --time=02:00:00
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
python -u scripts/mice_behavior/finalize_patchgrid_dinov2.py
