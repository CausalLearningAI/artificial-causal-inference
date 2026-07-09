#!/bin/bash
#SBATCH --job-name=mice_pg_final
#SBATCH --output=logs/mice_pg_final_%j.out
#SBATCH --error=logs/mice_pg_final_%j.err
#SBATCH --time=03:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:A100:1

module load conda
conda activate crl
export PYTHONUNBUFFERED=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
mkdir -p logs

python -u scripts/mice_behavior/train_patchgrid_final.py
