#!/bin/bash
#
# One-off: finish promoting patchgrid_concat after the original search's final
# retrain crashed with CUDA OOM. See finish_concat_promotion.py docstring.
#
# Usage:
#   sbatch scripts/mice_behavior/finish_concat_promotion.sh
#
#SBATCH --job-name=mice_finish_concat_promotion
#SBATCH --output=logs/mice_finish_concat_promotion_%j.out
#SBATCH --error=logs/mice_finish_concat_promotion_%j.err
#SBATCH --time=00:30:00
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

python -u scripts/mice_behavior/finish_concat_promotion.py
