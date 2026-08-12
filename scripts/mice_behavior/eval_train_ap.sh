#!/bin/bash
#SBATCH --job-name=eval_train_ap
#SBATCH --output=logs/eval_train_ap_%j.out
#SBATCH --error=logs/eval_train_ap_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --gres=gpu:H100:1

module load conda
conda activate crl
export PYTHONUNBUFFERED=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
mkdir -p logs

ENCODER=${ENCODER:-dinov2}
python -u scripts/mice_behavior/eval_train_ap.py --encoder "${ENCODER}"
