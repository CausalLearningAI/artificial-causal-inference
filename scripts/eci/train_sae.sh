#!/bin/bash
#
# ECI step 2: Matryoshka BatchTopK SAE (1024 latents, prefixes 128/256/512/1024,
# k=16, 20 epochs) on final-layer DINOv2-base patch tokens of mice v1, split by pool.
# Array index = SAE seed (0 and 1, to assess stability).
#
# Usage:
#   sbatch scripts/eci/train_sae.sh
#   EXTRA_ARGS="--overwrite" sbatch scripts/eci/train_sae.sh
#   sbatch scripts/eci/sae_stability.sh            # after both seeds finish
#
#SBATCH --job-name=eci_sae
#SBATCH --output=logs/eci_sae_%A_%a.out
#SBATCH --error=logs/eci_sae_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=02:00:00
#SBATCH --partition=gpu100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:H100:1

module load conda
conda activate crl

export PYTHONUNBUFFERED=1
set -euo pipefail

cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
mkdir -p logs

python -u scripts/eci/train_sae.py \
    --seed ${SLURM_ARRAY_TASK_ID} \
    --n-latents 1024 \
    --prefixes 128,256,512,1024 \
    --k 16 \
    --epochs 20 \
    --batch-size 4096 \
    --lr 5e-4 \
    --grad-clip 1.0 \
    ${EXTRA_ARGS:-}
