#!/bin/bash
#
# Foreground pipeline step 3: Matryoshka BatchTopK SAE (1024 latents, prefixes 128/256/512/1024,
# k=16, AuxK, lr 5e-4 warmup+cosine, batch 4096, 5 epochs = ~82.5k steps) on the foreground tokens of
# dataset/mice/v1/eci/train_tokens/dinov2_base_l-1_fg448_fps1 (held in CPU RAM).
# Same 8 held-out pools as the ep20 SAE. Array index = seed.
# Output: dataset/mice/v1/eci/sae/matryoshka_btk_1024_k16_fg448_s{seed}/
#
# Usage: sbatch scripts/eci/train_sae_fg.sh
#
#SBATCH --job-name=eci_sae_fg
#SBATCH --output=logs/eci_sae_fg_%A_%a.out
#SBATCH --error=logs/eci_sae_fg_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=06:00:00
#SBATCH --partition=gpu100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --gres=gpu:H100:1

module load conda
conda activate crl

export PYTHONUNBUFFERED=1
set -euo pipefail

cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
mkdir -p logs

python -u scripts/eci/train_sae_fg.py --seed ${SLURM_ARRAY_TASK_ID} ${EXTRA_ARGS:-}
