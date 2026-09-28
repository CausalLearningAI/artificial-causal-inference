#!/bin/bash
#
# ECI step 2 (v2): Matryoshka BatchTopK SAE (1024 latents, prefixes 128/256/512/1024,
# k=16, AuxK, lr 5e-4 warmup+cosine, batch 4096) - same hparams as v1 - trained on the
# 1 fps token store (518,400 frames, ~118M train tokens), streamed from disk in random
# chunks of 16,384 frames. 2 epochs (~57.6k optimizer steps; v1 had 30,720).
# Same 8 held-out pools as v1. Array index = SAE seed (0 and 1).
# Output: dataset/mice/v1/eci/sae/matryoshka_btk_1024_k16_fps1_s{seed}/
#
# Usage:
#   sbatch scripts/eci/extract_train_tokens_fps1.sh   # first (plus --finalize, see there)
#   sbatch scripts/eci/train_sae_fps1.sh
#   python scripts/eci/sae_stability.py --a dataset/mice/v1/eci/sae/matryoshka_btk_1024_k16_fps1_s0 \
#       --b dataset/mice/v1/eci/sae/matryoshka_btk_1024_k16_fps1_s1
#   sbatch scripts/eci/compare_sae.sh                 # v1 vs v2 on the same held-out tokens
#
#SBATCH --job-name=eci_sae_fps1
#SBATCH --output=logs/eci_sae_fps1_%A_%a.out
#SBATCH --error=logs/eci_sae_fps1_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=06:00:00
#SBATCH --partition=gpu100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --gres=gpu:H100:1

module load conda
conda activate crl

export PYTHONUNBUFFERED=1
set -euo pipefail

cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
mkdir -p logs

python -u scripts/eci/train_sae.py \
    --tokens-dir dataset/mice/v1/eci/train_tokens/dinov2_base_l-1_fps1 \
    --tag fps1 \
    --seed ${SLURM_ARRAY_TASK_ID} \
    --n-latents 1024 \
    --prefixes 128,256,512,1024 \
    --k 16 \
    --epochs 2 \
    --chunk-frames 16384 \
    --batch-size 4096 \
    --lr 5e-4 \
    --grad-clip 1.0 \
    ${EXTRA_ARGS:-}
