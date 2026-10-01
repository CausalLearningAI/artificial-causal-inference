#!/bin/bash
#
# DINOv2 vs DINOv3 foreground SAE comparison (scripts/eci/compare_encoders.py), CPU only.
# Output: results/vision/eci_encoders/<domain>/compare.json
#
# Usage:
#   DOMAIN=mice EXTRA_ARGS="--sae matryoshka_btk_1024_k16_fg448_s0 --sae matryoshka_btk_1024_k16_fg512v3_s0" \
#       sbatch --export=ALL scripts/eci/compare_encoders.sh
#   DOMAIN=ants EXTRA_ARGS="--sae matryoshka_btk_1024_k16_antsfg_s0 --sae matryoshka_btk_1024_k16_antsfgv3_s0 --nes-sub pairs" \
#       sbatch --export=ALL scripts/eci/compare_encoders.sh
#
#SBATCH --job-name=eci_enc_cmp
#SBATCH --output=logs/eci_enc_cmp_%j.out
#SBATCH --error=logs/eci_enc_cmp_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=defaultp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G

module load conda
conda activate crl

export PYTHONUNBUFFERED=1
set -euo pipefail

cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
mkdir -p logs

python -u scripts/eci/compare_encoders.py --domain "${DOMAIN:-mice}" ${EXTRA_ARGS:-}
