#!/bin/bash
#
# Raw-token encoder check (scripts/eci/encoder_probe.py): DINOv2 (448) vs DINOv3 (512) linear-probe AUC on
# pooled tokens of labelled frames (mice contact, ants grooming) and the patch-position share of the token
# variance. Same frames, same foreground patches. Output: results/vision/eci_encoders/<domain>/probe.json
#
# Usage: DOMAIN=mice sbatch --export=ALL scripts/eci/encoder_probe.sh
#        DOMAIN=ants EXTRA_ARGS="--n-pos 15 --n-neg 15" sbatch --export=ALL scripts/eci/encoder_probe.sh
#   DINOv3 ViT-S/16 (384-dim) instead of ViT-B/16, same frames and patches:
#        DOMAIN=mice EXTRA_ARGS="--encoder2 dinov3_small --out-dir results/vision/eci_encoders/dinov3_small" sbatch ...
#   Measured (pooled foreground tokens, C=0.1): mice contact 0.782 (DINOv2 0.813, ViT-B 0.795), ants grooming 0.912
#   (DINOv2 0.937 in the same run, 0.931 earlier; ViT-B 0.927); patch-position share 34% / 27% (DINOv2 19% / 13%).
#
#SBATCH --job-name=eci_enc_probe
#SBATCH --output=logs/eci_enc_probe_%j.out
#SBATCH --error=logs/eci_enc_probe_%j.err
#SBATCH --time=03:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1

module load conda
conda activate crl

export PYTHONUNBUFFERED=1
set -euo pipefail

cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
mkdir -p logs

python -u scripts/eci/encoder_probe.py --domain "${DOMAIN:-mice}" --num-workers 14 ${EXTRA_ARGS:-}
