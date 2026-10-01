#!/bin/bash
#
# Raw-token encoder check (scripts/eci/encoder_probe.py): DINOv2 (448) vs DINOv3 (512) linear-probe AUC on
# pooled tokens of labelled frames (mice contact, ants grooming) and the patch-position share of the token
# variance. Same frames, same foreground patches. Output: results/vision/eci_encoders/<domain>/probe.json
#
# Usage: DOMAIN=mice sbatch --export=ALL scripts/eci/encoder_probe.sh
#        DOMAIN=ants EXTRA_ARGS="--n-pos 15 --n-neg 15" sbatch --export=ALL scripts/eci/encoder_probe.sh
#
#SBATCH --job-name=eci_enc_probe
#SBATCH --output=logs/eci_enc_probe_%j.out
#SBATCH --error=logs/eci_enc_probe_%j.err
#SBATCH --time=03:00:00
#SBATCH --partition=gpu100
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
