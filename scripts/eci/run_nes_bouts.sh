#!/bin/bash
#
# NES with bout outcomes from max-pooled SAE codes (scripts/eci/run_nes_bouts.py), mice v1.
# Needs dataset/mice/v1/eci/codes/<sae>/DONE. CPU only. First run computes pooled thresholds and
# streams codes_max once; caches under results/vision/mice/eci/nes/<sae>/_cache/.
# Output: results/vision/mice/eci/nes/<sae>/maxpool_bouts/
#
# Usage:
#   sbatch scripts/eci/run_nes_bouts.sh
#   SAE=matryoshka_btk_1024_k16_fps1_s0 sbatch scripts/eci/run_nes_bouts.sh
#   DOMAIN=ants SAE=<ants sae> EXTRA_ARGS="--compare-pooling max" sbatch --export=ALL scripts/eci/run_nes_bouts.sh
#
#SBATCH --job-name=eci_nes_bouts
#SBATCH --output=logs/eci_nes_bouts_%j.out
#SBATCH --error=logs/eci_nes_bouts_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=defaultp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

module load conda
conda activate crl

export PYTHONUNBUFFERED=1
set -euo pipefail

cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
mkdir -p logs

python -u scripts/eci/run_nes_bouts.py --domain "${DOMAIN:-mice}" --sae "${SAE:-matryoshka_btk_1024_k16_ep20_s0}" ${EXTRA_ARGS:-}
