#!/bin/bash
#
# Video-level Neural Effect Search analyses (scripts/eci/run_nes.py) on the mice v1 SAE codes.
# Needs dataset/mice/v1/eci/codes/<sae>/DONE. CPU only; the first run streams both code
# memmaps once (~10 GB) and caches per-video summaries under results/.../nes/<sae>/_cache/.
#
# Usage:
#   sbatch scripts/eci/run_nes.sh
#   SAE=matryoshka_btk_1024_k16_ep20_s0 sbatch scripts/eci/run_nes.sh
#
#SBATCH --job-name=eci_nes
#SBATCH --output=logs/eci_nes_%j.out
#SBATCH --error=logs/eci_nes_%j.err
#SBATCH --time=06:00:00
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

python -u scripts/eci/run_nes.py --sae "${SAE:-matryoshka_btk_1024_k16_ep20_s0}" ${EXTRA_ARGS:-}
