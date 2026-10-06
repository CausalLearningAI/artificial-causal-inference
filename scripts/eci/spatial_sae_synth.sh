#!/bin/bash
#
# Synthetic sanity test of the Spatial-SAE (scripts/eci/spatial_sae_synth.py): 3 seeds x {B, SS, SS-mu0, SS-stat},
# full 16 x 16 grids and (MASK_FRAC=0.5) half-grid coherent masks like the foreground-only ECI tokens.
# Usage: OUT=<dir> sbatch --export=ALL scripts/eci/spatial_sae_synth.sh
#        OUT=<dir> MASK_FRAC=0.5 sbatch --export=ALL scripts/eci/spatial_sae_synth.sh
# Measured 2026-10-06 (results/vision/eci_spatial_sae/synth_mask{0,0.5}.json, mean +- sd over 3 seeds), full grids:
#   B MCC 0.571+-0.003 F1 0.083 R2 0.513 | SS 0.748+-0.012 F1 0.215 R2 0.311 | SS-mu0 0.741 F1 0.203 R2 0.333 |
#   SS-stat 0.662 F1 0.129 R2 0.470.  Half-grid masks: B 0.567 / SS 0.745 / SS-mu0 0.739 / SS-stat 0.618 (MCC).
#
#SBATCH --job-name=eci_ssae_synth
#SBATCH --output=logs/eci_ssae_synth_%j.out
#SBATCH --time=03:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

module load conda
conda activate crl
export PYTHONUNBUFFERED=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
mkdir -p logs
MF=${MASK_FRAC:-0}
nvidia-smi -L
python -u scripts/eci/spatial_sae_synth.py --seeds 0 1 2 --mask-frac $MF --out "${OUT}/synth_mask${MF}.json"
