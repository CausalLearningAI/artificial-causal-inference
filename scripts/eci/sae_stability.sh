#!/bin/bash
#
# ECI step 2b: decoder-atom stability of the SAE across seeds 0 and 1 (CPU, seconds).
#
# Usage:
#   sbatch scripts/eci/sae_stability.sh
#
#SBATCH --job-name=eci_sae_stab
#SBATCH --output=logs/eci_sae_stab_%j.out
#SBATCH --error=logs/eci_sae_stab_%j.err
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

module load conda
conda activate crl

export PYTHONUNBUFFERED=1
set -euo pipefail

cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
python -u scripts/eci/sae_stability.py
