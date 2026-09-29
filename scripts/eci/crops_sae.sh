#!/bin/bash
#
# Crop prototype steps 2-3: single and pair/contact crop SAEs (2 seeds each), then the gates vs fg448.
# Output: dataset/mice/v1/eci/crops/sae/*, dataset/mice/v1/eci/crops/gates.json, top16_*.jpg
# Usage: sbatch scripts/eci/crops_sae.sh   (after crops_encode.sh)
#
#SBATCH --job-name=eci_crops_sae
#SBATCH --output=logs/eci_crops_sae_%j.out
#SBATCH --error=logs/eci_crops_sae_%j.err
#SBATCH --time=03:00:00
#SBATCH --partition=gpu100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1

module load conda
conda activate crl
export PYTHONUNBUFFERED=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
for kind in single pair; do for s in 0 1; do
    python -u scripts/eci/crops_train_sae.py --kind $kind --seed $s
done; done
python -u scripts/eci/crops_gates.py
