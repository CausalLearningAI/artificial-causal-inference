#!/bin/bash
# Behaviour alignment of the atlas models v0 / v1 / v2 (scripts/eci/atlas_align.py), CPU only.
#   sbatch scripts/eci/atlas_align.sh            (SERIES=mice,ants_v3 for a subset)
#SBATCH --job-name=atlas_align
#SBATCH --output=results/vision/eci_atlas_v37/logs/%x_%j.out
#SBATCH --time=03:00:00
#SBATCH --partition=defaultp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
module load conda
conda activate crl
export PYTHONUNBUFFERED=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
echo "node $(hostname)"
python -u scripts/eci/atlas_align.py ${SERIES:+--series $SERIES}
