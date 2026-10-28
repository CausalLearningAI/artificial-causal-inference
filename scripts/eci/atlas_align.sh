#!/bin/bash
# Behaviour alignment of the atlas models v0 / v1 / v2 (scripts/eci/atlas_align.py), CPU only.
#   sbatch scripts/eci/atlas_align.sh            (SERIES=mice,ants_v3 for a subset)
#   ECI v2.0 as 'v2': ATLAS_ALIGN_V2_ROOT=results/vision/eci_v2_0B ATLAS_ALIGN_OUT=results/vision/eci_atlas_v20/align \
#     sbatch --export=ALL --output=results/vision/eci_atlas_v20/logs/%x_%j.out scripts/eci/atlas_align.sh
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
