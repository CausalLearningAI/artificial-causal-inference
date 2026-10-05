#!/bin/bash
#
# Tadpole (frogs stage 44-48) body crops, one array task per valid row of data/frogs/v1_tadpole/experiment.csv
# (scripts/eci/tadpoles_crops.py: raw 60 fps video -> 5 fps 96 px body crops at 512 px + the tadpole-free background).
# Needs scripts/eci/frogs_prepare.py --stage 44-48 --step source and /usr/bin/python3 scripts/eci/frogs_sleap.py
# --stage 44-48. The array size must equal the number of valid rows (scripts/eci/frogs_chain.sh sets it).
#
# Usage: sbatch --array=0-172 scripts/eci/tadpoles_crops.sh
#        OBS=WT_100_1,FoxP1_169_3 sbatch --array=0-1 scripts/eci/tadpoles_crops.sh   (selected videos, testing)
#
#SBATCH --job-name=tadpole_crops
#SBATCH --output=logs/tadpole_crops_%A_%a.out
#SBATCH --error=logs/tadpole_crops_%A_%a.err
#SBATCH --time=03:00:00
#SBATCH --partition=defaultp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=12G

module load conda
conda activate crl

export PYTHONUNBUFFERED=1
set -euo pipefail

cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
mkdir -p logs

if [ -n "${OBS:-}" ]; then
    IFS=, read -ra ids <<< "$OBS"
    python -u scripts/eci/tadpoles_crops.py --obs "${ids[$SLURM_ARRAY_TASK_ID]}" ${EXTRA_ARGS:-}
else
    python -u scripts/eci/tadpoles_crops.py --task ${SLURM_ARRAY_TASK_ID} ${EXTRA_ARGS:-}
fi
