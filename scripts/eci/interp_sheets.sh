#!/bin/bash
#
# Contact sheets of the explorer page's neurons for interpretation (scripts/eci/interp_sheets.py): one PNG per model x
# selectable neuron in <res>/interp/sheets/ + index.json. GPU (per-patch heat of the top frames).
# Usage: sbatch scripts/eci/interp_sheets.sh      EXTRA_ARGS="--res <res> --overwrite" sbatch --export=ALL ...
#
#SBATCH --job-name=interp_sheets
#SBATCH --output=logs/interp_sheets_%j.out
#SBATCH --error=logs/interp_sheets_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

module load conda
conda activate crl
export PYTHONUNBUFFERED=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
mkdir -p logs
python -u scripts/eci/interp_sheets.py ${EXTRA_ARGS:-}
