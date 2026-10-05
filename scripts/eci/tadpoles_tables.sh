#!/bin/bash
#
# Tadpole (frogs stage 44-48) tables after the body crops (scripts/eci/frogs_chain.sh STAGE=44-48 step 2):
# frogs_prepare.py --stage 44-48 --step tables (annotations.csv / experiment.csv of the domain 'tadpoles'), then
# tadpoles_background.py (the crop-mask backgrounds). TADPOLE_TAG + SUBSET=--subset for a test on a few videos.
#
# Usage: sbatch scripts/eci/tadpoles_tables.sh
#
#SBATCH --job-name=tadpole_tables
#SBATCH --output=logs/tadpole_tables_%j.out
#SBATCH --error=logs/tadpole_tables_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=defaultp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

module load conda
conda activate crl

export PYTHONUNBUFFERED=1
set -euo pipefail

cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
mkdir -p logs

python -u scripts/eci/frogs_prepare.py --stage 44-48 --step tables ${SUBSET:-}
python -u scripts/eci/tadpoles_background.py
