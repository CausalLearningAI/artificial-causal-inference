#!/bin/bash
#
# Frogs v1 frames, one array task per video (scripts/eci/frogs_frames.py: standardize 5 fps / 512 px, mirrored when
# experiment.csv hflip = 1, then JPEG frames; 18,000 frames per 1 h video). Needs scripts/eci/frogs_prepare.py
# --step source. Then the tables (scripts/eci/frogs_chain.sh submits both with a dependency):
#   python src/data/get_metadata.py experiment=frogs/v1 && python scripts/eci/frogs_prepare.py --step tables
#
# Usage: sbatch scripts/eci/frogs_frames.sh
#
#SBATCH --job-name=frogs_frames
#SBATCH --output=logs/frogs_frames_%A_%a.out
#SBATCH --error=logs/frogs_frames_%A_%a.err
#SBATCH --array=0-34
#SBATCH --time=02:00:00
#SBATCH --partition=defaultp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

module load conda
conda activate crl

export PYTHONUNBUFFERED=1
set -euo pipefail

cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
mkdir -p logs

python -u scripts/eci/frogs_frames.py --task ${SLURM_ARRAY_TASK_ID}
