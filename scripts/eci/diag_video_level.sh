#!/bin/bash
#
# Video-level NES usefulness of mice contact neurons (scripts/eci/diag_video_level.py). CPU only.
# Stages annotations.csv and the deployed fg448al codes_mean / codes_max (~10 GB) to /localhome/$USER/$SLURM_JOB_ID,
# reads them there, writes results/vision/eci_repr_diag/mice/video_level/, removes the staging dir.
#
# Usage:
#   sbatch scripts/eci/diag_video_level.sh
#   srun --partition=defaultp --cpus-per-task=4 --mem=32G --time=01:00:00 bash scripts/eci/diag_video_level.sh
#SBATCH --job-name=diag_video_level
#SBATCH --output=logs/diag_video_level_%j.out
#SBATCH --time=01:00:00
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

module load conda
conda activate crl
export PYTHONUNBUFFERED=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
mkdir -p logs
STAGE=/localhome/$USER/${SLURM_JOB_ID:-manual}_video_level
mkdir -p $STAGE
trap 'rm -rf $STAGE' EXIT
df -h /localhome | tail -1
python -u scripts/eci/diag_video_level.py --stage $STAGE ${EXTRA:-}
python -u scripts/eci/diag_video_level.py --summary
