#!/bin/bash
# Exact frames of the online model's published clips (scripts/eci/atlas_clip_frames.py), CPU only; codes staged to
# local disk. Reads results/vision/eci_atlas_v37/site (run after an assemble), writes results/vision/eci_atlas_v37/clip_frames/.
#   sbatch scripts/eci/atlas_clip_frames.sh
#SBATCH --job-name=atlas_frames
#SBATCH --output=results/vision/eci_atlas_v37/logs/%x_%j.out
#SBATCH --time=01:00:00
#SBATCH --partition=defaultp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
module load conda
conda activate crl
export PYTHONUNBUFFERED=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
export LOCAL_DIR=/localhome/$USER/${SLURM_JOB_ID:-local}_frames
mkdir -p $LOCAL_DIR
trap 'rm -rf $LOCAL_DIR' EXIT
python -u scripts/eci/atlas_clip_frames.py
