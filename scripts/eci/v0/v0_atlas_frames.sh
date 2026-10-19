#!/bin/bash
# Frame strips of the v0 neurons for the ECI atlas (scripts/eci/v0/v0_atlas_frames.py), CPU only.
#   sbatch scripts/eci/v0/v0_atlas_frames.sh            (KEYS=ants_v2,mice_v1 for a subset)
#SBATCH --job-name=v0frames
#SBATCH --output=results/vision/eci_atlas_v37/logs/%x_%j.out
#SBATCH --time=02:00:00
#SBATCH --partition=defaultp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
module load conda
conda activate crl
export PYTHONUNBUFFERED=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
LOC=/localhome/$USER/${SLURM_JOB_ID:-local}_v0frames
mkdir -p $LOC
trap 'rm -rf $LOC' EXIT
echo "node $(hostname), local disk free: $(df -h /localhome | tail -1)"
python -u scripts/eci/v0/v0_atlas_frames.py --local $LOC ${KEYS:+--keys $KEYS}
