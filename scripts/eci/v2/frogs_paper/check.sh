#!/bin/bash
# ECI v2 frog discoveries vs the lab's paper + side-alignment audit (scripts/eci/v2/frogs_paper/check.py). CPU only:
# reads the 35 juvenile token npz (BeeGFS), SLEAP 5 fps npz, the lab's kinematic tables, decodes 140 single frames.
#SBATCH --job-name=eci_v2_frogpaper
#SBATCH --output=/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference/results/vision/eci_v2/frogs_paper/%x_%j.out
#SBATCH --partition=defaultp
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
module load conda
conda activate crl
export PYTHONUNBUFFERED=1 MPLCONFIGDIR=/localhome/$USER/${SLURM_JOB_ID}_mpl
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
trap 'rm -rf "$MPLCONFIGDIR"' EXIT
python -u scripts/eci/v2/frogs_paper/check.py
