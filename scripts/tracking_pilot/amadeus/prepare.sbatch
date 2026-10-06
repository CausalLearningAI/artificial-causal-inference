#!/bin/bash
#SBATCH --job-name=amd_prep
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference/results/tracking_pilot/logs/prep_%x_%j.log
# Usage: sbatch --job-name=prep_<video_id> prepare.sbatch <video_id>
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference/scripts/tracking_pilot/amadeus
export PATH=/nfs/scistore19/locatgrp/rcadei/.conda/envs/crl/bin:$PATH   # ffprobe
/nfs/scistore19/locatgrp/rcadei/tools/AMADEUS/.venv/bin/python prepare_video.py "$1"
