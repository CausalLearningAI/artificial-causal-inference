#!/bin/bash
# top-activating frame sheets (CPU): sbatch --export=ALL,DOM=mice|ants scripts/eci/v2/v21_extended_sheet.sh
#SBATCH --job-name=v21_ext_sheet
#SBATCH --output=/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference/results/vision/eci_v2_1/extended/logs/%x_%j.out
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --partition=defaultp
#SBATCH --exclude=eta310
module load conda
conda activate crl
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1 HF_HUB_OFFLINE=1
set -euo pipefail
export LOCAL_DIR=/localhome/$USER/$SLURM_JOB_ID
mkdir -p $LOCAL_DIR
trap 'rm -rf $LOCAL_DIR' EXIT
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
python scripts/eci/v2/v21_extended_sheet.py --dom ${DOM}
