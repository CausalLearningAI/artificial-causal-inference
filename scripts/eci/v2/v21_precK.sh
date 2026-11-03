#!/bin/bash
# precision@K on the v2.1 ladder: sbatch --export=ALL,STEP=boot,DOM=mice|ants (or STEP=report) scripts/eci/v2/v21_precK.sh
#SBATCH --job-name=v21_precK
#SBATCH --output=/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference/results/vision/eci_v2_1/precK/%x_%j.out
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --partition=defaultp
#SBATCH --exclude=eta310
module load conda
conda activate crl
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
python scripts/eci/v2/v21_precK.py ${STEP} ${DOM:-}
