#!/bin/bash
# v2.2 ants tags: KEYS="k1 k2 ..." sbatch --array=0-(n-1) v22_ants.sh
#SBATCH --output=/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference/results/vision/eci_v2_2/logs/%x_%A_%a.out
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --partition=defaultp
#SBATCH --exclude=eta310
module load conda
conda activate crl
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
read -ra K <<< "$KEYS"
python scripts/eci/v2/v22_ants.py tag --model ${K[$SLURM_ARRAY_TASK_ID]}
