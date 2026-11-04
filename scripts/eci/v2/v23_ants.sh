#!/bin/bash
# v2.3 ants tags: KEYS="k1 k2 ..." sbatch --array=0-(n-1) v23_ants.sh
#SBATCH --output=/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference/results/vision/eci_v2_3/logs/%x_%A_%a.out
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=defaultp
#SBATCH --exclude=eta310,eta380
module load conda
conda activate crl
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1 NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
read -ra K <<< "$KEYS"
python scripts/eci/v2/v23_ants.py tag --model ${K[$SLURM_ARRAY_TASK_ID]}
