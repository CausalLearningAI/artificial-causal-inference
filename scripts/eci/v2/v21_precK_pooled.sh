#!/bin/bash
# pooled ants (post hoc) array task: tag index -> v1_s0..10, S_s1001..6, V_s1001..6
#SBATCH --job-name=v21_pK_pool
#SBATCH --output=/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference/results/vision/eci_v2_1/precK/%x_%A_%a.out
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --partition=defaultp
#SBATCH --exclude=eta310
#SBATCH --array=0-22
module load conda
conda activate crl
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
TAGS=(v1_s0 v1_s1 v1_s2 v1_s3 v1_s4 v1_s5 v1_s6 v1_s7 v1_s8 v1_s9 v1_s10 S_s1001 S_s1002 S_s1003 S_s1004 S_s1005 S_s1006 V_s1001 V_s1002 V_s1003 V_s1004 V_s1005 V_s1006)
python scripts/eci/v2/v21_precK.py pooled ${TAGS[$SLURM_ARRAY_TASK_ID]}
