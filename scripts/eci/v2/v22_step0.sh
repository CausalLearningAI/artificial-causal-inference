#!/bin/bash
# v2.2 Step 0: readout cache + per-neuron diagnosis for the stored V (1001-1006) and v1 (s0-10) mice models (CPU array)
#SBATCH --job-name=v22_s0
#SBATCH --output=/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference/results/vision/eci_v2_2/logs/%x_%A_%a.out
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=defaultp
#SBATCH --exclude=eta310
#SBATCH --array=0-16
module load conda
conda activate crl
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
K=(V_s1001 V_s1002 V_s1003 V_s1004 V_s1005 V_s1006 v1_s0 v1_s1 v1_s2 v1_s3 v1_s4 v1_s5 v1_s6 v1_s7 v1_s8 v1_s9 v1_s10)
M=${K[$SLURM_ARRAY_TASK_ID]}
python scripts/eci/v2/v22_core.py cache --model $M
python scripts/eci/v2/v22_core.py diag --model $M
