#!/bin/bash
# extended v1 vs v2 evaluation: array over (dom, tag); or STEP=report
#SBATCH --job-name=v21_ext
#SBATCH --output=/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference/results/vision/eci_v2_1/extended/logs/%x_%A_%a.out
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --partition=defaultp
#SBATCH --exclude=eta310
module load conda
conda activate crl
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
if [ "${STEP:-tag}" = report ]; then python scripts/eci/v2/v21_extended.py report; exit; fi
JOBS=()
for d in mice ants; do for t in v1_s0 v1_s1 v1_s2 v1_s3 v1_s4 v1_s5 v1_s6 v1_s7 v1_s8 v1_s9 v1_s10 V_s1001 V_s1002 V_s1003 V_s1004 V_s1005 V_s1006; do JOBS+=("$d $t"); done; done
set -- ${JOBS[$SLURM_ARRAY_TASK_ID]}
python scripts/eci/v2/v21_extended.py tag --dom $1 --model $2
