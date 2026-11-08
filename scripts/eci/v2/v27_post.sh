#!/bin/bash
# v2.5 per-model scoring chain (CPU): DOM=mice KEY=W1_s2501 [FWD=1] [V25_EDGE=0] sbatch v25_post.sh -> cache (edge mode), tag, (mice) prep, (FWD) fwdcheck
#SBATCH --partition=defaultp
#SBATCH --exclude=eta310,eta380
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference/results/vision/eci_v2_7/logs/%x_%j.out
set -euo pipefail
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
R=/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference/scripts/eci/v2/v22_run.sh
if [ "${V25_EDGE:-1}" == 1 ]; then bash $R v27_eval.py cache --dom $DOM --key $KEY; fi
bash $R v27_eval.py tag --dom $DOM --key $KEY
if [ $DOM == mice ]; then bash $R v27_eval.py prep --dom $DOM --key $KEY; fi
if [ "${FWD:-0}" == 1 ]; then bash $R v27_eval.py fwdcheck --dom $DOM --key $KEY; fi
