#!/bin/bash
# v2.4 per-model scoring chain (CPU): DOM=mice KEY=P2W_s2401 NT=24 sbatch v24_post.sh  -> merge codes parts, HMM cache, tag, (mice) prep, fwdcheck
#SBATCH --partition=defaultp
#SBATCH --exclude=eta310,eta380
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference/results/vision/eci_v2_4/logs/%x_%j.out
set -euo pipefail
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
R=/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference/scripts/eci/v2/v22_run.sh
bash $R v24_train.py merge --dom $DOM --keys $KEY --ntask $NT
bash $R v24_eval.py cache --dom $DOM --key $KEY
bash $R v24_eval.py tag --dom $DOM --key $KEY
if [ $DOM == mice ]; then bash $R v24_eval.py prep --dom $DOM --key $KEY; fi
bash $R v24_eval.py fwdcheck --dom $DOM --key $KEY
