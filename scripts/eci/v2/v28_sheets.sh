#!/bin/bash
# v2.8 sheets (CPU): sbatch -J NAME v28_sheets.sh <v28_sheets.py args...>   (stages /localhome via v22_run.sh)
#SBATCH --partition=defaultp
#SBATCH --exclude=eta310,eta380
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference/results/vision/eci_v2_8/logs/%x_%j.out
set -euo pipefail
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
bash /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference/scripts/eci/v2/v22_run.sh v28_sheets.py "$@"
