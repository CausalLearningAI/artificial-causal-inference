#!/bin/bash
# v2.9 split addition (POST HOC): sbatch --array=0-N v29_split.sh <comma-separated keys>  (one dictionary per array task)
#SBATCH --partition=defaultp
#SBATCH --exclude=eta310,eta380,eta338
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference/results/vision/eci_v2_9/logs/%x_%A_%a.out
set -euo pipefail
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16} OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
IFS=',' read -ra K <<< "$1"
bash /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference/scripts/eci/v2/v22_run.sh v29_split.py --key ${K[${SLURM_ARRAY_TASK_ID:-0}]}
