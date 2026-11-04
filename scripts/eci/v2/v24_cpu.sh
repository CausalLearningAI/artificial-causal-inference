#!/bin/bash
# v2.4 CPU steps: sbatch [--array=0-71] v24_cpu.sh <script.py> <args...>  (the array index is passed as --task by the caller's args via $T)
#SBATCH --partition=defaultp
#SBATCH --exclude=eta310,eta380
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference/results/vision/eci_v2_4/logs/%x_%A_%a.out
set -euo pipefail
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
args=("$@")
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then args=("${args[@]//@TASK@/$SLURM_ARRAY_TASK_ID}"); fi
bash /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference/scripts/eci/v2/v22_run.sh "${args[@]}"
