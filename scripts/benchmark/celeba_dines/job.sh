#!/bin/bash
#
# DINES vs NES on the CelebA semi-synthetic benchmark: generic CPU job wrapper (never a GPU partition).
# CMD = the python command line to run (relative to the repo root); it may read $LOCAL_DIR, a per-job staging directory
# on the node's local disk (/localhome/$USER/<job>), removed on exit.
#   CMD="tests/eci/test_dines.py results/eci/dines_celeba/unit_tests.json" sbatch scripts/benchmark/celeba_dines/job.sh
#   CMD="scripts/benchmark/celeba_dines/run_original.py --task \$SLURM_ARRAY_TASK_ID" sbatch --array=0-79 ... job.sh
#SBATCH --job-name=dines
#SBATCH --output=results/eci/dines_celeba/logs/%x_%A_%a.out
#SBATCH --partition=defaultp
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
PY=/nfs/scistore19/locatgrp/rcadei/.conda/envs/crl/bin/python
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export LOCAL_DIR=/localhome/$USER/${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}
mkdir -p $LOCAL_DIR
trap 'rm -rf $LOCAL_DIR' EXIT
echo "host $(hostname) job ${SLURM_JOB_ID:-} task ${SLURM_ARRAY_TASK_ID:-} start $(date -Is) cmd $CMD"
t0=$(date +%s)
eval $PY -u $CMD
echo "done $(date -Is) elapsed $(( $(date +%s) - t0 )) s"
