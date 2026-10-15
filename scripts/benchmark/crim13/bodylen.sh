#!/bin/bash
# CRIM13 body length second pass (see bodylen.py), CPU job: sbatch scripts/benchmark/crim13/bodylen.sh
#SBATCH --job-name=crim_bodylen
#SBATCH --output=results/vision/benchmark_crim13/logs/%x_%j.out
#SBATCH --time=02:00:00
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
export PYTHONUNBUFFERED=1 NWORK=12 OMP_NUM_THREADS=1
L=/localhome/$USER/$SLURM_JOB_ID
trap 'rm -rf $L' EXIT
/nfs/scistore19/locatgrp/rcadei/.conda/envs/crl/bin/python -I scripts/benchmark/crim13/bodylen.py \
  /mnt/beegfs/locatgrp/shared/rcadei/benchmark/crim13 $L results/vision/benchmark_crim13/inventory/bodylen.json
echo "ALLDONE $(date)"
