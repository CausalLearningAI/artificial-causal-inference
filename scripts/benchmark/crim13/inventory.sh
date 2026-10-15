#!/bin/bash
# CRIM13 inventory, CPU job: sbatch scripts/benchmark/crim13/inventory.sh
# inventory.py stages the small files once and each top-view .seq one at a time (per worker) to
# /localhome/$USER/$SLURM_JOB_ID, writes results/vision/benchmark_crim13/inventory/; the staging dir is removed.
#SBATCH --job-name=crim_inv
#SBATCH --output=results/vision/benchmark_crim13/logs/%x_%j.out
#SBATCH --time=04:00:00
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
PY=/nfs/scistore19/locatgrp/rcadei/.conda/envs/crl/bin/python
export PYTHONUNBUFFERED=1 NWORK=12 OMP_NUM_THREADS=1 OPENCV_FOR_THREADS_NUM=1
D=/mnt/beegfs/locatgrp/shared/rcadei/benchmark/crim13
L=/localhome/$USER/$SLURM_JOB_ID
OUT=results/vision/benchmark_crim13/inventory
trap 'rm -rf $L' EXIT
mkdir -p $L $OUT
echo "host $(hostname) start $(date)"
$PY -I scripts/benchmark/crim13/inventory.py $D $L $OUT
[ -f scripts/benchmark/crim13/inventory_md.py ] && $PY -I scripts/benchmark/crim13/inventory_md.py $OUT
echo "ALLDONE $(date)"
