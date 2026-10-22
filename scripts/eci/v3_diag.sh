#!/bin/bash
# v3 alignment-drop diagnosis (scripts/eci/v3_diag.py), CPU only. STEP = labels | score | seeds; EXTRA = more arguments.
#   sbatch -p defaultp --export=ALL,STEP=score scripts/eci/v3_diag.sh
#SBATCH --job-name=v3diag
#SBATCH --output=results/vision/eci_v3_diag/logs/%x_%j.out
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
module load conda
conda activate crl
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
export LOCAL_DIR=/localhome/$USER/${SLURM_JOB_ID:-local}_v3diag
mkdir -p $LOCAL_DIR
trap 'rm -rf $LOCAL_DIR' EXIT
df -h /localhome | tail -1
echo "host $(hostname) job ${SLURM_JOB_ID:-} step $STEP extra ${EXTRA:-}"
python -u scripts/eci/v3_diag.py $STEP ${EXTRA:-}
