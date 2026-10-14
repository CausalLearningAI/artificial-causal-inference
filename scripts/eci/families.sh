#!/bin/bash
#
# Neuron families for mice nn / nt (scripts/eci/families.py). STEP = run (encode + analyse + sheets) | encode |
# analyse | sheets | table. GPU on the 'gpu' partition (never gpu100), excluding gpu150 / gpu242 / gpu230 / gpu241.
# analyse / sheets also run on CPU:  STEP=analyse sbatch -p defaultp --gres=none --export=ALL scripts/eci/families.sh
# Heavy inputs are staged to /localhome/$USER/$SLURM_JOB_ID (removed on exit).
#   mkdir -p results/vision/eci_families/logs; STEP=run sbatch --export=ALL scripts/eci/families.sh
#SBATCH --job-name=eci_fam
#SBATCH --output=results/vision/eci_families/logs/%x_%j.out
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu150,gpu242,gpu230,gpu241
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G

module load conda
conda activate crl
export PYTHONUNBUFFERED=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
export LOCAL_DIR=/localhome/$USER/${SLURM_JOB_ID}
mkdir -p $LOCAL_DIR
trap 'rm -rf $LOCAL_DIR' EXIT
df -h /localhome | tail -1
nvidia-smi -L 2>/dev/null || true
python -u scripts/eci/families.py ${STEP:-run} ${EXTRA:-}
