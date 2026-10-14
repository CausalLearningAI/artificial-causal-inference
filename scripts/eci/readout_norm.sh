#!/bin/bash
#
# Readout normalisation of the C1 deployment SAEs (scripts/eci/readout_norm.py). CPU only (defaultp).
# STEP = run (per DOMAIN = mice | ants | frogs) | figure. codes_max is staged to /localhome/$USER/$SLURM_JOB_ID
# (removed on exit); the normalised codes live there only.
#
#   mkdir -p results/vision/eci_readout_norm/logs
#   DOMAIN=ants STEP=run sbatch --export=ALL scripts/eci/readout_norm.sh
#   DOMAIN=mice STEP=run sbatch --time=04:00:00 --mem=96G --export=ALL scripts/eci/readout_norm.sh
#   STEP=figure sbatch --time=00:10:00 --mem=4G --cpus-per-task=1 --export=ALL scripts/eci/readout_norm.sh
#SBATCH --job-name=eci_rnorm
#SBATCH --output=results/vision/eci_readout_norm/logs/%x_%j.out
#SBATCH --time=02:30:00
#SBATCH --partition=defaultp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

module load conda
conda activate crl
export PYTHONUNBUFFERED=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
export LOCAL_DIR=/localhome/$USER/${SLURM_JOB_ID}
mkdir -p $LOCAL_DIR
trap 'rm -rf $LOCAL_DIR' EXIT
hostname; df -h /localhome | tail -1
STEP=${STEP:-run}
DOMAIN=${DOMAIN:-mice}
python -u scripts/eci/readout_norm.py ${STEP} --domain $DOMAIN
