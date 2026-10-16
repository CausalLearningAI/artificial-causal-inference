#!/bin/bash
# R4 background-neuron diagnosis (CPU). Usage: sbatch scripts/eci/diag_r4bg/diag.sh <domain ...>
#SBATCH --job-name=r4bgdiag
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference/results/vision/eci_r4_bgdiag/logs/%x_%j.out
set -euo pipefail
REPO=/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
export LOCAL_DIR=/localhome/$USER/$SLURM_JOB_ID
mkdir -p "$LOCAL_DIR"
trap 'rm -rf "$LOCAL_DIR"' EXIT
source ~/.bashrc >/dev/null 2>&1 || true
PY=${PY:-/nfs/scistore19/locatgrp/rcadei/.conda/envs/crl/bin/python}
cd "$REPO"
echo "host $(hostname) python $(which $PY)"
$PY -u scripts/eci/diag_r4bg/diag.py "$@"
