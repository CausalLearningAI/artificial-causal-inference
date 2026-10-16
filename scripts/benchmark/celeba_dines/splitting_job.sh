#!/bin/bash
#SBATCH --job-name=splitdiag
#SBATCH --output=results/eci/ness_splitting/splitdiag_%j.out
#SBATCH --partition=defaultp
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
export LOCAL_DIR=/localhome/$USER/$SLURM_JOB_ID
mkdir -p $LOCAL_DIR; trap 'rm -rf $LOCAL_DIR' EXIT
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
/nfs/scistore19/locatgrp/rcadei/.conda/envs/crl/bin/python -u scripts/benchmark/celeba_dines/splitting_diagnostic.py results/eci/ness_splitting
