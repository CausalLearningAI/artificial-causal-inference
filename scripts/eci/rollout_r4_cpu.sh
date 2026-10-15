#!/bin/bash
# R4 CPU steps of one domain in order (scripts/eci/rollout_r4.py), e.g. STEPS="ftheta summaries" DOMAIN=ants_v2
#SBATCH --job-name=r4_cpu
#SBATCH --output=results/vision/eci_rollout_r4/logs/%x_%j.out
#SBATCH --time=01:00:00
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
module load conda
conda activate crl
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
for s in ${STEPS:?}; do python -u scripts/eci/rollout_r4.py $s --domain ${DOMAIN:?} ${EXTRA:-}; done
