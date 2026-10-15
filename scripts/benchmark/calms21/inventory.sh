#!/bin/bash
# CalMS21 inventory, CPU job: sbatch scripts/benchmark/calms21/inventory.sh
#SBATCH --job-name=calms_inv
#SBATCH --output=results/vision/benchmark_calms21/logs/%x_%j.out
#SBATCH --time=02:00:00
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
module load conda; conda activate crl
export PYTHONUNBUFFERED=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
python -I scripts/benchmark/calms21/inventory.py
