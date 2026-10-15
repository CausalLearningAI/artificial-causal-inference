#!/bin/bash
# CPU job: sbatch scripts/benchmark/mabe22/inventory.sh
#SBATCH --job-name=mabe_inv
#SBATCH --output=results/vision/benchmark_mabe22/logs/%x_%j.out
#SBATCH --time=02:00:00
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
module load conda; conda activate crl
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
python -I scripts/benchmark/mabe22/inventory.py
