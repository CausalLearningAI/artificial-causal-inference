#!/bin/bash
# CPU job: sbatch scripts/benchmark/mabe22/unzip.sh   (then probe.sh)
#SBATCH --job-name=mabe_unzip
#SBATCH --output=results/vision/benchmark_mabe22/logs/%x_%j.out
#SBATCH --time=03:00:00
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
module load conda; conda activate crl
set -e
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
python -I scripts/benchmark/mabe22/unzip.py
echo ALLDONE
