#!/bin/bash
#SBATCH --job-name=mabe_md
#SBATCH --output=results/vision/benchmark_mabe22/logs/%x_%j.out
#SBATCH --time=00:30:00
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
module load conda; conda activate crl
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
python -I scripts/benchmark/mabe22/make_md.py
