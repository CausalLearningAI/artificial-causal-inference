#!/bin/bash
# CPU job: print raw structure of MABe22 npy files + a few ffprobe calls.
#SBATCH --job-name=mabe_probe
#SBATCH --output=results/vision/benchmark_mabe22/logs/%x_%j.out
#SBATCH --time=00:40:00
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
module load conda; conda activate crl
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
python -I scripts/benchmark/mabe22/probe.py
