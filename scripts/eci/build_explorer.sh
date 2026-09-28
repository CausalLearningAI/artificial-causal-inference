#!/bin/bash
#
# NES explorer page "Mouse Concept Atlas" (scripts/eci/build_explorer.py). GPU for the per-patch
# heatmaps of the top clips (DINOv2 + SAE); the rest is CPU / IO (codes scan, ffmpeg). Incremental.
#
# Usage: sbatch scripts/eci/build_explorer.sh
#   Default: --res results/vision/mice/eci/nes/matryoshka_btk_1024_k16_ep20_s0, outcomes = bout rate
#   (maxpool_bouts/, default) + mean activation (.), each included when its summary.csv exists.
#   Other result set / outcomes (see the build_explorer.py docstring for the LABEL=SUBDIR:k=v spec):
#   EXTRA_ARGS="--res results/vision/mice/eci/nes/<sae> --outcome 'Mean activation=.'" sbatch ...
#
#SBATCH --job-name=nes_explorer
#SBATCH --output=logs/nes_explorer_%j.out
#SBATCH --error=logs/nes_explorer_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1

module load conda
conda activate crl
export PYTHONUNBUFFERED=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
mkdir -p logs
eval python -u scripts/eci/build_explorer.py ${EXTRA_ARGS:-}
