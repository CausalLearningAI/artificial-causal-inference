#!/bin/bash
#
# ECI neuron galleries (src/eci/viz.py) on the full per-frame SAE codes of mice v1.
# Needs dataset/mice/v1/eci/codes/<sae>/DONE (scripts/eci/encode_merge.sh). The scan over the
# (2.59M, 1024) memmap is CPU/IO bound; the GPU recomputes the patch heatmaps (~24 frames/neuron).
#
# Usage:
#   sbatch scripts/eci/neuron_gallery.sh                                   # first 128 neurons
#   EXTRA_ARGS="--prefix 1024" sbatch scripts/eci/neuron_gallery.sh
#   EXTRA_ARGS="--stats results/.../nes.csv --tag nes" sbatch scripts/eci/neuron_gallery.sh
#
#SBATCH --job-name=eci_gallery
#SBATCH --output=logs/eci_gallery_%j.out
#SBATCH --error=logs/eci_gallery_%j.err
#SBATCH --time=04:00:00
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

python -u scripts/eci/neuron_gallery.py \
    --sae matryoshka_btk_1024_k16_ep20_s0 \
    --source full \
    ${EXTRA_ARGS:---prefix 128}
