#!/bin/bash
#
# NES explorer page "Exploratory Causal Inference x Mice / Ants" (scripts/eci/build_explorer.py, design in
# scripts/eci/explorer_template.html). One page; the title's domain word selects mice / ants, the Model bar the
# SAEs of that domain (mice: mouse mask fg448 (default) and full frame ep20; ants: ants mask antsfg; any other
# finished SAE result set under the NES roots is added automatically). Every listed neuron gets clips, packed
# into a few large H.264 files (clip packs). GPU for the per-patch heatmaps of the clips and the arena maps
# (DINOv2 + SAE); the rest is CPU / IO (codes, ffmpeg). Incremental (cache in <res>/_cache/explorer of each SAE).
#
# Usage: sbatch scripts/eci/build_explorer.sh
#   Default: DEFAULT_RES + discovered result sets -> results/vision/mice/eci/nes/matryoshka_btk_1024_k16_fg448_s0/explorer/
#   Other result sets:  EXTRA_ARGS="--res results/vision/mice/eci/nes/<sae> --res ..." sbatch scripts/eci/build_explorer.sh
#   After uploading the packs to the artifact asset store:
#     EXTRA_ARGS="--steps page,check --asset-map <map.json>" sbatch scripts/eci/build_explorer.sh
#   Outcomes default to bout rate (maxpool_bouts/) + mean activation (.), each when its summary.csv exists
#   (see the build_explorer.py docstring for the --outcome LABEL=SUBDIR:k=v spec).
#
#SBATCH --job-name=nes_explorer
#SBATCH --output=logs/nes_explorer_%j.out
#SBATCH --error=logs/nes_explorer_%j.err
#SBATCH --time=16:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

module load conda
conda activate crl
export PYTHONUNBUFFERED=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
mkdir -p logs
eval python -u scripts/eci/build_explorer.py ${EXTRA_ARGS:-}
