#!/bin/bash
# Tadpole frame strips with real per-token heat maps for the atlas (scripts/eci/tadpole_atlas_strips.py), one GPU.
#SBATCH --job-name=tad_strips
#SBATCH --output=results/vision/eci_atlas_v37/logs/%x_%j.out
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
module load conda
conda activate crl
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used --format=csv,noheader -l 60 | sed -u 's/^/[gpu] /' &
SMI=$!
trap 'kill $SMI 2>/dev/null || true' EXIT
python -u scripts/eci/tadpole_atlas_strips.py
