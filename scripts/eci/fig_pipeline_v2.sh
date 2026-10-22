#!/bin/bash
#
# ECI v2 pipeline figure (scripts/eci/fig_pipeline_v2.py). GPU: DINOv3-L on frame t per domain (the frozen bf16 path,
# to verify against the token store and draw the token / pre-activation glyphs); reads one full-pass token npz per
# domain and one column of the primary smoothed codes. readout (CPU): continuous video representation of the example
# video, verified against the NES runner caches. Plot only: STAGE_ARG=plot (light, also fine without Slurm).
# Usage: sbatch -p gpu --gres=gpu:1 --constraint="L40S|A40" scripts/eci/fig_pipeline_v2.sh
#        STAGE_ARG=readout sbatch -p defaultp --export=ALL scripts/eci/fig_pipeline_v2.sh   (then STAGE_ARG=plot)
# Output: results/vision/eci_v2/fig/fig_pipeline_v2.png, fig_pipeline_v2_preview.png, crop_a..crop_d.png, cache/,
#         facts.json, readout.json
#
#SBATCH --job-name=eci_v2_pipefig
#SBATCH --output=/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference/results/vision/eci_v2/logs/%x_%j.out
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
module load conda
conda activate crl
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
export STAGE=/localhome/$USER/${SLURM_JOB_ID:-local}_pipefig
mkdir -p "$STAGE" 2>/dev/null || { export STAGE=${TMPDIR:-/tmp}/${USER}_${SLURM_JOB_ID:-local}; mkdir -p "$STAGE"; }
trap 'rm -rf "$STAGE"' EXIT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLCONFIGDIR=$STAGE/mpl
nvidia-smi -L 2>/dev/null || true
python -u scripts/eci/fig_pipeline_v2.py ${STAGE_ARG:-all} ${EXTRA_ARGS:-}
