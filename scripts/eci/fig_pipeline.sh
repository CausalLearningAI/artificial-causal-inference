#!/bin/bash
#
# ECI representation figure (one row per domain: ants, mice, frogs). CPU only: DINOv2-base on 2 frames per domain,
# ffmpeg decode of 7 native frames per domain, ffprobe of the ants v2 / v3 sources, codes_max reads of the example clips.
# Usage: sbatch scripts/eci/fig_pipeline.sh                      (compute + plot)
#        STAGE_ARG=plot sbatch scripts/eci/fig_pipeline.sh       (re-render from the cache only)
#        STAGE_ARG=plot EXTRA_ARGS="--encoder-label 'V-JEPA 2' --grid 28" sbatch scripts/eci/fig_pipeline.sh
# Output: results/vision/eci_pipeline_fig/fig_pipeline.{pdf,png}, cache/, facts.json
#
#SBATCH --job-name=eci_pipefig
#SBATCH --output=logs/eci_pipefig_%j.out
#SBATCH --error=logs/eci_pipefig_%j.err
#SBATCH --time=00:45:00
#SBATCH --partition=defaultp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G

module load conda
conda activate crl
export PYTHONUNBUFFERED=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
mkdir -p logs
export STAGE=/localhome/$USER/$SLURM_JOB_ID
mkdir -p "$STAGE"
trap 'rm -rf "$STAGE"' EXIT
export MPLCONFIGDIR=$STAGE/mpl
eval python -u scripts/eci/fig_pipeline.py ${STAGE_ARG:-all} ${EXTRA_ARGS:-}
