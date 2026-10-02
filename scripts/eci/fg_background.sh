#!/bin/bash
#
# Foreground pipeline step 1: per-video backgrounds (200 frames per video, DINOv2-base on the
# whole frame at 448, no crop) -> dataset/mice/v1/eci/fg448/background/{observation_id}.npz.
# 8 tasks x 54 videos; finished videos are skipped (resubmit to resume).
#
# Usage: sbatch scripts/eci/fg_background.sh
#   odor-aligned frames (odor corner top right) -> dataset/mice/v1/eci/fg448al/background/:
#   EXTRA_ARGS="--align odor" sbatch --export=ALL -p gpu scripts/eci/fg_background.sh
#
#SBATCH --job-name=eci_fg_bg
#SBATCH --output=logs/eci_fg_bg_%A_%a.out
#SBATCH --error=logs/eci_fg_bg_%A_%a.err
#SBATCH --array=0-7
#SBATCH --time=02:00:00
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

python -u scripts/eci/fg_background.py --domain "${DOMAIN:-mice}" --task ${SLURM_ARRAY_TASK_ID} --n-tasks 8 --num-workers 14 ${EXTRA_ARGS:-}
