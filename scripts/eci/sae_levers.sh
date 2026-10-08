#!/bin/bash
#
# SAE levers for ECI (scripts/eci/sae_levers.py). STEP = selftest | train | evaluate | both (train then evaluate).
#   train     stage the train frames to /localhome, pools + centring means, CONFIGS x 3 seeds -> $OUT/sae/
#   evaluate  stage the eval frames (loaded into RAM, staging dir removed), encode, cross-fitted scores, ants
#             dot-Voronoi read, contact sheets -> $OUT/{eval,align,codes_best,sheets}/, $OUT/summary.json
# GPU jobs go to the 'gpu' partition (not gpu100), excluding gpu150 (11 GB 2080 Ti) and gpu242 (GPU 0 failed 2026-10-06).
#
# Usage:
#   mkdir -p results/vision/eci_sae_levers/logs
#   OUT=results/vision/eci_sae_levers/mice
#   t=$(DOMAIN=mice OUT=$OUT STEP=train CONFIGS="base cell video merged hard" sbatch --parsable --export=ALL scripts/eci/sae_levers.sh)
#   DOMAIN=mice OUT=$OUT STEP=evaluate CONFIGS="base cell video merged hard" sbatch --export=ALL --dependency=afterok:$t scripts/eci/sae_levers.sh
#   smoke: STEP=both EXTRA_TRAIN="--max-train-frames 3000 --steps 300 --seeds 0" EXTRA_EVAL="--max-eval-frames 3000 --seeds 0"
#   ants: add EXTRA_EVAL="--deployed" to the job that evaluates 'base' (scores deployed antsfg neuron 90 as reference)
#SBATCH --job-name=eci_levers
#SBATCH --output=results/vision/eci_sae_levers/logs/%x_%j.out
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu150,gpu242
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G

module load conda
conda activate crl
export PYTHONUNBUFFERED=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
export LOCAL_DIR=/localhome/$USER/${SLURM_JOB_ID}
mkdir -p $LOCAL_DIR
trap 'rm -rf $LOCAL_DIR' EXIT
df -h /localhome | tail -1
nvidia-smi -L
DOMAIN=${DOMAIN:-mice}
CONFIGS=${CONFIGS:-base}
EXTRA_TRAIN=${EXTRA_TRAIN:-}
EXTRA_EVAL=${EXTRA_EVAL:-}
case ${STEP} in
  selftest)
    python -u scripts/eci/sae_levers.py selftest ;;
  train)
    python -u scripts/eci/sae_levers.py train --domain $DOMAIN --out-dir $OUT --configs $CONFIGS $EXTRA_TRAIN ;;
  evaluate)
    python -u scripts/eci/sae_levers.py evaluate --domain $DOMAIN --out-dir $OUT --configs $CONFIGS $EXTRA_EVAL ;;
  both)
    python -u scripts/eci/sae_levers.py train --domain $DOMAIN --out-dir $OUT --configs $CONFIGS $EXTRA_TRAIN
    python -u scripts/eci/sae_levers.py evaluate --domain $DOMAIN --out-dir $OUT --configs $CONFIGS $EXTRA_EVAL ;;
esac
