#!/bin/bash
#
# Second-level (relational) SAEs for ECI (scripts/eci/l2sae.py). STEP = selftest | train | evaluate | video | both
#   train     stage the train frames, level-1 sparse codes, level-2 + level-3 SAEs x seeds -> $OUT/sae/
#   evaluate  stage the eval frames, level-1/2/3 read-outs, cross-fitted scores, ants dot-Voronoi read, collective
#             proxies, contact sheets -> $OUT/{eval,codes_best,sheets}/, $OUT/summary.json, $OUT/table.md
#   video     (mice, CPU: submit with -p defaultp --gres=none) video-level measurements raw + fg-count controlled
# GPU jobs go to the 'gpu' partition (not gpu100), excluding gpu150 (11 GB 2080 Ti) and gpu242 (GPU 0 failed 2026-10-06).
#
# Usage:
#   mkdir -p results/vision/eci_l2sae/logs
#   OUT=results/vision/eci_l2sae/mice
#   t=$(DOMAIN=mice OUT=$OUT STEP=train sbatch --parsable --time=01:30:00 --export=ALL scripts/eci/l2sae.sh)
#   e=$(DOMAIN=mice OUT=$OUT STEP=evaluate sbatch --parsable --time=01:30:00 --export=ALL --dependency=afterok:$t scripts/eci/l2sae.sh)
#   DOMAIN=mice OUT=$OUT STEP=video sbatch -p defaultp --gres=none --mem=32G --time=00:20:00 --export=ALL --dependency=afterok:$e scripts/eci/l2sae.sh
#   ants: EXTRA_EVAL="--deployed" (dot-Voronoi read of the deployed antsfg n90 as reference)
#   smoke: STEP=both EXTRA_TRAIN="--max-train-frames 3000 --steps 300 --seeds 0" EXTRA_EVAL="--max-eval-frames 3000 --seeds 0"
#SBATCH --job-name=eci_l2sae
#SBATCH --output=results/vision/eci_l2sae/logs/%x_%j.out
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
nvidia-smi -L 2>/dev/null || echo "no GPU"
DOMAIN=${DOMAIN:-mice}
EXTRA_TRAIN=${EXTRA_TRAIN:-}
EXTRA_EVAL=${EXTRA_EVAL:-}
case ${STEP} in
  selftest)
    python -u scripts/eci/l2sae.py selftest ;;
  train)
    python -u scripts/eci/l2sae.py train --domain $DOMAIN --out-dir $OUT $EXTRA_TRAIN ;;
  evaluate)
    python -u scripts/eci/l2sae.py evaluate --domain $DOMAIN --out-dir $OUT $EXTRA_EVAL ;;
  video)
    python -u scripts/eci/l2sae.py video --domain $DOMAIN --out-dir $OUT ;;
  both)
    python -u scripts/eci/l2sae.py train --domain $DOMAIN --out-dir $OUT $EXTRA_TRAIN
    python -u scripts/eci/l2sae.py evaluate --domain $DOMAIN --out-dir $OUT $EXTRA_EVAL ;;
esac
