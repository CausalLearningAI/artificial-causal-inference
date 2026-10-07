#!/bin/bash
#
# Tracker-free multi-scale SAEs (scripts/eci/multiscale_sae.py). STEP = selftest | train | evaluate | regional.
#   train     stage the train frames to /localhome, build window pools, 6 configs x 3 seeds -> $OUT/sae/
#   evaluate  stage the eval frames, encode, align, size control, discrimination, proxies, contact sheets
#             -> $OUT/{eval,align,codes_best,sheets}/, $OUT/summary.json (copied back once at the end)
#   regional  ants only, after evaluate: yellow-dot / blue-dot Voronoi read near the focal (diag_ants_pairs.py rule
#             mark_vor2 on window centres) for every config + the deployed SAE -> $OUT/regional*/
# GPU jobs go to the 'gpu' partition (not gpu100), excluding gpu150 (11 GB 2080 Ti) and gpu242 (GPU 0 failed 2026-10-06).
#
# Usage (chain, per domain):
#   mkdir -p results/vision/eci_multiscale/logs
#   OUT=results/vision/eci_multiscale/mice
#   t=$(DOMAIN=mice OUT=$OUT STEP=train sbatch --parsable --export=ALL scripts/eci/multiscale_sae.sh)
#   DOMAIN=mice OUT=$OUT STEP=evaluate sbatch --export=ALL --dependency=afterok:$t scripts/eci/multiscale_sae.sh
#   EXTRA="--max-train-frames 3000 --steps 300 --seeds 0 1" for a smoke test (evaluate: --max-eval-frames N)
#SBATCH --job-name=eci_multiscale
#SBATCH --output=results/vision/eci_multiscale/logs/%x_%j.out
#SBATCH --time=04:00:00
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
EXTRA=${EXTRA:-}
case ${DOMAIN} in
  mice) CAP=${CAP:-4000000} ;;
  ants) CAP=${CAP:-2000000} ;;
esac
case ${STEP} in
  selftest)
    python -u scripts/eci/multiscale_sae.py selftest --domain $DOMAIN ;;
  train)
    python -u scripts/eci/multiscale_sae.py train --domain $DOMAIN --out-dir $OUT --cap $CAP $EXTRA ;;
  evaluate)
    python -u scripts/eci/multiscale_sae.py evaluate --domain $DOMAIN --out-dir $OUT $EXTRA ;;
  regional)
    python -u scripts/eci/multiscale_sae.py regional --domain ants --out-dir $OUT $EXTRA ;;
esac
