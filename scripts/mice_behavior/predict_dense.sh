#!/bin/bash
#
# Dense (stride-1) inference for one model on one cohort. See predict_dense.py docstring.
#
#   TAG=xfit_f1 VERSION=v1 sbatch scripts/mice_behavior/predict_dense.sh
#   TAG=xfit_f2 VERSION=v1 EXTRA="--only-labelled --n-obs 3 --out-suffix _VALIDATE" \
#       sbatch scripts/mice_behavior/predict_dense.sh
#
#SBATCH --job-name=predict_dense
#SBATCH --output=logs/predict_dense_%j.out
#SBATCH --error=logs/predict_dense_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40S:1
# Frames are loaded and freed one OBSERVATION at a time (~9k frames, ~320 MB of JPEG bytes), so
# this does not need the 180G the training runs do. The jpeg cache, when passed, is memory-mapped
# and therefore reclaimable file-backed pages rather than anonymous RSS.
#SBATCH --mem=80G
#SBATCH --cpus-per-task=32

module load conda
conda activate crl
export PYTHONUNBUFFERED=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
mkdir -p logs

CACHE_ARGS=""
# v1 has a prebuilt cache covering the ANNOTATED frames only; the unlabelled pools are not in it,
# so this saves the read for a --only-labelled pass and is a harmless no-op otherwise.
if [ -n "${JPEG_CACHE_FILE:-}" ]; then
    CACHE_ARGS="--jpeg-cache-file ${JPEG_CACHE_FILE}"
fi

python -u scripts/mice_behavior/predict_dense.py \
    --tag "${TAG:?set TAG}" \
    --version "${VERSION:-v1}" \
    --chunk "${CHUNK:-512}" \
    --head-batch "${HEAD_BATCH:-64}" \
    --decode-threads "${DECODE_THREADS:-16}" \
    --read-workers "${READ_WORKERS:-32}" \
    ${CACHE_ARGS} ${EXTRA:-}
