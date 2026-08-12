#!/bin/bash
#
# Online per-batch encoding from a JPEG-bytes RAM cache, with optional D4 augmentation.
# See train_online_aug.py docstring.
#   AUGMENT=d4 CONTEXT_K=2 sbatch scripts/mice_behavior/train_online_aug.sh
#
#SBATCH --job-name=online_aug
#SBATCH --output=logs/online_aug_%j.out
#SBATCH --error=logs/online_aug_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=gpu100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=150G
#SBATCH --gres=gpu:H100:1

module load conda
conda activate crl
export PYTHONUNBUFFERED=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
mkdir -p logs

python -u scripts/mice_behavior/train_online_aug.py \
    --context-k "${CONTEXT_K:-2}" \
    --augment "${AUGMENT:-d4}" \
    --neg-ratio "${NEG_RATIO:-1}" \
    --max-train-frames "${MAX_TRAIN_FRAMES:-300000}" \
    --input-size "${INPUT_SIZE:-224}" \
    --batch-size "${BATCH_SIZE:-64}" \
    --read-workers "${READ_WORKERS:-32}" \
    --decode-workers "${DECODE_WORKERS:-16}" \
    --tag "${TAG:-online_aug}"
