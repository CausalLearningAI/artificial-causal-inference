#!/bin/bash
#
# Successive-halving search for the online-augmented per-frame classifier.
# See search_online_aug.py docstring.
#   N_CONFIGS=32 INPUT_SIZE=504 sbatch scripts/mice_behavior/search_online_aug.sh
#
#SBATCH --job-name=search_oa
#SBATCH --output=logs/search_oa_%j.out
#SBATCH --error=logs/search_oa_%j.err
#SBATCH --time=20:00:00
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

WANDB_ARGS=""
if [ "${WANDB:-1}" = "1" ]; then
    WANDB_ARGS="--wandb"
fi

python -u scripts/mice_behavior/search_online_aug.py \
    --n-configs "${N_CONFIGS:-32}" \
    --rungs ${RUNGS:-4 8 14 24} \
    --input-size "${INPUT_SIZE:-504}" \
    --context-k "${CONTEXT_K:-2}" \
    --max-train-frames "${MAX_TRAIN_FRAMES:-300000}" \
    --batch-size "${BATCH_SIZE:-64}" \
    --read-workers "${READ_WORKERS:-32}" \
    --decode-workers "${DECODE_WORKERS:-8}" \
    --final-full-val "${FINAL_FULL_VAL:-3}" \
    ${WANDB_ARGS} \
    --tag "${TAG:-search_online_aug}"
