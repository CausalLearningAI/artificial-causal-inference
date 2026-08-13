#!/bin/bash
#
# TTA + temporal smoothing + checkpoint ensembling. See eval_ensemble.py docstring.
#   TTA=rot4 sbatch scripts/mice_behavior/eval_ensemble.sh
#
#SBATCH --job-name=ens_eval
#SBATCH --output=logs/ens_eval_%j.out
#SBATCH --error=logs/ens_eval_%j.err
#SBATCH --time=16:00:00
#SBATCH --partition=gpu100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --gres=gpu:H100:1

module load conda
conda activate crl
export PYTHONUNBUFFERED=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
mkdir -p logs

python -u scripts/mice_behavior/eval_ensemble.py \
    --tta "${TTA:-rot4}" \
    --tta-top-k "${TTA_TOP_K:-3}" \
    --n-train-obs "${N_TRAIN_OBS:-12}" \
    --batch-size "${BATCH_SIZE:-64}" \
    --read-workers "${READ_WORKERS:-32}" \
    --decode-workers "${DECODE_WORKERS:-8}" \
    --tag "${TAG:-ensemble}"
