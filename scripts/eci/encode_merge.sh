#!/bin/bash
#
# ECI step 3b: merge the 16 encode shards into the final arrays and verify them
# (row count, NaN / all-zero rows, alignment with the step-1 stored tokens).
#
# Usage:
#   sbatch --dependency=afterok:<encode_all job id> scripts/eci/encode_merge.sh
#
#SBATCH --job-name=eci_encode_merge
#SBATCH --output=logs/eci_encode_merge_%j.out
#SBATCH --error=logs/eci_encode_merge_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

module load conda
conda activate crl

export PYTHONUNBUFFERED=1
set -euo pipefail

cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
mkdir -p logs

python -u scripts/eci/encode_all.py \
    --sae ${SAE:-matryoshka_btk_1024_k16_ep20_s0} \
    --outputs ${OUTPUTS:-codes_mean,codes_max,cls_l-1} \
    --tokens-dir ${TOKENS_DIR:-dataset/mice/v1/eci/train_tokens/dinov2_base_l-1} \
    --n-shards 16 \
    --merge --verify
