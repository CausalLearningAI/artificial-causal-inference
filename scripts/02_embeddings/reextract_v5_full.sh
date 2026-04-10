#!/bin/bash
#
# Re-extract full embeddings for ants/v5 with overwrite.
#
#SBATCH --job-name=reextract_v5_full
#SBATCH --output=logs/reextract_v5_full_%j.out
#SBATCH --error=logs/reextract_v5_full_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:3090:1

module load conda
conda activate crl

export PYTHONUNBUFFERED=1
set -euo pipefail

ENCODER=${ENCODER:-dinov2}
TOKEN=${TOKEN:-class}
BATCH_SIZE=${BATCH_SIZE:-192}
NUM_WORKERS=${NUM_WORKERS:-8}
DEVICE=${DEVICE:-cuda}

echo "=========================================="
echo "Re-extracting ants/v5 full embeddings"
echo "Encoder: ${ENCODER}  Token: ${TOKEN}"
echo "=========================================="

START=$(date +%s)
python -u src/embedding/get_embeddings.py \
    experiment="ants/v5" \
    encoder="${ENCODER}" \
    token="${TOKEN}" \
    batch_size="${BATCH_SIZE}" \
    num_workers="${NUM_WORKERS}" \
    device="${DEVICE}" \
    overwrite.embeddings=true
ELAPSED=$(($(date +%s) - START))
printf "[done in %02d:%02d:%02d]\n" $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60))
