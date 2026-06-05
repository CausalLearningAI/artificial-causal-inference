#!/bin/bash
#
# Extract embeddings for a single encoder/token — submit directly with sbatch.
# For all encoders × tokens in one go, use all.sh instead.
#
# Usage:
#   ENCODER=dinov2 TOKEN=mean sbatch scripts/02_embeddings/single.sh
#   ENCODER=siglip2 TOKEN=class sbatch scripts/02_embeddings/single.sh
#
#SBATCH --job-name=get_embeddings
#SBATCH --output=logs/get_embeddings_%j.out
#SBATCH --error=logs/get_embeddings_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:H100:1

# Load environment
module load conda
conda activate crl

# Disable Python output buffering for real-time logs
export PYTHONUNBUFFERED=1

# Fail fast on any error
set -euo pipefail

# ---------------------------------------------------------------------------
# Encoder options (set ENCODER= before submitting or override on CLI)
#
# DINOv2 family (Meta, 2023):
#   ENCODER=dinov2             BATCH_SIZE=192   # base — recommended default
#   ENCODER=dinov2_large       BATCH_SIZE=80
#   ENCODER=dinov2_giant       BATCH_SIZE=32
#   ENCODER=dinov2_reg         BATCH_SIZE=192   # base + register tokens
#   ENCODER=dinov2_reg_large   BATCH_SIZE=80
#
# DINOv3 family (Meta, Aug 2025 — trained on 1.7B images, LVD-1689M):
#   ENCODER=dinov3             BATCH_SIZE=192   # base
#   ENCODER=dinov3_large       BATCH_SIZE=80
#
# SigLIP family (Google):
#   ENCODER=siglip             BATCH_SIZE=80    # base, 512px
#   ENCODER=siglip_large       BATCH_SIZE=112   # SO400M, 384px
#   ENCODER=siglip2            BATCH_SIZE=112   # SO400M, Feb 2025
#
# CLIP family (OpenAI):
#   ENCODER=clip               BATCH_SIZE=256
#   ENCODER=clip_large         BATCH_SIZE=112
#
# Other:
#   ENCODER=vit                BATCH_SIZE=256
#   ENCODER=mae                BATCH_SIZE=128
#   ENCODER=resnet             BATCH_SIZE=512
#   ENCODER=aimv2              BATCH_SIZE=160   # Apple AIMv2, 2025
# ---------------------------------------------------------------------------
ENCODER=${ENCODER:-dinov2}
TOKEN=${TOKEN:-class}
LAYER=${LAYER:--2}
BATCH_SIZE=${BATCH_SIZE:-192}
NUM_WORKERS=${NUM_WORKERS:-8}
DEVICE=${DEVICE:-cuda}

process_experiment() {
    local SUBJECT=$1
    local VERSION=$2
    local START_TIME=$(date +%s)

    STEP_START=$(date +%s)
    if python -u src/embedding/get_embeddings.py \
        experiment="${SUBJECT}/${VERSION}" \
        encoder="${ENCODER}" \
        token="${TOKEN}" \
        layer="${LAYER}" \
        batch_size="${BATCH_SIZE}" \
        num_workers="${NUM_WORKERS}" \
        device="${DEVICE}"; then
        STEP_ELAPSED=$(($(date +%s) - STEP_START))
        echo "[Embeddings completed in ${STEP_ELAPSED}s]"
    else
        echo "[Embeddings FAILED]"
        return 1
    fi
    echo ""

    local ELAPSED=$(($(date +%s) - START_TIME))
    echo "=========================================="
    printf "TOTAL TIME: %02d:%02d:%02d\n" $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60))
    echo "=========================================="
    echo ""
}

# Process experiments
# process_experiment "ants" "v1"
# process_experiment "ants" "v2"
# process_experiment "ants" "v3"
# process_experiment "ants" "v4"
# process_experiment "ants" "v5"

process_experiment "mice" "v1"
# process_experiment "mice" "v2"
