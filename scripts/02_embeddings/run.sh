#!/bin/bash
#
# Extract embeddings for all experiments
#
#SBATCH --job-name=get_embeddings
#SBATCH --output=logs/get_embeddings_%j.out
#SBATCH --error=logs/get_embeddings_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=visualize
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

# Load environment
module load conda
conda activate crl

# Disable Python output buffering for real-time logs
export PYTHONUNBUFFERED=1

# Fail fast on any error
set -euo pipefail

# Embedding settings (override as needed)
ENCODER=${ENCODER:-dinov2}
TOKEN=${TOKEN:-class}
BATCH_SIZE=${BATCH_SIZE:-32}
NUM_WORKERS=${NUM_WORKERS:-4}
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
process_experiment "ants" "v1"
process_experiment "ants" "v2"
process_experiment "ants" "v3"
process_experiment "ants" "v4"

# process_experiment "mice" "v1"
# process_experiment "mice" "v2"
