#!/bin/bash
#
# Re-extract POV embeddings for ants/v5 (blue + yellow) with overwrite.
# Run after dataset/ants/v5/hf/ has been built by scripts/01_dataset/run.sh.
#
#SBATCH --job-name=reextract_v5_pov
#SBATCH --output=logs/reextract_v5_pov_%j.out
#SBATCH --error=logs/reextract_v5_pov_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=gpu100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:H100:1

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
echo "Re-extracting ants/v5 POV embeddings"
echo "Encoder: ${ENCODER}  Token: ${TOKEN}"
echo "=========================================="

for POV in blue yellow; do
    echo ""
    echo "--- POV: ${POV} ---"
    START=$(date +%s)
    python -u src/embedding/get_embeddings.py \
        experiment="ants/v5" \
        encoder="${ENCODER}" \
        token="${TOKEN}" \
        batch_size="${BATCH_SIZE}" \
        num_workers="${NUM_WORKERS}" \
        device="${DEVICE}" \
        +frame_type=pov \
        +pov_identity="${POV}" \
        overwrite.embeddings=true
    ELAPSED=$(($(date +%s) - START))
    printf "[%s done in %02d:%02d:%02d]\n" "${POV}" $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60))
done

echo ""
echo "=========================================="
echo "DONE: v5 POV embeddings re-extracted"
echo "=========================================="
