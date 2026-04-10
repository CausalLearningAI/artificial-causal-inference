#!/bin/bash
#
# Step 3 job: extract POV embeddings for one {SUBJECT}/{VERSION}/{ENCODER}/{TOKEN}.
#
# Usage:
#   VERSION=v3 ENCODER=dinov2 TOKEN=class sbatch scripts/03_tracking/job_embed.sh
#   SUBJECT=ants VERSION=v5 ENCODER=dinov3 TOKEN=class OVERWRITE_EMBEDDINGS=false sbatch scripts/03_tracking/job_embed.sh
#
#SBATCH --job-name=emb
#SBATCH --output=logs/emb_%x_%j.out
#SBATCH --error=logs/emb_%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1

set -euo pipefail
export PYTHONUNBUFFERED=1

SUBJECT=${SUBJECT:-ants}
VERSION=${VERSION:-v3}
ENCODER=${ENCODER:-dinov2}
TOKEN=${TOKEN:-class}
POV_IDENTITY=${POV_IDENTITY:-blue}
NUM_WORKERS=${NUM_WORKERS:-8}
DEVICE=${DEVICE:-cuda}
OVERWRITE_EMBEDDINGS=${OVERWRITE_EMBEDDINGS:-${OVERWRITE:-false}}

# Optional resource overrides at submission time.
# Examples:
#   EMBED_CPUS=4 EMBED_MEM=32G EMBED_TIME=12:00:00 EMBED_PARTITION=gpu EMBED_GRES=gpu:1
EMBED_PARTITION=${EMBED_PARTITION:-gpu}
EMBED_GRES=${EMBED_GRES:-gpu:1}
EMBED_CPUS=${EMBED_CPUS:-8}
EMBED_MEM=${EMBED_MEM:-48G}
EMBED_TIME=${EMBED_TIME:-24:00:00}

if [ -z "${BATCH_SIZE:-}" ]; then
    case "${ENCODER}" in
        dinov2|dinov3) BATCH_SIZE=192 ;;
        siglip2) BATCH_SIZE=112 ;;
        siglip) BATCH_SIZE=80 ;;
        *) BATCH_SIZE=96 ;;
    esac
else
    BATCH_SIZE=${BATCH_SIZE}
fi

module load conda
conda activate crl

mkdir -p logs

echo "========================================================"
echo " Step 3: Embeddings — ${SUBJECT}/${VERSION}"
echo " encoder=${ENCODER} token=${TOKEN} pov_identity=${POV_IDENTITY} overwrite_embeddings=${OVERWRITE_EMBEDDINGS}"
echo " $(date)"
echo "========================================================"

STEP_START=$(date +%s)

python -u src/embedding/get_embeddings.py \
    experiment="${SUBJECT}/${VERSION}" \
    encoder="${ENCODER}" \
    token="${TOKEN}" \
    batch_size="${BATCH_SIZE}" \
    num_workers="${NUM_WORKERS}" \
    device="${DEVICE}" \
    +frame_type="pov" \
    +pov_identity="${POV_IDENTITY}" \
    overwrite.embeddings="${OVERWRITE_EMBEDDINGS}"

STEP_ELAPSED=$(($(date +%s) - STEP_START))
printf "[Embeddings done in %02d:%02d:%02d]\n" \
    $((STEP_ELAPSED/3600)) $((STEP_ELAPSED%3600/60)) $((STEP_ELAPSED%60))

echo ""
echo "Output: dataset/${SUBJECT}/${VERSION}/embeddings/pov/${POV_IDENTITY}/${ENCODER}/${TOKEN}/"
