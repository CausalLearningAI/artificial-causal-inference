#!/bin/bash
#
# Re-extract blue POV embeddings for ants/v2 with num_workers=1 to fix
# worker-0 NaN corruption (22/132000 frames affected).
#
#SBATCH --job-name=reextract_v2_blue_pov
#SBATCH --output=logs/reextract_v2_blue_pov_%j.out
#SBATCH --error=logs/reextract_v2_blue_pov_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:3090:1

module load conda
conda activate crl

export PYTHONUNBUFFERED=1
set -euo pipefail

echo "=========================================="
echo "Re-extracting ants/v2 blue POV embeddings"
echo "Using num_workers=1 to avoid worker-0 NaN bug"
echo "=========================================="

START=$(date +%s)
python -u src/embedding/get_embeddings.py \
    experiment="ants/v2" \
    encoder=dinov2 \
    token=class \
    batch_size=192 \
    num_workers=1 \
    device=cuda \
    +frame_type=pov \
    +pov_identity=blue \
    overwrite.embeddings=true
ELAPSED=$(($(date +%s) - START))
printf "[done in %02d:%02d:%02d]\n" $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60))
