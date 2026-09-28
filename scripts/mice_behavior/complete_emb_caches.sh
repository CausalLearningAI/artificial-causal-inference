#!/bin/bash
#
# Append the frames missing from the legacy mice v1 embedding caches (rd64 for class_l-2;
# the 15 newly annotated observations for patch_grid4), dinov2 + dinov3.
#
# Usage:
#   MODE=check   sbatch scripts/mice_behavior/complete_emb_caches.sh   # prove settings on stored rows
#   MODE=extract sbatch scripts/mice_behavior/complete_emb_caches.sh   # append + re-check new rows
#
#SBATCH --job-name=mice_complete_emb
#SBATCH --output=logs/mice_complete_emb_%j.out
#SBATCH --error=logs/mice_complete_emb_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1

module load conda
conda activate crl

export PYTHONUNBUFFERED=1
set -euo pipefail

cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference

mkdir -p logs

MODE=${MODE:-check}
PY=scripts/mice_behavior/complete_emb_caches.py

for ENC in dinov2 dinov3; do
    for KIND in cls patch; do
        echo "=== ${MODE} ${ENC} ${KIND} ==="
        if [ "${MODE}" = "extract" ]; then
            N_LEGACY=$(python -c "from src.mice_behavior.emb_index import n_embedding_rows as n; print(n('dataset/mice/v1/embeddings/full/${ENC}/$([ ${KIND} = cls ] && echo class_l-2 || echo patch_grid4)'))")
            python -u ${PY} --encoder ${ENC} --kind ${KIND} --mode extract --num-workers 8
            python -u ${PY} --encoder ${ENC} --kind ${KIND} --mode check --rows new --n-legacy "${N_LEGACY}" --n-check 20
        else
            python -u ${PY} --encoder ${ENC} --kind ${KIND} --mode check --n-check 20
        fi
    done
done
echo "ALL DONE"
