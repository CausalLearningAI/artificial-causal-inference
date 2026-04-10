#!/bin/bash
#
# Run one hyperparameter search job for PPCI ants (called as a SLURM array).
#
# Stages (run in order — each builds on the best from the previous):
#   backbone_context ~40 runs  (encoder × token × context_window × context_mode) — full pretrain+finetune
#   arch             ~19 runs  (hidden_dim, layers, dropout) — full pretrain+finetune
#   finetune         ~18 runs  (method, lr, weight_decay)    — loads saved pretrain
#   augmentation      ~9 runs  (noise_std, mixup_alpha)      — loads saved pretrain
#
# Usage:
#   # Step 0 — check exact config count for a stage:
#   python src/ppci/hparam_search.py --stage backbone_context --list
#
#   # Step 1 — submit each stage after the previous one completes:
#   STAGE=backbone_context sbatch --array=0-39  scripts/04_train/hparam_worker.sh
#   STAGE=arch             sbatch --array=0-25  scripts/04_train/hparam_worker.sh
#   STAGE=finetune         sbatch --array=0-19  scripts/04_train/hparam_worker.sh
#   STAGE=augmentation     sbatch --array=0-9   scripts/04_train/hparam_worker.sh
#
#   # Or submit the full dependency chain at once:
#   bash scripts/04_train/hparam_submit.sh
#
# GPU notes:
#   - Each job loads 2–60 GB of embeddings (concat mode with large encoders needs most).
#   - MLP training is fast (~30 s/epoch); full pretrain+finetune ≈ 30–60 min.
#   - Finetune-only jobs ≈ 10 min total.
#   - Use --array=0-N%K to cap concurrent jobs: e.g. --array=0-39%8
#
#SBATCH --job-name=ppci_hparam
#SBATCH --output=logs/hparam_%A_%a.out
#SBATCH --error=logs/hparam_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=96G

STAGE=${STAGE:-backbone_context}
FRAME_TYPE=${FRAME_TYPE:-full}

cd "${SLURM_SUBMIT_DIR}"

module load conda
conda activate crl

export PYTHONUNBUFFERED=1
set -euo pipefail

mkdir -p logs

echo "========================================================"
echo "PPCI hparam search"
echo "  stage      : ${STAGE}"
echo "  frame_type : ${FRAME_TYPE}"
echo "  job-idx    : ${SLURM_ARRAY_TASK_ID}"
echo "  node       : $(hostname)"
echo "  gpu        : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'n/a')"
echo "  started    : $(date)"
echo "========================================================"

START=$(date +%s)

python -u src/ppci/hparam_search.py \
    --stage      "${STAGE}" \
    --frame-type "${FRAME_TYPE}" \
    --job-idx    "${SLURM_ARRAY_TASK_ID}"

ELAPSED=$(( $(date +%s) - START ))
printf "\nTotal time: %02d:%02d:%02d\n" \
    $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60))
