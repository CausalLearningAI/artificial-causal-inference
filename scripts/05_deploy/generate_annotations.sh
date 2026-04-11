#!/bin/bash
#
# Generate per-observation annotation CSVs from the deployed model.
#
# Runs generate_annotations.py for the target version (default: v5).
# Auto-detects the deployed model from results/ppci/ants/performances/.
#
# Usage:
#   sbatch scripts/05_deploy/generate_annotations.sh
#   sbatch scripts/05_deploy/generate_annotations.sh --version v4
#   bash   scripts/05_deploy/generate_annotations.sh --obs 5_17_7
#
# Output:
#   results/ppci/ants/annotations/{version}/
#
#SBATCH --job-name=ppci_ann
#SBATCH --output=logs/generate_annotations_%j.out
#SBATCH --error=logs/generate_annotations_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

cd "${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}"

module load conda 2>/dev/null || true
conda activate crl

export PYTHONUNBUFFERED=1
set -euo pipefail

mkdir -p logs

echo "========================================================"
echo "PPCI generate annotations"
echo "  node    : $(hostname)"
echo "  gpu     : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'n/a')"
echo "  started : $(date)"
echo "========================================================"

START=$(date +%s)

python -u src/ppci/generate_annotations.py "$@"

ELAPSED=$(( $(date +%s) - START ))
printf "\nTotal time: %02d:%02d:%02d\n" \
    $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60))
