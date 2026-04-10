#!/bin/bash
#
# Deploy the best PPCI ants model across all named configurations.
#
# Automatically selects the best frame type (pov > full) from the hparam search.
#
# Usage:
#   sbatch scripts/04_train/deploy.sh                          # auto-detect best, all configs
#   bash   scripts/04_train/deploy.sh --config validate        # one config only
#   bash   scripts/04_train/deploy.sh --eval-only              # re-eval + plots
#   python src/ppci/deploy_model.py --dry-run                  # preview, no training
#
# Output:
#   results/ppci/ants/hparam/deploy/{validate,final}/
#
#SBATCH --job-name=ppci_deploy
#SBATCH --output=logs/deploy_%j.out
#SBATCH --error=logs/deploy_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G

cd "${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}"

module load conda 2>/dev/null || true
conda activate crl

export PYTHONUNBUFFERED=1
set -euo pipefail

mkdir -p logs

echo "========================================================"
echo "PPCI deploy"
echo "  node    : $(hostname)"
echo "  gpu     : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'n/a')"
echo "  started : $(date)"
echo "========================================================"

START=$(date +%s)

python -u src/ppci/deploy_model.py "$@"

ELAPSED=$(( $(date +%s) - START ))
printf "\nTotal time: %02d:%02d:%02d\n" \
    $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60))
