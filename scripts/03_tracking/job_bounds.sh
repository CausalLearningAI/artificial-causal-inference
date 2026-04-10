#!/bin/bash
#
# Optimize HSV color bounds for ant tracking.
# Called by launch_full.sh; versions passed via VERSIONS env var.
#
#SBATCH --job-name=bounds
#SBATCH --output=logs/bounds_%x_%j.out
#SBATCH --error=logs/bounds_%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=defaultp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G

set -euo pipefail
export PYTHONUNBUFFERED=1

VERSIONS=${VERSIONS:-"v2 v3 v4"}

module load conda
conda activate crl

echo "========================================================"
echo " Optimizing color bounds: ${VERSIONS}"
echo " $(date)"
echo "========================================================"

python src/tracking/optimize_bounds.py ${VERSIONS}

echo ""
echo "Done. $(date)"
