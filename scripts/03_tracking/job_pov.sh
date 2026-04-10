#!/bin/bash
#
# Step 2 job: extract POV crops for one {SUBJECT}/{VERSION}.
#
# Usage:
#   VERSION=v3 sbatch scripts/03_tracking/job_pov.sh
#   SUBJECT=ants VERSION=v5 OVERWRITE_POV=false sbatch scripts/03_tracking/job_pov.sh
#
#SBATCH --job-name=pov
#SBATCH --output=logs/pov_%x_%j.out
#SBATCH --error=logs/pov_%x_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=gpu100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:H100:1

set -euo pipefail
export PYTHONUNBUFFERED=1

SUBJECT=${SUBJECT:-ants}
VERSION=${VERSION:-v3}
OVERWRITE_POV=${OVERWRITE_POV:-${OVERWRITE:-false}}

module load conda
conda activate crl

mkdir -p logs

echo "========================================================"
echo " Step 2: POV extraction — ${SUBJECT}/${VERSION}"
echo " overwrite_pov=${OVERWRITE_POV}"
echo " $(date)"
echo "========================================================"

STEP_START=$(date +%s)

python -u src/tracking/get_pov_frames.py \
    --config-name "${SUBJECT}/${VERSION}" \
    +overwrite="${OVERWRITE_POV}"

STEP_ELAPSED=$(($(date +%s) - STEP_START))
printf "[POV done in %02d:%02d:%02d]\n" \
    $((STEP_ELAPSED/3600)) $((STEP_ELAPSED%3600/60)) $((STEP_ELAPSED%60))

echo ""
echo "Output: dataset/${SUBJECT}/${VERSION}/frames/pov/"
