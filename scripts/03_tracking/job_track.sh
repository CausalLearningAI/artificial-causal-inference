#!/bin/bash
#
# Track ants + generate demo videos for a single {SUBJECT}/{VERSION}.
# Submittable directly with sbatch or called by launcher scripts.
#
# Usage:
#   VERSION=v3 sbatch scripts/03_tracking/job_track.sh
#   VERSION=v1 SUBJECT=ants sbatch scripts/03_tracking/job_track.sh
#
#SBATCH --job-name=tracking
#SBATCH --output=logs/tracking_%x_%j.out
#SBATCH --error=logs/tracking_%x_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=defaultp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail
export PYTHONUNBUFFERED=1

SUBJECT=${SUBJECT:-ants}
VERSION=${VERSION:-v3}
OVERWRITE_TRACKING=${OVERWRITE_TRACKING:-${OVERWRITE:-false}}
SKIP_EXISTING_DEMOS=true
if [ "${OVERWRITE_TRACKING}" = "true" ]; then
    SKIP_EXISTING_DEMOS=false
fi

# ── Environment ───────────────────────────────────────────────────────────────
module load conda
conda activate crl

mkdir -p logs

echo "========================================================"
echo " Stage 3: Ant tracking — ${SUBJECT}/${VERSION}"
echo " $(date)"
echo "========================================================"

# ── Step 1: Track all videos ─────────────────────────────────────────────────
STEP_START=$(date +%s)
echo ""
echo ">>> Step 1/2: Tracking all videos..."

python -u src/tracking/get_tracking.py \
    --config-name "${SUBJECT}/${VERSION}" \
    +overwrite="${OVERWRITE_TRACKING}"

STEP_ELAPSED=$(($(date +%s) - STEP_START))
printf "[Tracking done in %02d:%02d:%02d]\n" \
    $((STEP_ELAPSED/3600)) $((STEP_ELAPSED%3600/60)) $((STEP_ELAPSED%60))

# ── Step 2: Generate demo videos (all experiments, full video) ────────────────
STEP_START=$(date +%s)
echo ""
echo ">>> Step 2/2: Generating full demo videos (all experiments)..."

python -u src/tracking/visualize_tracking.py \
    --config-name "${SUBJECT}/${VERSION}" \
    +n_sample=10 \
    +skip_existing="${SKIP_EXISTING_DEMOS}"

STEP_ELAPSED=$(($(date +%s) - STEP_START))
printf "[Visualization done in %02d:%02d:%02d]\n" \
    $((STEP_ELAPSED/3600)) $((STEP_ELAPSED%3600/60)) $((STEP_ELAPSED%60))

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo " Results for ${SUBJECT}/${VERSION}"
echo "========================================================"
echo ""
echo "  Tracking CSVs:  dataset/${SUBJECT}/${VERSION}/tracking/"
echo "  Demo videos:    results/${SUBJECT}/${VERSION}/tracking_viz/"
echo ""
echo "  How to view:"
echo "    ffplay results/${SUBJECT}/${VERSION}/tracking_viz/<obs_id>.mp4"
echo "    # or: vlc / mpv / scp to local machine"
echo ""
echo "  What to check:"
echo "    BLUE   circle → blue-marked nestmate (stays on blue ant)"
echo "    YELLOW circle → yellow/orange-marked nestmate"
echo "    GREEN  circle → focal (unmarked) ant"
echo "    Ring = POV crop boundary"
echo "    n_ants=3 throughout (drops to 1-2 only when ants clump)"
echo "    No persistent identity swaps between BLUE and YELLOW"
echo ""
echo "  If tracking looks wrong, tune in:"
echo "    configs/tracking/${SUBJECT}/${VERSION}.yaml"
echo "  Then re-run:"
echo "    VERSION=${VERSION} OVERWRITE=true sbatch scripts/03_tracking/job_track.sh"
echo ""
