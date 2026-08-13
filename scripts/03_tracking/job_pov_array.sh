#!/bin/bash
#
# Step 2 (parallel): POV extraction, one observation per array task.
#
# The serial job_pov.sh walks all observations in one process at ~3.4 min each,
# so a 176-observation version needs ~10h and does not fit a single wall clock.
# The work is embarrassingly parallel — each observation reads its own tracking
# CSV and its own full frames — and get_pov_frames.py already takes obs_id, so
# one task per observation turns 10h into a few waves of a few minutes.
#
# The task -> obs_id map is sorted(dataset/{subject}/{version}/tracking/*.csv),
# the same order the serial script iterates, so array indices are stable across
# resubmissions as long as the tracking directory does not change.
#
# Resume-friendly: with OVERWRITE_POV=false the per-frame existence check in
# get_pov_frames.py skips frames already on disk, so a re-run costs only the
# missing ones. NOTE this is a per-FRAME check — an observation left partially
# written by a killed job is resumed, but its last JPG may be truncated. Delete
# incomplete observation directories before resuming rather than trusting them.
#
# Usage:
#   SUBJECT=ants VERSION=vA sbatch --array=0-175%32 scripts/03_tracking/job_pov_array.sh
#
# Size the array from the observation count:
#   ls dataset/ants/vA/tracking/*.csv | wc -l
#
#SBATCH --job-name=pov_arr
#SBATCH --output=logs/pov_arr_%x_%A_%a.out
#SBATCH --error=logs/pov_arr_%x_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --partition=defaultp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail
export PYTHONUNBUFFERED=1

cd "${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}"

SUBJECT=${SUBJECT:-ants}
VERSION=${VERSION:-v3}
OVERWRITE_POV=${OVERWRITE_POV:-${OVERWRITE:-false}}

module load conda
conda activate crl

mkdir -p logs

TRACKING_DIR="dataset/${SUBJECT}/${VERSION}/tracking"
mapfile -t OBS_IDS < <(find "${TRACKING_DIR}" -maxdepth 1 -name '*.csv' -printf '%f\n' \
                       | sed 's/\.csv$//' | sort)

IDX=${SLURM_ARRAY_TASK_ID:-0}
if [ "${IDX}" -ge "${#OBS_IDS[@]}" ]; then
    echo "[skip] task ${IDX} >= ${#OBS_IDS[@]} observations in ${TRACKING_DIR}"
    exit 0
fi
OBS_ID=${OBS_IDS[$IDX]}

echo "========================================================"
echo " Step 2 (array): POV extraction — ${SUBJECT}/${VERSION}"
echo " task=${IDX}/${#OBS_IDS[@]}  obs_id=${OBS_ID}"
echo " overwrite_pov=${OVERWRITE_POV}  node=$(hostname)"
echo " $(date)"
echo "========================================================"

STEP_START=$(date +%s)

python -u src/tracking/get_pov_frames.py \
    --config-name "${SUBJECT}/${VERSION}" \
    +obs_id="${OBS_ID}" \
    +overwrite="${OVERWRITE_POV}"

STEP_ELAPSED=$(($(date +%s) - STEP_START))
printf "[POV %s done in %02d:%02d:%02d]\n" "${OBS_ID}" \
    $((STEP_ELAPSED/3600)) $((STEP_ELAPSED%3600/60)) $((STEP_ELAPSED%60))
