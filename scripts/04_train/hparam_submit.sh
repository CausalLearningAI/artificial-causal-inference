#!/bin/bash
#
# Submit the full PPCI ants hyperparameter search as a SLURM dependency chain.
# Each stage starts automatically once ALL jobs from the previous stage finish.
#
# Usage:
#   FRAME_TYPE=full bash scripts/04_train/hparam_submit.sh                    # full frame (skips completed runs)
#   FRAME_TYPE=pov  bash scripts/04_train/hparam_submit.sh                    # pov frame
#   FRAME_TYPE=full bash scripts/04_train/hparam_submit.sh --dry-run          # preview
#   FRAME_TYPE=full bash scripts/04_train/hparam_submit.sh --clean            # delete previous results & rerun
#   FRAME_TYPE=full bash scripts/04_train/hparam_submit.sh --clean --dry-run  # preview with clean
#
# Monitor: squeue -u $USER
# Results: python src/ppci/hparam_search.py --print-results --frame-type $FRAME_TYPE
#
set -euo pipefail

# Parse flags (order-independent)
DRY=""
CLEAN=""
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY="--dry-run" ;;
        --clean)   CLEAN="--clean" ;;
    esac
done
FRAME_TYPE="${FRAME_TYPE:-full}"
SCRIPT="scripts/04_train/hparam_worker.sh"
RUNNER="src/ppci/hparam_search.py"

mkdir -p logs

# ── helper: submit one stage, return job ID ───────────────────────────────────
submit() {
    local stage="$1"
    local max_idx="$2"    # --array upper bound (safe to over-provision; NOOP beyond n_configs)
    local dep_job="$3"    # job ID of preceding stage ("" = no dependency)

    local dep_arg=""
    [[ -n "${dep_job}" ]] && dep_arg="--dependency=afterany:${dep_job}"

    if [[ "${DRY}" == "--dry-run" ]]; then
        local concurrency="${MAX_CONCURRENT:-4}"
        echo "[DRY] FRAME_TYPE=${FRAME_TYPE} STAGE=${stage} sbatch --array=0-${max_idx}%${concurrency} ${dep_arg} --parsable ${SCRIPT}" >&2
        echo "dry_${stage}"
        return
    fi

    local jobid
    local concurrency="${MAX_CONCURRENT:-4}"
    jobid=$(FRAME_TYPE="${FRAME_TYPE}" STAGE="${stage}" sbatch --array=0-"${max_idx}%${concurrency}" ${dep_arg} --parsable "${SCRIPT}")
    printf "[SUBMITTED] %-12s  job_id=%-8s  array=0-%-2s  dep=%s\n" \
        "${stage}" "${jobid}" "${max_idx}" "${dep_job:-none}" >&2
    echo "${jobid}"
}

# ── results directory ─────────────────────────────────────────────────────────
RESULTS_DIR="results/ppci/ants/hparam/${FRAME_TYPE}"

# ── optionally clean previous results (must pass --clean explicitly) ──────────
# NOTE: Do NOT clean automatically — it causes a race condition if this script
# is invoked multiple times (later cleanup deletes dirs while earlier arch jobs
# are still running). Completed runs are skipped via metrics.json check in run_one.
if [[ "${CLEAN}" == "--clean" ]]; then
    for stage_dir in "${RESULTS_DIR}"/backbone_context "${RESULTS_DIR}"/arch \
                     "${RESULTS_DIR}"/finetune "${RESULTS_DIR}"/augmentation \
                     "${RESULTS_DIR}"/training; do
        [[ -d "${stage_dir}" ]] && rm -rf "${stage_dir}" \
            && echo "  Cleaned ${stage_dir}"
    done
fi

# ── compute exact backbone_context array size ─────────────────────────────────
N_BK=$(python "${RUNNER}" --stage backbone_context --frame-type "${FRAME_TYPE}" --list 2>/dev/null \
    | awk '/^Stage/ { for(i=1;i<=NF;i++) if($i=="configs") print $(i-1) }')

if [[ -z "${N_BK}" || "${N_BK}" -eq 0 ]]; then
    echo "[ERROR] No backbone_context configs available." \
         "Check that embeddings exist for at least one encoder." >&2
    exit 1
fi

MAX_BK=$(( N_BK - 1 ))

echo "========================================================================"
echo "PPCI ants hyperparameter search  (frame_type=${FRAME_TYPE})"
echo "  backbone_context configs : ${N_BK}"
if [[ "${FRAME_TYPE}" == "pov" ]]; then
    echo "  remaining stages         : arch → training (joint finetune+augmentation)"
else
    echo "  remaining stages         : arch → finetune → augmentation"
fi
echo "  each stage starts only after all jobs in the previous stage succeed"
echo "========================================================================"
[[ "${DRY}" == "--dry-run" ]] && echo "  *** DRY RUN — no jobs will be submitted ***"
echo ""

# ── submit dependency chain ───────────────────────────────────────────────────
#
#  Generous --array upper bounds: the Python script exits cleanly (NOOP)
#  when job-idx >= n_configs.
#
#  Stage              max_idx   reason
#  backbone_context   dynamic   exact count from --list
#  arch               39        full: top-2 backbone × 19 = 38 configs
#  finetune           19        exactly 18 configs (2×3×3) — full only
#  augmentation       9         exactly 9 configs  (3×3)   — full only
#  training           79        72 configs (2×2×2×3×3)     — pov only
#
JOB1=$(submit "backbone_context" "${MAX_BK}" "")
JOB2=$(submit "arch"             39          "${JOB1}")

if [[ "${FRAME_TYPE}" == "pov" ]]; then
    JOB3=$(submit "training"     79          "${JOB2}")
else
    JOB3=$(submit "finetune"         19          "${JOB2}")
    JOB4=$(submit "augmentation"     9           "${JOB3}")
fi

echo ""
echo "All stages queued. Monitor:"
echo "  squeue -u \$USER"
echo ""
echo "Results (available after each stage):"
echo "  python ${RUNNER} --print-results --frame-type ${FRAME_TYPE}"
echo "  python ${RUNNER} --print-results --frame-type ${FRAME_TYPE} --stage backbone_context"
