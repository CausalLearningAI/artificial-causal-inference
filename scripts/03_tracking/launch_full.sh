#!/bin/bash
# =============================================================================
# Canonical Stage-3 launcher (4-step pipeline):
#   0) bounds
#   1) tracking (+demo)
#   2) pov crops
#   3) embeddings
#
# Defaults are resume-friendly and match your current preferred setup:
# - versions: v2 v3 v4 v5
# - encoders: dinov2 dinov3
# - token: class
# - pov identities for embedding dataset: blue yellow
# - overwrite flags: false for steps 1-3
# - bounds: auto (skip if bounds already in all configs)
#
# Usage:
#   bash scripts/03_tracking/launch_full.sh
#   bash scripts/03_tracking/launch_full.sh v2 v3 v4 v5
#
# Toggle steps:
#   RUN_BOUNDS=true|false|auto   (default: auto)
#   RUN_TRACK=true|false         (default: true)
#   RUN_POV=true|false           (default: true)
#   RUN_EMBED=true|false         (default: true)
#
# Per-step overwrite flags:
#   OVERWRITE_BOUNDS=true|false    (used only when RUN_BOUNDS=true)
#   OVERWRITE_TRACKING=true|false
#   OVERWRITE_POV=true|false
#   OVERWRITE_EMBEDDINGS=true|false
#
# Encoder selection:
#   ENCODERS="dinov2 dinov3" TOKEN=class POV_IDENTITIES="blue yellow"
#
# Embedding cluster overrides (optional):
#   EMBED_PARTITION=gpu EMBED_GRES=gpu:1 EMBED_CPUS=8 EMBED_MEM=48G EMBED_TIME=24:00:00
# =============================================================================
set -euo pipefail

if [ $# -eq 0 ]; then
    VERSIONS=(v2 v3 v4 v5)
else
    VERSIONS=("$@")
fi

SUBJECT=${SUBJECT:-ants}
ENCODERS=${ENCODERS:-"dinov2 dinov3"}
TOKEN=${TOKEN:-class}
POV_IDENTITIES=${POV_IDENTITIES:-"blue yellow"}
RUN_BOUNDS=${RUN_BOUNDS:-false}
RUN_TRACK=${RUN_TRACK:-false}
RUN_POV=${RUN_POV:-false}
RUN_EMBED=${RUN_EMBED:-true}

OVERWRITE_BOUNDS=${OVERWRITE_BOUNDS:-false}
OVERWRITE_TRACKING=${OVERWRITE_TRACKING:-false}
OVERWRITE_POV=${OVERWRITE_POV:-false}
OVERWRITE_EMBEDDINGS=${OVERWRITE_EMBEDDINGS:-true}

EMBED_PARTITION=${EMBED_PARTITION:-gpu}
EMBED_GRES=${EMBED_GRES:-gpu:1}
EMBED_CPUS=${EMBED_CPUS:-8}
EMBED_MEM=${EMBED_MEM:-48G}
EMBED_TIME=${EMBED_TIME:-24:00:00}

BOUNDS_SCRIPT="scripts/03_tracking/job_bounds.sh"
TRACK_SCRIPT="scripts/03_tracking/job_track.sh"
POV_SCRIPT="scripts/03_tracking/job_pov.sh"
EMBED_SCRIPT="scripts/03_tracking/job_embed.sh"

mkdir -p logs

echo "================================================================"
echo " Stage-3 pipeline — ${SUBJECT}: ${VERSIONS[*]}"
echo " run: bounds=${RUN_BOUNDS} track=${RUN_TRACK} pov=${RUN_POV} embed=${RUN_EMBED}"
echo " overwrite: bounds=${OVERWRITE_BOUNDS} track=${OVERWRITE_TRACKING} pov=${OVERWRITE_POV} embed=${OVERWRITE_EMBEDDINGS}"
echo " encoders=${ENCODERS} token=${TOKEN} pov_identities=${POV_IDENTITIES}"
echo " embed resources: partition=${EMBED_PARTITION} gres=${EMBED_GRES} cpus=${EMBED_CPUS} mem=${EMBED_MEM} time=${EMBED_TIME}"
echo "================================================================"
echo ""

have_bounds_cfg() {
    local v="$1"
    local cfg="configs/tracking/${SUBJECT}/${v}.yaml"
    [ -f "$cfg" ] || return 1
    grep -q '^blue_marking_lb:' "$cfg" || return 1
    grep -q '^blue_marking_ub:' "$cfg" || return 1
    grep -q '^yellow_marking_lb:' "$cfg" || return 1
    grep -q '^yellow_marking_ub:' "$cfg" || return 1
    return 0
}

BOUNDS_ID=""
submit_bounds=false
if [ "$RUN_BOUNDS" = "true" ]; then
    submit_bounds=true
elif [ "$RUN_BOUNDS" = "false" ]; then
    submit_bounds=false
elif [ "$RUN_BOUNDS" = "auto" ]; then
    if [ "$OVERWRITE_BOUNDS" = "true" ]; then
        submit_bounds=true
    else
        all_ok=true
        for V in "${VERSIONS[@]}"; do
            if ! have_bounds_cfg "$V"; then
                all_ok=false
                break
            fi
        done
        if [ "$all_ok" = false ]; then
            submit_bounds=true
        fi
    fi
fi

if [ "$submit_bounds" = true ]; then
    BOUNDS_ID=$(
        VERSIONS="${VERSIONS[*]}" \
        sbatch \
            --job-name="bounds_${SUBJECT}" \
            --output="logs/bounds_${SUBJECT}_%j.out" \
            --error="logs/bounds_${SUBJECT}_%j.err" \
            "${BOUNDS_SCRIPT}" \
        | awk '{print $NF}'
    )
    echo "  step0 bounds  -> ${BOUNDS_ID}"
else
    echo "  step0 bounds  -> skipped (using config bounds)"
fi
echo ""

TRACK_IDS=()
POV_IDS=()
declare -A EMB_IDS

for VERSION in "${VERSIONS[@]}"; do
    dep=""
    if [ -n "$BOUNDS_ID" ]; then
        dep="--dependency=afterok:${BOUNDS_ID}"
    fi

    TRACK_ID=""
    if [ "$RUN_TRACK" = "true" ]; then
        TRACK_NAME="track_${SUBJECT}_${VERSION}"
        TRACK_ID=$(
            SUBJECT="${SUBJECT}" VERSION="${VERSION}" OVERWRITE_TRACKING="${OVERWRITE_TRACKING}" \
            sbatch \
                --job-name="${TRACK_NAME}" \
                --output="logs/${TRACK_NAME}_%j.out" \
                --error="logs/${TRACK_NAME}_%j.err" \
                ${dep} \
                "${TRACK_SCRIPT}" \
            | awk '{print $NF}'
        )
        TRACK_IDS+=("${TRACK_ID}")
    fi

    POV_ID=""
    if [ "$RUN_POV" = "true" ]; then
        POV_NAME="pov_${SUBJECT}_${VERSION}"
        dep_pov="$dep"
        if [ -n "$TRACK_ID" ]; then
            dep_pov="--dependency=afterok:${TRACK_ID}"
        fi
        POV_ID=$(
            SUBJECT="${SUBJECT}" VERSION="${VERSION}" OVERWRITE_POV="${OVERWRITE_POV}" \
            sbatch \
                --job-name="${POV_NAME}" \
                --output="logs/${POV_NAME}_%j.out" \
                --error="logs/${POV_NAME}_%j.err" \
                ${dep_pov} \
                "${POV_SCRIPT}" \
            | awk '{print $NF}'
        )
        POV_IDS+=("${POV_ID}")
    fi

    echo "  ${SUBJECT}/${VERSION}"
    [ -n "$TRACK_ID" ] && echo "    step1 track    -> ${TRACK_ID}"
    [ -n "$POV_ID" ] && echo "    step2 pov      -> ${POV_ID}"

    if [ "$RUN_EMBED" = "true" ]; then
        dep_emb="$dep"
        if [ -n "$POV_ID" ]; then
            dep_emb="--dependency=afterok:${POV_ID}"
        elif [ -n "$TRACK_ID" ] && [ "$RUN_POV" = "false" ]; then
            dep_emb="--dependency=afterok:${TRACK_ID}"
        fi

        for ENCODER in ${ENCODERS}; do
            for POV_IDENTITY in ${POV_IDENTITIES}; do
                EMB_NAME="emb_${SUBJECT}_${VERSION}_${POV_IDENTITY}_${ENCODER}"
                EMB_ID=$(
                    SUBJECT="${SUBJECT}" VERSION="${VERSION}" \
                    ENCODER="${ENCODER}" TOKEN="${TOKEN}" POV_IDENTITY="${POV_IDENTITY}" OVERWRITE_EMBEDDINGS="${OVERWRITE_EMBEDDINGS}" \
                    sbatch \
                        --job-name="${EMB_NAME}" \
                        --output="logs/${EMB_NAME}_%j.out" \
                        --error="logs/${EMB_NAME}_%j.err" \
                        --partition="${EMBED_PARTITION}" \
                        --gres="${EMBED_GRES}" \
                        --cpus-per-task="${EMBED_CPUS}" \
                        --mem="${EMBED_MEM}" \
                        --time="${EMBED_TIME}" \
                        ${dep_emb} \
                        "${EMBED_SCRIPT}" \
                    | awk '{print $NF}'
                )
                EMB_IDS["${VERSION}_${POV_IDENTITY}_${ENCODER}"]="$EMB_ID"
                echo "    step3 embed ${POV_IDENTITY}/${ENCODER} -> ${EMB_ID}"
            done
        done
    fi
    echo ""
done

echo "All requested jobs submitted."
echo ""
echo "Monitor:"
echo "  watch -n 30 squeue -u \$USER"
echo ""
echo "Logs:"
if [ -n "$BOUNDS_ID" ]; then
    echo "  tail -f logs/bounds_${SUBJECT}_${BOUNDS_ID}.out"
fi
for V in "${VERSIONS[@]}"; do
    echo "  tail -f logs/track_${SUBJECT}_${V}_<jobid>.out"
    echo "  tail -f logs/pov_${SUBJECT}_${V}_<jobid>.out"
    for POV_IDENTITY in ${POV_IDENTITIES}; do
        for ENCODER in ${ENCODERS}; do
            echo "  tail -f logs/emb_${SUBJECT}_${V}_${POV_IDENTITY}_${ENCODER}_<jobid>.out"
        done
    done
done
