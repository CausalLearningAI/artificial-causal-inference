#!/bin/bash
# =============================================================================
# Summary of tracking pipeline results for the most recent submission.
#
# Usage:
#   bash scripts/03_tracking/report_summary.sh              # default: v1 v2 v3 v4 v5
#   bash scripts/03_tracking/report_summary.sh v2 v3 v4
# =============================================================================
set -euo pipefail

SUBJECT=${SUBJECT:-ants}
if [ $# -eq 0 ]; then
    VERSIONS=(v1 v2 v3 v4 v5)
else
    VERSIONS=("$@")
fi

BASE="$(cd "$(dirname "$0")/../.." && pwd)"

# ── Find logs ────────────────────────────────────────────────────────────────
# BOUNDS_LOG is used for bounds metrics/progress parsing.
BOUNDS_LOG=$(ls -t "${BASE}"/logs/bounds_${SUBJECT}*.out "${BASE}"/logs/bounds_demos*.out 2>/dev/null | head -1 || true)

# STATUS_LOG is used for the "Last submission" header/job status and should
# reflect the most recent relevant run (tracking or bounds).
STATUS_LOG=$(ls -t \
    "${BASE}"/logs/track_${SUBJECT}_v*_*.out \
    "${BASE}"/logs/bounds_${SUBJECT}*.out \
    "${BASE}"/logs/bounds_demos*.out \
    2>/dev/null | head -1 || true)

IS_RUNNING=false
JOB_STATE=""
LAST_JOB_ID=""
RUN_MODE="unknown"   # tracking | bounds | unknown

if [ -n "${STATUS_LOG:-}" ] && [ -f "${STATUS_LOG}" ]; then
    LAST_JOB_ID=$(basename "$STATUS_LOG" .out | grep -oP '[0-9]+$' || true)
    SUBMIT_TIME=$(head -5 "$STATUS_LOG" | grep -oP '\w+ \w+ \d+ .* \d{4}' | head -1 || true)

    if [ -n "${LAST_JOB_ID:-}" ]; then
        JOB_STATE=$(sacct -j "$LAST_JOB_ID" --format="State" --noheader 2>/dev/null | head -1 | xargs || true)
        [[ "$JOB_STATE" == "RUNNING" ]] && IS_RUNNING=true
    fi

    LOG_KIND="submission"
    STATUS_BASENAME=$(basename "$STATUS_LOG")
    if [[ "$STATUS_BASENAME" == track_* ]]; then
        LOG_KIND="tracking"
        RUN_MODE="tracking"
    elif [[ "$STATUS_BASENAME" == bounds_* || "$STATUS_BASENAME" == bounds_demos* ]]; then
        LOG_KIND="bounds"
        RUN_MODE="bounds"
    fi

    echo "================================================================"
    echo "  Last submission (${LOG_KIND}): ${STATUS_BASENAME}  [${JOB_STATE:-UNKNOWN}]"
    [ -n "${SUBMIT_TIME:-}" ] && echo "  Started: ${SUBMIT_TIME}"
    echo "================================================================"
else
    echo "================================================================"
    echo "  No bounds/pipeline logs found"
    echo "================================================================"
fi

# ── Detect which steps the running job has completed ─────────────────────────
hms_to_sec() { echo "$1" | awk -F: '{print $1*3600 + $2*60 + $3}'; }

BOUNDS_RECOMPUTED=false   # bounds results in log are fresh
BOUNDS_RECOMPUTING=false  # bounds optimization is in progress
declare -A DEMO_DONE=()   # DEMO_DONE[v3]=1 means demo for v3 is done
declare -A TRACK_RECOMPUTING=()
declare -A TRACK_JOB_ID=()
declare -A TRACK_STATE=()
declare -A TRACK_PROGRESS=()
declare -A TRACK_PHASE=()  # tracking | demo | unknown

COMPLETED_STEPS=()
if [ -n "${BOUNDS_LOG:-}" ] && [ -f "${BOUNDS_LOG}" ]; then
    while IFS= read -r line; do
        STEP_NAME=$(echo "$line" | grep -oP '^\[.*(?= done in)' | sed 's/^\[//' || true)
        STEP_TIME=$(echo "$line" | grep -oP '\d{2}:\d{2}:\d{2}' || true)
        if [ -n "$STEP_NAME" ] && [ -n "$STEP_TIME" ]; then
            STEP_SECS=$(hms_to_sec "$STEP_TIME")
            COMPLETED_STEPS+=("${STEP_NAME}|${STEP_TIME}|${STEP_SECS}")
            [[ "$STEP_NAME" == "Bounds" ]] && BOUNDS_RECOMPUTED=true
            [[ "$STEP_NAME" == v* ]] && DEMO_DONE["$STEP_NAME"]=1
        fi
    done < <(grep 'done in' "$BOUNDS_LOG" 2>/dev/null || true)

    # If a bounds job is running and bounds not yet done, they're being recomputed
    if [ "$RUN_MODE" = "bounds" ] && [ "$IS_RUNNING" = true ] && [ "$BOUNDS_RECOMPUTED" = false ]; then
        BOUNDS_RECOMPUTING=true
    fi

    # For full pipeline jobs, detect which track jobs are pending/running
    LOG_NAME=$(basename "$BOUNDS_LOG")
    if [ "$RUN_MODE" = "bounds" ] && [[ "$LOG_NAME" == bounds_ants* ]] && [ -n "${LAST_JOB_ID:-}" ]; then
        JOB_LO=$((LAST_JOB_ID))
        JOB_HI=$((LAST_JOB_ID + 20))
        while IFS= read -r line; do
            JNAME=$(echo "$line" | awk '{print $1}')
            JSTATE=$(echo "$line" | awk '{print $2}')
            if [[ "$JNAME" == track_* ]] && [[ "$JSTATE" == "RUNNING" || "$JSTATE" == "PENDING" ]]; then
                VER=$(echo "$JNAME" | grep -oP 'v\d+$' || true)
                [ -n "$VER" ] && TRACK_RECOMPUTING["$VER"]=1
            fi
        done < <(sacct -u "$USER" --starttime=2026-01-01 \
            --format="JobName%30,State%15" --noheader 2>/dev/null \
            | awk -v lo="$JOB_LO" -v hi="$JOB_HI" 'NR>=lo-1 && NR<=hi' || true)
    fi
fi

# Detect active tracking jobs directly (works for tracking-only submissions too)
while IFS= read -r line; do
    [ -z "$line" ] && continue
    JID=$(echo "$line" | awk '{print $1}')
    JNAME=$(echo "$line" | awk '{print $2}')
    JSTATE=$(echo "$line" | awk '{print $3}')

    if [[ "$JNAME" =~ ^track_${SUBJECT}_(v[0-9]+)$ ]]; then
        VER="${BASH_REMATCH[1]}"
        TRACK_RECOMPUTING["$VER"]=1
        TRACK_JOB_ID["$VER"]="$JID"
        TRACK_STATE["$VER"]="$JSTATE"

        LOG_PATH="${BASE}/logs/${JNAME}_${JID}.out"
        if [ -f "$LOG_PATH" ]; then
            PCT=$(grep -oE '[0-9]+\.?[0-9]*%' "$LOG_PATH" | tail -1 | tr -d '%' || true)
            if [ -n "${PCT:-}" ]; then
                TRACK_PROGRESS["$VER"]="$PCT"
            fi

            # Detect which phase is active in job_track.sh
            if grep -q "\[Tracking done in" "$LOG_PATH" 2>/dev/null || grep -q ">>> Step 2/2" "$LOG_PATH" 2>/dev/null; then
                TRACK_PHASE["$VER"]="demo"
            elif grep -q ">>> Step 1/2" "$LOG_PATH" 2>/dev/null; then
                TRACK_PHASE["$VER"]="tracking"
            else
                TRACK_PHASE["$VER"]="unknown"
            fi
        fi
    fi
done < <(squeue -h -u "$USER" -o "%i %j %T" 2>/dev/null || true)

# ── SLURM jobs from this submission ──────────────────────────────────────────
if [ -n "${LAST_JOB_ID:-}" ]; then
    echo ""
    JOB_LO=$((LAST_JOB_ID))
    JOB_HI=$((LAST_JOB_ID + 20))
    sacct -u "$USER" --starttime=2026-01-01 \
        --format="JobID%12,JobName%25,State%15,Elapsed%12,ExitCode%10" \
        2>/dev/null \
        | awk -v lo="$JOB_LO" -v hi="$JOB_HI" '
            /^---/ { print; next }
            /JobID/ { print; next }
            { split($1, a, "."); if (a[1]+0 >= lo && a[1]+0 <= hi && $1 !~ /\./) print }
        ' \
        || true
fi

# ── Per-version results ─────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  Tracking results — ${SUBJECT}"
echo "================================================================"

for VERSION in "${VERSIONS[@]}"; do
    echo ""
    echo "── ${SUBJECT}/${VERSION} ──────────────────────────────────"

    TRACK_DIR="${BASE}/dataset/${SUBJECT}/${VERSION}/tracking"
    VIZ_DIR="${BASE}/results/${SUBJECT}/${VERSION}/tracking_viz"
    POV_DIR="${BASE}/dataset/${SUBJECT}/${VERSION}/frames/pov"
    EMB_DIR="${BASE}/dataset/${SUBJECT}/${VERSION}/embeddings/pov"
    CFG="${BASE}/configs/tracking/${SUBJECT}/${VERSION}.yaml"

    # ── Bounds ────────────────────────────────────────────────────────────
    if [ "$BOUNDS_RECOMPUTING" = true ]; then
        echo "  bounds:  (recomputing...)"
    elif [ -f "$CFG" ]; then
        BLUE_LB=$(grep 'blue_marking_lb' "$CFG" | sed 's/.*: //')
        BLUE_UB=$(grep 'blue_marking_ub' "$CFG" | sed 's/.*: //')
        YEL_LB=$(grep 'yellow_marking_lb' "$CFG" | sed 's/.*: //')
        YEL_UB=$(grep 'yellow_marking_ub' "$CFG" | sed 's/.*: //')

        # Extract metrics from the bounds log
        BLUE_METRICS="" ; YEL_METRICS=""
        if [ -n "${BOUNDS_LOG:-}" ] && [ -f "${BOUNDS_LOG}" ]; then
            BLUE_SCORE=$(awk "/^  ${VERSION}:$/{found=1; next} found && /blue.*score=/{print; exit}" "$BOUNDS_LOG" \
                | grep -oP 'score=\K-?[0-9.]+' || true)
            BLUE_DETAIL=$(awk "/^  ${VERSION}:$/{found=1; next} found && /blue/{got=1; next} got && /prec=/{print; exit}" "$BOUNDS_LOG" \
                | grep -oP '(prec|no_det|over_det)=[0-9.]+' | tr '\n' '  ' || true)
            YEL_SCORE=$(awk "/^  ${VERSION}:$/{found=1; next} found && /yellow.*score=/{print; exit}" "$BOUNDS_LOG" \
                | grep -oP 'score=\K-?[0-9.]+' || true)
            YEL_DETAIL=$(awk "/^  ${VERSION}:$/{found=1; next} found && /yellow/{got=1; next} got && /prec=/{print; exit}" "$BOUNDS_LOG" \
                | grep -oP '(prec|no_det|over_det)=[0-9.]+' | tr '\n' '  ' || true)

            [ -n "$BLUE_SCORE" ] && BLUE_METRICS="  score=${BLUE_SCORE}"
            [ -n "$BLUE_DETAIL" ] && BLUE_METRICS="${BLUE_METRICS}  ${BLUE_DETAIL}"
            [ -n "$YEL_SCORE" ] && YEL_METRICS="  score=${YEL_SCORE}"
            [ -n "$YEL_DETAIL" ] && YEL_METRICS="${YEL_METRICS}  ${YEL_DETAIL}"
        fi

        echo "  bounds:  blue   ${BLUE_LB} → ${BLUE_UB}${BLUE_METRICS}"
        echo "           yellow ${YEL_LB} → ${YEL_UB}${YEL_METRICS}"
    else
        echo "  bounds:  (no config found)"
    fi

    # ── Tracking CSVs ─────────────────────────────────────────────────────
    # If this version's tracking is being recomputed, show that instead of stale data
    TRACK_STALE=false
    if [ "$RUN_MODE" = "bounds" ] && [ "$IS_RUNNING" = true ]; then
        # bounds_demos: tracking is recomputed for each version after bounds
        LOG_NAME=$(basename "$BOUNDS_LOG")
        if [[ "$LOG_NAME" == bounds_demos* ]]; then
            # Stale if bounds done but this version's demo not yet done
            if [ "$BOUNDS_RECOMPUTED" = true ] && [ -z "${DEMO_DONE[$VERSION]:-}" ]; then
                TRACK_STALE=true
            # Also stale if bounds not even done yet
            elif [ "$BOUNDS_RECOMPUTED" = false ]; then
                TRACK_STALE=true
            fi
        fi
    fi

    # Active per-version track job: hide metrics only while Step 1 tracking is ongoing.
    # During Step 2 (demo rendering), tracking CSVs are already finalized and can be shown.
    if [ -n "${TRACK_RECOMPUTING[$VERSION]:-}" ]; then
        PHASE="${TRACK_PHASE[$VERSION]:-unknown}"
        if [ "$PHASE" = "tracking" ] || [ "$PHASE" = "unknown" ]; then
            TRACK_STALE=true
        fi
    fi

    if [ "$TRACK_STALE" = true ]; then
        if [ -n "${TRACK_PROGRESS[$VERSION]:-}" ]; then
            echo "  tracking: (recomputing... ${TRACK_PROGRESS[$VERSION]}%  ${TRACK_STATE[$VERSION]:-RUNNING})"
        elif [ -n "${TRACK_STATE[$VERSION]:-}" ]; then
            echo "  tracking: (recomputing... ${TRACK_STATE[$VERSION]})"
        else
            echo "  tracking: (recomputing...)"
        fi
    elif [ -d "$TRACK_DIR" ]; then
        N_CSV=$(find "$TRACK_DIR" -name '*.csv' | wc -l)

        if [ "$N_CSV" -gt 0 ]; then
            METRICS=$(python "${BASE}/src/tracking/tracking_summary.py" "$SUBJECT" "$VERSION" 2>/dev/null || echo "ERROR")
            if [ "$METRICS" = "ERROR" ] || [ "$METRICS" = "NOT_FOUND" ]; then
                echo "  tracking: NOT FOUND"
            elif [ "$METRICS" = "EMPTY" ]; then
                echo "  tracking: (empty)"
            else
                echo "$METRICS" | awk 'NR==1 {printf "  tracking: %s\n", $0} NR>1 {printf "             %s\n", $0}'
            fi
        else
            echo "  tracking: (empty)"
        fi
    else
        echo "  tracking: NOT FOUND"
    fi

    # Demo videos
    if [ "$TRACK_STALE" = true ]; then
        echo "  demos:   (pending...)"
    elif [ -n "${TRACK_RECOMPUTING[$VERSION]:-}" ] && [ "${TRACK_PHASE[$VERSION]:-}" = "demo" ]; then
        if [ -d "$VIZ_DIR" ]; then
            N_VIZ=$(find "$VIZ_DIR" -name '*.mp4' | wc -l)
            echo "  demos:   ${N_VIZ} videos (generating...)"
        else
            echo "  demos:   (generating...)"
        fi
    elif [ -d "$VIZ_DIR" ]; then
        N_VIZ=$(find "$VIZ_DIR" -name '*.mp4' | wc -l)
        VIZ_SIZE=$(du -sh "$VIZ_DIR" 2>/dev/null | cut -f1)
        echo "  demos:   ${N_VIZ} videos (${VIZ_SIZE})"
    else
        echo "  demos:   NOT FOUND"
    fi

    # POV frames
    if [ -d "$POV_DIR" ]; then
        N_OBS=$(find "$POV_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
        N_FRAMES=$(find "$POV_DIR" -name '*.jpg' -o -name '*.png' | wc -l)
        POV_SIZE=$(du -sh "$POV_DIR" 2>/dev/null | cut -f1)
        echo "  pov:     ${N_OBS} obs, ${N_FRAMES} frames (${POV_SIZE})"
    else
        echo "  pov:     NOT FOUND"
    fi

    # POV embeddings
    if [ -d "$EMB_DIR" ]; then
        for ENC_DIR in "$EMB_DIR"/*/; do
            [ -d "$ENC_DIR" ] || continue
            ENC_NAME=$(basename "$ENC_DIR")
            for TOK_DIR in "$ENC_DIR"/*/; do
                [ -d "$TOK_DIR" ] || continue
                TOK_NAME=$(basename "$TOK_DIR")
                N_NPY=$(find "$TOK_DIR" -name '*.npy' | wc -l)
                EMB_SIZE=$(du -sh "$TOK_DIR" 2>/dev/null | cut -f1)
                echo "  embeds:  ${ENC_NAME}/${TOK_NAME}: ${N_NPY} files (${EMB_SIZE})"
            done
        done
    else
        echo "  embeds:  NOT FOUND"
    fi
done

# ── Progress & ETA ───────────────────────────────────────────────────────────
if { [ "$RUN_MODE" = "bounds" ] && [ -n "${BOUNDS_LOG:-}" ] && [ -f "${BOUNDS_LOG}" ]; } ||
    { [ "$RUN_MODE" = "tracking" ] && [ -n "${STATUS_LOG:-}" ] && [ -f "${STATUS_LOG}" ]; }; then
    echo ""
    echo "================================================================"
    echo "  Progress"
    echo "================================================================"
    echo ""

    PROGRESS_LOG="$BOUNDS_LOG"
    if [ "$RUN_MODE" = "tracking" ]; then
        PROGRESS_LOG="$STATUS_LOG"
    fi
    LOG_NAME=$(basename "$PROGRESS_LOG")

    if [ "$RUN_MODE" = "bounds" ]; then
        for step in "${COMPLETED_STEPS[@]}"; do
            NAME="${step%%|*}"
            TIME="${step#*|}" ; TIME="${TIME%%|*}"
            echo "  [done] ${NAME}  (${TIME})"
        done
    else
        for VERSION in "${VERSIONS[@]}"; do
            VLOG=$(ls -t "${BASE}"/logs/track_${SUBJECT}_${VERSION}_*.out 2>/dev/null | head -1 || true)
            if [ -n "${VLOG:-}" ] && grep -q "Results for ${SUBJECT}/${VERSION}" "$VLOG" 2>/dev/null; then
                echo "  [done] ${SUBJECT}/${VERSION}"
            fi
        done
    fi

    # Detect current step from log
    if [ "$IS_RUNNING" = true ]; then
        CURRENT_STEP=$(grep -E '>>> |^── ' "$PROGRESS_LOG" | tail -1 | sed 's/^>>> //' | sed 's/^── //' | sed 's/ ──.*$//' | sed 's/\.\.\.$//' || true)
        [ -n "$CURRENT_STEP" ] && echo "  [running] ${CURRENT_STEP}..."
    fi

    # ── Per-version bounds progress (from [progress] markers) ─────────
    # Parse structured progress lines: [progress] <version> <color> <phase> <cur>/<total>
    # Phases per version: loading → blue proxy → blue verify → blue done → yellow proxy → yellow verify → yellow done
    # Total weight: loading=5% + blue_proxy=10% + blue_verify=30% + yellow_proxy=10% + yellow_verify=30% + overhead=15%
    if [ "$RUN_MODE" = "bounds" ] && [ "$IS_RUNNING" = true ] && [ "$BOUNDS_RECOMPUTED" = false ]; then
        declare -A VER_PCT=()
        for VERSION in "${VERSIONS[@]}"; do
            pct=0

            # Check if this version is fully done (config was saved)
            if grep -q "saved.*${VERSION}.yaml" "$BOUNDS_LOG" 2>/dev/null; then
                pct=100
            else
                # Loading phase (5%)
                LOAD_LINE=$(grep "\[progress\] ${VERSION} loading" "$BOUNDS_LOG" | tail -1 || true)
                if [ -n "$LOAD_LINE" ]; then
                    LOAD_CUR=$(echo "$LOAD_LINE" | grep -oP 'loading \K\d+')
                    LOAD_TOT=$(echo "$LOAD_LINE" | grep -oP '/\K\d+')
                    if [ -n "$LOAD_CUR" ] && [ -n "$LOAD_TOT" ] && [ "$LOAD_TOT" -gt 0 ]; then
                        pct=$(( 5 * LOAD_CUR / LOAD_TOT ))
                    fi
                fi

                for color in blue yellow; do
                    if [ "$color" = "blue" ]; then
                        BASE_PCT=5; PROXY_W=10; VERIFY_W=30
                    else
                        BASE_PCT=55; PROXY_W=10; VERIFY_W=30
                    fi

                    # Check if color is done
                    if grep -q "\[progress\] ${VERSION} ${color} done" "$BOUNDS_LOG" 2>/dev/null; then
                        pct=$((BASE_PCT + PROXY_W + VERIFY_W))
                        continue
                    fi

                    # Verify phase (heavy)
                    VERIFY_LINE=$(grep "\[progress\] ${VERSION} ${color} verify" "$BOUNDS_LOG" | tail -1 || true)
                    if [ -n "$VERIFY_LINE" ]; then
                        V_CUR=$(echo "$VERIFY_LINE" | grep -oP 'verify \K\d+')
                        V_TOT=$(echo "$VERIFY_LINE" | grep -oP '/\K\d+')
                        if [ -n "$V_CUR" ] && [ -n "$V_TOT" ] && [ "$V_TOT" -gt 0 ]; then
                            pct=$((BASE_PCT + PROXY_W + VERIFY_W * V_CUR / V_TOT))
                        fi
                        continue
                    fi

                    # Proxy phase
                    PROXY_LINE=$(grep "\[progress\] ${VERSION} ${color} proxy" "$BOUNDS_LOG" | tail -1 || true)
                    if [ -n "$PROXY_LINE" ]; then
                        P_CUR=$(echo "$PROXY_LINE" | grep -oP 'proxy \K\d+')
                        P_TOT=$(echo "$PROXY_LINE" | grep -oP '/\K\d+')
                        if [ -n "$P_CUR" ] && [ -n "$P_TOT" ] && [ "$P_TOT" -gt 0 ]; then
                            pct=$((BASE_PCT + PROXY_W * P_CUR / P_TOT))
                        fi
                    fi
                done
            fi

            VER_PCT["$VERSION"]=$pct
        done

        # Show per-version progress bars
        echo ""
        echo "  Bounds optimization:"
        TOTAL_PCT=0
        for VERSION in "${VERSIONS[@]}"; do
            P=${VER_PCT[$VERSION]:-0}
            TOTAL_PCT=$((TOTAL_PCT + P))
            # Build a 20-char progress bar
            FILLED=$((P * 20 / 100))
            BAR=""
            for ((i=0; i<FILLED; i++)); do BAR="${BAR}#"; done
            for ((i=FILLED; i<20; i++)); do BAR="${BAR}-"; done
            printf "    %-4s [%s] %3d%%\n" "$VERSION" "$BAR" "$P"
        done

        # Overall percentage and ETA
        N_VERSIONS=${#VERSIONS[@]}
        OVERALL_PCT=$((TOTAL_PCT / N_VERSIONS))

        JOB_ELAPSED=$(sacct -j "$LAST_JOB_ID" --format="Elapsed" --noheader 2>/dev/null | head -1 | xargs || true)
        if [ -n "$JOB_ELAPSED" ] && [ "$OVERALL_PCT" -gt 2 ]; then
            ELAPSED_SEC=$(hms_to_sec "$JOB_ELAPSED")
            # ETA = elapsed * (remaining_pct / done_pct)
            REMAINING_SEC=$(( ELAPSED_SEC * (100 - OVERALL_PCT) / OVERALL_PCT ))
            REM_H=$((REMAINING_SEC / 3600))
            REM_M=$(((REMAINING_SEC % 3600) / 60))
            echo ""
            if [ $REM_H -gt 0 ]; then
                printf "  Bounds: %d%% — ETA ~%dh %dm\n" "$OVERALL_PCT" "$REM_H" "$REM_M"
            else
                printf "  Bounds: %d%% — ETA ~%dm\n" "$OVERALL_PCT" "$REM_M"
            fi
        elif [ "$OVERALL_PCT" -gt 0 ]; then
            echo ""
            printf "  Bounds: %d%% (estimating ETA...)\n" "$OVERALL_PCT"
        else
            echo ""
            echo "  Bounds: starting..."
        fi
    fi

    # ── Demo ETA (after bounds done) ──────────────────────────────────
    if [ "$RUN_MODE" = "bounds" ] && [ "$IS_RUNNING" = true ] && [ "$BOUNDS_RECOMPUTED" = true ]; then
        N_VERSIONS=${#VERSIONS[@]}
        N_DEMOS_DONE=0
        for step in "${COMPLETED_STEPS[@]}"; do
            NAME="${step%%|*}"
            [[ "$NAME" == v* ]] && N_DEMOS_DONE=$((N_DEMOS_DONE + 1))
        done
        N_DEMOS_LEFT=$((N_VERSIONS - N_DEMOS_DONE))
        if [ $N_DEMOS_LEFT -gt 0 ] && [ $N_DEMOS_DONE -gt 0 ]; then
            TOTAL_DEMO_SEC=0
            for step in "${COMPLETED_STEPS[@]}"; do
                NAME="${step%%|*}"
                SECS="${step##*|}"
                [[ "$NAME" == v* ]] && TOTAL_DEMO_SEC=$((TOTAL_DEMO_SEC + SECS))
            done
            AVG_DEMO=$((TOTAL_DEMO_SEC / N_DEMOS_DONE))
            REMAINING_SEC=$((N_DEMOS_LEFT * AVG_DEMO))
            REM_M=$((REMAINING_SEC / 60))
            echo ""
            echo "  Demos: ${N_DEMOS_DONE}/${N_VERSIONS} done — ETA ~${REM_M}m"
        elif [ $N_DEMOS_LEFT -gt 0 ]; then
            echo ""
            echo "  Demos: 0/${N_VERSIONS} done"
        fi
    fi

    # ── Full pipeline ETA (bounds_ants* jobs) ─────────────────────────
    if [ "$RUN_MODE" = "bounds" ] && [ "$IS_RUNNING" = true ] && [[ "$LOG_NAME" == bounds_ants* ]]; then
        N_TRACK_PENDING=$(sacct -u "$USER" --starttime=2026-01-01 --format="JobName,State" --noheader 2>/dev/null \
            | grep -c "track.*PENDING\|track.*RUNNING" || true)
        N_POV_PENDING=$(sacct -u "$USER" --starttime=2026-01-01 --format="JobName,State" --noheader 2>/dev/null \
            | grep -c "pov.*PENDING\|pov.*RUNNING" || true)
        [ "$N_TRACK_PENDING" -gt 0 ] && echo "  Track jobs pending: ${N_TRACK_PENDING}"
        [ "$N_POV_PENDING" -gt 0 ] && echo "  POV jobs pending: ${N_POV_PENDING}"
    fi

    if [ "$IS_RUNNING" = false ] && [ -n "$JOB_STATE" ]; then
        echo ""
        echo "  Job status: ${JOB_STATE}"
    fi

    # Log tail
    echo ""
    echo "── log tail ──"
    tail -8 "$PROGRESS_LOG"
fi

echo ""
echo "================================================================"
echo "  Done."
echo "================================================================"
