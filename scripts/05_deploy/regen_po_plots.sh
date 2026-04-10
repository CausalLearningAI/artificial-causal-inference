#!/usr/bin/env bash
# Regenerate plot_summary_*.png (APO bar plots) with 95% CI error bars
# for both full and pov best models, validate and final (deploy) configs.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

RESULTS="results/ppci/ants/hparam"

for FRAME_TYPE in full pov; do
    for CONFIG in validate final; do
        MODEL_PT="$RESULTS/$FRAME_TYPE/deploy/$CONFIG/model.pt"
        if [[ ! -f "$MODEL_PT" ]]; then
            echo "[SKIP] $FRAME_TYPE/$CONFIG — model.pt not found"
            continue
        fi
        echo "==> Regenerating plots: $FRAME_TYPE/$CONFIG"
        python src/ppci/deploy_model.py \
            --results-dir "$RESULTS/$FRAME_TYPE" \
            --config "$CONFIG" \
            --eval-only
        echo "    Done."
    done
done

echo ""
echo "All done. Plots saved under results/ppci/ants/hparam/{full,pov}/deploy/{validate,final}/"
