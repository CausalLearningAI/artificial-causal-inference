#!/bin/bash
#
# The ssl_dapt arm of the head-vs-encoder ablation, both stages, chained.
# ==============================================================================
# ablate_head_vs_encoder.sh names this arm and deliberately does not launch it:
#
#   (3) domain-adaptive SSL would do better than supervised adaptation
#         ssl_dapt   DELIBERATELY NOT LAUNCHED HERE. It is the expensive arm and it only
#                    addresses hypothesis (2); run it if and only if ft_bitfit shows that
#                    cheap recalibration is what mattered.
#
# THAT GATE IS THIS SCRIPT'S PRECONDITION AND IT IS CHECKED BELOW, not left to memory. The
# check is a guard, not a veto -- FORCE=1 runs the arm anyway, which is a legitimate choice
# if you would rather have the answer than the tidy decision tree. It is not a free choice:
# ~11 h of A100 on a single node that the rest of the ablation is already queueing for.
#
# THE TWO STAGES
#   A  ssl_dapt.sh          adapt the last 2 encoder blocks on 244,800 UNLABELLED frames
#                           spanning all 68 non-val pools (of which 288 observations, 1.73M
#                           frames, have no behaviour annotation at all). ~5 h + ~45 min of
#                           first-time JPEG cache build.
#   B  train_online_aug.sh  freeze that encoder and train the head with flags IDENTICAL to
#                           frozen_ctrl_s42. ~5 h.
#
# Stage B is submitted with --dependency=afterok so a collapsed or crashed Stage A does not
# silently hand a stock encoder to a 5-hour head training run. ssl_dapt.py also aborts itself
# on collapse, so afterok is the second of two independent guards rather than the only one.
#
# WHAT THE RESULT MEANS -- the comparison is against frozen_ctrl_s42, NOT against zero and NOT
# against the fine-tuned runs. All three of these share --unfreeze-blocks 0 at head-training
# time and differ only in where the frozen encoder came from:
#
#     frozen_ctrl_s42                      stock DINOv2
#     res448_k2_frozen_d4photo_sslinit     DINOv2 + unlabelled adaptation   <- this arm
#     res448_k2_ft2_d4photo                DINOv2 + supervised adaptation (encoder trained)
#
#     sslinit ~= frozen_ctrl   unlabelled adaptation buys nothing; if bitfit DID win, then what
#                              mattered was supervision, cheap as it was -- not domain shift
#     sslinit -> ft2           the gain never needed labels; recalibration was the whole story
#     sslinit > ft2            the 1.73M unannotated frames carry more than the 24 labelled pools
#     sslinit < frozen_ctrl    adaptation actively hurt; check drift before believing it
#
# Judge "~=" against frozen_ctrl_s42 vs frozen_ctrl_s1, which is what that seed replicate is
# for. And read Stage A's `drift` first: a null with drift ~0 is a failed experiment, not a
# result -- it means the encoder never moved and Stage B re-measured the frozen baseline.
#
# Usage:
#     bash scripts/mice_behavior/ablate_ssl_dapt.sh            # both stages, chained
#     SMOKE=1 bash scripts/mice_behavior/ablate_ssl_dapt.sh    # Stage A pipeline check only
#     DRY=1   bash scripts/mice_behavior/ablate_ssl_dapt.sh    # print, submit nothing
#     FORCE=1 bash scripts/mice_behavior/ablate_ssl_dapt.sh    # skip the bitfit gate
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference

SSL_TAG="${SSL_TAG:-ssl_dapt}"
STAGE_B_TAG="${STAGE_B_TAG:-res448_k2_frozen_d4photo_sslinit}"
FRAME_DIR=results/vision/mice/frame

# ---- the gate ---------------------------------------------------------------------------
# The bitfit arms are what license this run. Both must have LANDED (a config.json only appears
# when train_online_aug.py finishes), and at least one must have beaten the frozen control.
gate_ok=1
BITFIT_LO=$FRAME_DIR/res448_k2_bit2_d4photo_elr1e05
BITFIT_HI=$FRAME_DIR/res448_k2_bit2_d4photo_elr0.001
CTRL=$FRAME_DIR/res448_k2_frozen_d4photo_decay30_seed42
for d in "$BITFIT_LO" "$BITFIT_HI" "$CTRL"; do
    if [ ! -e "$d/config.json" ]; then
        echo "GATE: $(basename "$d") has not finished yet (no config.json)"; gate_ok=0
    fi
done
if [ "$gate_ok" = 1 ]; then
    read -r verdict <<<"$(python - "$BITFIT_LO" "$BITFIT_HI" "$CTRL" <<'PY'
import json, sys
def ap(d):
    return json.load(open(f'{d}/config.json'))['ap_report']['macro/tol0']['ap']
lo, hi, ctrl = (ap(p) for p in sys.argv[1:4])
best = max(lo, hi)
print(f'{"WIN" if best > ctrl else "FLAT"} bitfit_lo={lo:.4f} bitfit_hi={hi:.4f} ctrl={ctrl:.4f}')
PY
)"
    echo "GATE: $verdict"
    case "$verdict" in
        FLAT*) echo "GATE: neither bitfit arm beat the frozen control, so hypothesis (2)"
               echo "      (recalibration) is not supported and this arm addresses only that"
               echo "      hypothesis. Re-run with FORCE=1 to test it anyway."; gate_ok=0 ;;
    esac
fi
if [ "$gate_ok" != 1 ] && [ -z "${FORCE:-}" ] && [ -z "${SMOKE:-}" ]; then
    echo; echo "not submitting. FORCE=1 overrides."; exit 1
fi

# ---- already done? ---------------------------------------------------------------------
if [ -e "$FRAME_DIR/$STAGE_B_TAG/config.json" ]; then
    echo "SKIP: $STAGE_B_TAG already has results"; exit 0
fi
for j in ssl_dapt "$STAGE_B_TAG"; do
    if squeue -u "$USER" -h -o '%j' 2>/dev/null | grep -qx "$j"; then
        echo "SKIP: job '$j' is already queued or running"; exit 0
    fi
done

SB_A=(--partition="${PARTITION:-gpu}" --gres="${GRES:-gpu:A100:1}" --time="${TIME_A:-10:00:00}"
      --mem="${MEM_A:-120G}" --cpus-per-task=32)

if [ -n "${SMOKE:-}" ]; then
    echo "[smoke] Stage A only, tiny corpus"
    [ -n "${DRY:-}" ] && { echo "[dry] would submit smoke Stage A"; exit 0; }
    SMOKE=1 sbatch "${SB_A[@]}" --job-name=ssl_smoke \
        --output=logs/ssl_smoke_%j.out --error=logs/ssl_smoke_%j.err \
        scripts/mice_behavior/ssl_dapt.sh
    exit 0
fi

if [ -n "${DRY:-}" ]; then
    echo "[dry] Stage A: TAG=$SSL_TAG ssl_dapt.sh"
    echo "[dry] Stage B: TAG=$STAGE_B_TAG INIT_ENCODER=$FRAME_DIR/$SSL_TAG/best_encoder.pt"
    exit 0
fi

jid_a=$(TAG="$SSL_TAG" WANDB=1 sbatch "${SB_A[@]}" --job-name=ssl_dapt \
            --output=logs/ssl_dapt_%j.out --error=logs/ssl_dapt_%j.err \
            --parsable scripts/mice_behavior/ssl_dapt.sh)
echo "submitted $jid_a  Stage A (ssl_dapt) -> $FRAME_DIR/$SSL_TAG/"

# Stage B: every flag below is copied from the frozen_ctrl_s42 arm of ablate_head_vs_encoder.sh.
# UNFREEZE_BLOCKS=0 is the point of the arm -- the encoder is frozen during head training, so
# the only difference from frozen_ctrl_s42 is which weights it was frozen AT.
jid_b=$(INPUT_SIZE=448 CONTEXT_K=2 STRIDE=1 AUGMENT=d4_photo PHOTO_STRENGTH=1.0 \
        OPTIMIZER=adamw LR=3e-4 WEIGHT_DECAY=0.05 DROPOUT=0.4 \
        WARMUP_EPOCHS=3 LR_DECAY_EPOCHS=30 N_EPOCHS=30 PATIENCE=10 \
        BATCH_SIZE=64 NEG_RATIO=1 MAX_TRAIN_FRAMES=300000 \
        JPEG_CACHE_FILE=dataset/mice/v1/jpegcache_k2 WANDB=1 \
        UNFREEZE_BLOCKS=0 SEED=42 \
        INIT_ENCODER="$FRAME_DIR/$SSL_TAG/best_encoder.pt" TAG="$STAGE_B_TAG" \
        sbatch --partition="${PARTITION:-gpu}" --gres="${GRES:-gpu:A100:1}" \
               --time="${TIME_B:-14:00:00}" --mem="${MEM_B:-180G}" --cpus-per-task=32 \
               --dependency=afterok:"$jid_a" --job-name="$STAGE_B_TAG" \
               --output="logs/${STAGE_B_TAG}_%j.out" --error="logs/${STAGE_B_TAG}_%j.err" \
               --parsable scripts/mice_behavior/train_online_aug.sh)
echo "submitted $jid_b  Stage B ($STAGE_B_TAG) -> afterok:$jid_a"
echo
echo "read Stage A's drift before trusting Stage B:  grep drift logs/ssl_dapt_${jid_a}.out"
