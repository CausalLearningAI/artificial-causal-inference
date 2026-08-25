#!/bin/bash
#
# DOES DERM COST MORE WHEN THE PIXELS ARE POORER? Riccardo's hypothesis, 2026-08-25.
#
# THE ARGUMENT. DERM's whole bet is "stop scoring a frame by which phase it looks like, score it by
# what the animals are doing". That trade is only cheap if the honest evidence is actually there.
# Two measurements say we may be on the wrong side of it:
#
#   the SHORTCUT is resolution-free. A leave-one-POOL-out linear probe on a grey thumbnail of a
#   QUIET frame reads O vs {H,P} at 0.81 balanced accuracy from a 4x4 thumbnail and 0.86 from 8x8,
#   flat to 128x128 (24 pools). Sixteen pixels for the whole cage. The bag is an enormous
#   low-frequency cue and downsampling removes none of it.
#
#   the SIGNAL is resolution-hungry. The pixel_source ablation, same head, same split:
#   112px -> 0.3139 macro AP, 224px -> 0.3996, full 448 -> ~0.42, STILL CLIMBING.
#
# So lowering resolution strictly widens the gap between how easy the shortcut is and how hard the
# honest signal is -- and DERM, which forbids the shortcut, pays that gap. If that is right, the
# ERM-minus-DERM AP gap should SHRINK as the pixel budget rises, and the fix is more pixels rather
# than a different objective.
#
# --pixel-source resizes each frame down to N and back up to input_size, so the TOKEN COUNT and the
# encoder are held fixed and only the pixel information moves. That is the knob we want; changing
# input_size would confound resolution with sequence length.
#
# ERM at 112 and 224 already exist (res448_k2_frozen_d4photo_px{112,224}); every field below is
# copied from px224's config so the new arms are matched by construction. The full-resolution pair
# is run here too rather than reused, so all three pixel budgets sit on identical settings.
#
#   bash scripts/mice_behavior/ablate_pixel_derm.sh          # 4 arms
#   DRY=1 bash scripts/mice_behavior/ablate_pixel_derm.sh
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
export PATH=/opt/slurm/bin:$PATH
mkdir -p logs

export INPUT_SIZE=448 CONTEXT_K=2 STRIDE=1 AUGMENT=d4_photo PHOTO_STRENGTH=1.0
export OPTIMIZER=adamw LR=3e-4 WEIGHT_DECAY=0.05 DROPOUT=0.4
export WARMUP_EPOCHS=3 LR_DECAY_EPOCHS=30 N_EPOCHS=30
export BATCH_SIZE=64 NEG_RATIO=1 MAX_TRAIN_FRAMES=300000
export UNFREEZE_BLOCKS=0 CROSS_ATTN_DIM=64 PATCH_POOL_DIM=256
export JPEG_CACHE_FILE=dataset/mice/v1/jpegcache_k2
export VAL_POOLS="rd11_2,rd13,rd14,rd18"
export WANDB=1
# A FIXED EPOCH BUDGET for every arm. Selecting on unweighted AP asks for the epoch that best
# exploits the phase prior, which is what DERM removes -- so it would confound exactly the gap
# this experiment measures.
export SELECT=last

SB=(--partition="${PARTITION:-gpu}" --gres="${GRES:-gpu:L40S:1}" --time="${TIME:-12:00:00}"
    --mem=80G --cpus-per-task=32)

# arm -> "PIXEL_SOURCE objective"
# BOTH objectives at EVERY pixel budget, all six under SELECT=last. The existing
# res448_k2_frozen_d4photo_px{112,224} arms are ERM but were selected on unweighted monitor AP, and
# a gap measured across two different selection rules is not a gap -- so the ERM legs are rerun
# here rather than reused. Six runs is the price of the comparison being worth reading.
declare -A ARM=(
  [px112_erm]="112 erm"    [px112_derm]="112 derm"
  [px224_erm]="224 erm"    [px224_derm]="224 derm"
  [pxfull_erm]="0 erm"     [pxfull_derm]="0 derm"
)
ORDER="px112_erm px112_derm px224_erm px224_derm pxfull_erm pxfull_derm"

for a in ${ARMS:-$ORDER}; do
    read -r px obj <<<"${ARM[$a]}"
    tag="res448_k2_frozen_d4photo_${a}_last"
    if [ -e "results/vision/mice/frame/$tag/config.json" ]; then
        echo "SKIP  $tag already has results"; continue
    fi
    if squeue -u "$USER" -h -o '%j' 2>/dev/null | grep -qx "$tag"; then
        echo "SKIP  $tag already queued or running"; continue
    fi
    if [ "$obj" = "derm" ]; then env_args="ENV_KEY=phase DERM=1 VREX_BETA=0"
    else env_args="ENV_KEY=none"; fi
    if [ -n "${DRY:-}" ]; then echo "[dry] $tag  PIXEL_SOURCE=$px  $env_args"; continue; fi
    jid=$(env PIXEL_SOURCE="$px" $env_args TAG="$tag" SEED=42 sbatch "${SB[@]}" \
              --job-name="$tag" --output="logs/${tag}_%j.out" --error="logs/${tag}_%j.err" \
              --parsable scripts/mice_behavior/train_online_aug.sh)
    echo "submitted $jid  $tag  (PIXEL_SOURCE=$px  $env_args)"
done

cat <<'EOF'

Read it as a GAP, never as a level: ERM minus DERM macro AP at each pixel budget. Falling with
resolution supports the hypothesis (the fix is pixels); flat rejects it (the cost is intrinsic).
All six arms are run here under SELECT=last, so every gap is within a matched selection rule.
The older res448_k2_frozen_d4photo_px{112,224} arms are ERM on the OLD rule -- useful as a check on
the ERM legs reproducing, not as the other half of a gap.
EOF
