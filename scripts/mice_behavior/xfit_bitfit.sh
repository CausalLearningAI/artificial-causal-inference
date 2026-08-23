#!/bin/bash
#
# Cross-fit the ACCURACY-LEADING configuration, then dense-predict with it.
#
# WHY
# ===
# Every estimate in the status report rests on xfit_f1/f2/f3, which were trained on the
# SSL-adapted encoder with the plain 5.03 M head for 20 epochs. That configuration was chosen for
# label-free adaptation reaching v2, not for accuracy -- and it is not the leader:
#
#     xfit_f1/f2/f3 (deployed)   macro AP 0.378 / 0.408 / 0.360   mean 0.382
#     res448_k2_bit6_d4          macro AP 0.5409                  <- leader on every accuracy axis
#
# BitFit-6 trains 70,656 encoder parameters (biases, LayerNorm and LayerScale gains) at
# encoder-lr 1e-3 with d4 augmentation, and beats full six-block fine-tuning at 1/602nd the
# trainable parameters. This script runs it over the same three folds and then over every
# unannotated pool of both cohorts, which is what it takes to put the effects on it.
#
# The folds are the ones already on disk, copied verbatim from the deployed runs' own configs so
# the out-of-fold structure is identical and the two deployments are directly comparable:
#     f1  rd19 rd23 rd24 rd25 rd27 rd30 rd35_2 rd41_3
#     f2  rd15 rd20 rd21 rd26 rd28 rd31 rd32   rd35_3
#     f3  rd11_2 rd13 rd14 rd18 rd22 rd29 rd34 rd64
# Three folds of 8 tile all 24 annotated pools, so every annotated pool is scored by a model that
# never saw it -- the condition PPI++'s rectifier needs.
#
# WHAT EACH ESTIMATOR GETS OUT OF THIS
# ====================================
# PPI++  needs the out-of-fold predictions. This is the only reason cross-fitting is required.
# PPCI   needs no label anywhere and is not bound by cross-fitting at all. It still benefits,
#        for a different reason: its 72-pool average currently mixes 24 pools scored out-of-fold
#        with 48 scored by an average of three models, and fold-to-fold spread in mean predicted
#        occupancy (nt: 4.5 / 8.3 / 7.7 pp) is larger than the labelled-vs-unlabelled gap
#        (5.3 vs 6.8). Averaging the folds on both sides is what keeps that average homogeneous.
# CI     unaffected. It never touches a model.
#
# ONE THING THIS GIVES UP, deliberately: BitFit-6 starts from stock DINOv2, so its encoder never
# saw a v2 frame, where the deployed SSL encoder adapted on 374k frames spanning both cohorts.
# If the v2 numbers move more than the v1 numbers do, that is the first thing to suspect.
#
# Usage:
#     bash scripts/mice_behavior/xfit_bitfit.sh          # 3 trainings, then 6 dense passes
#     DRY=1 bash scripts/mice_behavior/xfit_bitfit.sh    # print, submit nothing
#     STAGE=predict bash scripts/mice_behavior/xfit_bitfit.sh   # dense passes only, no chaining
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
export PATH=/opt/slurm/bin:$PATH
mkdir -p logs

# Exactly res448_k2_bit6_d4's config. AUGMENT=d4 not d4_photo because that is what the winning
# run used; ENCODER_LR=1e-3 because BitFit gets better with a bigger step where full fine-tuning
# gets worse (0.4509 at 1e-5 -> 0.4902 at 1e-3).
export INPUT_SIZE=448 CONTEXT_K=2 STRIDE=1 AUGMENT=d4 PHOTO_STRENGTH=1.0
export OPTIMIZER=adamw LR=3e-4 WEIGHT_DECAY=0.05 DROPOUT=0.4
export WARMUP_EPOCHS=3 LR_DECAY_EPOCHS=30 N_EPOCHS=30 PATIENCE=10
export BATCH_SIZE=64 NEG_RATIO=1 MAX_TRAIN_FRAMES=300000
export UNFREEZE_BLOCKS=6 FT_MODE=bitfit ENCODER_LR=1e-3 LAYERWISE_DECAY=0.65
export CROSS_ATTN_DIM=64 PATCH_POOL_DIM=256      # the 0.52 M head the winning run used
export JPEG_CACHE_FILE=dataset/mice/v1/jpegcache_k2
export WANDB=1

# MEM=180G for the same reason ablate_derm.sh uses it: the cgroup charges the mapped pages of
# jpegcache_k2 to every job sharing the node. Two arms were OOM-killed at 100G.
SB=(--partition="${PARTITION:-gpu}" --gres="${GRES:-gpu:L40S:1}" --time="${TIME:-14:00:00}"
    --mem="${MEM:-180G}" --cpus-per-task=32)

declare -A FOLD=(
  [1]="rd19,rd23,rd24,rd25,rd27,rd30,rd35_2,rd41_3"
  [2]="rd15,rd20,rd21,rd26,rd28,rd31,rd32,rd35_3"
  [3]="rd11_2,rd13,rd14,rd18,rd22,rd29,rd34,rd64"
)

train_ids=()
if [ "${STAGE:-all}" != "predict" ]; then
  for f in 1 2 3; do
      tag="xfit_bit6_f${f}"
      if [ -e "results/vision/mice/frame/$tag/config.json" ]; then
          echo "SKIP  $tag already has results"; continue
      fi
      if squeue -u "$USER" -h -o '%j' 2>/dev/null | grep -qx "$tag"; then
          echo "SKIP  $tag already queued or running"; continue
      fi
      if [ -n "${DRY:-}" ]; then
          echo "[dry] $tag  VAL_POOLS=${FOLD[$f]}"; continue
      fi
      jid=$(env VAL_POOLS="${FOLD[$f]}" TAG="$tag" SEED=42 sbatch "${SB[@]}" \
                --job-name="$tag" --output="logs/${tag}_%j.out" --error="logs/${tag}_%j.err" \
                --parsable scripts/mice_behavior/train_online_aug.sh)
      train_ids+=("$jid")
      echo "submitted $jid  $tag  (holds out ${FOLD[$f]})"
  done
fi

# Dense inference waits for its OWN fold only -- there is no reason for fold 2's v1 pass to wait
# on fold 3's training. afterok, so a failed training does not produce a garbage dump.
i=0
for f in 1 2 3; do
    tag="xfit_bit6_f${f}"
    dep=""
    if [ "${STAGE:-all}" != "predict" ] && [ ${#train_ids[@]} -gt $i ]; then
        dep="--dependency=afterok:${train_ids[$i]}"; i=$((i+1))
    fi
    for v in v1 v2; do
        if [ -n "${DRY:-}" ]; then echo "[dry] predict_dense $tag $v $dep"; continue; fi
        jid=$(env TAG="$tag" VERSION="$v" sbatch $dep --job-name="pd_${tag}_${v}" \
                  --parsable scripts/mice_behavior/predict_dense.sh)
        echo "submitted $jid  predict_dense $tag $v${dep:+  (after ${dep#*:})}"
    done
done

cat <<'EOF'

When all six dense passes have landed, rebuild the report from the new predictions:
    # point FOLDS at the new tags in build_estimates.py / build_examples.py first
    python scripts/mice_behavior/build_estimates.py --report-mismatch
    python scripts/mice_behavior/build_models.py
    python scripts/mice_behavior/build_examples.py
    python scripts/mice_behavior/build_report.py -o /tmp/mice_report.html
EOF
