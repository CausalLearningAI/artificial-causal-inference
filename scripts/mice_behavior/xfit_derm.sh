#!/bin/bash
#
# Cross-fit the DERM / ERM MATCHED PAIR, so a_O - a_H is measured on 24 pools instead of 4.
#
# WHY
# ===
# A physical bag sits in a corner of the cage during the O phase, so the treatment is legible in a
# frame that carries no behaviour at all. Measured (build_derm.py, `probe`): a leave-one-pool-out
# linear probe on a 32x32 grey thumbnail of a QUIET frame -- no scored behaviour, >=5 s from any
# bout -- identifies O against H-and-P at 0.946 balanced accuracy. The exposure (fear vs social) is
# at chance, which is the negative control: the probe reads the protocol, not the cage or the hour.
#
# ERM's objective contains nothing that penalises using that. Its optimum is P(Y=1|x), which
# INCLUDES the phase-conditional prior; DERM's optimum divides the prior odds out. So the shortcut
# is available and only one of the two objectives is trying to close it.
#
# What that shortcut becomes, if it is used, is the second term of
#
#     E[D_f] = b * E[D_Y] + (a_O - a_H)
#
# a bias in the estimand itself -- and on the standing split it measures at
#
#     nose-to-tail   ERM +0.163   DERM +0.100     against a pooled true effect of -0.062
#     nose-to-nose   ERM -0.002   DERM +0.025     against +0.425
#
# Both nose-to-tail values positive with DERM nearer zero, which is correction rather than
# overshoot. But the 95% intervals are [-0.13, +0.46] and [-0.12, +0.32] and the paired test gives
# p = 0.44, because the standing split has FOUR pools. That is the only reason this is unresolved.
#
# WHAT THIS BUYS
# ==============
# Three folds of 8 tile all 24 annotated pools, so a_O - a_H is measured on 24 instead of 4 -- the
# standard error falls by about sqrt(6) = 2.4x, taking +-0.29 to about +-0.12, which resolves +0.163
# from zero. Both arms run over the same folds, so the comparison stays paired.
#
# It is also the only way to get the quantities that actually matter downstream:
#   * the real PPI++ interval under each objective (needs out-of-fold predictions)
#   * the real PPCI point estimate under each objective, against CI as ground truth -- PPCI has no
#     rectifier, so a_O - a_H IS its bias, and this is the one place it can be checked
#
# DO NOT judge the outcome on macro AP. A model that has stopped using the phase prior must be
# slightly worse at frame classification, because the prior is genuinely informative for that task.
# The -0.02 AP already measured is the expected price of the correction, not evidence against it.
#
# STAGE B, and it is the cheaper answer
# ====================================
# The cue is spatially localised: O-against-the-rest reads 0.903 from the bottom-left quadrant
# alone against 0.507-0.581 from the other three, and the same corner in all four probed pools.
# Blanking the bottom-left 25% of each side (6.2% of the frame) drops the probe from 0.946 to 0.657.
# So `ARMS=mask` runs the ERM control with that corner masked, which removes the cue at the INPUT
# rather than asking the model to unlearn it. Run it alongside: if masking alone kills the bias,
# it is the better fix, and it is verifiable without any training at all.
#
# The residual 0.657 is not a failure. The bag also changes WHERE THE ANIMALS ARE, and that is real
# behaviour which must not be removed -- masking takes out the non-behavioural cue and leaves it.
# D4 augmentation is not a substitute: it randomises which corner the bag lands in, so the model
# cannot use its POSITION, but presence anywhere on the border still reads 0.951.
#
# COST
# ====
# Frozen stock DINOv2, so these fit on an L40S and do not need the A100 that BitFit-6 does. Six
# trainings for the pair (plus three for the mask arm). Dense inference over the unannotated pools
# is a SEPARATE stage and is only worth running once the pair has answered -- pass STAGE=predict.
#
# Usage:
#     bash scripts/mice_behavior/xfit_derm.sh                  # the matched pair, 6 trainings
#     ARMS="erm derm mask" bash scripts/mice_behavior/xfit_derm.sh
#     DRY=1 bash scripts/mice_behavior/xfit_derm.sh            # print, submit nothing
#     STAGE=predict bash scripts/mice_behavior/xfit_derm.sh    # dense passes for what has landed
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
export PATH=/opt/slurm/bin:$PATH
mkdir -p logs

# WHICH HEAD, AND WHY IT IS NOT THE ONE THE STANDING ARMS USE
# ===========================================================
# The comparison this script exists for is DERM against ERM at 24 pools. That stays valid for any
# head, as long as BOTH arms carry the SAME one -- and it is a paired comparison over the same three
# folds, so the head is held fixed by construction.
#
# So use the better head. `ablate_derm.sh` never set CROSS_ATTN_DIM/PATCH_POOL_DIM, which left every
# standing DERM/vREx arm on the plain 5.03 M head -- an accident of that launcher, and the weakest of
# the five heads measured in 04.5 (macro AP 0.4163/0.4202 against 0.4289/0.4200 for the 0.52 M
# cross-attention head at 10x fewer parameters). The 0.52 M head is also what every accuracy-leading
# run in the pipeline uses: BitFit-6, the head-vs-encoder family, and the xfit_bit6 folds. Matching
# it means this result composes with those instead of sitting on an island.
#
# WHAT THAT GIVES UP, stated so nobody trips on it: the 24-pool numbers from this script are NOT
# stackable against the 4-pool numbers already in section 04.6, because the head moves at the same
# time as the sample size. Nothing is lost for the question at hand -- xfit_erm IS the matched
# control for xfit_derm -- but do not read a 4-pool value and a 24-pool value as one series.
#
# Encoder stays FROZEN STOCK: it keeps the single variable the objective, it fits on an L40S, and it
# does not contend for the A100s the BitFit folds are already holding. The probe shows 1024
# downsampled pixels carry the treatment at 0.946, so a frozen DINOv2's features certainly do -- the
# shortcut is available to this model too.
export INPUT_SIZE=448 CONTEXT_K=2 STRIDE=1 AUGMENT=d4_photo PHOTO_STRENGTH=1.0
export OPTIMIZER=adamw LR=3e-4 WEIGHT_DECAY=0.05 DROPOUT=0.4
export WARMUP_EPOCHS=3 LR_DECAY_EPOCHS=30 N_EPOCHS=30 PATIENCE=10
export BATCH_SIZE=64 NEG_RATIO=1 MAX_TRAIN_FRAMES=300000
export UNFREEZE_BLOCKS=0
export CROSS_ATTN_DIM=64 PATCH_POOL_DIM=256      # the 0.52 M head, not the launcher's 5.03 M default
export JPEG_CACHE_FILE=dataset/mice/v1/jpegcache_k2
export WANDB=1

# The three deployment folds, copied verbatim from xfit_bitfit.sh so all three cross-fitted
# deployments share one out-of-fold structure and are directly comparable.
declare -A FOLD=(
  [1]="rd19,rd23,rd24,rd25,rd27,rd30,rd35_2,rd41_3"
  [2]="rd15,rd20,rd21,rd26,rd28,rd31,rd32,rd35_3"
  [3]="rd11_2,rd13,rd14,rd18,rd22,rd29,rd34,rd64"
)

# arm -> tag prefix | extra env. `mask` needs MASK_CORNER support in train_online_aug.py; the
# script refuses rather than silently training an unmasked arm under a masked tag.
declare -A ARM_ENV=(
  [erm]="ENV_KEY=none"
  [derm]="ENV_KEY=phase DERM=1 VREX_BETA=0"
  [mask]="ENV_KEY=none MASK_CORNER=bottom-left MASK_FRAC=0.25"
)
declare -A ARM_TAG=([erm]="xfit_erm" [derm]="xfit_derm" [mask]="xfit_ermMask")

# L40S is right for a frozen encoder; MEM high because the cgroup charges the mapped pages of
# jpegcache_k2 to every job sharing the node (two arms were OOM-killed at 100G).
SB=(--partition="${PARTITION:-gpu}" --gres="${GRES:-gpu:L40S:1}" --time="${TIME:-10:00:00}"
    --mem="${MEM:-180G}" --cpus-per-task=32)

if [[ " ${ARMS:-erm derm} " == *" mask "* ]] \
   && ! grep -q 'mask-corner\|MASK_CORNER' scripts/mice_behavior/train_online_aug.py \
                                            scripts/mice_behavior/train_online_aug.sh; then
  echo "REFUSING the mask arm: train_online_aug has no --mask-corner flag yet." >&2
  echo "Implement it (blank the corner AFTER the resize, BEFORE augmentation and normalisation)," >&2
  echo "or run without it:  ARMS=\"erm derm\" bash \$0" >&2
  exit 1
fi

train_ids=(); train_tags=()
for arm in ${ARMS:-erm derm}; do
  [ -n "${ARM_TAG[$arm]:-}" ] || { echo "unknown arm '$arm'" >&2; exit 1; }
  for f in 1 2 3; do
      tag="${ARM_TAG[$arm]}_f${f}"
      if [ -e "results/vision/mice/frame/$tag/config.json" ]; then
          echo "SKIP  $tag already has results"; continue
      fi
      if squeue -u "$USER" -h -o '%j' 2>/dev/null | grep -qx "$tag"; then
          echo "SKIP  $tag already queued or running"; continue
      fi
      if [ "${STAGE:-all}" = "predict" ]; then continue; fi
      if [ -n "${DRY:-}" ]; then
          echo "[dry] $tag  VAL_POOLS=${FOLD[$f]}  ${ARM_ENV[$arm]}"; continue
      fi
      jid=$(env ${ARM_ENV[$arm]} VAL_POOLS="${FOLD[$f]}" TAG="$tag" SEED=42 sbatch "${SB[@]}" \
                --job-name="$tag" --output="logs/${tag}_%j.out" --error="logs/${tag}_%j.err" \
                --parsable scripts/mice_behavior/train_online_aug.sh)
      train_ids+=("$jid"); train_tags+=("$tag")
      echo "submitted $jid  $tag  (holds out ${FOLD[$f]})"
  done
done

# Dense inference is NOT chained by default. a_O - a_H, r-delta and the paired test all come from
# each fold's own val_probs.npz, so the question this script exists for is answered without a
# single dense pass. Only PPI++ and PPCI on the 48 unannotated pools need them -- run STAGE=predict
# once the pair has actually separated.
if [ "${STAGE:-all}" = "predict" ]; then
  for arm in ${ARMS:-erm derm}; do
    for f in 1 2 3; do
      tag="${ARM_TAG[$arm]}_f${f}"
      [ -e "results/vision/mice/frame/$tag/config.json" ] || { echo "SKIP  $tag not landed"; continue; }
      for v in v1 v2; do
        if [ -n "${DRY:-}" ]; then echo "[dry] predict_dense $tag $v"; continue; fi
        jid=$(env TAG="$tag" VERSION="$v" sbatch --job-name="pd_${tag}_${v}" \
                  --parsable scripts/mice_behavior/predict_dense.sh)
        echo "submitted $jid  predict_dense $tag $v"
      done
    done
  done
fi

cat <<'EOF'

When the trainings have landed, the answer is one command -- no dense pass needed:
    python scripts/mice_behavior/build_derm.py     # add the xfit_* tags to its FAMS map first
Read `estimand_bias`: the mean a_O - a_H per family, now over 24 pools. Judge it there.
Do NOT judge it on macro AP -- giving up the phase prior costs frame accuracy by construction.
EOF
