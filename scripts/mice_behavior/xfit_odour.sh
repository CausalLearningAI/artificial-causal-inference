#!/bin/bash
#
# Train on one exposure session, test on the other, as pure PPCI. ERM against DERM.
#
# WHY THIS SPLIT, AND WHY IT IS SHARPER THAN HOLDING OUT POOLS
# ===========================================================
# Every pool is filmed twice -- fear and social -- through the same three phases. Splitting on the
# EXPOSURE therefore holds the cage, the animals, the annotator, the lighting and the camera fixed
# and varies only the treatment episode. Under a pool-held-out split the model's error on a test
# pool mixes "it has never seen this cage" with "it reads the phase"; here the cage is known, so
# what is left is much more purely the phase channel.
#
# It also gives 24 units of a_O - a_H from ONE training run (24 pools x 1 exposure) against 16 per
# cross-fitting fold -- the standing 4-pool split gives 8.
#
# THE SIGN TEST, which is the real reason for four arms
# ====================================================
# The two exposures carry OPPOSITE true effects on nose-to-tail: H->O reads +0.18 bouts/min under
# fear and -0.31 under social. A model that has learnt its training session's phase prior imports
# that session's prevalence gap into the test session, so the bias it produces is
#
#     bias on test T  ~  (prevalence gap in S) - (prevalence gap in T)
#
# which FLIPS SIGN when the direction flips. Train on fear and the social estimate is pushed
# positive; train on social and the fear estimate is pushed negative. A plain generalisation gap --
# "the model is simply worse on an exposure it never saw" -- cannot produce a sign flip tied to
# which session was trained on. That is the control, and it is what makes the result attributable
# to the prior rather than to difficulty. Hence 2 directions x 2 objectives:
#
#     odourF_erm    train fear,   test social    ERM
#     odourF_derm   train fear,   test social    DERM, environments = the 3 phases
#     odourS_erm    train social, test fear      ERM
#     odourS_derm   train social, test fear      DERM
#
# Drop to one direction with ARMS="odourF_erm odourF_derm" if the queue is tight; the comparison
# still works, it just loses the control.
#
# WHAT THIS IS NOT
# ================
# NOT a deployment estimate. The model has seen every test pool's cage and animals, so its bias
# there is SMALLER than on a genuinely unseen pool -- this is a lower bound on what PPCI suffers on
# the 48 unannotated pools, useful precisely because a large lower bound is a strong statement, but
# it must never be quoted as the deployment number. `xfit_derm.sh` is the deployment-valid version.
#
# NOT usable by PPI++. Its rectifier would sit on pools the model trained on, which is the exact
# failure cross-fitting exists to prevent. PPCI uses no labels anywhere and is not bound by it.
#
# Early stopping never touches the test session: --train-odour keeps the monitor set inside the
# TRAINING exposure (the four held-out pools' same-odour recordings). The test session is scored
# afterwards by predict_dense.py --held-out-odour, which takes every ANNOTATED observation whose
# exposure is not the training one: 24 pools x 3 phases = 72 per arm.
#
# THAT FLAG EXISTS BECAUSE THIS SCRIPT GOT IT WRONG ONCE (2026-08-24). The line above used to read
# "predict_dense.py, which dumps every v1 observation". It does not: it dumps the UNANNOTATED
# pools, and admits labelled ones only behind --labelled-too, gated on the run's held-out POOLS.
# This design holds out an EXPOSURE, so the first four passes scored 288 unannotated observations
# each and none of the test session that carries the truth -- four trained models, ~10 GPU-hours
# of inference, and derm.json's odour_split stuck at n_obs 0. The annotated-only pass is a quarter
# of that work and hits the prebuilt JPEG cache, which the unannotated one could not.
#
# Usage:
#     bash scripts/mice_behavior/xfit_odour.sh                    # 4 trainings + chained dense pass
#     ARMS="odourF_erm odourF_derm" bash scripts/mice_behavior/xfit_odour.sh
#     DRY=1 bash scripts/mice_behavior/xfit_odour.sh
#     STAGE=predict bash scripts/mice_behavior/xfit_odour.sh      # dense passes for what landed
#     STAGE=train   bash scripts/mice_behavior/xfit_odour.sh      # trainings only, no dense pass
#     BIT6=1 ...                                                  # BitFit-6 backbone, tag _bit6
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
export PATH=/opt/slurm/bin:$PATH
mkdir -p logs

# Same recipe as xfit_derm.sh, so the two experiments are one comparison at two split designs:
# frozen stock DINOv2, 448 px, the 0.52 M cross-attention head, D4 + photometric augmentation.
export INPUT_SIZE=448 CONTEXT_K=2 STRIDE=1 AUGMENT=d4_photo PHOTO_STRENGTH=1.0
export OPTIMIZER=adamw LR=3e-4 WEIGHT_DECAY=0.05 DROPOUT=0.4
export WARMUP_EPOCHS=3 LR_DECAY_EPOCHS=30 N_EPOCHS=30 PATIENCE=10
export BATCH_SIZE=64 NEG_RATIO=1 MAX_TRAIN_FRAMES=300000
export UNFREEZE_BLOCKS=0
export CROSS_ATTN_DIM=64 PATCH_POOL_DIM=256
export JPEG_CACHE_FILE=dataset/mice/v1/jpegcache_k2
export WANDB=1

# Fold 3's pools are the monitor set, for one reason only: they are the four the standing split
# already uses, so the monitor is the same cages every other arm on this page was early-stopped on.
# The remaining 20 pools train. All 24 are then scored on the OTHER exposure, monitor pools
# included -- for PPCI that is admissible, since it uses no labels anywhere.
MONITOR="rd11_2,rd13,rd14,rd18"

declare -A ARM_ENV=(
  [odourF_erm]="TRAIN_ODOUR=F ENV_KEY=none"
  [odourF_derm]="TRAIN_ODOUR=F ENV_KEY=phase DERM=1 VREX_BETA=0"
  [odourS_erm]="TRAIN_ODOUR=S ENV_KEY=none"
  [odourS_derm]="TRAIN_ODOUR=S ENV_KEY=phase DERM=1 VREX_BETA=0"
)
declare -A ARM_TAG=(
  [odourF_erm]="odour_trF_erm"   [odourF_derm]="odour_trF_derm"
  [odourS_erm]="odour_trS_erm"   [odourS_derm]="odour_trS_derm"
)
ORDER="odourF_erm odourF_derm odourS_erm odourS_derm"

# SELECT=last gives every arm a FIXED EPOCH BUDGET, which takes the selection rule out of the
# ERM-vs-DERM contrast. The default keeps the epoch with the highest UNWEIGHTED monitor AP -- for a
# DERM arm that is a request for the epoch that best exploits the phase prior, i.e. the selection
# rule pulls against the objective and the comparison inherits it. The last ten epochs of all four
# arms sit inside 0.01 AP of each other, so nothing is being given up by fixing the budget.
#
#     SELECT=last bash scripts/mice_behavior/xfit_odour.sh
#
# Separate tags, so the two selection rules can be read side by side rather than one overwriting
# the other.
if [ "${SELECT:-monitor_ap}" = "last" ]; then
  for a in $ORDER; do ARM_TAG[$a]="${ARM_TAG[$a]}_last"; done
  export SELECT=last
fi

# WEIGHTS=corrected -- DERM's target mass set by the POPULATION Var(Y|E) instead of the
# subsampled one. Training subsamples twice (--max-train-frames keeps positives preferentially,
# then neg_ratio balances 1:1), which drives p_e from 0.4-1.4% to ~25% where p(1-p) saturates and
# the ratio across environments collapses. Verified end to end, env mass share against target:
#
#                old (sampled Var)   new (population Var)   target
#   fear nt           1.55x                3.56x            3.56x
#   fear nn           1.28x                2.43x            2.43x
#   social nt         1.17x                1.29x            1.29x
#   social nn         1.42x                1.77x            1.77x
#
# positive/negative mass balanced to 1e-9 in every cell. It also removes the phase-duration
# artefact for free: the target is a RATE, so H's extra 15 minutes buys it no extra mass.
#
# ERM is untouched by this flag, so only the DERM arms are rerun; they pair against the
# already-trained odour_tr{F,S}_erm_last. Requires SELECT=last so the pairing is within one
# selection rule.
#
#     SELECT=last WEIGHTS=corrected bash scripts/mice_behavior/xfit_odour.sh
if [ "${WEIGHTS:-}" = "corrected" ]; then
  [ "${SELECT:-monitor_ap}" = "last" ] || { echo "REFUSING: WEIGHTS=corrected needs SELECT=last, or the ERM leg it pairs against uses a different selection rule." >&2; exit 1; }
  export DERM_PREVALENCE=population
  ORDER="odourF_derm odourS_derm"
  for a in $ORDER; do ARM_TAG[$a]="${ARM_TAG[$a]}_popw"; done
fi

# BIT6=1 -- THE SAME SPLIT ON THE BitFit-6 BACKBONE, and the reason it is an opt-in here rather
# than a copied launcher is the reason POPW=1 lives inside xfit_bit6_derm.sh: every other export
# in this file is then literally the same line for both backbones, which a copy cannot promise.
#
# WHY THE ARM HAD TO EXIST. DERM was promoted on this page. On the fear direction, seed-averaged
# over three seeds, nose-to-tail H->O reads ERM +0.1843 against DERM -0.0148, paired p 0.0023
# (results/vision/mice/frame/_figures/derm.json, odour_split.seed_avg.train_fear_popw_seedavg).
# Every arm behind that number ran a FROZEN encoder -- n_encoder_trainable 0 in all fourteen
# landed configs. BitFit-6 now leads on accuracy, calibration and in-distribution estimand bias,
# so the one criterion still standing against it is the one criterion it has never been run on.
#
# EXACTLY ONE THING MOVES against the frozen arms, and it is the backbone. UNFREEZE_BLOCKS 0 -> 6
# is the change. FT_MODE and ENCODER_LR are INERT while the encoder is frozen -- every frozen arm
# records encoder_lr 1e-05 and trains nothing with it -- so setting them here cannot retro-alter
# the comparison. AUGMENT deliberately stays d4_photo: xfit_bitfit.sh uses d4, but following it
# would move the augmentation and the backbone in the same step and the contrast against the
# frozen odour arms would then answer nothing. That is the ONE place where "match the odour arms"
# and "match xfit_bit6_f*" disagree, and this split is the thing being matched.
#
# GRAD_CHECKPOINT=1 by default, because the default GRES here is the L40S: BitFit-6 at batch 64
# and 448 px stores backprop activations for six unfrozen blocks over 64 x 5 = 320 images of 1025
# tokens, peaks at 42.53 GiB, and dies 1.88 GiB short on a 44.42 GiB card without it. Checkpointing
# recomputes those blocks in backward instead -- same loss, same gradients, A/B verified in commit
# 7558b6d -- at about 1.64x per epoch, which is why TIME defaults to 20:00:00 here against the
# frozen arms' 10:00:00. Nineteen L40S cards beat queueing eight arms behind the one A100 node.
#
#     SELECT=last BIT6=1 ARMS=odourF_erm bash scripts/mice_behavior/xfit_odour.sh
#     SELECT=last WEIGHTS=corrected BIT6=1 ARMS=odourF_derm bash scripts/mice_behavior/xfit_odour.sh
#
# The tag gains `_bit6` AFTER the weighting marker and BEFORE the seed, so the grammar stays
# `odour_tr{F,S}_{erm,derm}[_last][_popw][_bit6][_s{n}]` and build_derm.py's ODOUR_VARIANTS picks
# the pairs up as one more internally-matched variant.
if [ "${BIT6:-0}" = "1" ]; then
    export UNFREEZE_BLOCKS=6 FT_MODE=bitfit ENCODER_LR=1e-3
    export GRAD_CHECKPOINT="${GRAD_CHECKPOINT:-1}"
    TIME="${TIME:-20:00:00}"
    for a in $ORDER; do ARM_TAG[$a]="${ARM_TAG[$a]}_bit6"; done
fi

if ! grep -q 'train-odour' scripts/mice_behavior/train_online_aug.sh; then
  echo "REFUSING: train_online_aug.sh does not forward --train-odour." >&2; exit 1
fi
if [ "${BIT6:-0}" = "1" ] && ! grep -q 'grad-checkpoint' scripts/mice_behavior/train_online_aug.sh; then
  echo "REFUSING: train_online_aug.sh does not forward --grad-checkpoint, so BIT6=1 on an L40S would OOM." >&2; exit 1
fi

# SEED=n replicates an arm under a different seed, tagged `_s{n}`. The headline paired test rests
# on one seed per arm otherwise, and this project has already seen a metric move 0.18 -> 0.85
# between two seeds of the same config. SEED pairs WITHIN seed only: an ERM_s1 leg pairs against
# a DERM_s1 leg, never across. 42 is the original and keeps the unsuffixed tags.
#
#     SELECT=last SEED=1 ARMS=odourF_erm bash scripts/mice_behavior/xfit_odour.sh
#     SELECT=last WEIGHTS=corrected SEED=1 ARMS=odourF_derm bash scripts/mice_behavior/xfit_odour.sh
if [ -n "${SEED:-}" ] && [ "$SEED" != 42 ]; then
  for a in $ORDER; do ARM_TAG[$a]="${ARM_TAG[$a]}_s${SEED}"; done
fi

# Frozen encoder -> L40S, and it does not contend for the A100s the BitFit folds hold.
SB=(--partition="${PARTITION:-gpu}" --gres="${GRES:-gpu:L40S:1}" --time="${TIME:-10:00:00}"
    --mem="${MEM:-180G}" --cpus-per-task=32)

declare -A TRAIN_JID=()
for arm in ${ARMS:-$ORDER}; do
    tag="${ARM_TAG[$arm]:-}"
    [ -n "$tag" ] || { echo "unknown arm '$arm'" >&2; exit 1; }
    if [ -e "results/vision/mice/frame/$tag/config.json" ]; then
        echo "SKIP  $tag already has results"; continue
    fi
    if squeue -u "$USER" -h -o '%j' 2>/dev/null | grep -qx "$tag"; then
        echo "SKIP  $tag already queued or running"; continue
    fi
    if [ "${STAGE:-all}" = "predict" ]; then continue; fi
    if [ -n "${DRY:-}" ]; then
        echo "[dry] $tag  ${ARM_ENV[$arm]}  MONITOR=$MONITOR" \
             "  gres=${GRES:-gpu:L40S:1} time=${TIME:-10:00:00}" \
             "unfreeze=${UNFREEZE_BLOCKS} ft_mode=${FT_MODE:-full}" \
             "grad_checkpoint=${GRAD_CHECKPOINT:-0}"
        TRAIN_JID[$arm]="<dry>"          # so the dry run also shows the chained dense pass
        continue
    fi
    jid=$(env ${ARM_ENV[$arm]} VAL_POOLS="$MONITOR" TAG="$tag" SEED="${SEED:-42}" sbatch "${SB[@]}" \
              --job-name="$tag" --output="logs/${tag}_%j.out" --error="logs/${tag}_%j.err" \
              --parsable scripts/mice_behavior/train_online_aug.sh)
    TRAIN_JID[$arm]="$jid"
    echo "submitted $jid  $tag  (${ARM_ENV[$arm]})"
done

# The TEST session comes from a dense pass, because the trainer only ever scores its own monitor
# set. --held-out-odour selects it by EXPOSURE (see the note at the top of this file), and the
# _heldout suffix keeps it clear of the unannotated dump the first attempt wrote to the plain name.
#
# THIS IS NOW CHAINED BY DEFAULT, and that is a fix rather than a convenience. A trained odour arm
# with no dense pass contributes NOTHING: derm.json's odour_split reads `pred_dense_v1_heldout.csv`
# and nothing else, so the arm lands in `absent` and the GPU hours are spent for no measurement.
# This project has already paid for that twice -- once with four arms dumping the wrong 288
# observations, and once with a STAGE=train launch whose predict half was never submitted. So the
# default submits both halves, with afterok so a failed training cannot produce a garbage dump.
# STAGE=train opts out (a training run purely to answer a question, where the dense pass is a
# decision to take afterwards); STAGE=predict submits only the dense half for what already landed.
if [ "${STAGE:-all}" != "train" ]; then
  for arm in ${ARMS:-$ORDER}; do
    tag="${ARM_TAG[$arm]}"
    dep=""
    if [ -n "${TRAIN_JID[$arm]:-}" ]; then
        dep="--dependency=afterok:${TRAIN_JID[$arm]}"
    elif [ ! -e "results/vision/mice/frame/$tag/config.json" ]; then
        echo "SKIP  $tag not landed and no training submitted this pass"; continue
    fi
    if [ -e "results/vision/mice/frame/$tag/pred_dense_v1_heldout.csv" ]; then
        echo "SKIP  $tag already has its test session"; continue
    fi
    if squeue -u "$USER" -h -o '%j' 2>/dev/null | grep -qx "pd_${tag}"; then
        echo "SKIP  pd_${tag} already queued or running"; continue
    fi
    if [ -n "${DRY:-}" ]; then
        echo "[dry] predict_dense --held-out-odour $tag v1 ${dep}"; continue
    fi
    jid=$(env TAG="$tag" VERSION=v1 \
              EXTRA="--held-out-odour --out-suffix _heldout" \
              sbatch $dep --job-name="pd_${tag}" --parsable scripts/mice_behavior/predict_dense.sh)
    echo "submitted $jid  predict_dense --held-out-odour $tag v1${dep:+  (after ${dep#*:})}"
  done
fi

cat <<'EOF'

The dense pass is chained above (afterok). When it has landed:
    python scripts/mice_behavior/build_derm.py               # reads pred_dense_v1_heldout.csv
If a training was already queued when this ran, its dense pass was NOT chained; submit it with
    STAGE=predict bash scripts/mice_behavior/xfit_odour.sh   # dense pass, the TEST session

Read a_O - a_H on the HELD-OUT exposure, per direction. The prediction under a phase-prior
shortcut is that ERM's bias flips sign between the two directions and DERM's is closer to zero in
both. Do NOT judge this on macro AP: dropping the phase prior costs frame accuracy by construction.
EOF
