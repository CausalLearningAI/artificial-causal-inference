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
#     bash scripts/mice_behavior/xfit_odour.sh                    # 4 trainings
#     ARMS="odourF_erm odourF_derm" bash scripts/mice_behavior/xfit_odour.sh
#     DRY=1 bash scripts/mice_behavior/xfit_odour.sh
#     STAGE=predict bash scripts/mice_behavior/xfit_odour.sh      # dense passes for what landed
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

# WEIGHTS=corrected -- DERM's weight table computed for the POPULATION rather than for the
# 1:1-subsampled epoch, and with a duration-neutral P(E). Verified on the real labels:
#
#                   w(y=0) spread          w(y=1) spread     true Var(Y|E)
#   fear nt      2.67x -> 3.59x           1.19x -> 1.01x        3.56x
#   fear nn      1.79x -> 2.45x           1.27x -> 1.00x        2.43x
#   social nt    2.43x -> 1.30x           2.35x -> 1.00x        1.29x
#   social nn    1.88x -> 1.78x           2.51x -> 1.01x        1.77x
#
# The new table reproduces the true ratios; the old one was attenuated on fear (1:1 subsampling
# pushes every p_e to 16-39%, where p(1-p) saturates and compresses the ratio) and on social it was
# reading DURATION -- H is 30 min against O and P's 15, so P_e = 0.50/0.25/0.25 and the positive
# weights varied 2.5x on phase length alone, carrying no prevalence signal at all.
#
# ERM is untouched by these flags, so only the DERM arms are rerun; they pair against the
# already-trained odour_tr{F,S}_erm_last. Requires SELECT=last so the pairing is within one
# selection rule.
#
#     SELECT=last WEIGHTS=corrected bash scripts/mice_behavior/xfit_odour.sh
if [ "${WEIGHTS:-}" = "corrected" ]; then
  [ "${SELECT:-monitor_ap}" = "last" ] || { echo "REFUSING: WEIGHTS=corrected needs SELECT=last, or the ERM leg it pairs against uses a different selection rule." >&2; exit 1; }
  export DERM_PREVALENCE=population DERM_ENV_PRIOR=uniform
  ORDER="odourF_derm odourS_derm"
  for a in $ORDER; do ARM_TAG[$a]="${ARM_TAG[$a]}_popw"; done
fi

if ! grep -q 'train-odour' scripts/mice_behavior/train_online_aug.sh; then
  echo "REFUSING: train_online_aug.sh does not forward --train-odour." >&2; exit 1
fi

# Frozen encoder -> L40S, and it does not contend for the A100s the BitFit folds hold.
SB=(--partition="${PARTITION:-gpu}" --gres="${GRES:-gpu:L40S:1}" --time="${TIME:-10:00:00}"
    --mem="${MEM:-180G}" --cpus-per-task=32)

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
        echo "[dry] $tag  ${ARM_ENV[$arm]}  MONITOR=$MONITOR"; continue
    fi
    jid=$(env ${ARM_ENV[$arm]} VAL_POOLS="$MONITOR" TAG="$tag" SEED=42 sbatch "${SB[@]}" \
              --job-name="$tag" --output="logs/${tag}_%j.out" --error="logs/${tag}_%j.err" \
              --parsable scripts/mice_behavior/train_online_aug.sh)
    echo "submitted $jid  $tag  (${ARM_ENV[$arm]})"
done

# The TEST session comes from a dense pass, because the trainer only ever scores its own monitor
# set. --held-out-odour selects it by EXPOSURE (see the note at the top of this file), and the
# _heldout suffix keeps it clear of the unannotated dump the first attempt wrote to the plain name.
if [ "${STAGE:-all}" = "predict" ]; then
  for arm in ${ARMS:-$ORDER}; do
    tag="${ARM_TAG[$arm]}"
    [ -e "results/vision/mice/frame/$tag/config.json" ] || { echo "SKIP  $tag not landed"; continue; }
    if [ -e "results/vision/mice/frame/$tag/pred_dense_v1_heldout.csv" ]; then
        echo "SKIP  $tag already has its test session"; continue
    fi
    if squeue -u "$USER" -h -o '%j' 2>/dev/null | grep -qx "pd_${tag}"; then
        echo "SKIP  pd_${tag} already queued or running"; continue
    fi
    if [ -n "${DRY:-}" ]; then echo "[dry] predict_dense --held-out-odour $tag v1"; continue; fi
    jid=$(env TAG="$tag" VERSION=v1 \
              EXTRA="--held-out-odour --out-suffix _heldout" \
              sbatch --job-name="pd_${tag}" --parsable scripts/mice_behavior/predict_dense.sh)
    echo "submitted $jid  predict_dense --held-out-odour $tag v1"
  done
fi

cat <<'EOF'

When the four trainings have landed:
    STAGE=predict bash scripts/mice_behavior/xfit_odour.sh   # dense pass, the TEST session
    python scripts/mice_behavior/build_derm.py               # add the odour_* tags to FAMS

Read a_O - a_H on the HELD-OUT exposure, per direction. The prediction under a phase-prior
shortcut is that ERM's bias flips sign between the two directions and DERM's is closer to zero in
both. Do NOT judge this on macro AP: dropping the phase prior costs frame accuracy by construction.
EOF
