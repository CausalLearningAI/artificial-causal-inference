#!/bin/bash
#
# Cross-fit BitFit-6 + DERM -- the one cell of the 2x2 that has never been cross-fitted -- then
# dense-predict with it.
#
# WHY
# ===
# Two cross-fitted families already exist over the SAME three folds of 8 pools, and they bracket
# this one on the two axes that matter:
#
#     xfit_bit6_f{1,2,3}   BitFit-6 encoder, ERM objective      3-fold mean macro AP 0.510
#     xfit_derm_f{1,2,3}   frozen stock encoder, DERM objective 3-fold mean macro AP 0.388
#
# The first is the accuracy leader. The second is the currently deployed predictor, chosen for the
# estimand rather than for accuracy. Nobody has run the combination at 24 pools, and the 4-pool
# evidence says the combination is not a compromise between the two -- it is better than either on
# the thing the report is actually about. From `results/vision/mice/frame/_figures/derm.json`
# (`estimand` block), bias in bouts/min on the H->O transition, mean over 2 seeds:
#
#                    nose-to-tail    nose-to-nose    macro AP
#     frozen ERM        +0.167          -0.010         0.418
#     frozen DERM       +0.100          +0.023         0.398
#     BitFit ERM        +0.190          -0.179         0.537   <- accuracy leader, WORST nt shortcut
#     BitFit DERM       -0.013          +0.008         0.494   <- cleanest arm on BOTH behaviours
#
# BitFit ERM is the arm that leans hardest on the phase shortcut: fine-tuning gives the encoder the
# capacity to read the bag, and ERM's optimum is P(Y=1|x), which INCLUDES the phase-conditional
# prior. DERM divides that prior odds out. So the pairing is not "a better model" against "a fairer
# model" -- it is the same capacity with and without the licence to use the shortcut.
#
# The 4-pool numbers cannot resolve any of that: their 95% intervals are about +-0.29 wide, which
# does not separate +0.190 from -0.013. Three folds of 8 tile all 24 annotated pools, cutting the
# standard error by about sqrt(6) = 2.4x, to roughly +-0.12. That is the entire point of this run.
#
# WHAT MUST BE HELD FIXED, and why deviating makes this worthless
# ===============================================================
# The comparison this script exists for is against xfit_bit6_f*, so this arm is xfit_bit6_f* with
# the objective switched and NOTHING else moved. Every export below is copied verbatim from
# xfit_bitfit.sh. The only additions are ENV_KEY=phase and DERM=1 (plus VREX_BETA=0, which is
# already the trainer's default and is stated only so the arm reads unambiguously as DERM-alone
# rather than DERM-plus-vREx). Verified against the two reference configs on disk: the objective
# controls exactly three recorded fields -- derm, env_key and n_environments -- and if anything
# else differs, the pairing against the ERM cross-fit is broken and the run answers nothing.
#
# Two settings that look like knobs and are NOT, because the siblings used them:
#   AUGMENT=d4      xfit_derm_f* used d4_photo, xfit_bit6_f* used d4. Follow xfit_bit6_f*.
#   SELECT unset    i.e. monitor_ap. `--select last` is the cleaner rule for an objective
#                   comparison (monitor AP rewards the prior-exploitation DERM removes), but both
#                   cross-fit families selected on monitor_ap, so changing it here would move the
#                   selection rule and the objective at the same time.
#
# THE PREVALENCE WEIGHTING: sampled, deliberately, and it is not the "best" one
# ============================================================================
# DERM's target mass can come from the SUBSAMPLED epoch (`sampled`) or from every annotated frame
# of the training observations (`population`). Population is where the confound actually lives --
# 1:1 negative subsampling pushes p_e from 0.4-1.4% up to 16-39%, saturating p(1-p) and collapsing
# the across-environment ratio 3.56x -> 1.58x, so ~77% of the confound is gone before DERM sees it.
# On the odour split the population weighting is the strongest DERM evidence this project has.
#
# This script DEFAULTS to `sampled`, because xfit_derm_f* did (their configs predate the field
# being dumped; the wrapper's default has always been sampled) and a paired comparison beats a
# stronger arm that lines up with nothing.
#
# POPW=1 IS THAT FOLLOW-UP, and it lives here rather than in a copy of this file
# ============================================================================
# The sampled cross-fit has now been scored, and DERM made the estimand bias WORSE on this
# backbone -- nn went from 0.50x to 1.00x of the true effect, paired p 0.0295. That is exactly
# the failure the odour split already diagnosed and fixed: on `nn` there, sampled-DERM did
# -0.292 of harm where ERM sat at -0.128, and population weights moved it to +0.036. So the
# open question is whether BitFit's DERM failure is the WEIGHTING or the BACKBONE, and the only
# way to ask it is one variable moved against the runs above.
#
# Because it is one variable, it is an opt-in here and NOT a second script: every export in this
# file is then literally the same line for both arms, which a copied launcher cannot promise.
#
#     POPW=1 GRES=gpu:L40S:1 GRAD_CHECKPOINT=1 TIME=20:00:00 STAGE=train \
#         bash scripts/mice_behavior/xfit_bit6_derm.sh
#
# IT MOVES TWO RECORDED FIELDS, NOT ONE, AND THAT IS FORCED
# ---------------------------------------------------------
# derm_prevalence sampled -> population, AND derm_floor 0.02 -> 1e-4. The second is not a choice:
# the population Var(Y|E) on these folds is 0.0057-0.0144, and a floor of 0.02 gives a variance
# floor of 0.0196 that sits ABOVE ALL SIX cells, clipping every environment to one value and
# making the correction an exact no-op. The trainer refuses outright rather than training it
# (`--derm-floor ... is above EVERY population Var(Y|E) here`). So leave DERM_FLOOR unset and let
# the trainer pick 1e-4 -- which is what odour_tr{F,S}_derm_last_popw recorded.
#
# THE TARGET WEIGHT TABLE, computed BEFORE launch so the log can be checked and not rationalised
# ---------------------------------------------------------------------------------------------
# Population Var(Y|E) per fold, over every annotated frame of the 16 training pools, and the
# `target ratio` the trainer prints at epoch 1 (max/min over H,O,P):
#
#     fold 1   H nt 0.00931 nn 0.00824   O nt 0.01444 nn 0.00974   P nt 0.01158 nn 0.00683
#              -> target ratio  nt 1.55x   nn 1.43x
#     fold 2   H nt 0.00811 nn 0.00594   O nt 0.00905 nn 0.00806   P nt 0.01080 nn 0.00569
#              -> target ratio  nt 1.33x   nn 1.42x
#     fold 3   H nt 0.00912 nn 0.00797   O nt 0.01338 nn 0.00864   P nt 0.01177 nn 0.00654
#              -> target ratio  nt 1.47x   nn 1.32x
#
# EXPECT A SMALLER CORRECTION THAN THE ODOUR SPLIT PROMISED. Those 1.3-1.6x are nowhere near the
# fear session's 3.56x/2.43x that made population weighting look decisive. The reason is that the
# cross-fit trains on BOTH odours pooled: fear suppresses the home-cage phase hard (H nt Var
# 0.00374 against O's 0.01329) while social is nearly flat (1.29x), and pooling them averages the
# confound away before DERM sees it. For reference the SAMPLED arm above logged env mass share
# ratios of 1.31x / 1.09x on fold 1, so this moves the spread 1.31x -> 1.55x and 1.09x -> 1.43x.
# Real, one-directional, and much smaller than the odour split. A null result here is a genuine
# answer -- it says the BitFit backbone, not the weighting, is what breaks DERM -- so do not read
# a null as a botched launch.
#
# DO NOT JUDGE THIS ARM ON MACRO AP
# =================================
# A model that has stopped using the phase prior must be slightly worse at frame classification,
# because the prior is genuinely informative for that task. The measured price on this project is
# about -0.02 to -0.05 AP (frozen: 0.418 -> 0.398; BitFit at 4 pools: 0.537 -> 0.494). Expect this
# to land near 0.47-0.49 against xfit_bit6_f*'s 0.510. That is the cost of the correction, not
# evidence against it. What decides is the estimand bias, and that is measured by build_derm.py.
#
# COST
# ====
# A100, NOT L40S. BitFit-6 unfreezes six blocks, so backprop retains activations through them at
# 1024 tokens x batch 64, and that does not fit in the L40S's 44 GiB -- ablate_derm.sh records the
# bit6 arms dying there 1.88 GiB short. xfit_bit6_f1 ran 7h36m and peaked at 131.8 GB host RSS, so
# 14:00:00 and 180G are the sibling's numbers and are not padding: the cgroup charges the mapped
# pages of jpegcache_k2 to every job sharing the node, and arms have been OOM-killed at 100G.
# Partition `gpu`, not `gpu100` -- H100s are reserved unless explicitly asked for.
#
# Dense inference IS chained here, unlike xfit_derm.sh. That script deferred it because a_O - a_H
# comes from each fold's own val_probs.npz and needs no dense pass; this arm is meant to ENTER the
# effects grid, and build_estimates.py's predictor_ready() skips a predictor unless ALL THREE folds
# have both pred_dense_v1.csv and val_probs.npz. Two folds of three would leave eight annotated
# pools with no out-of-fold prediction and silently invalidate PPI++, so it refuses to half-build.
#
# Usage:
#     bash scripts/mice_behavior/xfit_bit6_derm.sh          # 3 trainings, then 6 dense passes
#     DRY=1 bash scripts/mice_behavior/xfit_bit6_derm.sh    # print, submit nothing
#     STAGE=predict bash scripts/mice_behavior/xfit_bit6_derm.sh   # dense passes only, no chaining
#     STAGE=train   bash scripts/mice_behavior/xfit_bit6_derm.sh   # trainings only, no dense passes
#     POPW=1 ...                                            # population target mass, tag _popw
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
export PATH=/opt/slurm/bin:$PATH
mkdir -p logs

# ---- copied verbatim from xfit_bitfit.sh; do not "improve" any of these ----------------------
export INPUT_SIZE=448 CONTEXT_K=2 STRIDE=1 AUGMENT=d4 PHOTO_STRENGTH=1.0
export OPTIMIZER=adamw LR=3e-4 WEIGHT_DECAY=0.05 DROPOUT=0.4
export WARMUP_EPOCHS=3 LR_DECAY_EPOCHS=30 N_EPOCHS=30 PATIENCE=10
export BATCH_SIZE=64 NEG_RATIO=1 MAX_TRAIN_FRAMES=300000
export UNFREEZE_BLOCKS=6 FT_MODE=bitfit ENCODER_LR=1e-3 LAYERWISE_DECAY=0.65
export CROSS_ATTN_DIM=64 PATCH_POOL_DIM=256      # the 0.52 M head the winning run used
export JPEG_CACHE_FILE=dataset/mice/v1/jpegcache_k2
export WANDB=1
# ---- the objective, and the ONLY thing this script adds -------------------------------------
# DERM_PREVALENCE and DERM_FLOOR stay UNSET on purpose: unset gives sampled + floor 0.02, which is
# what xfit_derm_f* trained with. Forwarding DERM_FLOOR unconditionally once turned the whole
# correction into an exact no-op, which is why the wrapper only passes it when it is set.
export ENV_KEY=phase DERM=1 VREX_BETA=0

# POPW=1 switches the target mass to the population Var(Y|E) and renames the arm. DERM_FLOOR is
# deliberately NOT set: the wrapper only forwards it when non-empty, so the trainer applies its
# mode-dependent default of 1e-4. Setting it here to "match" the sampled arm's 0.02 would be the
# one edit that silently destroys the run.
SUF=""
if [ "${POPW:-0}" = "1" ]; then
    export DERM_PREVALENCE=population
    SUF="_popw"
fi

# THE L40S ESCAPE HATCH -- opt-in, defaults unchanged
# ==================================================
# The COST section above is still the truth about the DEFAULT recipe: BitFit-6 at batch 64 and
# 448 px stores backprop activations for six unfrozen blocks over 64 x 5 = 320 images of 1025
# tokens, and that peaks at 42.53 GiB on a 44.42 GiB L40S -- dying in the encoder forward,
# 1.88 GiB short, before one optimiser step. But there is exactly ONE A100 node on this cluster
# and nineteen free L40S cards, so the default recipe serialises three folds over ~36 hours
# while most of the fleet sits idle.
#
# GRAD_CHECKPOINT=1 recomputes the six unfrozen blocks in backward instead of storing them,
# which buys back that memory for roughly one extra encoder forward per step. It changes the
# arithmetic not at all -- same loss, same gradients, verified A/B -- so a fold trained this way
# is still paired with xfit_bit6_f*. Both overrides are needed together:
#
#     GRES=gpu:L40S:1 GRAD_CHECKPOINT=1 TIME=20:00:00 bash scripts/mice_behavior/xfit_bit6_derm.sh
#
# The defaults below stay the A100 recipe, because that is what this script's COST section
# describes and what the sibling cross-fits ran on.
SB=(--partition="${PARTITION:-gpu}" --gres="${GRES:-gpu:A100:1}" --time="${TIME:-14:00:00}"
    --mem="${MEM:-180G}" --cpus-per-task=32)
export GRAD_CHECKPOINT="${GRAD_CHECKPOINT:-0}"

# The three deployment folds, copied verbatim from xfit_bitfit.sh / xfit_derm.sh so all four
# cross-fitted families share one out-of-fold structure and are directly comparable.
declare -A FOLD=(
  [1]="rd19,rd23,rd24,rd25,rd27,rd30,rd35_2,rd41_3"
  [2]="rd15,rd20,rd21,rd26,rd28,rd31,rd32,rd35_3"
  [3]="rd11_2,rd13,rd14,rd18,rd22,rd29,rd34,rd64"
)

train_ids=()
if [ "${STAGE:-all}" != "predict" ]; then
  for f in 1 2 3; do
      tag="xfit_bit6_derm${SUF}_f${f}"
      # config.json only appears when a run FINISHES, so this check alone would happily submit a
      # second copy of an arm that is queued or mid-training, and both would write the same dir.
      if [ -e "results/vision/mice/frame/$tag/config.json" ]; then
          echo "SKIP  $tag already has results"; continue
      fi
      if squeue -u "$USER" -h -o '%j' 2>/dev/null | grep -qx "$tag"; then
          echo "SKIP  $tag already queued or running"; continue
      fi
      if [ -n "${DRY:-}" ]; then
          echo "[dry] $tag  VAL_POOLS=${FOLD[$f]}  ENV_KEY=phase DERM=1 VREX_BETA=0" \
               "  gres=${GRES:-gpu:A100:1} grad_checkpoint=${GRAD_CHECKPOINT}"; continue
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
#
# The encoder reload is safe for this family: unfreeze_blocks=6 > 0, so predict_unannotated.py's
# build_model() takes the encoder from the run's own best_encoder.pt (the ~300 KB of BitFit
# tensors) and refuses if it is missing. The silent-substitution trap only bites runs that froze
# the encoder at someone else's --init-encoder checkpoint and therefore left no artefact behind.
# STAGE=train stops here. The dense passes are what lets an arm ENTER the effects grid, and that
# is a decision to take AFTER the folds land and the estimand bias is read -- not a side effect of
# launching a training. Chaining them by default is right for an arm already meant for deployment;
# it is wrong for an arm being run to answer a question.
i=0
for f in 1 2 3; do
    if [ "${STAGE:-all}" = "train" ]; then break; fi
    tag="xfit_bit6_derm${SUF}_f${f}"
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

When the three trainings have landed, the question this script exists for is one command away --
no dense pass needed for it:
    python scripts/mice_behavior/build_derm.py    # add the xfit_bit6_derm_* tags to its FAMS map
Read `estimand_bias`: the mean a_O - a_H per family, now over 24 pools. Judge it THERE.
Do NOT judge it on macro AP -- giving up the phase prior costs frame accuracy by construction.

The six dense passes are what lets this arm enter the effects grid afterwards; predictor_ready()
needs pred_dense_v1.csv AND val_probs.npz on all three folds before build_estimates.py will use it.
EOF
