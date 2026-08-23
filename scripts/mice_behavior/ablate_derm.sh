#!/bin/bash
#
# DERM (Deconfounded ERM, ours) on the mice v1 frame classifier.
#
# WHY THIS EXISTS
# ===============
# The status report closed a section titled "DERM / vREx -- no measurable help". That heading
# was wrong: `train_online_aug.py` only ever implemented vREx (--vrex-beta, "mean risk + beta *
# variance of risk"), and the four runs behind the claim are named vrexAnn_b10/b100 and
# vrexCond_b1/b10. DERM had never been run on this dataset at all. The two methods attack the
# same failure by OPPOSITE means:
#
#   vREx  keeps the training distribution, adds a PENALTY on the spread of risk across envs.
#   DERM  keeps the loss, and REWEIGHTS each sample by Var(Y|E)/P(Y,E) -- it changes the
#         distribution the risk is averaged over.
#
# For binary labels DERM's weight collapses to w(y=1,e) = (1-p_e)/P(e), w(y=0,e) = p_e/P(e)
# with p_e = P(Y=1|E=e). Verified numerically: positives and negatives then carry EQUAL mass
# inside every environment, the effective prevalence in every environment becomes exactly 0.5
# (from a raw 3.5x spread), and each environment's total mass is exactly proportional to
# Var(Y|E). Mean weight is normalised to 1, so the effective learning rate is unchanged and a
# DERM-vs-ERM comparison is not confounded by a quietly different step size.
#
# WHAT IT IS SUPPOSED TO FIX
# ==========================
# The deployed model's bias moves WITH the treatment: predicted/true rate ratio 2.26 (H) /
# 1.80 (O) / 3.51 (P). Inside v1 that costs nothing, because PPI's rectifier corrects any
# predictor. On v2 there are no labels, no rectifier can be built, and a treatment-linked bias
# is exactly what corrupts the estimate. Phase predicts PREVALENCE here, so the classifier can
# score a frame by which phase it looks like rather than by what the mice are doing; DERM
# removes that route by construction while leaving what a contact LOOKS like untouched.
#
# HOW TO READ THE RESULT -- and this is the part the vREx round got wrong
# ======================================================================
# Do NOT rank these arms on macro AP. The quantity that decides whether this pays is r_delta,
# the correlation between true and predicted WITHIN-POOL PHASE DIFFERENCES, because PPI's
# variance reduction is a function of r_delta and nothing else. The vREx arms were judged on AP
# alone, and on AP they are all inside seed noise -- which says nothing about r_delta. Screen
# with:
#     python scripts/mice_behavior/event_eval.py --tag <tag>
#
# and read r_delta against the controls, WITH the caveat that r_delta on the standing 4-pool
# split rests on 4 pools x 2 odours x 2 transitions = 16 points and its operating threshold is
# picked by max-F1 on that same split. It is a SCREEN, not a result. If an arm clears the
# controls by a margin worth having, cross-fit it (3 folds, ~18 GPU-h) and only then quote a
# CI shrinkage.
#
# Controls already on disk, no need to re-run:
#     res448_k2_frozen_d4photo_decay30_seed42   AP 0.4289   r_delta nt --    nn --
#     res448_k2_frozen_d4photo_decay30_seed1    AP 0.4200   r_delta nt 0.417 nn 0.768
#     -> seed noise on AP is 0.0089
#
# Usage:
#     bash scripts/mice_behavior/ablate_derm.sh              # submit every arm
#     ARMS="derm_phase" bash scripts/mice_behavior/ablate_derm.sh
#     DRY=1 bash scripts/mice_behavior/ablate_derm.sh        # print, submit nothing
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
export PATH=/opt/slurm/bin:$PATH
mkdir -p logs

# Held fixed at exactly the config the controls and the vREx arms used, so the only thing that
# varies is the method. Copied from ablate_head_vs_encoder.sh rather than re-derived.
export INPUT_SIZE=448 CONTEXT_K=2 STRIDE=1 AUGMENT=d4_photo PHOTO_STRENGTH=1.0
export OPTIMIZER=adamw LR=3e-4 WEIGHT_DECAY=0.05 DROPOUT=0.4
export WARMUP_EPOCHS=3 LR_DECAY_EPOCHS=30 N_EPOCHS=30 PATIENCE=10
export BATCH_SIZE=64 NEG_RATIO=1 MAX_TRAIN_FRAMES=300000
export LAYERWISE_DECAY=0.65
export JPEG_CACHE_FILE=dataset/mice/v1/jpegcache_k2   # pre-built, skips the ~33 min NFS read
export WANDB=1

# L40S rather than the A100s the earlier sweeps used: 8 nodes sit idle, the encoder is frozen
# so this is not compute-bound, and gpu100/H100 is reserved unless asked for.
#
# MEM=180G, matching ablate_head_vs_encoder.sh, and NOT the 80-100G its comment suggests is enough
# for a run that memory-maps the prebuilt cache. Measured here: two arms were OOM-killed at 100G
# with MaxRSS 104.8 GB. On this cluster the cgroup charges the mapped pages of jpegcache_k2
# (18.6 GiB) to the job, and several arms sharing a node each pay it, so the "clean file-backed
# pages the kernel reclaims" argument does not hold. Do not lower this again without checking
# `sacct -o MaxRSS`.
SB=(--partition="${PARTITION:-gpu}" --gres="${GRES:-gpu:L40S:1}" --time="${TIME:-14:00:00}"
    --mem="${MEM:-180G}" --cpus-per-task=32)

declare -A ARM=(
  # THE motivated arm. Environments are the 3 phases, which is the estimand's own treatment
  # variable -- deconfound the label from the thing whose effect we are estimating.
  [derm_phase]="res448_k2_frozen_d4photo_dermPhase       | ENV_KEY=phase DERM=1 SEED=42"
  # The 6 phase x odour cells. Finer, and arguably the better match: the two odours are
  # distinct treatments with opposite signs on nt, so their prevalence profiles differ and
  # pooling them into 3 phase environments averages two different confounds.
  [derm_cond]="res448_k2_frozen_d4photo_dermCond         | ENV_KEY=condition DERM=1 SEED=42"
  # Seed replicate of the phase arm. Every vREx arm was a single seed, which is why that round
  # could not tell a real move from a draw; do not repeat the mistake.
  [derm_phase_s1]="res448_k2_frozen_d4photo_dermPhase_s1 | ENV_KEY=phase DERM=1 SEED=1"
  # Annotator as the environment. Included as a CONTRAST, not a candidate: annotator is exactly
  # balanced across phase (every scorer took all 6 observations of each pool they touched, so
  # H/O/P = 48/48/48), which means annotator already cancels in a within-pool phase contrast.
  # If this arm moves r_delta as much as the phase arms do, the mechanism is not deconfounding
  # the treatment and the phase result needs a different explanation.
  [derm_ann]="res448_k2_frozen_d4photo_dermAnn           | ENV_KEY=annotator DERM=1 SEED=42"
)
ORDER="derm_phase derm_cond derm_phase_s1 derm_ann"

for arm in ${ARMS:-$ORDER}; do
    spec="${ARM[$arm]:-}"
    [ -z "$spec" ] && { echo "unknown arm: $arm"; exit 1; }
    tag=$(echo "${spec%%|*}" | xargs)
    envs=$(echo "${spec##*|}" | xargs)
    if [ -e "results/vision/mice/frame/$tag/config.json" ]; then
        echo "SKIP  $arm -> $tag already has results"; continue
    fi
    # config.json only appears when a run FINISHES, so that check alone would happily submit a
    # second copy of an arm that is queued or mid-training, and both would write the same
    # results directory.
    if squeue -u "$USER" -h -o '%j' 2>/dev/null | grep -qx "$arm"; then
        echo "SKIP  $arm -> already queued or running"; continue
    fi
    if [ -n "${DRY:-}" ]; then
        echo "[dry] $arm: TAG=$tag $envs"; continue
    fi
    jid=$(env $envs TAG="$tag" sbatch "${SB[@]}" --job-name="$arm" \
              --output="logs/${arm}_%j.out" --error="logs/${arm}_%j.err" \
              --parsable scripts/mice_behavior/train_online_aug.sh)
    echo "submitted $jid  $arm -> $tag"
done
