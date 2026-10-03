#!/bin/bash
#
# ff448al: odor-aligned FULL-FRAME mice SAE (replaces the full-frame ep20 SAE: 224 center crop, unaligned).
# Not an sbatch script: run it with plain bash on a login node; it submits the whole chain with dependencies.
#
# Same code path as the ants full-frame SAE antsfull and the odor-aligned mouse-mask SAE fg448al
# (src/eci/foreground.py): the whole 512 px frame, turned so the odor corner is at the top right (align_rot90, before
# DINOv2), resized to 448 (32 x 32 = 1024 patches); rule 'all' = every patch (no mouse mask, no backgrounds read);
# training tokens = a seeded 25% of the 1024 patches of every 1 fps frame of the 432 videos (~133M tokens, ~205 GB).
# SAE: Matryoshka BatchTopK, 1024 latents, k=16, prefixes 128/256/512/1024, same 8 held-out pools as ep20.
#
#   1. fg_extract_train.sh  -> dataset/mice/v1/eci/train_tokens/dinov2_base_l-1_ff448al_fps1_pf25/  (16 tasks)
#   2. train_sae_fg.sh      -> dataset/mice/v1/eci/sae/matryoshka_btk_1024_k16_ff448al_s0/  (checkpoint: fg_rule 'all',
#                              align 'odor', so every later step rotates the frames and takes all 1024 patches)
#   3. fg_encode_all.sh     -> dataset/mice/v1/eci/codes/matryoshka_btk_1024_k16_ff448al_s0/  codes_max / codes_mean
#                              (max / mean over all 1024 patches), n_fg (= 1024); 24 shards
#   4. fg_encode_merge.sh   merge + verify + delete shards
#   5. somp_encode_all.sh   -> codes/matryoshka_btk_1024_k16_ff448al_s0_somp/  (SOMP over all 1024 patches; 24 shards)
#   6. somp_encode_merge.sh merge + verify + delete shards, and codes/..._ff448al_s0_mean (links to codes_mean)
#
# Usage: bash scripts/eci/ff448al_chain.sh            (prints the job ids)
#        DRY=1 bash scripts/eci/ff448al_chain.sh      (prints the sbatch commands only)
# Finished shards / SAEs are skipped, so the chain can be resubmitted after a failure.
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
mkdir -p logs
export DOMAIN=mice  # every step below is mice; no stray EXTRA_ARGS / SAE from the caller's shell
unset EXTRA_ARGS SAE

T=dataset/mice/v1/eci/train_tokens/dinov2_base_l-1_ff448al_fps1_pf25
SAE=matryoshka_btk_1024_k16_ff448al_s0

run() {  # submit (or print with DRY=1) and echo the job id
    if [ -n "${DRY:-}" ]; then echo "DOMAIN=${DOMAIN:-} SAE=${SAE:-} EXTRA_ARGS='${EXTRA_ARGS:-}' $*" >&2; echo "DRY"
    else "$@"; fi
}
dep() { [ "$1" = "DRY" ] && echo "" || echo "--dependency=afterok:$1"; }

j1=$(EXTRA_ARGS="--rule all --patch-frac 0.25 --align odor --out-dir $T" \
    run sbatch --parsable --export=ALL -p gpu scripts/eci/fg_extract_train.sh)
j2=$(EXTRA_ARGS="--tokens-dir $T --tag ff448al" \
    run sbatch --parsable --export=ALL -p gpu --gres=gpu:1 --mem=480G --array=0 $(dep "$j1") scripts/eci/train_sae_fg.sh)
j3=$(SAE=$SAE run sbatch --parsable --export=ALL $(dep "$j2") scripts/eci/fg_encode_all.sh)
j4=$(SAE=$SAE run sbatch --parsable --export=ALL $(dep "$j3") scripts/eci/fg_encode_merge.sh)
j5=$(SAE=$SAE run sbatch --parsable --export=ALL $(dep "$j4") scripts/eci/somp_encode_all.sh)
j6=$(SAE=$SAE run sbatch --parsable --export=ALL $(dep "$j5") scripts/eci/somp_encode_merge.sh)
echo "tokens $j1 -> SAE $j2 -> codes $j3 -> merge $j4 -> SOMP $j5 -> merge $j6"
