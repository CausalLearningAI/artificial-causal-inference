#!/bin/bash
#
# MABe22 mouse ECI - PRE-REGISTERED CONFIRMATION RUN (submit-only launcher; the jobs run scripts/benchmark/mabe22/eci.sh).
# Rule copied verbatim from the coordinator's message (2026-10-15 ~06:30 CEST), fixed BEFORE any new number:
#
# - **Arms:**
#   - A0 (control);
#   - D2 (Δ 2 s);
#   - D4 (Δ 4 s);
#   - A7 (D4 orbit, Δ 1 s);
#   - A7D2 (orbit + Δ 2 s).
#   Everything else is the A0 recipe: Matryoshka BTK 4096 k16, θ*, your MABe mask, 448 upsampled.
# - **Seeds:** 6 FRESH SAE seeds per arm (e.g. 100–105; never used before), same train pool, same eval frames, 2 split
#   seeds as before.
# - **Primary metric:** mean cross-fitted AUROC over the 4 labels oral_oral, oral_genital, approaches, huddles. oral_ear
#   is dropped from the rule, because its AP is at chance in every arm. Still report oral_ear.
# - **Win:**
#   - the mean over seeds of (arm − A0) is ≥ +0.010;
#   - AND its 95% CI excludes 0. Use a Welch t interval on the per-seed primary means, 6 vs 6.
#   - AND AP on oral_oral and on oral_genital are each ≥ 0.9× A0's.
# - **Also report:**
#   - the bimodality of each arm (per-seed primary means, listed);
#   - the honest top-1% on the oral labels.
# - **Pre-registered decision:** if both A7 and D2 win, A7D2 is preferred only if it also beats each of them by ≥ 0.01
#   with a CI excluding 0. Otherwise the cheaper winner is preferred, where D2 is cheaper than A7 (A7 costs 8× the
#   encoding).
#
# Implementation: arm names in eci_train.ARMS: A0, D2, X_D4 (= D4), A7, X_A7D2 (= A7D2); SAE seeds 100-105, results in
# results/vision/benchmark_mabe22/eci/confirm/ (MABE_RUN). The token store is re-encoded once (the earlier stores were
# deleted): the MABe mask ('arena') recomputed, 448 tokens of t, t-1 s, t-2 s, t-4 s and D4 orbit tokens of t, t-1 s,
# t-2 s; all arms of this run read the same new store. Scored by scripts/benchmark/mabe22/eci_confirm.py.
#
# Usage (from the repo root): bash scripts/benchmark/mabe22/eci_confirm.sh encode | train
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
G="-p gpu --gres=gpu:1 --exclude=gpu242,gpu150,gpu230,gpu241 --cpus-per-task=8"
KEEP="m448_t m448_p m448_p60 m448_p120 m448o_t m448o_p m448o_p60 pos448"
case "${1:-}" in
  encode)
    for i in $(seq 0 7); do
      PY=eci_encode EXTRA="--set train --shard $i --nshard 8 --pixbg arena --extra-lags 60 120 --orbit --orbit-lag 60 --keep $KEEP" \
        sbatch $G --mem=48G --time=01:00:00 --job-name=mabe_cenc_tr$i scripts/benchmark/mabe22/eci.sh
    done
    for i in $(seq 0 23); do
      PY=eci_encode EXTRA="--set eval --shard $i --nshard 24 --pixbg arena --extra-lags 60 120 --orbit --orbit-lag 60 --keep $KEEP" \
        sbatch $G --mem=48G --time=01:00:00 --job-name=mabe_cenc_ev$i scripts/benchmark/mabe22/eci.sh
    done ;;
  train)
    for a in A0 D2 X_D4 A7 X_A7D2; do
      for S in "100 101 102" "103 104 105"; do
        MABE_RUN=results/vision/benchmark_mabe22/eci/confirm PY=eci_train EXTRA="--arm $a --seeds $S" \
          sbatch $G --mem=64G --time=01:00:00 --job-name=mabe_conf_$a scripts/benchmark/mabe22/eci.sh
      done
    done ;;
  *) echo "usage: $0 encode | train"; exit 1 ;;
esac
