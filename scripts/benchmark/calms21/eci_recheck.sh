#!/bin/bash
#
# CalMS21 ECI - PHASE 2: RE-CHECK OF THE MABe22 DESIGN CHOICES at 1024-px source resolution. Submit-only launcher;
# every job runs scripts/benchmark/calms21/eci.sh. PRE-REGISTERED, written and committed 2026-10-15 ~14:00 CEST BEFORE
# any phase-2 job (phase-1 grid results were partly known: see eci_res.sh; the base is chosen MECHANICALLY below).
#
# WHY: several choices were made only on MABe22 (224-px video) or only on lab data. A choice STANDS only if both
# benchmarks agree; if they disagree it is marked UNSURE and the simpler option is kept.
#
# BASE (p2_base.py, run after the phase-1 grid is scored, writes results/vision/benchmark_calms21/p2/base.txt):
#   the arm with the highest mean primary AUROC among the eight pre-registered grid arms D2-32, D2-64, D3-32 (L10),
#   D3-64 (L10), D3L-32 (L20), D3L-64 (L20), VJ-32, VJ-64; if that arm is V-JEPA (no hand-made Delta, no image orbit),
#   the best IMAGE-encoder arm of the eight is the base. The base keeps EVERYTHING of the grid recipe (same frames,
#   mask cells, canvas, layer, bf16): plain BatchTopK d 4096, k 16, theta*, x = [z_t, z_t - z_{t-60}], frame value =
#   max over the kept tokens. Its 6 grid seeds (0-5) are the control of every contrast.
#
# ARMS (6 seeds 0-5 each, everything else = base):
#   P_D0     change window none: x = z_t only
#   P_D1     1 s: x = [z_t, z_t - z_{t-30}]
#   P_D4     4 s: x = [z_t, z_t - z_{t-120}]           (lag frames clamped to frame 0, as MABe22)
#   P_W512   plain d = 512
#   P_N512   d 512 with Matryoshka prefixes 128 / 256 / 512   (contrast vs P_W512, not vs the base)
#   P_K8     k = 8
#   P_NOMASK no animal mask: training tokens = a uniformly random subset of the frame's VIDEO cells (padding excluded)
#            of the same size as its masked count (same training-token count); eval frame value over ALL video cells
#   P_ORB    D4 rotation averaging of the encoder tokens (8 transforms of the square canvas, read back at the original
#            positions); cost reported first (eci_p2_encode.py JSONs); run because the measured encode cost (8 x base)
#            fits the 25 GPU-h phase-2 budget for every image-encoder base (D3L-64: ~4.6 GPU-h by the pass-D rate)
#   FRAME VALUE: max vs mean over the kept tokens, from the same codes, reported for EVERY arm (grid arms included,
#            rescored from their saved SAEs); contrast "mean vs max" on the base.
# RULE (round-3 rule): primary = mean cross-fitted best-neuron AUROC over attack, investigation, mount (AUROC-picked
#   neuron, split seeds 0 and 1 averaged first); delta = arm - control on the 6 vs 6 per-seed primaries, Welch 95% CI:
#     WIN delta >= +0.010 AND CI excludes 0;  LOSES delta <= -0.010 AND CI excludes 0;  otherwise EQUIVALENT.
#   Contrasts and the MABe22 verdict each one is compared with (agree = same verdict word, for the change window the
#   pairwise MABe22 statement):
#     2 s vs 1 s (base vs P_D1)           MABe22: WINS (confirmation run: D2 vs A0 (1 s) +0.030 [+0.004, +0.056])
#     none vs 1 s (P_D0 vs P_D1)          MABe22: EQUIVALENT (A4 vs A0 -0.002), but sniffing AP x0.5
#     4 s vs 1 s (P_D4 vs P_D1)           MABe22: WINS (confirmation run: D4 vs A0 +0.034 [+0.009, +0.059])
#     (descriptive, no MABe22 verdict: none vs 2 s, 4 s vs 2 s)
#     plain 512 vs 4096 (P_W512 vs base)  MABe22: EQUIVALENT
#     nested 128/256/512 vs plain 512     MABe22: EQUIVALENT (but weak)
#     k 8 vs 16 (P_K8 vs base)            MABe22: LOSES
#     no mask vs mask (P_NOMASK vs base)  MABe22: LOSES
#     orbit vs none (P_ORB vs base)       MABe22: WINS
#     mean vs max frame value (base)      MABe22: never tested (no verdict to agree with)
#   Report: choice, MABe22 verdict, CalMS21 verdict, agree yes / no; AP and honest top-1% per label; primary metric name
#   next to every number.
# MASK CAVEAT: the grid's 'dark' mask misses the white intruder entirely in 8.3% of frames (intruder keypoint recall
#   0.748). It is NOT changed here; the mask arm is reported with this caveat.
# BUDGET ~25 GPU-hours; BeeGFS writes only from L40S / A100 / A40 nodes (gpu240-242 excluded: no BeeGFS mount);
#   several pass-A shards per encode job (fewer, longer jobs); nvidia-smi utilisation logged by eci.sh.
# ADDENDUM 2026-10-15 14:20 CEST (coordinator), before P_ORB ran: "rotation averaging on top of the best LARGE arm" is
#   P_ORB itself as long as the base d3l_1024_L20 (a Large arm) is the best Large arm; MABe22 verdict for that
#   comparison: EQUIVALENT (+0.015 [-0.005, +0.035]). If the size-matched D2L-64 (eci_res.sh addendum 3) beats the base,
#   that is reported and no second orbit arm is run unless the budget allows (stated in the report).
# ======================================================================================================================
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
X="--exclude=gpu240,gpu242,gpu150,gpu230,gpu241"
BIG='L40S|A100|A40'
LOGS=results/vision/benchmark_calms21/logs
sub() { sbatch --parsable -p gpu --gres=gpu:1 $X --constraint="$BIG" --output=$LOGS/%x_%j.out "$@" scripts/benchmark/calms21/eci.sh; }
B=$(head -1 results/vision/benchmark_calms21/p2/base.txt 2>/dev/null || echo NONE)
case "${1:-}" in
  encode)  # extra tokens for the base: 1 s / 4 s lags, no-mask, orbit; 2 jobs per set x 3 pass-A shards
    for S in train eval; do for SH in "0 1 2" "3 4 5"; do
      PY=eci_p2_encode EXTRA="--set $S --shards $SH --config $B --parts lags nomask orbit" \
        sub --cpus-per-task=16 --mem=128G --time=06:00:00 --job-name=p2enc_$S${SH// /}
    done; done ;;
  train)   # bash eci_recheck.sh train <deps>: one job per arm, 6 seeds
    D=${2:-}
    s() { PY=eci_train EXTRA="--config $B --seeds 0 1 2 3 4 5 $2 --run $1_$B" sub ${D:+--dependency=afterok:$D} --cpus-per-task=8 --mem=160G --time=02:00:00 --job-name=$1; }
    s P_D0 "--lag 0"; s P_D1 "--lag 30"; s P_D4 "--lag 120"; s P_W512 "--width 512"
    s P_N512 "--width 512 --prefixes 128 256 512"; s P_K8 "--k 8"; s P_NOMASK "--tok nomask"; s P_ORB "--tok orbit" ;;
  rescore) # max vs mean for the grid arms: rerun eci_train on the saved SAEs (no retraining; 6 seeds each)
    for C in d2_448 d2_896 d3_512 d3_1024 vj_512 d3_512_L10 d3_1024_L10 d3l_512_L20 d3l_1024_L20 d2l_448 d3_1024lo_L10 d3l_1024lo_L20; do PY=eci_train EXTRA="--config $C --seeds 0 1 2 3 4 5" sub --cpus-per-task=8 --mem=160G --time=01:00:00 --job-name=rs_$C; done ;;
  *) echo "usage: $0 encode | train [deps] | rescore"; exit 1 ;;
esac
