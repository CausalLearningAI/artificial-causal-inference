#!/bin/bash
#
# CRIM13 ECI - SINGLE-NEURON ALIGNMENT WITH PAIR BEHAVIOURS (sniffing above all). Submit-only launcher; every job runs
# scripts/benchmark/crim13/align_job.sh. PRE-REGISTERED RULES, written and committed 2026-10-15 ~20:00 CEST AFTER the
# inventory (results/vision/benchmark_crim13/inventory/) and the frame plan (align_plan.py, job 854981) and BEFORE any
# token store or SAE job of this test (only the plan job had run; a truncation check and a 2-video smoke encode were
# queued). The rules below are FROZEN; commands may be added further down later; anything added after an SAE score of
# this test was seen is labelled EXPLORATORY.
#
# ======================================================================================================================
# DATA (inventory.md): CRIM13 top view, 236 videos (103 released train / 133 released test), 640 x 480 (151) or
#   656 x 490 (85), 25 fps, 8-bit grey JPEG in Norpix .seq. Labels: S1 stream of <name>_t.txt, dense and mutually
#   exclusive; label frame L = 0-based frame L - 1 (offset +0). Residents A22 / A24 / A25 / A29 appear in both halves.
# SCORED BEHAVIOURS (the inventory's rule-passing set): approach, attack, copulation, chase, circle, drink, eat, clean,
#   sniff, up, walk_away. NOT scored: other (catch-all), human (experimenter's hand). One-vs-rest per frame.
# FRAMES (plan/frames.parquet, rng 0): 5 fps grid (v % 5 == 0, v >= 50). Per half: SAE pool = 10,000 'train' + 2,000
#   'calib' (theta*) frames uniform over the half's grid, label-free, disjoint from the half's eval frames; eval =
#   15,000 frames stratified by the frame's own label (sniff 1,500, every other scored behaviour 1,000 capped at the
#   stratum size, rest = other + human the remainder) with inverse-probability weights w (weighted base rates =
#   grid rates). Train half: sniff weighted base rate 0.1377; test half 0.1514.
#
# INPUT GEOMETRY: about 10 patches per body length, never above native: the resident is ~100-131 px long (bodylen.py
#   second pass median 131 px), native 640 px is the cap. Each frame is PADDED (no crop) to a W x W square in the
#   ImageNet mean colour; the 656 x 490 videos are padded to a 656 square and that canvas is RESIZED to 640 (bicubic,
#   antialias, x 0.976); every encoder sees the same 640 x 640 canvas.
#     C1 / C2 / C4: DINOv3-L/16 (facebook/dinov3-vitl16-pretrain-lvd1689m) at 640 px = 40 x 40 grid (8.2 patches per
#                   131-px body length), block 20 of 24 after the final LayerNorm (d3l_640_L20)
#     C3:           DINOv2-L/14 (facebook/dinov2-large) at 560 px = 40 x 40 grid, block 20 of 24 (d2l_560_L20)
#   CLS and registers dropped, bf16 autocast + SDPA; the blocks after 20 are removed (align_encode.py --mode check
#   measures the cosine to the hooked full model).
# MASK (label-free; the CalMS21 'dark' rule, NOT changed after seeing any score): on the 40 x 40 grid of the 640
#   canvas, per video: token cue (DINOv2-B/14 at 560 px, last layer; cosine distance to the video's per-position
#   background token of 32 evenly spaced frames > the FG_RULE video threshold, isolated cells dropped) OR pixel cue
#   (> 15% of the cell's pixels with grey < 60 and > 40 below the per-pixel median background) -> 1-cell dilation;
#   pure-padding cells never kept. The SAME cells for every arm. CHECK (never used to build the mask): per plan frame,
#   is the released track point of the resident (track row 1) / intruder (row 2) in a kept cell; reported per animal,
#   the intruder split by strain (BC = pale BALB/c, C57 = dark), plus a mask sheet (align_sheets.py).
# INPUT x: [z_t, z_t - z_{t-50}] (Delta 2 s at 25 fps) for C1 / C3 / C4; z_t only for C2.
# FIXED SAE RECIPE: plain BatchTopK (no Matryoshka) d 4096, k 16, theta* on the held-out calib tokens, 6000 steps x
#   4096 tokens (mabe22 eci_train.train_one), frame value = max over the frame's kept tokens. 6 SAE seeds 500-505 per
#   arm and fold, all seeds of one arm in one job.
# FOLDS: the released halves. Fold A: SAE trained on the train half's pool (labels ignored), each behaviour's neuron
#   picked on the train half's eval frames (labels), scored on the test half's eval frames. Fold B: roles swapped.
#   Picks: by AUROC (primary), by AP, and honest top-1% (top-1%-precision among neurons firing on >= 1% of the
#   selection half's weighted frames). No video subsampling, so no split seeds (one frame plan, rng 0).
#
# ARMS: C1 = D3L, Delta 2 s (reference); C2 = D3L, no Delta; C3 = DINOv2-L/14 block 20 at 560 px, Delta 2 s.
# PRIMARY METRIC ("pair primary AUROC"), one value per SAE seed: for each behaviour the test AUROC of the AUROC-picked
#   neuron averaged over folds A and B (same seed), then the mean over the 7 PAIR behaviours sniff, approach, attack,
#   chase, circle, walk_away, copulation. Also reported: the all-behaviour mean (11 behaviours).
# RULE: contrast X vs C1: delta = mean(X per-seed pair primary) - mean(C1 per-seed pair primary), Welch 95% CI on the
#   6 vs 6 per-seed values. WIN  delta >= +0.010 AND CI excludes 0;  LOSES  delta <= -0.010 AND CI excludes 0;
#   else EQUIVALENT.  Contrasts: C2 vs C1 (does the 2 s difference help? C2 LOSES = it helps), C3 vs C1.
# SNIFFING BAR: C1's sniff honest top-1% (mean over seeds x folds) >= 0.25 -> PASSES. Reported with the AP lift
#   (AP / weighted base rate of the scoring half) and the honest top-1% lift (top-1% / base rate).
# PER BEHAVIOUR (every arm, mean over seeds x folds): AUROC, AP (AP-picked), AP lift, honest top-1%, top-1% lift and
#   the weighted firing rate of the AUROC-picked neuron on the scoring half.
# SENSITIVITY (DESCRIPTIVE): leave one resident out: within each fold, pick on the selection half's eval frames of the
#   other 3 residents, score on the held-out resident's eval frames of the scoring half (the label-free SAE is
#   unchanged); per held-out resident the pair primary (behaviours with no positive in that subset are skipped and
#   counted).
# CEILING (DIAGNOSTIC, never a selection criterion): supervised mil_ctx probe (scripts/eci/diag_mice_patches, MIL with
#   3 x 3 context, 3 seeds, 3000 steps) on C1's masked z_t tokens of the eval frames, folds = the released halves,
#   sniff.
# CONTACT SHEETS: the C1 model (fold, seed) whose AUROC-picked sniff neuron has the highest SELECTION-half AUROC; its
#   16 highest scoring-half eval frames, crops around the peak patch, true positives marked ->
#   results/vision/benchmark_crim13/align/sheets/ (and the same for its honest top-1% pick).
# THROUGHPUT: nvidia-smi every 60 s per GPU job, frames / s per encode job. BUDGET ~15 GPU-hours. Token stores:
#   /mnt/beegfs/locatgrp/shared/rcadei/eci_tok/crim13 (4 shards x a few large arrays), deleted after scoring;
#   SAE checkpoints and JSONs kept in results/vision/benchmark_crim13/align/.
# ======================================================================================================================
# ADDENDUM 1, 2026-10-15 19:50 CEST (coordinator), written BEFORE any token store or SAE job: CalMS21 measured that the
#   'dark' rule misses the white intruder entirely in 19.1% of frames (keypoint recall 0.643; resident 0.998). The mask
#   check reports BC and C57 intruders separately (above). EXPLORATORY arm C4 = C1 with NO mask: SAE trained on random
#   video cells, the same number per frame as the mask keeps (stored by the encode job for the pool frames, rng seeded
#   by video and frame), read out on ALL video cells of the eval frames (align_c4.py, re-encodes the eval frames).
#   C4 runs if the BC-intruder track recall (own cell) of the mask is < 0.90 and the budget allows. C4 is OUTSIDE the
#   pre-registered verdicts; it is reported next to C1, and the pair results of C1 and C4 are split by intruder type
#   (BC vs C57 videos of the scoring half; neuron pick unchanged).
# ADDENDUM 2, 2026-10-15 19:52 CEST (coordinator), BEFORE any scoring: AUROC can reward broad neurons. SECONDARY
#   metrics for every arm and behaviour, not changing any verdict: R-precision (weighted precision among the top-N
#   weighted frames, N = the weighted positive mass; ties at the cut pro rata, not capped at the firing frames) and
#   F1-max (the best F1 over all thresholds of the neuron), each (a) for the AUROC-picked neuron and (b) with the
#   neuron picked by the same metric on the selection half (cross-fitted exactly like AUROC); base rate next to each.
# ======================================================================================================================
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
X="--exclude=gpu242,gpu150,gpu230,gpu241,gpu240"
BIG='L40S|A100|A40'
LOGS=results/vision/benchmark_crim13/align/logs
mkdir -p $LOGS
sub() { sbatch --parsable -p gpu --gres=gpu:1 $X --output=$LOGS/%x_%j.out "$@" scripts/benchmark/crim13/align_job.sh; }
NS=2
case "${1:-}" in
  encode)  # 2 halves x 2 shards, decode once for all encoders
    for H in train test; do for i in $(seq 0 $((NS - 1))); do
      \
        sub --export=ALL,PY=align_encode,EXTRA="--half $H --shard $i --nshard $NS --workers 6 --threads 3" \
        --constraint="$BIG" --cpus-per-task=16 --mem=110G --time=01:30:00 --job-name=cr_enc_${H}$i
    done; done ;;
  train)   # bash eci_align.sh train <arm> <dependency job ids, colon separated or empty>: one job, 6 seeds x 2 folds
    A=$2; D=${3:-}
    sub --export=ALL,PY=align_train,EXTRA="--arm $A" ${D:+--dependency=afterok:$D} --constraint="$BIG" \
      --cpus-per-task=8 --mem=160G --time=01:45:00 --job-name=cr_tr_$A ;;
  probe)   # bash eci_align.sh probe [deps]
    D=${2:-}
    sub --export=ALL,PY=align_probe,EXTRA="" ${D:+--dependency=afterok:$D} --constraint="$BIG" --cpus-per-task=8 \
      --mem=120G --time=01:30:00 --job-name=cr_probe ;;
  c4)      # EXPLORATORY (addendum 1): bash eci_align.sh c4 [deps] -> train on random cells, then the all-cell read-out
    D=${2:-}
    t=$(sub --export=ALL,PY=align_train,EXTRA="--arm C4" ${D:+--dependency=afterok:$D} --constraint="$BIG" \
      --cpus-per-task=8 --mem=160G --time=01:30:00 --job-name=cr_tr_C4)
    sub --export=ALL,PY=align_c4,EXTRA="" --dependency=afterok:$t --constraint="$BIG" --cpus-per-task=16 \
      --mem=110G --time=02:00:00 --job-name=cr_c4_eval ;;
  sheets)  # CPU: mask sheet + sniff contact sheets. bash eci_align.sh sheets [deps]
    D=${2:-}
    sbatch --parsable -p defaultp --cpus-per-task=4 --mem=16G --time=00:30:00 --job-name=cr_sheets \
      --output=$LOGS/%x_%j.out ${D:+--dependency=afterok:$D} --export=ALL,PY=align_sheets,EXTRA="" \
      scripts/benchmark/crim13/align_job.sh ;;
  gate)    # after the encodes: C4 gate (addendum 1), sheets, score, cleanup. bash eci_align.sh gate <C1> <C2> <C3> <probe> <encode deps>
    sbatch --parsable -p defaultp --dependency=afterok:$6 --export=ALL,C1=$2,C2=$3,C3=$4,PR=$5 \
      scripts/benchmark/crim13/align_gate.sh ;;
  *) echo "usage: $0 encode | train <arm> [deps] | probe [deps] | c4 [deps] | sheets [deps]"; exit 1 ;;
esac
