#!/bin/bash
#
# MABe22 mouse ECI - ROUND 3: follow-ups and untested assumptions / priors of the current recipe R (submit-only launcher;
# every job runs scripts/benchmark/mabe22/eci.sh). PRE-REGISTERED RULES, written and committed 2026-10-15 ~08:40 CEST
# BEFORE any round-3 job was submitted. The rules below are FROZEN: commands may be added further down this file later,
# the rules may not change. Anything added after a round-3 number was seen is labelled EXPLORATORY.
# WORDING CLARIFICATIONS (coordinator, ~08:50 CEST, before any round-3 score; no rule changed): every number is named
# by its metric ("primary AUROC +0.030", "oral_oral AP 1.5x"); E3 tests TOKEN DENSITY (patches per mouse body
# length), not image resolution - MABe22 frames are 224 px at the source, so upscaling adds tokens, not pixel detail.
#
# ======================================================================================================================
# R (the control, decided by the confirmation run, eci_confirm.sh / eci_confirm.py, arm X_A7D2):
#   224 px frames bicubic-upscaled to 448; DINOv2-base (fp32, TF32 off) patch tokens averaged over the 8 D4 transforms
#   ("orbit"); MABe-fixed animal mask (pixbg 'arena'); token input x = [z_t, z_t - z_{t-60}] (lag 60 frames = 2 s at
#   30 fps, clipped to frame 0); Matryoshka BatchTopK d 4096, k 16, prefixes 512/1024/2048/4096, 6000 steps x 4096
#   tokens, theta* calibrated on the held-out userTrain calibration frames; frame value = max over the kept tokens.
#
# DEFAULT RULE (every comparison unless an experiment says otherwise):
#   primary  = mean cross-fitted AUROC over oral_oral, oral_genital, approaches, huddles (each label's cf AUROC averaged
#              over the 2 split seeds first; read-out = the full width of the dictionary unless named otherwise).
#   seeds    = 6 FRESH SAE seeds per arm, 200-205 (never used before); the matched-pool control of E2 additionally uses
#              210-215, 220-225, 230-235. Same train pool (plan/train_frames.parquet: 13 train + 3 calib frames per
#              userTrain clip), same eval frames (plan/eval_frames.parquet, 18,496 frames, IPW weights), split seeds 0, 1.
#   control  = R trained in THIS round on the same round-3 token store with the same seeds 200-205.
#   delta    = primary-AUROC delta = mean(arm per-seed primaries) - mean(control per-seed primaries); 95% CI = Welch t interval (Welch-
#              Satterthwaite df) on the 6 vs 6 per-seed primary means (eci_confirm.welch).
#   WIN        iff primary-AUROC delta >= +0.010 AND the CI excludes 0 AND cf AP (mean over seeds x splits) on oral_oral and on
#              oral_genital are each >= 0.9 x the control's.
#   EQUIVALENT iff |primary-AUROC delta| < 0.010 OR the CI contains 0. When equivalent, the SIMPLER option is preferred.
#   LOSES      iff primary-AUROC delta <= -0.010 AND the CI excludes 0.
#   (delta >= +0.010 with a CI excluding 0 but an AP guard failing = "not a win (AP guard)".)
#   Always reported: per-label cf AUROC / AP (oral_ear and the secondary labels too) and the honest top-1% (top-1%-picked
#   neuron among neurons firing on >= 1% of the selection half). Sniffing-neuron evidence = any honest top-1% >= 0.25 on
#   oral_oral or oral_genital.
#
# HARD CONSTRAINTS: no labels in any representation choice. SAEs train only on userTrain clips; theta* on held-out
# userTrain tokens; evaluation on submission clips only, cross-fitted at the clip level. Keypoints only for checks and
# references, never as features.
#
# ONE shared round-3 token store: results/vision/benchmark_mabe22/eci/tok_r3/ (<= ~250 GB; deleted only after every
# experiment is scored, the deletion written down). Main mouse store (train + eval plan frames): mask 'arena'
# recomputed; orbit 448 tokens of t and of t-15, t-30, t-45, t-60, t-120 (frames; 0.5 / 1 / 1.5 / 2 / 4 s), plain 448
# tokens of t and t-60 (for a no-orbit fallback), all at t's mask positions.
#
# ----------------------------------------------------------------------------------------------------------------------
# E1  WIDTH. Arms: R (d 4096; its 512 prefix read out for free, plus 1024 / 2048); R512 = d 512 trained directly,
#     Matryoshka prefixes 64/128/256/512, k 16; R512plain = d 512, no Matryoshka (one prefix 512), k 16.
#     Contrasts (default rule): R512 vs R; R512plain vs R; R512plain vs R512 (does Matryoshka matter at 512; if
#     equivalent the plain one is simpler). R's 512 prefix vs R and the 64 / 128 / 256 prefixes of R512 are reported
#     descriptively with the same delta / CI (how coarse it can go).
#     Simplicity rule: if R512 is equivalent to R (or wins), R512 is preferred.
#     WIDTH DECISION FOR E2: W = 1024 (prefixes 128/256/512/1024) if R512 is equivalent to R or wins ("width does not
#     matter"), else W = 4096 (prefixes 512/1024/2048/4096). All E2 dictionaries use W; the E2 control is R at width W
#     (= R itself when W = 4096), seeds 200-205.
#
# E2  TIME SCALES (orbit tokens, width W, k 16, Matryoshka, theta*).
#     Per-lag dictionaries x = [z_t, z_t - z_{t-L}], L = 15, 30, 60, 120 frames (0.5 / 1 / 2 / 4 s; L = 60 is R_W),
#     seeds 200-205 each.
#     "per-lag union", run j (j = 0..5): the 4 per-lag dictionaries of seed 200+j, their eval codes concatenated
#       (4W neurons); the usual cross-fit picks the best neuron from the pooled set (selection on one clip half,
#       scored on the other, both directions averaged, 2 split seeds).
#     "matched pool", run j: 4 R_W dictionaries with seeds 200+j, 210+j, 220+j, 230+j pooled the same way.
#     UNION WINS iff it beats the matched pool under the default rule (6 vs 6 runs, Welch CI, AP guards vs the matched
#     pool). Also reported: union vs R_W, and per behaviour the lag the selected neuron comes from (AUROC pick and AP
#     pick, counted over 6 runs x 2 splits x 2 directions).
#     "context-mean": x = [z_t, mean(z_{t-15}, z_{t-30}, z_{t-45}, z_{t-60})] (the 4 frames of the last 2 s at 0.5 s
#       steps, t itself excluded), vs R_W, default rule (tests "change" form vs "context" alone).
#     "multi-lag concat": x = [z_t, z_t - z_{t-30}, z_t - z_{t-60}, z_t - z_{t-120}] (3,072-d, TokenNorm per 768 block),
#       vs R_W, default rule.
#
# E3  TOKEN DENSITY (prior "same number of tokens per body length"; NOT image resolution: the source is 224 px). Input sizes 336 / 448 (R) / 672 px, all bicubic-
#     upscaled from 224 (grids 24^2 / 32^2 / 48^2). Mask: the round-3 448 mask transferred by patch centre (a patch of
#     the S grid is kept iff its centre lies inside a kept 448 patch). Orbit tokens of t and t-60 at each size; arms
#     R336 and R672 vs R (448), default rule, same SAE recipe and steps.
#     ORBIT DECISION (by cost only, fixed before any E3 number): with orbit iff the measured 672-px orbit throughput
#     projects the E3 encoding at <= 6 GPU-hours; otherwise E3 runs WITHOUT orbit (plain tokens, 2 s delta) at
#     336 / 448 / 672 with a matched no-orbit 448 control, and this is stated.
#     Reported: the mouse length in patches per size = median over the eval frames and mice of the keypoint chain
#     nose -> neck -> centre back -> tail base (keypoints 512-px scale, (y, x)) x S / 512 / 14; the cost (GPU-s per frame
#     and the job GPU-hours). If R672 wins, the implied "patches per body length" rule for the lab domains is noted.
#
# E4  ETHOGRAM (assumption "time spent, bouts and latency summarise a video"). R codes = the 6 CONFIRMED R
#     dictionaries (confirm/sae/X_A7D2_s100..s105, their own theta* from confirm/eval), applied to full-clip encodings
#     of the 1,828 submission clips: 5 fps grid label index i = 0, 6, ..., 1794 (video frame i + 1, lag t-60 clipped to
#     frame 0), arena mask per frame (bg / threshold from the clip's 30 background frames as in eci_encode.clip_mask).
#     Firing rule: neuron code > 0 under theta*.
#     Per clip and neuron: time budget = fraction of grid frames firing; bout rate = number of maximal runs of
#     consecutive firing grid frames per minute; latency = seconds to the first firing grid frame (60 s if never).
#     Annotation per clip and behaviour: time budget = fraction of the 1,800 label frames positive; bout rate = maximal
#     positive runs of the 30 fps labels per minute.
#     Cross-fit at the clip level (halves from eci_train.halves, split seeds 0 and 1): on half A pick the neuron with the
#     largest Spearman rho between its per-clip time budget and the annotated time budget (eligible: fires in >= 10
#     clips of half A); report its rho on half B; both directions averaged; then over split seeds; mean +- sd over the
#     6 SAE seeds. Same for bout rate (neuron bout rate vs annotated bout rate). All 9 behaviours reported.
#     Condition decoding (strain, lights; both binary and constant per clip): features (a) the ethogram vector (time
#     budget, bout rate, latency of every neuron: 3 x 4096) and (b) the mean frame code per clip (4096); standardised,
#     L2 logistic regression, C from {1e-3, 1e-2, 1e-1, 1} by inner 3-fold CV, outer 5-fold stratified CV with one row
#     per clip (clip-grouped); per SAE seed; reported: macro F1, AUROC, chance (majority-class macro F1; AUROC 0.5; and a
#     label-permutation run).
#     Frame rate: on a subset = 300 random clips (rng 0) + every clip with an oral_oral or oral_genital positive label
#     frame, the summaries are recomputed at 15 fps (i = 0, 2, ..., 1798, lag t-60) and the same cross-fitted
#     correlations computed at 5 fps and at 15 fps on the same subset (and the 5 vs 15 fps rank agreement of each
#     selected neuron's summary). Pre-registered reading: "5 fps is enough" for a summary iff |rho_15 - rho_5| < 0.05 on
#     every primary behaviour (mean over seeds).
#     PRECISION (by cost only): fp32 TF32-off as R if the measured throughput projects E4 at <= 14 GPU-hours; otherwise
#     bf16 autocast for E4's encoder only, with the firing-bit agreement against fp32 codes at the eval-plan frames
#     reported.
#
# E5  PER-DOMAIN vs SHARED DICTIONARY. MABe22 beetle userTrain clips encoded with the same pipeline; the mask adapted
#     label-free to pale animals on a dark floor (checked against the beetle keypoints and shown on a mask sheet; the
#     variant is fixed from the sheet / keypoint recall before any mouse score of the shared dictionary exists).
#     Shared dictionary = R recipe (d 4096, k 16, Matryoshka, 6000 steps x 4096) trained on mouse + beetle tokens with
#     EQUAL token counts (all mouse train-pool tokens + the same number of beetle tokens, sampled uniformly from beetle
#     userTrain frames); theta* calibrated on mouse calib tokens + an equal number of held-out beetle tokens. Scored on
#     the mouse labels vs R, default rule. SHARED IS PREFERRED IF EQUIVALENT (or better). Beetle clip-level check only if
#     cheap, on the corrected beetle label rows (inventory/row_mapping_check.json).
#
# E6  FROZEN vs ADAPTED FEATURES. DINOv2-base adapted self-supervised on unlabelled MABe mouse userTrain frames only
#     (no labels; the repo's scripts/mice_behavior/ssl_dapt.py objective: masked patch-feature prediction against an
#     EMA teacher, last 2 blocks unfrozen, 448 px input), fixed budget ~2 GPU-hours. Then R on the adapted encoder
#     (orbit tokens of t and t-60 at the round-3 mask positions, masks unchanged), seeds 200-205, vs R, default rule.
#     If equivalent, the frozen encoder is preferred (simpler). Skipped (and said why) if it cannot be done cleanly.
#
# Priority E1 > E2 > E3 > E4 > E5 > E6; dropped from the end if the ~40 GPU-hour budget runs out.
# GPU jobs: -p gpu --exclude=gpu242,gpu150,gpu230,gpu241 (never gpu100); CPU jobs: -p defaultp.
# ======================================================================================================================
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
echo "commands are added below as the round-3 scripts land"; exit 1
