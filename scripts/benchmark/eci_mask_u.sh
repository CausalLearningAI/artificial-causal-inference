#!/bin/bash
#
# ECI animal mask - ONE RULE "U" FOR EVERY DOMAIN vs today's per-domain variants. Submit-only launcher; every job runs
# scripts/benchmark/mask_u/job.sh. PRE-REGISTERED RULES, written and committed 2026-10-15 ~15:30 CEST BEFORE any token
# store, any SAE job and any full-set mask QA of this test. Before this commit only DEV mask QA ran (maskqa.py --pick dev:
# 6 MABe22 submission clips x <= 12 plan frames, 6 beetle clips x 16 frames, 6 CalMS21 TRAIN videos x 12 frames, all rng 7)
# to set U's parameters (dev versions v1-v4 in results/vision/mask_u/maskqa/*_dev*.json; v4 is the frozen rule). The rule
# and the decision below are FROZEN; commands may be added further down; anything added after an SAE score of this test
# was seen is labelled EXPLORATORY.
#
# ======================================================================================================================
# DECISION DATA: the two VALIDATION benchmarks only (MABe22, CalMS21). Lab data (ants, mice, frogs) = TEST set,
# descriptive only, cannot change the decision.
#
# RULE U (src/eci/mask_u.py, U_RULE; identical parameters in every domain, all fractions / quantiles; no tracker points)
#   Mask grid: the 32 x 32 grid of the DINOv2-base 448 input (cell = image side / 32 px: MABe22 7 px of 224, CalMS21 32 px
#   of the 1024 canvas, lab 16 px of 512). Per video: n sample frames evenly spaced (MABe22 30 per clip, CalMS21 32 per
#   video, as the current rules), their grey frames and DINOv2-448 last-layer tokens (the current mask encoder).
#   blur        box blur, side k = 2 floor(cell / 4) + 1 px (MABe22 3, CalMS21 17, lab 9)
#   M           per-pixel temporal median of the blurred sample frames
#   floor       Otsu split of M (valid pixels); floor class = the side holding >= 50% of the central box (middle 50% of
#               rows and columns); opened by a cell x cell square; C = largest component; F = convex hull(C);
#               f0 = median(M over C); sig_s = 1.4826 MAD(M over C)
#   thr_pix     6 x 1.4826 x max(MAD(blurred sample frames - M over F), 1 grey level)          [still_k = 6]
#   B           per time block (sample frames split into 4 consecutive blocks; a frame uses the nearest block):
#               B_b = per-pixel median of the block; "still" pixels set to f0 = pixels of F with |M_b - f0| >
#               max(thr_pix, 6 sig_s) in components of area <= 0.10 |C| AND area <= 25 r_max^2 (compact)
#   pixel cue   fraction of the cell's pixels with |blurred grey - B| > thr_pix (either sign) inside F; core if > 0.15
#   token cue   cosine distance to the per-position reference token = masked median over sample frames whose cell is
#               not within 1 cell of a pixel-cue core cell; positions with < 25% such frames filled from clean
#               8-neighbours (foreground.fill_leaks); threshold = 0.995 quantile of the sample distances > 2 cells from any
#               pixel-cue core cell; isolated token-cue cells dropped
#   mask        dilate by 1 cell onto ALL 8 neighbours; cells without a valid pixel never kept (CalMS21 padding)
# CURRENT RULES (the controls): MABe22 mouse 'arena' (mabe22 eci_encode.clip_mask --pixbg arena), MABe22 beetle 'b15hc'
#   (eci_r3_beetle.py), CalMS21 'dark' (plan/mask_rule.txt, calms21 eci_encode.Masker), lab ants 'ants' / mice 'fg448' /
#   frogs 'frogs' (src/eci/foreground.py RULES, existing per-video backgrounds).
#
# MASK QUALITY (keypoints only measure, never enter a mask):
#   MABe22 mouse: all 18,496 submission eval frames (IPW weights), keypoints 512 px (y, x), video frame = label + 1;
#   CalMS21: all 10,000 test eval frames (IPW), MARS keypoints score > 0.5, video frame = label + 0, per animal
#   (resident black, intruder white); MABe22 beetle: the round-3 beetle mask-check clips (24 clips x 16 frames, rng 1).
#   Per mask: keypoint recall per animal, frames with the animal missed, kept cells per frame (fraction of grid),
#   hull precision (kept cells within 1 cell of some animal's keypoint convex hull; beetle: kept cells within 40 px of a
#   keypoint, eci_r3_beetle.kp_far). Mask sheets in results/vision/mask_u/sheets/.
#
# SAE ARMS (6 FRESH seeds 400-405 per arm; the two arms read ONE union token store per benchmark and keep the tokens of
#   their own mask; frame value = max over the frame's kept tokens; x = [z_t, z_t - z_{t-60}] (2 s, lag clipped to 0)):
#   MABe22 (encode_mabe.py): DINOv3-L/16 block 20, 512 px from the 224 source (32^2 grid), bf16 + SDPA, no orbit;
#     SAE = the encoder study's D3L-L20 recipe (Matryoshka BatchTopK d 4096, prefixes 512/1024/2048/4096, k 16,
#     6000 steps x 4096 tokens, theta* on the userTrain calib frames); round-3 train / eval plans. Arms M_cur, M_U.
#   CalMS21 (encode_calms.py; OWN store under /mnt/beegfs/.../eci_tok/mask_u/calms, the phase-2 stores are not read):
#     the phase-2 base D3L-64 = DINOv3-L/16 block 20 at the 1024 canvas (64^2 grid), bf16; plain BatchTopK d 4096, k 16,
#     theta*; a 64-grid patch is kept iff its 32-grid parent cell is. Arms C_cur, C_U.
#
# PRE-REGISTERED DECISION (scripts/benchmark/mask_u/score.py):
#   primary AUROC = mean cross-fitted best-neuron AUROC (AUROC-picked neuron, full width), each label averaged over split
#   seeds 0, 1 first; MABe22 over oral_oral, oral_genital, approaches, huddles; CalMS21 over attack, investigation, mount.
#   delta = mean(U per-seed) - mean(cur per-seed), Welch 95% CI on 6 vs 6 per-seed primaries.
#   U is ADOPTED iff ALL hold:
#     (i)   on BOTH benchmarks: primary AUROC delta > -0.010 OR the CI contains 0
#     (ii)  MABe22 cf AP (AP-picked) of U >= 0.9 x current on oral_oral AND on oral_genital
#     (iii) keypoint recall of U >= current - 0.01 for EVERY animal on MABe22 mouse, MABe22 beetle and CalMS21
#   otherwise the current per-domain rules stay, and the failed condition(s) are reported.
# LAB (descriptive, TEST set; no SAE, no behaviour labels): a few frames of ants v2 / v3 (tracker CSV recall), mice v1
#   (kept fraction + sheet), frogs (SLEAP node recall), U vs the current rule.
#
# Storage: token stores on BeeGFS /mnt/beegfs/locatgrp/shared/rcadei/eci_tok/mask_u/ (deleted after scoring); SAE
# checkpoints and JSONs under results/vision/mask_u/. GPU: -p gpu, L40S / A100 / A40, never gpu100, exclude
# gpu240,gpu242,gpu150,gpu230,gpu241; CPU: -p defaultp. Budget ~30 GPU-hours.
# ======================================================================================================================
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
LOGS=results/vision/mask_u/logs
mkdir -p $LOGS
G="-p gpu --gres=gpu:1 --constraint=L40S|A100|A40 --exclude=gpu240,gpu242,gpu150,gpu230,gpu241"
sub() { local PYX=$1 EX=$2; shift 2; sbatch --parsable $G --output=$LOGS/%x_%j.out --export=ALL,PY=$PYX,EXTRA="$EX" "$@" scripts/benchmark/mask_u/job.sh; }
# 2026-10-15 15:55 (before any token store existed): job sizes cut so the jobs fit the scheduler's backfill (the first
# submission was estimated to start at 23:39): one shard per encode job (MABe22 train 4 / eval 6 shards, CalMS21 4 / 4), <= 30 min limits, 3 seeds per
# train job. Nothing else changed.
case "${1:-}" in
  encode_mabe)
    for SH in 0 1 2 3; do sub encode_mabe "--set train --shards $SH --nshard 4" --cpus-per-task=8 --mem=48G --time=00:25:00 --job-name=mu_enc_mtr$SH; done
    for SH in 0 1 2 3 4 5; do sub encode_mabe "--set eval --shards $SH --nshard 6" --cpus-per-task=8 --mem=48G --time=00:25:00 --job-name=mu_enc_mev$SH; done ;;
  encode_calms)
    for S in train eval; do for SH in 0 1 2 3; do
      sub encode_calms "--set $S --shards $SH --nshard 4" --cpus-per-task=8 --mem=64G --time=00:28:00 --job-name=mu_enc_c$S$SH
    done; done ;;
  beetle)
    sub maskqa "--domain beetle --pick final" --cpus-per-task=8 --mem=48G --time=00:40:00 --job-name=mu_beetle ;;
  train)  # bash eci_mask_u.sh train <mabe|calms> [dependency]: one job per arm, 6 seeds (17:05: <= 30 min, less memory, for backfill)
    B=$2; D=${3:-}
    for A in cur U; do
      sub train "--bench $B --mask $A --seeds 400 401 402 403 404 405" ${D:+--dependency=afterok:$D} --cpus-per-task=8 \
        --mem=$([ $B = calms ] && echo 80G || echo 40G) --time=00:$([ $B = calms ] && echo 30 || echo 25):00 --job-name=mu_tr_${B}_$A
    done ;;
  *) echo "usage: $0 encode_mabe | encode_calms | beetle | train <mabe|calms> [dep]"; exit 1 ;;
esac
