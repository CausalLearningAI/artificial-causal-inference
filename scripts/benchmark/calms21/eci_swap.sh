#!/bin/bash
#
# CalMS21 ECI - TWO-WAY ROLE SWAP: do single SAE neurons align with the task-3 PAIR behaviours (sniff_face = nose-to-
# nose sniffing, approach, mount_attempt, intromission)? Submit-only launcher; every job runs eci_swap_job.sh.
# PRE-REGISTERED RULES, written and committed 2026-10-15 ~20:00 CEST AFTER the frame plans (eci_swap_plan.py) and the
# label-only drop check (eci_swap_halves.py) and BEFORE any token store or SAE job of this test. Frozen; commands may
# be added below later; anything added after an SAE score of this test was seen is labelled EXPLORATORY.
# ======================================================================================================================
# WHY: the task-3 pair labels lie on the 70 task1 TRAIN videos (inventory: every matched sequence = one whole train
#   video, offset 0), which were the grid's SAE pool. Here the roles swap: SAE pool = the 19 task1 TEST videos (labels
#   never read), scoring = the 70 TRAIN videos. No eval video is ever seen by an SAE.
# DATA / PLANS (eci_swap_plan.py; results/vision/benchmark_calms21/swap/plan/swap_plan.json):
#   NOTE: store / plan SET NAMES ARE SWAPPED because the shared encoders map --set train -> train videos, anything else ->
#   test videos: plan/eval_frames.parquet + store set 'eval' = SAE POOL (test videos); plan/train_frames.parquet + store
#   set 'train' = EVAL frames (train videos).
#   SAE pool: 10,000 train + 2,000 theta*-calibration frames, uniform over the 5 fps grid of the 19 test videos (rng 0).
#   Eval: 15,519 frames of the 70 train videos (every frame with 60 <= i <= N - 65), 12 disjoint strata with known
#     inclusion probabilities (ma_near 519 = all, in_near 1500, ap_near 1500, sf_near 2500, at_near 1500, mo_near 1500,
#     inv 1500, ma_rest / in_rest / ap_rest / sf_rest 1000 each, rest 1000; 'near' = a positive within +-6 frames);
#     inverse-probability weights w (weighted base rates = population rates, checked in swap_plan.json).
#   SCORED LABELS: task-3 sniff_face (50 videos, 47 with a sampled positive), approach (28 / 28), mount_attempt
#     (10 / 10), intromission (8 / 8), each scored ONLY on the frames of its own sequences (in_<b>); task1 attack /
#     investigation / mount on all eval frames (sanity reference).
# CROSS-FITTING: video-level halves (calms21 eci_train.halves over all 7 labels: videos with positives of each label
#   dealt alternately, rarest label first, the rest at random) for split seeds 0-9; best neuron picked on one half, scored
#   on the other, both directions averaged (contact_sae.cross_fit, weighted), per label on its own frames.
#   DROP RULE (label-only, decided by eci_swap_halves.py BEFORE any SAE, plan/halves.json): a pair label is dropped from
#   the pair primary iff in some split seed a half holds < 3 source videos with a positive. Result: NONE dropped (min
#   per half over split seeds 0-9: sniff_face 20, approach 12, mount_attempt 5, intromission 4; task1 attack 8,
#   investigation 33, mount 17).
# RECIPE (all arms; the grid's): canvas pad 1024 x 1024, 'dark' mask (grid's choice, plan/mask_rule.txt, computed per
#   video with the DINOv2-448 token cue), 64-grid patch kept iff its 32-grid parent is, bf16 + SDPA, plain BatchTopK
#   d 4096, k 16, theta* on the calibration tokens, 6000 steps x 4096 tokens, frame value = max over the kept tokens.
#   SAE seeds 400-405 (6 per arm, one job per arm).
# ARMS:
#   S1  d3l_1024_L20 (DINOv3-L/16 block 20, 1024 px, 64^2 grid), x = [z_t, z_t - z_{t-60}] (2 s), no orbit (reference)
#   S2  as S1 with x = z_t only (no Delta)
#   S3  d2l_896_L20 (DINOv2-L/14 block 20, 896 px, 64^2 grid), Delta 2 s
#   S4  as S1 + orbit (D4 rotation averaging of the encoder tokens, eci_p2_encode.py --parts orbit) on ALL eval frames,
#       with a REDUCED SAE pool: the plan's orbit_pool subset of 4,000 train + 1,000 calib frames (rng 1). Cost first:
#       measured orbit rate 24.4 inputs / GPU-s (p2 encode, L40S) -> 16 inputs per frame -> eval 15,519 frames ~2.8
#       GPU-h + pool 5,000 ~0.9 GPU-h. Run only if its encode finishes by ~22:30 CEST.
#   S1r as S1 trained / calibrated on the SAME reduced pool as S4 (no extra encode) = S4's matched control.
# PRIMARY: "pair primary AUROC" = mean over the kept pair labels (sniff_face, approach, mount_attempt, intromission) of
#   the cross-fitted best-neuron AUROC (AUROC-picked), each label averaged over split seeds 0-9 first; one value per
#   SAE seed. Sanity: "task-1 primary AUROC" = the same over attack, investigation, mount.
#   Contrasts (6 vs 6 per-seed pair primaries, Welch 95% CI):
#     S2 vs S1 (does DINOv3-L need temporal context for pair behaviours?), S3 vs S1 (DINOv3 vs DINOv2, both L, 64 grid),
#     S4 vs S1r (orbit, pool-matched = the orbit verdict), S4 vs S1 (requested; confounds orbit with a smaller pool).
#     WIN  delta >= +0.010 AND CI excludes 0;   LOSES  delta <= -0.010 AND CI excludes 0;   else EQUIVALENT.
#   Also reported per label: cf AUROC (AUROC pick), cf AP (AP pick), AP lift = cf AP / weighted base rate of the label's
#   frames, honest top-1% (top-1%-picked among neurons firing on >= 1% of the selection half's weighted frames) and the
#   weighted firing rate of the AUROC pick and of the top-1% pick; task-1 contrasts reported descriptively.
#   SNIFFING BAR: a neuron "passes" on sniff_face iff the arm's mean honest top-1% on sniff_face >= 0.25.
# CONTACT SHEETS (descriptive): S1 (and S4) seed 400, the sniff_face neuron picked most often by AUROC over the 20
#   split x direction selections; its top-16 frames among the sniff_face-sequence eval frames, crop around the peak patch,
#   true positives marked -> results/vision/benchmark_calms21/swap/sheets/.
# BUDGET ~30 GPU-h (shared with the MABe22 P2 test, eci_p2.sh), <= 300 GB BeeGFS; stores in
#   /mnt/beegfs/locatgrp/shared/rcadei/eci_tok/pairpush/calms21{,_orbpool}, deleted after scoring; SAE checkpoints and
#   JSONs kept under results/vision/benchmark_calms21/swap/. GPU: -p gpu, L40S | A100 | A40, never gpu100; CPU: defaultp.
# ======================================================================================================================
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
X="--exclude=gpu240,gpu242,gpu150,gpu230,gpu241"
BIG='L40S|A100|A40'
LOGS=results/vision/benchmark_calms21/swap/logs
TOKM=/mnt/beegfs/locatgrp/shared/rcadei/eci_tok/pairpush/calms21
TOKO=/mnt/beegfs/locatgrp/shared/rcadei/eci_tok/pairpush/calms21_orbpool
mkdir -p $LOGS
sub() { sbatch --parsable -p gpu --gres=gpu:1 $X --constraint="$BIG" --output=$LOGS/%x_%j.out "$@" scripts/benchmark/calms21/eci_swap_job.sh; }
cpu() { sbatch --parsable -p defaultp --output=$LOGS/%x_%j.out "$@" scripts/benchmark/calms21/eci_swap_job.sh; }
ENC="--nshard 8 --bf16 --configs d3l_1024_L20 d2l_896_L20 --mask file --workers 4 --threads 3"
case "${1:-}" in
  plan)    PY=eci_swap_plan cpu --cpus-per-task=2 --mem=48G --time=00:30:00 --job-name=swap_plan
           PY=eci_swap_halves cpu --cpus-per-task=1 --mem=4G --time=00:10:00 --job-name=swap_halves ;;
  encode)  # base stores (mask + D3L-64 + D2L-64, t and t - 60), 2 jobs per set x 4 shards of 8
    for S in train eval; do for SH in "0 1 2 3" "4 5 6 7"; do
      PY=eci_encode EXTRA="--set $S --shards $SH $ENC" sub --cpus-per-task=16 --mem=128G --time=01:30:00 --job-name=swE_$S${SH// /}
    done; done ;;
  orbit)   # bash eci_swap.sh orbit <base encode job ids of set train, colon sep> <of set eval>: S4 tokens
    for SH in "0 1" "2 3" "4 5" "6 7"; do
      PY=eci_p2_encode EXTRA="--set train --shards $SH --nshard 8 --config d3l_1024_L20 --parts orbit" \
        sub --dependency=afterok:$2 --cpus-per-task=16 --mem=128G --time=02:00:00 --job-name=swO_train${SH// /}
    done
    o=$(PY=eci_swap_orbpool EXTRA="$TOKO" cpu --dependency=afterok:$3 --cpus-per-task=1 --mem=8G --time=00:10:00 --job-name=swO_pool)
    for SH in "0 1 2 3" "4 5 6 7"; do
      CALMS_TOK=$TOKO PY=eci_p2_encode EXTRA="--set eval --shards $SH --nshard 8 --config d3l_1024_L20 --parts orbit" \
        sub --dependency=afterok:$o --cpus-per-task=16 --mem=128G --time=02:00:00 --job-name=swO_eval${SH// /}
    done ;;
  train)   # bash eci_swap.sh train <arm> <deps>
    D=${3:-}; SEEDS="--seeds 400 401 402 403 404 405"
    case $2 in
      S1)  E="--arm S1 --config d3l_1024_L20 --lag 60 --sheet" ;;
      S2)  E="--arm S2 --config d3l_1024_L20 --lag 0" ;;
      S3)  E="--arm S3 --config d2l_896_L20 --lag 60" ;;
      S1r) E="--arm S1r --config d3l_1024_L20 --lag 60 --pool orbit" ;;
      S4)  E="--arm S4 --config d3l_1024_L20 --lag 60 --tok orbit --pool orbit --pool-tok $TOKO --sheet" ;;
      *) echo "unknown arm $2"; exit 1 ;;
    esac
    PY=eci_swap_train EXTRA="$E $SEEDS" sub ${D:+--dependency=afterok:$D} --cpus-per-task=8 --mem=160G --time=02:00:00 --job-name=swT_$2 ;;
  *) echo "usage: $0 plan | encode | orbit <train deps> <eval deps> | train <arm> [deps]"; exit 1 ;;
esac
