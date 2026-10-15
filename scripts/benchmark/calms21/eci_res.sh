#!/bin/bash
#
# CalMS21 ECI - RESOLUTION x ENCODER TEST. Does a higher input resolution help single-neuron alignment, and does DINOv3
# (RoPE, any input size) beat DINOv2 at high resolution? Submit-only launcher; every job runs scripts/benchmark/calms21/
# eci.sh. PRE-REGISTERED RULES, written and committed 2026-10-15 ~12:10 CEST AFTER the inventory and the frame plans
# and BEFORE any token store, SAE or probe job of this test (only the inventory, the frame plans, the mask QA and smoke
# tests ran before). The rules below are FROZEN; commands may be added further down later; anything added after an SAE
# score of this test was seen is labelled EXPLORATORY.
#
# ======================================================================================================================
# DATA (inventory: results/vision/benchmark_calms21/inventory/inventory.md)
#   CalMS21 task 1 MP4 videos (the only videos in the record): 70 train / 19 test, 1024 x 570, 30 fps, mpeg4.
#   Label / keypoint index i = video frame i (offset 0, from the grey level under moving keypoints of both mice).
#   SAE pool, LABEL-FREE: task1 TRAIN videos with their labels ignored (unlabeled_videos.zip holds no video):
#     plan/train_frames.parquet = 10,000 train + 2,000 theta*-calibration frames, uniform over the 5 fps grid of 69
#     videos (rng 0). Scoring ONLY on the held-out task1 TEST videos: plan/eval_frames.parquet = 10,000 frames of the
#     19 test videos, stratified (attack-near 2,500 / mount-near 2,500 / investigation 2,500 / rest 2,500) with
#     inverse-probability weights w (weighted base rates = population rates).
#   SCORED LABELS (pre-registered from the inventory, before any SAE result): attack, investigation, mount
#     (task 1, dense frame-level, one expert; positives in 5 / 19 / 14 of the 19 test videos).
#     NOT scored: 'other' (the complement); the task-3 labels: the inventory found the task-3 sequences are cut from
#     task1 videos and almost all lie on task1 TRAIN videos (= the SAE pool; sniff_face 50 / 50 sequences, approach
#     28 / 28); on test videos only groom (5), intromission (3), mount_attempt (6) sequences - too few videos for
#     video-level cross-fitting. Task-2 sequences come from no task1 video (no video exists for them).
#   Cross-fitting at the VIDEO level: 2 halves x split seeds 0, 1 (videos with positives of each scored label dealt
#     alternately, rarest first, the rest at random), best neuron on one half, scored on the other, both directions.
#
# INPUT GEOMETRY: each 1024 x 570 frame is PADDED (no crop, no distortion) to a 1024 x 1024 canvas, 227 rows above and
#   below in the ImageNet mean colour; then resized (bicubic, antialias when downscaling) to
#     grid 32 x 32: DINOv2-B/14 448 px, DINOv3-B/16 512 px, V-JEPA 2-L 512 px   (1 patch = 32 x 32 canvas px)
#     grid 64 x 64: DINOv2-B/14 896 px, DINOv3-B/16 1024 px (native width), V-JEPA 2-L 1024 px (16 x 16 canvas px)
#   Models: facebook/dinov2-base, facebook/dinov3-vitb16-pretrain-lvd1689m (web, 768-d), facebook/vjepa2-vitl-fpc64-256
#   (encoder only; crop_size = S so its 3D RoPE uses integer grid positions 0..G-1 = the lab convention of
#   scripts/eci/vjepa.py; clip = 16 frames at a stride of 8 video frames, t at clip position 7, token = tubelet 3 =
#   frames (t-8, t)). Tokens = last block after the final LayerNorm, prefix tokens dropped; bf16 autocast, SDPA.
#
# MASK (label-free; keypoints used only to CHECK it): on the 32 x 32 canvas grid, per video:
#   token cue (DINOv2-448 cosine distance to the video's per-position background token of 32 evenly spaced frames >
#   the FG_RULE video threshold, isolated cells dropped) OR pixel cue (> 15% of the cell's pixels) -> 1-cell dilation;
#   pure-padding cells never kept. The SAME cells are used at every grid (a 64-grid patch is kept iff its 32-grid
#   parent is). Pixel-cue variant: chosen by the mask QA (eci_encode.py --mode maskqa, 8 videos x 24 frames, before any
#   token store) by THIS criterion, fixed now: the variant with the highest mean keypoint recall (resident and
#   intruder averaged, keypoints with score > 0.5) among the variants keeping <= 25% of the video cells; a variant
#   within 0.01 recall of a simpler one loses to it (order: dark, dark_bright, absdiff, absdiff_floor). The choice is
#   written to plan/mask_rule.txt and read by every encode job.
#
# FIXED RECIPE (all arms): plain BatchTopK SAE (one prefix, no Matryoshka), d = 4096, k = 16, theta* on the held-out
#   calibration tokens, 6000 steps x 4096 tokens, frame value = max over the frame's kept tokens, no rotation averaging.
#   DINO arms x = [z_t, z_t - z_{t-60}] (Delta = 2 s); V-JEPA arms x = z_t only (its tubelets carry time).
#   6 SAE seeds per arm: 0 1 2 3 4 5. Code: eci_encode.py, eci_train.py, eci_probe.py, eci_res_score.py.
#
# ARMS: D2-32 (d2_448), D2-64 (d2_896), D3-32 (d3_512), D3-64 (d3_1024), VJ-32 (vj_512), VJ-64 (vj_1024).
#
# PRIMARY METRIC: "primary AUROC" = mean over attack, investigation, mount of the cross-fitted best-neuron AUROC
#   (AUROC-picked neuron), each label averaged over split seeds 0 and 1 first; one value per SAE seed.
#   Contrast A vs B: delta = mean(A per-seed) - mean(B per-seed), Welch 95% CI on the 6 vs 6 per-seed values.
#     WIN  delta >= +0.010 AND CI excludes 0;   LOSES  delta <= -0.010 AND CI excludes 0;   else NO DIFFERENCE.
# PRE-REGISTERED QUESTIONS:
#   (a) does 64 beat 32 within each encoder?  D2-64 vs D2-32, D3-64 vs D3-32, VJ-64 vs VJ-32 (WIN = yes)
#   (b) at 64, does DINOv3 beat DINOv2?       D3-64 vs D2-64 (WIN = yes)
#   (c) does V-JEPA 2's ranking hold at 64?   verdicts VJ vs D2 and VJ vs D3 at 32 and at 64; HOLDS iff both
#       verdicts at 64 equal those at 32.
#   Always reported: per-label cf AUROC / AP (AP-picked) / honest top-1% (top-1%-picked among neurons firing on >= 1%
#   of the selection half's weighted frames) and the weighted firing rate of the AUROC pick.
# DIAGNOSTIC, NOT A SELECTION CRITERION: supervised ceiling per arm (eci_probe.py = diag_mice_patches.cross_validate,
#   mean_lin / mil_ctx, 5 video-grouped folds, 3 MIL seeds).
# THROUGHPUT: frames / s and GPU utilisation per job type (nvidia-smi every 60 s); random-input bench of every config
#   plus the 2064-equivalent stress grid DINOv3 129 x 129 (2064 px; throughput / memory only, no SAE).
# BUDGET: ~35 GPU-hours; token stores (BeeGFS /mnt/beegfs/locatgrp/shared/rcadei/eci_tok/calms21, a few large
#   shards) deleted after scoring; SAE checkpoints and JSONs kept.
# GPU jobs: -p gpu --exclude=gpu242,gpu150,gpu230,gpu241 (never gpu100) with a GPU-type constraint; CPU: -p defaultp.
# ======================================================================================================================
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
X="--exclude=gpu242,gpu150,gpu230,gpu241"
BIG='L40S|A100|A40'
ANY='L40S|A100|A40|A10|RTX3090Ti'
LOGS=results/vision/benchmark_calms21/logs
mkdir -p $LOGS
sub() { sbatch --parsable -p gpu --gres=gpu:1 $X --output=$LOGS/%x_%j.out "$@" scripts/benchmark/calms21/eci.sh; }
NS=6
A_CFG="d2_448 d2_896 d3_512 d3_1024 vj_512"
case "${1:-}" in
  encode)  # pass A (mask + DINO 32 / 64 + V-JEPA 32) and pass B (V-JEPA 64, reuses pass A's mask), 6 + 6 shards each
    HOLD=${HOLD:-}
    for S in train eval; do
      for i in $(seq 0 $((NS - 1))); do
        a=$(PY=eci_encode EXTRA="--set $S --shard $i --nshard $NS --bf16 --configs $A_CFG --mask file --workers 4 --threads 3" \
            sub $HOLD --constraint="$ANY" --cpus-per-task=16 --mem=96G --time=01:30:00 --job-name=encA_${S}$i)
        b=$(PY=eci_encode EXTRA="--set $S --shard $i --nshard $NS --bf16 --configs vj_1024 --reuse-mask --workers 4 --threads 3 --vj-bs 1" \
            sub $HOLD --dependency=afterok:$a --constraint="$BIG" --cpus-per-task=16 --mem=128G --time=03:00:00 --job-name=encB_${S}$i)
        echo "$S $i A $a B $b"
      done
    done ;;
  train)   # bash eci_res.sh train <config> <dependency job ids, colon separated or empty>: one job, 6 seeds
    C=$2; D=${3:-}
    PY=eci_train EXTRA="--config $C --seeds 0 1 2 3 4 5" sub ${D:+--dependency=afterok:$D} --constraint="$BIG" \
      --cpus-per-task=8 --mem=160G --time=02:00:00 --job-name=tr_$C ;;
  probe)   # bash eci_res.sh probe <config> [deps]
    C=$2; D=${3:-}
    PY=eci_probe EXTRA="--config $C" sub ${D:+--dependency=afterok:$D} --constraint="$BIG" --cpus-per-task=8 \
      --mem=128G --time=03:00:00 --job-name=pr_$C ;;
  *) echo "usage: $0 encode | train <config> [deps] | probe <config> [deps]"; exit 1 ;;
esac
