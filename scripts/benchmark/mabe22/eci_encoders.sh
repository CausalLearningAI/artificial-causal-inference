#!/bin/bash
#
# MABe22 mouse ECI - ENCODER STUDY: is DINOv3 / V-JEPA 2 really worse than DINOv2 here, and can V-JEPA 2's tubelets
# replace the hand-made 2 s token difference? Submit-only launcher; every job runs scripts/benchmark/mabe22/eci_enc_job.sh.
# PRE-REGISTERED RULES, written and committed 2026-10-15 ~11:25 CEST BEFORE any screen / SAE job of this study (only
# smoke tests and the throughput benchmark ran before). The rules below are FROZEN; commands may be added further down
# later; anything added after a score of this study was seen is labelled EXPLORATORY.
#
# ======================================================================================================================
# SHARED (all arms): the round-3 frames and mask, READ-ONLY from results/vision/benchmark_mabe22/eci/tok_r3/
#   (<set>_<i>of<n>_frames.parquet, _pos448.npy: mask 'arena' at frame t on the 32 x 32 grid of the 448-px DINOv2
#   input; train = plan/train_frames.parquet 13 train + 3 calib frames per userTrain clip, eval = plan/eval_frames
#   .parquet 18,496 submission frames with IPW weights; video frame = label i + 1; lag frame = max(t - 60, 0)).
#   Inputs: the 224 frame bicubic-upscaled to S px (as R), ImageNet mean / std (the processors of all three models).
#   Mask on another grid: identical positions when the grid is 32 x 32 (DINOv3 / V-JEPA at 512 px tile the frame into
#   the same 7-source-px cells as DINOv2 at 448); 16 x 16 grid: a patch is kept iff any of its 2 x 2 child cells is kept.
#   Tokens = the chosen block's output passed through the model's FINAL LayerNorm (= last_hidden_state at the last
#   block; the get_intermediate_layers(norm=True) / out_layers convention otherwise); prefix tokens dropped (DINOv2 CLS;
#   DINOv3 CLS + 4 registers). Code: eci_enc_encode.py (encode), eci_enc_probe.py (ceiling), eci_enc_train.py (SAE),
#   eci_enc_score.py (verdicts). Outputs: results/vision/benchmark_mabe22/eci/encoders/.
#   NO ORBIT in any arm (D4 averaging is 8x the encoder cost); encoders are compared without it.
#
# 1. CEILING SCREEN (DIAGNOSTIC ONLY - says whether the information is there and at which layer).
#    Eval-frame t tokens at layers: DINOv2-B 448 px (32^2): 6 8 10 12; DINOv3-B/16 512 px (32^2): 6 8 10 12;
#    DINOv3-L/16 512 px (32^2): 12 16 20 24; V-JEPA 2-L (facebook/vjepa2-vitl-fpc64-256, trained at 256 px = 16^2
#    RoPE grid): 12 16 20 24 in three position settings - vj_256 (native 256 px, 16^2 grid), vj_512x (512 px, RoPE
#    positions 0..31 = extrapolated beyond training), vj_512i (512 px, RoPE height / width positions x 16/32 =
#    interpolated into the trained range). V-JEPA clip: 16 frames at a stride of 8 video frames (3.75 fps, nearest to
#    its 4 fps pretraining), t at clip position 7, token = tubelet 3 = frames (t-8, t); indices clamped to the clip.
#    Probes = eci_probe.py's mean_lin and mil_ctx (5 clip-grouped folds, inner val 17%, 3000 steps, seeds 0-2),
#    on oral_oral and oral_genital. CEILING SCORE = mil_ctx AUROC averaged over oral_oral and oral_genital.
#    Sanity check: DINOv2 448 L12 must reproduce the round-1 A0 probe (mil_ctx AUROC oral_oral 0.883 / oral_genital
#    0.869) within seed noise, and its tokens the round-3 store's m448_t (max abs diff at fp16 rounding).
#
# 2. SAE ARMS (R's SAE recipe: Matryoshka BatchTopK d 4096, prefixes 512/1024/2048/4096, k 16, 6000 steps x 4096
#    tokens, theta* on the held-out userTrain calibration tokens, frame value = max over kept tokens; 6 FRESH seeds
#    300-305 per arm; TokenNorm per D-dim block, D = the encoder width):
#    D2    CONTROL. DINOv2-B 448, last layer, x = [z_t, z_t - z_{t-60}] (the round-3 store's plain m448_t / m448_p60).
#    D3B   DINOv3-B/16 512, LAST layer (12; label-free default, fixed now), x = [z_t, z_t - z_{t-60}].
#    D3L   DINOv3-L/16 512, LAST layer (24), x = [z_t, z_t - z_{t-60}].
#    D3B* / D3L*   only if the best-ceiling layer of that encoder is not the last: the same arm at that layer
#          (pre-registered secondary arms; the layer choice uses eval labels, so they are flagged "label-chosen layer").
#    VJ    V-JEPA 2-L at the best-ceiling CONFIG (position setting x layer, argmax of the ceiling score over the 12
#          V-JEPA screen cells; label-chosen, flagged), x = z_t ONLY (no hand-made temporal difference).
#    VJD   the same tokens with x = [z_t, z_t - z_{t-60}] (z_{t-60} = the tubelet-3 token of the clip centred on t-60).
#    Each arm is scored against D2 with the round-3 DEFAULT RULE (eci_round3.sh): primary = mean cross-fitted AUROC over
#    oral_oral, oral_genital, approaches, huddles (each label averaged over split seeds 0, 1 first; full width 4096);
#    delta = mean(arm per-seed primaries) - mean(D2 per-seed primaries), Welch 95% CI on the 6 vs 6 per-seed means;
#      WIN        delta >= +0.010 AND CI excludes 0 AND cf AP on oral_oral and on oral_genital each >= 0.9 x D2's;
#      EQUIVALENT |delta| < 0.010 OR the CI contains 0;   LOSES   delta <= -0.010 AND CI excludes 0;
#      (delta >= +0.010, CI excluding 0, an AP guard failing = "not a win (AP guard)").
#    Always reported: per-label cf AUROC / AP (all 9 labels) and the honest top-1% (oral_oral, oral_genital).
#    Descriptive contrasts (same interval, no decision): VJD vs VJ, D3L vs D3B.
#    V-JEPA QUESTION, answered by VJ vs D2: "V-JEPA's tubelets can replace the hand-made 2 s difference" iff VJ is
#    EQUIVALENT to or WINS against D2. (VJ still uses a 4 s clip around t: the temporal prior moves into the encoder
#    input; that is stated with the answer.)
#    Orbit follow-up (only if time allows, EXPLORATORY): the D4-orbit version of a WINNING arm vs R.
#
# 3. RESOLUTION CAPACITY: facts from the model cards / papers, plus measured throughput (tokens / s, random input,
#    same GPU type L40S) at 448 / 512 and 896 / 1024 px (mode bench). MABe22 frames are 224 px at the source, so no
#    resolution effect is claimed from this benchmark.
#
# ADDENDUM 1 (coordinator request, ~11:30 CEST, after smoke tests only, before any screen job; no rule above changed):
#   STORAGE: token stores on BeeGFS /mnt/beegfs/locatgrp/shared/rcadei/eci_tok/encoders/ (NFS is 95% full); JSONs,
#   checkpoints, figures under results/vision/benchmark_mabe22/eci/encoders/. Jobs log GPU utilisation every 60 s
#   (eci_enc_job.sh) and frames / s; decoding uses parallel ffmpeg workers, each clip decoded once per job for all lags.
#   PRECISION (cost rule, fixed before the V-JEPA precision check finished): image encoders stay fp32 TF32 off as R and
#   the D2 control (bf16 autocast measured on 8 eval clips: token cosine to fp32 mean >= 0.999, 1st pct >= 0.994 at
#   every screen layer of DINOv2-B / DINOv3-B / DINOv3-L, so it would be safe, but the full image-encoder encoding is
#   < 1 GPU-hour and fp32 keeps the arms identical to the control). V-JEPA: bf16 autocast for a position setting iff on
#   the same 8 clips, at every screen layer, cosine mean >= 0.995 AND 1st percentile >= 0.98; otherwise fp32 TF32 on.
#
# Storage: token stores (<= ~150 GB), deleted after scoring; SAE checkpoints and eval JSONs kept.
# GPU jobs: -p gpu --gres=gpu:L40S:1 --exclude=gpu242,gpu150,gpu230,gpu241 (never gpu100); CPU jobs: -p defaultp.
# ======================================================================================================================
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
G="-p gpu --gres=gpu:L40S:1 --exclude=gpu242,gpu150,gpu230,gpu241 --cpus-per-task=8"
LOGS=results/vision/benchmark_mabe22/eci/encoders/logs
mkdir -p $LOGS
sub() { sbatch $G --output=$LOGS/%x_%j.out "$@" scripts/benchmark/mabe22/eci_enc_job.sh; }
enc() { N=$1; T=$2; shift 2; PY=eci_enc_encode EXTRA="$*" sub --mem=64G --time=$T --job-name=$N; }
EV="0 1 2 3 4 5 6 7 8 9 10 11"
case "${1:-}" in
  bench)
    PY=eci_enc_encode EXTRA="--mode bench --bench dinov2b:448 dinov2b:896 dinov3b:512 dinov3b:1024 dinov3l:512 dinov3l:1024 vjepal:256 vjepal:512 vjepal:1024" sub --mem=48G --time=00:30:00 --job-name=enc_bench ;;
  screen)  # eval t tokens at the screen layers (image encoders: one job each; V-JEPA 512: 4 jobs x 3 shards)
    enc sc_d2 01:00:00 --set eval --shard $EV --encoder dinov2b --size 448 --layers 6 8 10 12 --tag d2_448 --check-r3
    enc sc_d3b 01:00:00 --set eval --shard $EV --encoder dinov3b --size 512 --layers 6 8 10 12 --tag d3b_512
    enc sc_d3l 01:30:00 --set eval --shard $EV --encoder dinov3l --size 512 --layers 12 16 20 24 --tag d3l_512
    enc sc_vj256 02:00:00 --set eval --shard $EV --encoder vjepal --size 256 --layers 12 16 20 24 --tag vj_256 --bs 8
    for s in "0 1 2" "3 4 5" "6 7 8" "9 10 11"; do
      enc sc_vjx 03:00:00 --set eval --shard $s --encoder vjepal --size 512 --layers 12 16 20 24 --tag vj_512x --bs 4
      enc sc_vji 03:00:00 --set eval --shard $s --encoder vjepal --size 512 --vj-pos interp --layers 12 16 20 24 --tag vj_512i --bs 4
    done ;;
  probe)   # bash eci_encoders.sh probe <tag> <grid> <layers...>
    T=$2; GR=$3; shift 3
    PY=eci_enc_probe EXTRA="--tag $T --grid $GR --layers $*" sub --mem=96G --time=02:30:00 --job-name=pr_$T ;;
  store)   # bash eci_encoders.sh store <encoder> <size> <tag> <bs> [vj-pos] <layers...>: train t + p60, eval p60
    E=$2; S=$3; T=$4; B=$5; shift 5
    VP=""; if [ "${1:-}" = interp ] || [ "${1:-}" = extrap ]; then VP="--vj-pos $1"; shift; fi
    for s in "0 1" "2 3" "4 5" "6 7"; do enc st_${T}_tr 03:00:00 --set train --nshard 8 --shard $s --encoder $E --size $S $VP --layers $* --tag $T --bs $B --kinds t p60; done
    for s in "0 1 2" "3 4 5" "6 7 8" "9 10 11"; do enc st_${T}_ev 03:00:00 --set eval --shard $s --encoder $E --size $S $VP --layers $* --tag $T --bs $B --kinds p60; done ;;
  train)   # bash eci_encoders.sh train <arm> [tag] [layer]: 2 jobs x 3 seeds
    A=$2; T=${3:-}; L=${4:-0}
    for S in "300 301 302" "303 304 305"; do
      PY=eci_enc_train EXTRA="--arm $A ${T:+--tag $T} --layer $L --seeds $S" sub --mem=128G --time=01:30:00 --job-name=tr_${A}
    done ;;
  *) echo "usage: $0 bench | screen | probe <tag> <grid> <layers> | store ... | train <arm> [tag] [layer]"; exit 1 ;;
esac
