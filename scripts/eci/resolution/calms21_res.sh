#!/bin/bash
#
# CalMS21 resolution test, EXTRA TOKEN GRIDS for DINOv3-L/16 block 20 (extends scripts/benchmark/calms21/eci_res.sh).
# Submit-only launcher; every job runs scripts/eci/resolution/calms21_res_job.sh -> calms21_res.py (thin wrapper over
# the unchanged calms21 code: eci_encode / eci_train / eci_res_score, patched in memory; see its docstring).
#
# ======================================================================================================================
# PRE-REGISTRATION, written 2026-10-19 BEFORE any token store, SAE or score of the two new arms existed. Already seen:
# every result of eci_res.sh (res.md / cost.md), incl. D3L-32 (d3l_512_L20) primary AUROC 0.8509 +- 0.0069 SE and
# D3L-64 (d3l_1024_L20) primary AUROC 0.8813 +- 0.0091 SE (6 SAE seeds each). Those two arms are NOT rerun: their
# existing per-seed eval JSONs (results/vision/benchmark_calms21/eval/<config>_s<seed>.json, the files eci_res_score.py
# reads) are used as they are.
#
# NEW ARMS (DINOv3-L/16 facebook/dinov3-vitl16-pretrain-lvd1689m, block 20 after the final LayerNorm, 1024-d):
#   D3L-48 = d3l_768_L20   768 px input, 48 x 48 grid, 7.5 patches per body length (body 160 px native): real detail,
#                          the 1024 canvas downscaled (bicubic, antialias)
#   D3L-80 = d3l_1280_L20  1280 px input, 80 x 80 grid, 12.5 patches per body length: UPSAMPLED beyond the native 1024
#                          canvas (bicubic, no antialias) - tests TOKEN DENSITY, not image detail
# SAME as the old arms: frames (plan/train_frames.parquet, plan/eval_frames.parquet), mask cells (pass A's 32-grid
#   pos32, --reuse-mask; a G-grid patch is kept iff its 32-grid parent floor((g + 0.5) * 32 / G) per axis is kept,
#   identical to the old rule at G = 64), fixed recipe (plain BatchTopK d 4096, k 16, theta*, 6000 steps x 4096
#   tokens, x = [z_t, z_t - z_{t-60}] (Delta 2 s), frame value = max over kept tokens), SAE seeds 0 1 2 3 4 5,
#   cross-fitting (video halves, split seeds 0 and 1).
# PRIMARY METRIC (unchanged): primary AUROC = mean over attack / investigation / mount of the cross-fitted best-neuron
#   AUROC (AUROC-picked neuron), each label averaged over split seeds 0 and 1; one value per SAE seed.
#   Contrast A vs B: delta = mean(A) - mean(B) over the 6 vs 6 per-seed values, Welch 95% CI.
#     WIN  delta >= +0.010 AND CI excludes 0;   LOSES  delta <= -0.010 AND CI excludes 0;   else NO DIFFERENCE.
# QUESTIONS:
#   (e1) adjacent grids for D3L, same rule: D3L-48 vs D3L-32, D3L-64 vs D3L-48, D3L-80 vs D3L-64 (WIN = the finer grid
#        helps).
#   (e2) CalMS21 grid choice = the SMALLEST grid among 32 / 48 / 64 / 80 whose primary AUROC does NOT 'LOSE' (rule
#        above) against the grid with the best mean primary AUROC (the best grid itself qualifies).
# ALWAYS REPORTED: per-label cf AUROC, encoder tokens per frame (2 x (G^2 + 5) for t and t-60), SAE tokens per frame
#   after the mask, encode frames per GPU-second, GPU-hours (sacct elapsed x 1 GPU).
# BUDGET ~6 GPU-hours. New token stores on BeeGFS (TOK = /mnt/beegfs/locatgrp/shared/rcadei/eci_tok/calms21), DELETED
#   after the SAEs are trained and scored; SAE checkpoints / eval JSONs kept (results/vision/benchmark_calms21/{sae,
#   eval}/d3l_{768,1280}_L20_s*, new names). Summary: results/vision/eci_resolution/calms21/{res.json, res.md}.
# ======================================================================================================================
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
X="--exclude=gpu240,gpu241,gpu242,gpu150,gpu230"  # gpu240-242 have no BeeGFS mount
BIG='L40S|A100|A40'
LOGS=results/vision/eci_resolution/calms21/logs
mkdir -p $LOGS
sub() { sbatch --parsable -p gpu --gres=gpu:1 $X --constraint="$BIG" --output=$LOGS/%x_%j.out "$@" scripts/eci/resolution/calms21_res_job.sh; }
NS=6
CFG="d3l_768_L20 d3l_1280_L20"
SMOKE=/mnt/beegfs/locatgrp/shared/rcadei/eci_tok/calms21_res_smoke
case "${1:-}" in
  smoke)   # scratch TOK / OUT (never the real stores): 2 videos x 16 frames of eval shard 0, both configs
    CALMS_TOK=$SMOKE/tok CALMS_OUT=$SMOKE/out MODE=encode \
      EXTRA="--set eval --shard 0 --nshard $NS --bf16 --configs $CFG --reuse-mask --max-per-video 16 --workers 2 --threads 3" \
      sub --cpus-per-task=8 --mem=48G --time=00:30:00 --job-name=res_calms_smoke ;;
  encode)  # one job per set, all 6 shards, both configs (decode once). bash calms21_res.sh encode <train|eval>
    MODE=encode EXTRA="--set $2 --shards 0 1 2 3 4 5 --nshard $NS --bf16 --configs $CFG --reuse-mask --workers 4 --threads 3" \
      sub ${DEP:+--dependency=afterok:$DEP} --cpus-per-task=16 --mem=128G --time=04:00:00 --job-name=res_calms_enc_$2 ;;
  train)   # bash calms21_res.sh train <config> <dependency job ids, colon separated>
    C=$2; D=${3:-}
    MODE=train EXTRA="--config $C --seeds 0 1 2 3 4 5" sub ${D:+--dependency=afterok:$D} \
      --cpus-per-task=8 --mem=180G --time=04:00:00 --job-name=res_calms_tr_$C ;;
  score)   # login safe
    module load conda; conda activate crl; python scripts/eci/resolution/calms21_res.py score ;;
  *) echo "usage: $0 smoke | encode <set> | train <config> [deps] | score"; exit 1 ;;
esac
