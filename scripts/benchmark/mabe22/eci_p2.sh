#!/bin/bash
#
# MABe22 mouse ECI - EXPERIMENT P2: does the 2 s token difference matter for sniffing on DINOv3-L?
# Submit-only launcher; every job runs scripts/benchmark/mabe22/eci_enc_job.sh (encoder-study wrapper, GPU util / 60 s).
# PRE-REGISTERED RULES, written and committed 2026-10-15 ~20:00 CEST BEFORE any P2 job (encode or SAE). FROZEN; any
# command or analysis added after a P2 score was seen is labelled EXPLORATORY.
#
# QUESTION. On DINOv2-B (round 1, eci_summary.py / OUT/table.md) removing the temporal difference halved sniffing AP:
#   A4 (z_t only) vs A0 (x = [z_t, z_t - z_{t-30}], 1 s lag), 3 graded SAE seeds: cf AP oral_oral 0.0089 vs 0.0180
#   (0.49x), oral_genital 0.0127 vs 0.0183 (0.69x), primary AUROC -0.002 (equivalent). Does that hold on DINOv3-L/16
#   block 20 with the 2 s difference of the encoder study?
#
# DATA / TOKENS (identical to the encoder-study arm D3L_d3l_512_L20, eci_encoders.sh addendum 2 C1, except precision):
#   frames = plan/train_frames.parquet (13 train + 3 calib frames per userTrain clip) and plan/eval_frames.parquet
#   (18,496 submission frames, IPW weights); mask = the round-3 'arena' mask, READ-ONLY from
#   /mnt/beegfs/locatgrp/shared/rcadei/eci_tok/encoders/r3mask/ (32 x 32 grid = identical positions at 512 px / patch 16);
#   video frame = label i + offset; lag frame = max(t - 60, 0) (2 s at 30 fps).
#   Encoder facebook/dinov3-vitl16-pretrain-lvd1689m, 224 frame bicubic-upscaled to 512 px (32 x 32 grid), ImageNet
#   mean / std, block 20 output through the final LayerNorm, CLS + 4 registers dropped. SDPA attention, bf16 autocast
#   (encoder-study precision check on 8 eval clips, block 20: token cosine to fp32 mean 0.99989, 1st pct 0.99922; it is
#   re-measured at the start of the encode job and the job ABORTS if the block-20 mean cosine < 0.9996).
#   ONE encode job decodes each clip once and writes z_t and z_{t-60} (tag p2d3l_512) to the BeeGFS store
#   /mnt/beegfs/locatgrp/shared/rcadei/eci_tok/pairpush/mabe22/ (deleted after scoring).
#
# ARMS (SAE recipe of the encoder study, unchanged: Matryoshka BatchTopK d 4096, prefixes 512/1024/2048/4096, k 16,
#   6000 steps x 4096 tokens, TokenNorm per 1024-d block, theta* on the held-out userTrain calibration tokens, frame
#   value = max over kept tokens, cross-fitted scoring with split seeds 0 and 1, full width m4096):
#   P2D  Delta 2 s: x = [z_t, z_t - z_{t-60}]   run P2D_p2d3l_512_L20
#   P2N  no Delta:  x = z_t (the same store's t part)   run P2N_p2d3l_512_L20
#   SAE seeds 320-325 for BOTH arms (fresh; no collision with the study's 300-305 / 310-315); one GPU job per arm.
#   Outputs: results/vision/benchmark_mabe22/eci/p2/{sae,eval,logs,encode_json}/, p2.md, p2.json (eci_p2_score.py).
#
# METRICS (per arm, mean over seeds x split seeds): cross-fitted AUROC, AP and honest top-1% (top-1%-picked neuron
#   among neurons firing on >= 1% of frames) for oral_oral_contact and oral_genital_contact, weighted base rates; and
#   the study's PRIMARY = mean cf AUROC over oral_oral_contact, oral_genital_contact, approaches, huddles (each label
#   averaged over the 2 split seeds first), one value per SAE seed.
# CONTRAST P2D - P2N (Delta minus no-Delta), Welch 95% CI on the 6 vs 6 per-seed values:
#   WIN (Delta better) = delta >= +0.010 AND CI excludes 0; LOSES = delta <= -0.010 AND CI excludes 0; else EQUIVALENT.
#   Applied to the primary AUROC (the decision) and, descriptively, to the per-seed cf AP of oral_oral_contact and of
#   oral_genital_contact (the same absolute 0.010 threshold, literally; at base rates ~0.002 it is hard to reach, so
#   the CI and the no-Delta / Delta AP ratio are reported next to it).
# ANSWER to the question (fixed now): the DINOv2-B AP loss HOLDS on DINOv3-L iff, on BOTH oral_oral and oral_genital,
#   the no-Delta / Delta cf AP ratio is <= 0.70 AND the Welch CI of (P2D - P2N) per-seed AP excludes 0 (Delta higher);
#   PARTIAL if that is true for exactly one of the two labels; DOES NOT HOLD otherwise.
#   Caveat stated with the answer: the DINOv2-B reference used a 1 s difference and 3 graded seeds (6-seed robustness:
#   0.50x / 0.68x); P2 uses the 2 s difference of the encoder study.
#
# BUDGET: <= 6 GPU-h, <= 80 GB BeeGFS (expected ~10 GB store, ~1.5 GPU-h). GPU jobs: -p gpu, L40S | A100 | A40,
#   --exclude=gpu242,gpu150,gpu230,gpu241,gpu240 (never gpu100); staging in /localhome/$USER/<job>_mabe.
# ======================================================================================================================
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
export MABE_ENC_TOK=/mnt/beegfs/locatgrp/shared/rcadei/eci_tok/pairpush/mabe22
export MABE_R3_MASK=/mnt/beegfs/locatgrp/shared/rcadei/eci_tok/encoders/r3mask
G="-p gpu --gres=gpu:1 --constraint=L40S|A100|A40 --exclude=gpu242,gpu150,gpu230,gpu241,gpu240 --cpus-per-task=8"
LOGS=results/vision/benchmark_mabe22/eci/p2/logs
mkdir -p $LOGS
sub() { sbatch $G --output=$LOGS/%x_%j.out "$@" scripts/benchmark/mabe22/eci_enc_job.sh; }
case "${1:-}" in
  encode)  # precision check + train (8 shards) + eval (12 shards), t and p60, one decode per clip
    PY=eci_p2_encode EXTRA="" sub --mem=64G --time=01:45:00 --job-name=p2_enc ;;
  train)   # one job per arm, 6 seeds each (needs the encode job)
    for A in P2D P2N; do
      PY=eci_p2_train EXTRA="--arm $A --seeds 320 321 322 323 324 325" sub --mem=128G --time=01:30:00 \
        ${DEP:+--dependency=afterok:$DEP} --job-name=p2_tr_$A
    done ;;
  *) echo "usage: [DEP=<encode job id>] $0 encode | train"; exit 1 ;;
esac
