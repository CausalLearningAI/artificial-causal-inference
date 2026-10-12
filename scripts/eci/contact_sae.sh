#!/bin/bash
#
# Contact-token SAE (ECI, mice v1): an SAE trained only on the patches WHERE TWO MICE TOUCH (scripts/eci/contact_sae*.py).
# STEP = split | tokens | train | eval. Outputs in results/vision/eci_contact_sae/ (gitignored).
#   split   GPU  pre-registered combined splitter on 30,000 unlabelled training frames (contact_sae_split.py)
#   tokens  GPU  contact tokens (train frames + usable B-subset frames) and the C1 M5 patch-SAE codes of the B frames
#   train   GPU  Matryoshka BatchTopK SAEs on the contact tokens (static / motion x 1024 / 4096 x 3 seeds)
#   eval    GPU  cross-fitted best-neuron scores, C1 + supervised ceiling on the same frames, sheets
# SAM 2 from tools/sam2 (+ tools/sam2_deps) on PYTHONPATH (split only). Inputs staged to /localhome/$USER/$SLURM_JOB_ID,
# removed at the end.
#
# Measured 2026-10-12 (results/vision/eci_contact_sae/{split/shard_*.json, tokens/tokens.json, eval/results.json, table.md}):
#   splitter on 30,000 training frames: 77.7% pass the automatic rule (23,300; propagation branch on 63.4% of frames,
#     91-98% usable there; per-frame fallback 40-73%); 1.8 L40S GPU-h. 68,046 train contact tokens (34,023 touching
#     pairs x 2 orders) from 17,149 frames; eval 53,946 tokens, 81% of usable B frames (92.6% of nose_nose, 95.1% of
#     nose_tail positives) have >= 1. Held-out FVE static 0.66 / 0.65, motion 0.58 / 0.57 (w1024 / w4096) vs end-of-
#     training batch FVE 0.78 / 0.83, 0.66 / 0.73 (361 epochs of a small pool: w4096 overfits).
#   16,775 usable B frames, weighted, cf AUROC / AP / honest top-1% (3 seeds):
#     nose_nose  C1 M5 0.679 / 0.0517 / 0.114   static_w1024 0.644 / 0.0486 / 0.149   static_w4096 0.648 / 0.0484 / 0.143
#                motion_w1024 0.652 / 0.0421 / 0.118   motion_w4096 0.639 / 0.0414 / 0.121   ceiling P2 0.874 / 0.274 / 0.465
#     nose_tail  C1 M5 0.714 / 0.0485 / 0.102   static_w1024 0.654 / 0.0475 / 0.115   static_w4096 0.638 / 0.0414 / 0.108
#                motion_w1024 0.664 / 0.0432 / 0.107   motion_w4096 0.673 / 0.0376 / 0.084   ceiling P2 0.880 / 0.199 / 0.296
#   Pre-registered "clearly helps" (AP >= 2x C1 and top-1% >= 0.25): NO for every config (best AP ratio 0.98x).
#
# Usage:
#   STEP=split EXTRA="--shard 0 --n-shards 8" sbatch --export=ALL --time=01:00:00 scripts/eci/contact_sae.sh
#SBATCH --job-name=contact_sae
#SBATCH --output=logs/contact_sae_%j.out
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --exclude=gpu150,gpu242,gpu230
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G

module load conda
conda activate crl
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
mkdir -p logs
export STAGE=/localhome/$USER/${SLURM_JOB_ID:-local}
export LOCAL_DIR=$STAGE
mkdir -p $STAGE
trap 'rm -rf $STAGE' EXIT
echo "host $(hostname) job ${SLURM_JOB_ID:-} step $STEP"
df -h /localhome | tail -1
nvidia-smi -L || true
case $STEP in
  split)
    T=/nfs/scistore19/locatgrp/rcadei/tools
    export PYTHONPATH=$T/sam2:$T/sam2_deps:${PYTHONPATH:-}
    python -u scripts/eci/contact_sae_split.py --workers 6 ${EXTRA:-} ;;
  tokens|train|eval)
    python -u scripts/eci/contact_sae.py $STEP ${EXTRA:-} ;;
esac
