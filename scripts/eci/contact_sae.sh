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
