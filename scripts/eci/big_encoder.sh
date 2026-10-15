#!/bin/bash
#
# big_encoder launcher (scripts/eci/big_encoder.py -> the invariant_tokens.py harness with --model large|giant).
# STEP = encode | train; EXTRA = args after the step.
# Staging in /localhome/$USER/<job> ($TMPDIR when /localhome is missing), removed on exit.
# Outputs: results/vision/eci_invariant/ and results/vision/eci_big_encoder/ (gitignored).
#
# Usage (from the repo root; the pool1 plan of invariant_tokens must exist):
#   STEP=encode EXTRA="--model large --set pool0" sbatch scripts/eci/big_encoder.sh
#   STEP=encode EXTRA="--model giant --set eval --shard 0 --nshard 2 --bs 8 --fwd 8" sbatch --time=02:00:00 scripts/eci/big_encoder.sh
#   STEP=train EXTRA="--model large --kinds id --probe" sbatch scripts/eci/big_encoder.sh
#   python scripts/eci/big_encoder.py summary              # login safe (reads JSONs)
#SBATCH --job-name=big_enc
#SBATCH --output=results/vision/eci_big_encoder/logs/%x_%j.out
#SBATCH --time=01:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu150,gpu242,gpu230,gpu241
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G

module load conda
conda activate crl
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1 HF_HUB_OFFLINE=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
if [ -d /localhome ] && [ -w /localhome/$USER -o -w /localhome ]; then
    export LOCAL_DIR=/localhome/$USER/${SLURM_JOB_ID:-local}_inv
else
    export LOCAL_DIR=${TMPDIR:-/tmp}/${SLURM_JOB_ID:-local}_inv
    export TMPDIR=${TMPDIR:-/tmp}
fi
mkdir -p $LOCAL_DIR
trap 'rm -rf $LOCAL_DIR' EXIT
echo "host $(hostname) job ${SLURM_JOB_ID:-} step $STEP extra ${EXTRA:-} local $LOCAL_DIR"
df -h $LOCAL_DIR | tail -1
nvidia-smi -L 2>/dev/null || true
python -u scripts/eci/big_encoder.py $STEP ${EXTRA:-}
