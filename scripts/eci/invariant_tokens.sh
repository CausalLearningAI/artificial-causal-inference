#!/bin/bash
#
# invariant_tokens launcher (scripts/eci/invariant_tokens.py; also used by idea B, big_encoder). STEP = plan | encode |
# train; EXTRA = args after the step. GPU: 'gpu' partition (never gpu100), excluding gpu150 / gpu242 / gpu230 / gpu241.
# Staging in /localhome/$USER/<job> ($TMPDIR when /localhome is missing), removed on exit.
# Outputs: results/vision/eci_invariant/ and results/vision/eci_big_encoder/ (gitignored).
#
# Usage (from the repo root):
#   STEP=plan sbatch -p defaultp --gres=none --cpus-per-task=4 --mem=64G --time=00:45:00 scripts/eci/invariant_tokens.sh
#   STEP=encode EXTRA="--orbit --set pool0" sbatch scripts/eci/invariant_tokens.sh
#   STEP=encode EXTRA="--orbit --set eval --shard 0 --nshard 2" sbatch scripts/eci/invariant_tokens.sh
#   STEP=train EXTRA="--kinds id orb --probe" sbatch scripts/eci/invariant_tokens.sh
#   python scripts/eci/invariant_tokens.py summary           # login safe (reads JSONs)
#   smoke: INV_SMOKE=1 INV_OUT=results/vision/eci_invariant/smoke STEP=encode EXTRA="--orbit --set pool0 --smoke" ...
#SBATCH --job-name=inv_tok
#SBATCH --output=results/vision/eci_invariant/logs/%x_%j.out
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
python -u scripts/eci/invariant_tokens.py $STEP ${EXTRA:-}
