#!/bin/bash
#
# temporal_ctx launcher (scripts/eci/temporal_ctx.py). EXTRA = args after "run". GPU: 'gpu' partition, never gpu100,
# excluding gpu150 / gpu242 / gpu230 / gpu241. Staging in /localhome/$USER/<job> ($TMPDIR when /localhome is missing),
# removed on exit (decoded frames, videos, tokens never leave the node). Frames are decoded once and every JPG is encoded
# once per job; every arm in --arms trains on the same tokens.
# Outputs: results/vision/eci_temporal_ctx/<domain>/ (gitignored).
#
# Usage (from the repo root; mkdir -p results/vision/eci_temporal_ctx/logs first):
#   TC_OUT=results/vision/eci_temporal_ctx/smoke EXTRA="--domain mice --smoke --seeds 0 1" \
#       sbatch --time=00:40:00 scripts/eci/temporal_ctx.sh
#   EXTRA="--domain mice" sbatch scripts/eci/temporal_ctx.sh
#   EXTRA="--domain ants" sbatch --time=04:00:00 scripts/eci/temporal_ctx.sh
#   python scripts/eci/temporal_ctx.py summary
#SBATCH --job-name=tctx
#SBATCH --output=results/vision/eci_temporal_ctx/logs/%x_%j.out
#SBATCH --time=03:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu150,gpu242,gpu230,gpu241
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

module load conda
conda activate crl
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1 HF_HUB_OFFLINE=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
if [ -d /localhome ] && [ -w /localhome/$USER -o -w /localhome ]; then
    export LOCAL_DIR=/localhome/$USER/${SLURM_JOB_ID:-local}_tc
else
    export LOCAL_DIR=${TMPDIR:-/tmp}/${SLURM_JOB_ID:-local}_tc
    export TMPDIR=${TMPDIR:-/tmp}
fi
mkdir -p $LOCAL_DIR
trap 'rm -rf $LOCAL_DIR' EXIT
echo "host $(hostname) job ${SLURM_JOB_ID:-} extra ${EXTRA:-} out ${TC_OUT:-default} local $LOCAL_DIR"
df -h $LOCAL_DIR | tail -1
nvidia-smi -L 2>/dev/null || true
python -u scripts/eci/temporal_ctx.py run ${EXTRA:-}
