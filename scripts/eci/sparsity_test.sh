#!/bin/bash
#
# sparsity_test launcher (scripts/eci/sparsity_test.py). EXTRA = args after "run". GPU: 'gpu' partition, never gpu100,
# excluding gpu150 / gpu242 / gpu230 / gpu241. Staging in /localhome/$USER/<job> ($TMPDIR when /localhome is missing),
# removed on exit. Tokens are encoded ONCE per job; every arm in --arms trains on them.
# Outputs: results/vision/eci_sparsity/ (gitignored).
#
# Usage (from the repo root):
#   EXTRA="--arms c448 btk4 btk8 tk4 tk8" sbatch scripts/eci/sparsity_test.sh
#   EXTRA="--arms btk16 btk32 tk16 tk32" sbatch scripts/eci/sparsity_test.sh
#   SP_OUT=results/vision/eci_sparsity/smoke EXTRA="--arms c448 btk4 tk4 --smoke --seeds 0 1" \
#       sbatch --time=00:30:00 scripts/eci/sparsity_test.sh
#   python scripts/eci/sparsity_test.py summary
#SBATCH --job-name=sparsity
#SBATCH --output=results/vision/eci_sparsity/logs/%x_%j.out
#SBATCH --time=02:30:00
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
    export LOCAL_DIR=/localhome/$USER/${SLURM_JOB_ID:-local}_sp
else
    export LOCAL_DIR=${TMPDIR:-/tmp}/${SLURM_JOB_ID:-local}_sp
    export TMPDIR=${TMPDIR:-/tmp}
fi
mkdir -p $LOCAL_DIR
trap 'rm -rf $LOCAL_DIR' EXIT
echo "host $(hostname) job ${SLURM_JOB_ID:-} extra ${EXTRA:-} out ${SP_OUT:-default} local $LOCAL_DIR"
nvidia-smi -L 2>/dev/null || true
python -u scripts/eci/sparsity_test.py run ${EXTRA:-}
