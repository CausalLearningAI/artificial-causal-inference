#!/bin/bash
#
# gated_sae launcher (scripts/eci/gated_sae.py). EXTRA = args after "run". GPU: 'gpu' partition, never gpu100,
# excluding gpu150 / gpu242 / gpu230 / gpu241. Staging in /localhome/$USER/<job> ($TMPDIR when /localhome is missing),
# removed on exit. Tokens are encoded ONCE per job; every arm in --arms trains on them.
# Outputs: results/vision/eci_gated_sae/ (gitignored).
#
# Usage (from the repo root):
#   sbatch scripts/eci/gated_sae.sh                                   # gate16 + gate8, seeds 0 1 2
#   GS_OUT=results/vision/eci_gated_sae/smoke EXTRA="--smoke --steps 800 --seeds 0" \
#       sbatch --time=00:40:00 scripts/eci/gated_sae.sh
#   python scripts/eci/gated_sae.py summary
#   unit tests (CPU): sbatch -p defaultp -c 4 --mem 8G -t 00:20:00 --wrap "bash scripts/eci/gated_sae.sh test"
#SBATCH --job-name=gated_sae
#SBATCH --output=results/vision/eci_gated_sae/logs/%x_%j.out
#SBATCH --time=02:00:00
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
if [ "${1:-}" = test ]; then
    exec python -u scripts/eci/gated_sae_test.py
fi
if [ -d /localhome ] && [ -w /localhome/$USER -o -w /localhome ]; then
    export LOCAL_DIR=/localhome/$USER/${SLURM_JOB_ID:-local}_gs
else
    export LOCAL_DIR=${TMPDIR:-/tmp}/${SLURM_JOB_ID:-local}_gs
    export TMPDIR=${TMPDIR:-/tmp}
fi
mkdir -p $LOCAL_DIR
trap 'rm -rf $LOCAL_DIR' EXIT
echo "host $(hostname) job ${SLURM_JOB_ID:-} extra ${EXTRA:-} out ${GS_OUT:-default} local $LOCAL_DIR"
nvidia-smi -L 2>/dev/null || true
python -u scripts/eci/gated_sae.py run ${EXTRA:-}
