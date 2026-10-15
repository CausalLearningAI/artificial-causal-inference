#!/bin/bash
#
# MABe22 mouse ECI - false-positive figure of the D2 oral_oral neuron (scripts/benchmark/mabe22/eci_fig_fp.py).
# STAGE = data (GPU: selection, decode, mask recompute, mm scale, numbers) | plot (CPU: figure only) | all.
#   STAGE=data sbatch -p gpu --gres=gpu:3090:1 --exclude=gpu242,gpu150,gpu230,gpu241 scripts/benchmark/mabe22/eci_fig_fp.sh
#   STAGE=plot sbatch -p defaultp --gres=none --cpus-per-task=2 --mem=4G --time=00:10:00 scripts/benchmark/mabe22/eci_fig_fp.sh
# Staging in /localhome/$USER/<job> ($TMPDIR when missing), removed on exit. Outputs: results/.../eci/fig_fp/.
#SBATCH --job-name=mabe_figfp
#SBATCH --output=results/vision/benchmark_mabe22/eci/logs/%x_%j.out
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

module load conda
conda activate crl
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1 HF_HUB_OFFLINE=1 MPLCONFIGDIR=${TMPDIR:-/tmp}/mpl_${SLURM_JOB_ID:-x}
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
if [ -d /localhome ] && [ -w /localhome/$USER -o -w /localhome ]; then
    export LOCAL_DIR=/localhome/$USER/${SLURM_JOB_ID:-local}_figfp
else
    export LOCAL_DIR=${TMPDIR:-/tmp}/${SLURM_JOB_ID:-local}_figfp
    export TMPDIR=${TMPDIR:-/tmp}
fi
mkdir -p $LOCAL_DIR
trap 'rm -rf $LOCAL_DIR $MPLCONFIGDIR' EXIT
echo "host $(hostname) job ${SLURM_JOB_ID:-} stage ${STAGE:-all} local $LOCAL_DIR"
nvidia-smi -L 2>/dev/null || true
python -u scripts/benchmark/mabe22/eci_fig_fp.py --stage ${STAGE:-all}
