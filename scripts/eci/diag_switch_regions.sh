#!/bin/bash
#
# Where do the mice contact neurons' switch reactions come from (scripts/eci/diag_switch_regions.py)?
# STEP = prep (CPU: bag zone, regions, dark cue of the 172,800 eval frames) | encode (GPU: per-token codes of the
# candidate neurons on both token stores, ~80 GB staged in turn) | analyse (CPU). Inputs are staged to
# /localhome/$USER/$SLURM_JOB_ID, which is removed at exit. Outputs: results/vision/eci_repr_diag/mice/switch_regions/
#
# Usage:
#   mkdir -p logs
#   STEP=prep    sbatch --export=ALL --partition=defaultp --cpus-per-task=16 --mem=48G --time=00:45:00 scripts/eci/diag_switch_regions.sh
#   STEP=encode  sbatch --export=ALL --partition=gpu --gres=gpu:1 --exclude=gpu150,gpu242 --cpus-per-task=8 --mem=120G --time=00:50:00 scripts/eci/diag_switch_regions.sh
#   STEP=analyse sbatch --export=ALL --partition=defaultp --cpus-per-task=4 --mem=64G --time=00:40:00 scripts/eci/diag_switch_regions.sh
#SBATCH --job-name=switch_regions
#SBATCH --output=logs/switch_regions_%x_%j.out

module load conda
conda activate crl
export PYTHONUNBUFFERED=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
STAGE=/localhome/$USER/${SLURM_JOB_ID:-manual}_switch_regions
mkdir -p $STAGE
trap 'rm -rf $STAGE' EXIT
df -h /localhome | tail -1
[ "${STEP}" = encode ] && nvidia-smi -L
export LOCAL_DIR=$STAGE
python -u scripts/eci/diag_switch_regions.py ${STEP} --stage $STAGE --workers ${SLURM_CPUS_PER_TASK:-4}
