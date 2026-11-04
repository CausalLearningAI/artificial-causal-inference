#!/bin/bash
# ECI probe job wrapper: sbatch [--array=...] scripts/eci/probe/job.sh <script.py> <step> [args]; --task defaults to SLURM_ARRAY_TASK_ID
#SBATCH --output=/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference/results/vision/eci_probe/logs/%x_%A_%a.out
module load conda
conda activate crl
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TQDM_DISABLE=1 TRANSFORMERS_VERBOSITY=error
set -eo pipefail
if [ -d /localhome ] && mkdir -p /localhome/$USER 2>/dev/null; then BASE=/localhome/$USER; else BASE=${TMPDIR:-/tmp}; fi
export LOCAL_DIR=$BASE/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}_probe
mkdir -p $LOCAL_DIR
trap "rm -rf $LOCAL_DIR" EXIT
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
S=$1; shift
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then T="--task $SLURM_ARRAY_TASK_ID"; else T=""; fi
python -u scripts/eci/probe/$S "$@" $T
