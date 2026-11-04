#!/bin/bash
# v2.2 job wrapper: bash v22_run.sh <script.py> <args...>; stages a /localhome dir, logs GPU utilisation every 2 min.
module load conda
conda activate crl
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TQDM_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LOCAL_DIR=/localhome/$USER/${SLURM_JOB_ID}_v22
mkdir -p $LOCAL_DIR
trap "rm -rf $LOCAL_DIR" EXIT
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
if command -v nvidia-smi >/dev/null && [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  (nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 120 2>/dev/null | sed -u "s/^/[gpu] /") &
fi
python -u scripts/eci/v2/"$@"
