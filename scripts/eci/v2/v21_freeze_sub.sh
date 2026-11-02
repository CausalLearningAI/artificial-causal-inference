#!/bin/bash
# helpers for v21_freeze.py jobs: source this file, then  csub <sbatch args> -- <step args> | gsub ...
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
FL=results/vision/eci_v2_1/freeze/logs
PRE='export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TQDM_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True LOCAL_DIR=/localhome/$USER/${SLURM_JOB_ID}_fz; mkdir -p $LOCAL_DIR; trap "rm -rf $LOCAL_DIR" EXIT; cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference; (nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 300 2>/dev/null | sed -u "s/^/[gpu] /") & python -u scripts/eci/v2/v21_freeze.py'
GPU=(-p gpu --gres=gpu:1 --constraint="L40S|A100|A40|A10" --exclude=gpu240,gpu241,gpu242,gpu278)
csub() { local n=$1; shift; local args=$1; shift; sbatch --parsable -p defaultp $args -J $n -o $FL/%x_%j.out --wrap "$PRE $*"; }
gsub() { local n=$1; shift; local args=$1; shift; sbatch --parsable "${GPU[@]}" $args -J $n -o $FL/%x_%j.out --wrap "$PRE $*"; }
