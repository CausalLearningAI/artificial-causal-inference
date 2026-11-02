#!/bin/bash
# v21_v1gap.py submission (pre-registration v21_v1gap_prereg.md). Prints job ids.
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
L=results/vision/eci_v2_1/v1gap/logs
PRE='export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TQDM_DISABLE=1 TRANSFORMERS_VERBOSITY=error PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True LOCAL_DIR=/localhome/$USER/${SLURM_JOB_ID}_v1g; mkdir -p $LOCAL_DIR; trap "rm -rf $LOCAL_DIR" EXIT; cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference; (nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 120 2>/dev/null | sed -u "s/^/[gpu] /") & python -u scripts/eci/v2/v21_v1gap.py'
GPU=(-p gpu --gres=gpu:1 --exclude=gpu240,gpu241,gpu242,gpu278 -c 16)
csub() { sbatch --parsable -p defaultp "$@"; }
gsub() { sbatch --parsable "${GPU[@]}" "$@"; }
cols=()
for s in 1001 1002 1003; do
  cols+=($(csub -c 16 --mem=64G --time=03:00:00 -J v1g_sc_SN2raw_s$s -o $L/%x_%j.out --wrap "$PRE cols --keys SN2raw_s$s"))
done
tk=$(gsub --mem=128G --time=04:00:00 -J v1g_tok -o $L/%x_%j.out --wrap "$PRE tok")
echo "tok $tk"
for arm in D2r1 D2r2 D3r1 D3r2; do
  tr=$(gsub --mem=96G --time=03:00:00 -d afterok:$tk -J v1g_tr_$arm -o $L/%x_%j.out --wrap "$PRE train --arm $arm")
  echo "train $arm $tr"
  for s in 1 2 3; do
    cols+=($(csub -c 16 --mem=64G --time=03:00:00 -d afterok:$tr -J v1g_sc_${arm}_s$s -o $L/%x_%j.out --wrap "$PRE cols --keys ${arm}_s$s"))
  done
done
echo "cols ${cols[*]}"
dep=$(IFS=:; echo "${cols[*]}")
echo "boot $(csub -c 32 --mem=96G --time=03:00:00 -d afterok:$dep -J v1g_boot -o $L/%x_%j.out --wrap "$PRE boot")"
