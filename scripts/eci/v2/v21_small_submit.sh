#!/bin/bash
# v21_small.py submission (pre-registration v21_small_prereg.md). Prints job ids; appends them to small/myids.txt.
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
L=results/vision/eci_v2_1/small/logs
IDS=results/vision/eci_v2_1/small/myids.txt
PRE='export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TQDM_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True LOCAL_DIR=/localhome/$USER/${SLURM_JOB_ID}_sm; mkdir -p $LOCAL_DIR; trap "rm -rf $LOCAL_DIR" EXIT; cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference; (nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 120 2>/dev/null | sed -u "s/^/[gpu] /") & python -u scripts/eci/v2/v21_small.py'
GPU=(-p gpu --gres=gpu:1 --exclude=gpu240,gpu241,gpu242,gpu278 -c 16)
csub() { local j; j=$(sbatch --parsable -p defaultp --exclude=eta310 "$@"); echo "$j" >> $IDS; echo "$j"; }
gsub() { local j; j=$(sbatch --parsable "${GPU[@]}" "$@"); echo "$j" >> $IDS; echo "$j"; }
for dom in ants mice; do
  # identity: stored V_s1004 re-scored (no dependency)
  idc=$(csub -c 32 --mem=200G --time=06:00:00 -J sm_${dom}_cols_V_s1004 -o $L/%x_%j.out --wrap "$PRE cols --dom $dom --model V_s1004")
  echo "$dom idcheck $idc"
  tr=()
  for arm in Ssm Vsm; do
    tr+=($(gsub --mem=240G --time=03:00:00 -J sm_${dom}_tr_$arm -o $L/%x_%j.out --wrap "$PRE train --dom $dom --arm $arm"))
  done
  dep=$(IFS=:; echo "${tr[*]}")
  cd_=$(gsub --constraint="L40S|A100|A40" --mem=160G --time=04:00:00 -d afterok:$dep -J sm_${dom}_codes -o $L/%x_%j.out --wrap "$PRE codes --dom $dom")
  echo "$dom train ${tr[*]} codes $cd_"
  cols=($idc)
  for arm in Ssm Vsm; do for s in 1004 1005 1006; do for g in max top10; do
    ag=""; [ $g = top10 ] && ag="--agg top10"
    cols+=($(csub -c 32 --mem=200G --time=06:00:00 -d afterok:$cd_ -J sm_${dom}_cols_${arm}_s${s}_$g -o $L/%x_%j.out --wrap "$PRE cols --dom $dom --model ${arm}_s$s $ag"))
  done; done; done
  echo "$dom cols ${cols[*]}"
  dep=$(IFS=:; echo "${cols[*]}")
  echo "$dom boot $(csub -c 32 --mem=120G --time=04:00:00 -d afterok:$dep -J sm_${dom}_boot -o $L/%x_%j.out --wrap "$PRE boot --dom $dom")"
done
