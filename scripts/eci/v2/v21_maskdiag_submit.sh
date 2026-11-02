#!/bin/bash
# v21_maskdiag.py submission (pre-registration v21_maskdiag_prereg.md). Prints job ids.
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
L=results/vision/eci_v2_1/maskdiag/logs
PRE='export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True LOCAL_DIR=/localhome/$USER/${SLURM_JOB_ID}_md; mkdir -p $LOCAL_DIR; trap "rm -rf $LOCAL_DIR" EXIT; cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference; (nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 120 2>/dev/null | sed -u "s/^/[gpu] /") & python -u scripts/eci/v2/v21_maskdiag.py'
GPU=(-p gpu --gres=gpu:1 --constraint="L40S|A100|A40|A10" --exclude=gpu240,gpu241,gpu242,gpu278 -c 16)
csub() { sbatch --parsable -p defaultp "$@"; }
gsub() { sbatch --parsable "${GPU[@]}" "$@"; }
for dom in mice ants; do
  st=$(csub -c 4 --mem=32G --time=03:00:00 -J md_stats_$dom -o $L/%x_%j.out --wrap "$PRE stats --dom $dom")
  pr=$(csub -c 24 --mem=96G --time=06:00:00 -J md_prep_$dom -o $L/%x_%j.out --wrap "$PRE prep --dom $dom")
  echo "$dom stats $st prep $pr"
  tr=()
  for arm in SA SMN SU; do for s in 1001 1002 1003; do
    tr+=($(gsub --mem=120G --time=03:00:00 -d afterok:$pr -J md_tr_${dom}_${arm}${s} -o $L/%x_%j.out --wrap "$PRE train --dom $dom --arm $arm --seed $s"))
  done; done
  dep=$(IFS=:; echo "${tr[*]}")
  cn=$(gsub --mem=120G --time=08:00:00 -d afterok:$dep -J md_codes_${dom}_new -o $L/%x_%j.out --wrap "$PRE codes --dom $dom --spec SA=SA@A,SMN=SMN@MN,SU=SU@U")
  echo "$dom train ${tr[*]} codes $cn"
  for arm in SA SMN SU; do for s in 1001 1002 1003; do
    echo "  score $dom $arm $s: $(csub -c 32 --mem=200G --time=04:00:00 -d afterok:$cn -J md_sc_${dom}_${arm}_s$s -o $L/%x_%j.out --wrap "$PRE score --dom $dom --arms $arm --seeds $s --only ${arm}_s$s")"
  done; done
done
# mice cross readouts (no training)
cx=$(gsub --mem=120G --time=08:00:00 -J md_codes_mice_cross -o $L/%x_%j.out --wrap "$PRE codes --dom mice --spec S=S@M,SN2=SN2@N,SxN2=S@N,SN2xM=SN2@M --peaks --check_only_existing")
echo "mice cross codes $cx"
for arm in SxN2 SN2xM; do for s in 1001 1002 1003; do
  echo "  score mice $arm $s: $(csub -c 32 --mem=200G --time=04:00:00 -d afterok:$cx -J md_sc_mice_${arm}_s$s -o $L/%x_%j.out --wrap "$PRE score --dom mice --arms $arm --seeds $s --only ${arm}_s$s")"
done; done
