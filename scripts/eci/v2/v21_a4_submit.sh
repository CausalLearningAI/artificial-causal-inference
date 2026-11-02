#!/bin/bash
# AMENDMENT 4 submission (v21_train.py): fresh seeds 1004-1006 for S / V / W (train -> codes max + aggs -> score),
# and top10 / mean codes + scores for the existing W / B (bit0) and SN2 (bit1) seeds 1001-1003. Prints job ids.
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
L=results/vision/eci_v2_1/logs
PRE='export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True LOCAL_DIR=/localhome/$USER/${SLURM_JOB_ID}_v21; mkdir -p $LOCAL_DIR; trap "rm -rf $LOCAL_DIR" EXIT; cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference; nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 120 | sed -u "s/^/[gpu] /" & python -u scripts/eci/v2/v21_train.py'
GPU=(-p gpu --gres=gpu:1 --exclude=gpu240,gpu241,gpu242,gpu278 -c 16 --mem=240G)
gsub() { sbatch --parsable "${GPU[@]}" "$@"; }
csub() { sbatch --parsable -p defaultp -c 32 --mem=200G --time=12:00:00 "$@"; }
SCO='cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference; python -u scripts/eci/v2/v21_train.py score'
for dom in mice ants; do
  tr=()
  for arm in S V W; do for s in 1004 1005 1006; do
    tr+=($(gsub --constraint="L40S|A100|A40|A10" --time=04:00:00 -J v21tr_${dom}_${arm}${s} -o $L/%x_%j.out --wrap "$PRE train --dom $dom --arm $arm --seed $s"))
  done; done
  dep=$(IFS=:; echo "${tr[*]}")
  cm=$(gsub --constraint="L40S|A100" --time=10:00:00 -d afterok:$dep -J v21codes_${dom}_fresh -o $L/%x_%j.out --wrap "$PRE codes --dom $dom --arms S,V,W --seeds 1004,1005,1006 --no_rows")
  ca=$(gsub --constraint="L40S|A100" --time=10:00:00 -d afterok:$dep -J v21agg_${dom}_fresh -o $L/%x_%j.out --wrap "$PRE codes --dom $dom --arms S,V,W --seeds 1004,1005,1006 --aggs top10,mean --no_rows")
  echo "$dom train ${tr[*]} codes $cm aggcodes $ca"
  for arm in S V W; do for s in 1004 1005 1006; do
    a=$(csub -d afterok:$cm -J v21sc_${dom}_${arm}_s$s -o $L/%x_%j.out --wrap "$SCO --dom $dom --arms $arm --seeds $s --only ${arm}_s$s")
    b=$(csub -d afterok:$ca:$cm -J v21sc_${dom}_${arm}_s${s}_top10 -o $L/%x_%j.out --wrap "$SCO --dom $dom --arms $arm --seeds $s --only ${arm}_s$s --agg top10")
    c=$(csub -d afterok:$ca:$cm -J v21sc_${dom}_${arm}_s${s}_mean -o $L/%x_%j.out --wrap "$SCO --dom $dom --arms $arm --seeds $s --only ${arm}_s$s --agg mean")
    echo "  score $dom $arm $s: $a $b $c"
  done; done
  # existing seeds: aggs for W, B (bit0) and SN2 (bit1)
  cw=$(gsub --constraint="L40S|A100" --time=10:00:00 -J v21agg_${dom}_WB -o $L/%x_%j.out --wrap "$PRE codes --dom $dom --arms W,B --aggs top10,mean --no_rows")
  sdep=""; [ "$dom" = mice ] && sdep="-d afterok:${SN2CODES}"
  cn=$(gsub --constraint="L40S|A100" --time=10:00:00 $sdep -J v21agg_${dom}_SN2 -o $L/%x_%j.out --wrap "$PRE codes --dom $dom --arms SN2 --aggs top10,mean --no_rows")
  echo "$dom aggcodes W,B $cw SN2 $cn"
  for arm in W B SN2; do for s in 1001 1002 1003; do for g in top10 mean; do
    d=$cw; [ $arm = SN2 ] && d=$cn
    x=$(csub -d afterok:$d -J v21sc_${dom}_${arm}_s${s}_$g -o $L/%x_%j.out --wrap "$SCO --dom $dom --arms $arm --only ${arm}_s$s --agg $g")
    echo "  score $dom $arm $s $g: $x"
  done; done; done
done
