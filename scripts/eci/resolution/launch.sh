#!/bin/bash
# Submit the resolution sweep (rule pre-registered in res.py). bash launch.sh mice | ants   (prints job ids)
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
X="--exclude=gpu240,gpu241,gpu242"
G=${GATE:-}  # e.g. the self-test job ids, colon separated: every first-stage job waits for them (afterok)
sub() { local name=$1 dep=$2 t=$3 mem=$4 step=$5; shift 5
  sbatch --parsable -p gpu --gres=gpu:1 --constraint="L40S|A100|A40" $X --job-name=$name --time=$t --mem=$mem \
    $( d=${dep:-}; [ -n "$G" ] && d=${d:+$d:}$G; [ -n "$d" ] && echo --dependency=afterok:$d ) --export=ALL,STEP=$step,EXTRA="$*" scripts/eci/resolution/res.sh; }
case "${1:-}" in
  mice)
    for A in d3_512 d3_768 d3_1408 d2_672 d2_896 d3_1024; do
      T=""; [ $A != d3_1024 ] && T=$(sub res_mtok_$A "" 10:00:00 160G mtokens --arm $A)
      P=$(sub res_mpool_$A "" 06:00:00 160G mpool --arm $A)
      S=$(sub res_msweep_$A "${T:+$T:}$P" 04:00:00 200G msweep --arm $A)
      echo "$A tokens $T pool $P sweep $S"
    done
    S=$(sub res_msweep_d3_2048 "" 05:00:00 240G msweep --arm d3_2048); echo "d3_2048 sweep $S" ;;
  ants)
    for A in aN a7 a10; do
      P=$(sub res_aprep_$A "" 08:00:00 160G aprep --arm $A)
      T=$(sub res_atrain_$A "$P" 04:00:00 200G atrain --arm $A)
      E2=$(sub res_aenc_v2_$A "$T" 10:00:00 128G aencode --arm $A --series v2)
      E3=$(sub res_aenc_v3_$A "$T" 16:00:00 128G aencode --arm $A --series v3)
      C=$(sub res_ascore_$A "$E2:$E3" 01:00:00 64G ascore --arm $A)
      echo "$A prep $P train $T enc $E2 $E3 score $C"
    done ;;
  *) echo "usage: $0 mice | ants"; exit 1 ;;
esac
