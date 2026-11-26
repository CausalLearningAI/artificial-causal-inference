#!/bin/bash
# v2.10 full chain (v210_prereg.md): identity check -> PREDk16 prep / train (cal, mab, crim, frogs); recipe V prep / train
# (mab, crim); codes; per-dictionary scores; report.  usage: bash v210_submit.sh
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
S=scripts/eci/v2
G="$S/v210_gpu.sh"; C="$S/v210_cpu.sh"
sb() { sbatch --parsable "$@"; }
ID=$(sb -J v210id -t 00:40:00 $G v210_train.py train --dom mice --cfgs PREDk16:2701 --max_steps 5001 --idcheck --tag idcheck)
echo "idcheck $ID"
declare -A TP TV CO
for d in cal mab crim frogs; do
  p=$(sb -J v210prep_$d -c 16 --mem 64G -t 01:30:00 $C v210_train.py prep --dom $d)
  TP[$d]=$(sb -J v210train_$d -t 04:00:00 --dependency=afterok:$ID:$p $G v210_train.py train --dom $d)
  echo "$d prepP $p trainP ${TP[$d]}"
done
for d in mab crim; do
  p=$(sb -J v210prepV_$d -t 02:30:00 $G v210_bench.py prep --dom $d)
  TV[$d]=$(sb -J v210trainV_$d -t 02:00:00 --dependency=afterok:$p $G v210_bench.py trainV --dom $d)
  echo "$d prepV $p trainV ${TV[$d]}"
done
CO[cal]=$(sb -J v210codes_cal -t 01:30:00 --dependency=afterok:${TP[cal]} $G v210_bench.py codes --dom cal)
CO[mab]=$(sb -J v210codes_mab -t 04:00:00 --dependency=afterok:${TP[mab]}:${TV[mab]} $G v210_bench.py codes --dom mab)
CO[crim]=$(sb -J v210codes_crim -t 06:00:00 --dependency=afterok:${TP[crim]}:${TV[crim]} $G v210_bench.py codes --dom crim)
CO[frogs]=$(sb -J v210codes_frogs -t 06:00:00 --dependency=afterok:${TP[frogs]} $G v210_bench.py codes --dom frogs)
echo "codes cal ${CO[cal]} mab ${CO[mab]} crim ${CO[crim]} frogs ${CO[frogs]}"
SC=""
for d in cal mab crim; do
  case $d in cal) R="-c 8 --mem 48G -t 01:30:00";; mab) R="-c 32 --mem 140G -t 05:00:00";; crim) R="-c 32 --mem 180G -t 06:00:00";; esac
  for k in V_s2001 V_s2002 V_s2003 PREDk16_s3001 PREDk16_s3002 PREDk16_s3003; do
    j=$(sb -J v210score_${d}_$k $R --dependency=afterok:${CO[$d]} $C v210_bench.py score --dom $d --key $k)
    SC="$SC:$j"; echo -n "$d $k $j; "
  done; echo
done
RP=$(sb -J v210report -c 2 --mem 8G -t 00:20:00 --dependency=afterok$SC:${CO[frogs]} $C v210_bench.py report)
echo "report $RP"
