#!/bin/bash
# v2.5: after the training job TRJ of DOM: codes CPU array + merge + per-model scoring chains. usage: v25_score_submit.sh DOM TRJ NT
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
DOM=$1; TRJ=$2; NT=$3
j=$(sbatch --parsable --dependency=afterok:$TRJ --export=ALL -J v25codes_$DOM --array=0-$((NT-1)) -c 8 --mem=48G --time=03:00:00 scripts/eci/v2/v25_cpu.sh v25_train.py codes --dom $DOM --task @TASK@ --ntask $NT)
m=$(sbatch --parsable --dependency=afterok:$j --export=ALL -J v25merge_$DOM -c 4 --mem=64G --time=01:00:00 scripts/eci/v2/v25_cpu.sh v25_train.py merge --dom $DOM --ntask $NT)
echo "codes $j merge $m"
for a in ${ARMS:-N0 W1 Wh C1}; do for s in 2501 2502 2503; do
  DOM=$DOM KEY=${a}_s$s FWD=1 V25_MULTI=${V25_MULTI:-0} sbatch --parsable --export=ALL --dependency=afterok:$m -J v25post_${DOM}_${a}_s$s scripts/eci/v2/v25_post.sh
done; done
