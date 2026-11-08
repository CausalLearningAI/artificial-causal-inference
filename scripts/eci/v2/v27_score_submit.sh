#!/bin/bash
# v2.6: after training job TRJ of DOM: codes CPU array + merge + per-model chains.  usage: v26_score_submit.sh DOM TRJ NT
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
DOM=$1; TRJ=$2; NT=$3
KEYS=""; for a in PREDc PREDd PREDk PREDh; do for s in 2701 2702 2703; do KEYS="$KEYS,${a}_s$s"; done; done; KEYS=${KEYS#,}
j=$(sbatch --parsable --dependency=afterok:$TRJ -J v27codes_$DOM --array=0-$((NT-1)) -c 8 --mem=64G --time=03:00:00 scripts/eci/v2/v27_cpu.sh v27_eval.py codes --dom $DOM --task @TASK@ --ntask $NT --keys $KEYS)
m=$(sbatch --parsable --dependency=afterok:$j -J v27merge_$DOM -c 4 --mem=96G --time=01:00:00 scripts/eci/v2/v27_cpu.sh v27_eval.py merge --dom $DOM --ntask $NT --keys $KEYS)
echo "codes $j merge $m"
for k in ${KEYS//,/ }; do DOM=$DOM KEY=$k FWD=1 sbatch --parsable --export=ALL --dependency=afterok:$m -J v27post_${DOM}_$k scripts/eci/v2/v27_post.sh; done | tr '\n' ' '; echo
