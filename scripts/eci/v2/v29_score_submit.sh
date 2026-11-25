#!/bin/bash
# v2.9: after training job TRJ of DOM: codes CPU array + merge (parts removed) + per-model chains + per-model sgap.  usage: v29_score_submit.sh DOM TRJ NT
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
DOM=$1; TRJ=$2; NT=$3
KEYS=""; for s in 2901 2902 2903 2904 2905 2906; do KEYS="$KEYS,PREDk16_s$s"; done; for a in PREDk8 PREDk32; do for s in 2901 2902 2903; do KEYS="$KEYS,${a}_s$s"; done; done; KEYS=${KEYS#,}
j=$(sbatch --parsable --dependency=afterok:$TRJ -J v29codes_$DOM --array=0-$((NT-1)) -c 8 --mem=64G --time=03:00:00 scripts/eci/v2/v29_cpu.sh v29_eval.py codes --dom $DOM --task @TASK@ --ntask $NT --keys $KEYS)
m=$(sbatch --parsable --dependency=afterok:$j -J v29merge_$DOM -c 4 --mem=96G --time=01:30:00 scripts/eci/v2/v29_cpu.sh v29_eval.py merge --dom $DOM --ntask $NT --keys $KEYS)
echo "codes $j merge $m"
for k in ${KEYS//,/ }; do
  p=$(DOM=$DOM KEY=$k FWD=1 sbatch --parsable --export=ALL --dependency=afterok:$m -J v29post_${DOM}_$k scripts/eci/v2/v29_post.sh)
  s=$(sbatch --parsable --dependency=afterok:$p -J v29sgap_${DOM}_$k scripts/eci/v2/v29_sgap.sh $DOM $k)
  echo -n "$k post $p sgap $s; "
done; echo
