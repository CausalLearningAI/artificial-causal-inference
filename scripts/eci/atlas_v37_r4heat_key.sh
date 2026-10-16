#!/bin/bash
#
# Submit every GPU part of one key (one job group per GPU model its R4 full pass ran on, ~TARGET frames per job) and the
# CPU assemble (afterok; MOVE=1 installs the real heat files into patch/ if the checks pass). Login node: only a
# planning read (picks.json, plan.json, encode logs; < 1 s) and sbatch calls.
#   bash scripts/eci/atlas_v37_r4heat_key.sh ants_v3 [MOVE=1 via env]
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
KEY=$1
TARGET=${TARGET:-9000}
D=$(python scripts/eci/atlas_v37_r4heat.py --key $KEY --dry)
G=$(echo "$D" | sed -n 's/^GPUS=//p')
W=$(echo "$D" | sed -n 's/^WORK=//p')
[ -d "$W" ] || { echo "no work dir $W"; exit 1; }
rm -rf "$W/r4heat_parts"  # parts of an earlier submission
echo "$KEY: frames per encode GPU $G"
ids=()
while IFS=$'\t' read -r gpu frames; do
  case "$gpu" in
    *L40S*) con=L40S ;; *A40*) con=A40 ;; *A100*) con=A100 ;; *2080*) con=2080ti ;;
    *) echo "no constraint for $gpu"; exit 1 ;;
  esac
  n=$(( (frames + TARGET - 1) / TARGET ))
  for t in $(seq 0 $((n - 1))); do
    j=$(sbatch --parsable --constraint=$con -J r4heat_${KEY} --export=ALL,KEY=$KEY,GPU="$gpu",TASK=$t,NTASKS=$n,MODE=gpu \
        scripts/eci/atlas_v37_r4heat.sh)
    echo "  $gpu ($con) task $t/$n -> $j"
    ids+=($j)
  done
done < <(python -c "import json,sys; [print(f'{g}\t{n}') for g,n in json.loads(sys.argv[1]).items()]" "$G")
dep=$(IFS=:; echo "${ids[*]}")
a=$(sbatch --parsable -p defaultp --gres=none --cpus-per-task=4 --mem=32G --time=00:30:00 -J r4heat_asm_${KEY} \
    --dependency=afterok:$dep --export=ALL,KEY=$KEY,MODE=assemble,MOVE=${MOVE:-} scripts/eci/atlas_v37_r4heat.sh)
echo "  assemble -> $a (after $dep, MOVE=${MOVE:-})"
