#!/bin/bash
# Atlas v37 sweeper, full chain (1 CPU, sleeps inside the job): for each R4 build key in KEYS (e.g. the pooled ants
# series "ants_v2p ants_v3p ants_v4p"), waits for the R4 NES result set <nes root>/<best d2 SAE>_fnorm (root,
# maxpool_bouts/ and maxpool_latency/ summary.csv), then submits DINES (scripts/eci/atlas_v37_methods.sh), the atlas
# build after it (scripts/eci/build_atlas_v37.sh) and an assemble after every build so far.
#   sbatch --export=ALL,KEYS="ants_v2p ants_v3p ants_v4p" scripts/eci/build_atlas_v37_sweep2.sh
#SBATCH --job-name=a37sweep2
#SBATCH --output=results/vision/eci_atlas_v37/logs/%x_%j.out
#SBATCH --time=08:00:00
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
set -uo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
KEYS=${KEYS:?}
declare -A ROOTS=([ants_v2p]=results/vision/ants/eci_r4v2p/nes [ants_v3p]=results/vision/ants/eci_r4v3p/nes
                  [ants_v4p]=results/vision/ants/eci_r4v4p/nes [mice]=results/vision/mice/eci/nes
                  [frogs]=results/vision/frogs/eci/nes)
declare -A RDOM=([ants_v2p]=ants_pooled [ants_v3p]=ants_pooled [ants_v4p]=ants_pooled [mice]=mice [frogs]=frogs)
deps=""
for k in $KEYS; do
  while true; do
    best=results/vision/eci_rollout_r4/${RDOM[$k]}/sae/best.json
    if [ -f $best ]; then
      R=${ROOTS[$k]}/$(python -c "import json;print(json.load(open('$best'))['d2']['name'])")_fnorm
      if [ -f $R/summary.csv ] && [ -f $R/maxpool_bouts/summary.csv ] && [ -f $R/maxpool_latency/summary.csv ] \
         && ! squeue -u $USER -h -o "%j" | grep -q "^r4_nes_.*${k#ants_}"; then
        break
      fi
    fi
    sleep 120
  done
  echo "$(date +%H:%M) $k: NES ready ($R)"
  m=$(RES=$R sbatch --parsable --export=ALL -J v37m_$k scripts/eci/atlas_v37_methods.sh)
  j=$(sbatch --parsable --dependency=afterok:$m --export=ALL,ARGS="--dom $k --model new --steps prep,checks,data,patch,arena,render,dump" \
      -J a37new_$k scripts/eci/build_atlas_v37.sh)
  deps="$deps:$j"
  a=$(sbatch --parsable --dependency=afterok${deps} --export=ALL,ARGS="--steps assemble" -J a37asm --cpus-per-task=2 \
      --mem=8G --time=00:30:00 scripts/eci/build_atlas_v37.sh)
  echo "$(date +%H:%M) $k: methods $m -> build $j -> assemble $a"
done
