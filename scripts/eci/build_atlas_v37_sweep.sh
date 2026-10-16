#!/bin/bash
# Atlas v37 sweeper (1 CPU, sleeps inside the job, no polling from the login node): for each R4 build key in KEYS
# (default "mice ants_v4"), waits until its DINES / family C outputs of scripts/eci/atlas_v37_methods.py exist (they are
# written after the R4 NES run), then submits its new-model build (scripts/eci/build_atlas_v37.sh) and, after every
# submitted build, one assemble job.
#   sbatch --export=ALL,KEYS="mice ants_v4" scripts/eci/build_atlas_v37_sweep.sh
#SBATCH --job-name=a37sweep
#SBATCH --output=results/vision/eci_atlas_v37/logs/%x_%j.out
#SBATCH --time=09:00:00
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
set -uo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
KEYS=${KEYS:-"mice ants_v4"}
declare -A RDOM=([mice]=mice [ants_v4]=ants_v4)
declare -A MKEY=([mice]=mice [ants_v4]=ants_eci_r4v4)
deps=""
for k in $KEYS; do
  while true; do
    best=results/vision/eci_rollout_r4/${RDOM[$k]}/sae/best.json
    if [ -f $best ]; then
      sae=$(python -c "import json;print(json.load(open('$best'))['d2']['name'])")_fnorm
      M=results/vision/eci_atlas_v37/methods/${MKEY[$k]}/$sae
      if [ -f $M/dines/maxpool_latency/summary.csv ] && [ -f $M/dines/summary.csv ] && [ -f $M/dines/maxpool_bouts/summary.csv ] \
         && { [ "$k" != "mice" ] || [ -d $M/familyC_dines/maxpool_latency ]; }; then
        break
      fi
    fi
    sleep 120
  done
  echo "$(date +%H:%M) $k: methods ready ($M)"
  j=$(sbatch --parsable --export=ALL,ARGS="--dom $k --model new --steps prep,data,patch,arena,render,dump" -J a37new_$k \
      scripts/eci/build_atlas_v37.sh)
  echo "$(date +%H:%M) $k: build -> $j"
  deps="$deps:$j"
  a=$(sbatch --parsable --dependency=afterok${deps} --export=ALL,ARGS="--steps assemble" -J a37asm --cpus-per-task=2 \
      --mem=8G --time=00:30:00 scripts/eci/build_atlas_v37.sh)
  echo "$(date +%H:%M) assemble after$deps -> $a"
done
