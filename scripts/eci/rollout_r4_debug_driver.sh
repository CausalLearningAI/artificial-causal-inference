#!/bin/bash
# Submits the ants v2 encode shards to debug_gpu (QOS: at most 2 submitted jobs per user) as slots free up, after the
# train job succeeded; then the CPU tail. Runs as a small defaultp job (no polling from the login node).
#   sbatch --export=ALL,TRAIN=<jobid>,NSH=4 scripts/eci/rollout_r4_debug_driver.sh
#SBATCH --job-name=r4_driver
#SBATCH --output=results/vision/eci_rollout_r4/logs/%x_%j.out
#SBATCH --time=05:00:00
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
set -uo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
DOM=${DOM:-ants_v2}
st() { sacct -n -X -j $1 -o State%20 | head -1 | tr -d ' '; }
while true; do s=$(st $TRAIN); case $s in COMPLETED) break;; PENDING|RUNNING|"") sleep 60;; *) echo "train $TRAIN $s: stop"; exit 1;; esac; done
echo "$(date +%H:%M) train done"
ids=()
for i in $(seq 0 $((NSH - 1))); do
  while [ $(squeue -u $USER -p debug_gpu -h | wc -l) -ge 2 ]; do sleep 60; done
  j=$(sbatch --parsable -p debug_gpu --gres=gpu:1 --cpus-per-task=6 --mem=32G --time=00:55:00 -J r4_enc_${DOM}_$i \
      --export=ALL,STEP=encode,DOMAIN=$DOM,R4_AMP=fp16,EXTRA="--shard $i" scripts/eci/rollout_r4.sh)
  echo "$(date +%H:%M) shard $i -> $j"; ids+=($j); sleep 20
done
dep=$(IFS=:; echo "${ids[*]}")
p=$(sbatch --parsable --dependency=afterany:$dep -J r4_post_$DOM --export=ALL,DOMAIN=$DOM scripts/eci/rollout_r4_post.sh)
echo "$(date +%H:%M) post -> $p"
