#!/bin/bash
# Submit the post-processing chain of one domain after its encode jobs: merge -> fire -> link (CPU) -> align (GPU) and
# NES / NESS per series (CPU, scripts/eci/v2/nes.sh) -> heat (CPU) -> manifest.
#   DEPS=<colon list of encode job ids> bash scripts/eci/v2/chain.sh <domain>
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
D=$1
C="sbatch --parsable -p defaultp --cpus-per-task=8"
G="sbatch --parsable -p gpu --gres=gpu:1 --constraint=L40S|A100|A40 --exclude=gpu240,gpu241,gpu242"
m=$($C -J v2_merge_$D --time=04:00:00 --mem=64G --dependency=afterok:$DEPS --export=ALL,STEP=merge,EXTRA="--domain $D" scripts/eci/v2/run.sh)
f=$($C -J v2_fire_$D --time=06:00:00 --mem=120G --dependency=afterok:$m --export=ALL,STEP=fire,EXTRA="--domain $D" scripts/eci/v2/run.sh)
l=$($C -J v2_link_$D --time=00:20:00 --mem=4G --dependency=afterok:$f --export=ALL,STEP=link,EXTRA="--domain $D" scripts/eci/v2/run.sh)
a=$($G -J v2_align_$D --time=03:00:00 --mem=150G --dependency=afterok:$l --export=ALL,STEP=align,EXTRA="--domain $D" scripts/eci/v2/run.sh)
echo "$D merge $m fire $f link $l align $a"
