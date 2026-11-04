#!/bin/bash
# v2.4: after a training job's checkpoints exist: codes CPU array + per-model scoring chains.  usage: v24_score_submit.sh DOM K1,K2,...
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
DOM=$1; K=$2; NT=24
j=$(sbatch --parsable -J v24codes_$DOM --array=0-$((NT-1)) -c 8 --mem=32G --time=01:30:00 scripts/eci/v2/v24_cpu.sh v24_train.py codes --dom $DOM --keys $K --task @TASK@ --ntask $NT)
echo "codes $j"
for k in ${K//,/ }; do DOM=$DOM KEY=$k NT=$NT sbatch --export=ALL --dependency=afterok:$j -J v24post_$k scripts/eci/v2/v24_post.sh; done
