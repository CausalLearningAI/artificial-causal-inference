#!/bin/bash
#
# CRIM13 SAE-knob test - submit-only launcher (rules: the docstring of scripts/benchmark/crim13/align_knobs.py).
# Every job runs scripts/benchmark/crim13/align_job.sh (staging in /localhome, nvidia-smi every 60 s). GPU jobs:
# -p gpu, L40S | A40 | A100, never gpu100, BeeGFS-less nodes excluded.
#   bash align_knobs.sh encode                  rebuild the d3l_640_L20 token store (4 jobs, prints colon-joined ids)
#   bash align_knobs.sh smoke                   2-video smoke store per half (selftest input)
#   bash align_knobs.sh selftest [deps]         tiny run of align_knobs.py on the smoke store
#   bash align_knobs.sh train <arm> [deps]      one arm x 3 seeds x 2 folds
#   bash align_knobs.sh sheets <arm> [deps]     CPU: sniff contact sheets of <arm> (align_sheets.sniff_sheet)
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
X="--exclude=gpu242,gpu150,gpu230,gpu241,gpu240"
BIG='L40S|A40|A100'
LOGS=results/vision/benchmark_crim13/align/logs
mkdir -p $LOGS
sub() { sbatch --parsable -p gpu --gres=gpu:1 $X --constraint="$BIG" --output=$LOGS/%x_%j.out "$@" \
          scripts/benchmark/crim13/align_job.sh; }
case "${1:-}" in
  encode)
    ids=()
    for H in train test; do for i in 0 1; do
      ids+=($(sub --export=ALL,PY=align_knobs_encode,EXTRA="--half $H --shard $i --nshard 2 --workers 6 --threads 3" \
        --cpus-per-task=16 --mem=110G --time=01:30:00 --job-name=kn_enc_${H}$i))
    done; done
    (IFS=:; echo "${ids[*]}") ;;
  smoke)
    ids=()
    for H in train test; do
      ids+=($(sub --export=ALL,PY=align_knobs_encode,EXTRA="--half $H --shard 0 --nshard 2 --smoke 3 --max-per-video 60" \
        --cpus-per-task=8 --mem=48G --time=00:30:00 --job-name=kn_smoke_$H))
    done
    (IFS=:; echo "${ids[*]}") ;;
  selftest)
    D=${2:-}
    sub --export=ALL,PY=align_knobs,EXTRA="--selftest" ${D:+--dependency=afterok:$D} --cpus-per-task=8 --mem=48G \
      --time=00:40:00 --job-name=kn_selftest ;;
  train)
    A=$2; D=${3:-}
    T=${TIME:-03:00:00}
    sub --export=ALL,PY=align_knobs,EXTRA="--arm $A" ${D:+--dependency=afterok:$D} --cpus-per-task=8 --mem=180G \
      --time=$T --job-name=kn_tr_$A ;;
  sheets)
    A=$2; D=${3:-}
    sbatch --parsable -p defaultp --cpus-per-task=4 --mem=16G --time=00:30:00 --job-name=kn_sheets_$A \
      --output=$LOGS/%x_%j.out ${D:+--dependency=afterok:$D} --export=ALL,PY=align_knobs,EXTRA="--sheets $A" \
      scripts/benchmark/crim13/align_job.sh ;;
  *) echo "usage: $0 encode | smoke | selftest [deps] | train <arm> [deps] | sheets <arm> [deps]"; exit 1 ;;
esac
