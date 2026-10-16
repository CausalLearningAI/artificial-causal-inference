#!/bin/bash
#
# CalMS21 ECI role swap - SAE KNOB SWEEP for a sniffing neuron (rules: the docstring of eci_knobs.py, pre-registered
# before any sweep SAE was trained). Submit-only launcher; every job runs eci_swap_job.sh (staging in /localhome,
# nvidia-smi log every 60 s to swap/logs/gpu_<job>_<id>.csv). GPU: -p gpu, L40S | A100 | A40, never gpu100, no
# BeeGFS-less nodes. CPU (sheets): -p defaultp.
#   bash eci_knobs.sh selftest          tiny run (B2, 300 steps, seed 600, 2 split seeds) -> swap/knobs_selftest
#   bash eci_knobs.sh train [deps]      the 8 arms in 6 jobs (B0+B1, B2, B3, B4+B5, B6, B7), seeds 600-602
#   bash eci_knobs.sh sheet <json>...   contact sheets (eci_swap_sheet.py) of sheet JSONs
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
export CALMS_OUT=results/vision/benchmark_calms21/swap
export CALMS_TOK=/mnt/beegfs/locatgrp/shared/rcadei/eci_tok/pairpush/calms21
X="--exclude=gpu240,gpu242,gpu150,gpu230,gpu241"
BIG='L40S|A100|A40'
LOGS=$CALMS_OUT/logs
mkdir -p $LOGS
sub() { sbatch --parsable -p gpu --gres=gpu:1 $X --constraint="$BIG" --cpus-per-task=8 --mem=120G --output=$LOGS/%x_%j.out "$@" scripts/benchmark/calms21/eci_swap_job.sh; }
case "${1:-}" in
  selftest) PY=eci_knobs EXTRA="--arms B2 --selftest" sub --time=00:40:00 --job-name=knob_selftest ;;
  train)
    D=${2:-}
    for g in "B0 B1:01:00:00" "B2:01:00:00" "B3:01:40:00" "B4 B5:01:30:00" "B6:01:00:00" "B7:01:00:00"; do
      arms=${g%%:*}; t=${g#*:}
      PY=eci_knobs EXTRA="--arms $arms" sub ${D:+--dependency=afterok:$D} --time=$t --job-name=knob_${arms// /}
    done ;;
  sheet) shift
    PY=eci_swap_sheet EXTRA="$*" sbatch --parsable -p defaultp --cpus-per-task=6 --mem=16G --time=00:20:00 \
      --output=$LOGS/%x_%j.out --job-name=knob_sheet scripts/benchmark/calms21/eci_swap_job.sh ;;
  *) echo "usage: $0 selftest | train [deps] | sheet <json>..."; exit 1 ;;
esac
