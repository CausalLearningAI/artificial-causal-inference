#!/bin/bash
#
# MABe22 mouse ECI - KNOB SWEEP on the P2 tokens. Rules (pre-registered) and arms: scripts/benchmark/mabe22/eci_knobs.py.
# Submit-only launcher; every job runs scripts/benchmark/mabe22/eci_enc_job.sh (GPU utilisation logged every 60 s,
# staging in /localhome/$USER/<job>_mabe, removed on exit). Needs the P2 store (tag p2d3l_512): `eci_p2.sh encode`.
#   selftest            R-precision GPU vs numpy reference (no store needed)
#   smoke               2 shards per set, 300 steps, arms B1 B6 B2D (width 512, centring, delta), outputs *_smoke
#   full                3 GPU jobs: z_t arms B0 B1 B2 B4 B5 | B3 B7 B6 | change-channel arms B0D B2D
#   sheet ARM           CPU job: contact sheet of ARM's oral_oral neuron (seed 600)
# DEP=<job id> adds --dependency=afterok:<job id>.
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
export MABE_ENC_TOK=/mnt/beegfs/locatgrp/shared/rcadei/eci_tok/pairpush/mabe22
export MABE_KNOB_STORE=/mnt/beegfs/locatgrp/shared/rcadei/eci_tok/knobs_mabe22
G="-p gpu --gres=gpu:1 --constraint=L40S|A100|A40 --exclude=gpu242,gpu150,gpu230,gpu241,gpu240 --cpus-per-task=8"
LOGS=results/vision/benchmark_mabe22/eci/knobs/logs
mkdir -p $LOGS
D=${DEP:+--dependency=afterok:$DEP}
sub() { sbatch --parsable $G --output=$LOGS/%x_%j.out $D "$@" scripts/benchmark/mabe22/eci_enc_job.sh; }
case "${1:-}" in
  selftest) PY=eci_knobs EXTRA="selftest" sub --mem=16G --time=00:15:00 --job-name=kn_self ;;
  smoke)    PY=eci_knobs EXTRA="run --arms B1 B6 B2D --seeds 600 --smoke" sub --mem=64G --time=00:40:00 --job-name=kn_smoke ;;
  full)
    PY=eci_knobs EXTRA="run --arms B0 B1 B2 B4 B5" sub --mem=96G --time=02:30:00 --job-name=kn_zt1
    PY=eci_knobs EXTRA="run --arms B3 B7 B6" sub --mem=96G --time=02:30:00 --job-name=kn_zt2
    PY=eci_knobs EXTRA="run --arms B0D B2D" sub --mem=128G --time=02:00:00 --job-name=kn_delta ;;
  sheet)
    sbatch --parsable -p defaultp --cpus-per-task=4 --mem=16G --time=00:30:00 \
      --output=$LOGS/%x_%j.out $D --job-name=kn_sheet \
      --export=ALL,PY=eci_knobs,EXTRA="sheet --arm ${2:?arm}" scripts/benchmark/mabe22/eci_enc_job.sh ;;
  *) echo "usage: [DEP=<job id>] $0 selftest | smoke | full | sheet ARM"; exit 1 ;;
esac
