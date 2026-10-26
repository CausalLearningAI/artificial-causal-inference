#!/bin/bash
# ECI v2.0: NES + NESS (scripts/eci/v2/nes.sh, unchanged search methods) on the v2.0 outcome sets of one domain, prefixes
# 128 / 256 / 2048. Sets: <sae>_hmm (P(on); per-video mean = mean P(on), the PRIMARY outcome), <sae>_smooth (continuous,
# secondary), <sae>_hmmv (Viterbi on-state, V2_ZERO_THR=1: bout rate and latency from the HMM on-state).
# Mice: the v1 layout only (mice v2 is unlabelled and has no genotype contrast; it enters training and inference).
#   bash scripts/eci/v2/v20_nes.sh mice|ants|frogs
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
D=$1
NAME=$(python -c "import json;print(json.load(open('results/vision/eci_v2_0/$D/best.json'))['name'])")
sub() {  # KIND ANTS_ECI SET ZERO
  sbatch --parsable -J v20nes_$1${2:+_$2}$3 --output=results/vision/eci_v2_0/logs/%x_%j.out \
    --export=ALL,KIND=$1,ANTS_ECI=$2,SAE=$NAME$3,PREFIXES=128,256,2048,V2_ZERO_THR=$4,TADPOLE_TAG=_r4 scripts/eci/v2/nes.sh
}
for set in _hmm _smooth _hmmv; do
  z=0; [ $set = _hmmv ] && z=1
  case $D in
    mice) sub mice "" $set $z ;;
    ants) for e in eci_r4v2p eci_r4v3p eci_r4v4p; do sub ants $e $set $z; done ;;
    frogs) sub frogs "" $set $z; sub tadpoles "" $set $z ;;
  esac
done
