#!/bin/bash
#
# Latency-to-first-bout NES (scripts/eci/run_nes_latency.py) on the max-, mean- and SOMP-pooled frame codes of one SAE:
#   <nes root>/<SAE>/[<set>/]maxpool_latency/, <SAE>_mean/[<set>/]meanpool_latency/, <SAE>_somp/[<set>/]somp_latency/
# CPU only. Usage: DOMAIN=ants SAE=matryoshka_btk_1024_k16_antsfg_s0 EXTRA_ARGS="--analysis-set pairs" sbatch --export=ALL scripts/eci/run_nes_latency.sh
#
#SBATCH --job-name=eci_nes_latency
#SBATCH --output=logs/eci_nes_latency_%j.out
#SBATCH --error=logs/eci_nes_latency_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=defaultp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

module load conda
conda activate crl
export PYTHONUNBUFFERED=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
mkdir -p logs
for agg in ${AGGS:-max mean somp}; do
  sae="${SAE}"; [ "$agg" != max ] && sae="${SAE}_${agg}"
  echo "=== ${sae}: latency NES (codes_${agg})"
  python -u scripts/eci/run_nes_latency.py --domain "${DOMAIN:-mice}" --sae "$sae" --frame-pooling "$agg" ${EXTRA_ARGS:-}
done
