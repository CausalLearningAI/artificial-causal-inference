#!/bin/bash
#
# NES (scripts/eci/run_nes.py, mean activation) and bout NES (scripts/eci/run_nes_bouts.py) on the SOMP and the
# mean-pooled frame codes of one SAE, each as its own result set:
#   <nes root>/<SAE>_somp/            codes/<SAE>_somp/codes_somp.npy  (scripts/eci/somp_encode_all.py)
#   <nes root>/<SAE>_somp/somp_bouts/
#   <nes root>/<SAE>_mean/            codes/<SAE>_mean/codes_mean.npy  (symlink, --link-mean)
#   <nes root>/<SAE>_mean/meanpool_bouts/
# Full grid (no --primary-only), domain default nuisance. CPU only.
#
# Usage: DOMAIN=ants SAE=matryoshka_btk_1024_k16_antsfg_s0 sbatch --export=ALL scripts/eci/run_nes_somp.sh
#        AGGS="somp" ... to run one aggregation only
#
#SBATCH --job-name=eci_nes_somp
#SBATCH --output=logs/eci_nes_somp_%j.out
#SBATCH --error=logs/eci_nes_somp_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=defaultp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

module load conda
conda activate crl
export PYTHONUNBUFFERED=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
mkdir -p logs
for agg in ${AGGS:-somp mean}; do
  echo "=== ${SAE}_${agg}: mean-activation NES"
  python -u scripts/eci/run_nes.py --domain "${DOMAIN:-mice}" --sae "${SAE}_${agg}" --primary-pooling "$agg" --poolings "$agg"
  echo "=== ${SAE}_${agg}: bout NES"
  python -u scripts/eci/run_nes_bouts.py --domain "${DOMAIN:-mice}" --sae "${SAE}_${agg}" --frame-pooling "$agg" \
    --compare-pooling "$agg"
done
