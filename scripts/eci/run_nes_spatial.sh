#!/bin/bash
#
# NES (scripts/eci/run_nes.py, average-time outcome) and bout NES (scripts/eci/run_nes_bouts.py, event-rate outcome)
# on the spatial feature sets of scripts/eci/spatial_encode_all.py, each as its own result set:
#   <nes root>/<SAE>_pairs/[pairs/]              codes/<SAE>_pairs/codes_pairs.npy (pair co-activation strengths)
#   <nes root>/<SAE>_pairs/[pairs/]pairs_bouts/
#   <nes root>/<SAE>_zones/[pairs/]              codes/<SAE>_zones/codes_zones.npy (latent x zone max)
#   <nes root>/<SAE>_zones/[pairs/]zones_bouts/
# One prefix = every column (--prefixes all): these features have no Matryoshka order. Feature ids = columns of
# the codes folder's features.csv. Full grid, domain default nuisance, label-shuffle nulls. CPU only.
#
# Usage: DOMAIN=ants SAE=matryoshka_btk_1024_k16_antsfg_s0 EXTRA_ARGS="--analysis-set pairs" sbatch --export=ALL scripts/eci/run_nes_spatial.sh
#        TAGS="zones" ... to run one set only; NES_ARGS="--skip-frame" ... to run_nes.py only (the frame-level
#        illustration holds every frame of its videos in memory: ~20 GB at 8000 pair columns), BOUT_ARGS ... to run_nes_bouts.py only
#
#SBATCH --job-name=eci_nes_spatial
#SBATCH --output=logs/eci_nes_spatial_%j.out
#SBATCH --error=logs/eci_nes_spatial_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=defaultp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G

module load conda
conda activate crl
export PYTHONUNBUFFERED=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
mkdir -p logs
for tag in ${TAGS:-zones pairs}; do
  echo "=== ${SAE}_${tag}: mean-activation NES"
  python -u scripts/eci/run_nes.py --domain "${DOMAIN:-mice}" --sae "${SAE}_${tag}" --primary-pooling "$tag" --poolings "$tag" \
    --prefixes all ${NES_ARGS:-} ${EXTRA_ARGS:-}
  echo "=== ${SAE}_${tag}: bout NES"
  python -u scripts/eci/run_nes_bouts.py --domain "${DOMAIN:-mice}" --sae "${SAE}_${tag}" --frame-pooling "$tag" \
    --compare-pooling "$tag" --prefixes all ${BOUT_ARGS:-} ${EXTRA_ARGS:-}
done
