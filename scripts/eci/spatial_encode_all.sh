#!/bin/bash
#
# Spatial frame features (pairwise co-activation, zone pooling, blob measures) for all frames of one foreground SAE
# (scripts/eci/spatial_encode_all.py): tokens, mask and SAE patch codes recomputed with the SAE's own pipeline
# -> <domain eci dir>/codes/<SAE>_spatial/shards/. N_SHARDS shards (array 0..N_SHARDS-1), finished shards are
# skipped (resubmit to resume). Then merge + verify + delete shards:
#
#   jid=$(DOMAIN=mice SAE=matryoshka_btk_1024_k16_fg448_s0 sbatch --parsable --export=ALL --array=0-23 scripts/eci/spatial_encode_all.sh)
#   DOMAIN=mice SAE=matryoshka_btk_1024_k16_fg448_s0 sbatch --export=ALL --dependency=afterok:${jid} scripts/eci/spatial_encode_merge.sh
#   (ants: N_SHARDS=8 and --array=0-7 for both)
#
#SBATCH --job-name=eci_spatial_enc
#SBATCH --output=logs/eci_spatial_enc_%A_%a.out
#SBATCH --error=logs/eci_spatial_enc_%A_%a.err
#SBATCH --array=0-23
#SBATCH --time=06:00:00
#SBATCH --partition=gpu100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --gres=gpu:1

module load conda
conda activate crl

export PYTHONUNBUFFERED=1
set -euo pipefail

cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
mkdir -p logs

python -u scripts/eci/spatial_encode_all.py --domain "${DOMAIN:-mice}" --sae "${SAE:?set SAE}" --n-shards "${N_SHARDS:-24}" \
    --shard ${SLURM_ARRAY_TASK_ID} --num-workers 22 ${EXTRA_ARGS:-}
