#!/bin/bash
#
# SOMP spatial aggregation for all frames of one SAE (scripts/eci/somp_encode_all.py): DINOv2 tokens
# recomputed with the SAE's own pipeline (foreground 448 or full-frame 224) -> SOMP over the SAE decoder
# -> <domain eci dir>/codes/<SAE>_somp/. 24 shards, finished shards are skipped (resubmit to resume).
# Then merge + verify + delete shards:
#
#   jid=$(DOMAIN=mice SAE=matryoshka_btk_1024_k16_fg448_s0 sbatch --parsable --export=ALL scripts/eci/somp_encode_all.sh)
#   DOMAIN=mice SAE=matryoshka_btk_1024_k16_fg448_s0 sbatch --export=ALL --dependency=afterok:${jid} scripts/eci/somp_encode_merge.sh
#
#SBATCH --job-name=eci_somp_enc
#SBATCH --output=logs/eci_somp_enc_%A_%a.out
#SBATCH --error=logs/eci_somp_enc_%A_%a.err
#SBATCH --array=0-23
#SBATCH --time=05:00:00
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

python -u scripts/eci/somp_encode_all.py --domain "${DOMAIN:-mice}" --sae "${SAE:?set SAE}" --k "${K:-16}" --n-shards 24 \
    --shard ${SLURM_ARRAY_TASK_ID} --num-workers 22 ${EXTRA_ARGS:-}
