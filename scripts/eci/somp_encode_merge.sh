#!/bin/bash
#
# Merge + verify (64 random frames recomputed from scratch, per-shard codes_max alignment) + delete shards
# for scripts/eci/somp_encode_all.sh, then the <SAE>_mean link folder (--link-mean).
#
#SBATCH --job-name=eci_somp_merge
#SBATCH --output=logs/eci_somp_merge_%j.out
#SBATCH --error=logs/eci_somp_merge_%j.err
#SBATCH --time=03:00:00
#SBATCH --partition=gpu100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

module load conda
conda activate crl

export PYTHONUNBUFFERED=1
set -euo pipefail

cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
python -u scripts/eci/somp_encode_all.py --domain "${DOMAIN:-mice}" --sae "${SAE:?set SAE}" --k "${K:-16}" --n-shards 24 \
    --merge --verify --delete-shards
python -u scripts/eci/somp_encode_all.py --domain "${DOMAIN:-mice}" --sae "${SAE}" --link-mean
