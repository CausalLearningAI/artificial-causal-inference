#!/bin/bash
#
# Merge + verify (alignment recomputed from scratch on 64 random frames) + delete shards for
# scripts/eci/fg_encode_all.sh.
#
#SBATCH --job-name=eci_fg_merge
#SBATCH --output=logs/eci_fg_merge_%j.out
#SBATCH --error=logs/eci_fg_merge_%j.err
#SBATCH --time=03:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1

module load conda
conda activate crl

export PYTHONUNBUFFERED=1
set -euo pipefail

cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
python -u scripts/eci/fg_encode_all.py --sae ${SAE:-matryoshka_btk_1024_k16_fg448_s0} --n-shards 24 \
    --merge --verify --delete-shards
