#!/bin/bash
#
# Foreground pipeline step 5: all 2,592,000 frames -> DINOv2 (448, no crop) + foreground mask +
# fg448 SAE -> dataset/mice/v1/eci/codes/<SAE>/ (codes_max, codes_mean, n_fg). 24 shards,
# finished shards are skipped (resubmit to resume). Then merge + verify + delete shards:
#
#   jid=$(sbatch --parsable scripts/eci/fg_encode_all.sh)
#   sbatch --dependency=afterok:${jid} scripts/eci/fg_encode_merge.sh
#
#SBATCH --job-name=eci_fg_enc
#SBATCH --output=logs/eci_fg_enc_%A_%a.out
#SBATCH --error=logs/eci_fg_enc_%A_%a.err
#SBATCH --array=0-23
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

module load conda
conda activate crl

export PYTHONUNBUFFERED=1
set -euo pipefail

cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
mkdir -p logs

python -u scripts/eci/fg_encode_all.py --sae ${SAE:-matryoshka_btk_1024_k16_fg448_s0} --n-shards 24 \
    --shard ${SLURM_ARRAY_TASK_ID} --num-workers 22 ${EXTRA_ARGS:-}
