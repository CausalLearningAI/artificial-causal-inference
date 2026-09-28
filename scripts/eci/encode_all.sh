#!/bin/bash
#
# ECI step 3: DINOv2-base + seed-0 20-epoch Matryoshka BatchTopK SAE on ALL 2,592,000 mice v1
# frames, sharded by row range (16 shards). Finished shards are skipped, so the
# array can simply be resubmitted. Then merge + verify on CPU:
#
# Usage:
#   jid=$(sbatch --parsable scripts/eci/encode_all.sh)          # SAE=<name> to pick another SAE
#   sbatch --dependency=afterok:${jid} scripts/eci/encode_merge.sh
#
#   # v2 (1 fps SAE, no CLS; verification against the 1 fps token store):
#   export SAE=matryoshka_btk_1024_k16_fps1_s0 OUTPUTS=codes_mean,codes_max \
#          TOKENS_DIR=dataset/mice/v1/eci/train_tokens/dinov2_base_l-1_fps1
#   jid=$(sbatch --parsable --export=ALL scripts/eci/encode_all.sh)
#   sbatch --export=ALL --dependency=afterok:${jid} scripts/eci/encode_merge.sh
#
# 24 CPUs per shard so at most ~4 shards share a node (16-CPU shards packed 6 per
# 96-CPU node ran at half speed from JPEG-decoding contention).
#
#SBATCH --job-name=eci_encode
#SBATCH --output=logs/eci_encode_%A_%a.out
#SBATCH --error=logs/eci_encode_%A_%a.err
#SBATCH --array=0-15
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --gres=gpu:1

module load conda
conda activate crl

export PYTHONUNBUFFERED=1
set -euo pipefail

cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
mkdir -p logs

python -u scripts/eci/encode_all.py \
    --sae ${SAE:-matryoshka_btk_1024_k16_ep20_s0} \
    --outputs ${OUTPUTS:-codes_mean,codes_max,cls_l-1} \
    --tokens-dir ${TOKENS_DIR:-dataset/mice/v1/eci/train_tokens/dinov2_base_l-1} \
    --n-shards 16 \
    --shard ${SLURM_ARRAY_TASK_ID} \
    --batch-size 256 \
    --num-workers ${WORKERS:-22} \
    ${EXTRA_ARGS:-}
