#!/bin/bash
#
# Re-score finished runs on full val and redraw their error figures.
# See regen_confusion_figs.py docstring.
#
#   TAGS="patchgrid256_dinov2_ft_b4 patchgrid256_dinov2_ft_b2" \
#       sbatch scripts/mice_behavior/regen_confusion_figs.sh
#
# Redrawing after a figure-code change needs no GPU at all once val_probs.npz exists:
#   python scripts/mice_behavior/regen_confusion_figs.py --tag <tag> --from-cache
#
#SBATCH --job-name=regen_figs
#SBATCH --output=logs/regen_figs_%j.out
#SBATCH --error=logs/regen_figs_%j.err
#SBATCH --time=04:00:00
# gpu, not gpu100: this is a single inference pass over 144k val samples, and the H100
# partition is where the training sweep lives. No reason to contend for it here.
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
# The JPEG cache is MEMORY-MAPPED rather than read into a dict (which is what forces
# train_online_aug.sh to 180G), but the cgroup still charges those pages, and 12 decode workers
# fork the 444k-entry index. 64G survived one run and was OOM-killed entering the second.
#SBATCH --mem=120G
# L40S (48 GB), not A100: gpu238 is the cluster's only A100 node and the training sweep occupies
# it, so pinning A100 queues this behind every arm for hours. Inference on DINOv2-base needs
# neither the memory nor the throughput. Override with GRES=gpu:A100:1 when the node is free.
#SBATCH --gres=gpu:L40S:1

module load conda
conda activate crl
export PYTHONUNBUFFERED=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
mkdir -p logs

TAG_ARGS=""
for t in ${TAGS:-patchgrid256_dinov2_ft_b4}; do
    TAG_ARGS="${TAG_ARGS} --tag ${t}"
done

python -u scripts/mice_behavior/regen_confusion_figs.py \
    ${TAG_ARGS} \
    --n-rows "${N_ROWS:-10}" \
    --context "${CONTEXT:-3}" \
    --batch-size "${BATCH_SIZE:-64}" \
    --decode-workers "${DECODE_WORKERS:-12}" \
    --read-workers "${READ_WORKERS:-32}" \
    --jpeg-cache-file "${JPEG_CACHE_FILE:-dataset/mice/v1/jpegcache_k2}"
