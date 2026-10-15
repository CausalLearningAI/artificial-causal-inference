#!/bin/bash
# Plain file download of DINOv2-large (ViT-L/14) and DINOv2-giant (ViT-g/14) to the HF hub cache (no model loading).
# CPU job:  sbatch scripts/eci/big_encoder_download.sh
#SBATCH --job-name=be_dl
#SBATCH --output=results/vision/eci_big_encoder/logs/%x_%j.out
#SBATCH --time=01:00:00
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
module load conda
conda activate crl
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
python -c "
from huggingface_hub import snapshot_download
for r in ['facebook/dinov2-large', 'facebook/dinov2-giant']:
    p = snapshot_download(r, allow_patterns=['*.json', '*.safetensors', '*.txt', '*.md'])
    print(r, p, flush=True)
"
ls -la ~/.cache/huggingface/hub/models--facebook--dinov2-large/snapshots/*/ ~/.cache/huggingface/hub/models--facebook--dinov2-giant/snapshots/*/
