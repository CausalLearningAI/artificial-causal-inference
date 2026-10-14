#!/bin/bash
# Plain file download of the V-JEPA 2 ViT-L weights to the HF hub cache (no model loading). CPU job.
#   sbatch scripts/eci/vjepa_download.sh
#SBATCH --job-name=vjepa_dl
#SBATCH --output=results/vision/eci_vjepa/logs/%x_%j.out
#SBATCH --time=00:30:00
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
module load conda
conda activate crl
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
python -c "
from huggingface_hub import snapshot_download
for r in ['facebook/vjepa2-vitl-fpc64-256']:
    p = snapshot_download(r, allow_patterns=['*.json', '*.safetensors', '*.txt', '*.md'])
    print(r, p, flush=True)
"
ls -la ~/.cache/huggingface/hub/models--facebook--vjepa2-vitl-fpc64-256/snapshots/*/
