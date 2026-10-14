#!/bin/bash
# Plain file download of DINOv2-with-registers-base to the HF hub cache (no model loading). CPU job.
#   sbatch scripts/eci/quick_repr_download.sh
#SBATCH --job-name=qr_dl
#SBATCH --output=results/vision/eci_quick_repr/logs/%x_%j.out
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
for r in ['facebook/dinov2-with-registers-base']:
    p = snapshot_download(r, allow_patterns=['*.json', '*.safetensors', '*.txt', '*.md'])
    print(r, p, flush=True)
"
ls -la ~/.cache/huggingface/hub/models--facebook--dinov2-with-registers-base/snapshots/*/
