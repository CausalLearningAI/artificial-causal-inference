#!/bin/bash
# Encoder study (eci_encoders.sh): plain file download of the extra encoder checkpoints to the HF hub cache (CPU job,
# no model loading). DINOv3 repos are gated: an access error is printed per repo and the job continues.
#   sbatch scripts/benchmark/mabe22/eci_enc_download.sh
#SBATCH --job-name=enc_dl
#SBATCH --output=results/vision/benchmark_mabe22/eci/encoders/logs/%x_%j.out
#SBATCH --time=01:00:00
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
module load conda
conda activate crl
set -uo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
python -u - <<'PY'
from huggingface_hub import snapshot_download
for r in ['facebook/dinov3-vitl16-pretrain-lvd1689m', 'facebook/dinov3-vith16plus-pretrain-lvd1689m',
          'facebook/vjepa2-vith-fpc64-256', 'facebook/vjepa2-vitg-fpc64-256', 'facebook/vjepa2-vitg-fpc64-384']:
    try:
        p = snapshot_download(r, allow_patterns=['*.json', '*.safetensors', '*.md'])
        print('OK', r, p, flush=True)
    except Exception as e:
        print('FAIL', r, type(e).__name__, str(e)[:300].replace('\n', ' '), flush=True)
PY
