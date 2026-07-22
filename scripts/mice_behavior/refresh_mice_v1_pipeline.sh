#!/bin/bash
#
# Refreshes the mice v1 pipeline artifacts needed after new annotated videos are
# added (15 new observations across pools rd64, rd29, rd41_3 as of 2026-07-20):
# frames -> annotations.csv -> HF dataset -> pair_labels.parquet.
#
# Deliberately does NOT re-extract CLS or patch_grid4 embeddings here — the
# patchgrid256 (online) search/training scripts read raw frames + annotations.csv
# + pair_labels.parquet directly and never touch those cached embedding files, so
# they aren't a dependency for re-queuing that search. Re-extract them separately
# (get_embeddings.py, extract_patch_embeddings.py --overwrite) before re-running
# any of the 4x4-pooled-patch-grid variants (patchgrid4x4_*).
#
# Usage:
#   sbatch scripts/mice_behavior/refresh_mice_v1_pipeline.sh
#
#SBATCH --job-name=refresh_mice_v1_pipeline
#SBATCH --output=logs/refresh_mice_v1_pipeline_%j.out
#SBATCH --error=logs/refresh_mice_v1_pipeline_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=visualize
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

module load conda
conda activate crl
export PYTHONUNBUFFERED=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
mkdir -p logs

echo "=== [1/4] Extract frames (fills gaps only, overwrite.frames=false) ==="
python -u src/dataset/get_frames.py experiment=mice/v1

echo "=== [2/4] Regenerate annotations.csv ==="
python -u -m src.dataset.get_annotations experiment=mice/v1 overwrite.annotations=true

echo "=== [3/4] Regenerate HF dataset ==="
python -u src/dataset/get_dataset.py experiment=mice/v1 overwrite.hf=true

echo "=== [4/4] Regenerate pair_labels.parquet ==="
python -u -c "
from src.mice_behavior.build_pair_labels import build_pair_labels
out = build_pair_labels(overwrite=True)
print(f'Wrote {out}')
"

echo "=== Pipeline refresh complete ==="
