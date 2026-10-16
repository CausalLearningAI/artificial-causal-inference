#!/bin/bash
#
# ECI v0 (original ECI paper recipe) on ants v2: frames -> DINOv2-S tokens -> TopK SAE -> mean-pooled
# codes -> Welch search + gate checks. See scripts/eci/v0/ants_v2_v0.py. One GPU, ~1 h.
#
#   sbatch scripts/eci/v0/ants_v2_v0.sh
#
# Big intermediates (token shards, frame codes, SAE): /mnt/beegfs/locatgrp/shared/rcadei/eci_v0/ants_v2/
# Small results: results/vision/ants/eci_v0/   GPU utilisation log: logs/eci_v0_ants_v2_<job>_gpu.csv
#
#SBATCH --job-name=eci_v0_ants_v2
#SBATCH --output=logs/eci_v0_ants_v2_%j.out
#SBATCH --error=logs/eci_v0_ants_v2_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1

module load conda
conda activate crl

export PYTHONUNBUFFERED=1
set -euo pipefail

REPO=/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
cd "$REPO"
mkdir -p logs results/vision/ants/eci_v0

STAGE=/localhome/$USER/$SLURM_JOB_ID
mkdir -p "$STAGE/videos" "$STAGE/out"
trap 'rm -rf "$STAGE"; kill $SMI_PID 2>/dev/null || true' EXIT

nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total --format=csv -l 30 \
    > "logs/eci_v0_ants_v2_${SLURM_JOB_ID}_gpu.csv" &
SMI_PID=$!

# stage the 44 valid source videos (~1.3 GB) to local disk
python - <<'EOF' > "$STAGE/videos.txt"
import pandas as pd
e = pd.read_csv('data/ants/v2/experiment.csv')
print('\n'.join(e[e.valid == 1].observation_file))
EOF
while read -r f; do cp "data/ants/v2/observations/source/$f" "$STAGE/videos/"; done < "$STAGE/videos.txt"
echo "staged $(ls "$STAGE/videos" | wc -l) videos"

python -u scripts/eci/v0/ants_v2_v0.py --video-dir "$STAGE/videos" --out-dir "$STAGE/out" \
    --num-workers 14 ${EXTRA_ARGS:-}

cp "$STAGE/out/"* results/vision/ants/eci_v0/
echo "copied results to results/vision/ants/eci_v0/"
