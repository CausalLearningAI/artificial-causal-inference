#!/bin/bash
#
# v0 audit on ants v2: retrain the v0 SAE over pre-registered variants (batch 128 / 20, 5 / 3 seeds) and
# encode frames three ways (mean, tok0, patchavg). See scripts/eci/v0_audit/audit.py.
#   sbatch scripts/eci/v0_audit/audit.sh
# Results: results/vision/ants/eci_v0_audit/audit_<variant>.json
#
#SBATCH --job-name=eci_v0_audit
#SBATCH --output=logs/eci_v0_audit_%j.out
#SBATCH --error=logs/eci_v0_audit_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=L40S|A100|A40
#SBATCH --exclude=gpu240,gpu241,gpu242
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1

module load conda
conda activate crl
export PYTHONUNBUFFERED=1
set -euo pipefail

REPO=/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
SRC=/mnt/beegfs/locatgrp/shared/rcadei/eci_v0/ants_v2
OUT=$REPO/results/vision/ants/eci_v0_audit
cd "$REPO"
mkdir -p logs "$OUT"
STAGE=/localhome/$USER/$SLURM_JOB_ID
mkdir -p "$STAGE/out"
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used --format=csv -l 60 > "logs/eci_v0_audit_${SLURM_JOB_ID}_gpu.csv" &
SMI_PID=$!
trap 'rm -rf "$STAGE"; kill $SMI_PID 2>/dev/null || true' EXIT

cp "$SRC"/tokens_batch_*.npy "$SRC/frames.csv" "$STAGE/"
echo "staged: $(du -sh "$STAGE" | cut -f1)"

python -u scripts/eci/v0_audit/audit.py --token-dir "$STAGE" --out "$STAGE/out" ${EXTRA_ARGS:-}
cp "$STAGE"/out/audit_*.json "$OUT/"
echo "copied to $OUT"
