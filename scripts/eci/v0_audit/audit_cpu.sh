#!/bin/bash
#
# CPU twin of audit.sh (the GPU queue was ~17 h deep on 2026-10-19). ARGS selects one variant, e.g.
# runs audit.py on CPU for one pre-registered variant per job (eta nodes have no /localhome).
#   ARGS="--plan b128:1" sbatch --export=ALL scripts/eci/v0_audit/audit_cpu.sh
# Results: results/vision/ants/eci_v0_audit/audit_<variant>.json
#
#SBATCH --job-name=eci_v0_audit_cpu
#SBATCH --output=logs/eci_v0_audit_cpu_%j.out
#SBATCH --error=logs/eci_v0_audit_cpu_%j.err
#SBATCH --time=16:00:00
#SBATCH --partition=defaultp
#SBATCH --constraint=delta|beta
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=80G

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
trap 'rm -rf "$STAGE"' EXIT

cp "$SRC"/tokens_batch_*.npy "$SRC/frames.csv" "$STAGE/"
echo "staged: $(du -sh "$STAGE" | cut -f1)"

python -u scripts/eci/v0_audit/audit.py --token-dir "$STAGE" --out "$STAGE/out" --device cpu --threads 32 $ARGS
cp "$STAGE"/out/audit_*.json "$OUT/"
echo "copied to $OUT"
