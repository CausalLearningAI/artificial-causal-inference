#!/bin/bash
#
# Round 2 of the v0 audit, our re-implementation path (audit2.py). One SAE per job:
# CPU, delta|beta nodes (eta nodes have no /localhome).
#   KEY=b128_s1 sbatch --export=ALL scripts/eci/v0_audit/audit2_cpu.sh   (KEY: cached, b128_s1..4, b20_s0..2)
# Results: results/vision/ants/eci_v0_audit/audit2_<KEY>.json + per-code .npz
#
#SBATCH --job-name=eci_v0_audit2
#SBATCH --output=logs/eci_v0_audit2_%j.out
#SBATCH --error=logs/eci_v0_audit2_%j.err
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
: "${KEY:?}"

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

python -u scripts/eci/v0_audit/audit2.py --token-dir "$STAGE" --out "$STAGE/out" --threads 32 --key "$KEY"
cp "$STAGE"/out/audit2_* "$OUT/"
echo "copied to $OUT"
