#!/bin/bash
# Fly-vs-Fly inventory, CPU job: sbatch scripts/benchmark/flyvsfly/inventory.sh
# Stages the dataset (minus the *_feat.mat feature files) from BeeGFS to /localhome/$USER/$SLURM_JOB_ID, runs
# inventory.py there, writes results/vision/benchmark_flyvsfly/inventory/, removes the staging dir.
#SBATCH --job-name=fvf_inv
#SBATCH --output=results/vision/benchmark_flyvsfly/logs/%x_%j.out
#SBATCH --time=04:00:00
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
PY=/nfs/scistore19/locatgrp/rcadei/.conda/envs/crl/bin/python
export PYTHONUNBUFFERED=1
D=/mnt/beegfs/locatgrp/shared/rcadei/benchmark/flyvsfly
L=/localhome/$USER/fvf_stage  # kept between reruns on the same node (KEEP_STAGE=1); removed otherwise
OUT=results/vision/benchmark_flyvsfly/inventory
[ -n "${KEEP_STAGE:-}" ] || trap 'rm -rf $L' EXIT
mkdir -p $L $OUT
echo "host $(hostname) staging $(date)"
rsync -a --exclude '*_feat.mat' $D/ $L/fvf/
echo "staged $(du -sh $L | cut -f1) $(date)"
$PY -I scripts/benchmark/flyvsfly/inventory.py $L/fvf $OUT
[ -f scripts/benchmark/flyvsfly/inventory_md.py ] && $PY -I scripts/benchmark/flyvsfly/inventory_md.py $OUT
echo "ALLDONE $(date)"
