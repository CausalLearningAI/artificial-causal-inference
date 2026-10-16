#!/bin/bash
# CPU smoke test of scripts/eci/v0/v0_domains.py (1 video / pool per arm value, 1 SAE epoch); outputs tagged _smoke.
#   DOMAIN=mice_v1 sbatch scripts/eci/v0/v0_domains_cpusmoke.sh
#SBATCH --job-name=v0cpu
#SBATCH --output=logs/eci_v0_cpusmoke_%j.out
#SBATCH --partition=defaultp
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
module load conda
conda activate crl
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=12
set -euo pipefail
S=/localhome/$USER/$SLURM_JOB_ID
mkdir -p $S/tok $S/out
trap 'rm -rf $S' EXIT
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
echo "smoke $DOMAIN on $(hostname)"
python -u scripts/eci/v0/v0_domains.py --domain $DOMAIN --smoke 1 --epochs 1 --num-workers 4 --stage-dir $S/tok \
    --out-dir results/vision/${DOMAIN%%_*}/eci_v0_${DOMAIN}_smoke --big-dir /mnt/beegfs/locatgrp/shared/rcadei/eci_v0/${DOMAIN}_smoke
