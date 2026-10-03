#!/bin/bash
#
# Camera-period checks (scripts/eci/period_check.py) of the family-B NES results of mice v1 SAEs: period-dependence
# flag per latent + day-adjusted re-test of the primary picks. CPU only, reads the NES caches / summary.csv.
#
# Usage: sbatch scripts/eci/period_check.sh
#        SPECS="matryoshka_btk_1024_k16_ff448al_s0:max" sbatch --export=ALL scripts/eci/period_check.sh
#
#SBATCH --job-name=eci_period
#SBATCH --output=logs/eci_period_%j.out
#SBATCH --error=logs/eci_period_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=defaultp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

module load conda
conda activate crl

export PYTHONUNBUFFERED=1
set -euo pipefail

cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
mkdir -p logs

SPECS="${SPECS:-matryoshka_btk_1024_k16_fg448al_s0:max matryoshka_btk_1024_k16_fg448al_s0_mean:mean}"
for spec in $SPECS; do
    python -u scripts/eci/period_check.py --sae "${spec%%:*}" --pooling "${spec##*:}" ${EXTRA_ARGS:-}
done
