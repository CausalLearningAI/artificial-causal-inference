#!/bin/bash
#
# Recording-session checks (scripts/eci/session_check.py) of the frogs family-B NES results: within-group session
# dependence flag per latent + leave-one-session-out re-test of the primary picks. CPU only, reads the NES caches /
# summary.csv (needs run_nes.sh, run_nes_bouts.sh and, for <sae>_mean, run_nes_somp.sh first).
#
# Usage: SPECS="matryoshka_btk_1024_k16_frogsfg_s0:max matryoshka_btk_1024_k16_frogsfg_s0_mean:mean" \
#            sbatch --export=ALL scripts/eci/session_check.sh
#        tadpoles, session and sub-stage checks: DOMAIN=tadpoles LEVELS="session substage" SPECS=... sbatch ...
#
#SBATCH --job-name=eci_session
#SBATCH --output=logs/eci_session_%j.out
#SBATCH --error=logs/eci_session_%j.err
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

for spec in ${SPECS:?set SPECS}; do
    for level in ${LEVELS:-session}; do
        python -u scripts/eci/session_check.py --domain "${DOMAIN:-frogs}" --sae "${spec%%:*}" --pooling "${spec##*:}" \
            --level "$level" ${EXTRA_ARGS:-}
    done
done
