#!/bin/bash
#
# NES on one SAE's codes with the C1-rollout decisions (scripts/eci/rollout_c1.py): primary codes_max, per-video mean
# (run_nes.py), bout rate (run_nes_bouts.py) and latency (run_nes_latency.py); neurons picked by smallest p
# (--select p); nuisance = per-video mean foreground patch count (--nuisance nfg) in every domain, plus the recording
# day (--day) where the design has one with both arms on a day (ants: recording_date; frogs: group is confounded with
# the session, mice: no day column). Mice also get the camera-period adjusted family B (--period adjust) as a
# sensitivity. CPU only.
#
#   DOMAIN=mice  SAE=matryoshka_btk_4096_k16_fg448m5_s0  PREFIXES=512,1024 sbatch --export=ALL scripts/eci/rollout_c1_nes.sh
#   DOMAIN=ants  SAE=matryoshka_btk_4096_k16_antsfgm5_s0 PREFIXES=512,1024 sbatch --export=ALL scripts/eci/rollout_c1_nes.sh
#   DOMAIN=frogs SAE=matryoshka_btk_4096_k16_frogsfgm5_s0 PREFIXES=512,1024 sbatch --export=ALL scripts/eci/rollout_c1_nes.sh
#   the deployed SAEs under the same settings (comparison, outside the atlas roots):
#   DOMAIN=ants SAE=matryoshka_btk_1024_k16_antsfg_s0 PREFIXES=128,256 OUTROOT=results/vision/eci_rollout_c1/nes_deployed/ants \
#       sbatch --export=ALL scripts/eci/rollout_c1_nes.sh
#SBATCH --job-name=eci_c1_nes
#SBATCH --output=results/vision/eci_rollout_c1/logs/%x_%j.out
#SBATCH --time=06:00:00
#SBATCH --partition=defaultp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

module load conda
conda activate crl
export PYTHONUNBUFFERED=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
DOMAIN=${DOMAIN:?}
SAE=${SAE:?}
PREFIXES=${PREFIXES:-512,1024}
OR=""; [ -n "${OUTROOT:-}" ] && OR="--out-root ${OUTROOT}"
COMMON="--domain $DOMAIN --sae $SAE --prefixes $PREFIXES --nuisance nfg --select p $OR"
DAY=""; [ "$DOMAIN" = "ants" ] && DAY="--day"
SETS="core"; [ "$DOMAIN" = "ants" ] && SETS="core pairs"
for set in $SETS; do
    echo "=== $SAE set $set: activation (per-video mean of codes_max)"
    python -u scripts/eci/run_nes.py $COMMON $DAY --analysis-set $set --primary-pooling max
    echo "=== $SAE set $set: bout rate"
    python -u scripts/eci/run_nes_bouts.py $COMMON $DAY --analysis-set $set --compare-pooling max
    echo "=== $SAE set $set: latency"
    python -u scripts/eci/run_nes_latency.py $COMMON $DAY --analysis-set $set --frame-pooling max
done
if [ "$DOMAIN" = "mice" ]; then
    echo "=== $SAE: camera-period adjusted family B (sensitivity)"
    python -u scripts/eci/run_nes.py $COMMON --primary-pooling max --period adjust --primary-only
fi
if [ "$DOMAIN" = "ants" ]; then
    echo "=== $SAE: without the day covariate (sensitivity; the day-confounded pairs are testable only here)"
    python -u scripts/eci/run_nes.py $COMMON --analysis-set pairs --primary-pooling max --subdir pairs_noday --primary-only
    python -u scripts/eci/run_nes_bouts.py $COMMON --analysis-set pairs --compare-pooling max --subdir pairs_noday --primary-only
fi
echo "done"
