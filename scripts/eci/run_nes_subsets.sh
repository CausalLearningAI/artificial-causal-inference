#!/bin/bash
#
# Subgroup NES (gene line and / or sex) on the mice v1 SAE codes: the primary settings of
# scripts/eci/run_nes.py (mean activation) and scripts/eci/run_nes_bouts.py (bout rate) for the 11
# subgroups (3 lines, 2 sexes, 6 line x sex), for each SAE:pooling of SPECS, into
# results/vision/mice/eci/nes/<sae>/subsets/<line>_<sex>/[<P>pool_bouts/] (prefixes 128, 256, 1024).
#   <sae>:max / <sae>:mean   codes_max / codes_mean primary of the SAE's own result set (both poolings loaded)
#   <sae>_mean:mean, <sae>_somp:somp   the aggregation sibling result sets (one codes file; bouts on those
#                            frame values: <sae>_mean/subsets/<name>/meanpool_bouts/), as scripts/eci/run_nes_somp.sh
# Default SPECS: the odor-aligned mouse-mask SAE fg448al with its max-pool (core of the atlas) and mean-pool
# (sibling <sae>_mean) outcomes. The earlier runs (prefixes 128 / 1024 only) used
# SPECS="matryoshka_btk_1024_k16_fg448_s0:max matryoshka_btk_1024_k16_ep20_s0:mean".
# The null check (NULL_SUBSET, default ash1l_all, 10 shuffles) runs on the NULL_SAE result set.
# CPU only; needs the per-video caches of the full-cohort runs (<sae>/_cache/). ~20 s per run.
#
# Usage: sbatch scripts/eci/run_nes_subsets.sh
#        SPECS="<sae>:max <sae>_mean:mean" NULL_SAE=<sae> sbatch --export=ALL scripts/eci/run_nes_subsets.sh
#        EXTRA_ARGS="--period adjust" ... passed to both runners (camera period, family B only)
#
#SBATCH --job-name=eci_nes_subsets
#SBATCH --output=logs/eci_nes_subsets_%j.out
#SBATCH --error=logs/eci_nes_subsets_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=defaultp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

module load conda
conda activate crl
export PYTHONUNBUFFERED=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
mkdir -p logs
NULL_SUBSET=${NULL_SUBSET:-ash1l_all}
AL=matryoshka_btk_1024_k16_fg448al_s0
SPECS=${SPECS:-"$AL:max ${AL}_mean:mean"}
NULL_SAE=${NULL_SAE:-$AL}
for sp in $SPECS; do
  sae=${sp%%:*}; pool=${sp##*:}
  case "$sae" in
    *_mean|*_somp) nes_args="--poolings $pool"; bout_args="--frame-pooling $pool" ;;  # one-aggregation sibling
    *) nes_args=""; bout_args="" ;;
  esac
  for line in all ash1l kdm6b kmt5b; do
    for sex in all f m; do
      [ "$line/$sex" = "all/all" ] && continue
      nsh=0
      [ "${line}_${sex}" = "$NULL_SUBSET" ] && [ "$sae" = "$NULL_SAE" ] && nsh=10
      echo "=== $sae ($pool) ${line}_${sex} shuffles $nsh"
      python -u scripts/eci/run_nes.py --sae "$sae" --primary-pooling "$pool" $nes_args --line "$line" --sex "$sex" \
        --primary-only --n-shuffles "$nsh" ${EXTRA_ARGS:-}
      python -u scripts/eci/run_nes_bouts.py --sae "$sae" --compare-pooling "$pool" $bout_args --line "$line" --sex "$sex" \
        --primary-only --n-shuffles "$nsh" ${EXTRA_ARGS:-}
    done
  done
done
