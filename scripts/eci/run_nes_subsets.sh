#!/bin/bash
#
# Subgroup NES (gene line and / or sex) on the mice v1 SAE codes: the primary settings of
# scripts/eci/run_nes.py (mean activation) and scripts/eci/run_nes_bouts.py (bout rate) for the 11
# subgroups (3 lines, 2 sexes, 6 line x sex), for both SAEs, into
# results/vision/mice/eci/nes/<sae>/subsets/<line>_<sex>/[maxpool_bouts/].
# The null check (NULL_SUBSET, default ash1l_all, 10 shuffles) runs on the mouse-only (fg448) SAE.
# CPU only; needs the per-video caches of the full-cohort runs (<sae>/_cache/). ~20 s per run.
#
# Usage: sbatch scripts/eci/run_nes_subsets.sh
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
# SAE and its primary pooling of the mean-activation outcome (as its full-cohort run)
for sp in matryoshka_btk_1024_k16_fg448_s0:max matryoshka_btk_1024_k16_ep20_s0:mean; do
  sae=${sp%%:*}; pool=${sp##*:}
  for line in all ash1l kdm6b kmt5b; do
    for sex in all f m; do
      [ "$line/$sex" = "all/all" ] && continue
      nsh=0
      [ "${line}_${sex}" = "$NULL_SUBSET" ] && [ "$sae" = matryoshka_btk_1024_k16_fg448_s0 ] && nsh=10
      echo "=== $sae ${line}_${sex} shuffles $nsh"
      python -u scripts/eci/run_nes.py --sae "$sae" --primary-pooling "$pool" --line "$line" --sex "$sex" \
        --primary-only --n-shuffles "$nsh"
      python -u scripts/eci/run_nes_bouts.py --sae "$sae" --compare-pooling "$pool" --line "$line" --sex "$sex" \
        --primary-only --n-shuffles "$nsh"
    done
  done
done
