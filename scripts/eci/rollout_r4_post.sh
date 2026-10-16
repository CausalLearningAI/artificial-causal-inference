#!/bin/bash
#
# R4 rollout, CPU tail of one domain after its encode array (scripts/eci/rollout_r4.py): merge (fails loud when a video
# is missing) -> frame-level firing rules (fnorm primary, ftheta, gt0) -> per-video summaries under both rules -> behaviour alignment test check (mice, ants) -> NES jobs for both variants
# (scripts/eci/rollout_r4_nes.sh, submitted from here because the SAE names hold the chosen seed).
#   sbatch --dependency=afterany:<encode array> --export=ALL,DOMAIN=ants_v2 scripts/eci/rollout_r4_post.sh
#SBATCH --job-name=r4_post
#SBATCH --output=results/vision/eci_rollout_r4/logs/%x_%j.out
#SBATCH --time=02:00:00
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
module load conda
conda activate crl
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
DOMAIN=${DOMAIN:?}
python -u scripts/eci/rollout_r4.py merge --domain $DOMAIN
python -u scripts/eci/rollout_r4.py ftheta --domain $DOMAIN
python -u scripts/eci/rollout_r4.py summaries --domain $DOMAIN
case $DOMAIN in
  mice) ND=mice; AE="";;
  ants_v2) ND=ants; AE=eci_r4v2;;
  ants_v3) ND=ants; AE=eci_r4v3;;
  ants_v4) ND=ants; AE=eci_r4v4;;
  ants_v2p) ND=ants; AE=eci_r4v2p;;
  ants_v3p) ND=ants; AE=eci_r4v3p;;
  ants_v4p) ND=ants; AE=eci_r4v4p;;
  frogs) ND=frogs; AE="";;
esac
NR=$( [ "$ND" = "ants" ] && echo "results/vision/ants/$AE/nes" || echo "results/vision/$ND/eci/nes" )
for v in d2 d0; do
  SAE=$(python -c "import sys,json;sys.argv=['x'];sys.path.insert(0,'scripts/eci');import rollout_r4 as R;print(json.load(open(R.sae_root('$DOMAIN')/'best.json'))['$v']['name'])")
  for suf in _fnorm _ftheta ""; do
    EXP="ALL,DOMAIN=$ND,ANTS_ECI=$AE,SAE=${SAE}${suf},TAG=$DOMAIN"; [ -n "$suf" ] && EXP="$EXP,POOLINGS=max"
    j=$(sbatch --parsable -J r4_nes_${DOMAIN}_${v}${suf} --export=$EXP scripts/eci/rollout_r4_nes.sh)
    # NESS (= DINES, src/eci/dines.py) on the same NES result set, committed code (git archive of HEAD, recorded)
    sbatch --dependency=afterok:$j -J r4_ness_${DOMAIN}_${v}${suf} -o results/vision/eci_rollout_r4/logs/%x_%j.out \
      --export=ALL,RES=$NR/${SAE}${suf},EXTRA="--out-root $PWD/results/vision/eci_rollout_r4/ness" scripts/eci/atlas_v37_methods.sh
  done
done
# alignment test check (ants_v4p runs r4 only: no C1 codes for v4)
if [ "$DOMAIN" != "frogs" ] && [ "$DOMAIN" != "ants_v4" ]; then python -u scripts/eci/rollout_r4.py align --domain $DOMAIN; fi
echo done
