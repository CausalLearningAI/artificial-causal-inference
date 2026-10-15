#!/bin/bash
#
# R4 rollout, CPU tail of one domain after its encode array (scripts/eci/rollout_r4.py): merge (fails loud when a video
# is missing) -> frame theta (ftheta, the primary firing rule) -> per-video summaries under both rules -> behaviour alignment test check (mice, ants) -> NES jobs for both variants
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
  frogs) ND=frogs; AE="";;
esac
for v in d2 d0; do
  SAE=$(python -c "import json;print(json.load(open('results/vision/eci_rollout_r4/$DOMAIN/sae/best.json'))['$v']['name'])")
  sbatch -J r4_nes_${DOMAIN}_${v}ft --export=ALL,DOMAIN=$ND,ANTS_ECI=$AE,SAE=${SAE}_ftheta,TAG=$DOMAIN,POOLINGS=max scripts/eci/rollout_r4_nes.sh
  sbatch -J r4_nes_${DOMAIN}_$v --export=ALL,DOMAIN=$ND,ANTS_ECI=$AE,SAE=$SAE,TAG=$DOMAIN scripts/eci/rollout_r4_nes.sh
done
if [ "$DOMAIN" != "frogs" ]; then python -u scripts/eci/rollout_r4.py align --domain $DOMAIN; fi
echo done
