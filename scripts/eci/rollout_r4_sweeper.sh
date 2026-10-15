#!/bin/bash
# R4 sweeper for one domain (small defaultp job, no polling from the login node): waits until no encode job of the domain
# is pending / running, resubmits the shards that still have unencoded videos (encode resumes per video; time-outs on
# slower GPUs), repeats, then submits the CPU tail (rollout_r4_post.sh) once every video is encoded.
#   sbatch --export=ALL,DOM=frogs scripts/eci/rollout_r4_sweeper.sh
#SBATCH --job-name=r4_sweep
#SBATCH --output=results/vision/eci_rollout_r4/logs/%x_%j.out
#SBATCH --time=10:00:00
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
module load conda
conda activate crl
set -uo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
DOM=${DOM:?}
GPU='-p gpu --gres=gpu:1 --constraint=L40S|A100|A40 --exclude=gpu242,gpu150,gpu230,gpu241,gpu240'
for round in 1 2 3 4 5 6; do
  while squeue -u $USER -h -o "%j" | grep -q "^r4_enc_${DOM}"; do sleep 120; done
  MISS=$(python - <<PY
import json, pathlib
P = json.load(open('results/vision/eci_rollout_r4/$DOM/plan.json'))
d = pathlib.Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/r4/$DOM/videos')
print(','.join(str(j) for j, s in enumerate(P['shards']) if any(not (d / f"{P['videos'][i]['obs']}.npz").exists() for i in s)))
PY
)
  echo "$(date +%H:%M) round $round: shards with missing videos: '${MISS}'"
  if [ -z "$MISS" ]; then
    p=$(sbatch --parsable -J r4_post_$DOM --export=ALL,DOMAIN=$DOM scripts/eci/rollout_r4_post.sh); echo "post -> $p"; exit 0
  fi
  j=$(sbatch --parsable $GPU --cpus-per-task=24 --mem=96G --time=01:00:00 -J r4_enc_$DOM --array=$MISS \
      --export=ALL,STEP=encode,DOMAIN=$DOM scripts/eci/rollout_r4.sh); echo "resubmitted $MISS -> $j"
  sleep 60
done
echo "gave up after 6 rounds"; exit 1
