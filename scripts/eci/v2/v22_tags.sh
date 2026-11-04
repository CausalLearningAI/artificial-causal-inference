#!/bin/bash
# v2.2 tags: bash v22_tags.sh "KEY1[:codes_path] KEY2 ..." -> one array task per key; builds the readout cache if missing
#SBATCH --output=/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference/results/vision/eci_v2_2/logs/%x_%A_%a.out
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --partition=defaultp
#SBATCH --exclude=eta310
module load conda
conda activate crl
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
read -ra K <<< "$KEYS"
E=${K[$SLURM_ARRAY_TASK_ID]}
M=${E%%:*}; P=""; [ "$E" != "$M" ] && P=${E#*:}
B=/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_2/cache/mice
if [ ! -f $B/${M}_raw.npy ]; then
  if [ -n "$P" ]; then python scripts/eci/v2/v22_core.py cache --model $M --codes $P; else python scripts/eci/v2/v22_core.py cache --model $M; fi
fi
python scripts/eci/v2/v22_eval.py tag --model $M
