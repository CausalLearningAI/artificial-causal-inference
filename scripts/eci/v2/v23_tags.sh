#!/bin/bash
# v2.3 ext tags: KEYS="K1 K2 ..." sbatch --array=0-(n-1) v23_tags.sh ; builds the ext cache (codes assembled + HMM P(on)) if missing, then the tag.
module load conda
conda activate crl
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1 NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} V22_FRAMES=ext
export LOCAL_DIR=/localhome/$USER/${SLURM_JOB_ID}_v23t
mkdir -p $LOCAL_DIR; trap "rm -rf $LOCAL_DIR" EXIT
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
read -ra K <<< "$KEYS"
M=${K[$SLURM_ARRAY_TASK_ID]}
B=/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_3/ext/cache/mice
if [ ! -f $B/${M}_raw.npy ] || { [[ $M != v1_* ]] && [ ! -f $B/${M}_pon.npy ]; }; then
  python scripts/eci/v2/v22_full.py cache --model $M
fi
python scripts/eci/v2/v22_eval.py tag --model $M
