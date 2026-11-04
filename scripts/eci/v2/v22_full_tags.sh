#!/bin/bash
# v2.2 FULL WINDOW tags: V22_FRAMES=full|fullsub KEYS="K1 K2 ..." sbatch --array=0-(n-1) v22_full_tags.sh
# full: builds the full-window cache (v22_full.py cache) if missing, then v22_eval.py tag; fullsub: old caches.
module load conda
conda activate crl
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1 NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export LOCAL_DIR=/localhome/$USER/${SLURM_JOB_ID}_v22f
mkdir -p $LOCAL_DIR; trap "rm -rf $LOCAL_DIR" EXIT
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
read -ra K <<< "$KEYS"
M=${K[$SLURM_ARRAY_TASK_ID]}
B=/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_2/full/cache/mice
if [ "$V22_FRAMES" = full ] && { [ ! -f $B/${M}_raw.npy ] || { [[ $M != v1_* ]] && [ ! -f $B/${M}_pon.npy ]; }; }; then
  python scripts/eci/v2/v22_full.py cache --model $M
fi
[ "${CACHE_ONLY:-0}" = 1 ] && exit 0
python scripts/eci/v2/v22_eval.py tag --model $M
