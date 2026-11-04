#!/bin/bash
# precision@top3/video: sbatch --array=0-21 v24_top3.sh prep ; sbatch v24_top3.sh report
#SBATCH --output=/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference/results/vision/eci_v2_4/metric_top3/logs/%x_%A_%a.out
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --partition=defaultp
#SBATCH --exclude=eta310,eta380
module load conda
conda activate crl
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1 NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export LOCAL_DIR=/localhome/$USER/${SLURM_JOB_ID}_v24t
mkdir -p $LOCAL_DIR 2>/dev/null; trap "rm -rf $LOCAL_DIR" EXIT
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
S=scripts/eci/v2/v24_top3.py
if [ "$1" == prep ]; then
  EXT=(V_s1001 V_s1002 V_s1003 V_s1004 V_s1005 V_s1006 Vall_s2001 Vall_s2002 Vall_s2003 v1_s0)
  i=$SLURM_ARRAY_TASK_ID
  if [ $i -lt 10 ]; then V22_FRAMES=ext python $S mice-prep --key ${EXT[$i]}
  elif [ $i -lt 21 ]; then V22_FRAMES=full python $S mice-prep --key v1_s$((i-10))
  else for k in ${EXT[@]} v1_s1 v1_s2 v1_s3 v1_s4 v1_s5 v1_s6 v1_s7 v1_s8 v1_s9 v1_s10; do python $S ants-prep --key $k; done
  fi
else
  V22_FRAMES=ext python $S mice-report
  V22_FRAMES=full python $S mice-report
  python $S ants-report
  python $S summary
fi
