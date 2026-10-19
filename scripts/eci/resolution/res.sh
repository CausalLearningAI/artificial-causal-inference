#!/bin/bash
#
# Resolution sweep job wrapper (scripts/eci/resolution/res.py). STEP = <res.py step> | selftest_mice | selftest_ants;
# EXTRA = more arguments (lists with ':' instead of ',': sbatch --export splits on commas). Staging in
# /localhome/$USER/<job>_<task>_res (removed on exit); GPU utilisation / memory logged every 60 s ([gpu] lines).
#   sbatch -p gpu --gres=gpu:1 --constraint="L40S|A100|A40" --exclude=gpu240,gpu241,gpu242 \
#          --export=ALL,STEP=mtokens,EXTRA="--arm d3_512" scripts/eci/resolution/res.sh
#SBATCH --job-name=res
#SBATCH --output=/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference/results/vision/eci_resolution/logs/%x_%j.out
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
module load conda
conda activate crl
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
export LOCAL_DIR=/localhome/$USER/${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}_res
mkdir -p $LOCAL_DIR
SMI=""
if nvidia-smi -L >/dev/null 2>&1; then
  nvidia-smi -L
  nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used --format=csv,noheader -l 60 | sed -u 's/^/[gpu] /' &
  SMI=$!
fi
trap 'rm -rf $LOCAL_DIR; [ -n "$SMI" ] && kill $SMI 2>/dev/null || true' EXIT
df -h /localhome | tail -1
echo "host $(hostname) job ${SLURM_JOB_ID:-} step $STEP extra ${EXTRA:-}"
EXTRA=${EXTRA:-}; EXTRA=${EXTRA//:/,}
PY="python -u scripts/eci/resolution/res.py"
if [ "$STEP" = "selftest_mice" ]; then
  # tiny token / pool stores on local disk (36 videos x 30 frames; 40 videos x 2 bursts), 300-step SAEs, scored
  export RES_OUT=$LOCAL_DIR/out
  for ARM in ${ARMS:-d2_896 d3_1408}; do
    $PY mtokens --arm $ARM --selftest --max-videos 36 --max-frames 30
    mv $LOCAL_DIR/diag $LOCAL_DIR/diag_$ARM
    $PY mpool --arm $ARM --selftest --max-videos 40
    mv $LOCAL_DIR/pool $LOCAL_DIR/pool_$ARM
    $PY msweep --arm $ARM --selftest --pool-dir $LOCAL_DIR/pool_$ARM --diag-dir $LOCAL_DIR/diag_$ARM \
      --sae-dir $LOCAL_DIR/sae --steps 300 --seeds 0 --recipes A
  done
  mkdir -p results/vision/eci_resolution/selftest
  cp -r $RES_OUT/* results/vision/eci_resolution/selftest/
  exit 0
fi
if [ "$STEP" = "selftest_ants" ]; then
  export RES_OUT=$LOCAL_DIR/out
  ARM=${ARM:-a10}
  $PY aprep --arm $ARM --selftest --max-frames 20
  $PY atrain --arm $ARM --selftest --pool-dir $LOCAL_DIR/prep --sae-dir $LOCAL_DIR/sae --steps 300 --seeds 0
  for SER in v2 v3; do
    $PY aencode --arm $ARM --series $SER --selftest --sae-dir $LOCAL_DIR/sae --seeds 0 --max-videos 4 --max-frames 30
  done
  $PY ascore --arm $ARM --selftest
  mkdir -p results/vision/eci_resolution/selftest
  cp -r $RES_OUT/* results/vision/eci_resolution/selftest/
  exit 0
fi
$PY $STEP ${EXTRA:-}
