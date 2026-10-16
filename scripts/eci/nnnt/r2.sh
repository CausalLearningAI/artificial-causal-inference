#!/bin/bash
#
# NN/NT round 2 job wrapper (scripts/eci/nnnt/r2.py). STEP = tokens | probe | selftest | <any r2.py step>; EXTRA = more
# arguments. Staging in /localhome/$USER/<job>_<task>_r2 (removed on exit); GPU utilisation / memory logged every 60 s
# into the job log ([gpu] lines).
#   sbatch -p gpu --gres=gpu:1 --constraint=L40S --export=ALL,STEP=tokens scripts/eci/nnnt/r2.sh
#SBATCH --job-name=nnnt_r2
#SBATCH --output=/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference/results/vision/eci_nnnt/logs/%x_%A_%a.out
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
module load conda
conda activate crl
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
export LOCAL_DIR=/localhome/$USER/${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}_r2
mkdir -p $LOCAL_DIR
SMI=""
if nvidia-smi -L >/dev/null 2>&1; then
  nvidia-smi -L
  nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used --format=csv,noheader -l 60 | sed -u 's/^/[gpu] /' &
  SMI=$!
fi
trap 'rm -rf $LOCAL_DIR; [ -n "$SMI" ] && kill $SMI 2>/dev/null || true' EXIT
df -h /localhome | tail -1
echo "host $(hostname) job ${SLURM_JOB_ID:-} task ${SLURM_ARRAY_TASK_ID:-} step $STEP extra ${EXTRA:-}"
PY="python -u scripts/eci/nnnt/r2.py"
if [ "$STEP" = "selftest" ]; then
  # diagnosis on 36 videos x 30 frames (6 pools), written to local disk only
  export R2_OUT=$LOCAL_DIR/out
  $PY tokens --selftest --max-videos 36 --max-frames 30
  mkdir -p results/vision/eci_nnnt/r2
  cp $R2_OUT/tokens_selftest.json results/vision/eci_nnnt/r2/
  $PY probe --selftest --diag-dir $LOCAL_DIR/diag --n-splits 1 --steps 100
  $PY pool --selftest --max-videos 40 --bursts-per-video 2
  cp $R2_OUT/pool_selftest.json results/vision/eci_nnnt/r2/
  $PY sweep --selftest --pool-dir $LOCAL_DIR/pool --diag-dir $LOCAL_DIR/diag --steps 300 --seeds 0 --arms S0,S1,S6
  cp $R2_OUT/sweep_selftest.json results/vision/eci_nnnt/r2/
  cp $R2_OUT/diag_selftest.json results/vision/eci_nnnt/r2/
  exit 0
fi
$PY $STEP ${EXTRA:-}
