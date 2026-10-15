#!/bin/bash
# CalMS21 ECI smoke / QA GPU job (before any SAE): mask QA (all MASKS rules), bf16 vs fp32 token agreement, throughput
# bench (incl. the 129 x 129 DINOv3 stress grid), and a 1-video encode of every config.
#   VIDS="test:mouseXXX ..." sbatch -p gpu --gres=gpu:L40S:1 --exclude=gpu242,gpu150,gpu230,gpu241 scripts/benchmark/calms21/smoke.sh
#SBATCH --job-name=calms_smoke
#SBATCH --output=results/vision/benchmark_calms21/logs/%x_%j.out
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
module load conda
conda activate crl
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1 HF_HUB_OFFLINE=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
export LOCAL_DIR=/localhome/$USER/${SLURM_JOB_ID:-local}_calms
mkdir -p $LOCAL_DIR
nvidia-smi -L
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 60 > results/vision/benchmark_calms21/logs/gpu_smoke_${SLURM_JOB_ID}.csv &
SMI=$!
trap 'rm -rf $LOCAL_DIR; kill $SMI 2>/dev/null || true' EXIT
P="python -u scripts/benchmark/calms21/eci_encode.py"
${STEPS_MASK:-true} && $P --mode maskqa --bf16 --videos $VIDS --mask ${MASK:-absdiff}
${STEPS_PREC:-true} && $P --mode prec --configs d2_448 d2_896 d3_512 d3_1024 vj_512 vj_1024
${STEPS_BENCH:-true} && $P --mode bench --bf16 --configs d2_448 d2_896 d3_512 d3_1024 vj_512 vj_1024 d3_2064 --tag bf16
${STEPS_ENC:-true} && CALMS_TOK=$LOCAL_DIR/tok $P --mode encode --set eval --shard 0 --nshard 8 --smoke 1 --bf16 \
    --configs d2_448 d2_896 d3_512 d3_1024 vj_512 vj_1024 --mask ${MASK:-absdiff}
echo SMOKEDONE
