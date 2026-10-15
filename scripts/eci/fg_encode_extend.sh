#!/bin/bash
#
# Codes of a new ECI layout reusing an old layout's codes (scripts/eci/fg_encode_extend.py). STEP = setup | encode |
# merge; SAES = space-separated SAE names (encoded in one DINOv2 pass); the new layout is the domain's (ANTS_ECI).
# GPU jobs on the 'gpu' partition (never gpu100), excluding gpu150 (11 GB), gpu242 (GPU 0 failed) and gpu230
# (/localhome nearly full). Encode stages the shard's JPGs to /localhome/$USER/$SLURM_JOB_ID_<task> (removed on exit)
# and logs GPU utilisation every 60 s next to the job log.
#
#   S="matryoshka_btk_1024_k16_antsfg_s0 matryoshka_btk_1024_k16_antsfull_s0"
#   ANTS_ECI=eci30 STEP=setup SAES="$S" sbatch -p defaultp --gres=none --export=ALL scripts/eci/fg_encode_extend.sh
#   ANTS_ECI=eci30 STEP=encode SAES="$S" sbatch --array=0-23 --time=04:00:00 --export=ALL scripts/eci/fg_encode_extend.sh
#   ANTS_ECI=eci30 STEP=merge SAES="$S" sbatch --time=03:00:00 --export=ALL scripts/eci/fg_encode_extend.sh
#SBATCH --job-name=eci_extend
#SBATCH --output=results/vision/eci_ants30/logs/%x_%A_%a.out
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu150,gpu242,gpu230
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G

module load conda
conda activate crl
export PYTHONUNBUFFERED=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
mkdir -p results/vision/eci_ants30/logs
SA=""; for s in ${SAES:?}; do SA="$SA --sae $s"; done
if [ "${STEP:?}" = "encode" ]; then
    export LOCAL_DIR=/localhome/$USER/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}
    mkdir -p $LOCAL_DIR
    trap 'rm -rf $LOCAL_DIR; kill %1 2>/dev/null || true' EXIT
    nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader -l 60 \
        > results/vision/eci_ants30/logs/gpu_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}.csv &
    python -u scripts/eci/fg_encode_extend.py encode $SA --shard ${SLURM_ARRAY_TASK_ID:-0} --n-shards ${N_SHARDS:-24} ${EXTRA:-}
else
    python -u scripts/eci/fg_encode_extend.py ${STEP} $SA --n-shards ${N_SHARDS:-24} ${EXTRA:-}
fi
