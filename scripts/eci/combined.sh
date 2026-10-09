#!/bin/bash
#
# COMBINED test (scripts/eci/combined.py): C0 base / C1 today's mask + M5 motion / C2 T2b mask + M5 motion.
#   STEP=choose                 login safe (reads a small npz): python scripts/eci/combined.py choose
#   STEP=apply                  CPU: T2b masks, encode plan, mask statistics, sheets
#   STEP=encode                 GPU (A40 for bit identity with the stores), array over tasks: TASK = array index
#   STEP=train|evaluate         GPU, DOMAIN
#   STEP=readouts               CPU: video level, tables, decision
# GPU jobs on the 'gpu' partition (never gpu100), excluding gpu150 / gpu242. Staging in /localhome/$USER/<job>.
# Outputs: results/vision/eci_combined/ (gitignored).
# Usage:
#   mkdir -p results/vision/eci_combined/logs
#   STEP=apply sbatch -p defaultp --gres=none --cpus-per-task=32 --mem=96G --time=01:00:00 scripts/eci/combined.sh
#   STEP=encode DOMAIN=ants NTASKS=3 sbatch --array=0-2 --constraint=A40 --time=00:45:00 scripts/eci/combined.sh
#   STEP=train DOMAIN=ants sbatch --time=00:40:00 scripts/eci/combined.sh
#SBATCH --job-name=eci_comb
#SBATCH --output=results/vision/eci_combined/logs/%x_%A_%a.out
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu150,gpu242
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G

module load conda
conda activate crl
export PYTHONUNBUFFERED=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
export LOCAL_DIR=/localhome/$USER/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}_comb
mkdir -p $LOCAL_DIR
trap 'rm -rf $LOCAL_DIR' EXIT
df -h /localhome | tail -1
nvidia-smi -L 2>/dev/null || true
DOMAIN=${DOMAIN:-mice}
EXTRA=${EXTRA:-}
case ${STEP} in
  apply|readouts)
    python -u scripts/eci/combined.py $STEP --domains ${DOMAINS:-mice,ants} --workers ${SLURM_CPUS_PER_TASK:-8} $EXTRA ;;
  encode)
    python -u scripts/eci/combined.py encode --domain $DOMAIN --task ${SLURM_ARRAY_TASK_ID:-0} --n-tasks ${NTASKS:-1} \
      --num-workers $(( ${SLURM_CPUS_PER_TASK:-8} - 2 )) $EXTRA ;;
  train|evaluate)
    python -u scripts/eci/combined.py $STEP --domain $DOMAIN $EXTRA ;;
esac
