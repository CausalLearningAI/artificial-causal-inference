#!/bin/bash
#
# Mice v2 ECI v2.0 input layout (scripts/eci/v2/mice_v2_setup.py), one STEP per job:
#   plan / check     CPU:  sbatch -p defaultp --export=ALL,STEP=plan scripts/eci/v2/mice_v2_setup.sh
#   corner           CPU:  scripts/eci/mice_odor_corner.py --version v2 (needs fg448 backgrounds)
#   fgbg             GPU array 0-7, ALIGN=none|odor: scripts/eci/fg_background.py unchanged (v1 parameters: 200 frames,
#                    rule fg448, DINOv2-base 448) on the v2 frames, through a dataset-dir shim in $LOCAL_DIR whose
#                    mice/v1/annotations.csv (+ eci/odor_corner.csv) are the v2 files (the script reads the mice domain's
#                    annotations path) -> dataset/mice/v2/eci/fg448[al]/background
#     sbatch -p gpu --gres=gpu:1 --array=0-7 --cpus-per-task=16 --mem=64G --export=ALL,STEP=fgbg,ALIGN=none scripts/eci/v2/mice_v2_setup.sh
#SBATCH --job-name=mv2setup
#SBATCH --output=/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference/results/vision/eci_v2_0/logs/%x_%A_%a.out
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
module load conda
conda activate crl
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error
set -euo pipefail
R=/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
cd $R
if [ -d /localhome ] && mkdir -p /localhome/$USER 2>/dev/null; then BASE=/localhome/$USER; else BASE=${TMPDIR:-/tmp}; fi
export LOCAL_DIR=$BASE/${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}_mv2
mkdir -p $LOCAL_DIR
trap 'rm -rf $LOCAL_DIR' EXIT
echo "host $(hostname) job ${SLURM_JOB_ID:-} step $STEP align ${ALIGN:-} git $(git rev-parse --short HEAD)"
case $STEP in
  plan|check) python -u scripts/eci/v2/mice_v2_setup.py $STEP ;;
  corner) python -u scripts/eci/mice_odor_corner.py --version v2 --sheet results/vision/eci_v2_0/mice/v2_odor_corner_sheet.png ;;
  fgbg)
    D=$LOCAL_DIR/ds
    mkdir -p $D/mice/v1/eci $D/mice/v2
    ln -s $R/dataset/mice/v2/annotations.csv $D/mice/v1/annotations.csv
    ln -s $R/dataset/mice/v2/eci/odor_corner.csv $D/mice/v1/eci/odor_corner.csv
    ln -s $R/dataset/mice/v2/frames $D/mice/v2/frames
    TAG=fg448; [ "${ALIGN:-none}" = odor ] && TAG=fg448al
    python -u scripts/eci/fg_background.py --domain mice --dataset-dir $D --out-dir $R/dataset/mice/v2/eci/$TAG/background \
      --align ${ALIGN:-none} --task ${SLURM_ARRAY_TASK_ID} --n-tasks 8 --num-workers 14 ;;
  *) echo "unknown STEP $STEP"; exit 1 ;;
esac
echo done
