#!/bin/bash
#
# T2 tight animal mask + T4 read-outs (scripts/eci/tight_mask.py, tight_mask_sae.py, readouts.py).
#   STEP=background|choose|apply|sheet   CPU (tight_mask.py): pixel backgrounds + calibration, parameter rule, apply
#                                        to the store frames, visual check. Frames are read from dataset/ JPGs once.
#   STEP=encode                          GPU (tight_mask_encode.py): tight-mask patches absent from the store, DOMAIN,
#                                        TASK / NTASKS split
#   STEP=train|evaluate                  GPU (tight_mask_sae.py): T2 arm SAEs; baseline + T2 evaluation with presence
#                                        and extent read-outs, level map, contact sheets. Tokens staged to /localhome.
#   STEP=readouts                        CPU (readouts.py): per-video read-outs, video-level check, tables, decisions.
# Outputs: results/vision/eci_t2t4/{mice,ants}/ (gitignored).
# Usage:
#   mkdir -p results/vision/eci_t2t4/logs
#   STEP=background sbatch --partition=defaultp --cpus-per-task=32 --mem=96G --time=01:00:00 scripts/eci/tight_mask.sh
#   STEP=train DOMAIN=mice sbatch --partition=gpu --gres=gpu:1 --exclude=gpu150,gpu242 --cpus-per-task=8 --mem=150G \
#       --time=01:00:00 scripts/eci/tight_mask.sh
#SBATCH --job-name=t2t4
#SBATCH --output=results/vision/eci_t2t4/logs/%x_%j.out

module load conda
conda activate crl
export PYTHONUNBUFFERED=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
export LOCAL_DIR=/localhome/$USER/${SLURM_JOB_ID:-manual}_t2t4
mkdir -p $LOCAL_DIR
trap 'rm -rf $LOCAL_DIR' EXIT
df -h /localhome | tail -1
DOMAINS=${DOMAINS:-mice,ants}
DOMAIN=${DOMAIN:-mice}
EXTRA=${EXTRA:-}
case ${STEP} in
  background|choose|apply|sheet)
    python -u scripts/eci/tight_mask.py $STEP --domains $DOMAINS --workers ${SLURM_CPUS_PER_TASK:-8} $EXTRA ;;
  encode)
    nvidia-smi -L
    python -u scripts/eci/tight_mask_encode.py --domain $DOMAIN --task ${TASK:-0} --n-tasks ${NTASKS:-1} $EXTRA ;;
  train|evaluate)
    nvidia-smi -L
    python -u scripts/eci/tight_mask_sae.py $STEP --domain $DOMAIN $EXTRA ;;
  readouts)
    python -u scripts/eci/readouts.py --stage $LOCAL_DIR $EXTRA ;;
esac
