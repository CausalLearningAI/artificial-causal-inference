#!/bin/bash
#
# Behavioural neuron map v2 (scripts/eci/neuron_map_v2.py): instance-mask social level, place-bound, halo, background,
# with validation. Outputs in results/vision/eci_neuron_map_v2/ (gitignored). Inputs staged to
# /localhome/$USER/$SLURM_JOB_ID, removed at exit.
#   STEP=encode  DOMAIN=mice|ants  GPU: tokens (+ d5 prev) staged, all dictionaries of the domain encoded -> CSR codes
#   STEP=prep    DOMAIN=mice|ants  CPU: instance labels per patch, distances, zones, blob sites, strata
#   STEP=analyse DICT=<dict>       CPU: statistics, old / new classes, anchors, collective, confusion
#   STEP=sheets  DICT=<dict>       CPU: hand-check contact sheets
# Usage:
#   DOMAIN=ants STEP=encode sbatch --export=ALL --partition=gpu --gres=gpu:1 --exclude=gpu150,gpu242,gpu230 --cpus-per-task=8 --mem=96G --time=01:00:00 scripts/eci/neuron_map_v2.sh
#   DOMAIN=ants STEP=prep sbatch --export=ALL --partition=defaultp --cpus-per-task=16 --mem=64G --time=01:00:00 scripts/eci/neuron_map_v2.sh
#   DICT=antsfgm5 STEP=analyse sbatch --export=ALL --partition=defaultp --cpus-per-task=8 --mem=120G --time=01:30:00 scripts/eci/neuron_map_v2.sh
#SBATCH --job-name=nmap_v2
#SBATCH --output=results/vision/eci_neuron_map_v2/logs/%x_%j.out

module load conda
conda activate crl
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
export LOCAL_DIR=/localhome/$USER/${SLURM_JOB_ID:-manual}_nmap_v2
mkdir -p $LOCAL_DIR
trap 'rm -rf $LOCAL_DIR' EXIT
echo "host $(hostname) job ${SLURM_JOB_ID:-} step $STEP ${DOMAIN:-} ${DICT:-}"
df -h /localhome | tail -1
case $STEP in
  encode)
    nvidia-smi -L
    python -u scripts/eci/neuron_map_v2.py encode --domain $DOMAIN ;;
  prep)
    if [ "$DOMAIN" = "mice" ]; then cp results/vision/eci_mice_pairs/bsub/labels_s*.npz $LOCAL_DIR/; fi
    python -u scripts/eci/neuron_map_v2.py prep --domain $DOMAIN --workers ${SLURM_CPUS_PER_TASK:-8} ;;
  analyse)
    python -u scripts/eci/neuron_map_v2.py analyse --dict $DICT ${VARIANT:+--variant $VARIANT} ;;
  sheets)
    python -u scripts/eci/neuron_map_v2.py sheets --dict $DICT --workers ${SLURM_CPUS_PER_TASK:-8} ;;
esac
