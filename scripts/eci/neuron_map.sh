#!/bin/bash
#
# Post-hoc behavioural map of the ECI SAE dictionaries' neurons (scripts/eci/neuron_map.py).
#   STEP=encode   GPU: mice eval tokens staged (raw store: all 172,800 eval frames for CellMeans, ~40 GB; aligned store:
#                 the 30,240 subset frames only), both mice SAEs encoded on the subset -> results/vision/eci_neuron_map/mice/codes
#   STEP=prep     CPU: frame tables, dark cue, dark blobs / site maps, ants dish circles (both domains)
#   STEP=analyse  CPU: per-neuron statistics, classes, figures, contact sheets (ants patch codes staged locally)
# Inputs are staged to /localhome/$USER/$SLURM_JOB_ID, results copied back once, staging removed at exit.
# Usage:
#   mkdir -p logs
#   e=$(STEP=encode sbatch --parsable --export=ALL --partition=gpu --gres=gpu:1 --exclude=gpu150,gpu242 --cpus-per-task=8 --mem=64G --time=00:45:00 scripts/eci/neuron_map.sh)
#   p=$(STEP=prep sbatch --parsable --export=ALL --partition=defaultp --cpus-per-task=32 --mem=48G --time=00:40:00 scripts/eci/neuron_map.sh)
#   STEP=analyse sbatch --export=ALL --partition=defaultp --cpus-per-task=8 --mem=96G --time=01:00:00 --dependency=afterok:$e:$p scripts/eci/neuron_map.sh
#SBATCH --job-name=neuron_map
#SBATCH --output=logs/neuron_map_%j.out

module load conda
conda activate crl
export PYTHONUNBUFFERED=1
set -euo pipefail
REPO=/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
cd $REPO
STAGE=/localhome/$USER/${SLURM_JOB_ID:-manual}_neuron_map
mkdir -p $STAGE
trap 'rm -rf $STAGE' EXIT
df -h /localhome | tail -1
export LOCAL_DIR=$STAGE
case ${STEP} in
  encode)
    nvidia-smi -L
    python -u scripts/eci/neuron_map.py encode --stage $STAGE ;;
  prep)
    python -u scripts/eci/neuron_map.py prep --workers ${SLURM_CPUS_PER_TASK:-8} ;;
  analyse)
    cp -r results/vision/eci_repr_diag/ants/patch_codes $STAGE/ants_pc
    python -u scripts/eci/neuron_map.py analyse --stage $STAGE ${DICTS:+--dicts $DICTS} ;;
esac
