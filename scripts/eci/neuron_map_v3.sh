#!/bin/bash
#
# Behavioural neuron map v3, MICE ONLY (scripts/eci/neuron_map_v3.py): tails attached to the SAM instances, animal
# count per contact cluster, a cage-object map and the environment-interaction rule. Reads the v2 codes / frames /
# top frames (results/vision/eci_neuron_map_v2/), writes results/vision/eci_neuron_map_v3/ (gitignored). Inputs staged
# to /localhome/$USER/$SLURM_JOB_ID, removed at exit. CPU only.
#   STEP=look | objects | prep            (mice)
#   STEP=analyse | sheets  DICT=fg448m5|fg448al  CUE=dark|mid|rel (tail cue, default dark)
# Usage:
#   STEP=prep sbatch --export=ALL --partition=defaultp --cpus-per-task=16 --mem=64G --time=02:00:00 scripts/eci/neuron_map_v3.sh
#SBATCH --job-name=nmap_v3
#SBATCH --output=results/vision/eci_neuron_map_v3/logs/%x_%j.out

module load conda
conda activate crl
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
export LOCAL_DIR=/localhome/$USER/${SLURM_JOB_ID:-manual}_nmap_v3
mkdir -p $LOCAL_DIR
trap 'rm -rf $LOCAL_DIR' EXIT
echo "host $(hostname) job ${SLURM_JOB_ID:-} step $STEP ${DICT:-}"
python -u scripts/eci/neuron_map_v3.py $STEP ${DICT:+--dict $DICT} ${CUE:+--cue $CUE} --workers ${SLURM_CPUS_PER_TASK:-8}
