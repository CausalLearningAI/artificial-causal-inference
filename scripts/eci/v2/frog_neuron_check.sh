#!/bin/bash
# Where on the frog do ECI v2 frogs neurons fire (scripts/eci/v2/frog_neuron_check.py). CPU: reads the 35 juvenile
# full-pass token npz from BeeGFS once each + SLEAP poses; decodes 24 frames for the contact sheet.
#SBATCH --job-name=eci_v2_frogcheck
#SBATCH --output=/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference/results/vision/eci_v2/logs/%x_%j.out
#SBATCH --partition=defaultp
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
module load conda
conda activate crl
export PYTHONUNBUFFERED=1 MPLCONFIGDIR=/localhome/$USER/${SLURM_JOB_ID}_mpl
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
trap 'rm -rf "$MPLCONFIGDIR"' EXIT
python -u scripts/eci/v2/frog_neuron_check.py ${EXTRA_ARGS:-}
