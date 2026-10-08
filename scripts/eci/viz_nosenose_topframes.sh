#!/bin/bash
#
# Top-10 frames of the best nose-nose neuron of three mice SAEs (Spatial-SAE pilot matryoshka / spatial, deployed
# fg448), raw and with the per-patch activation overlay (scripts/eci/viz_nosenose_topframes.py).
# This job runs the 'select' step (frame-level max from the stored eval codes, greedy top-10 with a 10 s per-video
# window, re-encoding of the 30 picked frames' tokens on CPU), staged under /localhome, then 'render'.
#   sbatch scripts/eci/viz_nosenose_topframes.sh
# Output: results/vision/eci_spatial_sae/mice/viz_nosenose_top10/
#SBATCH --job-name=eci_nn_top10
#SBATCH --output=logs/eci_nn_top10_%j.out
#SBATCH --time=01:00:00
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G

module load conda
conda activate crl
export PYTHONUNBUFFERED=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
mkdir -p logs
STAGE=/localhome/$USER/$SLURM_JOB_ID
trap 'rm -rf "$STAGE"' EXIT
python -u scripts/eci/viz_nosenose_topframes.py select --stage "$STAGE" --threads ${SLURM_CPUS_PER_TASK:-8}
python -u scripts/eci/viz_nosenose_topframes.py render
