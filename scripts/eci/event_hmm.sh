#!/bin/bash
#
# Label-free event confidence of the C1 SAE neurons: zero-inflated two-state HMM per neuron (scripts/eci/event_hmm.py).
# CPU only (defaultp). codes_max is read sequentially in row chunks (key columns cached in results/), nothing staged.
#
#   mkdir -p results/vision/eci_event_hmm/logs
#   for D in mice ants frogs; do DOMAIN=$D STEP=run sbatch --export=ALL scripts/eci/event_hmm.sh; done
#   DOMAIN=ants STEP=all sbatch --time=06:00:00 --mem=64G --cpus-per-task=16 --export=ALL scripts/eci/event_hmm.sh
#   STEP=figure sbatch --time=00:10:00 --mem=4G --cpus-per-task=1 --export=ALL scripts/eci/event_hmm.sh
#SBATCH --job-name=eci_ehmm
#SBATCH --output=results/vision/eci_event_hmm/logs/%x_%j.out
#SBATCH --time=01:30:00
#SBATCH --partition=defaultp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

module load conda
conda activate crl
export PYTHONUNBUFFERED=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
hostname
STEP=${STEP:-run}
DOMAIN=${DOMAIN:-mice}
python -u scripts/eci/event_hmm.py ${STEP} --domain $DOMAIN ${EXTRA:-}
