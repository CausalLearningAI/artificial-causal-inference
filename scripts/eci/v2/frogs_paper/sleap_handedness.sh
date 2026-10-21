#!/bin/bash
# Left/right convention of the original vs re-predicted SLEAP files of the juvenile frogs (sleap_handedness.py).
#SBATCH --job-name=eci_v2_froghand
#SBATCH --output=/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference/results/vision/eci_v2/frogs_paper/%x_%j.out
#SBATCH --partition=defaultp
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
# h5py is in /usr/bin/python3 on the login node only; on compute nodes install it into the job's local staging dir
PY=/usr/bin/python3
if ! $PY -c 'import h5py' 2>/dev/null; then
  LIB=/localhome/$USER/${SLURM_JOB_ID}_pylib; trap 'rm -rf "$LIB"' EXIT
  module load conda; conda activate crl; PY=python
  python -m pip install -q --target "$LIB" h5py
  export PYTHONPATH=$LIB
fi
$PY -u scripts/eci/v2/frogs_paper/sleap_handedness.py
