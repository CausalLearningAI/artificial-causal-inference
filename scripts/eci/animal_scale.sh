#!/bin/bash
#
# Animal scale measurement (scripts/eci/animal_scale.py), one CPU job per domain; videos staged to /localhome (removed by
# the script). Usage:
#   for d in ants_v2 ants_v3 mice_v1 frogs_v1 mabe22_mouse mabe22_beetle calms21; do
#       DOM=$d sbatch --export=ALL --job-name=ascale_$d scripts/eci/animal_scale.sh; done
#   python scripts/eci/animal_scale.py report          # aggregation (reads json only)
# DOM=<domain> TEST=--test for a 3-video smoke test.
#SBATCH --partition=defaultp
#SBATCH --output=logs/animal_scale_%x_%j.out
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

module load conda
conda activate crl
export PYTHONUNBUFFERED=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
echo "host $(hostname) job ${SLURM_JOB_ID:-} dom $DOM"
python -u scripts/eci/animal_scale.py measure --domain $DOM ${TEST:-}
