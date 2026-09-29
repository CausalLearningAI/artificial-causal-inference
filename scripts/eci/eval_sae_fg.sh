#!/bin/bash
#
# Foreground pipeline step 4b: metrics table fg448 SAEs vs ep20 (token FVE/L0/dead, stability,
# frame firing, behaviour AUROC) -> dataset/mice/v1/eci/fg448/eval_fg_vs_ep20.json
#
#SBATCH --job-name=eci_fg_eval
#SBATCH --output=logs/eci_fg_eval_%j.out
#SBATCH --error=logs/eci_fg_eval_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1

module load conda
conda activate crl

export PYTHONUNBUFFERED=1
set -euo pipefail

cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
python -u scripts/eci/eval_sae_fg.py ${EXTRA_ARGS:-}
