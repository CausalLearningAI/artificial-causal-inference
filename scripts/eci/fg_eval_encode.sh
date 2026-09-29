#!/bin/bash
#
# Foreground pipeline step 4a: encode every 5 fps frame of the 18 annotated held-out videos with
# both foreground SAEs (for the behaviour AUROC check in scripts/eci/eval_sae_fg.py).
# Output: dataset/mice/v1/eci/fg448/eval_codes/task_XX.npz
#
# Usage: sbatch scripts/eci/fg_eval_encode.sh ; then python scripts/eci/eval_sae_fg.py (GPU)
#
#SBATCH --job-name=eci_fg_evenc
#SBATCH --output=logs/eci_fg_evenc_%A_%a.out
#SBATCH --error=logs/eci_fg_evenc_%A_%a.err
#SBATCH --array=0-5
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

module load conda
conda activate crl

export PYTHONUNBUFFERED=1
set -euo pipefail

cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
mkdir -p logs

python -u scripts/eci/fg_eval_encode.py --task ${SLURM_ARRAY_TASK_ID} --n-tasks 6 ${EXTRA_ARGS:-}
