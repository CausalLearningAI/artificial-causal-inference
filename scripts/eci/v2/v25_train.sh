#!/bin/bash
# v2.5 SAE training, one job per domain, all arms x seeds (gpu100, approved; v2.5 total <= 8 H100-hours): DOM=mice sbatch v25_train.sh
#SBATCH --partition=gpu100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --output=/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference/results/vision/eci_v2_5/logs/%x_%j.out
set -euo pipefail
bash /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference/scripts/eci/v2/v22_run.sh v25_train.py train --dom $DOM ${EXTRA:-}
