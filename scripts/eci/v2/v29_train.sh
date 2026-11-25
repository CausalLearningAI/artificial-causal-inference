#!/bin/bash
# v2.9 SAE training (gpu100 H100 as v2.7; v2.9 total <= 10 H100-hours): DOM=mice [EXTRA="--cfgs ... --max_steps N --tag T"] sbatch v29_train.sh
#SBATCH --partition=gpu100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=07:00:00
#SBATCH --output=/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference/results/vision/eci_v2_9/logs/%x_%j.out
set -euo pipefail
bash /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference/scripts/eci/v2/v22_run.sh v29_train.py train --dom $DOM ${EXTRA:-}
