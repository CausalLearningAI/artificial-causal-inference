#!/bin/bash
# v2.3 SAE training (gpu100): DOM=mice sbatch v23_train.sh ; a 3,000-step benchmark first (same job), then the full pass.
#SBATCH --partition=gpu100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference/results/vision/eci_v2_3/logs/%x_%j.out
set -euo pipefail
S=/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference/scripts/eci/v2
bash $S/v22_run.sh v23_train.py train --dom $DOM --seeds ${SEEDS:-2001,2002,2003} --max_steps 3000 --bench
bash $S/v22_run.sh v23_train.py train --dom $DOM --seeds ${SEEDS:-2001,2002,2003}
