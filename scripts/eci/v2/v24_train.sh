#!/bin/bash
# v2.4 SAE training (gpu100, approved, <= 5 H100-hours in total): DOM=mice P=2 ARMS=W,N sbatch v24_train.sh
#SBATCH --partition=gpu100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH --output=/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference/results/vision/eci_v2_4/logs/%x_%j.out
set -euo pipefail
S=/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference/scripts/eci/v2
bash $S/v22_run.sh v24_train.py train --dom $DOM --P $P --arms $ARMS --seeds ${SEEDS:-2401,2402,2403} ${EXTRA:-}
