#!/bin/bash
# v2.10 GPU steps (gpu100 = v29_train.sh): sbatch [-t ...] v210_gpu.sh <script.py> <args...>
#SBATCH --partition=gpu100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference/results/vision/eci_v2_10/logs/%x_%j.out
set -euo pipefail
bash /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference/scripts/eci/v2/v22_run.sh "$@"
