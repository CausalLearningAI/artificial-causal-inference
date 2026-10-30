#!/bin/bash
# CPU wrapper for scripts/eci/v2/mask5_kpg.py: SET = cal|mab|crim (score then sheet)
#SBATCH --output=/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference/results/vision/eci_v2_0B/mask_approval_v3/logs/%x_%j.out
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
module load conda
conda activate crl
export PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
python -u scripts/eci/v2/mask5_kpg.py score --set $SET && python -u scripts/eci/v2/mask5_kpg.py sheet --set $SET && echo done
