#!/bin/bash
# run several mask4_val.py steps in one job: CHAIN = "step args;step args;..." (":" for ",")
#SBATCH --output=/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference/results/vision/eci_v2_0B/mask_approval_v2/logs/%x_%j.out
module load conda
conda activate crl
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error PYTHONWARNINGS=ignore
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
nvidia-smi -L 2>/dev/null
echo "host $(hostname) job ${SLURM_JOB_ID:-} git $(git rev-parse --short HEAD 2>/dev/null)"
IFS=';' read -ra ST <<< "${CHAIN//:/,}"
for s in "${ST[@]}"; do echo "== $s"; python -u scripts/eci/v2/mask4_val.py $s || echo "FAILED: $s"; done
echo done
