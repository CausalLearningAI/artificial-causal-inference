#!/bin/bash
#
# Fused encoding (scripts/eci/fused_encode_all.py): all frames -> ONE DINOv2 forward (448, the SAE's foreground rule
# and alignment) -> pooled SAE codes (codes/<SAE>/shards) AND SOMP codes (codes/<SAE>_somp/shards). Replaces
# fg_encode_all.sh + somp_encode_all.sh for static dinov2_base SAEs. 24 shards, finished shards are skipped
# (resubmit to resume). Then merge + verify + delete shards + the _mean link with the existing scripts:
#
#   jid=$(SAE=<sae> sbatch --parsable --export=ALL scripts/eci/fused_encode_all.sh)
#   SAE=<sae> sbatch --export=ALL --dependency=afterok:${jid} scripts/eci/fused_encode_merge.sh
#
#SBATCH --job-name=eci_fused_enc
#SBATCH --output=logs/eci_fused_enc_%A_%a.out
#SBATCH --error=logs/eci_fused_enc_%A_%a.err
#SBATCH --array=0-23
#SBATCH --time=08:00:00
#SBATCH --partition=gpu100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --gres=gpu:1

module load conda
conda activate crl

export PYTHONUNBUFFERED=1
set -euo pipefail

cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
mkdir -p logs

python -u scripts/eci/fused_encode_all.py --domain "${DOMAIN:-mice}" --sae "${SAE:?set SAE}" --k "${K:-16}" --n-shards 24 \
    --shard ${SLURM_ARRAY_TASK_ID} --num-workers 22 ${EXTRA_ARGS:-}
