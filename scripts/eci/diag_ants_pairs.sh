#!/bin/bash
#
# Region-read diagnostic of the deployed ants SAE (scripts/eci/diag_ants_pairs.py): Y2F vs B2F from location.
#   STEP=encode   GPU or CPU (the SAE pass is ~12 TFLOP; a 16-core CPU job takes minutes): store tokens -> sparse patch codes (+ check against stored codes_max) -> $OUT/patch_codes
#   STEP=analyze  CPU: region reads, neuron scan, baselines, contact sheet                     -> $OUT
# Inputs are staged to /localhome/$USER/$SLURM_JOB_ID, results copied back once, staging removed.
# Usage:
#   e=$(STEP=encode sbatch --parsable --export=ALL --partition=defaultp --gres=none scripts/eci/diag_ants_pairs.sh)
#   STEP=analyze sbatch --export=ALL --partition=defaultp --gres=none --dependency=afterok:$e scripts/eci/diag_ants_pairs.sh
#SBATCH --job-name=diag_ants_pairs
#SBATCH --output=logs/diag_ants_pairs_%j.out
#SBATCH --time=03:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu150,gpu242
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G

module load conda
conda activate crl
export PYTHONUNBUFFERED=1
set -euo pipefail
REPO=/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
cd $REPO
mkdir -p logs
OUT=${OUT:-$REPO/results/vision/eci_repr_diag/ants}
STAGE=/localhome/$USER/$SLURM_JOB_ID
mkdir -p $STAGE $OUT
trap 'rm -rf $STAGE' EXIT
CODES=dataset/ants/eci/codes/matryoshka_btk_1024_k16_antsfg_s0/codes_max.npy
cp $CODES $STAGE/codes_max.npy
case ${STEP} in
  encode)
    command -v nvidia-smi >/dev/null && nvidia-smi -L || true
    cp -r dataset/ants/eci/train_tokens/dinov2_base_l-1_antsfg_fps1 $STAGE/store
    python -u scripts/eci/diag_ants_pairs.py encode --store $STAGE/store --codes-max $STAGE/codes_max.npy \
        --out $STAGE/out
    mkdir -p $OUT/patch_codes
    cp $STAGE/out/patch_codes/* $OUT/patch_codes/ ;;
  analyze)
    mkdir -p $STAGE/out
    cp -r $OUT/patch_codes $STAGE/out/
    python -u scripts/eci/diag_ants_pairs.py analyze --codes-max $STAGE/codes_max.npy --out $STAGE/out
    rm -rf $STAGE/out/patch_codes
    cp -r $STAGE/out/* $OUT/ ;;
esac
