#!/bin/bash
#
# Spatial-SAE pilot (scripts/eci/spatial_sae_pilot.py). One script, three steps (STEP=train | encode | align).
#   train   array 0-3 = btk, matryoshka, spatial, spatial_static -> $OUT/<domain>/<model>/sae.pt
#   encode  all four + the deployed SAE on the eval frames   -> $OUT/<domain>/codes/<name>/codes_{max,mean}.npy
#   align   CPU, per-latent AUROC / AP per behaviour          -> $OUT/<domain>/codes/align_best.json
# Defaults: mice (fg448 store, deployed fg448), 100 train videos; ants: DOMAIN=ants NTRAIN=128 DEPLOYED=antsfg.
# GPU jobs go to the 'gpu' partition (any GPU but the 11 GB 2080 Ti node gpu150 and gpu242, whose GPU 0 failed on 2026-10-06); do not use gpu100.
#
# Usage (chain):
#   OUT=results/vision/eci_spatial_sae
#   a=$(OUT=$OUT STEP=train sbatch --parsable --export=ALL --array=0-3 scripts/eci/spatial_sae_pilot.sh)
#   e=$(OUT=$OUT STEP=encode sbatch --parsable --export=ALL --dependency=afterok:$a scripts/eci/spatial_sae_pilot.sh)
#   OUT=$OUT STEP=align sbatch --export=ALL --partition=defaultp --gres=none --dependency=afterok:$e scripts/eci/spatial_sae_pilot.sh
#
# Measured 2026-10-06 (results/vision/eci_spatial_sae/<domain>/codes*/align_best.json; best single latent, max pooling,
# AUROC / max AP; mice = 172,800 frames of the 144 annotated videos, none seen in training; ants = 76,800 frames of
# 128 held-out videos). Mice nose-nose / nose-tail, seeds 0 and 1:
#   btk         0.642/0.039, 0.670/0.046 | 0.683/0.030, 0.723/0.035   FVE 0.854
#   matryoshka  0.739/0.059, 0.644/0.036 | 0.765/0.034, 0.714/0.037   FVE 0.848
#   spatial     0.621/0.031, 0.618/0.032 | 0.673/0.026, 0.674/0.021   FVE 0.688-0.690
#   deployed fg448 (trained on 67.6M tokens incl. the eval videos) 0.759/0.075 | 0.722/0.025, FVE 0.863
# Ants grooming (any): btk 0.843, matryoshka 0.933, spatial 0.817 (0.877 with --frames-per-batch 84 = ~4.1k tokens),
# deployed antsfg 0.904. The Spatial-SAE does not give better behaviour latents here.
#SBATCH --job-name=eci_ssae_pilot
#SBATCH --output=logs/eci_ssae_pilot_%A_%a.out
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu150,gpu242
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G

module load conda
conda activate crl
export PYTHONUNBUFFERED=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
mkdir -p logs
DOMAIN=${DOMAIN:-mice}
NTRAIN=${NTRAIN:-100}
DEPLOYED=${DEPLOYED:-fg448}
EPOCHS=${EPOCHS:-5}
D=${OUT}/${DOMAIN}
MODELS=(btk matryoshka spatial spatial_static)
case ${STEP} in
  train)
    M=${MODELS[${SLURM_ARRAY_TASK_ID:-0}]}
    nvidia-smi -L
    python -u scripts/eci/spatial_sae_pilot.py train --domain $DOMAIN --model $M --n-train-videos $NTRAIN \
        --epochs $EPOCHS --out-dir $D/$M ;;
  encode)
    nvidia-smi -L
    CK=""
    for M in "${MODELS[@]}"; do [ -f $D/$M/sae.pt ] && CK="$CK $M=$D/$M/sae.pt"; done
    python -u scripts/eci/spatial_sae_pilot.py encode --domain $DOMAIN --n-train-videos $NTRAIN --out-dir $D/codes \
        --ckpt $CK deployed=dataset/${DOMAIN/mice/mice\/v1}/eci/sae/matryoshka_btk_1024_k16_${DEPLOYED}_s0/sae.pt ;;
  align)
    python -u scripts/eci/spatial_sae_pilot.py align --domain $DOMAIN --codes-dir $D/codes --deployed $DEPLOYED \
        --check-equal ;;
esac
