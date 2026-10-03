#!/bin/bash
#
# Foreground pipeline step 3: Matryoshka BatchTopK SAE (1024 latents, prefixes 128/256/512/1024,
# k=16, AuxK, lr 5e-4 warmup+cosine, batch 4096, 5 epochs = ~82.5k steps) on the foreground tokens of
# dataset/mice/v1/eci/train_tokens/dinov2_base_l-1_fg448_fps1 (held in CPU RAM).
# Same 8 held-out pools as the ep20 SAE. Array index = seed.
# Output: dataset/mice/v1/eci/sae/matryoshka_btk_1024_k16_fg448_s{seed}/
#
# Usage: sbatch scripts/eci/train_sae_fg.sh
#   mask v3, static tokens / motion input [token_t, token_t - token_(t-2)]:
#   T=dataset/mice/v1/eci/train_tokens/dinov2_base_l-1_fgv3_fps1_d2
#   EXTRA_ARGS="--tokens-dir $T --tag fgv3" sbatch --export=ALL scripts/eci/train_sae_fg.sh
#   EXTRA_ARGS="--tokens-dir $T --tag fgv3m2 --motion" sbatch --export=ALL scripts/eci/train_sae_fg.sh
#   ants full frame -> dataset/ants/eci/sae/matryoshka_btk_1024_k16_antsfull_s0 (35.5M train tokens, 43k steps, ~3 min):
#   DOMAIN=ants EXTRA_ARGS="--tokens-dir dataset/ants/eci/train_tokens/dinov2_base_l-1_antsfull448_fps1_pf25 --tag antsfull" \
#       sbatch --export=ALL --array=0 scripts/eci/train_sae_fg.sh
#   then codes: DOMAIN=ants SAE=matryoshka_btk_1024_k16_antsfull_s0 fg_encode_all.sh / fg_encode_merge.sh (rule all
#   = mean / max over all 1024 patches, n_fg = 1024), somp_encode_all.sh / somp_encode_merge.sh
#
#   DINOv3 tokens (fg_extract_train.sh --encoder dinov3_base) -> the checkpoint records 'encoder' and fg_encode_all uses it:
#   EXTRA_ARGS="--tokens-dir dataset/mice/v1/eci/train_tokens/dinov3_base_l-1_fg512_fps1 --tag fg512v3" \
#       sbatch --export=ALL --array=0 scripts/eci/train_sae_fg.sh
#   DOMAIN=ants EXTRA_ARGS="--tokens-dir dataset/ants/eci/train_tokens/dinov3_base_l-1_antsfg512_fps1 --tag antsfgv3" \
#       sbatch --export=ALL --array=0 scripts/eci/train_sae_fg.sh
#   then fg_encode_all.sh with EXTRA_ARGS="--batch-size 64 --num-workers 16" (two encoders' pixels per frame: the
#   default 128 x 22 workers prefetch does not fit in 64 GB; the same holds for fg_extract_train.sh with 96 GB)
#
#   Background-subtracted input (token - the video's empty-arena background token, leaked positions filled; same store
#   and mask) and motion input ([token_t, token_t - token_(t-5)], 1 s at 5 fps; store from fg_extract_train.sh
#   --motion-delta 5). GPU jobs go to the default 'gpu' partition (-p gpu --gres=gpu:1 overrides the H100 request):
#   EXTRA_ARGS="--bg-sub --tag fg448bg" sbatch --export=ALL -p gpu --gres=gpu:1 --mem=320G --array=0 scripts/eci/train_sae_fg.sh
#   EXTRA_ARGS="--tokens-dir dataset/mice/v1/eci/train_tokens/dinov2_base_l-1_fg448_fps1_d5 --motion --tag fg448mot" \
#       sbatch --export=ALL -p gpu --gres=gpu:1 --mem=450G --array=0 scripts/eci/train_sae_fg.sh
#   DOMAIN=ants EXTRA_ARGS="--tokens-dir dataset/ants/eci/train_tokens/dinov2_base_l-1_antsfg_fps1 --bg-sub --tag antsfgbg" ...
#   DOMAIN=ants EXTRA_ARGS="--tokens-dir dataset/ants/eci/train_tokens/dinov2_base_l-1_antsfg_fps1_d5 --motion --tag antsfgmot" ...
#
#   Measured (results/vision/eci_bgmot/<domain>/compare.json): held-out FVE@1024 mice 0.864 ref / 0.824 bg / 0.789 mot,
#   ants 0.880 / 0.840 / 0.832; contact readout 0.809 / 0.793 / 0.816, grooming 0.955 / 0.925 / 0.950; patch-position
#   share of the latents mice 0.24 / 0.11 / 0.13, ants 0.16 / 0.18 / 0.09. Neither input clearly beats the reference.
#
#   Odor-aligned frames (fg_background.sh / fg_extract_train.sh with --align odor; checkpoint records align='odor', so
#   fg_encode_all.sh rotates the frames and reads fg448al/background):
#   EXTRA_ARGS="--tokens-dir dataset/mice/v1/eci/train_tokens/dinov2_base_l-1_fg448al_fps1 --tag fg448al" \
#       sbatch --export=ALL -p gpu --gres=gpu:1 --mem=240G --array=0 scripts/eci/train_sae_fg.sh
#   Measured (results/vision/eci_align/mice/): held-out FVE 0.796/0.827/0.849/0.864 (ref 0.797/0.828/0.848/0.864);
#   contact readout 0.816 (ref 0.809), best single latent 0.680 (ref 0.745); patch-position share of the latents
#   0.20 (ref 0.24); arena maps from TR- and BL-corner videos correlate 0.88 (activation-weighted, ref 0.66).
#
#   Odor-aligned FULL frame ff448al (rule all, 25% patches; ~205 GB store, so ~2x the fg448al memory):
#   EXTRA_ARGS="--tokens-dir dataset/mice/v1/eci/train_tokens/dinov2_base_l-1_ff448al_fps1_pf25 --tag ff448al" \
#       sbatch --export=ALL -p gpu --gres=gpu:1 --mem=480G --array=0 scripts/eci/train_sae_fg.sh
#   (whole chain with dependencies: scripts/eci/ff448al_chain.sh)
#
#SBATCH --job-name=eci_sae_fg
#SBATCH --output=logs/eci_sae_fg_%A_%a.out
#SBATCH --error=logs/eci_sae_fg_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=06:00:00
#SBATCH --partition=gpu100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --gres=gpu:H100:1

module load conda
conda activate crl

export PYTHONUNBUFFERED=1
set -euo pipefail

cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
mkdir -p logs

python -u scripts/eci/train_sae_fg.py --domain "${DOMAIN:-mice}" --seed ${SLURM_ARRAY_TASK_ID} ${EXTRA_ARGS:-}
