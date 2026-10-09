#!/bin/bash
#
# Native-resolution pilot for mice ECI (scripts/eci/native_res.py): R448 vs U896 (512 frames upsampled) vs N896
# (native 2064 px frames downscaled), same frames, same fg448 mask, same SAE recipe. STEP = plan | decode | encode |
# train | evaluate | mil | summary. Outputs in results/vision/eci_native_res/ (gitignored). Inputs are staged to
# /localhome/$USER/$SLURM_JOB_ID and removed at the end. GPU: 'gpu' partition (never gpu100), excluding gpu150, gpu242.
#
# Usage (in order; decode is a CPU array of 16 tasks, at most 16 at a time):
#   STEP=plan     sbatch --export=ALL --partition=defaultp --gres=none --cpus-per-task=4 --mem=48G --time=00:40:00 scripts/eci/native_res.sh
#   STEP=decode   sbatch --export=ALL --partition=defaultp --gres=none --cpus-per-task=8 --mem=16G --time=03:00:00 --array=0-15%16 scripts/eci/native_res.sh
#   STEP=encode   sbatch --export=ALL --time=02:30:00 scripts/eci/native_res.sh
#   STEP=train    sbatch --export=ALL --time=01:30:00 scripts/eci/native_res.sh
#   STEP=evaluate EXTRA=--sheets sbatch --export=ALL --time=01:30:00 scripts/eci/native_res.sh
#   STEP=mil      sbatch --export=ALL --time=02:00:00 scripts/eci/native_res.sh
#   python scripts/eci/native_res.py summary      (login-node safe: reads JSONs)
#
# Measured 2026-10-09 (results/vision/eci_native_res/summary.json; B subset weighted to the natural base rate,
# cross-fitted best neuron, mean +- sd over 3 SAE seeds; ceiling = mil_ctx AP, 3 seeds x 5 pool folds):
#   decode  30,000 frames of 244 videos, 31.6 CPU-h of ffmpeg (16 tasks x 8 cores, 41-77 min wall each), 377 GB staged;
#           frame check: native->512 closest to the dataset JPEG of the same frame in 695/732 checks (rest near-ties)
#   tokens/frame  R448 151.9 (B) / 146.9 (train); U896 = N896 = 4x (607.7 / 587.5). Encode 122.7 GPU-s / 1k frames (A40)
#                    cf AUROC        cf AP          honest top-1%   size-ctrl AUROC   FVE     ceiling AP
#   nose_nose R448   0.694+-0.042    0.057+-0.015   0.125+-0.043    0.680             0.853   0.244
#             U896   0.699+-0.034    0.063+-0.012   0.136+-0.056    0.674             0.874   0.333
#             N896   0.682+-0.011    0.053+-0.002   0.105+-0.017    0.656             0.879   0.329
#   nose_tail R448   0.718+-0.053    0.022+-0.002   0.068+-0.036    0.672                     0.236
#             U896   0.751+-0.023    0.037+-0.004   0.074+-0.012    0.708                     0.251
#             N896   0.768+-0.015    0.063+-0.017   0.124+-0.021    0.723                     0.274
#   Pre-registered 'real detail helps': PASSES for nose_tail on AP only (N896 mean 0.063 above every R448 / U896 seed,
#   2.86x R448; AUROC +0.050 but not above every U896 seed); FAILS for nose_nose. Stretch top-1% >= 0.40: no.
#   Ceilings: nose_tail N896 above every seed of both (0.274 vs <= 0.257) but only 1.16x R448; nose_nose N896 = U896.
#   Video-level (count-controlled) check: not run - the B subset is a sparse positive-enriched sample, not whole videos.
#SBATCH --job-name=native_res
#SBATCH --output=logs/native_res_%x_%A_%a.out
#SBATCH --time=01:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --exclude=gpu150,gpu242
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G

module load conda
conda activate crl
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
mkdir -p logs
export STAGE=/localhome/$USER/${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}
mkdir -p $STAGE
trap 'rm -rf $STAGE' EXIT
echo "host $(hostname) job ${SLURM_JOB_ID:-} task ${SLURM_ARRAY_TASK_ID:-} step $STEP"
df -h /localhome | tail -1
case $STEP in
  plan)     python -u scripts/eci/native_res.py plan ${EXTRA:-} ;;
  decode)   python -u scripts/eci/native_res.py decode --workers ${SLURM_CPUS_PER_TASK:-8} ${EXTRA:-} ;;
  encode)   nvidia-smi -L; python -u scripts/eci/native_res.py encode ${EXTRA:-} ;;
  train)    nvidia-smi -L; python -u scripts/eci/native_res.py train ${EXTRA:-} ;;
  evaluate) nvidia-smi -L; python -u scripts/eci/native_res.py evaluate ${EXTRA:-} ;;
  mil)      nvidia-smi -L; python -u scripts/eci/native_res.py mil --steps 3000 ${EXTRA:-} ;;
esac
