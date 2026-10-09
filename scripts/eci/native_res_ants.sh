#!/bin/bash
#
# Native-resolution test for ants ECI (scripts/eci/native_res_ants.py): R448 vs U896 (512 frames upsampled) vs N896
# (native 770 / 824 px source frames upsampled), same frames, same antsfg mask, same SAE recipe. STEP = plan | offset |
# bench | encode | train | evaluate (summary runs on the login node: python scripts/eci/native_res_ants.py summary).
# Outputs in results/vision/eci_native_res_ants/ (gitignored). Staging in /localhome/$USER/$SLURM_JOB_ID, removed at the
# end. GPU: 'gpu' partition (never gpu100), excluding gpu150, gpu242.
#
# Usage (in order):
#   STEP=plan   sbatch --export=ALL --partition=defaultp --gres=none --cpus-per-task=4 --mem=32G --time=00:30:00 scripts/eci/native_res_ants.sh
#   STEP=offset sbatch --export=ALL --partition=defaultp --gres=none --cpus-per-task=4 --mem=16G --time=00:30:00 scripts/eci/native_res_ants.sh
#   STEP=bench  sbatch --export=ALL --time=00:20:00 scripts/eci/native_res_ants.sh
#   STEP=encode EXTRA="--arm n896 --dtype ..." sbatch --export=ALL --time=... scripts/eci/native_res_ants.sh   (and --arm u896)
#   STEP=train    sbatch --export=ALL --time=01:00:00 scripts/eci/native_res_ants.sh
#   STEP=evaluate EXTRA="--arms l448 r448 u896 n896 --sheets" sbatch --export=ALL --time=01:30:00 scripts/eci/native_res_ants.sh
#SBATCH --job-name=native_res_ants
#SBATCH --output=logs/native_res_ants_%x_%j.out
#SBATCH --time=01:00:00
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
export STAGE=/localhome/$USER/${SLURM_JOB_ID:-local}
mkdir -p $STAGE
trap 'rm -rf $STAGE' EXIT
echo "host $(hostname) job ${SLURM_JOB_ID:-} step $STEP"
df -h /localhome | tail -1
case $STEP in
  plan|offset|offset_all) python -u scripts/eci/native_res_ants.py $STEP ${EXTRA:-} ;;
  bench|encode|train|evaluate) nvidia-smi -L; python -u scripts/eci/native_res_ants.py $STEP ${EXTRA:-} ;;
  smoke)  # whole chain on CPU on tiny frame sets, in the job dir; results copied to OUT/smoke
    export NRA_OUT=$STAGE/smoke NRA_DEV=cpu
    mkdir -p $NRA_OUT
    cp results/vision/eci_native_res_ants/offset.json $NRA_OUT/
    P="python -u scripts/eci/native_res_ants.py"
    $P plan --smoke
    export NRA_FAKE_MODEL=1  # the real DINOv2 forward costs ~27 s / frame on CPU (checked on 16 frames, n896 train)
    $P encode --arm n896 --dtype fp32 --workers 2 --bs 4 --splits train
    $P encode --arm n896 --dtype fp32 --workers 2 --bs 4 --splits eval --part 0 --nparts 2
    $P encode --arm n896 --dtype fp32 --workers 2 --bs 4 --splits eval --part 1 --nparts 2
    $P encode --arm u896 --dtype fp32 --workers 2 --bs 4
    $P train --steps 50 --seeds 0 1
    $P evaluate --arms l448 r448 u896 n896 --seeds 0 1 --sheets
    $P summary
    rm -f $NRA_OUT/*/tok_*.npy
    mkdir -p results/vision/eci_native_res_ants/smoke
    cp -r $NRA_OUT/summary.json $NRA_OUT/eval $NRA_OUT/sheets $NRA_OUT/encode_*.json results/vision/eci_native_res_ants/smoke/ ;;
esac
