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
#   STEP=smoke  (CPU, tiny frame sets, stand-in model)   python scripts/eci/native_res_ants.py summary (login safe)
#
# Measured 2026-10-09 (results/vision/eci_native_res_ants/summary.json; 76,800 eval frames = all store frames of the 128
# eval videos at 1 fps; cross-fitted best neuron over the levers video halves; mean +- sd over 3 SAE seeds; video r =
# held-out per-video presence mean vs annotated rate, count-controlled = residual on the per-video kept-patch count):
#   frame map  exact reproduction of the standardisation on all 256 videos (start + end): v3 = 6k+1 (212), v2 = 6k+2 (44)
#   tokens/frame  R448 50.0 (eval) / 49.1 (train); U896 = N896 = 4x. Train 20,480 frames (160 x 128 videos), 24.6M
#                 presentations per SAE in every arm. L448 = levers w4096 checkpoints rescored here (reproduces levers / T2).
#                     cf AUROC       cf AP          top-1%   size-ctrl   video r   video r count-ctrl
#   groom_any    R448 0.855+-0.033   0.734+-0.031   0.873    0.808       0.820     0.547+-0.030
#                U896 0.868+-0.005   0.736+-0.025   0.918    0.809       0.768     0.464+-0.070
#                N896 0.891+-0.030   0.777+-0.055   0.947    0.854       0.788     0.502+-0.078
#   groom_yellow R448 0.825+-0.022   0.499+-0.010   0.601    0.770       0.685     0.445+-0.018
#                U896 0.836+-0.009   0.515+-0.043   0.769    0.765       0.647     0.379+-0.086
#                N896 0.848+-0.027   0.550+-0.040   0.757    0.805       0.648     0.387+-0.062
#   groom_blue   R448 0.823+-0.025   0.482+-0.028   0.735    0.783       0.669     0.457+-0.050
#                U896 0.833+-0.007   0.543+-0.021   0.863    0.790       0.642     0.416+-0.035
#                N896 0.858+-0.016   0.549+-0.005   0.848    0.823       0.650     0.434+-0.028
#   onlid_yellow R448 0.970+-0.004   0.621+-0.095   0.563    0.958       0.913     0.906+-0.014
#                U896 0.981+-0.001   0.728+-0.043   0.725    0.976       0.851     0.847+-0.032
#                N896 0.975+-0.008   0.724+-0.105   0.758    0.966       0.846     0.850+-0.067
#   Pre-registered rule N896 vs R448: WIN. No label loses AUROC; gains on groom_blue (+0.034, mean 0.858 > R448 best
#   seed 0.847) and groom_any (+0.036, mean 0.8913 > R448 best seed 0.8911: a 0.0002 margin, knife-edge). No AP gain
#   (best 1.17x on-lid), no video gain: count-controlled video r is LOWER than R448 on every label (-0.02 .. -0.06).
#   N896 vs U896 (real detail vs finer grid): no label passes (AUROC +0.013 .. +0.025 on grooming), not a win.
#   Caveats: N896 and R448 tokens are fp32; U896 eval parts 2-3 (38,400 frames) are fp16 (per-job precision rule,
#   mean cos to fp32 0.99999). N896 seed 0 is the high seed (groom_any 0.926; seeds 1-2 0.870 / 0.879).
#   Compute: 7.3 GPU-h (encode 6.6, train 0.3, evaluate 0.5; fp32 encode: 3090 0.115 s/frame, L40S 0.052; fp16 4-5x faster), ~1.3 CPU-h.
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
