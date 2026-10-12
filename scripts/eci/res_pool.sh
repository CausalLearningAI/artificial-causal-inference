#!/bin/bash
#
# RES_POOL test (scripts/eci/res_pool.py): input resolution (c448 / n896 / u896) x pooling read-out (max / topk / mean)
# inside the final ECI pipeline C1 (M5 motion), mice + ants. STEP = plan | decode | encode_train | train | evaluate |
# score | sheets (summary is login safe: python scripts/eci/res_pool.py summary). TASK = array index (decode tasks,
# eval shards). GPU: 'gpu' partition, L40S (fp32 896 encodes), never gpu100, excluding gpu150 / gpu242 / gpu230.
# Staging in /localhome/$USER/<job> (removed on exit). Outputs: results/vision/eci_res_pool/ (gitignored).
# RP_OUT=<dir> runs everything in another output dir (smoke test: plan --smoke).
#
# Usage (from the repo root; mkdir -p results/vision/eci_res_pool/logs first):
#   STEP=plan   sbatch -p defaultp --gres=none --cpus-per-task=4 --mem=64G --time=00:40:00 scripts/eci/res_pool.sh
#   STEP=decode EXTRA="--workers 8" sbatch -p defaultp --gres=none --cpus-per-task=8 --mem=32G --time=02:30:00 --array=0-15%16 scripts/eci/res_pool.sh
#   STEP=encode_train EXTRA="--domain mice --arm n896" sbatch --time=00:50:00 scripts/eci/res_pool.sh
#   STEP=train  EXTRA="--domain mice --arm n896" sbatch --time=00:40:00 scripts/eci/res_pool.sh
#   STEP=evaluate EXTRA="--domain mice --arms n896" sbatch --array=0-3 --time=01:15:00 scripts/eci/res_pool.sh
#   STEP=evaluate EXTRA="--domain mice --arms c448 c448dep" sbatch --time=01:15:00 scripts/eci/res_pool.sh
#   STEP=score  EXTRA="--domain mice --arms n896" sbatch --time=01:30:00 scripts/eci/res_pool.sh
#   STEP=sheets sbatch --time=00:30:00 scripts/eci/res_pool.sh
#
# Measured 2026-10-12 (results/vision/eci_res_pool/summary.json, tables.md; mean over 3 SAE seeds; cf = cross-fitted
# best neuron; video r ctl = count-controlled held-out per-video mean vs annotated rate). k_top: mice 11 / 45, ants 5 / 21.
#   c448dep (deployed C1) MAX reproduces eci_combined C1 exactly (max abs diff 0.0, all labels / seeds, both domains).
#   frame map: dataset frame k = sequential source frame 6k+2 exactly (res_pool_offset.py); native_res seek decode = 6k+3.
#                      cf AUROC  c448 / n896 / u896      cf AP  c448 / n896          video r ctl (MAX) c448 / n896
#   mice nose_nose     0.700 / 0.718 / -                 0.051 / 0.054               0.19 / 0.20
#   mice nose_tail     0.753 / 0.793 / -                 0.049 / 0.060               -0.06 / 0.04
#   ants groom_any     0.883 / 0.863 / 0.839             0.753 / 0.745               0.46 / 0.55
#   ants groom_yellow  0.836 / 0.817 / 0.800             0.480 / 0.505               0.34 / 0.33
#   ants groom_blue    0.842 / 0.836 / 0.812             0.499 / 0.527               0.39 / 0.47
#   ants onlid_yellow  0.963 / 0.977 / 0.980             0.394 / 0.518               0.89 / 0.82
#   Decision 1: mice n896 WINS on one criterion only (nose_tail ctl video r +0.105 >= 0.10; AUROC +0.039 but mean 0.793
#     < c448 best seed 0.796; AP x1.21). Ants n896 does NOT win (groom_any -0.0203 > 0.02 drop; no gain). u896 < c448.
#   Decision 2: neither TOPK nor MEAN replaces MAX (TOPK passes only mice n896: ctl r +0.052); MAX stays.
#   Decision 3: with motion there is no ants 896 video drop under MAX (n896 - c448 ctl r mean +0.020); under TOPK +0.052,
#     MEAN +0.072 (positive on all 4 labels).
#   Compute: GPU 11.2 h (L40S); mice decode 88.4 CPU-h of ffmpeg (148.8 CPU-h allocated incl. idle cores), over the
#     80 CPU-h budget; 896 fp32 encode 0.055 GPU-s / frame.
#SBATCH --job-name=res_pool
#SBATCH --output=results/vision/eci_res_pool/logs/%x_%A_%a.out
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --exclude=gpu150,gpu242,gpu230
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G

module load conda
conda activate crl
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
export LOCAL_DIR=/localhome/$USER/${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}_rp
mkdir -p $LOCAL_DIR
trap 'rm -rf $LOCAL_DIR' EXIT
echo "host $(hostname) job ${SLURM_JOB_ID:-} task ${SLURM_ARRAY_TASK_ID:-} step $STEP out ${RP_OUT:-default}"
df -h /localhome | tail -1
nvidia-smi -L 2>/dev/null || true
TASK=${TASK:-${SLURM_ARRAY_TASK_ID:-0}}
P="python -u scripts/eci/res_pool.py"
if [ "$STEP" = "offset" ]; then  # CPU frame-mapping check of the mice native decoding
  python -u scripts/eci/res_pool_offset.py ${EXTRA:-}
elif [ "$STEP" = "smoke" ]; then  # GPU chain on the plan --smoke frames of DOMAIN (RP_OUT must point to the smoke dir)
  D=${DOMAIN:-ants}
  $P encode_train --domain $D --arm n896 --workers 4
  $P train --domain $D --arm n896 --steps 200
  $P train --domain $D --arm c448 --steps 200
  $P evaluate --domain $D --arms c448 c448dep
  $P evaluate --domain $D --arms n896 --workers 4
  $P score --domain $D --arms c448 c448dep n896
  if [ "$D" = "mice" ]; then $P sheets; fi
  $P summary
else
  $P $STEP --task $TASK ${EXTRA:-}
fi
