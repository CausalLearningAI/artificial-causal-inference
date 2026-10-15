#!/bin/bash
#
# MASK test 2 launcher (scripts/eci/mask_test2.py): M_dino (DINOv2 token cue + plain 1-patch dilation) / M_dino_cur
# (token cue + the current dilation) vs M_today / M_pix. GPU steps on the 'gpu' partition (never gpu100), excluding
# gpu150 / gpu242 / gpu230 / gpu241; CPU steps on defaultp. Staging in /localhome/$USER/<job> ($TMPDIR when
# /localhome is missing), removed on exit. Outputs: results/vision/eci_mask_test2/ (gitignored).
# Usage (from the repo root):
#   mkdir -p results/vision/eci_mask_test2/logs
#   STEP=cue sbatch --cpus-per-task=16 --mem=160G --time=01:30:00 scripts/eci/mask_test2.sh
#   STEP=encode DOMAIN=ants NTASKS=4 sbatch --array=0-3 --time=00:45:00 scripts/eci/mask_test2.sh
#   STEP=accuracy sbatch -p defaultp --gres=none --cpus-per-task=8 --mem=64G --time=01:00:00 scripts/eci/mask_test2.sh
#   STEP=train DOMAIN=mice MASK=dino sbatch --time=01:15:00 scripts/eci/mask_test2.sh
#   STEP=evaluate DOMAIN=mice MASK=dino sbatch --mem=180G --time=01:45:00 scripts/eci/mask_test2.sh
#   STEP=readouts sbatch -p defaultp --gres=none --cpus-per-task=8 --mem=64G --time=01:00:00 scripts/eci/mask_test2.sh
#SBATCH --job-name=mask_test2
#SBATCH --output=results/vision/eci_mask_test2/logs/%x_%A_%a.out
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu150,gpu242,gpu230,gpu241
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

module load conda
conda activate crl
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1 HF_HUB_OFFLINE=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
TAG=${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}_mt2
if [ -d /localhome ] && [ -w /localhome/$USER -o -w /localhome ]; then
    export LOCAL_DIR=/localhome/$USER/$TAG
else
    export LOCAL_DIR=${TMPDIR:-/tmp}/$TAG
fi
mkdir -p $LOCAL_DIR
trap 'rm -rf $LOCAL_DIR' EXIT
echo "host $(hostname) step ${STEP} domain ${DOMAIN:-} mask ${MASK:-} local $LOCAL_DIR"
df -h $LOCAL_DIR | tail -1
nvidia-smi -L 2>/dev/null || true
DOMAIN=${DOMAIN:-mice}
MASK=${MASK:-dino}
EXTRA=${EXTRA:-}
case ${STEP} in
  cue|accuracy)
    python -u scripts/eci/mask_test2.py $STEP --domains ${DOMAINS:-mice,ants} --workers ${SLURM_CPUS_PER_TASK:-8} $EXTRA ;;
  readouts|clean)
    python -u scripts/eci/mask_test2.py $STEP $EXTRA ;;
  encode)
    python -u scripts/eci/mask_test2.py encode --domain $DOMAIN --task ${SLURM_ARRAY_TASK_ID:-0} --n-tasks ${NTASKS:-2} \
      --num-workers $(( ${SLURM_CPUS_PER_TASK:-8} - 2 )) $EXTRA ;;
  train|evaluate)
    python -u scripts/eci/mask_test2.py $STEP --domain $DOMAIN --mask $MASK $EXTRA ;;
esac
