#!/bin/bash
#
# Tadpole r4 job wrapper (scripts/eci/tadpole_r4.py). STEP = plan | locate | validate | prepass | train | encode | merge |
# ftheta | summaries | manifest; EXTRA = more arguments. Runs COMMITTED code only: CODE_COMMIT (required, pinned at
# submit time) is exported with `git archive` into the job's local dir and run from there; the commit is recorded in
# results/vision/eci_rollout_r4/frogs_tadpole/code/<step>_<job>.json. Staging in /localhome/$USER/<job>_<task>_tad
# (removed on exit); GPU utilisation / memory logged every 60 s.
#   CPU: sbatch -p defaultp --cpus-per-task=4 --mem=20G --time=00:40:00 --array=0-117 \
#          --export=ALL,STEP=locate,CODE_COMMIT=<sha> scripts/eci/tadpole_r4.sh        (locate: task = video)
#   GPU: sbatch -p gpu --gres=gpu:1 --exclude=gpu242,gpu150,gpu230,gpu241,gpu240 --cpus-per-task=24 --mem=64G \
#          --time=01:00:00 --array=0-29 --export=ALL,STEP=encode,CODE_COMMIT=<sha> scripts/eci/tadpole_r4.sh
#SBATCH --job-name=tad_r4
#SBATCH --output=results/vision/eci_rollout_r4/logs/%x_%A_%a.out
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
module load conda
conda activate crl
command -v git >/dev/null 2>&1 || module load git/2.35.1
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error
set -euo pipefail
REPO=/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
cd $REPO
STEP=${STEP:?}
CODE_COMMIT=${CODE_COMMIT:?}
export CODE_COMMIT
export LOCAL_DIR=/localhome/$USER/${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}_tad
C=$LOCAL_DIR/code
mkdir -p $C
SMI=""
if nvidia-smi -L >/dev/null 2>&1; then
  nvidia-smi -L
  nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used --format=csv,noheader -l 60 | sed -u 's/^/[gpu] /' &
  SMI=$!
fi
trap 'rm -rf $LOCAL_DIR; [ -n "$SMI" ] && kill $SMI 2>/dev/null || true' EXIT
git archive $CODE_COMMIT src scripts/eci scripts/benchmark configs | tar -x -C $C
for d in data dataset results; do ln -s $REPO/$d $C/$d; done
O=$REPO/results/vision/eci_rollout_r4/frogs_tadpole/code
mkdir -p $O
printf '{"commit": "%s", "step": "%s", "job": "%s", "task": "%s", "host": "%s", "extra": "%s", "time": "%s"}\n' \
  "$CODE_COMMIT" "$STEP" "${SLURM_JOB_ID:-}" "${SLURM_ARRAY_TASK_ID:-}" "$(hostname)" "${EXTRA:-}" "$(date -Is)" \
  > $O/${STEP}_${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}.json
echo "host $(hostname) job ${SLURM_JOB_ID:-} task ${SLURM_ARRAY_TASK_ID:-} step $STEP commit $CODE_COMMIT extra ${EXTRA:-}"
df -h /localhome | tail -1
T=""
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  case $STEP in locate) T="--task $SLURM_ARRAY_TASK_ID";; encode) T="--shard $SLURM_ARRAY_TASK_ID";; esac
fi
python -u $C/scripts/eci/tadpole_r4.py $STEP $T ${EXTRA:-}
