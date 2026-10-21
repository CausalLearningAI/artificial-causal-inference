#!/bin/bash
#
# ECI concept atlas v37 builds (scripts/eci/build_atlas_v37.py), CPU only. The committed code (git archive of HEAD) runs
# from the job's local dir, with data / dataset / results symlinked back, so edits other sessions make to the shared
# checkout do not leak in; the commit is written to results/vision/eci_atlas_v37/code/<job>.json.
#   ARGS="--dom ants_v2 --model new --steps prep,data,patch,arena,render,dump" sbatch scripts/eci/build_atlas_v37.sh
#   ARGS="--dom mice --model online --steps prep,dump" sbatch scripts/eci/build_atlas_v37.sh
#   ARGS="--steps assemble" sbatch scripts/eci/build_atlas_v37.sh
# WORKTREE=1 runs the working tree instead (tests of uncommitted changes to the v37 files only).
#SBATCH --job-name=atlas37
#SBATCH --output=results/vision/eci_atlas_v37/logs/%x_%j.out
#SBATCH --time=04:00:00
#SBATCH --partition=defaultp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
module load conda
conda activate crl
command -v git >/dev/null 2>&1 || module load git/2.35.1
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1
set -euo pipefail
REPO=/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
cd $REPO
export LOCAL_DIR=/localhome/$USER/${SLURM_JOB_ID:-local}_atlas37
mkdir -p results/vision/eci_atlas_v37/code
mkdir -p $LOCAL_DIR 2>/dev/null || { export LOCAL_DIR=${TMPDIR:-/tmp}/${USER}_${SLURM_JOB_ID:-local}_atlas37; mkdir -p $LOCAL_DIR; echo "no /localhome on $(hostname): staging in $LOCAL_DIR"; }
trap 'rm -rf $LOCAL_DIR' EXIT
HEAD=$(git rev-parse HEAD)
CODE=$REPO
if [ -z "${WORKTREE:-}" ]; then
  mkdir -p $LOCAL_DIR/code
  git archive HEAD src scripts/eci configs atlas | tar -x -C $LOCAL_DIR/code
  for d in data dataset results; do ln -s $REPO/$d $LOCAL_DIR/code/$d; done
  CODE=$LOCAL_DIR/code
fi
printf '{"head": "%s", "code_root": "%s", "worktree": "%s", "args": "%s", "time": "%s"}\n' "$HEAD" "$CODE" "${WORKTREE:-}" \
  "${ARGS:-}" "$(date -Is)" > results/vision/eci_atlas_v37/code/${SLURM_JOB_ID:-local}.json
echo "atlas v37: HEAD $HEAD, code $CODE, args ${ARGS:-}"
cd $CODE
eval python -u scripts/eci/build_atlas_v37.py ${ARGS:-}
