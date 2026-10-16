#!/bin/bash
#
# Atlas v37 methods (scripts/eci/atlas_v37_methods.py): DINES on the outcomes of one NES result set, and the mice
# genotype x phase-transition family C (NES + DINES). CPU only. Runs COMMITTED code only: HEAD is exported with
# `git archive` into the job's local dir (data / dataset / results symlinked back) and the script runs from there; the
# HEAD hash goes into <out>/code.json.
#   RES=results/vision/ants/eci_r4v2/nes/matryoshka_btk_512_k16_r4d2_s2_fnorm sbatch --export=ALL scripts/eci/atlas_v37_methods.sh
#   RES=results/vision/mice/eci/nes/matryoshka_btk_1024_k16_fg448al_s0 EXTRA="--hab-window whole" sbatch --export=ALL scripts/eci/atlas_v37_methods.sh
#SBATCH --job-name=v37_methods
#SBATCH --output=results/vision/eci_atlas_v37/methods/logs/%x_%j.out
#SBATCH --time=04:00:00
#SBATCH --partition=defaultp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
module load conda
conda activate crl
command -v git >/dev/null 2>&1 || module load git/2.35.1   # some CPU nodes (beta*) have no system git
command -v git >/dev/null 2>&1 || { echo 'no git: cannot snapshot the code'; exit 1; }
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
set -euo pipefail
REPO=/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
cd $REPO
RES=${RES:?}
EXTRA=${EXTRA:-}
export ATLAS_V37_HEAD=$(git rev-parse HEAD)
L=/localhome/$USER/${SLURM_JOB_ID:-local}_v37methods
mkdir -p $L
trap 'rm -rf $L' EXIT
git archive HEAD src scripts/eci configs | tar -x -C $L
for d in data dataset results; do ln -s $REPO/$d $L/$d; done
echo "code: HEAD $ATLAS_V37_HEAD exported to $L; RES $RES; EXTRA $EXTRA; host $(hostname)"
python -u $L/scripts/eci/atlas_v37_methods.py --res $L/$RES --workers ${SLURM_CPUS_PER_TASK:-1} $EXTRA
echo "done"
