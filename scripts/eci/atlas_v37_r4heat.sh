#!/bin/bash
#
# One GPU part of the atlas v37 R4 real heat maps (scripts/eci/atlas_v37_r4heat.py): the videos of KEY whose R4 full
# pass ran on GPU model GPU, every NTASKS-th from TASK. Inputs staged to /localhome/$USER/<job>_r4heat (removed on exit);
# GPU utilisation / memory logged every 60 s. Submitted by scripts/eci/atlas_v37_r4heat_key.sh (it sets the constraint
# matching GPU):
#   sbatch --constraint=L40S --export=ALL,KEY=ants_v3,GPU="NVIDIA L40S",TASK=0,NTASKS=2 scripts/eci/atlas_v37_r4heat.sh
#   MODE=assemble (CPU): sbatch -p defaultp --gres=none ... --export=ALL,KEY=ants_v3,MODE=assemble,MOVE=1 ...
#SBATCH --job-name=r4heat
#SBATCH --output=results/vision/eci_atlas_v37/logs/%x_%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
module load conda
conda activate crl
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
command -v git >/dev/null 2>&1 || module load git/2.35.1
export LOCAL_DIR=/localhome/$USER/${SLURM_JOB_ID:-local}_r4heat
mkdir -p $LOCAL_DIR
SMI=""
if nvidia-smi -L >/dev/null 2>&1; then
  nvidia-smi -L
  nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used --format=csv,noheader -l 60 | sed -u 's/^/[gpu] /' &
  SMI=$!
fi
trap 'rm -rf $LOCAL_DIR; [ -n "$SMI" ] && kill $SMI 2>/dev/null || true' EXIT
# the committed R4 recipe (git archive of HEAD: other sessions edit the shared checkout), data / dataset / results
# symlinked back; this script's own .py from the working tree
mkdir -p $LOCAL_DIR/code
git archive HEAD src scripts/eci configs | tar -x -C $LOCAL_DIR/code
for d in data dataset results; do ln -s $PWD/$d $LOCAL_DIR/code/$d; done
cp scripts/eci/atlas_v37_r4heat.py $LOCAL_DIR/code/scripts/eci/
echo "code: HEAD $(git rev-parse HEAD) in $LOCAL_DIR/code; working-tree diffs not used: $(git status --porcelain -- scripts/eci/rollout_r4.py src/eci scripts/eci/build_explorer.py | tr '\n' ' ')"
df -h /localhome | tail -1
echo "host $(hostname) job ${SLURM_JOB_ID:-} key $KEY mode ${MODE:-gpu} gpu ${GPU:-} task ${TASK:-0}/${NTASKS:-1} move ${MOVE:-}"
if [ "${MODE:-gpu}" = "assemble" ]; then
  MV=""; [ -n "${MOVE:-}" ] && MV="--move"
  python -u $LOCAL_DIR/code/scripts/eci/atlas_v37_r4heat.py --key $KEY --assemble $MV
else
  python -u $LOCAL_DIR/code/scripts/eci/atlas_v37_r4heat.py --key $KEY --gpu "$GPU" --task ${TASK:-0} --ntasks ${NTASKS:-1} ${EXTRA:-}
fi
