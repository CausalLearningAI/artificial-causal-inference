#!/bin/bash
#
# NN/NT round 1 job wrapper (scripts/eci/nnnt/nnnt.py). STEP = plan | prepass | train | encode | score | locality |
# selftest | cost; EXTRA = more arguments. Staging in /localhome/$USER/<job>_<task>_nnnt (removed on exit); GPU
# utilisation / memory logged every 60 s into the job log ([gpu] lines, parsed by the 'cost' step).
#   GPU: sbatch -p gpu --gres=gpu:1 --constraint="L40S|A100" --exclude=gpu240,gpu241,gpu242 \
#          --export=ALL,STEP=prepass scripts/eci/nnnt/nnnt.sh
#   encode: --array=0-N (SLURM_ARRAY_TASK_ID = the shard)
#SBATCH --job-name=nnnt
#SBATCH --output=/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference/results/vision/eci_nnnt/logs/%x_%A_%a.out
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
module load conda
conda activate crl
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
export LOCAL_DIR=/localhome/$USER/${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}_nnnt
mkdir -p $LOCAL_DIR
SMI=""
if nvidia-smi -L >/dev/null 2>&1; then
  nvidia-smi -L
  nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used --format=csv,noheader -l 60 | sed -u 's/^/[gpu] /' &
  SMI=$!
fi
trap 'rm -rf $LOCAL_DIR; [ -n "$SMI" ] && kill $SMI 2>/dev/null || true' EXIT
df -h /localhome | tail -1
echo "host $(hostname) job ${SLURM_JOB_ID:-} task ${SLURM_ARRAY_TASK_ID:-} step $STEP extra ${EXTRA:-}"
PY="python -u scripts/eci/nnnt/nnnt.py"
if [ "$STEP" = "selftest" ]; then
  # plan, a 40-burst prepass, 300-step SAEs for every arm, 1 video x 160 frames encoded to local disk, scorer check
  export NNNT_OUT=$LOCAL_DIR/out NNNT_BEE=$LOCAL_DIR/bee
  mkdir -p $NNNT_OUT
  cp results/vision/eci_nnnt/plan.json $NNNT_OUT/
  $PY prepass --limit 40
  cp $NNNT_OUT/prepass_selftest.json $NNNT_OUT/mask_sheet_selftest.jpg results/vision/eci_nnnt/ || true
  $PY train --selftest --steps 300
  cp $NNNT_OUT/train_selftest.json results/vision/eci_nnnt/ || true
  $PY encode --selftest --videos 0 --max-frames 160 --out-dir $LOCAL_DIR/enc_out
  python - <<EOF
import numpy as np, glob, json
f = glob.glob('$LOCAL_DIR/enc_out/*.npz')[0]
z = np.load(f)
print('keys', len(z.files), sorted(z.files)[:8])
print('k', z['k'][:5], z['k'][-1], 'ku', z['ku'], 'fps1_off', z['fps1_off'])
for k in ('nk_u64L', 'nk_r64L', 'nk_r64T', 'nk_r32T', 'nk_r64_onanimal'):
    print(k, z[k].shape, z[k].mean())
for k in ('A0_w512_s0|max', 'A0ref|max', 'A1_w512_s0|max', 'A2_w512_s0|top', 'A4_w1024_s0|max'):
    x = z[k].astype(np.float32); print(k, x.shape, 'frame L0', (x > 0).sum(1).mean(), 'max', x.max())
# A0ref vs the stored r4 codes of the same rows
r4 = np.load('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/r4/mice/codes/matryoshka_btk_512_k16_r4d2_s0/codes_max.npy', mmap_mode='r')
a = z['A0ref|max'].astype(np.float32); s = np.asarray(r4[z['ku']]).astype(np.float32)
print('A0ref vs stored r4: max abs diff', np.abs(a - s).max(), 'mean', np.abs(a - s).mean(), 'fire agree', ((a > 0) == (s > 0)).mean())
EOF
  exit 0
fi
if [ "$STEP" = "refs" ]; then  # scorer check (synthetic) + the reference codes re-scored (r4 stored, C1, deployed)
  python -u scripts/eci/nnnt/check_scorer.py
  $PY score --refs-only
  exit 0
fi
SH=""
if [ "$STEP" = "encode" ] && [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then SH="--shard $SLURM_ARRAY_TASK_ID"; fi
$PY $STEP $SH ${EXTRA:-}
