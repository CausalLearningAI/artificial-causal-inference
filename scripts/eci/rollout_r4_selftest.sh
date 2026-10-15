#!/bin/bash
# Self-test of the R4 pipeline end to end on a few bursts / frames, everything under the job's local dir (R4_OUT /
# R4_BEE): prepass (--limit bursts) -> train (short) -> encode (one held-out video, first frames) -> merge-free checks.
#   sbatch -p debug_gpu --gres=gpu:1 --cpus-per-task=6 --mem=32G --time=00:50:00 --export=ALL,DOMAIN=ants_v2 scripts/eci/rollout_r4_selftest.sh
#SBATCH --job-name=r4_selftest
#SBATCH --output=results/vision/eci_rollout_r4/logs/%x_%j.out
module load conda
conda activate crl
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error R4_TEST=1 R4_AMP=${R4_AMP:-fp16}
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
L=/localhome/$USER/${SLURM_JOB_ID}_r4st
export LOCAL_DIR=$L/stage R4_OUT=$L/out R4_BEE=$L/bee
mkdir -p $R4_OUT/$DOMAIN $LOCAL_DIR
trap 'rm -rf $L' EXIT
nvidia-smi -L
cp results/vision/eci_rollout_r4/$DOMAIN/plan.json $R4_OUT/$DOMAIN/
python -u scripts/eci/rollout_r4.py prepass --domain $DOMAIN --limit ${LIMIT:-60}
rm -rf $LOCAL_DIR; mkdir -p $LOCAL_DIR
python -u scripts/eci/rollout_r4.py train --domain $DOMAIN --steps 300
V=$(python -c "import json;P=json.load(open('$R4_OUT/$DOMAIN/plan.json'));print([i for i,v in enumerate(P['videos']) if v['role']=='eval'][0])")
python -u scripts/eci/rollout_r4.py encode --domain $DOMAIN --videos $V --max-frames ${MAXF:-400}
python - <<PY
import numpy as np, glob
for f in glob.glob('$R4_BEE/$DOMAIN/videos/*.npz'):
    z = np.load(f)
    print(f, {k: z[k].shape for k in z.files})
    for v in ('d2', 'd0'):
        m = z[f'max_{v}'].astype(np.float32)
        a = z[f'arg_{v}']
        print(v, 'frame L0', (m > 0).sum(1).mean(), 'token L0', z['l0'] / np.maximum(z['ntok'], 1), 'neurons ever firing', (m > 0).any(0).sum(),
              'argmax valid where firing', bool(((a >= 0) == (m > 0)).all()), 'n_fg mean', z['n_fg'].mean(), 'n_fg32 mean', z['n_fg32'].mean())
PY
cp $R4_OUT/$DOMAIN/prepass.json results/vision/eci_rollout_r4/$DOMAIN/selftest_prepass.json
cp $R4_OUT/$DOMAIN/mask_sheet.jpg results/vision/eci_rollout_r4/$DOMAIN/selftest_mask_sheet.jpg 2>/dev/null || true
echo SELFTEST_OK
