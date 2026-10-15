#!/bin/bash
#
# End-to-end test of the 30-min ants code path on two v5 videos (already extended to 30 min, 9000 frames), before the
# v3 / v4 / v4_infested frames exist: an OLD layout (the two videos at 10 min, encoded the standard way:
# fg_background.py + fg_encode_all.py) and a NEW layout (the same videos at 30 min) built by fg_encode_extend.py
# (setup: links the old backgrounds as the first 10-min windows; fg_background.py: windows 2 and 3; encode: frames
# 3000-8999 in one DINOv2 pass for both SAEs; merge + verify: reused rows byte-equal, reused and encoded rows
# re-encoded through the standard path). Plus: foreground size of frames 3000-8999 with the per-window backgrounds vs
# with the first window's background only. Test dirs dataset/ants/eci30_t_{old,new} are deleted at the end; the
# verify.json files and the log stay in results/vision/eci_ants30/tests/e2e/.
#
#   sbatch scripts/eci/ants30_e2e_test.sh
#SBATCH --job-name=ants30_e2e
#SBATCH --output=results/vision/eci_ants30/logs/%x_%j.out
#SBATCH --time=01:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu150,gpu242,gpu230
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G

module load conda
conda activate crl
export PYTHONUNBUFFERED=1
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
OUT=results/vision/eci_ants30/tests/e2e
OLD=dataset/ants/eci30_t_old
NEW=dataset/ants/eci30_t_new
FG=matryoshka_btk_1024_k16_antsfg_s0
FULL=matryoshka_btk_1024_k16_antsfull_s0
rm -rf $OLD $NEW && mkdir -p $OUT $OLD/sae $NEW
trap 'rm -rf $OLD $NEW ${LOCAL_DIR:-/nonexistent}; kill %1 2>/dev/null || true' EXIT
nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader -l 60 > $OUT/gpu_${SLURM_JOB_ID}.csv &
for s in $FG $FULL; do ln -s $PWD/dataset/ants/eci/sae/$s $OLD/sae/$s; done
python - <<'PY'
import numpy as np, pandas as pd
e = pd.read_csv('data/ants/v5/experiment.csv', dtype={'observation_id': str})
e = e[e['valid'] == 1].head(2)
for d, n in (('dataset/ants/eci30_t_old', 3000), ('dataset/ants/eci30_t_new', 9000)):
    rows = [{'observation_id': o, 'frame_idx': i, 'fps': 5.0, 'frame_path': f'ants/v5/frames/full/{o}/frame_{i:06d}.jpg',
             'experiment': 'v5', 'T': 0} for o in e['observation_id'] for i in range(n)]  # v5 treatments are letters
    pd.DataFrame(rows).to_csv(f'{d}/annotations.csv', index=False)
    e.assign(experiment='v5', T=0)[['observation_id', 'experiment', 'T']].to_csv(f'{d}/experiment.csv', index=False)
    print(d, len(rows), 'rows', e['observation_id'].tolist())
PY
echo "=== old layout: backgrounds + standard codes"
export ANTS_ECI=eci30_t_old
python -u scripts/eci/fg_background.py --domain ants --num-workers 14
for s in $FG $FULL; do
    python -u scripts/eci/fg_encode_all.py --domain ants --sae $s --n-shards 1 --shard 0 --merge --num-workers 14
done
echo "=== new layout: setup, window backgrounds, encode (2 shards), merge + verify"
export ANTS_ECI=eci30_t_new
SA="--sae $FG --sae $FULL"
python -u scripts/eci/fg_encode_extend.py setup $SA --old-eci $OLD
ls $NEW/fg448/background
python -u scripts/eci/fg_background.py --domain ants --num-workers 14
for i in 0 1; do
    export LOCAL_DIR=/localhome/$USER/${SLURM_JOB_ID}_e2e_$i
    python -u scripts/eci/fg_encode_extend.py encode $SA --old-eci $OLD --shard $i --n-shards 2
    rm -rf $LOCAL_DIR
done
python -u scripts/eci/fg_encode_extend.py merge $SA --old-eci $OLD --n-shards 2 --n-check 32
for s in $FG $FULL; do cp $NEW/codes/$s/verify.json $OUT/verify_$s.json; done
echo "=== foreground size of frames 3000-8999: per-window backgrounds vs the first window's background only"
python - <<'PY'
import json, numpy as np, pandas as pd, torch
from pathlib import Path
from src.eci.foreground import FgBackgrounds, RULES, FrameDatasetFG, encode_batch, load_encoder_fg, obs_rows
new = Path('dataset/ants/eci30_t_new')
ann = new / 'annotations.csv'
ranges = obs_rows(ann)
one = new / 'bg_first_window_only'
one.mkdir(exist_ok=True)
for o in ranges:
    (one / f'{o}.npz').symlink_to((new / 'fg448/background' / f'{o}.npz').resolve())
dev = torch.device('cuda')
win = FgBackgrounds(new / 'fg448/background', ranges, RULES['ants'], dev)
single = FgBackgrounds(one, ranges, RULES['ants'], dev)
fp = pd.read_csv(ann)['frame_path'].values
rng = np.random.default_rng(0)
rows = np.sort(np.concatenate([rng.choice(np.arange(lo, hi), 300, replace=False) for lo, hi in ranges.values()]))
_, proc, model = load_encoder_fg(device=dev)
dl = torch.utils.data.DataLoader(FrameDatasetFG([f'dataset/{p}' for p in fp[rows]], proc, rows), batch_size=50, num_workers=8)
res = {k: [] for k in ('win', 'single', 'row')}
for pix, grey, r in dl:
    tok = encode_batch(model, pix, dev); g = grey.to(dev); r = r.numpy()
    res['win'].append(win.mask(tok, g, r)[0].sum(1).cpu().numpy())
    res['single'].append(single.mask(tok, g, r)[0].sum(1).cpu().numpy())
    res['row'].append(r)
w, s, r = (np.concatenate(res[k]) for k in ('win', 'single', 'row'))
off = r - np.array([ranges[o][0] for o in ranges])[np.searchsorted([v[0] for v in ranges.values()], r, side='right') - 1]
out = {}
for name, m in (('minutes 0-10', off < 3000), ('minutes 10-20', (off >= 3000) & (off < 6000)), ('minutes 20-30', off >= 6000)):
    out[name] = {'n_frames': int(m.sum()), 'n_fg_mean_windows': float(w[m].mean()), 'n_fg_mean_first_window_bg': float(s[m].mean()),
                 'identical_frac': float((w[m] == s[m]).mean())}
print(json.dumps(out, indent=1))
Path('results/vision/eci_ants30/tests/e2e/window_vs_single_bg.json').write_text(json.dumps(out, indent=1))
PY
echo "e2e done"
