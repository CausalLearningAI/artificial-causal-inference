"""Is the phase SHORTCUT resolution-dependent? (It is not. That is the point.)

RESULT, 24 pools, leave-one-POOL-out, balanced accuracy, chance 0.500:

    thumbnail    4x4    8x8   16x16   32x32   64x64  128x128
    O vs H,P    0.808  0.857  0.854   0.859   0.851   0.847     <- saturated by 8 px a side
    behav       0.533  0.509  0.535   0.552   0.538   0.539     <- CHANCE at every size

Read the second row as a NULL, not as a resolution curve: a linear probe on grey pixels cannot
read behaviour at ANY size, so this file measures the shortcut only. The behaviour's resolution
dependence has to come from the encoder ablation instead (--pixel-source: macro AP 0.3139 at
112 px, 0.3996 at 224, ~0.42 at 448, still climbing).

Together those two facts are the argument: downsampling removes NONE of the shortcut and a lot of
the signal, so a poorer pixel budget strictly widens the gap between how cheap the shortcut is and
how expensive the honest signal is -- and DERM, which forbids the shortcut, pays that gap. Tested
directly by scripts/mice_behavior/ablate_pixel_derm.sh.

Note vs the 32x32 figure in derm.json (0.946): that one is 4 pools and adds an intensity
histogram. 0.859 here is 24 pools, pixels only -- the more honest number, same conclusion.

Original note: does downsampling hurt the SIGNAL more than the SHORTCUT?

Identical descriptor (grey thumbnail), identical leave-one-POOL-out linear probe, two targets:
  SHORTCUT  O vs {H,P}, on QUIET frames only (no scored behaviour within 5 s) -- the bag
  SIGNAL    behaviour-present vs quiet, within phase -- the thing we actually want read
Swept over thumbnail size. If the shortcut saturates at a size where the signal is still
climbing, downsampling shifts the cost/benefit toward using the shortcut, and DERM -- which
forbids it -- pays that gap.
"""
import io, sys
import numpy as np, pandas as pd
from pathlib import Path
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path('/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference')
sys.path.insert(0, str(ROOT/'scripts'/'mice_behavior'))
F = ROOT/'results'/'vision'/'mice'/'frame'
BIG, SIZES, DIST = 128, (4, 8, 16, 32, 64, 128), 25
N_PHASE_CELL, N_BEHAV_CELL = 30, 90

ix = np.load(ROOT/'dataset'/'mice'/'v1'/'jpegcache_k2.npz')
keys, offs = ix['all_needed'], ix['offsets']
blob = np.memmap(ROOT/'dataset'/'mice'/'v1'/'jpegcache_k2.bin', dtype=np.uint8, mode='r')
pos = {int(k): i for i, k in enumerate(keys)}
exp = pd.read_csv(ROOT/'data'/'mice'/'v1'/'experiment.csv')[
    ['observation_id', 'pool', 'phase', 'odor']]

frames = []
for tag in ('xfit_erm_f1', 'xfit_erm_f2', 'xfit_erm_f3'):        # 3 folds -> all 24 pools
    d = np.load(F/tag/'val_probs.npz', allow_pickle=True)
    frames.append(pd.DataFrame({'obs': d['obs'], 'gi': d['gi'],
                                'y': (d['labels'][:, 0] > 0.5) | (d['labels'][:, 1] > 0.5)}))
df = pd.concat(frames).sort_values(['obs', 'gi']).merge(
    exp, left_on='obs', right_on='observation_id')
dist = []
for _, g in df.groupby('obs', sort=False):
    y = g.y.to_numpy(); q = np.arange(len(y))
    dist.append(pd.Series(np.abs(q[:, None] - np.flatnonzero(y)[None, :]).min(1)
                          if y.any() else np.full(len(y), 10**6), index=g.index))
df['dist'] = pd.concat(dist)
rng = np.random.default_rng(0)


def take(g, n):
    return g.iloc[rng.choice(len(g), min(len(g), n), replace=False)]


quiet = df[(~df.y) & (df.dist >= DIST)]
s_phase = pd.concat([take(g, N_PHASE_CELL)
                     for _, g in quiet.groupby(['pool', 'phase', 'odor'], sort=True)])
s_beh = pd.concat([take(g, N_BEHAV_CELL) for _, g in df[df.y].groupby(['pool'], sort=True)]
                  + [take(g, N_BEHAV_CELL) for _, g in quiet.groupby(['pool'], sort=True)])
print(f'phase sample {len(s_phase)} frames, behaviour sample {len(s_beh)} '
      f'({int(s_beh.y.sum())} positive), {df.pool.nunique()} pools', flush=True)


def thumbs(samp):
    T, keep = [], []
    for k, gi in enumerate(samp.gi.to_numpy()):
        i = pos.get(int(gi))
        if i is None:
            continue
        im = Image.open(io.BytesIO(blob[offs[i]:offs[i+1]].tobytes())).convert('L')
        T.append(np.asarray(im.resize((BIG, BIG), Image.BILINEAR), dtype=np.float32)/255.0)
        keep.append(k)
    return np.stack(T), samp.iloc[keep].reset_index(drop=True)


Tp, sp = thumbs(s_phase)
Tb, sb = thumbs(s_beh)
print(f'decoded {len(Tp)} + {len(Tb)}', flush=True)


def probe(T, samp, target, S):
    """Leave-one-POOL-out balanced accuracy at thumbnail size S."""
    if S < BIG:
        k = BIG // S
        X = T.reshape(len(T), S, k, S, k).mean((2, 4))          # exact box downsample
    else:
        X = T
    X = X.reshape(len(X), -1)
    pool, y = samp['pool'].to_numpy(), target
    pred = np.empty(len(y), dtype=y.dtype)
    for p in np.unique(pool):
        te = pool == p
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=2000, C=0.05))
        m.fit(X[~te], y[~te]); pred[te] = m.predict(X[te])
    return balanced_accuracy_score(y, pred)


print(f"\n{'thumb':>6s} {'px/side':>8s} | {'SHORTCUT O vs H,P':>18s} | {'SIGNAL behav vs quiet':>22s}")
print(f"{'':>6s} {'':>8s} | {'chance 0.500':>18s} | {'chance 0.500':>22s}")
rows = []
for S in SIZES:
    a = probe(Tp, sp, (sp.phase.to_numpy() == 'O'), S)
    b = probe(Tb, sb, sb.y.to_numpy(), S)
    rows.append((S, a, b))
    print(f'{S:6d} {S:8d} | {a:18.3f} | {b:22.3f}', flush=True)
print()
base = rows[-1]
print('fraction of the 128px result already reached:')
for S, a, b in rows:
    print(f'  {S:3d}px  shortcut {100*(a-.5)/(base[1]-.5):5.1f}%   signal {100*(b-.5)/(base[2]-.5):5.1f}%')
