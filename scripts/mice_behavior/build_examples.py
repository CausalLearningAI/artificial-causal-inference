#!/usr/bin/env python3
"""Pick the qualitative example frames for the report's interactive error figure, and embed them.

WHAT THIS REPLACES
==================
Six pre-rendered PNG grids -- confusion_examples_{nt,nn}.png and confident_{nt,nn}_{v1,v2}.png --
each baked at one model, one behaviour, one cohort. Every axis a reader would want to move was
frozen at render time, and adding one more combination meant another million-pixel PNG in the page.
This writes ONE json the browser can slice: model x cohort x behaviour x annotated-or-not.

NO GPU PASS IS NEEDED, and that is the point of the two caches this reads:

    <run>/val_probs.npz        per-frame prob + LABEL + gi + obs, on the run's held-out pools.
                              `gi` indexes dataset/mice/v1/annotations.csv directly, which is
                              where the frame path comes from.
    <run>/pred_dense_{v}.npz   per-frame prob at stride 1 for EVERY observation of cohort v,
                              keyed by observation_id, index = frame_idx. No labels exist for
                              these pools -- that is the whole situation the figure illustrates.

BUCKETS
=======
annotated   TP / FP / FN / TN at the run's max-F1 operating point. Confident errors are the
            diagnostic ones: a high-p FP is a frame the model argues hard for and is wrong about,
            a low-p FN is a real bout it saw no evidence in at all. So TP/FP are drawn from the
            HIGH end and FN/TN from the LOW end -- each bucket shown where it is most sure.
unannotated confident positive / confident negative only. There is no ground truth, so no
            bucket here can be called correct or incorrect, and the figure must not imply it.

At most one frame per observation inside a bucket, so N columns are N independent cases rather
than N frames of one bout.

THE CROSS-FITTED ENTRY
======================
`xfit` is not a run directory. Its annotated side concatenates the three folds' val_probs, which
between them hold out all 24 annotated pools, so every labelled frame is scored by a model that
never saw its pool. Its unannotated side AVERAGES the three folds' per-frame predictions, matching
what build_estimates.py feeds the estimators. Runs with no pred_dense dumps get an annotated side
only, and the figure says so rather than showing an empty panel.

    python scripts/mice_behavior/build_examples.py
    python scripts/mice_behavior/build_examples.py --n 4 --px 140
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

FRAME = ROOT / 'results' / 'vision' / 'mice' / 'frame'
OUT = FRAME / '_figures'
FOLDS = ('xfit_f1', 'xfit_f2', 'xfit_f3')
LABELS = ('nt', 'nn')
BEHAV_NICE = {'nt': 'nose-to-tail', 'nn': 'nose-to-nose (mutual and directional together)'}

# Which models the figure offers. `folds` means the cross-fitted trio; otherwise a run directory.
MODELS = [
    {'key': 'xfit', 'name': 'cross-fitted deployment (3 folds)', 'folds': True,
     'note': 'What every estimate in this report rests on. Annotated frames are scored '
             'out-of-fold, so no frame is scored by a model that saw its pool; unannotated '
             'predictions are the mean of the three folds.'},
    {'key': 'res448_k2_bit6_d4', 'name': 'BitFit, 6 blocks — best macro AP',
     'note': 'Leads on frame accuracy but was never cross-fitted, so it has no predictions on '
             'the unannotated pools and its annotated frames come from the standing 4-pool split.'},
    {'key': 'res448_k2_frozen_d4photo_sslinit', 'name': 'SSL-adapted, frozen',
     'note': 'The encoder the deployment folds were built on, scored here on the standing '
             '4-pool split rather than out-of-fold.'},
]


def best_f1_threshold(y, p):
    """Frame-level max-F1 operating point. The figure must show the point it scores at."""
    best, bt = -1.0, 0.5
    for t in np.round(np.arange(0.05, 1.0, 0.05), 2):
        pr = p >= t
        tp = float((pr & (y == 1)).sum()); fp = float((pr & (y == 0)).sum())
        fn = float((~pr & (y == 1)).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        if f1 > best:
            best, bt = f1, float(t)
    return bt, best


# ------------------------------------------------------------------ frame paths and thumbnails
def annotated_paths():
    """Row index in annotations.csv -> frame path. `gi` in val_probs.npz is exactly that index."""
    a = pd.read_csv(ROOT / 'dataset' / 'mice' / 'v1' / 'annotations.csv',
                    usecols=['frame_path'], low_memory=False)
    return a['frame_path'].to_numpy()


def stem_map(version):
    """observation_id -> the frames/ subdirectory for that recording."""
    e = pd.read_csv(ROOT / 'data' / 'mice' / version / 'experiment.csv')
    return {r.observation_id: os.path.splitext(r.observation_file)[0] for r in e.itertuples()}


# The rig is fixed and the outer border of every frame is chamber wall, not cage. Trimming a
# symmetric 6% buys ~1.14x magnification for free. Measured before choosing it: the bedding spans
# about x 0.13-0.84 and y 0.14-0.86 of the frame across both cohorts, so this cannot clip an
# animal. It is NOT a per-animal crop -- that needs the tracking this report is waiting on.
CROP = 0.06


def thumb(path: Path, px: int, q: int) -> str | None:
    try:
        with Image.open(path) as im:
            im = im.convert('L')
            w, h = im.size
            im = im.crop((round(CROP * w), round(CROP * h),
                          round((1 - CROP) * w), round((1 - CROP) * h)))
            im = im.resize((px, px), Image.LANCZOS)
            b = io.BytesIO(); im.save(b, 'JPEG', quality=q, optimize=True)
        return 'data:image/jpeg;base64,' + base64.b64encode(b.getvalue()).decode()
    except Exception as e:                     # a missing frame must not kill the build
        print(f'    [skip] {path}: {e.__class__.__name__}')
        return None


def pick(order, obs, n):
    """Walk an ordering, taking at most one frame per observation -> n independent cases."""
    out, seen = [], set()
    for i in order:
        if obs[i] in seen:
            continue
        seen.add(obs[i]); out.append(int(i))
        if len(out) >= n:
            break
    return out


# ------------------------------------------------------------------------- the annotated side
def annotated(model, paths, n, px, q):
    tags = FOLDS if model.get('folds') else (model['key'],)
    parts = []
    for t in tags:
        f = FRAME / t / 'val_probs.npz'
        if not f.exists():
            return None
        d = np.load(f, allow_pickle=True)
        parts.append((d['probs'], d['labels'], d['gi'], d['obs']))
    probs = np.concatenate([p[0] for p in parts])
    labs = np.concatenate([p[1] for p in parts])
    gi = np.concatenate([p[2] for p in parts])
    obs = np.concatenate([p[3] for p in parts])
    out = {}
    for li, lab in enumerate(LABELS):
        y, p = labs[:, li].astype(int), probs[:, li]
        thr, f1 = best_f1_threshold(y, p)
        pr = p >= thr
        idx = {'TP': np.flatnonzero((y == 1) & pr), 'FP': np.flatnonzero((y == 0) & pr),
               'FN': np.flatnonzero((y == 1) & ~pr), 'TN': np.flatnonzero((y == 0) & ~pr)}
        buckets = {}
        for b, pool in idx.items():
            if not len(pool):
                buckets[b] = {'n': 0, 'share': 0.0, 'items': []}
                continue
            # TP/FP are most telling at high p, FN/TN at low p
            hi = b in ('TP', 'FP')
            order = pool[np.argsort(-p[pool] if hi else p[pool])]
            items = []
            for i in pick(order, obs, n):
                u = thumb(ROOT / 'dataset' / paths[gi[i]], px, q)
                if u:
                    items.append({'p': round(float(p[i]), 3), 'obs': str(obs[i]),
                                  'frame': int(gi[i]), 'img': u})
            buckets[b] = {'n': int(len(pool)), 'share': round(100 * len(pool) / len(y), 2),
                          'items': items}
        out[lab] = {'threshold': round(thr, 2), 'f1': round(f1, 3),
                    'n_frames': int(len(y)), 'prevalence': round(100 * float(y.mean()), 2),
                    'buckets': buckets}
        print(f'    annotated {lab}: thr {thr:.2f}, frame-F1 {f1:.3f}, '
              + ', '.join(f'{b} {buckets[b]["n"]:,}' for b in ('TP', 'FP', 'FN', 'TN')))
    return out


# ----------------------------------------------------------------------- the unannotated side
def unannotated(model, version, labelled_pools, n, px, q):
    """Confident positives and negatives on pools with NO ground truth. No bucket is 'correct'."""
    tags = FOLDS if model.get('folds') else (model['key'],)
    per_fold = []
    for t in tags:
        f = FRAME / t / f'pred_dense_{version}.npz'
        if not f.exists():
            return None
        per_fold.append(np.load(f, allow_pickle=True))
    keys = [k for k in per_fold[0].files if all(k in d for d in per_fold)]
    stems = stem_map(version)
    e = pd.read_csv(ROOT / 'data' / 'mice' / version / 'experiment.csv')
    pool_of = dict(zip(e.observation_id, e.pool))
    # v1's dense dump covers all 72 pools; only the 48 with no labels belong in this panel
    keys = [k for k in keys if pool_of.get(k) not in labelled_pools]
    if not keys:
        return None
    P, OBS, FI = [], [], []
    for k in keys:
        # cross-prediction: the same average the estimators are fed
        a = np.mean([d[k].astype(np.float32) for d in per_fold], axis=0)
        P.append(a); OBS += [k] * len(a); FI += list(range(len(a)))
    P = np.concatenate(P); OBS = np.array(OBS); FI = np.array(FI)
    out = {}
    for li, lab in enumerate(LABELS):
        p = P[:, li]
        buckets = {}
        for b, hi in (('POS', True), ('NEG', False)):
            order = np.argsort(-p if hi else p)
            items = []
            for i in pick(order, OBS, n):
                path = (ROOT / 'dataset' / 'mice' / version / 'frames' / 'full'
                        / stems[OBS[i]] / f'frame_{FI[i]:06d}.jpg')
                u = thumb(path, px, q)
                if u:
                    items.append({'p': round(float(p[i]), 3), 'obs': str(OBS[i]),
                                  'frame': int(FI[i]), 'img': u})
            buckets[b] = {'n': None, 'share': None, 'items': items}
        out[lab] = {'n_frames': int(len(p)), 'n_pools': len(set(pool_of[k] for k in keys)),
                    'n_obs': len(keys), 'mean_p': round(100 * float(p.mean()), 2),
                    'buckets': buckets}
        print(f'    unannotated {version} {lab}: {len(keys)} observations, '
              f'mean p {100 * p.mean():.2f}%')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=3, help='examples per bucket')
    ap.add_argument('--px', type=int, default=200, help='thumbnail edge in px')
    ap.add_argument('--q', type=int, default=72, help='JPEG quality')
    a = ap.parse_args()

    paths = annotated_paths()
    # annotations.csv carries a row for EVERY v1 frame, labelled or not (Y_* is NaN where no
    # annotator looked), so its observation_id list is all 72 pools and cannot be used to find
    # the annotated ones. The annotation FILE existing on disk is the actual record.
    e1 = pd.read_csv(ROOT / 'data' / 'mice' / 'v1' / 'experiment.csv')
    ann_dir = ROOT / 'data' / 'mice' / 'v1' / 'annotations'
    has = e1.annotation_file.apply(lambda f: isinstance(f, str) and (ann_dir / f).exists())
    labelled_pools = set(e1[has].pool)
    print(f'{len(labelled_pools)} annotated pools on v1, '
          f'{e1.pool.nunique() - len(labelled_pools)} without labels')

    models = []
    for m in MODELS:
        print(f'\n{m["name"]}')
        entry = {k: m[k] for k in ('key', 'name', 'note')}
        entry['annotated'] = annotated(m, paths, a.n, a.px, a.q)
        entry['unannotated'] = {}
        for v in ('v1', 'v2'):
            u = unannotated(m, v, labelled_pools, a.n, a.px, a.q)
            if u:
                entry['unannotated'][v] = u
        if entry['annotated'] is None and not entry['unannotated']:
            print('  [skip] no cached predictions'); continue
        models.append(entry)

    payload = {'meta': {'behaviours': BEHAV_NICE, 'n_per_bucket': a.n, 'px': a.px,
                        'buckets_annotated': ['TP', 'FP', 'FN', 'TN'],
                        'buckets_unannotated': ['POS', 'NEG']},
               'models': models}
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / 'examples.json'
    json.dump(payload, open(p, 'w'), separators=(',', ':'))
    print(f'\nwrote {p}  ({p.stat().st_size / 1e6:.2f} MB, {len(models)} models)')


if __name__ == '__main__':
    main()
