#!/usr/bin/env python3
"""Is PPCI's answer a property of the behaviour, or of the model that produced it?

PPCI is the mean of the MODEL's within-pool phase differences, uncalibrated, so it claims sign and
pattern rather than magnitude. That claim is only worth anything if sign and pattern survive
changing the model. This measures it by swapping the deployed 3-fold ensemble for the single
accuracy leader and recomputing every cell.

    deployed   mean of xfit_f1/f2/f3, each trained on 16 pools, SSL-adapted encoder,
               plain 5.03 M head.                                    macro AP 0.382
    single     res448_k2_bit6_d4, trained on 20 pools, BitFit on 6 blocks of stock
               DINOv2, 0.52 M cross-attention head.                  macro AP 0.5409

THE POOLS ARE HELD FIXED, which is the whole point -- otherwise the model change is confounded
with a change of sample. BitFit-6 trained on 20 of the 24 annotated pools, so the only pools where
BOTH predictors are out-of-sample are the 48 unannotated ones plus BitFit-6's own 4 held-out
pools: 52 of 72. Every number below is on those 52.

Thresholds are each model's own event-level max-F1 point, measured off data it did not train on,
because a bout count is not defined without one. Occupancy needs none and is reported too.

    python scripts/mice_behavior/build_ppci_robustness.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from event_eval import postprocess, runs                                   # noqa: E402

FRAME = ROOT / 'results' / 'vision' / 'mice' / 'frame'
OUT = FRAME / '_figures'
FPS = 5.0
SINGLE = 'res448_k2_bit6_d4'
FOLDS = ('xfit_f1', 'xfit_f2', 'xfit_f3')
# Event-level max-F1, each measured on data the model in question did not train on.
THR = {'single': {'nt': 0.85, 'nn': 0.95}, 'deployed': {'nt': 0.90, 'nn': 0.87}}


def per_observation(which, exp, unann_obs, single_val):
    """Per-observation occupancy and bouts/min for one predictor, on the 52-pool set."""
    if which == 'single':
        z = {k: v.astype(np.float32) for k, v in
             np.load(FRAME / SINGLE / 'pred_dense_v1.npz', allow_pickle=True).items()
             if k in unann_obs}
        d = np.load(FRAME / SINGLE / 'val_probs.npz', allow_pickle=True)
        for o in set(d['obs']):                       # its own held-out pools
            z[o] = d['probs'][d['obs'] == o]
    else:
        parts = [{k: v.astype(np.float32) for k, v in
                  np.load(FRAME / t / 'pred_dense_v1.npz', allow_pickle=True).items()
                  if k in unann_obs} for t in FOLDS]
        z = {k: np.mean([p[k] for p in parts], axis=0) for k in parts[0]}
        pool_of = dict(zip(exp.observation_id, exp.pool))
        for t in FOLDS:                               # out-of-fold on the annotated side
            d = np.load(FRAME / t / 'val_probs.npz', allow_pickle=True)
            for o in set(d['obs']):
                if pool_of.get(o) in single_val:
                    z[o] = d['probs'][d['obs'] == o]
    thr = THR[which]
    rows = []
    for o, p in z.items():
        n = len(p); r = {'observation_id': o}
        for j, lab in enumerate(('nt', 'nn')):
            r[f'time_{lab}'] = float(p[:, j].mean() * 100)
            r[f'events_{lab}'] = len(runs(postprocess(p[:, j] >= thr[lab], 1, 1))) / (n / FPS / 60)
        rows.append(r)
    return pd.DataFrame(rows).merge(
        exp[['observation_id', 'pool', 'phase', 'odor']], on='observation_id')


def mean_diff(df, col, od, x, y):
    w = df[df.odor == od].pivot_table(index='pool', columns='phase', values=col)
    if x not in w or y not in w:
        return None, 0
    v = (w[y] - w[x]).dropna()
    return float(v.mean()), int(len(v))


def main():
    exp = pd.read_csv(ROOT / 'data' / 'mice' / 'v1' / 'experiment.csv')
    single_val = set(json.load(open(FRAME / SINGLE / 'config.json'))['val_pools'])
    unann_obs = set(exp.loc[exp.annotator.isna(), 'observation_id'])
    frames = {w: per_observation(w, exp, unann_obs, single_val) for w in ('deployed', 'single')}
    common = set(frames['deployed'].observation_id) & set(frames['single'].observation_id)
    for w in frames:
        frames[w] = frames[w][frames[w].observation_id.isin(common)]
    n_pools = int(frames['single'].pool.nunique())
    print(f'{len(common)} observations, {n_pools} pools, both predictors out-of-sample')

    cells, agree = [], {'events': 0, 'time': 0}
    for unit in ('events', 'time'):
        for lab in ('nn', 'nt'):
            for od, odn in (('F', 'fear'), ('S', 'social')):
                for x, y in (('H', 'O'), ('O', 'P')):
                    a, n = mean_diff(frames['deployed'], f'{unit}_{lab}', od, x, y)
                    b, _ = mean_diff(frames['single'], f'{unit}_{lab}', od, x, y)
                    if a is None or b is None:
                        continue
                    same = (a > 0) == (b > 0)
                    agree[unit] += bool(same)
                    cells.append({'unit': unit, 'behav': lab, 'odour': odn,
                                  'trans': f'{x}->{y}', 'deployed': round(a, 3),
                                  'single': round(b, 3), 'same_sign': bool(same), 'n_pools': n})
    for unit in ('events', 'time'):
        k = sum(1 for c in cells if c['unit'] == unit)
        print(f'  {unit:7s}: signs agree in {agree[unit]} of {k} cells')

    # calibration of each predictor against truth, on pools it did not train on
    cal = {}
    for w, tags in (('single', (SINGLE,)), ('deployed', FOLDS)):
        P = np.concatenate([np.load(FRAME / t / 'val_probs.npz', allow_pickle=True)['probs']
                            for t in tags])
        L = np.concatenate([np.load(FRAME / t / 'val_probs.npz', allow_pickle=True)['labels']
                            for t in tags])
        cal[w] = {lab: round(float(P[:, j].mean() / L[:, j].mean()), 2)
                  for j, lab in enumerate(('nt', 'nn'))}
        print(f'  {w:8s} predicted/true occupancy: '
              + ', '.join(f'{k} {v}x' for k, v in cal[w].items()))

    payload = {'meta': {'n_pools': n_pools, 'n_observations': len(common),
                        'single': SINGLE, 'deployed': list(FOLDS), 'thresholds': THR,
                        'calibration': cal,
                        'sign_agreement': {u: {'agree': agree[u],
                                               'of': sum(1 for c in cells if c['unit'] == u)}
                                           for u in ('events', 'time')}},
               'cells': cells}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(payload, open(OUT / 'ppci_robustness.json', 'w'), indent=1)
    print(f"\nwrote {OUT / 'ppci_robustness.json'}")


if __name__ == '__main__':
    main()
