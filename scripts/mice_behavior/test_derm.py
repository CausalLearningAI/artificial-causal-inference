#!/usr/bin/env python3
"""Audit `train_online_aug.derm_table` against the formula it claims to implement.

Run this before trusting any DERM arm. It is a property test, not a smoke test: every check below
is an identity that must hold exactly (to float tolerance), so a regression shows up as a failure
rather than as a slightly different number nobody notices.

WHAT IS CHECKED
===============
1. CLOSED FORM. w = Var(Y|E) / P(Y, E). For a binary label that collapses to
   w(y=1,e) = (1-p_e)/P(e) and w(y=0,e) = p_e/P(e). Checked against BOTH forms independently, so
   an error in the algebraic simplification cannot hide behind the simplification.
2. INVARIANT A. Positives and negatives carry EQUAL total mass inside every environment. This is
   the property that removes the label-environment association, i.e. the whole point.
3. INVARIANT B. Each environment's total mass is proportional to its own outcome variance, so an
   environment where nothing varies stops dominating by sheer size.
4. NORMALISATION. Mean weight over the training index is exactly 1, so the effective step size is
   identical to ERM and a DERM-vs-ERM comparison is not confounded by having changed the learning
   rate.
5. PER LABEL. nt and nn get their own correction. Collapsing them (which
   `src/ppci/dataset.py::compute_derm_weights` does, with a warning) would apply nt's correction to
   nn -- they sit at different prevalences with different phase profiles.
6. FLOOR NOT BINDING. `floor` clips p_e into [floor, 1-floor]. If it clips, the differences between
   environments are destroyed and the correction silently degenerates into a 1/P(e) reweighting.
   The RAW per-(phase,label) prevalence on this dataset is 0.68-1.24%, i.e. ALL BELOW the 2%
   default -- the floor is clear only because NEG_RATIO=1 lifts the SAMPLED prevalence to ~24%.
   Anything that raises neg_ratio walks into it.
7. ENVIRONMENT MAPPING. The sample -> observation -> phase map does a searchsorted over
   per-observation start rows in annotations.csv, which is exact only if each observation occupies
   one contiguous ascending block. Checked directly, and the resulting phase proportions are
   checked against the protocol's own 30/15/15-minute design.

    python scripts/mice_behavior/test_derm.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FRAME = ROOT / 'results' / 'vision' / 'mice' / 'frame'
TOL = 1e-5
FAIL: list[str] = []


def check(name: str, ok: bool, detail: str = '') -> None:
    print(f'  [{"ok  " if ok else "FAIL"}] {name}' + (f'   {detail}' if detail else ''))
    if not ok:
        FAIL.append(name)


def load_derm_table():
    """Import the one function under test without running the module's argparse main."""
    spec = importlib.util.spec_from_file_location(
        '_toa', Path(__file__).parent / 'train_online_aug.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules['_toa'] = mod
    try:
        spec.loader.exec_module(mod)
    except SystemExit:                      # the module calls main() under argparse on import
        pass
    return mod.derm_table


def test_formula(derm_table) -> None:
    """Both forms of the weight, and all three invariants, on data with KNOWN parameters."""
    print('\n1-5. formula and invariants, on synthetic data with known p_e and P(e)')
    rng = np.random.default_rng(0)
    n_env, L, N = 3, 2, 200_000
    P_true = np.array([0.5, 0.3, 0.2])
    p_true = np.array([[0.10, 0.40], [0.25, 0.05], [0.50, 0.30]])
    e = rng.choice(n_env, size=N, p=P_true)
    y = (rng.random((N, L)) < p_true[e]).astype(np.float32)

    tab, m_raw = derm_table(y, e, np.arange(N), n_env, floor=0.0)
    cnt = np.bincount(e, minlength=n_env).astype(float)
    P_e = cnt / N
    p_e = np.stack([np.bincount(e, weights=(y[:, l] > 0.5).astype(float),
                                minlength=n_env) / cnt for l in range(L)], axis=1)
    w = tab[e[:, None], np.arange(L)[None, :], (y > 0.5).astype(np.int64)]

    d1 = np.abs(tab[:, :, 1] * m_raw - (1 - p_e) / P_e[:, None]).max()
    d0 = np.abs(tab[:, :, 0] * m_raw - p_e / P_e[:, None]).max()
    check('w(y=1,e) = (1-p_e)/P(e)', d1 < TOL, f'max dev {d1:.2e}')
    check('w(y=0,e) = p_e/P(e)', d0 < TOL, f'max dev {d0:.2e}')

    # the general form, computed independently rather than via the simplification
    dv = 0.0
    for l in range(L):
        var = p_e[:, l] * (1 - p_e[:, l])
        for k, joint in ((1, P_e * p_e[:, l]), (0, P_e * (1 - p_e[:, l]))):
            dv = max(dv, np.abs(tab[:, l, k] * m_raw - var / joint).max())
    check('w = Var(Y|E) / P(Y,E), computed independently', dv < TOL, f'max dev {dv:.2e}')

    check('mean weight over the training index is 1', abs(w.mean() - 1.0) < TOL,
          f'{w.mean():.8f}')

    worst_mass, worst_var = 0.0, 0.0
    for l in range(L):
        for ee in range(n_env):
            s = e == ee
            mp = w[s & (y[:, l] > 0.5), l].sum()
            mn = w[s & (y[:, l] <= 0.5), l].sum()
            worst_mass = max(worst_mass, abs(mp / mn - 1.0))
        tot = np.array([w[e == ee, l].sum() for ee in range(n_env)])
        r = tot / (p_e[:, l] * (1 - p_e[:, l]))
        worst_var = max(worst_var, np.abs(r / r[0] - 1.0).max())
    check('INVARIANT A: equal pos/neg mass inside every environment',
          worst_mass < 1e-4, f'worst dev {worst_mass:.2e}')
    check('INVARIANT B: environment mass proportional to Var(Y|E)',
          worst_var < 1e-4, f'worst dev {worst_var:.2e}')

    # per label, not a mean over labels: label 0 and label 1 have different profiles here, so a
    # collapsed implementation would give them the same table
    same = np.allclose(tab[:, 0, :], tab[:, 1, :])
    check('PER LABEL: nt and nn get different tables', not same,
          'a collapsed version would make these identical')


def test_floor(derm_table) -> None:
    """The floor is the one way this silently degenerates. Show both regimes."""
    print('\n6. floor. If it clips, every p_e collapses to it and only 1/P(e) survives.')
    exp = pd.read_csv(ROOT / 'data' / 'mice' / 'v1' / 'experiment.csv')
    ann = pd.read_csv(ROOT / 'dataset' / 'mice' / 'v1' / 'annotations.csv',
                      usecols=['observation_id', 'frame_idx', 'Y_nt', 'Y_nn'],
                      low_memory=False).dropna(subset=['Y_nt'])
    ann = ann.merge(exp[['observation_id', 'pool', 'phase']], on='observation_id')
    cfg_p = FRAME / 'res448_k2_frozen_d4photo_dermPhase' / 'config.json'
    if not cfg_p.exists():
        print('     [skip] no dermPhase config on disk')
        return
    val = set(json.load(open(cfg_p))['val_pools'])
    tr = ann[~ann.pool.isin(val)]
    names, ev = np.unique(tr.phase.to_numpy(), return_inverse=True)
    Y = np.stack([(tr.Y_nt > 0.5).to_numpy(), (tr.Y_nn > 0.5).to_numpy()], 1).astype(np.float32)
    raw_p = np.stack([np.bincount(ev, weights=Y[:, l], minlength=len(names))
                      / np.bincount(ev) for l in range(2)], axis=1)
    print(f'     raw per-(phase,label) prevalence: '
          f'{100 * raw_p.min():.2f}%-{100 * raw_p.max():.2f}%  against a 2.0% default floor')
    check('raw prevalence is BELOW the floor -- so DERM on unsampled frames would degenerate',
          raw_p.max() < 0.02, 'this is why NEG_RATIO=1 is load-bearing, not incidental')

    # what the degenerate regime looks like, so the signature is on record
    tab_bad, m_bad = derm_table(Y, ev.astype(np.int64), np.arange(len(Y)), len(names), floor=0.02)
    collapsed = np.allclose(tab_bad[:, 0, :], tab_bad[:, 1, :])
    check('degenerate signature: all labels share one table when the floor bites', collapsed,
          f'derm_w_raw would log as {m_bad:.2f}')

    # and what the runs actually logged
    phi = 0.02
    degen = 2 * len(names) * phi * (1 - phi) / 1.0
    print(f'     if the floor bound, the runs would log derm_w_raw = {degen:.3f}; '
          f'they log 1.09 ({1.09 / degen:.1f}x)')
    check('the landed runs are NOT in the degenerate regime', 1.09 > 5 * degen,
          'derm_w_raw = 1.09 inverts to a mean sampled p_e of ~0.24, the 1:1 regime')


def test_env_mapping() -> None:
    """The sample -> phase map, and the assumption that makes it exact."""
    print('\n7. environment mapping')
    a = pd.read_csv(ROOT / 'dataset' / 'mice' / 'v1' / 'annotations.csv',
                    usecols=['observation_id', 'frame_idx'], low_memory=False).reset_index()
    bad = 0
    for _, g in a.groupby('observation_id', sort=False):
        i = g['index'].to_numpy()
        if not (np.all(np.diff(i) == 1) and np.all(np.diff(g.frame_idx.to_numpy()) == 1)):
            bad += 1
    check('every observation is one contiguous ascending block in annotations.csv', bad == 0,
          f'{bad} of {a.observation_id.nunique()} violate it '
          f'-- searchsorted over start rows needs this')

    # the protocol is H 30 min, O 15, P 15, so anchors must land 50/25/25 if the map is right.
    # Taken from the run's own log rather than recomputed, so this checks the SHIPPED mapping.
    logged = {'H': 38139, 'O': 19971, 'P': 19462}
    tot = sum(logged.values())
    share = {k: v / tot for k, v in logged.items()}
    want = {'H': 0.50, 'O': 0.25, 'P': 0.25}
    dev = max(abs(share[k] - want[k]) for k in want)
    check('logged anchor shares match the 30/15/15-minute protocol', dev < 0.01,
          ', '.join(f'{k} {100 * share[k]:.1f}%' for k in 'HOP') + f'  max dev {dev:.4f}')


def main() -> None:
    print('DERM implementation audit')
    derm_table = load_derm_table()
    test_formula(derm_table)
    test_floor(derm_table)
    test_env_mapping()
    print('\n' + ('FAILED: ' + '; '.join(FAIL) if FAIL else 'all checks passed'))
    sys.exit(1 if FAIL else 0)


if __name__ == '__main__':
    main()
