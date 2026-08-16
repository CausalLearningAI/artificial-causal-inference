#!/usr/bin/env python3
"""PPI on v1 and prediction-powered transport to v2, for the phase-transition contrast.

v1 (PPI++). The estimand is a mean of WITHIN-POOL phase differences, so the unit is the pool and
the paired difference is the observation. With n labelled pools carrying both a true difference
D_Y and an out-of-fold predicted difference D_f, and N unlabelled pools carrying D_f only:

    theta_ppi = mean_unlabelled(lam * D_f) + mean_labelled(D_Y - lam * D_f)

Unbiased for ANY predictor: whatever lam and however wrong f is, the rectifier subtracts exactly
what the first term added. f only moves the VARIANCE. The power-tuned lam is

    lam* = Cov(D_Y, D_f) / (Var(D_f) * (1 + n/N))

which is also why a badly-calibrated model costs nothing here: lam absorbs the scale.

v2 (PPCI). There are no labels anywhere in v2, so there is no rectifier and the estimate is only
as good as the model transfers. We transport the v1 calibration slope
    beta = Cov(D_Y, D_f) / Var(D_f)          (fitted on v1's labelled pools)
and report beta * mean(D_f) on v2. The interval must carry BOTH the v2 sampling error and the
uncertainty in beta, which is done here by a paired bootstrap over v1 pools and v2 pools jointly.
This is an extrapolation, not an estimate with guarantees -- state it as such.

Usage:
    python scripts/mice_behavior/ppi_phase.py --labelled /tmp/xfit_pool_preds.csv \
        --unlabelled-v1 <csv> --v2 <csv>
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def pool_deltas(df, tcol, pcol):
    """-> (pools, D_Y, D_f) with one row per pool x odour x consecutive transition."""
    keys, dy, df_ = [], [], []
    for (pool, od), g in df.groupby(['pool', 'odor']):
        m = g.set_index('phase')
        for x, y in (('H', 'O'), ('O', 'P')):
            if x not in m.index or y not in m.index:
                continue
            keys.append(pool)
            dy.append((m.loc[y, tcol] - m.loc[x, tcol]) if tcol else np.nan)
            df_.append(m.loc[y, pcol] - m.loc[x, pcol])
    return np.array(keys), np.array(dy, float), np.array(df_, float)


def ppi(dy_lab, df_lab, df_unl):
    n, N = len(dy_lab), len(df_unl)
    v = df_lab.var(ddof=1)
    lam = (np.cov(dy_lab, df_lab)[0, 1] / (v * (1 + n / N))) if v > 0 else 0.0
    theta = (lam * df_unl).mean() + (dy_lab - lam * df_lab).mean()
    var = (dy_lab - lam * df_lab).var(ddof=1) / n + lam ** 2 * df_unl.var(ddof=1) / N
    return theta, np.sqrt(var), lam


def classical(dy_lab):
    return dy_lab.mean(), dy_lab.std(ddof=1) / np.sqrt(len(dy_lab))


def cluster_boot(keys, arrs, fn, reps=4000, seed=0):
    rng = np.random.default_rng(seed)
    pools = np.unique(keys); out = []
    for _ in range(reps):
        pick = rng.choice(pools, len(pools), replace=True)
        sel = np.concatenate([np.flatnonzero(keys == p) for p in pick])
        try:
            v = fn(*[a[sel] for a in arrs])
            if np.isfinite(v): out.append(v)
        except Exception:
            pass
    return np.percentile(out, [2.5, 97.5]) if out else (np.nan, np.nan)
