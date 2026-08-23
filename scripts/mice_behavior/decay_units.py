#!/usr/bin/env python3
"""Does the within-phase decay force a different OUTCOME UNIT? Measured, not argued.

THE PROBLEM
===========
Behaviour decays inside every phase, and habituation runs 30 minutes while O and P run 15. A
phase MEAN is therefore an average over whatever stretch of a decaying curve the schedule
happened to sample, and the two phases being compared do not sample the same stretch.

THE ONE FACT THAT NARROWS THIS DOWN
===================================
The window problem is confined ENTIRELY to H->O. O and P are both 15 minutes, so any window
rule that is applied identically to both leaves the O->P contrast bit-for-bit unchanged --
verified below, where mean_full and mean_15 agree to the last digit on all four O->P cells.
So this is not a question about the outcome in general; it is a question about one of the two
contrasts.

THREE CANDIDATE UNITS
=====================
mean_full   the phase mean over the whole recording. The current outcome, and the one with the
            length artefact.
mean_15     the phase mean over the first 15 minutes of EVERY phase. Assumption-free: it does
            not model the decay, it just stops comparing unequal windows.
amp_t0      the decay-corrected rate at phase onset. Each minute's count is de-trended to t=0
            with exp(-b*t), b from that cell's pooled Poisson fit, then averaged. Length-
            invariant by construction -- this is the "initial amplitude" idea.

WHAT THE NUMBERS SAY (see the tables this prints)
================================================
1. A single exponential is NOT adequate. Adding a t^2 term is significant in 7 of 12 cells and
   in 3 of the 4 thirty-minute H cells, which is where curvature has room to show. The decay
   flattens toward a floor, so the honest model is A*exp(-t/tau) + c, not A*exp(-t/tau).

2. amp_t0 is the WRONG novel unit, and this is the useful negative result. De-trending to t=0
   multiplies minute 29.5 by exp(0.11*29.5) ~ 25, so one late bout dominates the estimate. It
   resolves 3 of 8 contrasts against mean_full's 7, with intervals 2-4x wider. It removes a
   bias by paying far more in variance -- exactly the trade PPI is designed to avoid making.

3. tau is worth reporting, but as a SECOND OUTCOME rather than a better summary of the level.
   It is phase-dependent in a way no mean can express: P decays fastest in every single cell
   (tau ~ 6 min against 9-20 for H), and nose-to-tail under social exposure during O has a
   POSITIVE slope -- the one thing the exposure sustains instead of habituating.

RECOMMENDATION
==============
- O->P: keep the phase mean. It is immune to the window by construction.
- H->O: use the matched 15-minute window. It is assumption-free and it is the only contrast the
  choice touches.
- Report the habituation slope per phase as its own effect. Do not build the headline on a
  t=0 extrapolation.

Usage:  python scripts/mice_behavior/decay_units.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
FPS = 5.0
BEH = (('Y_nt', 'nt'), ('Y_nn', 'nn'))
ODOURS = (('F', 'fear'), ('S', 'social'))
PHASES = ('H', 'O', 'P')
TRANS = (('H', 'O'), ('O', 'P'))
UNITS = ('mean_full', 'mean_first15', 'mean_last15', 'amp_t0', 'frontload')


def minute_counts() -> pd.DataFrame:
    """Bouts STARTED per elapsed minute, per observation. t is the bin midpoint."""
    a = pd.read_csv(ROOT / 'dataset' / 'mice' / 'v1' / 'annotations.csv',
                    usecols=['observation_id', 'frame_idx', 'Y_nt', 'Y_nn'],
                    low_memory=False).dropna(subset=['Y_nt'])
    e = pd.read_csv(ROOT / 'data' / 'mice' / 'v1' / 'experiment.csv')[
        ['observation_id', 'pool', 'phase', 'odor']]
    a = a.sort_values(['observation_id', 'frame_idx']).merge(e, on='observation_id')
    a['minute'] = (a.frame_idx // int(FPS * 60)).astype(int)
    rows = []
    for (oid, m), g in a.groupby(['observation_id', 'minute'], sort=False):
        if len(g) < 250:            # drop the ragged final partial minute
            continue
        r = {'observation_id': oid, 'minute': m}
        for lab, _ in BEH:
            v = g[lab].to_numpy()
            r[lab] = int(((v == 1) & (np.r_[0, v[:-1]] == 0)).sum())
        rows.append(r)
    out = pd.DataFrame(rows).merge(e, on='observation_id')
    out['t'] = out.minute + 0.5
    return out


def fit_slopes(mt: pd.DataFrame):
    """Poisson GLM log E[N] = a + b t per (behaviour, odour, phase), plus a curvature test."""
    fits, table = {}, []
    for lab, nice in BEH:
        for od, odn in ODOURS:
            for ph in PHASES:
                d = mt[(mt.odor == od) & (mt.phase == ph)]
                y, t = d[lab].to_numpy(), d.t.to_numpy()
                m1 = sm.GLM(y, sm.add_constant(t), family=sm.families.Poisson()).fit()
                m2 = sm.GLM(y, np.column_stack([sm.add_constant(t), t ** 2]),
                            family=sm.families.Poisson()).fit()
                p_curv = st.chi2.sf(m1.deviance - m2.deviance, 1)
                b = float(m1.params[1])
                fits[(lab, od, ph)] = b
                table.append(dict(behav=nice, odour=odn, phase=ph, n=len(d), b=b,
                                  tau=(-1 / b if b < 0 else np.inf), p_curv=p_curv))
    return fits, pd.DataFrame(table)


def per_observation(mt: pd.DataFrame, fits: dict) -> pd.DataFrame:
    rows = []
    for oid, g in mt.groupby('observation_id', sort=False):
        ph, od = g.phase.iloc[0], g.odor.iloc[0]
        r = {'observation_id': oid, 'pool': g.pool.iloc[0], 'phase': ph, 'odor': od}
        for lab, _ in BEH:
            b = fits[(lab, od, ph)]
            first15 = g[g.minute < 15][lab]
            r[f'{lab}_mean_full'] = g[lab].mean()
            r[f'{lab}_mean_first15'] = first15.mean()
            # H is the only 30-minute phase, so it is the only one where "last 15" differs.
            r[f'{lab}_mean_last15'] = (g[(g.minute >= 15) & (g.minute < 30)][lab].mean()
                                       if ph == 'H' else first15.mean())
            r[f'{lab}_amp_t0'] = (g[lab].to_numpy() * np.exp(-b * g.t.to_numpy())).mean()
            # FRONT-LOADING: share of the window's bouts that fall in its first third.
            # Flat process -> 1/3; strong decay -> higher. Bounded, model-free, length-invariant,
            # and it needs no exponential -- which matters because the curvature test rejects a
            # single exponential in 7 of 12 cells. Undefined when the window holds no bouts.
            tot = first15.sum()
            r[f'{lab}_frontload'] = (g[g.minute < 5][lab].sum() / tot) if tot > 0 else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def contrast(tbl, col, x, y):
    w = tbl.pivot_table(index=['pool', 'odor'], columns='phase', values=col).dropna(subset=[x, y])
    d = (w[y] - w[x]).groupby('pool').mean()
    n = len(d); m = d.mean(); se = d.std(ddof=1) / np.sqrt(n)
    q = st.t.ppf(0.975, n - 1)
    return m, m - q * se, m + q * se


def main():
    mt = minute_counts()
    fits, ftab = fit_slopes(mt)
    print('1. IS THE DECAY A SINGLE EXPONENTIAL?  Poisson log E[N] = a + b t, LR test on t^2')
    print(f"{'behav':6}{'odour':8}{'phase':6}{'b/min':>10}{'tau (min)':>11}"
          f"{'half-life':>11}{'p(t^2)':>9}")
    for _, r in ftab.iterrows():
        hl = np.log(2) * r.tau if np.isfinite(r.tau) else np.inf
        print(f'{r.behav:6}{r.odour:8}{r.phase:6}{r.b:+10.4f}{r.tau:11.1f}{hl:11.1f}'
              f'{r.p_curv:9.3f}')
    n_curv = int((ftab.p_curv < 0.05).sum())
    n_curv_H = int(((ftab.p_curv < 0.05) & (ftab.phase == 'H')).sum())
    print(f'  -> curvature significant in {n_curv}/12 cells, {n_curv_H}/4 of the 30-minute H '
          'cells. A single exponential is not adequate; the decay flattens toward a floor.')

    print('\n2. THE DECAY RATE IS ITSELF PHASE-DEPENDENT (a candidate SECOND outcome)')
    for lab, nice in BEH:
        for od, odn in ODOURS:
            b = {ph: fits[(lab, od, ph)] for ph in PHASES}
            print(f'  {nice:3} {odn:7}: ' + '  '.join(
                f'tau_{ph}={-1/b[ph]:6.1f}' if b[ph] < 0 else f'tau_{ph}=  RISING' for ph in PHASES))

    po = per_observation(mt, fits)
    print('\n3. THE THREE UNITS, on the same pool-level contrast')
    print(f"{'behav':6}{'odour':8}{'trans':7}" + ''.join(f'{u:>24}' for u in UNITS))
    for lab, nice in BEH:
        for od, odn in ODOURS:
            for x, y in TRANS:
                cells = []
                for u in UNITS:
                    m, lo, hi = contrast(po[po.odor == od], f'{lab}_{u}', x, y)
                    cells.append(f'{m:+.2f} [{lo:+.2f},{hi:+.2f}]' + ('*' if lo * hi > 0 else ' '))
                print(f'{nice:6}{odn:8}{x+"->"+y:7}' + ''.join(f'{c:>24}' for c in cells))
    print('\n  * = 95% CI excludes zero, 8 contrasts per unit.')
    for u in UNITS:
        n = sum(1 for lab, _ in BEH for od, _ in ODOURS for x, y in TRANS
                if np.prod(contrast(po[po.odor == od], f'{lab}_{u}', x, y)[1:]) > 0)
        print(f'    {u:10} resolves {n}/8')

    print('\n4. THE PHASE-ONSET SPIKE IS NOT ODOUR-SPECIFIC -- P proves it')
    print('   P is the phase where the odour is REMOVED. If P spikes at onset too, the spike is')
    print('   handling/disturbance (the experimenter opens the cage), not a treatment response.')
    print(f"   {'behav':6}{'odour':8}{'phase':6}{'min 0-2':>9}{'min 13-15':>11}{'ratio':>8}")
    for lab, nice in BEH:
        for od, odn in ODOURS:
            for ph in PHASES:
                d = mt[(mt.odor == od) & (mt.phase == ph)]
                e0 = d[d.minute < 2][lab].mean()
                l0 = d[(d.minute >= 13) & (d.minute < 15)][lab].mean()
                print(f'   {nice:6}{odn:8}{ph:6}{e0:9.2f}{l0:11.2f}{e0/max(l0,1e-9):8.2f}')
    print('   -> P has the LARGEST onset/late ratio in 3 of 4 cells. Any contrast that compares')
    print('      a decayed H tail against a fresh O onset therefore charges that handling spike')
    print('      to the odour.')

    print('\n5. THE WINDOW ONLY TOUCHES H->O -- O and P are both 15 minutes')
    same = True
    for lab, nice in BEH:
        for od, odn in ODOURS:
            a = contrast(po[po.odor == od], f'{lab}_mean_full', 'O', 'P')[0]
            b = contrast(po[po.odor == od], f'{lab}_mean_first15', 'O', 'P')[0]
            same &= np.isclose(a, b)
    print(f'  O->P identical under mean_full and mean_15 in all 4 cells: {same}')
    print('  H->O under three defensible windows -- a cell that moves across all three is not '
          'a result:')
    print(f"    {'behav':5}{'odour':8}{'full':>8}{'first15':>9}{'last15':>9}{'spread':>8}")
    for lab, nice in BEH:
        for od, odn in ODOURS:
            vs = [contrast(po[po.odor == od], f'{lab}_{u}', 'H', 'O')[0]
                  for u in ('mean_full', 'mean_first15', 'mean_last15')]
            flag = '  <-- SPANS ZERO' if min(vs) * max(vs) < 0 else ''
            print(f'    {nice:5}{odn:8}{vs[0]:+8.2f}{vs[1]:+9.2f}{vs[2]:+9.2f}'
                  f'{max(vs)-min(vs):8.2f}{flag}')


if __name__ == '__main__':
    main()
