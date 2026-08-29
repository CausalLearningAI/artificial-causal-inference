#!/usr/bin/env python3
"""Per-minute activity across the WHOLE protocol, for the report's interactive decay figure.

The static figure this replaces drew four small panels on a 0-30 minute axis, one per
behaviour x exposure, with the three phases overplotted on a shared elapsed-minute axis. That
layout hides the two things the section is actually about: that the phases are consecutive, and
that habituation runs twice as long as the other two. Laying the six recordings end to end on a
single 120-minute axis -- (30 + 15 + 15) x 2 sessions -- shows both directly.

The axis is a LAYOUT, not one continuous recording: the six videos are separate, minutes apart,
and each phase segment restarts at its own elapsed zero. Segments are therefore drawn as
separate polylines with a rule at every boundary, never joined across the seam.

Session order is the protocol order -- social session first, then fear -- which is the order
19 of the 24 annotated pools actually ran in.

Two units, the same pair the rest of the report reasons about:
    bouts    bouts STARTED in the minute        (event onsets; the headline outcome)
    seconds  seconds spent in the behaviour     (occupancy x 60)

Everything is aggregated over POOLS, the independent unit -- 24 of them -- and every interval
is a 95% bootstrap over pools with a fixed seed, matching story_figures._minute_ci exactly.

    python scripts/mice_behavior/build_decay.py
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

from src.mice_behavior.truth import read_truth                              # noqa: E402
OUT = ROOT / 'results' / 'vision' / 'mice' / 'frame' / '_figures' / 'decay.json'
FPS = 5.0
BEH = (('Y_nt', 'nt', 'nose-to-tail'), ('Y_nn', 'nn', 'nose-to-nose'))
# protocol order: the social session runs first
SESSIONS = (('S', 'social exposure'), ('F', 'fear exposure'))
PHASES = (('H', 'habituation', 30), ('O', 'exposure', 15), ('P', 'post', 15))
UNITS = (('bouts', 'bouts / min'), ('seconds', 'seconds / min'))
REPS, SEED = 2000, 0


def minute_table() -> pd.DataFrame:
    """One row per (observation, elapsed minute): bouts started, and seconds in behaviour."""
    a = read_truth()
    e = pd.read_csv(ROOT / 'data' / 'mice' / 'v1' / 'experiment.csv')[
        ['observation_id', 'pool', 'phase', 'odor']]
    a = a.sort_values(['observation_id', 'frame_idx']).merge(e, on='observation_id')
    a['minute'] = (a.frame_idx // int(FPS * 60)).astype(int)
    rows = []
    for (oid, m), g in a.groupby(['observation_id', 'minute'], sort=False):
        if len(g) < 250:                      # drop a ragged final partial minute
            continue
        r = {'observation_id': oid, 'minute': m}
        for lab, _, _ in BEH:
            v = g[lab].to_numpy()
            # a bout that is already running at the bin edge belongs to the minute it STARTED in
            r[f'{lab}_bouts'] = int(((v == 1) & (np.r_[0, v[:-1]] == 0)).sum())
            r[f'{lab}_seconds'] = float(v.mean() * 60.0)
        rows.append(r)
    return pd.DataFrame(rows).merge(e, on='observation_id')


def boot_ci(x: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and 95% interval over the FIRST axis (pools), resampled with replacement."""
    idx = rng.integers(0, len(x), size=(REPS, len(x)))
    with np.errstate(invalid='ignore'):
        boots = np.nanmean(x[idx], axis=1)
        return (np.nanmean(x, axis=0),
                np.nanpercentile(boots, 2.5, axis=0), np.nanpercentile(boots, 97.5, axis=0))


def build() -> dict:
    mt = minute_table()
    pools = sorted(mt.pool.unique())
    segs, t = [], 0.0
    for od, odn in SESSIONS:
        for ph, phn, dur in PHASES:
            segs.append(dict(odour=od, odour_label=odn, phase=ph, phase_label=phn,
                             t0=t, t1=t + dur, dur=dur))
            t += dur

    series = {}
    for unit, _ in UNITS:
        series[unit] = {}
        for lab, key, _ in BEH:
            rng = np.random.default_rng(SEED)
            pts, bars = [], []
            for s in segs:
                d = mt[(mt.odor == s['odour']) & (mt.phase == s['phase'])
                       & (mt.minute < s['dur'])]
                # pools x minutes; a pool missing a minute stays NaN and is skipped by nanmean
                piv = (d.pivot_table(index='pool', columns='minute', values=f'{lab}_{unit}')
                       .reindex(index=pools).reindex(columns=range(s['dur'])))
                x = piv.to_numpy(float)
                mu, lo, hi = boot_ci(x, rng)
                pts.append(dict(t=[s['t0'] + m + 0.5 for m in range(s['dur'])],
                                mean=[round(float(v), 4) for v in mu],
                                lo=[round(float(v), 4) for v in lo],
                                hi=[round(float(v), 4) for v in hi]))
                # the bar is the mean OF THE PLOTTED CURVE: pool means over the segment, then
                # the same bootstrap over pools. It is the phase outcome the report estimates on.
                pm = np.nanmean(x, axis=1)
                bmu, blo, bhi = boot_ci(pm[:, None], rng)
                bars.append(dict(mean=round(float(bmu[0]), 4), lo=round(float(blo[0]), 4),
                                 hi=round(float(bhi[0]), 4),
                                 n=int(np.isfinite(pm).sum())))
            series[unit][key] = dict(seg=pts, bar=bars)

    return dict(meta=dict(n_pools=len(pools), reps=REPS, seed=SEED, fps=FPS,
                          total_min=int(sum(s['dur'] for s in segs)),
                          units=[dict(key=k, label=l) for k, l in UNITS],
                          behav=[dict(key=k, label=n) for _, k, n in BEH],
                          segs=segs),
                series=series, facts=facts())


# --------------------------------------------------------------------------- the section-02 facts
# WHY THESE LIVE IN THE PAYLOAD. Sections 01 and 02 quoted six groups of numbers about the human
# labels -- frame prevalence, per-phase half-lives, the H->O window table, the phase-onset spike,
# the wild-type negative control -- as plain strings typed into the prose. Every one of them went
# stale when the nose-to-nose truth was corrected to the directed-pair union, and nothing failed.
# They are computed here, from the same `read_truth()` the figure uses, so the prose reads them.
#
# The estimators are `decay_units.py`'s, imported rather than re-implemented: that script is where
# each choice is argued, and two copies of a Poisson fit is how the two drift apart.


def _facts_prevalence() -> dict:
    """Share of annotated frames carrying each behaviour, and the union -- 04.1's imbalance."""
    a = read_truth()
    nt, nn = a.Y_nt > 0.5, a.Y_nn > 0.5
    n = len(a)
    return {'n_frames': int(n), 'n_obs': int(a.observation_id.nunique()),
            'nt': round(float(nt.mean()), 6), 'nn': round(float(nn.mean()), 6),
            'any': round(float((nt | nn).mean()), 6),
            'neg': round(float(1 - (nt | nn).mean()), 6)}


def _facts_wt() -> dict:
    """The three wild-type strata: do three lines' unmutated animals behave alike?

    Two questions, deliberately at different units, because that is what each one is about.
    LEVEL is a property of a pool, so the unit is the pool (2 per line). The ESTIMAND is measured
    once per pool AND exposure, so the unit is the pool x odour cell (4 per line). Rates are over
    the whole recording -- this is the raw-level check, before 02's matched window.
    """
    import numpy as _np
    from scipy import stats as _st

    a = read_truth()
    e = pd.read_csv(ROOT / 'data' / 'mice' / 'v1' / 'experiment.csv')[
        ['observation_id', 'pool', 'phase', 'odor', 'line', 'genotype']]
    a = a.sort_values(['observation_id', 'frame_idx']).merge(e, on='observation_id')
    rows = []
    for oid, g in a.groupby('observation_id', sort=False):
        r = {'observation_id': oid}
        mins = len(g) / (FPS * 60)
        for lab, _, _ in BEH:
            v = g[lab].to_numpy()
            r[lab] = float(((v == 1) & (_np.r_[0, v[:-1]] == 0)).sum() / mins)
        rows.append(r)
    obs = pd.DataFrame(rows).merge(e, on='observation_id')
    wt = obs[obs.genotype == 'wt']
    out = {'n_pools': int(wt.pool.nunique()), 'lines': sorted(wt.line.unique()), 'behav': {}}
    for lab, key, _ in BEH:
        lv = wt.groupby(['pool', 'line'])[lab].mean().reset_index()
        gl = [g[lab].to_numpy() for _, g in lv.groupby('line')]
        w = wt.pivot_table(index=['pool', 'line', 'odor'], columns='phase', values=lab).reset_index()
        w['d'] = w['O'] - w['H']
        gd = [g.d.to_numpy() for _, g in w.groupby('line')]
        out['behav'][key] = {
            'level': {'means': [round(float(x.mean()), 3) for x in gl],
                      'p': round(float(_st.f_oneway(*gl).pvalue), 4),
                      'n_per_line': [int(len(x)) for x in gl]},
            'ho': {'means': [round(float(x.mean()), 3) for x in gd],
                   'p': round(float(_st.f_oneway(*gd).pvalue), 4),
                   'n_per_line': [int(len(x)) for x in gd]}}
    return out


def facts() -> dict:
    """Everything sections 01, 02 and 04.1 state about the labels, measured rather than typed."""
    import numpy as _np
    from decay_units import (BEH as _B, ODOURS, PHASES, TRANS, contrast, fit_slopes,
                             minute_counts, per_observation)

    mt = minute_counts()
    fits, ftab = fit_slopes(mt)
    po = per_observation(mt, fits)
    # Is a single exponential adequate? LR test on a t^2 term, per cell. 02b cites the count to
    # justify not fitting a slope, and it moved when the nose-to-nose truth was corrected.
    curv = {'n_sig': int((ftab.p_curv < 0.05).sum()), 'n_cells': int(len(ftab)),
            'n_sig_H': int(((ftab.p_curv < 0.05) & (ftab.phase == 'H')).sum()),
            'n_cells_H': int((ftab.phase == 'H').sum())}

    tau, onset, window = {}, {}, {}
    for lab, key in _B:
        tau[key], onset[key], window[key] = {}, {}, {}
        for od, odn in ODOURS:
            b = {ph: float(fits[(lab, od, ph)]) for ph in PHASES}
            tau[key][odn] = {ph: {'b': round(b[ph], 4),
                                  'tau': round(-1 / b[ph], 2) if b[ph] < 0 else None,
                                  'half_life': round(-_np.log(2) / b[ph], 2) if b[ph] < 0 else None}
                             for ph in PHASES}
            onset[key][odn] = {}
            for ph in PHASES:
                d = mt[(mt.odor == od) & (mt.phase == ph)]
                e0 = float(d[d.minute < 2][lab].mean())
                l0 = float(d[(d.minute >= 13) & (d.minute < 15)][lab].mean())
                onset[key][odn][ph] = {'early': round(e0, 3), 'late': round(l0, 3),
                                       'ratio': round(e0 / max(l0, 1e-9), 2)}
            vs = [float(contrast(po[po.odor == od], f'{lab}_{u}', 'H', 'O')[0])
                  for u in ('mean_full', 'mean_first15', 'mean_last15')]
            window[key][odn] = {'full': round(vs[0], 2), 'first15': round(vs[1], 2),
                                'last15': round(vs[2], 2), 'spread': round(max(vs) - min(vs), 2),
                                'spans_zero': bool(min(vs) * max(vs) < 0)}
    # O->P is immune to the window rule by construction -- both phases are 15 minutes. Asserted
    # here rather than in the prose, so the claim cannot outlive the schedule it describes.
    op_same = all(_np.isclose(contrast(po[po.odor == od], f'{lab}_mean_full', 'O', 'P')[0],
                              contrast(po[po.odor == od], f'{lab}_mean_first15', 'O', 'P')[0])
                  for lab, _ in _B for od, _ in ODOURS)
    return {'prevalence': _facts_prevalence(), 'tau': tau, 'onset': onset, 'window': window,
            'curvature': curv, 'op_window_invariant': bool(op_same), 'wt': _facts_wt(),
            'n_trans': len(TRANS)}



if __name__ == '__main__':
    out = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, separators=(',', ':')))
    print(f'wrote {OUT}  ({OUT.stat().st_size/1024:.1f} KB, {out["meta"]["n_pools"]} pools, '
          f'{out["meta"]["total_min"]} min)')
    for u, _ in UNITS:
        for _, k, n in BEH:
            b = out['series'][u][k]['bar']
            print(f"  {u:8} {n:14} " + '  '.join(
                f"{s['odour']}{s['phase']}={x['mean']:.2f}" for s, x in zip(out['meta']['segs'], b)))
