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
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
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
    a = pd.read_csv(ROOT / 'dataset' / 'mice' / 'v1' / 'annotations.csv',
                    usecols=['observation_id', 'frame_idx', 'Y_nt', 'Y_nn'],
                    low_memory=False).dropna(subset=['Y_nt'])
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
                series=series)


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
