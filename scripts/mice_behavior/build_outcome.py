#!/usr/bin/env python3
"""Every number the report's OUTCOME section quotes, computed from the labels. One JSON out.

WHY THIS SCRIPT EXISTS
======================
The outcome section used to carry six hand-entered numbers. Two of them did not survive being
recomputed (see below), which is the whole argument for not hand-entering numbers.

WHAT IT MEASURES, AND WHY EACH ONE IS HERE
==========================================
A behaviour is a continuous stream. Turning it into a number needs three separate decisions, none
of them given by the biology:

    what counts as ONE event   a bout = an uninterrupted run of annotated frames, at 5 fps
    what the DENOMINATOR is    per minute of recording -- which depends on how long you watch
    what you MEASURE           how often it starts / how much time it fills / how long each lasts

The three candidate units are exactly the third decision:

    counts       bouts per minute        how often the behaviour is initiated
    occupancy    % of frames in it       how much of the recording it fills
    duration     mean bout length        how long one bout lasts

occupancy is close to counts x duration, so it is not a third independent option so much as the
product of the other two, and it inherits both of their noise sources.

For each unit this prints:
  resolves      how many of the 8 exposure x transition x behaviour contrasts have a 95% CI
                excluding zero. REPORTED, BUT NOT AN ARGUMENT -- see the caveat below.
  cv            mean within-cell coefficient of variation over the 6 phase x exposure cells:
                how noisy the measurement is, independent of any effect
  one_frame     share of bouts lasting a single 5 fps frame: whether the unit is measurable
  tail          share of total behaviour time carried by the longest 10% of bouts
  r_delta       correlation between true and cross-fitted-predicted WITHIN-POOL phase
                differences -- the quantity PPI's variance reduction depends on
  bias_spread   max/min of the predicted/true ratio across the three phases: how much of the
                model's error moves WITH the treatment

It also emits `dist`, the two DISTRIBUTIONS the section argues from -- events per recording and
bout length -- binned so the figure can show their shape instead of quoting a percentile from it.

THE CAVEAT THAT HAS TO TRAVEL WITH `resolves`
=============================================
Picking the outcome that yields the most rejections of the null is selection on significance. It
cannot be evidence for that outcome. It is printed because it is the thing a reader will ask for,
and it is labelled so it is not mistaken for a reason.

TWO CLAIMS THAT DIED HERE
=========================
1. "counts halve the treatment-linked model bias." They do not. The predicted/true ratio moves
   by the same FACTOR across phases on both units (about 1.25x either way). The old claim
   compared absolute ranges on two scales whose absolute levels differ by an order of magnitude.
2. "counts are the steadier measurement." Only against occupancy. Mean bout DURATION has the
   lowest CV of the three and still resolves 1 of 8 -- so low variance is not what is doing the
   work, and the chain "steadier therefore resolves more" does not hold.

    python scripts/mice_behavior/build_outcome.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from src.mice_behavior.phase_ate import TRANSITIONS, pool_deltas             # noqa: E402
from build_estimates import labelled_truth, out_of_fold_predictions          # noqa: E402
from event_eval import runs                                                  # noqa: E402

OUT = ROOT / 'results' / 'vision' / 'mice' / 'frame' / '_figures'
FPS = 5.0
LABELS = ('nt', 'nn')
UNITS = ('counts', 'occupancy', 'duration')


def per_observation():
    """All three units per observation, plus every bout length, from the human labels."""
    a = pd.read_csv(ROOT / 'dataset' / 'mice' / 'v1' / 'annotations.csv',
                    usecols=['observation_id', 'frame_idx', 'Y_nt', 'Y_nn'],
                    low_memory=False).dropna(subset=['Y_nt'])
    a = a.sort_values(['observation_id', 'frame_idx'])
    rows, lens = [], {l: [] for l in LABELS}
    for oid, g in a.groupby('observation_id', sort=False):
        n = len(g); rec = {'observation_id': oid}
        for l in LABELS:
            v = g['Y_' + l].to_numpy() > 0.5
            L = [b - s + 1 for s, b in runs(v)]
            lens[l] += L
            rec[f'counts_{l}'] = len(L) / (n / FPS / 60)
            rec[f'occupancy_{l}'] = v.mean() * 100
            rec[f'duration_{l}'] = (np.mean(L) / FPS) if L else np.nan
        rows.append(rec)
    e = pd.read_csv(ROOT / 'data' / 'mice' / 'v1' / 'experiment.csv')[
        ['observation_id', 'pool', 'phase', 'odor']]
    return pd.DataFrame(rows).merge(e, on='observation_id'), lens


def contrast(d, col, x, y):
    w = d.pivot_table(index='pool', columns='phase', values=col)
    v = (w[y] - w[x]).dropna().to_numpy()
    if len(v) < 2:
        return np.nan, np.nan, np.nan
    m, se = v.mean(), v.std(ddof=1) / np.sqrt(len(v))
    q = stats.t.ppf(0.975, len(v) - 1)
    return m, m - q * se, m + q * se


# ------------------------------------------------------------------ distributions for the figure
# The outcome section argues about MEASURABILITY, and the two facts it argues from are both
# distributional: how many events a recording contains, and how long one event lasts. Both were
# quoted as single numbers and are now emitted in full so the figure can show the shape rather
# than the summary. Nothing here is a new measurement -- it is the same bouts `per_observation`
# already walks, binned.
LEN_MAX = 15                      # lengths 1..LEN_MAX individually, then one tail bucket


def distributions(t: pd.DataFrame, lens: dict) -> dict:
    """Per-observation event counts and pooled bout lengths, binned for the section-02 figure."""
    dur = t.phase.map({'H': 30.0, 'O': 15.0, 'P': 15.0})
    out = {'meta': {'fps': FPS, 'n_obs': int(len(t)), 'n_pools': int(t.pool.nunique()),
                    'len_max': LEN_MAX,
                    'phase_minutes': {'H': 30, 'O': 15, 'P': 15}},
           'counts': {}, 'lengths': {}}
    for l in LABELS:
        per_min = t[f'counts_{l}'].to_numpy(float)
        per_rec = np.rint(per_min * dur.to_numpy(float)).astype(int)
        rec = {}
        for key, v, step in (('per_min', per_min, 0.2), ('per_rec', per_rec.astype(float), 5.0)):
            hi = step * (np.floor(v.max() / step) + 1)
            edges = np.arange(0, hi + step / 2, step)
            h, _ = np.histogram(v, bins=edges)
            by = {}
            for ph, g in t.assign(_v=v).groupby('phase'):
                by[ph] = {'median': round(float(g._v.median()), 4),
                          'mean': round(float(g._v.mean()), 4), 'n': int(len(g))}
            rec[key] = {'step': step, 'edges': [round(float(x), 4) for x in edges],
                        'hist': [int(x) for x in h],
                        'median': round(float(np.median(v)), 4),
                        'mean': round(float(v.mean()), 4),
                        'max': round(float(v.max()), 4),
                        'zero': int((v == 0).sum()), 'by_phase': by}
        out['counts'][l] = rec

        L = np.asarray(lens[l], dtype=int)
        buckets = list(range(1, LEN_MAX + 1))
        n_b = [int((L == k).sum()) for k in buckets] + [int((L > LEN_MAX).sum())]
        t_b = [int(L[L == k].sum()) for k in buckets] + [int(L[L > LEN_MAX].sum())]
        tot_b, tot_t = sum(n_b), sum(t_b)
        srt = np.sort(L)[::-1]
        k10 = max(1, int(round(0.10 * len(L))))
        out['lengths'][l] = {
            'buckets': [str(k) for k in buckets] + [f'{LEN_MAX + 1}+'],
            'n_bouts': n_b, 'time_frames': t_b,
            'share_bouts': [round(x / tot_b, 5) for x in n_b],
            'share_time': [round(x / tot_t, 5) for x in t_b],
            'cum_time': [round(float(x), 5) for x in np.cumsum(t_b) / tot_t],
            'n': int(len(L)), 'total_frames': int(L.sum()),
            'median': float(np.median(L)), 'p90': float(np.percentile(L, 90)),
            'max': int(L.max()), 'one_frame': round(float((L == 1).mean()), 4),
            'tail10_time': round(float(srt[:k10].sum() / L.sum()), 4)}
    return out


def main():
    t, lens = per_observation()
    print(f'{t.observation_id.nunique()} observations, {t.pool.nunique()} pools')

    out = {'units': {}, 'bouts': {}, 'dist': distributions(t, lens)}
    for l in LABELS:
        L = np.array(lens[l]); s = np.sort(L)[::-1]
        k = max(1, int(round(0.10 * len(L))))
        out['bouts'][l] = {'n': int(len(L)), 'one_frame': round(float((L == 1).mean()), 3),
                           'median_frames': float(np.median(L)),
                           'tail10': round(float(s[:k].sum() / L.sum()), 3)}
        print(f"  {l}: {len(L)} bouts, {(L == 1).mean():.1%} single-frame, "
              f"median {np.median(L):.0f} frames, top 10% carry {s[:k].sum() / L.sum():.1%}")

    print('\nout-of-fold predictions (for r_delta and the bias spread):')
    oof, _, _ = out_of_fold_predictions()
    truth = labelled_truth()
    exp = pd.read_csv(ROOT / 'data' / 'mice' / 'v1' / 'experiment.csv')
    lab = (truth.merge(oof, on='observation_id')
           .merge(exp[['observation_id', 'pool', 'phase', 'odor']], on='observation_id'))
    # build_estimates names the model's two units events/time; this file names them by what they
    # mean. Same columns, one mapping, stated here so the two scripts cannot silently diverge.
    PRED = {'counts': 'events', 'occupancy': 'time'}

    for u in UNITS:
        rec = {}
        n_sig, detail = 0, []
        for l in LABELS:
            for od in ('F', 'S'):
                for x, y in TRANSITIONS:
                    m, lo, hi = contrast(t[t.odor == od], f'{u}_{l}', x, y)
                    sig = np.isfinite(lo) and lo * hi > 0
                    n_sig += bool(sig)
                    detail.append({'behav': l, 'odour': od, 'trans': f'{x}->{y}',
                                   'est': round(float(m), 3), 'resolved': bool(sig)})
        rec['resolves'] = n_sig
        rec['contrasts'] = detail
        rec['cv'] = {}
        for l in LABELS:
            cvs = [g[f'{u}_{l}'].dropna().std(ddof=1) / g[f'{u}_{l}'].dropna().mean()
                   for _, g in t.groupby(['phase', 'odor'])
                   if g[f'{u}_{l}'].dropna().mean() > 0]
            rec['cv'][l] = round(float(np.mean(cvs)), 3)
        if u in PRED:                                    # the model has no duration head
            p = PRED[u]
            rec['r_delta'], rec['bias_spread'] = {}, {}
            for l in LABELS:
                dy, df = [], []
                for od in ('F', 'S'):
                    for tr in TRANSITIONS:
                        d = pool_deltas(lab, f't_{p}_{l}', f'f_{p}_{l}', od, tr)
                        # PAIRED pools only: a correlation between the two sides needs both of
                        # them, and on `decay` either one can be undefined.
                        dy += list(d.d_true[d.paired]); df += list(d.d_pred[d.paired])
                rec['r_delta'][l] = round(float(np.corrcoef(dy, df)[0, 1]), 3)
                r = [lab[lab.phase == ph][f'f_{p}_{l}'].mean()
                     / lab[lab.phase == ph][f't_{p}_{l}'].mean() for ph in ('H', 'O', 'P')]
                rec['bias_spread'][l] = round(float(max(r) / min(r)), 2)
        out['units'][u] = rec
        print(f'\n{u}: resolves {n_sig}/8   CV ' + ', '.join(
            f'{l} {rec["cv"][l]:.2f}' for l in LABELS)
            + ('   r_delta ' + ', '.join(f'{l} {rec["r_delta"][l]:.2f}' for l in LABELS)
               + '   bias spread ' + ', '.join(f'{l} {rec["bias_spread"][l]:.2f}x' for l in LABELS)
               if 'r_delta' in rec else '   (no model head for this unit)'))

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / 'outcome.json', 'w'), indent=1)
    print(f'\nwrote {OUT / "outcome.json"}')


if __name__ == '__main__':
    main()
