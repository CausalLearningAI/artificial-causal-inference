#!/usr/bin/env python3
"""How good could ANY model be, given that the labels themselves disagree?

v1 has no double annotation -- 0 of 432 observations were labelled twice -- so annotator
agreement cannot be measured directly. But it can be BOUNDED from the experimental design,
because the design happens to contain the right comparison: within one genotype group the six
pools are exchangeable (same genotype, and every pool contributes the identical 3 phases x 2
odors), yet they were labelled by different people. Any variance that tracks annotator identity
after conditioning on the experimental cell is variance no behaviour model should be credited
or penalised for.

The estimator
-------------
Condition on the experimental cell, then decompose what is left:

    pool level        cell = genotype group          (4 cells x 6 pools)
    observation level cell = genotype x phase x odor (24 cells x 6 observations)

One-way ANOVA with annotator as the factor, computed WITHIN cell and pooled across cells, gives
sigma^2_A (between annotators) and sigma^2_W (within). Then

    rho   = sigma^2_W / (sigma^2_A + sigma^2_W)     an UPPER bound on label reliability
    r_max = sqrt(rho)                               the best correlation any model can reach

rho is an upper bound because sigma^2_W still contains annotation noise that no design without
replication can separate from real between-pool differences; the true reliability is lower.
Two annotators labelling the same videos would agree at rho, and a model that predicted the
TRUE rate perfectly would correlate with the observed labels at sqrt(rho) -- that is the number
to compare `rate_report`'s r against.

What this cannot do
-------------------
Annotator is nearly collinear with pool (22 of 24 annotated pools have a single annotator), so
sigma^2_A is aliased with cage/cohort/date. The permutation test below asks only whether the
ANNOTATOR grouping explains more than a random grouping of the same shape -- it cannot say the
cause is the annotator rather than the cage. That ambiguity does not change the conclusion for
model evaluation: either it is annotator bias, which no model can predict, or it is a cage
effect, which a model could only predict by reading nuisance appearance rather than behaviour.
Neither belongs in a behaviour score.

Usage:
    python scripts/mice_behavior/annotation_ceiling.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LABELS = ('Y_nt', 'Y_nn')
NICE = {'Y_nt': 'nt', 'Y_nn': 'nn'}


def load() -> pd.DataFrame:
    """Per-observation annotated rate joined to the design. 144 rows (the annotated half)."""
    a = pd.read_csv(ROOT / 'dataset' / 'mice' / 'v1' / 'annotations.csv',
                    usecols=['observation_id', 'Y_nt', 'Y_nn'], low_memory=False)
    obs = a.groupby('observation_id')[list(LABELS)].mean().reset_index()
    e = pd.read_csv(ROOT / 'data' / 'mice' / 'v1' / 'experiment.csv')
    m = obs.merge(e[['observation_id', 'pool', 'line', 'genotype', 'phase', 'odor', 'annotator']],
                  on='observation_id').dropna(subset=['annotator'])
    # The 4 groups the experiment actually contrasts: wild type, and het of each line. wt is
    # pooled across lines because a wt littermate is the same genotype whichever line it came
    # from; conditioning on this is what makes two pools exchangeable apart from their annotator.
    m['g4'] = np.where(m.genotype == 'wt', 'wt', 'het_' + m.line)
    return m


def cells(df: pd.DataFrame, value: str, cell: list[str]):
    """Split into usable cells as plain numpy (values, annotator codes).

    Cells with a single annotator are dropped: they carry no information about annotator
    disagreement, and keeping them would pull the estimate toward zero through design imbalance
    rather than through evidence. Extracting to numpy once is what makes the permutation test
    affordable -- the pandas version re-grouped 24 cells per replicate and took minutes.
    """
    out = []
    for _, g in df.groupby(cell):
        if g.annotator.nunique() < 2 or len(g) < 3:
            continue
        out.append((g[value].to_numpy(float), pd.factorize(g.annotator)[0]))
    return out


def _ss(vals, codes):
    """Between/within annotator sums of squares for one cell."""
    k = codes.max() + 1
    n = np.bincount(codes, minlength=k).astype(float)
    s = np.bincount(codes, weights=vals, minlength=k)
    means = s / n
    ssb = float((n * (means - vals.mean()) ** 2).sum())
    ssw = float(((vals - means[codes]) ** 2).sum())
    return ssb, ssw, n


def components(cs):
    """Pooled within-cell one-way ANOVA on annotator -> (sigma2_A, sigma2_W, eta2, n_cells)."""
    ssb = ssw = 0.0
    dfb = dfw = 0
    n0_num, n0_den = 0.0, 0.0
    for vals, codes in cs:
        b, w, n = _ss(vals, codes)
        ssb += b; ssw += w
        dfb += len(n) - 1; dfw += int(n.sum()) - len(n)
        n0_num += n.sum() - (n ** 2).sum() / n.sum()
        n0_den += len(n) - 1
    if dfb == 0 or dfw == 0:
        return float('nan'), float('nan'), float('nan'), 0
    msb, msw = ssb / dfb, ssw / dfw
    n0 = n0_num / n0_den if n0_den else 1.0
    s2a = max((msb - msw) / n0, 0.0)          # variance components are non-negative by definition
    return s2a, msw, ssb / (ssb + ssw), len(cs)


def permute(cs, reps=5000, seed=0):
    """Null: shuffle annotator labels WITHIN each cell, preserving group sizes.

    The right null because it holds the experimental design and each annotator's workload fixed,
    and asks only whether the ACTUAL assignment of people to pools explains more variance than a
    random assignment of the same shape.
    """
    rng = np.random.default_rng(seed)
    obs = components(cs)[2]
    null = np.empty(reps)
    for i in range(reps):
        null[i] = components([(v, rng.permutation(c)) for v, c in cs])[2]
    null = null[np.isfinite(null)]
    return obs, float(null.mean()), float((null >= obs).mean()) if len(null) else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=2000)
    args = ap.parse_args()
    m = load()

    # Pool level is the PRIMARY test: annotator is assigned per pool, so the 24 pools are the
    # independent units and the permutation null is exact. The observation-level test has more
    # cells and therefore looks more significant, but its p is ANTICONSERVATIVE -- each pool
    # contributes one observation to each of 6 cells carrying the SAME annotator, so shuffling
    # within cells independently generates nulls that break a dependency the real design has.
    # Read the observation-level row for its effect size, not its p-value.
    levels = [
        ('POOL LEVEL  (n=24 pools, cell = genotype group)  <- PRIMARY: independent units, exact null',
         m.groupby(['pool', 'g4', 'annotator'])[list(LABELS)].mean().reset_index(), ['g4']),
        ('OBSERVATION LEVEL  (n=144, cell = genotype x phase x odor)  <- effect size only, p is anticonservative',
         m, ['g4', 'phase', 'odor']),
    ]
    for title, d, cell in levels:
        print('=' * 96); print(title); print('=' * 96)
        for y in LABELS:
            cs = cells(d, y, cell)
            s2a, s2w, eta2, ncell = components(cs)
            if not np.isfinite(s2a):
                print(f'  {NICE[y]}: not estimable'); continue
            rho = s2w / (s2a + s2w)
            obs_e, null_e, p = permute(cs, reps=args.reps)
            print(f'  {NICE[y]}:  annotator share of within-cell variance = {eta2:5.1%}   '
                  f'(chance {null_e:5.1%}, permutation p={p:.3f}, {ncell} usable cells)')
            print(f'        reliability rho <= {rho:5.3f}   ->  two annotators would agree at '
                  f'r <= {rho:.2f};  BEST POSSIBLE MODEL r <= {np.sqrt(rho):.2f}')
        print()

    print('=' * 96)
    print('WHERE THE MODELS ACTUALLY SIT (observation-level r on the 24 val observations)')
    print('=' * 96)
    frame = ROOT / 'results' / 'vision' / 'mice' / 'frame'
    import json
    rows = []
    for cfg_p in sorted(frame.glob('*/config.json')):
        cfg = json.load(open(cfg_p))
        rr, apr = cfg.get('rate_report', {}), cfg.get('ap_report', {})
        if 'macro/tol0' not in apr or not rr:
            continue
        rows.append((apr['macro/tol0']['ap'], cfg_p.parent.name,
                     rr.get('nt', {}).get('pearson_r'), rr.get('nn', {}).get('pearson_r')))
    for ap_, tag, rnt, rnn in sorted(rows, reverse=True)[:5]:
        print(f'  {tag:34s} r_nt={rnt:+.3f}  r_nn={rnn:+.3f}')
    print('\nCompare each r against the ceiling above, not against 1.0.')


if __name__ == '__main__':
    main()
