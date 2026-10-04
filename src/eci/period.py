"""
Camera-period checks for the het vs wt (family B) NES results on mice v1.

The recording period (dataset/mice/v1/eci/odor_corner.csv 'odor_corner': BL, BR, TR; the camera zoom differs
between periods) is not randomized with respect to genotype (BR = 2 het mice only; wt is mostly TR). The primary
family-B analysis stays unadjusted on all 72 videos per stage. This module adds two read-only checks of its results:

1. Period-dependence flag (per latent, per outcome). Unit = mouse (pool): the per-video outcome averaged over the
   mouse's 6 stage videos (the period is constant within a mouse). Score = max over the within-genotype period pairs
   PAIRS (wt TR vs BL, het TR vs BL) of the folded AUC |AUC - 0.5| + 0.5, i.e. how well the latent's outcome
   separates the two periods among mice of ONE genotype (so a genotype effect cannot produce it). The pair that
   attains the max and its raw AUC (> 0.5 = higher in the first period of the pair) are kept.
   BR is left out of the score: it has 2 mice, so a pair with BR reaches AUC 1.0 by chance with probability
   1/60 (BR vs het BL) per latent and the max over 1024 latents would be 1.0 under the null. Its separation is kept
   as a descriptive z (BR mean minus het non-BR mean, in het non-BR standard deviations).
   Threshold: label-shuffle null. Period labels are permuted among the mice of each genotype (counts kept), the
   score is recomputed for every latent and the max over latents is taken; threshold = 95th percentile of that max
   (family-wise 5% that any latent of the model is flagged by chance). pointwise_threshold = 95th percentile of one
   (non-constant) latent's null score, for reference only (no multiplicity control). p_perm = per-latent permutation p
   (share of shuffles with a score >= the observed one, +1 smoothing).

2. Day-adjusted re-test of the primary picks (robustness chip, like size-adj). Covariate = recording day = days
   since the first recording, continuous, from the timestamp in the video file name (data/mice/v1/experiment.csv
   'observation_file'; its 'time' column has a placeholder for 9 rows and is not used). All 72 videos per stage are
   kept (BR included). Each pick of round r is re-tested with the Neural Effect Test (src/eci/nes.py) given
   S = [day] + the picks of rounds < r of the same search, i.e. exactly the test that round would have run had the
   search conditioned on the day from round 0 (as --nuisance). survives = p_day below that round's Bonferroni
   threshold and the same sign as the unadjusted tau.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from eci.nes import neural_effect_test

PAIRS = (('wt', 'TR', 'BL'), ('het', 'TR', 'BL'))  # (genotype, period a, period b): AUC > 0.5 = higher in a


def recording_days(design, experiment_csv):
    """Days since the first recording (float) of every design row, from the video file name timestamp."""
    e = pd.read_csv(experiment_csv).set_index('observation_id')
    ts = pd.to_datetime(e['observation_file'].str[:19], format='%Y-%m-%d_%H-%M-%S')
    if ts.isna().any():
        raise SystemExit(f'recording_days: {int(ts.isna().sum())} file names without a timestamp')
    ts = ts.reindex(design['observation_id'])
    if ts.isna().any():
        raise SystemExit(f'recording_days: {int(ts.isna().sum())} design rows missing from {experiment_csv}')
    return ((ts - ts.min()).dt.total_seconds() / 86400.0).values


def pool_periods(design, odor_corner_csv):
    """Per design row: the period label (fails loud on missing labels or a period that varies within a pool)."""
    oc = pd.read_csv(odor_corner_csv).set_index('observation_id')['odor_corner']
    per = oc.reindex(design['observation_id']).values
    if pd.isna(per).any():
        raise SystemExit(f'{int(pd.isna(per).sum())} observations without a period in {odor_corner_csv}')
    if pd.Series(per).groupby(design['pool'].values).nunique().max() > 1:
        raise SystemExit('the camera period varies within a pool')
    return per


def pool_table(design, Y, per):
    """-> (pools DataFrame [pool, genotype, period], (n_pools, m) mean of Y over each pool's rows)."""
    d = design.assign(period=per).reset_index(drop=True)
    g = d.groupby('pool', sort=True)
    pools = g[['genotype', 'period']].first().reset_index()
    if (g['genotype'].nunique() > 1).any():
        raise SystemExit('genotype varies within a pool')
    Yp = np.stack([Y[d.loc[idx, 'obs_row'].values].mean(0) for _, idx in g.groups.items()])
    return pools, Yp


def _ranks(Y):
    """Column-wise average ranks (ties averaged) of Y (n, m), 1-based."""
    from scipy.stats import rankdata
    return rankdata(Y, axis=0)


def _auc_from_ranks(R, ia):
    """AUC of group a (row mask ia) vs the other rows of the rank matrix R (ranks within these rows)."""
    na, nb = int(ia.sum()), int((~ia).sum())
    return (R[ia].sum(0) - na * (na + 1) / 2.0) / (na * nb)


def period_scores(pools, Yp, pairs=PAIRS, n_shuffles=1000, seed=0, q=0.95):
    """Score of every latent (module docstring 1) + label-shuffle null.
    Returns dict with arrays score, pair (index into pairs), auc (raw AUC of that pair), p_perm, br_z, and
    threshold, null_max (n_shuffles,), pair_aucs (n_pairs, m)."""
    m = Yp.shape[1]
    blocks = []  # per pair: (rank matrix within the pair's mice, labels a, genotype rows)
    for geno, a, b in pairs:
        rows = np.flatnonzero((pools['genotype'].values == geno) & pools['period'].isin([a, b]).values)
        ia = pools['period'].values[rows] == a
        if ia.sum() == 0 or (~ia).sum() == 0:
            raise SystemExit(f'pair {geno} {a} vs {b}: one side empty')
        blocks.append((_ranks(Yp[rows]), ia))
    aucs = np.stack([_auc_from_ranks(R, ia) for R, ia in blocks])  # (n_pairs, m)
    fold = np.abs(aucs - 0.5) + 0.5
    best = fold.argmax(0)
    score = fold[best, np.arange(m)]
    # null: period labels permuted among the pair's mice (one pair per genotype, so = within genotype)
    if len({p[0] for p in pairs}) != len(pairs):
        raise SystemExit('period_scores: one pair per genotype expected (the null permutes within a pair)')
    rng = np.random.default_rng(seed)
    null_max = np.empty(n_shuffles)
    null_all = np.empty((n_shuffles, m))
    ge = np.zeros(m)
    for s in range(n_shuffles):
        f = np.stack([np.abs(_auc_from_ranks(R, rng.permutation(ia)) - 0.5) + 0.5 for R, ia in blocks]).max(0)
        null_max[s], null_all[s] = f.max(), f
        ge += f >= score - 1e-12
    p_perm = (ge + 1) / (n_shuffles + 1)
    # descriptive: BR (het only) vs the other het mice
    het = pools['genotype'].values == 'het'
    br = het & (pools['period'].values == 'BR')
    rest = het & ~br
    sd = Yp[rest].std(0, ddof=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        br_z = np.where(sd > 0, (Yp[br].mean(0) - Yp[rest].mean(0)) / sd, 0.0) if br.any() else np.full(m, np.nan)
    return {'score': score, 'pair': best, 'auc': aucs[best, np.arange(m)], 'p_perm': p_perm, 'br_z': br_z,
            'pair_aucs': aucs, 'null_max': null_max, 'threshold': float(np.quantile(null_max, q)),
            'pointwise_threshold': float(np.quantile(null_all[:, np.ptp(Yp, 0) > 0], q)),
            'n_mice': {f'{g} {a} vs {b}': [int(blk[1].sum()), int((~blk[1]).sum())]
                       for (g, a, b), blk in zip(pairs, blocks)}, 'n_br': int(br.sum())}


def day_adjusted_picks(picks, design, values, day, analyses):
    """Re-test primary picks with the recording day conditioned on (module docstring 2).

    picks: DataFrame rows of one primary search setting per (analysis_id, prefix) (round > 0; columns round, neuron,
    tau, p, threshold, direction); values: (n_obs, m) per-video outcome matrix (full window); day: (n_obs,) days;
    analyses: {analysis_id: Analysis} (family B). Returns picks with tau_day, se_day, p_day, df_day, n_cells,
    survives."""
    out = []
    for (aid, prefix), grp in picks.groupby(['analysis_id', 'prefix'], sort=False):
        rows = analyses[aid].select(design).sort_values(analyses[aid].unit)
        Z = values[rows['obs_row'].values][:, :int(prefix)]
        T = rows['T'].values
        Za = np.column_stack([Z, day[rows['obs_row'].values]])
        dcol = Z.shape[1]
        grp = grp.sort_values('round')
        prev = []
        for r in grp.to_dict('records'):
            j = int(r['neuron'])
            try:
                tab, info = neural_effect_test(Za, T, [dcol] + prev, cols=[j])
            except ValueError as err:  # e.g. the arms do not overlap on the day (nes.check_overlap): not estimable
                out.append({**r, 'tau_day': np.nan, 'se_day': np.nan, 'p_day': np.nan, 'df_day': np.nan,
                            'n_cells': 0, 'n_units': len(T), 'survives': False, 'not_estimable': str(err)})
                prev.append(j)
                continue
            t = tab.iloc[0]
            out.append({**r, 'tau_day': float(t['tau']), 'se_day': float(t['se']), 'p_day': float(t['p']),
                        'df_day': float(t['df']), 'n_cells': info['n_cells'], 'n_units': len(T),
                        'survives': bool(t['p'] < r['threshold'] and np.sign(t['tau']) == np.sign(r['tau'])),
                        'not_estimable': ''})
            prev.append(j)
    return pd.DataFrame(out)
