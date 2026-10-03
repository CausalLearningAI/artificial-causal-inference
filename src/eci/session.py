"""
Recording-session checks for the frogs (mutant vs WT, family B) NES results.

Every frogs recording session holds one group only (WT 157/158/283, FoxP1 9 sessions, En1 220/222/293), so group and
session are fully confounded: no analysis of these data can separate a group effect from a session effect (dish,
lighting, water, the day's cohort). What CAN be measured is how much a latent depends on the session WITHIN a group;
a group effect carried by a latent that also differs strongly between sessions of the same group is more likely a
session artefact. This is the frogs analogue of the mice camera-period check (src/eci/period.py), read-only on the
NES caches.

1. Session-dependence flag (per latent, per outcome). Unit = frog (one video). Score = max over the groups with >= 2
   sessions of eta^2 of the within-group RANKS of the per-video outcome on session (between-session sum of squares /
   total sum of squares of the ranks = the tie-corrected Kruskal-Wallis H / (n - 1)), i.e. how well the session
   explains the latent among frogs of ONE group (a group effect cannot produce it). The group attaining the max is
   kept. Threshold: label-shuffle null; session labels permuted among the frogs of each group (session sizes kept), the
   score recomputed for every latent and the max over latents taken; threshold = 95th percentile of that max
   (family-wise 5% that any latent is flagged by chance). p_perm = per-latent permutation p (+1 smoothing).
   What it cannot do: a latent with no within-group session spread can still be a between-group session difference
   (e.g. a dish type used only for one group); not flagged != not confounded.

2. Leave-one-session-out re-test of the primary picks. Each pick of round r is re-tested with the Neural Effect Test
   (src/eci/nes.py, S = the picks of rounds < r of the same search) once per session of the analysis with that
   session's frogs dropped. Reported: the largest p and whether the sign of tau is the same in every drop; a pick that
   loses its sign or significance when one session is dropped rests on that session.
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from eci.nes import neural_effect_test


def _eta2_ranks(R, codes, n_lv):
    """R (n, m) ranks, codes (n,) session index 0..n_lv-1 -> (m,) between-session SS / total SS (0 for constant)."""
    mu = R.mean(0)
    tot = ((R - mu) ** 2).sum(0)
    cnt = np.bincount(codes, minlength=n_lv).astype(np.float64)
    sums = np.zeros((n_lv, R.shape[1]))
    np.add.at(sums, codes, R)
    between = (sums ** 2 / np.maximum(cnt, 1)[:, None]).sum(0) - R.shape[0] * mu ** 2
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(tot > 0, between / tot, 0.0)


def session_scores(design, Y, group_col='group', level_col='session', n_shuffles=1000, seed=0, q=0.95):
    """Score of every latent (module docstring 1) + label-shuffle null. design rows = videos (obs_row indexes Y).
    Returns dict: score, group (index into groups), p_perm, group_scores (n_groups, m), null_max, threshold,
    pointwise_threshold, groups, n_frogs ({group: {session: n}})."""
    blocks, groups, n_frogs = [], [], {}
    for g, d in design.groupby(group_col, sort=True):
        codes, lv = pd.factorize(d[level_col])
        n_frogs[str(g)] = {str(k): int(v) for k, v in d[level_col].value_counts().sort_index().items()}
        if len(lv) < 2:
            continue
        blocks.append((rankdata(Y[d['obs_row'].values], axis=0), codes, len(lv)))
        groups.append(str(g))
    if not blocks:
        raise SystemExit('session_scores: no group with >= 2 sessions')
    gs = np.stack([_eta2_ranks(R, c, k) for R, c, k in blocks])
    best = gs.argmax(0)
    m = gs.shape[1]
    score = gs[best, np.arange(m)]
    rng = np.random.default_rng(seed)
    null_max, null_all, ge = np.empty(n_shuffles), np.empty((n_shuffles, m)), np.zeros(m)
    for s in range(n_shuffles):
        f = np.stack([_eta2_ranks(R, rng.permutation(c), k) for R, c, k in blocks]).max(0)
        null_max[s], null_all[s] = f.max(), f
        ge += f >= score - 1e-12
    live = np.ptp(Y[design['obs_row'].values], 0) > 0
    return {'score': score, 'group': best, 'p_perm': (ge + 1) / (n_shuffles + 1), 'group_scores': gs,
            'null_max': null_max, 'threshold': float(np.quantile(null_max, q)),
            'pointwise_threshold': float(np.quantile(null_all[:, live], q)), 'groups': groups, 'n_frogs': n_frogs}


def leave_one_session_out(picks, design, values, analyses, level_col='session'):
    """Re-test primary picks without each session in turn (module docstring 2).

    picks: rows of one primary setting (round > 0; columns analysis_id, prefix, round, neuron, tau, p, threshold);
    values: (n_obs, m) per-video outcome (full window); analyses: {analysis_id: Analysis}. Returns picks with
    n_sessions, worst_session (the drop with the largest p), p_max, tau_min_abs, same_sign (every drop), p_max_below
    (p_max < the round threshold)."""
    out = []
    for (aid, prefix), grp in picks.groupby(['analysis_id', 'prefix'], sort=False):
        rows = analyses[aid].select(design).sort_values(analyses[aid].unit)
        Z = values[rows['obs_row'].values][:, :int(prefix)]
        T, ses = rows['T'].values, rows[level_col].values
        prev = []
        for r in grp.sort_values('round').to_dict('records'):
            j = int(r['neuron'])
            res = []
            for s in np.unique(ses):
                keep = ses != s
                tab, _ = neural_effect_test(Z[keep], T[keep], prev, cols=[j])
                res.append((s, float(tab.iloc[0]['tau']), float(tab.iloc[0]['p'])))
            worst = max(res, key=lambda x: x[2])
            out.append({**r, 'n_sessions': len(res), 'worst_session': worst[0], 'p_max': worst[2],
                        'tau_min_abs': min(abs(t) for _, t, _ in res),
                        'same_sign': all(np.sign(t) == np.sign(r['tau']) for _, t, _ in res),
                        'p_max_below': bool(worst[2] < r['threshold'])})
            prev.append(j)
    return pd.DataFrame(out)
