"""
Neural Effect Search (NES) for the ECI (Exploratory Causal Inference) pipeline on mice v1.

Reference: Mencattini, Cadei, Locatello, "Exploratory Causal Inference in SAEnce",
arXiv 2510.14073 -- Section 4 (Algorithm 1), Appendix A (theory, optional residualization),
Appendix B (Algorithm 2, Neural Effect Test) and Appendix C (reference snippet).

Problem. Z (n units x m neurons) are SAE concept activations. Testing every neuron for a
treatment effect with Bonferroni flags every neuron that *leaks* a true effect (noisy mixtures
of the truly affected concepts) once n is large: the "paradox of exploratory causal inference".
NES fixes this by recursion: find the strongest significant neuron, then re-test all others
*controlling* for the neurons already found, until nothing is significant.

Neural Effect Test (NET), two-sample, for neuron j given the selected set S (Algorithm 2):
    A) residualize Z_j on Z_S separately in each arm (arm-wise OLS),
    B) cut units into treatment-agnostic strata by pooled quantiles of Z_S (median split by
       default),
    C) tau_j = sum_g w_g (mu_1g - mu_0g), w_g = n_g / sum_h n_h, variance, Satterthwaite df,
       two-sided t p-value.
NES (Algorithm 1): run NET on every remaining neuron, reject at alpha / m_remaining
(Bonferroni), stop if nothing is rejected, else add the rejected neuron with the largest |tau|
(Algorithm 1, line 6) to S and recurse.

Residualization modes of the two-sample NET (`residualize`):
  'ols' (default)  Steps A and C done jointly and exactly, per arm t: OLS of Z_j on stratum
                   indicators + arm-t slopes on Z_S centred at the stratum's pooled mean of Z_S.
                   The stratum intercepts are the arm-t means of Z_j adjusted to that common
                   point (Lin 2013 interacted regression), tau = sum_g w_g (c_1g - c_0g) and the
                   variance is the exact OLS one, per arm, df_t = n_t - rank.
                   Why not the paper's plug-in (residuals, then Welch on stratum means): Z_S is
                   post-treatment, so the arm means of Z_S differ by O(1) from the centring point
                   and the slope's estimation error enters tau at the same O(n^-1/2) order as the
                   noise. Welch on residuals ignores that term and underestimates the SE (by
                   ~sqrt(1 + (tau_S / 2 sd_S)^2), e.g. 25% for a 1.5 sd effect), which makes the
                   rounds after the first anti-conservative. The OLS variance includes it.
                   Pooling the residual variance across strata within an arm also gives far more
                   df than per-cell Welch (the treated-low / control-high cells of a strongly
                   affected Z_S are small), which matters at Bonferroni-level alphas.
  'crossfit'       the paper's recipe as written: arm-wise OLS residuals cross-fitted over
                   n_folds, then per-stratum Welch contrast. Kept for comparison.
  None / False     stratification only (Appendix C snippet), per-stratum Welch.
Residuals are centred at the pooled Z_S mean, so the contrast is tau_j - b' tau_S: the part of
neuron j's effect not predicted by the effects of the neurons already found. The paper writes
r = Z_j - b_t' Z_S; this equals ours when b_1 = b_0 and differs by the arm constant
(b_1 - b_0)' mean(Z_S) otherwise.

Strata cap: q = n_strata bins per selected neuron gives up to q^|S| cells. We use the first k
selected neurons (selection order, i.e. most prominent first) and the largest k, then the
largest q' <= q, such that EVERY non-empty cell has >= min_per_cell units in each arm; else we
coarsen, down to a single stratum. The choice depends on (T, Z_S), never on the tested neuron,
and residualization always adjusts for ALL of S, so leakage removal does not depend on how
coarse the strata are.

Near-constant neurons (active in < min_active of the units, or zero variance) are not tested
and do not count in the Bonferroni m. Their count is reported.

Paired design (stage transitions, the same pool at stage a and b): see paired_effect_test().

Nuisance conditioning (optional `nuisance` argument of both searches): per-unit covariates that are
not concepts but can drive many neurons (e.g. the per-video mean foreground size = how spread out the
mice are). They are appended as extra columns and treated as ALREADY SELECTED from round 0: every
test (round 1 included) stratifies on them first and residualizes on them exactly as on selected
neurons, they are never tested themselves and do not count in the Bonferroni m. With nuisance=None
the searches are unchanged.

Functions:
    unit_means               per-unit (e.g. per-video) mean activations from frame rows
    active_neurons           mask of neurons that are not near-constant
    make_strata              treatment-agnostic strata from pooled quantiles, with the cap
    neural_effect_test       Algorithm 2 (two-sample) on a set of neurons given S
    neural_effect_search     Algorithm 1 (two-sample); unit = video/pool, or frame
    paired_effect_test       paired analogue of Algorithm 2 (t or sign-flip p-values)
    paired_effect_search     paired analogue of Algorithm 1
"""

import time

import numpy as np
import pandas as pd
from scipy import stats


def unit_means(Z, groups, T=None):
    """Average frame rows into units (e.g. one row per video).

    Z: (n_frames, m); groups: (n_frames,) unit id per frame; T: optional (n_frames,) treatment,
    which must be constant within a unit. Returns (unit_ids, Z_units[, T_units]), units sorted.
    """
    Z = np.asarray(Z, dtype=np.float64)
    codes, uniq = pd.factorize(np.asarray(groups), sort=True)
    Zu = np.zeros((len(uniq), Z.shape[1]))
    np.add.at(Zu, codes, Z)
    Zu /= np.bincount(codes)[:, None]
    if T is None:
        return uniq, Zu
    T = np.asarray(T)
    Tu = np.zeros(len(uniq), dtype=T.dtype)
    Tu[codes] = T
    if not np.array_equal(Tu[codes], T):
        raise ValueError('treatment is not constant within a unit')
    return uniq, Zu, Tu


def active_neurons(Z, min_active=0.01, eps=1e-8):
    """True for neurons active (|z| > eps) in at least a min_active fraction of the rows and
    with non-zero variance. Z may be two-sample units or stacked paired observations."""
    Z = np.asarray(Z)
    return ((np.abs(Z) > eps).mean(0) >= min_active) & (Z.std(0) > eps)


def _quantile_bins(x, q):
    """Bin index 0..q-1 by pooled quantile cutpoints; ties at a cut go to the lower bin, so a
    sparse neuron with median 0 splits into 'zero' vs 'active' (as the paper's `> median`)."""
    cuts = np.unique(np.quantile(x, np.arange(1, q) / q))
    return np.searchsorted(cuts, x, side='left')


def _cells_ok(cell, T, min_per_cell):
    tot = np.bincount(cell)
    if T is None:
        return tot[tot > 0].min() >= min_per_cell
    return all(np.bincount(cell[T == t], minlength=len(tot))[tot > 0].min() >= min_per_cell for t in (0, 1))


def make_strata(levels, n_strata=2, min_per_cell=3, T=None):
    """Treatment-agnostic strata from pooled quantiles of `levels` (n, k), columns in selection
    order. Tries the first k' = k..1 columns and q' = n_strata..2 bins (more stratifiers first,
    then finer bins) and returns the first partition where every non-empty cell has at least
    min_per_cell units in each arm (T given) or in total (T None, paired design).
    Returns (cell (n,) int, k_used, q_used); (zeros, 0, 1) = a single stratum."""
    levels = np.asarray(levels, dtype=np.float64)
    n, k = levels.shape
    for kk in range(k, 0, -1):
        for q in range(n_strata, 1, -1):
            bins = np.stack([_quantile_bins(levels[:, c], q) for c in range(kk)], 1)
            cell = np.unique(bins, axis=0, return_inverse=True)[1].reshape(-1)
            if _cells_ok(cell, T, min_per_cell):
                return cell, kk, q
    return np.zeros(n, dtype=np.int64), 0, 1


def _t_pvalue(tau, V, den):
    with np.errstate(divide='ignore', invalid='ignore'):
        se = np.sqrt(V)
        t = np.where(se > 0, tau / se, 0.0)
        df = np.where(den > 0, V ** 2 / den, 1.0)
    p = np.where(se > 0, 2 * stats.t.sf(np.abs(t), df), 1.0)
    return se, t, df, p


def _ols_contrast(Y, X, a):
    """OLS of every column of Y on X; returns (a'coef (m,), its variance (m,), residual df, h)
    with h = the n-vector such that a'coef = h'Y. Rank-deficient X uses the pseudo-inverse."""
    pinv = np.linalg.pinv(X)
    h = pinv.T @ a
    rank = np.linalg.matrix_rank(X)
    dof = len(X) - rank
    if dof < 1:
        raise ValueError(f'no residual df (n={len(X)}, rank={rank}); raise min_per_cell or cap max_rounds')
    resid = Y - X @ (pinv @ Y)
    s2 = (resid ** 2).sum(0) / dof
    return h @ Y, s2 * (h @ h), dof, h


def _onehot(cell):
    return (cell[:, None] == np.unique(cell)[None, :]).astype(np.float64)


def _two_sample_ols(Y, X, T, cell):
    """Per arm: Y ~ stratum indicators + slopes on X centred at the stratum pooled mean."""
    G = _onehot(cell)
    w = G.sum(0) / len(cell)
    Xc = X - G @ (G.T @ X / G.sum(0)[:, None])
    est, var, dof = [], [], []
    for t in (1, 0):
        ix = T == t
        e, v, d, _ = _ols_contrast(Y[ix], np.column_stack([G[ix], Xc[ix]]), np.r_[w, np.zeros(X.shape[1])])
        est.append(e), var.append(v), dof.append(d)
    V = var[0] + var[1]
    return est[0] - est[1], V, var[0] ** 2 / dof[0] + var[1] ** 2 / dof[1], len(cell)


def _stratified_welch(R, T, cell):
    """Post-stratified Welch contrast of Algorithm 2 (lines 19-28), vectorized over columns.
    Strata with < 2 units in an arm are dropped and weights renormalized over the kept ones."""
    parts = []
    for g in np.unique(cell):
        r1, r0 = R[(cell == g) & (T == 1)], R[(cell == g) & (T == 0)]
        if len(r1) >= 2 and len(r0) >= 2:
            parts.append((len(r1), len(r0), r1.mean(0) - r0.mean(0), r1.var(0, ddof=1) / len(r1), r0.var(0, ddof=1) / len(r0)))
    n_kept = sum(n1 + n0 for n1, n0, *_ in parts)
    if n_kept == 0:
        raise ValueError('no stratum has >= 2 units in both arms')
    tau, V, den = 0.0, 0.0, 0.0
    for n1, n0, d, v1, v0 in parts:
        w = (n1 + n0) / n_kept
        tau, V = tau + w * d, V + w ** 2 * (v1 + v0)
        den = den + (w ** 2 * v1) ** 2 / max(n1 - 1, 1) + (w ** 2 * v0) ** 2 / max(n0 - 1, 1)
    return tau, V, den, n_kept


def _crossfit_residuals(Y, X, T, n_folds, rng):
    """Paper's recipe: r = Y - b_T' (X - pooled mean X), b_t fit on the other folds of arm t."""
    Xc = X - X.mean(0)
    R = np.empty_like(Y)
    for t in (0, 1):
        idx = np.flatnonzero(T == t)
        nf = max(2, min(n_folds, len(idx)))
        fold = rng.permutation(np.arange(len(idx)) % nf)
        for f in range(nf):
            te, tr = idx[fold == f], idx[fold != f]
            A = np.column_stack([np.ones(len(tr)), Xc[tr]])
            R[te] = Y[te] - Xc[te] @ np.linalg.lstsq(A, Y[tr], rcond=None)[0][1:]
    return R


def neural_effect_test(Z, T, S=(), cols=None, n_strata=2, min_per_cell=3, residualize='ols', n_folds=5, seed=0):
    """Neural Effect Test (Algorithm 2) of every neuron in `cols` given the selected set S.

    Z: (n, m) unit-level activations; T: (n,) binary treatment; S: selected neuron indices, in
    selection order; cols: neurons to test (default: all not in S).
    residualize: 'ols' (default), 'crossfit' (paper's recipe, n_folds folds) or None; see the
    module docstring. Strata: make_strata(Z_S, n_strata, min_per_cell, T).
    With S = {} every mode is exactly Welch's two-sample t-test.
    Returns (DataFrame[neuron, tau, se, t, df, p], info dict with the strata actually used).
    """
    Z = np.asarray(Z, dtype=np.float64)
    T = np.asarray(T).astype(np.int64)
    if set(np.unique(T)) != {0, 1}:
        raise ValueError('T must be binary 0/1 with both arms present')
    S = list(S)
    cols = np.array([j for j in range(Z.shape[1]) if j not in set(S)] if cols is None else cols, dtype=np.int64)
    Y = Z[:, cols]
    if S:
        X = Z[:, S]
        cell, k_used, q_used = make_strata(X, n_strata, min_per_cell, T)
    else:
        cell, k_used, q_used = np.zeros(len(T), dtype=np.int64), 0, 1
    if S and residualize == 'ols':
        tau, V, den, n_kept = _two_sample_ols(Y, X, T, cell)
    elif S and residualize == 'crossfit':
        tau, V, den, n_kept = _stratified_welch(_crossfit_residuals(Y, X, T, n_folds, np.random.default_rng(seed)), T, cell)
    elif not S or residualize in (None, False, 'none'):
        tau, V, den, n_kept = _stratified_welch(Y, T, cell)
    else:
        raise ValueError(f'unknown residualize {residualize!r}')
    se, t, df, p = _t_pvalue(tau, V, den)
    table = pd.DataFrame({'neuron': cols, 'tau': tau, 'se': se, 't': t, 'df': df, 'p': p})
    info = {'n_cells': int(len(np.unique(cell))), 'n_stratifiers': k_used, 'bins': q_used, 'n_units_kept': int(n_kept)}
    return table, info


def _reject(p, alpha, correction):
    """Boolean rejections and the p-value threshold used."""
    m = len(p)
    if correction == 'bonferroni':
        return p < alpha / m, alpha / m
    if correction == 'none':
        return p < alpha, alpha
    if correction == 'bh':
        ok = np.flatnonzero(np.sort(p) <= alpha * np.arange(1, m + 1) / m)
        thr = alpha * (ok[-1] + 1) / m if len(ok) else 0.0
        return p <= thr, thr
    raise ValueError(f'unknown correction {correction!r}')


def _pick(table, rejected, select):
    sig = table[rejected]
    if select == 'tau':  # Algorithm 1 line 6: significant set ordered by |tau| (desc)
        return int(sig['neuron'].values[np.argmax(np.abs(sig['tau'].values))])
    if select == 'p':  # smallest p, ties (e.g. at the permutation floor) broken by |t|
        return int(sig['neuron'].values[np.lexsort((-np.abs(sig['t'].values), sig['p'].values))[0]])
    raise ValueError(f'unknown select {select!r}')


def _search(test_fn, keep, alpha, correction, select, max_rounds, keep_tables, S0=()):
    """S0: column indices (nuisance covariates) conditioned on from round 0, never tested/returned."""
    t0 = time.time()
    tested = np.flatnonzero(keep)
    S0 = list(S0)
    S, rounds, tables = [], [], []
    while max_rounds is None or len(S) < max_rounds:
        cols = np.array([j for j in tested if j not in set(S)], dtype=np.int64)
        if len(cols) == 0:
            break
        table, info = test_fn(S0 + S, cols)
        rejected, thr = _reject(table['p'].values, alpha, correction)
        table['significant'] = rejected
        if keep_tables or not tables:
            tables.append(table)
        if not rejected.any():
            break
        j = _pick(table, rejected, select)
        row = table[table['neuron'] == j].iloc[0]
        rounds.append({'round': len(S) + 1, 'neuron': j, 'tau': row['tau'], 'se': row['se'], 't': row['t'],
                       'df': row['df'], 'p': row['p'], 'threshold': thr, 'n_tested': len(cols),
                       'n_significant': int(rejected.sum()), **info})
        S.append(j)
    return {'selected': S, 'rounds': pd.DataFrame(rounds), 'first_round': tables[0] if tables else None,
            'tables': tables if keep_tables else None, 'n_tested': int(keep.sum()),
            'n_dropped': int((~keep).sum()) - len(S0), 'dropped': np.setdiff1d(np.flatnonzero(~keep), S0),
            'n_nuisance': len(S0),
            'elapsed_s': time.time() - t0}


def _with_nuisance(Z, nuisance):
    """-> (Z with the nuisance columns appended, their indices, keep-mask extension)."""
    if nuisance is None:
        return Z, [], None
    N = np.asarray(nuisance, dtype=np.float64)
    N = N[:, None] if N.ndim == 1 else N
    if len(N) != len(Z):
        raise ValueError(f'nuisance has {len(N)} rows, Z has {len(Z)}')
    if not np.isfinite(N).all():
        raise ValueError('nuisance covariates must be finite')
    m = Z.shape[1]
    return np.column_stack([Z, N]), list(range(m, m + N.shape[1])), N.shape[1]


def neural_effect_search(Z, T, alpha=0.05, correction='bonferroni', select='tau', n_strata=2, min_per_cell=3,
                         residualize='ols', n_folds=5, max_rounds=None, min_active=0.01, groups=None,
                         keep_tables=False, seed=0, nuisance=None):
    """Neural Effect Search (Algorithm 1), two-sample.

    Z: (n, m) activations, T: (n,) binary treatment (e.g. het=1 vs wt=0 within one stage).
    Unit of analysis:
      groups=None -> each row is a unit. Pass per-video means for the video/pool-level analysis,
                     or raw frames for the frame-level analysis (pseudo-replication: frames of
                     one video are not independent, so frame-level p-values are far too small;
                     use it only to illustrate the paradox).
      groups=ids  -> rows are frames, first averaged per id (unit_means); T constant per id.
    correction: 'bonferroni' (alpha / m_remaining, paper default), 'bh', 'none'.
    select: 'tau' (paper: largest |tau| among the rejected) or 'p' (smallest p).
    max_rounds: cap on |S|. min_active: neurons active in fewer units are not tested.
    nuisance: optional (n_units,) or (n_units, k) per-unit covariates conditioned on from round 0
      (see the module docstring); with groups, rows are units AFTER averaging (sorted unit ids).
    Returns dict: selected (ordered list), rounds (DataFrame, one row per selected neuron with
    its test stats, threshold, #tested, #significant, strata used), first_round (the naive
    per-neuron multiple test: all neurons, no conditioning; column 'significant'), tables
    (every round, if keep_tables), n_tested, n_dropped, dropped, elapsed_s.
    """
    Z = np.asarray(Z, dtype=np.float64)
    if groups is not None:
        _, Z, T = unit_means(Z, groups, T)
    T = np.asarray(T).astype(np.int64)
    keep = active_neurons(Z, min_active)
    Z, S0, k = _with_nuisance(Z, nuisance)
    if k:
        keep = np.r_[keep, np.zeros(k, dtype=bool)]

    def test_fn(S, cols):
        return neural_effect_test(Z, T, S, cols, n_strata, min_per_cell, residualize, n_folds, seed + len(S) - len(S0))

    return _search(test_fn, keep, alpha, correction, select, max_rounds, keep_tables, S0)


def _signflip_p(e0, h, Q, dof, t_obs, n_perm, rng, chunk=1024):
    """Freedman-Lane sign-flip p-value: flip the null-model residuals e0 (n, m), refit the full
    model (column-space basis Q, contrast functional h) and compare |t*| with |t_obs|.
    RSS* = sum e0^2 - ||Q' (eps * e0)||^2 because flipping leaves sum e0^2 unchanged."""
    ss = (e0 ** 2).sum(0)
    hh = h @ h
    thr = np.abs(t_obs) * (1 - 1e-10)
    exceed, done = np.zeros(e0.shape[1]), 0
    while done < n_perm:
        B = min(chunk, n_perm - done)
        E = rng.choice(np.array([-1.0, 1.0]), size=(B, e0.shape[0]))
        est = (E * h) @ e0
        proj = np.einsum('bn,nk,nm->bkm', E, Q, e0, optimize=True)
        s2 = np.maximum(ss - (proj ** 2).sum(1), 0) / dof
        with np.errstate(divide='ignore', invalid='ignore'):
            tp = np.where(s2 > 0, np.abs(est) / np.sqrt(s2 * hh), 0.0)
        exceed += (tp >= thr).sum(0)
        done += B
    return (1 + exceed) / (1 + n_perm)


def paired_effect_test(Za, Zb, S=(), cols=None, n_strata=2, min_per_cell=3, residualize=True, test='t',
                       n_perm=None, alpha=0.05, seed=0):
    """Paired Neural Effect Test: does neuron j change from stage a to stage b within pools,
    beyond what the already-selected neurons S explain?

    Za, Zb: (n_pools, m), row i = the same pool at stages a and b. Unit = pool.

    Design (the paired analogue of Algorithm 2):
      * Outcome D = Zb - Za per pool: pool random effects cancel, which is the point of pairing.
        For S = {} the test is the one-sample t of E[D_j] = 0.
      * Leakage removal = regression on D_S. The two-sample NET removes found effects by
        comparing arms at equal Z_S. A paired design has no control arm to compare with at equal
        D_S (every pool is "treated"), so stratifying on D_S cannot remove leakage: in a stratum
        where D_1 ~ c, the mean of a leaky D_j is still ~ a * c. What removes it is the
        regression D_j = c_g(i) + b' D_S + e across pools and a test of the intercept: the change
        of neuron j among pools whose found concepts do not change. The estimate is
        mean(D_j) - b' mean(D_S), the mirror of the two-sample tau_j - b' tau_S. The slope b is
        learned from within-pool changes -- exactly the variation that carries the leakage of a
        stage effect -- and is untouched by between-pool (random-effect) covariance, unlike
        residualizing Za and Zb separately on cross-sectional slopes.
        The SE is the exact OLS SE of the intercept, which includes the slope-estimation term
        (mean(D_S)^2 / S_xx); a plug-in one-sample t on residuals would omit it, and since
        mean(D_S) is a strong effect by construction that term is large (we measured round-2
        false discoveries of pure-noise neurons without it). df = n - rank.
        residualize=False keeps stratification only; here that does NOT remove leakage and is
        offered for comparison only.
      * Strata = pooled quantiles of the stage-averaged level L_S = (Za_S + Zb_S)/2, cap as in
        make_strata (min_per_cell pools per cell). Swapping a and b leaves L unchanged, so the
        strata are treatment-agnostic and invariant under the sign-flip null. They standardize
        over the level of the found concepts (effect modification), mirroring the paper's
        post-stratification: tau = sum_g w_g c_g, w_g = n_g / n.
      * test='signflip': Freedman-Lane sign-flip p-value of the same t statistic: fit the null
        model D_j = b0' D_S + e0, flip each pool's e0_i, refit, recompute t. Exact for S = {}
        under exchangeable stages; asymptotically valid otherwise and robust to non-normal
        residuals. The smallest attainable p is 1 / (n_perm + 1), so with Bonferroni n_perm
        must exceed m / alpha; default n_perm = 10 m / alpha.
    Returns (DataFrame[neuron, tau, se, t, df, p], info dict).
    """
    Za, Zb = np.asarray(Za, dtype=np.float64), np.asarray(Zb, dtype=np.float64)
    if Za.shape != Zb.shape:
        raise ValueError('Za and Zb must be aligned pool-for-pool')
    S = list(S)
    cols = np.array([j for j in range(Za.shape[1]) if j not in set(S)] if cols is None else cols, dtype=np.int64)
    D = Zb - Za
    Y, n = D[:, cols], len(D)
    if S:
        cell, k_used, q_used = make_strata((Za[:, S] + Zb[:, S]) / 2, n_strata, min_per_cell)
    else:
        cell, k_used, q_used = np.zeros(n, dtype=np.int64), 0, 1
    G = _onehot(cell)
    w = G.sum(0) / n
    DS = D[:, S] if (S and residualize) else np.zeros((n, 0))
    X = np.column_stack([G, DS])
    tau, V, dof, h = _ols_contrast(Y, X, np.r_[w, np.zeros(DS.shape[1])])
    se, t, df, p = _t_pvalue(tau, V, V ** 2 / dof)
    if test == 'signflip':
        n_perm = int(np.ceil(10 * len(cols) / alpha)) if n_perm is None else n_perm
        e0 = Y - DS @ np.linalg.lstsq(DS, Y, rcond=None)[0] if DS.shape[1] else Y
        U, sv, _ = np.linalg.svd(X, full_matrices=False)
        Q = U[:, sv > sv.max() * 1e-10]
        p = _signflip_p(e0, h, Q, dof, t, n_perm, np.random.default_rng(seed))
    elif test != 't':
        raise ValueError(f'unknown test {test!r}')
    table = pd.DataFrame({'neuron': cols, 'tau': tau, 'se': se, 't': t, 'df': df, 'p': p})
    info = {'n_cells': int(G.shape[1]), 'n_stratifiers': k_used, 'bins': q_used, 'n_units_kept': n}
    return table, info


def paired_effect_search(Za, Zb, alpha=0.05, correction='bonferroni', select='tau', n_strata=2, min_per_cell=3,
                         residualize=True, test='t', n_perm=None, max_rounds=None, min_active=0.01,
                         keep_tables=False, seed=0, nuisance=None):
    """Neural Effect Search (Algorithm 1) with the paired test of paired_effect_test().

    Za, Zb: (n_pools, m) per-pool mean activations at stages a and b (rows aligned by pool),
    e.g. stage 1 -> 2 within the het pools. tau > 0 means the concept increases from a to b.
    Neurons active in < min_active of the 2n stacked observations, or with constant Zb - Za,
    are not tested. Other options and the returned dict as in neural_effect_search().
    nuisance: optional pair (Na, Nb) of (n_pools,) or (n_pools, k) per-pool covariates at stages a and
      b, conditioned on from round 0 like selected neurons: regression on Nb - Na and strata on
      (Na + Nb) / 2 (see paired_effect_test).
    """
    Za, Zb = np.asarray(Za, dtype=np.float64), np.asarray(Zb, dtype=np.float64)
    keep = active_neurons(np.vstack([Za, Zb]), min_active) & ((Zb - Za).std(0) > 1e-8)
    S0 = []
    if nuisance is not None:
        Na, Nb = nuisance
        Za, S0, k = _with_nuisance(Za, Na)
        Zb, _, _ = _with_nuisance(Zb, Nb)
        keep = np.r_[keep, np.zeros(k, dtype=bool)]

    def test_fn(S, cols):
        return paired_effect_test(Za, Zb, S, cols, n_strata, min_per_cell, residualize, test, n_perm, alpha,
                                  seed + len(S) - len(S0))

    return _search(test_fn, keep, alpha, correction, select, max_rounds, keep_tables, S0)
