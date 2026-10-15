"""
Direct-Indirect Neural Effect Search (DINES), the successor of NES (src/eci/nes.py).

Theory: docs/eci/dines_theory.md (commit 7dd7c26; labels below refer to that version): Section 2.5 (algorithm),
Section 2.6 (Theorem 2, Corollaries 1-2), Sections 2.6-2.8 (margin delta, Theorem 3), Section 2.9 (overlap) and
Section 3 (tests in practice).

Problem. A randomized binary treatment T, a fixed dictionary Z (n units x m atoms, e.g. SAE codes). The target
is the DIRECT set: atoms j whose mean shift under T cannot be explained away by any OVERLAP set of other atoms,
    H0(j | A):  E[Z^j | T, Z^A] = E[Z^j | Z^A]   false for every A (other atoms) with 0 < pi_A < 1,
pi_A = P(T = 1 | Z^A). Under the latent-factor model of the theory (Theorem 1) this is the set of principal atoms of
the children of T: mediated atoms (T -> C -> s) and co-parents of children ("spouses", W -> C <- T) are excluded. NES
targets something closer to the Markov boundary (it keeps spouses; Proposition 1) and has no backward step.

Algorithm (theory Section 2.5). A conditioning set A is SKIPPED when more than overlap_frac of the units have an
estimated pi_A outside [overlap_eps, 1 - overlap_eps] (defaults 0.02, 0.10).
  1. Forward phase (IAMB grow): S = {}, K = {}; repeat: p_j(S) for every atom j not in S u K; take the atom j* with
     the SMALLEST p-value (not the largest raw effect: atoms have different scales); stop if p > alpha / |remaining|
     (Bonferroni gate). If S u {j*} is skipped, j* (nearly) reveals the treatment: it goes to K, the set-aside atoms,
     and is never conditioned on in later rounds (otherwise it would stall the search, as NES's bag neuron did on the
     mice data, Section 2.9); else S = S u {j*}. F = S u K.
  2. Terminal phase: every j in F is judged on subsets A of F \ {j} at level alpha / m (m = atoms entering the
     analysis, after the prefilter); judgements are simultaneous.
       DROPPED       p_j(A) > alpha / m at some non-skipped A,
       UNDETERMINED  not dropped, and p_j(A) > alpha / m at some skipped A,
       KEPT          otherwise.
     REPORTED = KEPT u UNDETERMINED (the estimate of the direct set; Theorem 2 covers the union). Undetermined atoms
     may be changed directly or only through a treatment-revealing factor; no sample size separates the two
     (Proposition 4).
  (3. The indirect output, Section 2.12, is not implemented here.)

The test (theory Section 3.1): the generalized covariance measure (GCM, Shah & Peters 2020). Residualize T and Z^j on
the conditioning variables V_A = (Z^A, W): R_i = (T_i - pi(V_A,i)) (Z^j_i - mu_j(V_A,i)), statistic sqrt(n) mean(R) /
sd(R), two-sided normal p-value. E[R] is the numerator of rho_j|A (Lemma 1.3). Z^A is post-treatment, so pi_A is NOT
known by design and is learned (only A = {} without covariates uses the share of treated units).
  * forward phase: one LINEAR GCM, vectorized over all candidate atoms at once: every atom is residualized on
    [1, Z^S, 1[Z^S > 0], W] by one orthonormal basis (in sample), the propensity is an L2 logistic regression on the
    same columns. Only consistency is needed there (Section 2.5, "two speeds"; Section 3.3). The overlap check of
    S u {j*} uses the same logistic model, cross-fitted (n_folds folds).
  * terminal phase: a CROSS-FITTED flexible GCM on the few atoms in F: n_folds folds (whole clusters when clusters
    are given), nuisance models fitted on the other folds. learner='ridge' (default): ridge regression (mu) and L2
    logistic regression (pi) on [X, 1[X > 0], pairwise products of X and of the indicators] of the conditioning atoms
    X = Z^A, plus W linearly; 'hgb': sklearn histogram gradient boosting (slower).
  The indicator columns let the nuisance models represent the point mass at zero of sparse codes (Section 3.1, "do
  not binarize conditioning atoms").

Relevance margin delta (theory Sections 2.6-2.8, Theorem 3; default 0). Effect size = the partial correlation
    rho_j,A = mean((T - pi)(Z^j - mu)) / sqrt(mean((T - pi)^2) mean((Z^j - mu)^2)),
and with delta > 0 every test (both phases) is the test of H0: |rho| <= delta, p = Phi((delta - |rho^|) / se) +
Phi((-delta - |rho^|) / se), se from the delta method (influence function of the ratio, which includes the
estimation error of the two denominators). At delta = 0 the plain GCM statistic is used.

Pre-treatment covariates W (optional): appended to every conditioning set in both phases (they add power, not
validity, in a randomized experiment; Section 3.2, Proposition 10). Clusters (optional): cluster-robust standard
errors (sums of the per-unit influence values within a cluster) with a t_{G-1} reference, and cluster-level folds
(Section 3.4). Prefilter: atoms active (|z| > 1e-8) in fewer than min_active of the units or with zero variance are
not tested and do not count in m (a screen on Z only, never on T; nes.active_neurons; Section 3.3).

Cost: the terminal phase tests up to 2^|F| conditioning sets. Order: {} first, then F \ {j} for each j (under the
theory's model the full complement separates every atom outside the blanket, Lemma 4), then the remaining subsets by
increasing size. Each set is fitted once and tests every still-undropped atom outside it. If 2^|F| > max_subsets,
only the first max_subsets sets of that order are tested and result['terminal_complete'] = False.

Functions:
    linear_gcm      vectorized forward-phase test of many atoms given one conditioning set
    crossfit_gcm    cross-fitted flexible test of a few atoms given one conditioning set (terminal phase)
    dines           the full search (entry point)
"""

import itertools
import time

import numpy as np
from scipy import stats

from .nes import active_neurons

ZERO_TOL = 1e-8   # an activation with |z| <= ZERO_TOL counts as "not firing"
RES_REL_TOL = 1e-6  # residual RMS below this fraction of the atom's RMS: exact fit, untestable (p = 1)


# ----------------------------------------------------------------------------------------------------------------
# the GCM statistic (shared by both phases)
# ----------------------------------------------------------------------------------------------------------------

def _cluster_matrix(groups):
    """(n, G) one-hot cluster membership, or None."""
    if groups is None:
        return None
    codes = np.unique(np.asarray(groups), return_inverse=True)[1].reshape(-1)
    G = np.zeros((len(codes), codes.max() + 1))
    G[np.arange(len(codes)), codes] = 1.0
    return G


def _se_of_mean(psi, Gm):
    """SE of the mean of the columns of psi (n, k; already centred): sqrt(sum_i psi_i^2) / n, or with clusters
    sqrt(sum_g (sum_{i in g} psi_i)^2) / n (cluster-robust, no small-sample factor)."""
    n = len(psi)
    if Gm is None:
        return np.sqrt((psi ** 2).sum(0)) / n
    return np.sqrt(((Gm.T @ psi) ** 2).sum(0)) / n


def _gcm_from_residuals(rT, RZ, delta=0.0, Gm=None, scale=None):
    """GCM / relevance test of every column of RZ (n, k) against rT (n,).

    Returns dict of arrays (k,): rho (partial correlation), stat (z or t; at delta > 0 (|rho| - delta) / se),
    logp (natural log of the p-value), se_rho, testable. scale: the atoms' RMS before residualization (an exact
    fit, residual RMS <= RES_REL_TOL * scale, is untestable: p = 1)."""
    n, k = RZ.shape
    R = rT[:, None] * RZ
    a = R.mean(0)
    b = (rT ** 2).mean()
    c = (RZ ** 2).mean(0)
    with np.errstate(divide='ignore', invalid='ignore'):
        s = np.sqrt(b * c)
        rho = np.where(s > 0, a / s, 0.0)
        ok = (s > 0) & np.isfinite(s)
        if scale is not None:
            ok &= np.sqrt(c) > RES_REL_TOL * np.asarray(scale)
        if delta == 0:  # plain GCM: sqrt(n) mean(R) / sd(R)
            se_a = _se_of_mean(R - a, Gm)
            ok &= se_a > 0
            stat = np.where(ok, a / se_a, 0.0)
            se_rho = np.where(ok, se_a / s, np.inf)
        else:  # delta method for rho = a / sqrt(b c)
            psi = (R - a) / s - (a / (2 * s)) * ((rT[:, None] ** 2 - b) / b + (RZ ** 2 - c) / c)
            se_rho = _se_of_mean(np.where(ok, psi, 0.0), Gm)
            ok &= se_rho > 0
            stat = np.where(ok, (np.abs(rho) - delta) / se_rho, -np.inf)
    df = None if Gm is None else Gm.shape[1] - 1
    dist = stats.norm if df is None else stats.t(df)
    if delta == 0:
        logp = np.where(ok, np.log(2.0) + dist.logsf(np.abs(stat)), 0.0)
    else:
        r = np.abs(rho)
        with np.errstate(divide='ignore', invalid='ignore'):
            logp = np.where(ok, np.logaddexp(dist.logcdf((delta - r) / se_rho), dist.logcdf((-delta - r) / se_rho)), 0.0)
    logp = np.minimum(logp, 0.0)
    return {'rho': rho, 'stat': stat, 'logp': logp, 'se_rho': se_rho, 'testable': ok}


# ----------------------------------------------------------------------------------------------------------------
# design columns and nuisance models
# ----------------------------------------------------------------------------------------------------------------

def _lin_columns(X):
    """Forward-phase columns of the conditioning atoms X (n, k): X and, for atoms that have zeros and non-zeros,
    the firing indicator 1[|X| > ZERO_TOL]."""
    if X.shape[1] == 0:
        return X
    ind = (np.abs(X) > ZERO_TOL).astype(np.float64)
    keep = (ind.min(0) == 0) & (ind.max(0) == 1)
    return np.column_stack([X, ind[:, keep]])


MAX_INTERACT = 6  # pairwise products only among the first MAX_INTERACT conditioning atoms (forward order)


def _flex_columns(X):
    """Terminal-phase ('ridge' learner) columns: X, firing indicators, and the pairwise products of the values and of
    the indicators among the first MAX_INTERACT columns (columns come in forward order, i.e. the atoms most predictive
    of T first; the products let the propensity represent collider structure such as P(T | child, spouse))."""
    k = X.shape[1]
    if k == 0:
        return X
    ind = (np.abs(X) > ZERO_TOL).astype(np.float64)
    keep = (ind.min(0) == 0) & (ind.max(0) == 1)
    cols = [X, ind[:, keep]]
    for a, b in itertools.combinations(range(min(k, MAX_INTERACT)), 2):
        cols.append((X[:, a] * X[:, b])[:, None])
        if keep[a] and keep[b]:
            cols.append((ind[:, a] * ind[:, b])[:, None])
    return np.column_stack(cols)


def _orth_basis(V):
    """Orthonormal basis of span([1, V]) (rank-revealing SVD; constant or duplicated columns are harmless)."""
    A = np.column_stack([np.ones(len(V)), V])
    U, s, _ = np.linalg.svd(A, full_matrices=False)
    return U[:, s > s[0] * 1e-10]


def _standardize(Xtr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    return (Xtr - mu) / sd, (Xte - mu) / sd


def _fit_propensity(Xtr, Ttr, Xte, learner, C):
    """P(T = 1 | x) fitted on (Xtr, Ttr), predicted at Xte. A training set with one arm only, or no columns, gives the
    training share of treated units."""
    p0 = Ttr.mean()
    if Xtr.shape[1] == 0 or p0 in (0.0, 1.0):
        return np.full(len(Xte), p0)
    if learner == 'hgb':
        from sklearn.ensemble import HistGradientBoostingClassifier
        mdl = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=15, min_samples_leaf=20,
                                             early_stopping=False)
        return mdl.fit(Xtr, Ttr).predict_proba(Xte)[:, 1]
    from sklearn.linear_model import LogisticRegression
    Xs, Xt = _standardize(Xtr, Xte)
    mdl = LogisticRegression(C=C, solver='newton-cholesky', max_iter=200)
    return mdl.fit(Xs, Ttr).predict_proba(Xt)[:, 1]


RIDGE_ALPHAS = np.logspace(-3, 4, 8)


def _fit_regression(Xtr, Ytr, Xte, learner):
    """E[Y | x] for every column of Ytr (n_tr, k), fitted on Xtr, predicted at Xte."""
    if Xtr.shape[1] == 0:
        return np.broadcast_to(Ytr.mean(0), (len(Xte), Ytr.shape[1])).copy()
    if learner == 'hgb':
        from sklearn.ensemble import HistGradientBoostingRegressor
        out = np.empty((len(Xte), Ytr.shape[1]))
        for c in range(Ytr.shape[1]):
            mdl = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, max_leaf_nodes=15,
                                                min_samples_leaf=20, early_stopping=False)
            out[:, c] = mdl.fit(Xtr, Ytr[:, c]).predict(Xte)
        return out
    from sklearn.linear_model import RidgeCV
    Xs, Xt = _standardize(Xtr, Xte)
    mdl = RidgeCV(alphas=RIDGE_ALPHAS, alpha_per_target=True)
    return mdl.fit(Xs, Ytr).predict(Xt).reshape(len(Xte), -1)


def _make_folds(n, n_folds, groups, rng):
    """Fold id per unit; whole clusters go to one fold when groups are given."""
    if groups is None:
        return rng.permutation(np.arange(n) % n_folds)
    codes = np.unique(np.asarray(groups), return_inverse=True)[1].reshape(-1)
    g_fold = rng.permutation(np.arange(codes.max() + 1) % n_folds)
    return g_fold[codes]


# ----------------------------------------------------------------------------------------------------------------
# the two tests
# ----------------------------------------------------------------------------------------------------------------

def linear_gcm(Z, T, S=(), cols=None, W=None, delta=0.0, groups=None, prop_C=10.0, chunk=2048):
    """Forward-phase test of every atom in `cols` given the conditioning set S (theory Section 7.6).

    Atoms are residualized (in sample) on [1, Z^S, 1[Z^S > 0], W] by one orthonormal basis; T on the same columns by
    an L2 logistic regression (C = prop_C on standardized columns), or the share of treated units when there are no
    columns. Vectorized over the atoms, `chunk` columns at a time. Returns a dict of arrays over cols
    (rho, stat, logp, se_rho, testable) plus 'ess' = sum pi (1 - pi).
    """
    Z = np.asarray(Z)
    T = np.asarray(T, dtype=np.float64)
    S = list(S)
    cols = np.array([j for j in range(Z.shape[1]) if j not in set(S)] if cols is None else cols, dtype=np.int64)
    V = _lin_columns(np.asarray(Z[:, S], dtype=np.float64))
    if W is not None:
        V = np.column_stack([V, W])
    pi = _fit_propensity(V, T, V, 'logistic', prop_C)
    rT = T - pi
    Q = _orth_basis(V)
    Gm = _cluster_matrix(groups)
    out = {k: np.empty(len(cols)) for k in ('rho', 'stat', 'logp', 'se_rho')}
    out['testable'] = np.empty(len(cols), dtype=bool)
    for s in range(0, len(cols), chunk):
        Zc = np.asarray(Z[:, cols[s:s + chunk]], dtype=np.float64)
        RZ = Zc - Q @ (Q.T @ Zc)
        res = _gcm_from_residuals(rT, RZ, delta, Gm, scale=np.sqrt((Zc ** 2).mean(0)))
        for k in out:
            out[k][s:s + chunk] = res[k]
    out['cols'] = cols
    out['ess'] = float((pi * (1 - pi)).sum())
    return out


def crossfit_gcm(Z, T, A, cols, W=None, delta=0.0, groups=None, folds=None, learner='ridge', prop_C=1.0):
    """Terminal-phase test of the atoms `cols` given the conditioning set A (theory Section 7.2): cross-fitted
    nuisance models (fold ids `folds`), GCM / relevance statistic on the out-of-fold residuals.

    A = {} without W uses the share of treated units and the atom means (no fitting). Returns the dict of
    _gcm_from_residuals plus 'pi' (out-of-fold propensity, n) and 'ess' = sum pi (1 - pi)."""
    Z = np.asarray(Z)
    T = np.asarray(T, dtype=np.float64)
    Y = np.asarray(Z[:, list(cols)], dtype=np.float64)
    X = np.asarray(Z[:, list(A)], dtype=np.float64)
    F = _flex_columns(X) if learner == 'ridge' else np.column_stack([X, (np.abs(X) > ZERO_TOL)]) if X.shape[1] else X
    if W is not None:
        F = np.column_stack([F, W])
    if F.shape[1] == 0:
        pi = np.full(len(T), T.mean())
        RZ = Y - Y.mean(0)
    else:
        pi, mu = np.empty(len(T)), np.empty_like(Y)
        for f in np.unique(folds):
            te, tr = folds == f, folds != f
            pi[te] = _fit_propensity(F[tr], T[tr], F[te], learner, prop_C)
            mu[te] = _fit_regression(F[tr], Y[tr], F[te], learner)
        RZ = Y - mu
    res = _gcm_from_residuals(T - pi, RZ, delta, _cluster_matrix(groups), scale=np.sqrt((Y ** 2).mean(0)))
    res['pi'] = pi
    res['ess'] = float((pi * (1 - pi)).sum())
    return res


# ----------------------------------------------------------------------------------------------------------------
# the search
# ----------------------------------------------------------------------------------------------------------------

def _forward_skipped(Z, T, S, W, prop_C, overlap_eps, overlap_frac, folds):
    """Overlap rule for a forward conditioning set S: cross-fitted logistic propensity on the forward columns.
    Returns (skipped, share of units with pi outside [eps, 1 - eps])."""
    V = _lin_columns(np.asarray(Z[:, S], dtype=np.float64))
    if W is not None:
        V = np.column_stack([V, W])
    T = np.asarray(T, dtype=np.float64)
    pi = np.empty(len(T))
    for f in np.unique(folds):
        te, tr = folds == f, folds != f
        pi[te] = _fit_propensity(V[tr], T[tr], V[te], 'logistic', prop_C)
    bad = float(((pi < overlap_eps) | (pi > 1 - overlap_eps)).mean())
    return bad > overlap_frac, bad


def _forward(Z, T, cand, W, delta, groups, alpha, gate, max_forward, prop_C, chunk, keep_first, overlap_eps,
             overlap_frac, folds, verbose=False):
    """Forward phase with the set-aside rule: returns (S, K, path, first)."""
    S, K, path, first = [], [], [], None
    t0 = time.time()
    while len(S) + len(K) < len(cand) and (max_forward is None or len(S) + len(K) < max_forward):
        rem = np.array([j for j in cand if j not in set(S) | set(K)], dtype=np.int64)
        res = linear_gcm(Z, T, S, rem, W, delta, groups, prop_C, chunk)
        if keep_first and first is None:
            first = {'atom': rem, 'rho': res['rho'], 'logp': res['logp']}
        thr = alpha / len(rem) if gate == 'bonferroni' else alpha
        # smallest p; ties (e.g. p underflowing to the same value) broken by the larger statistic
        k = np.lexsort((-res['stat'], res['logp']))[0]
        n_sig = int((res['logp'] <= np.log(thr)).sum())
        rec = {'round': len(path) + 1, 'atom': int(rem[k]), 'p': float(np.exp(res['logp'][k])),
               'log_p': float(res['logp'][k]), 'rho': float(res['rho'][k]), 'stat': float(res['stat'][k]),
               'threshold': thr, 'n_tested': len(rem), 'n_significant': n_sig, 'ess': res['ess'],
               'added': bool(res['logp'][k] <= np.log(thr)), 'set_aside': False, 'frac_extreme_pi': None}
        path.append(rec)
        if not rec['added']:
            break
        skipped, bad = _forward_skipped(Z, T, S + [int(rem[k])], W, prop_C, overlap_eps, overlap_frac, folds)
        rec['frac_extreme_pi'] = bad
        if verbose:
            print(f"  forward round {rec['round']}: atom {rec['atom']} p {rec['p']:.3g} rho {rec['rho']:.3f} "
                  f"extreme-pi share {bad:.3f} ({time.time() - t0:.1f}s)", flush=True)
        if skipped:  # j* (with S) reveals the treatment: report it, never condition on it
            rec['set_aside'] = True
            K.append(int(rem[k]))
        else:
            S.append(int(rem[k]))
    return S, K, path, first


def _subset_order(F, max_subsets):
    """Conditioning sets of the terminal phase, in test order: {}, then F \\ {j} for each j, then the rest by size.
    Returns (list of tuples, complete flag)."""
    k = len(F)
    first = [()] + ([tuple(a for a in F if a != j) for j in F] if k > 1 else [])  # tuples keep forward order
    seen = set(first)
    order = list(first)
    for size in range(1, k - 1):  # every proper subset of F is a conditioning set for some j
        for A in itertools.combinations(F, size):
            if A in seen:
                continue
            if max_subsets is not None and len(order) >= max_subsets:
                return order, False
            order.append(A)
            seen.add(A)
    return order, True


def _terminal(Z, T, F, W, delta, groups, level, learner, n_folds, prop_C, overlap_eps, overlap_frac,
              max_subsets, folds, verbose=False):
    t0 = time.time()
    order, complete = _subset_order(F, max_subsets)
    log_level = np.log(level)
    atoms = {j: {'atom': int(j), 'status': 'kept', 'max_p': 0.0, 'argmax_A': None, 'n_sets_tested': 0,
                 'fail_skipped': [], 'sepset': None, 'p_empty': None, 'rho_empty': None, 'p_rest': None,
                 'rho_rest': None, 'min_ess': None} for j in F}
    sets = []
    for A in order:
        targets = [j for j in F if j not in A and atoms[j]['status'] != 'dropped']
        if not targets:
            continue
        res = crossfit_gcm(Z, T, A, targets, W, delta, groups, folds, learner, prop_C)
        pi = res['pi']
        bad = float(((pi < overlap_eps) | (pi > 1 - overlap_eps)).mean())
        skipped = bad > overlap_frac
        if verbose:
            print(f'  terminal set {len(sets) + 1}/{len(order)} |A| {len(A)} targets {len(targets)} skipped {skipped} '
                  f'({time.time() - t0:.1f}s)', flush=True)
        sets.append({'A': [int(a) for a in A], 'skipped': skipped, 'frac_extreme_pi': bad, 'ess': res['ess'],
                     'n_targets': len(targets)})
        for i, j in enumerate(targets):
            rec = atoms[j]
            lp = float(res['logp'][i])
            rec['n_sets_tested'] += 1
            rec['min_ess'] = res['ess'] if rec['min_ess'] is None else min(rec['min_ess'], res['ess'])
            if len(A) == 0:
                rec['p_empty'], rec['rho_empty'] = float(np.exp(lp)), float(res['rho'][i])
            if set(A) == set(F) - {j}:
                rec['p_rest'], rec['rho_rest'] = float(np.exp(lp)), float(res['rho'][i])
            if lp > log_level:  # not significant at alpha / m given A
                if skipped:
                    rec['fail_skipped'].append([int(a) for a in A])
                else:
                    rec['status'], rec['sepset'] = 'dropped', [int(a) for a in A]
            if not skipped and np.exp(lp) >= rec['max_p']:
                rec['max_p'], rec['argmax_A'] = float(np.exp(lp)), [int(a) for a in A]
    for rec in atoms.values():
        if rec['status'] == 'kept' and rec['fail_skipped']:
            rec['status'] = 'undetermined'
    return atoms, sets, complete


def dines(Z, T, *, alpha=0.05, delta=0.0, W=None, clusters=None, min_active=0.01, gate='bonferroni',
          learner='ridge', n_folds=5, prop_C_forward=10.0, prop_C_terminal=1.0, overlap_eps=0.02, overlap_frac=0.10,
          max_forward=30, max_subsets=4096, keep_first=False, seed=0, chunk=2048, verbose=False):
    """Direct-Indirect Neural Effect Search (direct set only).

    Z: (n, m) atom activations (one row per unit, e.g. image, frame or video); T: (n,) binary 0/1 treatment;
    W: optional (n,) or (n, p) pre-treatment covariates, added to every conditioning set; clusters: optional (n,)
    cluster ids (cluster-robust SE with a t_{G-1} reference, cluster-level folds).
    alpha: level; terminal tests at alpha / m (m = atoms after the prefilter), forward gate alpha / |remaining|
      (gate='bonferroni') or alpha (gate='none').
    delta: relevance margin on the partial-correlation scale (0 = the exact null; see the module docstring).
    learner: terminal nuisance models, 'ridge' (ridge + L2 logistic on value, indicator and pairwise columns) or 'hgb'.
    n_folds: cross-fitting folds of the terminal phase. overlap_eps, overlap_frac: the overlap rule (module docstring).
    max_forward: cap on |F| (None = no cap); max_subsets: cap on the conditioning sets of the terminal phase.
    min_active: prefilter on activity rate (Z only).

    Returns dict:
      selected      the REPORTED set = kept u undetermined (column indices of Z), in forward order
      kept          atoms rejected at every tested subset of F \\ {j}
      undetermined  atoms whose only non-rejections were at sets skipped by the overlap rule (flag)
      set_aside     K: forward atoms that (with S) revealed the treatment and were never conditioned on
      forward_set   F = S u K in order of entry; forward = one dict per round (atom, p, rho, stat, threshold,
                    set_aside, share of extreme pi, ...)
      terminal      one dict per atom of F (status kept/dropped/undetermined, max_p over non-skipped sets, the set
                    where it was attained, sepset = the first set that dropped it, p and rho given {} and given
                    F \\ {j}, min effective sample size over its sets, ...)
      sets          one dict per conditioning set tested (A, skipped, share of extreme pi, ESS)
      terminal_complete, m, level, n_dropped (prefilter), elapsed_s, elapsed_forward_s, elapsed_terminal_s,
      first_round (if keep_first: rho and log p of every tested atom given {}).
    """
    t0 = time.time()
    Z = np.asarray(Z)
    T = np.asarray(T).astype(np.int64)
    if set(np.unique(T)) != {0, 1}:
        raise ValueError('T must be binary 0/1 with both arms present')
    if W is not None:
        W = np.asarray(W, dtype=np.float64)
        W = W[:, None] if W.ndim == 1 else W
        if len(W) != len(T) or not np.isfinite(W).all():
            raise ValueError('W must be finite with one row per unit')
    if gate not in ('bonferroni', 'none'):
        raise ValueError(f'unknown gate {gate!r}')
    keep = active_neurons(Z, min_active)
    cand = np.flatnonzero(keep)
    m = len(cand)
    if m == 0:
        raise ValueError('no atom passes the activity prefilter')
    level = alpha / m
    rng = np.random.default_rng(seed)
    folds = _make_folds(len(T), n_folds, clusters, rng)
    S, K, path, first = _forward(Z, T, cand, W, delta, clusters, alpha, gate, max_forward, prop_C_forward, chunk,
                                 keep_first, overlap_eps, overlap_frac, folds, verbose)
    F = [r['atom'] for r in path if r['added']]  # S u K in order of entry
    t1 = time.time()
    atoms, sets, complete = _terminal(Z, T, F, W, delta, clusters, level, learner, n_folds, prop_C_terminal,
                                      overlap_eps, overlap_frac, max_subsets, folds, verbose)
    t2 = time.time()
    out = {'selected': [j for j in F if atoms[j]['status'] in ('kept', 'undetermined')],
           'kept': [j for j in F if atoms[j]['status'] == 'kept'],
           'undetermined': [j for j in F if atoms[j]['status'] == 'undetermined'],
           'set_aside': list(K), 'forward_set': list(F), 'forward': path, 'terminal': [atoms[j] for j in F],
           'sets': sets, 'terminal_complete': complete, 'm': m, 'level': level, 'n_dropped': int((~keep).sum()),
           'delta': delta, 'forward_capped': max_forward is not None and len(F) >= max_forward,
           'elapsed_s': t2 - t0, 'elapsed_forward_s': t1 - t0, 'elapsed_terminal_s': t2 - t1}
    if keep_first:
        out['first_round'] = first
    return out
