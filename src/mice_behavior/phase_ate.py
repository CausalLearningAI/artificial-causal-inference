"""Estimators for the mice within-pool PHASE-TRANSITION ATE, in one place.

THE ESTIMAND
============
The mean WITHIN-POOL change in a behaviour rate across one phase transition, for one exposure.
The unit of analysis is the POOL: a pool is one cage of four littermates filmed through all
three phases of both exposures, so every pool supplies its own control and cage, genotype, sex,
line, date and annotator cancel by construction. Consecutive transitions only, in the order the
experiment ran (H -> O -> P): P - H is the sum of the other two, so quoting all three
triple-counts the same pools.

The two exposures are DISTINCT treatments and are never pooled -- they carry opposite signs on
nose-to-tail, so averaging them returns approximately zero.

THREE ESTIMATORS, AND WHAT EACH ONE COSTS
=========================================
Write D_Y for a pool's true phase difference and D_f for its predicted one.

  classical   mean(D_Y) over every pool whose HUMAN difference is defined. No model. Unbiased,
              and the only one of the three that needs no assumption beyond the design. Limited
              to 24 of v1's 72 pools -- fewer on `decay`, where a phase with no bout has no
              onset -- and unavailable on v2, which has no labels at all.
              Its pool set NEVER depends on the predictor: see PoolDeltas.

  PPI++       lam * mean_N(D_f)  +  mean_n(D_Y - lam * D_f)
              Unbiased for ANY predictor: whatever lam is and however wrong f is, the second
              term subtracts exactly what the first added. f moves only the VARIANCE. The
              power-tuned lam = Cov_n(D_Y, D_f) / (Var_{n+N}(D_f) * (1 + n/N)) is what makes a
              miscalibrated model harmless -- lam absorbs the scale, so calibration happens in
              the one place where getting it wrong costs variance instead of validity. Note the
              subscripts: the covariance can only be formed on the n LABELLED pools, but the
              variance is taken over ALL n+N of them, which is both what PPI++ specifies and
              what stops the ratio exploding at n=2. See `ppi` for the full argument.
              REQUIRES out-of-fold predictions on the labelled pools (see below).

  PPCI        b * mean(D_f) over ALL pools, labelled and unlabelled alike, with
              b = Cov(D_Y, D_f) / Var(D_f) fitted on the labelled pools.
              There is NO rectifier. This is the plug-in estimate, and it is the only thing
              available on v2, where nothing is labelled. It trades bias for variance: the
              interval is narrower than classical because it uses every pool, but the narrowing
              is not free -- it is exactly the term PPI would have subtracted. Report it as an
              extrapolation, never as an estimate with guarantees.

WHY ONLY THE SLOPE, AND NEVER AN INTERCEPT
==========================================
The estimand is a DIFFERENCE of two phases within the same pool. Under any affine calibration
Y = a + b f + e the intercept cancels: D_Y = b D_f. So the calibration this analysis needs has
exactly ONE free parameter, fitted on labelled pools, and no additive offset is identifiable
from -- or relevant to -- a within-pool contrast. That is worth stating because the raw
predictions overstate occupancy several-fold; none of that offset reaches the estimate.

CROSS-FITTING IS NOT OPTIONAL FOR PPI
=====================================
f must not have been trained on the labelled pools that enter the rectifier. If it was, f fits
Y in sample, (D_Y - lam D_f) shrinks toward zero, and the SE is understated -- the interval
undercovers while looking excellent. `assert_out_of_fold` enforces it rather than trusting the
caller.

The matching requirement, which is easy to miss: the labelled and unlabelled predictions must
come from the SAME predictor, or the rectifier does not cancel. With K fold models the correct
construction (cross-prediction) is f_{k(j)} for a labelled pool j held out by fold k(j), and the
AVERAGE of all K models for an unlabelled pool. Both then have expectation
(1/K) sum_k E[f_k], so the estimator stays unbiased. Using a single fold's model on the
unlabelled pools while the labelled side carries out-of-fold predictions mixes two different
functions and reintroduces a bias of lam * (E[f_single] - E[f_oof]).

INTERVALS
=========
All three use the SAME t_{n-1} quantile on the same pool-clustered scale, because a shrinkage
percentage compared across methods is meaningless if one of them used 1.96 and another 2.07.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

__all__ = ['PoolDeltas', 'Estimate', 'pool_deltas', 'classical', 'ppi', 'ppci',
           'assert_out_of_fold', 'scale_bootstrap', 'TRANSITIONS']

# Consecutive only, in experimental order. P-H is their sum, not a third contrast.
TRANSITIONS = (('H', 'O'), ('O', 'P'))


@dataclass
class PoolDeltas:
    """One row per pool: its true and/or predicted phase difference for one cell.

    TWO POOL SETS, AND WHY THEY ARE NOT INTERCHANGEABLE
    ===================================================
    A difference can be missing on either side independently. On the `decay` unit -- the mean
    bout-onset minute -- a phase with no bout in the window has NO defined onset, so a pool can
    carry a human difference and no predicted one, or the reverse. Every pool that has both
    phases of the transition gets a row here, NaN and all, and each estimator selects the set
    its own arithmetic needs:

        has_true      the human difference is defined     -> `classical`, and nothing else
        paired        BOTH are defined                    -> PPI's rectifier, r, PPCI's scale k
        has_pred      the predicted difference is defined -> PPCI's plug-in mean
        unlabelled    predicted defined, human not        -> PPI's unlabelled arm

    An earlier version dropped a pool inside `pool_deltas` whenever the PREDICTION was
    undefined, which silently handed `classical` the `paired` set -- making a HUMAN-ONLY,
    model-free number move when the loaded predictor changed. There is deliberately no single
    `labelled` mask left to reach for by accident: pick the one the estimator actually needs.
    """
    pools: np.ndarray
    d_true: np.ndarray          # NaN where the pool is unlabelled OR the difference is undefined
    d_pred: np.ndarray          # NaN where the model gives no defined difference

    @property
    def has_true(self):
        """Pools with a defined HUMAN difference. The classical estimator's set."""
        return np.isfinite(self.d_true)

    @property
    def has_pred(self):
        """Pools with a defined PREDICTED difference. PPCI's plug-in set."""
        return np.isfinite(self.d_pred)

    @property
    def paired(self):
        """Pools where both sides are defined. Anything comparing f to Y needs this one."""
        return self.has_true & self.has_pred

    @property
    def unlabelled(self):
        """Predicted difference defined, human one not. PPI's unlabelled arm.

        On `decay` this is NOT the same thing as "never annotated": an annotated pool whose
        annotator recorded no bout in one of the two phases has no defined human onset, so it
        lands here too.
        """
        return self.has_pred & ~self.has_true

    def r(self) -> float:
        """Within-cell correlation of true and predicted differences on the PAIRED pools.

        This -- not frame AP -- is what PPI's variance reduction is a function of. It is also
        much lower than the value obtained by pooling cells together, because pooling adds
        between-cell signal that a single-cell estimate cannot use.
        """
        m = self.paired
        if m.sum() < 3 or np.ptp(self.d_pred[m]) == 0 or np.ptp(self.d_true[m]) == 0:
            return float('nan')
        return float(np.corrcoef(self.d_true[m], self.d_pred[m])[0, 1])


@dataclass
class Estimate:
    method: str
    est: float
    lo: float
    hi: float
    n_lab: int
    n_unlab: int
    extra: dict = field(default_factory=dict)

    @property
    def width(self) -> float:
        return self.hi - self.lo

    def as_dict(self) -> dict:
        return {'method': self.method, 'est': _f(self.est), 'lo': _f(self.lo), 'hi': _f(self.hi),
                'n_lab': int(self.n_lab), 'n_unlab': int(self.n_unlab),
                **{k: _f(v) if isinstance(v, float) else v for k, v in self.extra.items()}}


def _f(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 6)


def _t(n: int) -> float:
    """Two-sided 95% quantile with n-1 df. Shared by every method on purpose."""
    return float(stats.t.ppf(0.975, max(n - 1, 1)))


def pool_deltas(df, true_col, pred_col, odour, transition) -> PoolDeltas:
    """Collapse per-observation rows to one difference per pool, for ONE exposure cell.

    `df` needs columns pool, phase, odor and the two value columns. A pool contributes a row
    only if it has BOTH phases of the transition; `true_col=None` marks the frame as unlabelled.

    EITHER difference may come back NaN -- on `decay`, a phase with no bout has no onset -- and
    NOTHING is filtered on that here, on purpose. Which side has to be defined is a property of
    the estimator, not of the data, so the masks on PoolDeltas decide it. Filtering here instead
    would impose the strictest estimator's requirement on all of them, and in particular would
    make the model-free `classical` estimate depend on the predictor.
    """
    x, y = transition
    sub = df[df.odor == odour]
    keys, dy, dp = [], [], []
    for pool, g in sub.groupby('pool', sort=True):
        m = g.drop_duplicates('phase').set_index('phase')
        if x not in m.index or y not in m.index:
            continue
        keys.append(pool)
        dy.append(m.loc[y, true_col] - m.loc[x, true_col] if true_col else np.nan)
        dp.append((m.loc[y, pred_col] - m.loc[x, pred_col]) if pred_col else np.nan)
    return PoolDeltas(np.array(keys, dtype=object), np.array(dy, float), np.array(dp, float))


def classical(d: PoolDeltas) -> Estimate:
    """Mean of within-pool differences over every pool with a HUMAN difference. No model anywhere.

    The set is `has_true`, never `paired`. Whether the predictor happens to define a difference
    for a pool is irrelevant to a human-only estimate, and conditioning on it would make this
    number move when the model does -- which is exactly what it used to do on `decay`.
    """
    v = d.d_true[d.has_true]
    n = len(v)
    if n < 2:
        return Estimate('classical', float(v.mean()) if n else float('nan'),
                        float('nan'), float('nan'), n, 0,
                        {'note': 'n<2: no interval is defined'})
    se = v.std(ddof=1) / np.sqrt(n)
    m, q = float(v.mean()), _t(n)
    return Estimate('classical', m, m - q * se, m + q * se, n, 0, {'se': float(se)})


def ppi(d: PoolDeltas) -> Estimate:
    """PPI++ with power-tuned lambda. Unbiased for any predictor; f moves only the variance.

    BOTH arms need a defined prediction, and that is correct rather than a limitation: the
    rectifier subtracts a PAIR (D_Y - lam D_f) on the labelled side, so a pool with no predicted
    difference has nothing to subtract and cannot enter either term.
    """
    lab, unl = d.paired, d.unlabelled
    dy, fl, fu = d.d_true[lab], d.d_pred[lab], d.d_pred[unl]
    n, N = len(dy), len(fu)
    if n < 2 or N < 2:
        return Estimate('ppi', float('nan'), float('nan'), float('nan'), n, N,
                        {'note': 'needs >=2 labelled and >=2 unlabelled pools'})
    # LAMBDA'S DENOMINATOR IS Var(f) OVER ALL n+N POOLS, NOT OVER THE n LABELLED ONES.
    # That is PPI++'s own plug-in for a mean -- Angelopoulos, Duchi & Zrnic (2023), Example 6.1:
    # lam_hat = Cov_n(D_Y, D_f) / [(1 + n/N) * Var_{n+N}(D_f)] -- and the pooled variance is what
    # keeps the ratio stable. A genotype substratum has n=2, where a labelled-only variance is one
    # squared difference and can be arbitrarily small for reasons that have nothing to do with the
    # predictor's quality. On kdm6b/wt the two labelled pools had a mathematically IDENTICAL
    # predicted delta that differed by a single float64 ULP: Var came out 2.5e-32, passed a bare
    # `> 0`, and drove lambda to 5e14 and the estimate to 2.6e14 bouts per minute. Five further
    # n=2 cells carried |lambda| of 6-17 the same way, for estimates of +24 and -14 bouts/min.
    # Pooling the unlabelled pools in makes the denominator a real spread, so a near-tie now
    # zeroes the COVARIANCE rather than the variance and lambda falls to ~0 -- which is the right
    # answer, not a patch: two pools the model scores identically carry no information about
    # lambda, and lambda=0 is exactly the classical estimate.
    #
    # THEN PROJECTED TO [0, 1], which is a SEPARATE fix and would NOT have caught the above.
    # Clipping the old lambda would have returned 1 -- vanilla PPI, importing the model's full
    # uncalibrated scale -- for an estimate of -0.213 against the classical -0.733, a plausible
    # wrong number instead of a visibly broken one. Fixing the denominator is what makes the cell
    # correct; the clip is what bounds everything else.
    #
    # Why clip at all, given PPI++ introduces lambda over R (section 2.2) and Example 6.1 says
    # mean estimation "is always convex for any choice of lambda in R, thus obviating the need for
    # clipping"? Because that passage is about the OPTIMISATION being well posed, and the paper's
    # "never worse than classical" result is asymptotic: it holds at the population lambda*, and
    # reaches finite samples only through lambda_hat -> lambda* (Corollary 4). At n=2 there is no
    # such convergence -- the covariance is a single 1-df product -- and measured over this grid's
    # 200 two-pool cells the unclipped plug-in makes the PPI interval on average WIDER than the
    # classical one it exists to narrow (mean width ratio 1.03, worst 5.4x). Projecting to [0,1]
    # brings that to 0.80 and 2.7x. At n>=6, where lambda_hat is a real estimate, the clip binds on
    # 21 of 259 cells and is a wash (mean ratio 0.8836 clipped against 0.8833 unclipped), so it
    # costs essentially nothing where the asymptotics do apply. Unbiasedness holds at EVERY lambda,
    # so the clip trades a sliver of asymptotic efficiency for finite-sample robustness and risks
    # no validity. `ppi.py` clips for the same reason, on its own simulation evidence.
    f_all = np.concatenate([fl, fu])
    v = float(f_all.var(ddof=1))
    lam = (float(np.cov(dy, fl)[0, 1] / (v * (1 + n / N)))
           if np.isfinite(v) and np.ptp(f_all) > 1e-9 * max(1.0, float(np.abs(f_all).max()))
           else 0.0)
    lam = float(np.clip(lam, 0.0, 1.0))
    est = float((lam * fu).mean() + (dy - lam * fl).mean())
    se = float(np.sqrt((dy - lam * fl).var(ddof=1) / n + lam ** 2 * fu.var(ddof=1) / N))
    q = _t(n)
    return Estimate('ppi', est, est - q * se, est + q * se, n, N,
                    {'se': se, 'lam': lam, 'r': d.r()})


def ppci(d: PoolDeltas, k: float = None, k_boot=None, reps: int = 4000,
         seed: int = 0, raw: bool = False) -> Estimate:
    """Plug-in on EVERY pool, with the scale taken from labels -- or, with raw=True, no scale.

    WHY THIS IS NOT beta, AND WHY beta WAS WRONG
    ============================================
    An earlier version rescaled by the regression slope beta = Cov(D_Y,D_f)/Var(D_f). That is
    the right multiplier for predicting ONE pool's D_Y from its own D_f, but it is the wrong
    multiplier for rescaling a MEAN, because beta = rho * sd_Y/sd_f is attenuated by noise in
    D_f. Measured on v1: beta ran 0.10-0.46 while the actual ratio of means E[D_Y]/E[D_f] ran
    0.44-2.18 -- a factor of SEVEN apart on nn/social/H->O (beta 0.209 against a ratio of 1.56).
    Every beta-rescaled estimate was therefore pulled toward zero by regression attenuation:
    +0.04 where the classical mean is +0.47. That is a bug, not a conservative choice.

    The correct scale for a mean is the ratio of means,

        k = mean(D_Y) / mean(D_f)      fitted on the labelled pools

    which is unbiased for the population mean whenever the predictor is off by a multiplicative
    factor: if E[D_f] = c E[D_Y] then k = 1/c and k * mean_all(D_f) targets E[D_Y]. One free
    parameter, and it is a calibration of the quantity actually being reported.

    IT STILL NEEDS LABELS, AND ON v2 THERE ARE NONE
    ===============================================
    k cannot be fitted on v2 -- nothing there is annotated. Passing `k` transports v1's value,
    and that transport is the whole load-bearing assumption, not a detail. `raw=True` is the
    honest alternative: the plug-in mean(D_f) with no calibration at all, which uses no labels
    anywhere and is therefore the only version that is genuinely available on an unannotated
    cohort. Its magnitude is on the MODEL's scale, not the behaviour's, so it supports
    statements about SIGN and relative pattern and nothing about effect size.

    k is fragile exactly where mean(D_f) approaches zero -- it is a ratio -- so it is refused
    when the denominator is not separated from zero, rather than returned as a large number.
    """
    rng = np.random.default_rng(seed)
    lab = d.paired                  # k relates the two sides, so it needs them both
    dy, fl = d.d_true[lab], d.d_pred[lab]
    f_all = d.d_pred[d.has_pred]    # the plug-in mean needs only a prediction
    if len(f_all) < 2:
        return Estimate('ppci', float('nan'), float('nan'), float('nan'),
                        int(lab.sum()), len(f_all), {'note': 'no predictions'})
    if raw:
        est = float(f_all.mean())
        bs = [f_all[rng.integers(0, len(f_all), len(f_all))].mean() for _ in range(reps)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        return Estimate('ppci', est, float(lo), float(hi), int(lab.sum()), len(f_all),
                        {'scale': 'model (uncalibrated)', 'k': None})
    local = k is None
    if local:
        if lab.sum() < 3:
            return Estimate('ppci', float('nan'), float('nan'), float('nan'),
                            int(lab.sum()), len(f_all), {'note': 'needs >=3 labelled pools'})
        # refuse a ratio whose denominator is not separated from zero
        se_f = fl.std(ddof=1) / np.sqrt(len(fl))
        if abs(fl.mean()) < 2 * se_f:
            return Estimate('ppci', float('nan'), float('nan'), float('nan'),
                            int(lab.sum()), len(f_all),
                            {'note': 'mean(D_f) not separated from zero: k is not identified'})
        k = float(dy.mean() / fl.mean())
    est = float(k * f_all.mean())
    bs = []
    for _ in range(reps):
        fb = f_all[rng.integers(0, len(f_all), len(f_all))]
        if local:
            i = rng.integers(0, len(dy), len(dy))
            den = fl[i].mean()
            if abs(den) < 1e-9:
                continue
            kk = dy[i].mean() / den
        else:
            kk = k if k_boot is None else k_boot[rng.integers(0, len(k_boot))]
        bs.append(kk * fb.mean())
    if len(bs) < reps // 10:
        return Estimate('ppci', est, float('nan'), float('nan'), int(lab.sum()), len(f_all),
                        {'k': float(k), 'note': 'bootstrap degenerate'})
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return Estimate('ppci', est, float(lo), float(hi), int(lab.sum()), len(f_all),
                    {'k': float(k), 'r': d.r(), 'transported': not local,
                     'scale': 'calibrated to labels'})


def scale_bootstrap(d: PoolDeltas, reps: int = 4000, seed: int = 0) -> np.ndarray:
    """Bootstrap sample of the ratio-of-means calibration k, for transporting it to a cohort."""
    rng = np.random.default_rng(seed)
    lab = d.paired
    dy, fl = d.d_true[lab], d.d_pred[lab]
    out = []
    for _ in range(reps):
        i = rng.integers(0, len(dy), len(dy))
        den = fl[i].mean()
        if abs(den) > 1e-9:
            out.append(dy[i].mean() / den)
    return np.array(out)


def assert_out_of_fold(labelled_pools, fold_train_pools: dict) -> None:
    """Refuse to proceed if any labelled pool was in the training set of the model scoring it.

    `fold_train_pools` maps fold tag -> the pools that fold TRAINED on. A labelled pool is only
    admissible if the fold whose prediction we are using held it out. This is the assumption
    that, when violated, makes a PPI interval look excellent while undercovering.
    """
    bad = {p: t for t, tr in fold_train_pools.items() for p in labelled_pools if p in tr}
    if bad:
        raise AssertionError(
            'these labelled pools carry IN-SAMPLE predictions, which invalidates the PPI '
            f'rectifier: {sorted(bad)}. Use the fold that held each pool out.')
