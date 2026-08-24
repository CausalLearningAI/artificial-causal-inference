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

  classical   mean(D_Y) over the labelled pools. No model. Unbiased, and the only one of the
              three that needs no assumption beyond the design. Limited to 24 of v1's 72 pools
              and unavailable on v2, which has no labels at all.

  PPI++       lam * mean_N(D_f)  +  mean_n(D_Y - lam * D_f)
              Unbiased for ANY predictor: whatever lam is and however wrong f is, the second
              term subtracts exactly what the first added. f moves only the VARIANCE. The
              power-tuned lam = Cov(D_Y, D_f) / (Var(D_f) * (1 + n/N)) is what makes a
              miscalibrated model harmless -- lam absorbs the scale, so calibration happens in
              the one place where getting it wrong costs variance instead of validity.
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
    """One row per pool: its true and/or predicted phase difference for one cell."""
    pools: np.ndarray
    d_true: np.ndarray          # NaN where the pool is unlabelled
    d_pred: np.ndarray

    @property
    def labelled(self):
        return np.isfinite(self.d_true)

    def r(self) -> float:
        """Within-cell correlation of true and predicted differences on the labelled pools.

        This -- not frame AP -- is what PPI's variance reduction is a function of. It is also
        much lower than the value obtained by pooling cells together, because pooling adds
        between-cell signal that a single-cell estimate cannot use.
        """
        m = self.labelled
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

    `df` needs columns pool, phase, odor and the two value columns. A pool contributes only if
    it has BOTH phases of the transition; `true_col=None` marks the frame as unlabelled.
    """
    x, y = transition
    sub = df[df.odor == odour]
    keys, dy, dp = [], [], []
    for pool, g in sub.groupby('pool', sort=True):
        m = g.drop_duplicates('phase').set_index('phase')
        if x not in m.index or y not in m.index:
            continue
        d_p = (m.loc[y, pred_col] - m.loc[x, pred_col]) if pred_col else np.nan
        if pred_col and not np.isfinite(d_p):
            continue                  # no prediction for this pool: it cannot enter any mean
        keys.append(pool)
        dy.append(m.loc[y, true_col] - m.loc[x, true_col] if true_col else np.nan)
        dp.append(d_p)
    return PoolDeltas(np.array(keys, dtype=object), np.array(dy, float), np.array(dp, float))


def classical(d: PoolDeltas) -> Estimate:
    """Mean of within-pool differences on labelled pools only. No model anywhere."""
    v = d.d_true[d.labelled]
    n = len(v)
    if n < 2:
        return Estimate('classical', float(v.mean()) if n else float('nan'),
                        float('nan'), float('nan'), n, 0,
                        {'note': 'n<2: no interval is defined'})
    se = v.std(ddof=1) / np.sqrt(n)
    m, q = float(v.mean()), _t(n)
    return Estimate('classical', m, m - q * se, m + q * se, n, 0, {'se': float(se)})


def ppi(d: PoolDeltas) -> Estimate:
    """PPI++ with power-tuned lambda. Unbiased for any predictor; f moves only the variance."""
    lab = d.labelled
    dy, fl, fu = d.d_true[lab], d.d_pred[lab], d.d_pred[~lab]
    n, N = len(dy), len(fu)
    if n < 2 or N < 2:
        return Estimate('ppi', float('nan'), float('nan'), float('nan'), n, N,
                        {'note': 'needs >=2 labelled and >=2 unlabelled pools'})
    v = fl.var(ddof=1)
    lam = float(np.cov(dy, fl)[0, 1] / (v * (1 + n / N))) if v > 0 else 0.0
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
    lab = d.labelled
    dy, fl = d.d_true[lab], d.d_pred[lab]
    f_all = d.d_pred[np.isfinite(d.d_pred)]
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
    lab = d.labelled
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
