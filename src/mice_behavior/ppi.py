"""PPI++ (prediction-powered inference) for the mice v1 genotype contrast.

WHY THIS MODULE EXISTS
----------------------
Until now the only downstream number the mice frame classifier produced was
`rate_report`'s per-observation Pearson r, plus a *projected* `ppi_variance_reduction`
of (1 - r^2). Nothing actually computed a prediction-powered estimate. That gap is what
makes people (correctly) nervous about calibration: with no rectifier, a plug-in estimate
built from predicted rates inherits every bit of the classifier's prior shift, and the
8-13x over-prediction documented in `metrics.rate_report` lands straight in the effect
size.

PPI++ removes that worry entirely. The estimator is

    theta_hat(lam) = mean_L(Y - lam * f) + lam * mean_U(f)

and E[theta_hat] = E[Y] for ANY lam and ANY f, because the two lam terms cancel in
expectation. The classifier can be arbitrarily miscalibrated, badly ranked, or plain
wrong: the labeled term rectifies it. f only affects VARIANCE, never validity. So the
honest way to use a miscalibrated model is not to calibrate it -- it is to rectify it.

THE UNIT OF ANALYSIS IS THE POOL, NOT THE OBSERVATION
-----------------------------------------------------
mice v1 genotype is constant within a pool (verified: 0 of 72 pools are mixed), so the
6 observations of a pool share one treatment value, one cage, one cohort and one set of
four physical mice. Treating observations as independent would quote a SE built on 144
units when the design only supplies 24. Everything here takes POOL-level rates as input,
which makes the clustering structural rather than a variance correction bolted on after.

At pool level the v1 design is:

    arm    labeled (n)   unlabeled (N)   n/N
    het        18            18          1.00
    wt          6            30          0.20

The `wt` arm carries only 6 labeled pools and therefore dominates the contrast's SE --
and it is exactly the arm with the most unlabeled data to borrow from. That asymmetry is
the whole reason PPI is worth doing here rather than just annotating more.

THE ONE ASSUMPTION THAT CAN ACTUALLY BREAK THIS
-----------------------------------------------
Unbiasedness needs labeled and unlabeled units to be exchangeable *within the stratum the
estimator is applied to*. In v1 they are NOT exchangeable marginally: annotation was
assigned 108 het / 36 wt at observation level while the full design is 216 / 216, so the
labeled set is 3:1 het-enriched. This is why every estimator here is stratified BY ARM and
never pools the two -- within an arm, `line`, `phase` and `odor` are balanced 48/48/48 and
96/96/96 between labeled and unlabeled, which is what the assumption needs. Do not "fix"
this by concatenating arms.

Second requirement, easy to violate by accident: **f must not have been trained on the
labeled units used in the rectifier**. In-sample predictions shrink (Y - lam*f) toward
zero, which understates the correction and produces an overconfident, biased interval.
Use cross-fitted predictions (see `assert_crossfitted`) or restrict the labeled set to
genuinely held-out pools.

References: Angelopoulos, Bates, Fannjiang, Jordan & Zrnic (2023), "Prediction-Powered
Inference"; Angelopoulos, Duchi & Zrnic (2023), "PPI++: Efficient Prediction-Powered
Inference" (the power-tuned lambda used here).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

__all__ = [
    "Estimate", "ArmData", "classical_mean", "ppi_mean", "ppi_contrast",
    "classical_contrast", "projected_variance_factor", "assert_crossfitted",
]


# ── result container ──────────────────────────────────────────────────────────

@dataclass
class Estimate:
    """A point estimate with an interval, plus everything needed to explain it."""
    theta:       float
    se:          float
    ci_low:      float
    ci_high:     float
    alpha:       float
    method:      str                       # 'classical' | 'ppi++'
    n_labeled:   int
    n_unlabeled: int
    lam:         dict = field(default_factory=dict)   # arm -> tuned lambda
    dof:         float = float("nan")
    per_arm:     dict = field(default_factory=dict)

    @property
    def ci_width(self) -> float:
        return self.ci_high - self.ci_low

    @property
    def significant(self) -> bool:
        return not (self.ci_low <= 0.0 <= self.ci_high)

    def as_dict(self) -> dict:
        d = {k: getattr(self, k) for k in
             ("theta", "se", "ci_low", "ci_high", "alpha", "method",
              "n_labeled", "n_unlabeled", "dof")}
        d["ci_width"] = self.ci_width
        d["significant"] = self.significant
        d["lam"] = dict(self.lam)
        d["per_arm"] = dict(self.per_arm)
        return d


@dataclass
class ArmData:
    """One treatment arm's pool-level rates.

    y_lab : true rate for each LABELED pool            (shape n)
    f_lab : predicted rate for those same pools        (shape n) -- must be out-of-fold
    f_unl : predicted rate for each UNLABELED pool     (shape N)
    """
    name:  str
    y_lab: np.ndarray
    f_lab: np.ndarray
    f_unl: np.ndarray

    def __post_init__(self):
        self.y_lab = np.asarray(self.y_lab, dtype=float).ravel()
        self.f_lab = np.asarray(self.f_lab, dtype=float).ravel()
        self.f_unl = np.asarray(self.f_unl, dtype=float).ravel()
        if self.y_lab.shape != self.f_lab.shape:
            raise ValueError(
                f"[{self.name}] y_lab {self.y_lab.shape} and f_lab {self.f_lab.shape} "
                "must be aligned pool-for-pool"
            )
        if self.n < 2:
            raise ValueError(f"[{self.name}] needs >=2 labeled pools, got {self.n}")

    @property
    def n(self) -> int:
        return len(self.y_lab)

    @property
    def N(self) -> int:
        return len(self.f_unl)

    @property
    def r(self) -> float:
        """Pearson r between true and predicted pool rate -- the quantity that sets the gain."""
        if self.n < 3 or np.ptp(self.y_lab) == 0 or np.ptp(self.f_lab) == 0:
            return float("nan")
        return float(stats.pearsonr(self.y_lab, self.f_lab)[0])


# ── lambda tuning ─────────────────────────────────────────────────────────────

def _tune_lambda(arm: ArmData, clip: bool = True) -> float:
    """Power-tuned lambda minimising ONE arm's variance.

    Var(theta_lam) = Var(Y - lam f)/n + lam^2 Var(f)/N
    d/dlam = 0  =>  lam* = Cov(Y, f) / (Var(f) * (1 + n/N))

    lam is clipped to [0, 1]. At lam=0 the estimator degrades gracefully to the classical
    labeled-only mean, so a useless f can never hurt.

    NOTE: this is the textbook per-arm tuner and is exposed for reference, but it is NOT
    the default for the contrast -- see `_shared_lambda` for why per-arm tuning loses at
    v1's sample sizes.
    """
    if arm.N == 0:
        return 0.0
    var_f = float(np.var(arm.f_lab, ddof=1))
    if not np.isfinite(var_f) or var_f <= 0:
        return 0.0
    cov = float(np.cov(arm.y_lab, arm.f_lab, ddof=1)[0, 1])
    lam = cov / (var_f * (1.0 + arm.n / arm.N))
    if not np.isfinite(lam):
        return 0.0
    return float(np.clip(lam, 0.0, 1.0)) if clip else float(lam)


def _shared_lambda(arms: list[ArmData], drop: tuple[str, int] | None = None,
                   clip: bool = True) -> float:
    """Contrast-optimal lambda SHARED across arms.

    Minimising Var(theta_treat - theta_control) = sum_a [Var_a(Y - lam f)/n_a + lam^2 Var_a(f)/N_a]
    over a single lam gives

        lam* = [sum_a Cov_a(Y,f)/n_a] / [sum_a Var_a(f) * (1/n_a + 1/N_a)]

    Why shared rather than per-arm, even though per-arm is optimal when lambda is KNOWN:
    lambda has to be estimated, and at v1's sizes the estimation noise dominates the
    theoretical gain. Simulated at the true v1 design (het 18/18, wt 6/30), 3000 reps,
    reporting realised sd of the contrast relative to the classical estimator, with 95%
    interval coverage in brackets:

        scheme            r=0.35          r=0.55          r=0.75
        per-arm tuned     1.017 [0.923]   0.955 [0.912]   0.824 [0.919]
        per-arm LOO       1.069 [0.944]   0.991 [0.940]   0.862 [0.941]
        shared LOO        0.994 [0.946]   0.933 [0.940]   0.814 [0.945]
        fixed lam=1      43.306 [0.953]  27.566 [0.942]  19.863 [0.949]

    Per-arm tuning has to estimate Cov(Y,f) from the wt arm's 6 pools and overfits it,
    which both undercovers (0.912-0.923 against a nominal 0.95) and, at low r, makes the
    estimator genuinely WORSE than classical. Sharing lambda pools all 24 labeled pools
    into one covariance estimate and wins on both axes.

    The `fixed lam=1` row is vanilla PPI, and it is the direct answer to "is calibration
    cheating?": with the mice classifier's measured ~13x prior-shift inflation, lam=1
    imports that scale straight into the estimate and inflates the sd by 20-43x. The
    estimator stays UNBIASED throughout -- it just becomes useless. Power tuning drives
    lam to ~1/13 automatically, which is calibration performed in the one place where
    getting it wrong costs variance instead of validity.

    `drop=(arm_name, i)` removes labeled pool i of that arm, for leave-one-out fitting.
    """
    num = den = 0.0
    for a in arms:
        y, f = a.y_lab, a.f_lab
        if drop is not None and drop[0] == a.name:
            keep = np.ones(a.n, dtype=bool)
            keep[drop[1]] = False
            y, f = y[keep], f[keep]
        n = len(y)
        if n < 2 or a.N == 0:
            continue
        var_f = float(np.var(f, ddof=1))
        if not np.isfinite(var_f) or var_f <= 0:
            continue
        num += float(np.cov(y, f, ddof=1)[0, 1]) / n
        den += var_f * (1.0 / n + 1.0 / a.N)
    if den <= 0 or not np.isfinite(num / den):
        return 0.0
    lam = num / den
    return float(np.clip(lam, 0.0, 1.0)) if clip else float(lam)


def _loo_lambdas(arms: list[ArmData], arm: ArmData) -> np.ndarray:
    """Leave-one-out shared lambda for each labeled pool of `arm`.

    lambda_i is fitted WITHOUT pool i, so it is independent of pool i's residual. That
    independence is what restores nominal coverage: fitting lambda on the same residuals
    it multiplies makes (Y - lam*f) look smaller than it is and shrinks the interval by
    ~3 percentage points of coverage at n=6 (0.912 -> 0.940 in the table above).
    """
    return np.array([_shared_lambda(arms, drop=(arm.name, i)) for i in range(arm.n)])


# ── single-arm estimators ─────────────────────────────────────────────────────

def _arm_components(arm: ArmData, lam) -> tuple[float, float, float, float, float]:
    """Return (theta, v_lab, v_unl, dof_lab, dof_unl) for one arm.

    `lam` is either a scalar or a per-labeled-pool vector of leave-one-out lambdas. In the
    vector case the unlabeled term uses their mean, which keeps the estimator unbiased:
    each lambda_i is independent of the unit it multiplies, and lam_bar is independent of
    the unlabeled pools entirely.
    """
    lam_vec = np.asarray(lam, dtype=float)
    lam_bar = float(lam_vec.mean()) if lam_vec.ndim else float(lam_vec)
    resid = arm.y_lab - lam_vec * arm.f_lab
    theta = float(resid.mean() + (lam_bar * arm.f_unl.mean() if arm.N else 0.0))
    v_lab = float(np.var(resid, ddof=1)) / arm.n
    v_unl = (lam_bar ** 2) * float(np.var(arm.f_unl, ddof=1)) / arm.N if arm.N > 1 else 0.0
    return theta, v_lab, v_unl, arm.n - 1, max(arm.N - 1, 1)


def _satterthwaite(components: list[tuple[float, float]]) -> float:
    """Welch-Satterthwaite dof for a sum of independent variance components [(v, dof)].

    Matters here because the labeled term of the `wt` arm has only 5 dof. A normal
    interval would be ~15% too narrow; the t interval is the honest one.
    """
    comps = [(v, d) for v, d in components if v > 0 and d > 0]
    if not comps:
        return float("inf")
    total = sum(v for v, _ in comps)
    denom = sum(v ** 2 / d for v, d in comps)
    return total ** 2 / denom if denom > 0 else float("inf")


def classical_mean(arm: ArmData, alpha: float = 0.05) -> Estimate:
    """Labeled-only mean -- the baseline PPI has to beat. Ignores every unlabeled pool."""
    y = arm.y_lab
    se = float(np.std(y, ddof=1) / np.sqrt(arm.n))
    dof = arm.n - 1
    half = stats.t.ppf(1 - alpha / 2, dof) * se
    m = float(y.mean())
    return Estimate(theta=m, se=se, ci_low=m - half, ci_high=m + half, alpha=alpha,
                    method="classical", n_labeled=arm.n, n_unlabeled=0, dof=float(dof))


def ppi_mean(arm: ArmData, alpha: float = 0.05, lam=None,
             context: list[ArmData] | None = None) -> Estimate:
    """PPI++ mean for one arm. Unbiased for any f; f only moves the variance.

    `context` supplies the other arms so lambda can be pooled across all labeled pools
    (see `_shared_lambda`); with no context it falls back to leave-one-out within this arm.
    """
    if lam is None:
        lam = _loo_lambdas(context or [arm], arm)
    theta, v_lab, v_unl, d_lab, d_unl = _arm_components(arm, lam)
    se = float(np.sqrt(v_lab + v_unl))
    dof = _satterthwaite([(v_lab, d_lab), (v_unl, d_unl)])
    half = stats.t.ppf(1 - alpha / 2, dof) * se
    return Estimate(theta=theta, se=se, ci_low=theta - half, ci_high=theta + half,
                    alpha=alpha, method="ppi++", n_labeled=arm.n, n_unlabeled=arm.N,
                    lam={arm.name: float(np.mean(lam))}, dof=dof)


# ── contrast (the actual estimand) ────────────────────────────────────────────

def classical_contrast(treat: ArmData, control: ArmData, alpha: float = 0.05) -> Estimate:
    """Difference in mean pool rate, labeled pools only (Welch two-sample)."""
    a, b = classical_mean(treat, alpha), classical_mean(control, alpha)
    theta = a.theta - b.theta
    se = float(np.hypot(a.se, b.se))
    dof = _satterthwaite([(a.se ** 2, treat.n - 1), (b.se ** 2, control.n - 1)])
    half = stats.t.ppf(1 - alpha / 2, dof) * se
    return Estimate(
        theta=theta, se=se, ci_low=theta - half, ci_high=theta + half, alpha=alpha,
        method="classical", n_labeled=treat.n + control.n, n_unlabeled=0, dof=dof,
        per_arm={treat.name: a.as_dict(), control.name: b.as_dict()},
    )


def ppi_contrast(treat: ArmData, control: ArmData, alpha: float = 0.05,
                 lam: dict | None = None, tuning: str = "shared-loo") -> Estimate:
    """PPI++ difference in mean pool rate between two arms. THE estimator for v1.

    Stratified by construction: each arm gets its own rectifier, so the v1 annotation
    imbalance (the labeled set is 3:1 het-enriched while the full design is 1:1) cannot
    bias the contrast. Exchangeability is only ever assumed WITHIN an arm.

    `tuning`:
      'shared-loo' (default) -- one lambda pooled over all labeled pools, fitted
                                leave-one-out. Best realised variance AND nominal
                                coverage at v1's sizes; see `_shared_lambda`.
      'shared'               -- same pooled lambda, no LOO. Slightly tighter intervals
                                that undercover; use only to reproduce older numbers.
      'per-arm'              -- textbook per-arm tuning. Undercovers at n=6, and at low r
                                is worse than doing nothing.
    Passing `lam={'het': ..., 'wt': ...}` fixes lambda explicitly and bypasses tuning
    (lam=1 for both is vanilla PPI).
    """
    arms = [treat, control]
    lam = lam or {}

    def _lam_for(a: ArmData):
        if a.name in lam:
            return float(lam[a.name])
        if tuning == "shared-loo":
            return _loo_lambdas(arms, a)
        if tuning == "shared":
            return _shared_lambda(arms)
        if tuning == "per-arm":
            return _tune_lambda(a)
        raise ValueError(f"unknown tuning={tuning!r}")

    lam_t, lam_c = _lam_for(treat), _lam_for(control)
    th_t, vl_t, vu_t, dl_t, du_t = _arm_components(treat, lam_t)
    th_c, vl_c, vu_c, dl_c, du_c = _arm_components(control, lam_c)

    theta = th_t - th_c
    se = float(np.sqrt(vl_t + vu_t + vl_c + vu_c))
    dof = _satterthwaite([(vl_t, dl_t), (vu_t, du_t), (vl_c, dl_c), (vu_c, du_c)])
    half = stats.t.ppf(1 - alpha / 2, dof) * se

    return Estimate(
        theta=theta, se=se, ci_low=theta - half, ci_high=theta + half, alpha=alpha,
        method="ppi++", n_labeled=treat.n + control.n, n_unlabeled=treat.N + control.N,
        lam={treat.name: float(np.mean(lam_t)), control.name: float(np.mean(lam_c))},
        dof=dof,
        per_arm={
            treat.name:   ppi_mean(treat,   alpha, lam_t).as_dict(),
            control.name: ppi_mean(control, alpha, lam_c).as_dict(),
        },
    )


# ── projection / diagnostics ──────────────────────────────────────────────────

def projected_variance_factor(r: float, n: int, N: int) -> float:
    """Var(PPI++) / Var(classical) at the optimal lambda = 1 - r^2 / (1 + n/N).

    NOT (1 - r^2). That is the N -> infinity limit, and it is what `metrics.rate_report`
    currently advertises. With a finite unlabeled pool the gain is strictly smaller,
    and in v1 the difference is large: the het arm has n/N = 1, so its best possible
    factor is 1 - r^2/2, i.e. HALF the advertised reduction. Quote this, not (1 - r^2).
    """
    if N <= 0:
        return 1.0
    return float(1.0 - (r ** 2) / (1.0 + n / N))


def assert_crossfitted(pool_ids, train_pools, arm_name: str = "") -> None:
    """Raise if any labeled pool used in the rectifier was also in the model's train set.

    In-sample f shrinks (Y - lam*f) toward zero: the rectifier looks smaller than it is,
    the interval comes out too narrow, and the estimate picks up the model's bias -- the
    exact failure PPI is supposed to prevent. This is the single easiest way to silently
    invalidate everything in this module, so it is a hard error, not a warning.
    """
    leaked = sorted(set(pool_ids) & set(train_pools))
    if leaked:
        raise ValueError(
            f"[{arm_name or 'ppi'}] {len(leaked)} labeled pool(s) in the rectifier were also "
            f"used to train the predictor: {leaked}. Use cross-fitted (out-of-fold) "
            f"predictions, or restrict the labeled set to held-out pools."
        )
