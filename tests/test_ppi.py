"""Simulation tests for the PPI++ estimator, run at the true mice v1 pool-level design.

These are the guarantees the whole approach rests on, so they are asserted rather than
argued: unbiasedness under an arbitrarily miscalibrated predictor, nominal interval
coverage, and never being meaningfully worse than the classical estimator.

Runs standalone (`python tests/test_ppi.py`) -- the repo has no pytest dependency and the
NumPy 2.4.1 environment is deliberately kept minimal -- but every test_* function is also
plain pytest-collectable if pytest is ever added.
"""
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mice_behavior.ppi import (  # noqa: E402
    ArmData, classical_contrast, ppi_contrast, projected_variance_factor,
    assert_crossfitted,
)

# mice v1 at pool level: genotype constant within pool, 72 pools, 24 annotated.
HET_L, HET_U = 18, 18
WT_L,  WT_U  = 6, 30
MU_HET, MU_WT = 0.0182, 0.0140      # ~ measured nn pool rates
SD = 0.006
TRUE_ATE = MU_HET - MU_WT
REPS = 3000
R_GRID = (0.35, 0.55, 0.75)


def _arm(rng, name, n_lab, n_unl, mu, r, slope=13.0, offset=0.05):
    """An arm whose predictor is deliberately broken: 13x inflated with a large offset,
    matching the mice classifier's measured prior shift (calibration slope 0.074-0.118)."""
    n = n_lab + n_unl
    y = rng.normal(mu, SD, n)
    f = slope * (y + rng.normal(0, SD * np.sqrt(1 / r ** 2 - 1), n)) + offset
    return ArmData(name, y[:n_lab], f[:n_lab], f[n_lab:])


def _trial(rng, r):
    return (_arm(rng, "het", HET_L, HET_U, MU_HET, r),
            _arm(rng, "wt",  WT_L,  WT_U,  MU_WT,  r))


def test_unbiased_under_broken_calibration():
    """The point of PPI: f can be 13x miscalibrated and the estimate stays centred."""
    for r in R_GRID:
        rng = np.random.default_rng(7)
        est = np.array([ppi_contrast(*_trial(rng, r)).theta for _ in range(REPS)])
        bias = est.mean() - TRUE_ATE
        tol = 3 * est.std() / np.sqrt(REPS)
        assert abs(bias) < tol, f"bias {bias:+.2e} exceeds {tol:.2e} at r={r}"


def test_nominal_coverage():
    """95% intervals must actually cover ~95% of the time, including the n=6 wt arm."""
    for r in R_GRID:
        rng = np.random.default_rng(11)
        hits = sum(e.ci_low <= TRUE_ATE <= e.ci_high
                   for e in (ppi_contrast(*_trial(rng, r)) for _ in range(REPS)))
        cov = hits / REPS
        assert 0.93 <= cov <= 0.97, f"coverage {cov:.3f} at r={r}"


def test_never_meaningfully_worse_than_classical():
    """lambda -> 0 must protect us: a weak predictor may fail to help, but must not hurt."""
    for r in R_GRID:
        rng = np.random.default_rng(13)
        ppi, cls = [], []
        for _ in range(REPS):
            het, wt = _trial(rng, r)
            ppi.append(ppi_contrast(het, wt).theta)
            cls.append(classical_contrast(het, wt).theta)
        ratio = np.std(ppi) / np.std(cls)
        assert ratio < 1.02, f"sd ratio {ratio:.3f} at r={r} -- PPI is hurting"


def test_vanilla_ppi_is_wrecked_by_miscalibration():
    """lam=1 stays unbiased but explodes the variance -- why power tuning is mandatory."""
    rng = np.random.default_rng(17)
    tuned, fixed, cls = [], [], []
    for _ in range(500):
        het, wt = _trial(rng, 0.55)
        tuned.append(ppi_contrast(het, wt).theta)
        fixed.append(ppi_contrast(het, wt, lam={"het": 1.0, "wt": 1.0}).theta)
        cls.append(classical_contrast(het, wt).theta)
    assert np.std(fixed) > 5 * np.std(cls), "vanilla PPI should be catastrophic here"
    assert np.std(tuned) < np.std(cls), "power-tuned PPI should still help"
    assert abs(np.mean(fixed) - TRUE_ATE) < 4 * np.std(fixed) / np.sqrt(500), \
        "vanilla PPI must remain UNBIASED even while being useless"


def test_lambda_absorbs_the_prior_shift():
    """The tuned lambda should land near 1/slope, i.e. it IS the calibration correction."""
    rng = np.random.default_rng(19)
    lams = [ppi_contrast(*_trial(rng, 0.75)).lam["wt"] for _ in range(300)]
    assert 0.02 < np.mean(lams) < 0.20, f"mean lambda {np.mean(lams):.3f} (expected ~1/13)"


def test_projection_is_not_one_minus_r_squared():
    """The finite unlabeled pool matters: het's n/N = 1 halves the advertised gain."""
    r = 0.55
    assert abs(projected_variance_factor(r, HET_L, HET_U) - (1 - r ** 2 / 2)) < 1e-12
    assert abs(projected_variance_factor(r, WT_L, WT_U) - (1 - r ** 2 / 1.2)) < 1e-12
    assert projected_variance_factor(r, HET_L, HET_U) > 1 - r ** 2, "gain must be strictly smaller"


def test_crossfit_guard():
    assert_crossfitted(["rd11_2", "rd13"], ["rd18", "rd25"])         # disjoint -> fine
    try:
        assert_crossfitted(["rd11_2", "rd13"], ["rd13", "rd25"])
    except ValueError as exc:
        assert "cross-fitted" in str(exc)
    else:
        raise AssertionError("leaked pool was not caught")


def test_arm_rejects_misaligned_input():
    try:
        ArmData("bad", np.zeros(5), np.zeros(4), np.zeros(3))
    except ValueError as exc:
        assert "aligned pool-for-pool" in str(exc)
    else:
        raise AssertionError("misaligned arm was not caught")


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception:
            failed += 1
            print(f"  FAIL  {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
