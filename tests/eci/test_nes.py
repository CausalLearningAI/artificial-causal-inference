"""Simulation tests for Neural Effect Search (src/eci/nes.py), run at the mice v1 design sizes
(72 videos per stage = 36 het vs 36 wt pools; 36 pools per genotype for stage transitions).

Synthetic world (m = 200 neurons): two latent concepts Y1, Y2 are shifted by the treatment
(or the stage). Neuron 0 ~ Y1 and neuron 1 ~ Y2 are the "true" neurons; neurons 2-11 are
"leaky": noisy linear mixtures a*Z_primary + b*Z_other with a in [0.3, 0.6], b in [0, 0.15];
the remaining 188 are noise (half Gaussian, half skewed log-normal). Optional pool random
effects give every pool its own offset on every neuron, shared across its observations.

What is asserted: (a) the naive per-neuron Bonferroni test flags more and more leaky neurons
as n grows while NES does not; (b) NES recovers exactly {0, 1} at the real sizes and larger;
(c) family-wise error rate (FWER) under the global null <= ~alpha for the two-sample and
paired searches (t and sign-flip), and no excess false discoveries in later rounds when only
one neuron is affected; (d) with large pool random effects the paired search finds
the effects while a two-sample search on the same data cannot; (e) frame-level testing under
the null is wrecked by pseudo-replication while video-level testing is not.

Runs standalone (`python tests/eci/test_nes.py`, prints the measured numbers) -- the repo has
no pytest dependency -- but every test_* function is plain pytest-collectable.
"""
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from eci.nes import (  # noqa: E402
    _ols_contrast, _onehot, check_overlap, check_overlap_paired, make_strata, neural_effect_search, neural_effect_test, paired_effect_search,
    paired_effect_test, unit_means,
)

M, TRUE, LEAKY = 200, {0, 1}, np.arange(2, 12)
_mix = np.random.default_rng(12345)
A_MIX = _mix.uniform(0.3, 0.6, len(LEAKY))
B_MIX = _mix.uniform(0.0, 0.15, len(LEAKY))
ALPHA = 0.05
FWER_TOL = ALPHA + 3 * np.sqrt(ALPHA * (1 - ALPHA) / 200)  # 0.096: alpha + 3 Monte Carlo SE at 200 sims


def simulate(rng, T, pool, d1, d2, re_sd=0.0, m=M):
    """Z (N, m) for observations with treatment T (N,) and pool index pool (N,)."""
    N, n_pool = len(T), pool.max() + 1
    U = rng.normal(0, re_sd, (n_pool, m))[pool] if re_sd > 0 else np.zeros((N, m))
    Z = np.empty((N, m))
    Z[:, 0] = d1 * T + U[:, 0] + rng.normal(0, 1, N) + rng.normal(0, 0.2, N)
    Z[:, 1] = d2 * T + U[:, 1] + rng.normal(0, 1, N) + rng.normal(0, 0.2, N)
    for k, j in enumerate(LEAKY):
        p = j % 2
        Z[:, j] = A_MIX[k] * Z[:, p] + B_MIX[k] * Z[:, 1 - p] + U[:, j] + rng.normal(0, 0.5, N)
    rest = np.arange(12, m)
    Z[:, rest] = U[:, rest] + rng.normal(0, 1, (N, len(rest)))
    skew = rest[::2]
    Z[:, skew] = np.exp(0.5 * Z[:, skew])
    return Z


def two_sample(rng, n, d1, d2, **kw):
    T = np.repeat([0, 1], n // 2)
    return simulate(rng, T, np.arange(n), d1, d2, **kw), T


def paired(rng, n_pairs, d1, d2, re_sd=0.0):
    T = np.repeat([0, 1], n_pairs)
    Z = simulate(rng, T, np.tile(np.arange(n_pairs), 2), d1, d2, re_sd=re_sd)
    return Z[:n_pairs], Z[n_pairs:]


def _leaky_flagged(res):
    fr = res['first_round']
    return int(fr.loc[fr['neuron'].isin(LEAKY), 'significant'].sum())


def test_first_round_is_welch():
    """With S = {} the NET must be exactly Welch's two-sample t-test."""
    Z, T = two_sample(np.random.default_rng(0), 72, 1.0, 0.5)
    table, info = neural_effect_test(Z, T)
    ref = stats.ttest_ind(Z[T == 1], Z[T == 0], equal_var=False)
    assert np.allclose(table['t'], ref.statistic) and np.allclose(table['p'], ref.pvalue)
    assert info['n_cells'] == 1


def test_strata_cap():
    """Strata coarsen until every cell has >= min_per_cell units per arm; a sparse neuron
    with median 0 splits into zero vs active."""
    rng = np.random.default_rng(1)
    T = np.repeat([0, 1], 36)
    L = rng.normal(size=(72, 4))
    cell, k, q = make_strata(L, 2, 3, T)
    counts = [np.bincount(cell[T == t]) for t in (0, 1)]
    assert min(c.min() for c in counts) >= 3 and 1 <= k < 4, (k, q)
    sparse = np.where(rng.random(72) < 0.7, 0.0, rng.random(72))
    cell, k, q = make_strata(sparse[:, None], 2, 3, T)
    assert k == 1 and np.array_equal(cell, (sparse > 0).astype(int))
    cell, k, q = make_strata(np.r_[np.zeros(70), 1, 2][:, None], 2, 3, T)
    assert k == 0 and q == 1  # too few active units -> single stratum


def test_unit_means():
    rng = np.random.default_rng(2)
    groups = np.repeat(['v2', 'v0', 'v1'], [3, 5, 4])
    Z, T = rng.normal(size=(12, 7)), np.repeat([1, 0, 0], [3, 5, 4])
    ids, Zu, Tu = unit_means(Z, groups, T)
    assert list(ids) == ['v0', 'v1', 'v2'] and np.allclose(Zu[2], Z[:3].mean(0)) and list(Tu) == [0, 0, 1]


def test_paradox_naive_flags_leaky_as_n_grows():
    """(a) Naive per-neuron Bonferroni (NES round 1) flags leaky neurons more as n grows;
    NES keeps returning the 2 true neurons."""
    rng, sims, flagged, exact = np.random.default_rng(3), 20, {}, {}
    for n in (72, 288, 1152):
        f, e = [], []
        for _ in range(sims):
            res = neural_effect_search(*two_sample(rng, n, 0.8, 0.6))
            f.append(_leaky_flagged(res))
            e.append(set(res['selected']) == TRUE)
        flagged[n], exact[n] = np.mean(f), np.mean(e)
    print(f'    two-sample d=(0.8,0.6): naive leaky flagged /10 {flagged}; NES exact recovery {exact}')
    assert flagged[72] < flagged[288] < flagged[1152] and flagged[1152] >= 9.5
    assert exact[1152] >= 0.9  # NES converges to the truth while the naive count saturates
    f, e = [], []
    for n in (36, 360):
        res = [paired_effect_search(*paired(rng, n, 0.8, 0.6)) for _ in range(sims)]
        f.append(np.mean([_leaky_flagged(r) for r in res]))
        e.append(np.mean([set(r['selected']) == TRUE for r in res]))
    print(f'    paired     d=(0.8,0.6): naive leaky flagged /10 at 36, 360 pairs {f}; NES exact {e}')
    assert f[0] < f[1] and f[1] >= 9.5 and e[1] >= 0.9


def _rates(results):
    return {'exact': np.mean([set(r['selected']) == TRUE for r in results]),
            'first_true': np.mean([len(r['selected']) > 0 and r['selected'][0] in TRUE for r in results]),
            'false_disc': np.mean([len(set(r['selected']) - TRUE) > 0 for r in results])}


def test_recovery_strong_effect():
    """(b) Exact recovery of {0, 1} with a strong effect (2.0 and 1.8 noise sd).
    At 36 pairs the second neuron is found only about half the time: separating it from
    leakage of the first means estimating the leakage slope, whose error is multiplied by the
    first effect (see paired_effect_test); this is a real finite-sample limit, not a bug, and
    it vanishes by 72 pairs."""
    rng, sims, out = np.random.default_rng(4), 40, {}
    for n in (72, 720):
        out[f'two-sample n={n}'] = _rates([neural_effect_search(*two_sample(rng, n, 2.0, 1.8)) for _ in range(sims)])
    for n in (36, 72, 360):
        out[f'paired n={n}'] = _rates([paired_effect_search(*paired(rng, n, 2.0, 1.8)) for _ in range(sims)])
    out['paired sign-flip n=36'] = _rates([paired_effect_search(*paired(rng, 36, 2.0, 1.8), test='signflip',
                                                                n_perm=20000) for _ in range(sims)])
    for k, v in out.items():
        print(f'    {k:32s} ' + '  '.join(f'{a} {b:.2f}' for a, b in v.items()))
    # two-sample n=72: over 400 sims (seed 100) exact recovery is 0.775 with the shared-slope estimator and 0.770 with
    # the per-arm one it replaced (2026-10-04); the 0.85 bound held at 40 sims only by luck of the seed (34/40)
    assert out['two-sample n=72']['exact'] >= 0.7, out['two-sample n=72']
    for k in ('two-sample n=720', 'paired n=72', 'paired n=360'):
        assert out[k]['exact'] >= 0.85, (k, out[k])
    # false_disc ~ alpha is nominal: the final, stopping round is itself a family tested at alpha.
    # Paired n=36: neuron 0 changes by 2 noise sd, so only ~3 of 36 pools have a change <= 0 and the round-2 test
    # (the change of neuron 1 at D_0 = 0) is often outside the data: since 2026-10-04 that round is untestable
    # (check_overlap_paired) and the search stops. Measured at 200 sims (seed 101): exact 0.300 with the rule,
    # 0.495 without (the leakage here is exactly linear, so the old extrapolation happened to be right); 57% of the
    # searches stop on the rule. At 72 pools: 0.900 vs 0.935.
    for k in ('paired n=36', 'paired sign-flip n=36'):
        assert out[k]['first_true'] >= 0.95 and out[k]['exact'] >= 0.1 and out[k]['false_disc'] <= 0.2, (k, out[k])


def test_fwer_global_null():
    """(c) P(NES selects anything | no effect at all) <= alpha (+ Monte Carlo error)."""
    rng, sims, out = np.random.default_rng(5), 200, {}
    out['two-sample n=72'] = np.mean([len(neural_effect_search(*two_sample(rng, 72, 0, 0))['selected']) > 0
                                      for _ in range(sims)])
    out['paired t n=36'] = np.mean([len(paired_effect_search(*paired(rng, 36, 0, 0, re_sd=1.0))['selected']) > 0
                                    for _ in range(sims)])
    out['paired sign-flip n=36'] = np.mean([len(paired_effect_search(*paired(rng, 36, 0, 0, re_sd=1.0),
                                                                      test='signflip', n_perm=8000)['selected']) > 0
                                            for _ in range(sims)])
    print(f'    FWER over {sims} global-null sims (alpha={ALPHA}, tol {FWER_TOL:.3f}): {out}')
    assert all(v <= FWER_TOL for v in out.values()), out


def test_false_discoveries_after_first_round():
    """(c') Partial null: only neuron 0 is affected (2 sd); leaky neurons load on it. Any
    selection beyond {0} is a false discovery made in round >= 2, where the conditioning on
    a post-treatment neuron matters. The paper's plug-in recipe (cross-fitted residuals, then
    Welch) is shown for contrast: it ignores the slope-estimation error and over-rejects."""
    rng, sims, out = np.random.default_rng(9), 200, {}
    for name, fn in [('two-sample ols n=72', lambda: neural_effect_search(*two_sample(rng, 72, 2.0, 0))),
                     ('two-sample crossfit n=72', lambda: neural_effect_search(*two_sample(rng, 72, 2.0, 0),
                                                                               residualize='crossfit')),
                     ('paired t n=36', lambda: paired_effect_search(*paired(rng, 36, 2.0, 0, re_sd=1.0)))]:
        res = [fn() for _ in range(sims)]
        out[name] = (np.mean([0 in r['selected'] for r in res]),
                     np.mean([len(set(r['selected']) - {0}) > 0 for r in res]))
    print('    partial null (found 0, P(any false discovery)): ' + str({k: tuple(round(x, 3) for x in v) for k, v in out.items()}))
    for k in ('two-sample ols n=72', 'paired t n=36'):
        assert out[k][0] >= 0.99 and out[k][1] <= FWER_TOL, (k, out[k])
    assert out['two-sample crossfit n=72'][1] > 0.1


def test_paired_beats_two_sample_under_pool_effects():
    """(d) Pool random effects (sd 3 vs noise sd 1), effects 1.5 / 1.2: pairing cancels the
    pool offsets, a two-sample search on the same 36 + 36 observations cannot see through them."""
    rng, sims = np.random.default_rng(6), 50
    res_p, hit_2 = [], []
    for _ in range(sims):
        Za, Zb = paired(rng, 36, 1.5, 1.2, re_sd=3.0)
        res_p.append(paired_effect_search(Za, Zb))
        T = np.repeat([0, 1], 36)
        hit_2.append(len(set(neural_effect_search(np.vstack([Za, Zb]), T)['selected']) & TRUE) > 0)
    rp = _rates(res_p)
    print(f'    pool re sd=3, 36 pools: paired {rp}; two-sample finds any true neuron {np.mean(hit_2):.2f}')
    assert rp['first_true'] >= 0.95 and rp['false_disc'] <= 0.15 and np.mean(hit_2) <= 0.2


def test_frame_level_pseudo_replication():
    """(e) No treatment effect, 72 videos x 50 frames, video random effect sd 1: frame-level
    NES 'discovers' neurons almost always; video-level NES keeps FWER near alpha."""
    rng, sims, n_vid, n_fr = np.random.default_rng(7), 20, 72, 50
    frame_fwer, video_fwer, n_sel = [], [], []
    for _ in range(sims):
        T_vid = np.repeat([0, 1], n_vid // 2)
        vid = np.repeat(np.arange(n_vid), n_fr)
        Z = simulate(rng, T_vid[vid], vid, 0, 0, re_sd=1.0)
        rf = neural_effect_search(Z, T_vid[vid], max_rounds=20)
        rv = neural_effect_search(Z, T_vid[vid], groups=vid)
        frame_fwer.append(len(rf['selected']) > 0)
        video_fwer.append(len(rv['selected']) > 0)
        n_sel.append(int(rf['first_round']['significant'].sum()))
    print(f'    null, frames: FWER {np.mean(frame_fwer):.2f}, naive #significant/200 {np.mean(n_sel):.1f}; '
          f'videos: FWER {np.mean(video_fwer):.2f}')
    assert np.mean(frame_fwer) >= 0.8 and np.mean(video_fwer) <= 0.15


def test_runtime_m1024():
    rng = np.random.default_rng(8)
    Z, T = two_sample(rng, 72, 1.5, 1.2, m=1024)
    t0 = time.time()
    res = neural_effect_search(Z, T)
    t_two = time.time() - t0
    Za, Zb = paired(rng, 36, 1.5, 1.2)
    Za, Zb = np.hstack([Za, rng.normal(size=(36, 824))]), np.hstack([Zb, rng.normal(size=(36, 824))])
    t0 = time.time()
    resp = paired_effect_search(Za, Zb, test='signflip')
    t_sf = time.time() - t0
    print(f'    m=1024: two-sample n=72 {t_two:.2f}s ({len(res["rounds"])} rounds, selected {res["selected"]}); '
          f'paired sign-flip n=36, n_perm={int(10 * 1024 / ALPHA)} {t_sf:.1f}s (selected {resp["selected"]})')
    assert t_two < 10


def _nuisance_world(rng, T, n_units, m=200):
    """Nuisance N (e.g. mice spread) shifted by T; neuron 0 = N + noise (no effect of its own),
    neuron 1 = a true effect independent of N, the rest noise."""
    N = 1.8 * T + rng.normal(0, 1, n_units)
    Z = rng.normal(0, 1, (n_units, m))
    Z[:, 0] = N + rng.normal(0, 0.5, n_units)
    Z[:, 1] = 1.8 * T + rng.normal(0, 1, n_units)
    return Z, N


def test_nuisance_conditioning_two_sample():
    """(f) a neuron driven only by a nuisance covariate is significant without conditioning and must
    not be selected with nuisance=...; the true effect must still be found."""
    rng = np.random.default_rng(21)
    T = np.repeat([0, 1], 36)
    n_sim, hit0_plain, hit0, hit1, extra = 50, 0, 0, 0, 0
    for _ in range(n_sim):
        Z, N = _nuisance_world(rng, T, 72)
        plain = neural_effect_search(Z, T)
        res = neural_effect_search(Z, T, nuisance=N)
        fr = plain['first_round']
        hit0_plain += bool(fr.loc[fr['neuron'] == 0, 'significant'].iloc[0])  # naive round-1 test flags it
        hit0 += 0 in res['selected']
        hit1 += 1 in res['selected']
        extra += len(set(res['selected']) - {1})
        assert res['n_tested'] == 200 and res['n_nuisance'] == 1 and all(j < 200 for j in res['selected'])
    print(f'    two-sample, {n_sim} sims: nuisance-driven neuron significant in the unconditioned round 1 {hit0_plain}/{n_sim}, selected '
          f'{hit0}/{n_sim} conditioned; true effect recovered {hit1}/{n_sim}; other selections {extra}')
    assert hit0_plain >= 0.7 * n_sim, 'the nuisance world is too weak to test anything'
    assert hit0 <= 0.1 * n_sim and hit1 >= 0.85 * n_sim and extra <= 0.15 * n_sim
    # nuisance=None is exactly the old search
    Z, N = _nuisance_world(rng, T, 72)
    a, b = neural_effect_search(Z, T), neural_effect_search(Z, T, nuisance=None)
    assert a['selected'] == b['selected'] and np.allclose(a['first_round']['p'], b['first_round']['p'])


def test_nuisance_conditioning_paired():
    """(f) paired analogue: the nuisance changes from stage a to b and drives neuron 0 only.
    Conditioning on a covariate that changes by mean(D_N) costs power exactly as for a selected
    neuron (intercept SE x sqrt(1 + mean(D_N)^2 / var(D_N)), here ~1.5), hence the larger true effect."""
    rng = np.random.default_rng(22)
    n, n_sim, hit0_plain, hit0, hit1, extra = 36, 50, 0, 0, 0, 0
    for _ in range(n_sim):
        U = rng.normal(0, 1, (n, 200))  # pool random effects
        Na, Nb = rng.normal(0, 1, n), 1.5 + rng.normal(0, 1, n)
        Za, Zb = U + rng.normal(0, 1, (n, 200)), U + rng.normal(0, 1, (n, 200))
        Za[:, 0], Zb[:, 0] = U[:, 0] + Na + rng.normal(0, .3, n), U[:, 0] + Nb + rng.normal(0, .3, n)
        Zb[:, 1] += 2.2
        plain = paired_effect_search(Za, Zb)
        res = paired_effect_search(Za, Zb, nuisance=(Na, Nb))
        fr = plain['first_round']
        hit0_plain += bool(fr.loc[fr['neuron'] == 0, 'significant'].iloc[0])  # naive round-1 test flags it
        hit0 += 0 in res['selected']
        hit1 += 1 in res['selected']
        extra += len(set(res['selected']) - {1})
    print(f'    paired, {n_sim} sims: nuisance-driven neuron significant in the unconditioned round 1 {hit0_plain}/{n_sim}, selected '
          f'{hit0}/{n_sim} conditioned; true effect recovered {hit1}/{n_sim}; other selections {extra}')
    assert hit0_plain >= 0.7 * n_sim
    assert hit0 <= 0.1 * n_sim and hit1 >= 0.85 * n_sim and extra <= 0.15 * n_sim


# ---- 2026-10-04 fix: shared slope (no extrapolation between per-arm fits), overlap rule, zero-SE rule -------------

WINDOW = 3600.8  # frogs latency window (s)


def _old_per_arm_tau(Y, X, T):
    """The estimator before 2026-10-04 (one stratum): per-arm OLS on X, both arms evaluated at the pooled mean of X.
    Kept here only to show the extrapolation it produced."""
    Xc = X - X.mean(0)
    est, var = [], []
    for t in (1, 0):
        ix = T == t
        e, v, _, _ = _ols_contrast(Y[ix], np.column_stack([np.ones(ix.sum()), Xc[ix]]), np.r_[1.0, np.zeros(X.shape[1])])
        est.append(e), var.append(v)
    return est[0] - est[1], var[0] + var[1]


def test_round1_unchanged_and_degenerate_untestable():
    """(a) round 1 (nothing conditioned) is still exactly Welch's t-test, in neural_effect_test and as the search's
    first-round table; a neuron constant within each arm (Welch variance 0, or ~1e-33 from rounding) is untestable
    (p = 1, testable False) instead of p ~ 0."""
    rng = np.random.default_rng(30)
    Z, T = two_sample(rng, 72, 1.0, 0.5)
    Z[:, 50] = np.where(T == 1, 0.3, 0.1) + 0.2  # constant within arms; float rounding can leave var ~1e-33
    table, info = neural_effect_test(Z, T)
    ref = stats.ttest_ind(Z[T == 1], Z[T == 0], equal_var=False)
    ok = np.arange(Z.shape[1]) != 50
    assert np.allclose(table['t'][ok], ref.statistic[ok], rtol=1e-12, atol=0)
    assert np.allclose(table['p'][ok], ref.pvalue[ok], rtol=1e-12, atol=0)
    assert table['testable'][ok].all() and not table['testable'][50] and table['p'][50] == 1.0
    fr = neural_effect_search(Z, T)['first_round']
    assert np.array_equal(fr['p'].values, table['p'].values) and np.array_equal(fr['tau'].values, table['tau'].values)
    print(f'    round 1: max |p - scipy Welch p| = {np.abs(table["p"][ok] - ref.pvalue[ok]).max():.1e}; '
          f'constant-within-arm neuron: se {table["se"][50]:.1e}, p {table["p"][50]}, testable {table["testable"][50]}')


def _frog38(rng, n_wt_spread):
    """Frogs neuron-38 shape. x = latency of the round-1 neuron: control (WT, 13) censored at the window except
    n_wt_spread videos, treated (FoxP1, 14) early (mean ~220 s). y = latency of the tested neuron: y = 2000 + 300 T
    - 0.3 x + noise in BOTH arms (true direct effect +300 s at equal x), noise sd 150 s."""
    n0, n1 = 13, 14
    x0 = np.full(n0, WINDOW)
    x1 = rng.exponential(220, n1)
    if n_wt_spread == 1:
        x0[0] = rng.uniform(100, 1500)
    else:  # partial overlap: 4 control and 4 treated videos interleaved in [800, 1500] s
        x0[:4] = [800, 1000, 1200, 1400] + rng.uniform(-50, 50, 4)
        x1[:4] = [900, 1100, 1300, 1500] + rng.uniform(-50, 50, 4)
    x = np.r_[x0, x1]
    T = np.r_[np.zeros(n0), np.ones(n1)].astype(int)
    y = 2000 + 300 * T - 0.3 * x + rng.normal(0, 150, n0 + n1)
    return y, x, T


def test_extrapolation_neuron38():
    """(b) the frogs neuron-38 extrapolation. Old per-arm fits compared at the pooled mean of x: |tau| exceeds the
    window. New code: with 12/13 control videos censored the arms share no support on x -> the round is untestable
    ('no overlap', the search stops); with partial overlap the shared-slope tau is bounded and near the truth (+300)."""
    rng = np.random.default_rng(31)
    old_big, n_sim = 0, 200
    for _ in range(n_sim):
        y, x, T = _frog38(rng, 1)
        Y = y[:, None].copy()
        # per-arm slopes estimated from noisy latencies: make the treated slope steep as in the real case (-5.3)
        Y[T == 1, 0] = 2300 - 5.3 * (x[T == 1] - x[T == 1].mean()) + rng.normal(0, 150, (T == 1).sum())
        tau_old, _ = _old_per_arm_tau(Y, x[:, None], T)
        old_big += abs(tau_old[0]) > WINDOW
        try:
            neural_effect_test(np.column_stack([x, Y]), T, S=[0], cols=[1])
            raise AssertionError('expected no overlap')
        except ValueError as err:
            assert str(err).startswith('no overlap'), err
    assert old_big >= 0.9 * n_sim, old_big
    # one search: round 1 picks x (huge raw effect), round 2 must stop with 'no overlap', not select y
    y, x, T = _frog38(rng, 1)
    Zs = np.column_stack([x, 2300 - 5.3 * (x - 220) * T + rng.normal(0, 150, 27), rng.normal(size=(27, 20))])
    res = neural_effect_search(Zs, T)
    assert res['selected'][:1] == [0] and 'no overlap' in res.get('stopped', ''), res.get('stopped')
    # partial overlap (4 + 4 videos in the common range): bounded, near the true +300
    err_new, err_old, taus = [], [], []
    for _ in range(n_sim):
        y, x, T = _frog38(rng, 4)
        tab, _ = neural_effect_test(np.column_stack([x, y]), T, S=[0], cols=[1])
        taus.append(tab['tau'][0])
        err_new.append(abs(tab['tau'][0] - 300))
        err_old.append(abs(_old_per_arm_tau(y[:, None], x[:, None], T)[0][0] - 300))
    taus = np.array(taus)
    print(f'    neuron-38 shape, {n_sim} sims: old |tau| > window in {old_big}/{n_sim}; new = no overlap in {n_sim}/{n_sim}; '
          f'search stopped: {res["stopped"]!r}')
    print(f'    partial overlap: new tau mean {taus.mean():.0f} (true 300), max |tau| {np.abs(taus).max():.0f}; '
          f'median abs error new {np.median(err_new):.0f} s vs old {np.median(err_old):.0f} s')
    assert np.abs(taus).max() < WINDOW and abs(taus.mean() - 300) < 60 and np.median(err_new) < np.median(err_old)


def test_zero_se_untestable():
    """(c) a neuron constant in one arm and in the other except one video that the conditioning neuron fits exactly
    (the frogsfull neuron-618 shape: fires in 1 of 27 videos): residuals are 0, SE ~1e-13 -> untestable, p = 1.
    The old per-arm estimator gave a non-zero tau over an SE of ~0 (p ~ 0) there."""
    T = np.r_[np.zeros(13), np.ones(14)].astype(int)
    x = np.zeros(27)
    x[-1], x[-5:-2] = 1.0, [0.2, 0.4, 0.6]  # selected neuron (overlap at 0: 13 / 11 units)
    y = np.full(27, WINDOW)
    y[-1] = 100.0  # the one video where the tested neuron fires
    y[-5:-2] = WINDOW - 3500.8 * np.array([0.2, 0.4, 0.6])  # on the same line: exact fit within the treated arm
    tab, _ = neural_effect_test(np.column_stack([x, y]), T, S=[0], cols=[1])
    tau_old, V_old = _old_per_arm_tau(y[:, None], x[:, None], T)
    p_old = 2 * stats.t.sf(abs(tau_old[0]) / np.sqrt(V_old[0]), 10) if V_old[0] > 0 else 1.0
    print(f'    one-video neuron: old tau {tau_old[0]:.1f}, se {np.sqrt(V_old[0]):.1e}, p {p_old:.1e}; '
          f'new tau {tab["tau"][0]:.1e}, se {tab["se"][0]:.1e}, p {tab["p"][0]}, testable {tab["testable"][0]}')
    assert not tab['testable'][0] and tab['p'][0] == 1.0
    # paired: a neuron whose change is constant across pools is untestable as well
    rng = np.random.default_rng(32)
    Za = rng.normal(size=(36, 5))
    Zb = Za.copy()
    Zb[:, 0] += 0.7
    Zb[:, 1:] += rng.normal(size=(36, 4))
    ptab, _ = paired_effect_test(Za, Zb)
    assert not ptab['testable'][0] and ptab['p'][0] == 1.0 and ptab['testable'][1:].all()


def test_shared_slope_recovers_effect():
    """(d) a real direct effect with overlapping covariates: x is shifted by T (a found concept, d = 1 sd) and y =
    0.5 x + 0.8 T + noise. The shared-slope tau given x recovers 0.8 (unbiased) with ~95% CI coverage; the overlap
    rule does not fire."""
    rng = np.random.default_rng(33)
    taus, cover, n_sim = [], 0, 400
    T = np.repeat([0, 1], 36)
    for _ in range(n_sim):
        x = 1.0 * T + rng.normal(size=72)
        y = 0.5 * x + 0.8 * T + rng.normal(size=72)
        tab, info = neural_effect_test(np.column_stack([x, y]), T, S=[0], cols=[1])
        r = tab.iloc[0]
        taus.append(r['tau'])
        cover += abs(r['tau'] - 0.8) <= stats.t.ppf(0.975, r['df']) * r['se']
    taus = np.array(taus)
    print(f'    {n_sim} sims: mean tau {taus.mean():.3f} (true 0.8), sd {taus.std():.3f}, 95% CI coverage {cover / n_sim:.3f}')
    assert abs(taus.mean() - 0.8) < 0.03 and 0.92 <= cover / n_sim <= 0.98
    check_overlap(np.column_stack([1.0 * T + rng.normal(size=72)]), T)  # no error


def test_support_along_adjustment_direction():
    """(e) several conditioning neurons, each overlapping between the arms on its own, while the combination the
    tested neuron is adjusted along separates them: x1 - x2 = +-2 by arm. The round-level rule passes, the
    neuron-level rule (common range of the adjustment score s_j = X b_j) makes the neuron untestable; a neuron
    adjusted along x1 + x2 (overlapping) stays testable (its tau is imprecise: T is nearly collinear with x1 - x2)."""
    rng = np.random.default_rng(34)
    T = np.repeat([0, 1], 36)
    u = rng.normal(0, 3, 72)
    x1 = u + np.where(T == 1, 1.0, -1.0) + rng.normal(0, 0.3, 72)
    x2 = u - np.where(T == 1, 1.0, -1.0) + rng.normal(0, 0.3, 72)
    check_overlap(np.column_stack([x1, x2]), T)  # marginal overlap: no error
    y_sep = 400 * (x1 - x2) + rng.normal(0, 1, 72)  # adjusted along the separating combination
    y_ok = 0.5 * (x1 + x2) + 0.3 * T + rng.normal(0, 1, 72)
    tab, _ = neural_effect_test(np.column_stack([x1, x2, y_sep, y_ok]), T, S=[0, 1], cols=[2, 3])
    print(f'    separating combination: support {bool(tab["support"][0])}, p {tab["p"][0]}; '
          f'overlapping combination: support {bool(tab["support"][1])}, tau {tab["tau"][1]:.2f} +- {tab["se"][1]:.2f} (true 0.3)')
    assert not tab['support'][0] and not tab['testable'][0] and tab['p'][0] == 1.0
    assert tab['support'][1] and tab['testable'][1]


def test_paired_support():
    """(f) paired: the paired tau is the change of neuron j at D_S = 0 (no change of the found concepts). When the
    found concept increases in every pool (all D_1 > 0) that point is outside the data: round-level 'no overlap'
    (the search stops after round 1); with >= 3 pools on each side of 0 the test runs."""
    rng = np.random.default_rng(35)
    Za = rng.normal(size=(36, 30))
    Zb = Za + rng.normal(size=(36, 30))
    Zb[:, 0] = Za[:, 0] + 3 + np.abs(rng.normal(size=36))  # every pool increases
    Zb[:, 1] = Za[:, 1] + 2.0 * (Zb[:, 0] - Za[:, 0]) + rng.normal(0, 0.5, 36)  # leaks neuron 0's change
    try:
        check_overlap_paired((Zb - Za)[:, [0]])
        raise AssertionError('expected no overlap')
    except ValueError as err:
        assert str(err).startswith('no overlap'), err
    res = paired_effect_search(Za, Zb)  # round 1 picks neuron 1 or 0 (both change in every pool), round 2 stops
    assert len(res['selected']) == 1 and 'round 2: no overlap' in res.get('stopped', ''), (res['selected'], res.get('stopped'))
    D = rng.normal(0.5, 1, 36)
    check_overlap_paired(D[:, None])  # ~11 pools below 0: no error
    print(f'    all pools change: selected {res["selected"]}, stopped {res["stopped"]!r}')


def main() -> int:
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        t0 = time.time()
        try:
            t()
            print(f"  PASS  {t.__name__}  ({time.time() - t0:.1f}s)")
        except Exception:
            failed += 1
            print(f"  FAIL  {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
