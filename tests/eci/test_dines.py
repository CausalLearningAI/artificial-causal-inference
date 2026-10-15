"""Simulation tests for DINES (src/eci/dines.py): linear-Gaussian toys with known structure.

Every toy has the factor atoms listed per case plus 20 noise atoms (10 Gaussian, 10 sparse ReLU codes, so the firing
indicators of the nuisance models are exercised). T ~ Bern(1/2), every noise term N(0, 1), exact alignment (each
factor's atom IS the factor) unless stated. Truth = the direct set (principal atoms of T's children).

  (a) two children            T -> C1, T -> C2                       direct {C1, C2}
  (b) mediator                T -> C -> s                            direct {C}; s must be dropped ({C} separates it)
  (c) spouse                  Sp -> C <- T, Sp -> s, C -> s          direct {C}; Sp and s must be dropped; s needs
                                                                     {C, Sp} as separator (Sp is needed)
  (d) descendant spouse       T -> A -> B -> Wd -> C <- T            direct {A, C}; B and Wd separated by {A} (Lemma 5)
  (e) global null             the factor graph of (d) with every edge out of T removed: select nothing in >= 90% of runs
  (f) noisy proxy             T -> C -> D, atom of C = C + 0.5 N(0, 1)  direct {C}; D leaks (its effect is not removed by
                                                                     a noisy proxy) and is selected at delta = 0 when n
                                                                     is large; delta = 0.05 removes it
  (g) treatment copy          Cp = T + 0.01 N(0, 1), T -> P -> V       Cp reveals T (no overlap given Cp). Theory Section
                                                                     2.9: Cp is set aside in the forward phase (never
                                                                     conditioned on), Cp and P are REPORTED (P flagged
                                                                     undetermined: given Cp it has no testable effect),
                                                                     V is dropped via {P}
'Selected' below = DINES's reported set (kept u undetermined).

Asserted (50 seeds per case, alpha = 0.05): exact recovery >= 0.85 for (a)-(d) at n = 2000 and delta = 0; (e) nothing
selected in >= 90% of 200 runs (the 50-seed rate is reported too); (f) at n = 16000 the leak D is selected in >= 50% of runs at delta = 0 and exact recovery
>= 0.90 at delta = 0.05; (g) Cp and P reported and V not reported in >= 90% of runs, Cp set aside in >= 90%. The measured rates (and NES on the same data, for comparison only) are printed.

Runs standalone (`python tests/eci/test_dines.py`), every test_* function is also pytest-collectable. Run it through
Slurm: it takes several minutes of CPU.
"""
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from eci.dines import _gcm_from_residuals, _subset_order, crossfit_gcm, dines, linear_gcm  # noqa: E402
from eci.nes import neural_effect_search  # noqa: E402

ALPHA, N_SEEDS, N_NOISE = 0.05, 50, 20


def _noise_atoms(rng, n):
    gauss = rng.normal(size=(n, N_NOISE // 2))
    relu = np.maximum(rng.normal(-0.5, 1.0, size=(n, N_NOISE // 2)), 0.0)
    return np.column_stack([gauss, relu])


def toy(case, rng, n, beta=0.5):
    """-> Z (n, m), T, truth (set of atom indices), roles {name: atom index}. Factor atoms come first."""
    T = rng.binomial(1, 0.5, n).astype(np.int64)
    e = lambda: rng.normal(size=n)  # noqa: E731
    if case == 'a':
        C1, C2 = beta * T + e(), beta * T + e()
        F, names, truth = [C1, C2], ['C1', 'C2'], {'C1', 'C2'}
    elif case == 'b':
        C = beta * T + e()
        s = 0.8 * C + e()
        F, names, truth = [C, s], ['C', 's'], {'C'}
    elif case == 'c':
        Sp = e()
        C = 0.8 * T + 0.8 * Sp + e()
        s = 0.8 * Sp + 0.8 * C + e()
        F, names, truth = [C, Sp, s], ['C', 'Sp', 's'], {'C'}
    elif case in ('d', 'e'):
        b = beta if case == 'd' else 0.0
        A = b * T + e()
        B = 0.8 * A + e()
        Wd = 0.8 * B + e()
        C = b * T + 0.8 * Wd + e()
        F, names, truth = [A, B, Wd, C], ['A', 'B', 'Wd', 'C'], ({'A', 'C'} if case == 'd' else set())
    elif case == 'g':
        Cp = T + 0.01 * e()
        P = beta * T + e()
        V = 0.8 * P + e()
        F, names, truth = [Cp, P, V], ['Cp', 'P', 'V'], {'Cp', 'P'}
    elif case == 'f':
        C = beta * T + e()
        D = 0.8 * C + e()
        F, names, truth = [C + 0.5 * e(), D], ['C', 'D'], {'C'}
    else:
        raise ValueError(case)
    Z = np.column_stack(F + [_noise_atoms(rng, n)])
    roles = {nm: i for i, nm in enumerate(names)}
    return Z, T, {roles[t] for t in truth}, roles


def run_case(case, n, delta, n_seeds=N_SEEDS, nes=False, **kw):
    """Run DINES (and optionally the repo NES) on n_seeds draws. Returns a dict of rates and per-role selection rates."""
    exact, any_sel, sizes, role_sel, nes_exact, nes_role, fwd_sizes, und = 0, 0, [], {}, 0, {}, [], 0
    role_und, role_aside = {}, {}
    t0 = time.time()
    for s in range(n_seeds):
        rng = np.random.default_rng(1000 * ord(case) + s)
        Z, T, truth, roles = toy(case, rng, n)
        res = dines(Z, T, alpha=ALPHA, delta=delta, seed=s, **kw)
        sel = set(res['selected'])
        exact += sel == truth
        any_sel += len(sel) > 0
        sizes.append(len(sel))
        fwd_sizes.append(len(res['forward_set']))
        und += len(res['undetermined']) > 0
        for nm, j in roles.items():
            role_sel[nm] = role_sel.get(nm, 0) + (j in sel)
            role_und[nm] = role_und.get(nm, 0) + (j in res['undetermined'])
            role_aside[nm] = role_aside.get(nm, 0) + (j in res['set_aside'])
        if nes:
            r = neural_effect_search(Z, T, alpha=ALPHA)
            sn = set(r['selected'])
            nes_exact += sn == truth
            for nm, j in roles.items():
                nes_role[nm] = nes_role.get(nm, 0) + (j in sn)
    out = {'case': case, 'n': n, 'delta': delta, 'seeds': n_seeds, 'exact': exact / n_seeds,
           'any_selected': any_sel / n_seeds, 'mean_size': float(np.mean(sizes)), 'mean_forward': float(np.mean(fwd_sizes)),
           'undetermined_runs': und / n_seeds, 'role_rate': {k: v / n_seeds for k, v in role_sel.items()},
           'role_undetermined': {k: v / n_seeds for k, v in role_und.items()},
           'role_set_aside': {k: v / n_seeds for k, v in role_aside.items()},
           'seconds': time.time() - t0}
    if nes:
        out['nes_exact'] = nes_exact / n_seeds
        out['nes_role_rate'] = {k: v / n_seeds for k, v in nes_role.items()}
    return out


REPORT = []


def _report(r):
    REPORT.append(r)
    roles = ' '.join(f'{k}:{v:.2f}' for k, v in r['role_rate'].items())
    nes = f" | NES exact {r['nes_exact']:.2f} ({' '.join(f'{k}:{v:.2f}' for k, v in r['nes_role_rate'].items())})" \
        if 'nes_exact' in r else ''
    print(f"    ({r['case']}) n={r['n']:>5} delta={r['delta']:<4} exact {r['exact']:.2f}  any {r['any_selected']:.2f}  "
          f"|S| {r['mean_size']:.2f}  |S~| {r['mean_forward']:.2f}  undetermined {r['undetermined_runs']:.2f}  "
          f"selected-rate {roles}{nes}  ({r['seconds']:.0f}s)")


# ---------------------------------------------------------------------------------------------------------------
# small deterministic checks
# ---------------------------------------------------------------------------------------------------------------

def test_gcm_matches_formula():
    rng = np.random.default_rng(0)
    rT, RZ = rng.normal(size=500), rng.normal(size=(500, 3))
    RZ[:, 0] += 0.3 * rT
    res = _gcm_from_residuals(rT, RZ)
    R = rT[:, None] * RZ
    z = np.sqrt(500) * R.mean(0) / R.std(0)
    assert np.allclose(res['stat'], z), (res['stat'], z)
    rho = R.mean(0) / np.sqrt((rT ** 2).mean() * (RZ ** 2).mean(0))
    assert np.allclose(res['rho'], rho)
    # delta > 0 makes every p larger; delta = 0 with the delta-method form must be close to the GCM
    r2 = _gcm_from_residuals(rT, RZ, delta=0.05)
    assert (r2['logp'] >= res['logp'] - 1e-12).all()
    r0 = _gcm_from_residuals(rT, RZ, delta=1e-12)
    assert np.allclose(np.exp(r0['logp'][1:]), np.exp(res['logp'][1:]), atol=0.02)


def test_subset_order():
    order, complete = _subset_order([3, 5, 7], None)
    assert complete and order[0] == () and set(order) == {(), (5, 7), (3, 7), (3, 5), (3,), (5,), (7,)}
    assert order[1:4] == [(5, 7), (3, 7), (3, 5)]
    order, complete = _subset_order(list(range(14)), 100)
    assert not complete and len(order) == 100
    assert _subset_order([4], None) == ([()], True)


def test_first_step_known_propensity():
    """A = {} without W: residuals of T are T - mean(T), the linear and the cross-fitted test agree."""
    rng = np.random.default_rng(1)
    T = rng.binomial(1, 0.5, 400)
    Z = rng.normal(size=(400, 4)) + 0.8 * T[:, None] * np.array([1, 0, 0, 0])
    a = linear_gcm(Z, T)
    b = crossfit_gcm(Z, T, (), [0, 1, 2, 3], folds=np.zeros(400, dtype=int))
    assert np.allclose(a['stat'], b['stat']) and a['logp'][0] < np.log(1e-4)


def test_clusters_inflate_se():
    """Cluster-shared noise: the cluster-robust p-value of a null atom is larger than the naive one on average."""
    rng = np.random.default_rng(2)
    worse = 0
    for _ in range(20):
        g = np.repeat(np.arange(40), 25)
        T = np.repeat(rng.binomial(1, 0.5, 40), 25)
        Z = (np.repeat(rng.normal(size=(40, 1)), 25, 0) + 0.3 * rng.normal(size=(1000, 1)))
        naive = linear_gcm(Z, T)['logp'][0]
        robust = linear_gcm(Z, T, groups=g)['logp'][0]
        worse += robust > naive
    assert worse >= 15, worse


# ---------------------------------------------------------------------------------------------------------------
# recovery on the toys (50 seeds each)
# ---------------------------------------------------------------------------------------------------------------

def test_a_two_children():
    r = run_case('a', 2000, 0.0, nes=True)
    _report(r)
    assert r['exact'] >= 0.85, r


def test_b_mediator():
    r = run_case('b', 2000, 0.0, nes=True)
    _report(r)
    assert r['exact'] >= 0.85 and r['role_rate']['s'] <= 0.1, r


def test_c_spouse_separator():
    r = run_case('c', 2000, 0.0, nes=True)
    _report(r)
    assert r['exact'] >= 0.85 and r['role_rate']['s'] <= 0.1 and r['role_rate']['Sp'] <= 0.1, r


def test_d_descendant_spouse():
    r = run_case('d', 2000, 0.0, nes=True)
    _report(r)
    assert r['exact'] >= 0.85, r


def test_e_global_null():
    """The 50-seed rate is reported; the assertion uses 200 seeds (seeds 0-199, the first 50 included): with a
    family-wise rate of ~0.05, 50 seeds fail a 90% bar by chance in ~5% of draws (seeds 0-49 at n = 500 gave 6/50 for
    DINES and for NES's Welch test alike; 400 further seeds gave 0.055)."""
    for n in (500, 2000):
        _report(run_case('e', n, 0.0, nes=True))
        r = run_case('e', n, 0.0, n_seeds=200)
        _report(r)
        assert 1 - r['any_selected'] >= 0.90, r


def test_f_noisy_proxy_delta():
    for n in (2000, 16000):
        r0 = run_case('f', n, 0.0, nes=True)
        _report(r0)
        r1 = run_case('f', n, 0.05)
        _report(r1)
        if n == 16000:
            assert r0['role_rate']['D'] >= 0.5, r0
            assert r1['exact'] >= 0.90, r1


def test_g_treatment_copy():
    r = run_case('g', 2000, 0.0, nes=True)
    _report(r)
    print(f"        undetermined-rate {r['role_undetermined']}  set-aside-rate {r['role_set_aside']}")
    rr = r['role_rate']
    assert rr['Cp'] >= 0.9 and rr['P'] >= 0.9 and rr['V'] <= 0.1 and r['role_set_aside']['Cp'] >= 0.9, r


def test_delta_on_exact_toys():
    """Report only: delta = 0.05 on the exactly aligned toys (recall cost at moderate n)."""
    for case in 'abcd':
        _report(run_case(case, 2000, 0.05))


def main() -> int:
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        t0 = time.time()
        try:
            t()
            print(f"  PASS  {t.__name__}  ({time.time() - t0:.1f}s)", flush=True)
        except Exception:
            failed += 1
            print(f"  FAIL  {t.__name__}", flush=True)
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    if len(sys.argv) > 1:  # optional JSON dump of the measured rates
        import json
        Path(sys.argv[1]).write_text(json.dumps(REPORT, indent=1))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
