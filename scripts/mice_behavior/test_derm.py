#!/usr/bin/env python3
"""Audit `train_online_aug.derm_table` against the formula it claims to implement.

Run this before trusting any DERM arm. It is a property test, not a smoke test: every check below
is an identity that must hold exactly (to float tolerance), so a regression shows up as a failure
rather than as a slightly different number nobody notices.

WHAT IS CHECKED
===============
1. CLOSED FORM. w = Var(Y|E) / P(Y, E). For a binary label that collapses to
   w(y=1,e) = (1-p_e)/P(e) and w(y=0,e) = p_e/P(e). Checked against BOTH forms independently, so
   an error in the algebraic simplification cannot hide behind the simplification.
2. INVARIANT A. Positives and negatives carry EQUAL total mass inside every environment. This is
   the property that removes the label-environment association, i.e. the whole point.
3. INVARIANT B. Each environment's total mass is proportional to its own outcome variance, so an
   environment where nothing varies stops dominating by sheer size.
4. NORMALISATION. Mean weight over the training index is exactly 1, so the effective step size is
   identical to ERM and a DERM-vs-ERM comparison is not confounded by having changed the learning
   rate.
5. PER LABEL. nt and nn get their own correction. Collapsing them (which
   `src/ppci/dataset.py::compute_derm_weights` does, with a warning) would apply nt's correction to
   nn -- they sit at different prevalences with different phase profiles.
6. FLOOR. `floor` clips Var(Y|E) from below, and it costs ONE of DERM's two channels, not both:
   invariant B goes (every environment gets the same mass regardless of its variance), invariant A
   stays (positives and negatives keep equal mass inside each environment, at a measured 40x-111x
   per-sample weight ratio), and the table stays per label because P(y,e) is untouched. An earlier
   version of this file asserted the opposite -- that a binding floor collapses every label onto
   one shared table -- and failed for that reason.
   On the RAW unsampled frames the 2% default clips nt in all 3 phases (prevalence 0.89-1.24%) and
   NO phase for nn (2.03-2.43% on the directed-pair union; the retired mutual-only column read
   0.68-0.89% and was below the floor, which is where the "all below" claim came from).
7. ENVIRONMENT MAPPING. The sample -> observation -> phase map does a searchsorted over
   per-observation start rows in annotations.csv, which is exact only if each observation occupies
   one contiguous ascending block. Checked directly, and the resulting phase proportions are
   checked against the protocol's own 30/15/15-minute design.
9. POPULATION PREVALENCE. `--derm-prevalence population` reads Var(Y|E) off the annotations rather
   than off the bounded anchor set. It had no test at all, which is how it spent a long time
   reading the mutual-only nn column. Checked against an independent recomputation, checked to be
   the union, and checked to actually reach the weights.
10. THE nn TRUTH. `read_truth()` returns Y_nn OR Y_np, and that is pinned against the head's own
   target from pair_labels.parquet: 1 frame of 864,000 disagrees with the union (nt disagrees with
   Y_nt on the same 1, so it is a shared boundary residual) against 11,754 for mutual-only.

    python scripts/mice_behavior/test_derm.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.mice_behavior.truth import read_truth                              # noqa: E402

FRAME = ROOT / 'results' / 'vision' / 'mice' / 'frame'
TOL = 1e-5
LABELS = ('nt', 'nn')
FAIL: list[str] = []


def check(name: str, ok: bool, detail: str = '') -> None:
    print(f'  [{"ok  " if ok else "FAIL"}] {name}' + (f'   {detail}' if detail else ''))
    if not ok:
        FAIL.append(name)


def load_module():
    """Import the functions under test without running the module's argparse main."""
    spec = importlib.util.spec_from_file_location(
        '_toa', Path(__file__).parent / 'train_online_aug.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules['_toa'] = mod
    try:
        spec.loader.exec_module(mod)
    except SystemExit:                      # the module calls main() under argparse on import
        pass
    return mod


def test_formula(derm_table) -> None:
    """Both forms of the weight, and all three invariants, on data with KNOWN parameters."""
    print('\n1-5. formula and invariants, on synthetic data with known p_e and P(e)')
    rng = np.random.default_rng(0)
    n_env, L, N = 3, 2, 200_000
    P_true = np.array([0.5, 0.3, 0.2])
    p_true = np.array([[0.10, 0.40], [0.25, 0.05], [0.50, 0.30]])
    e = rng.choice(n_env, size=N, p=P_true)
    y = (rng.random((N, L)) < p_true[e]).astype(np.float32)

    tab, m_raw = derm_table(y, e, np.arange(N), n_env, floor=0.0)
    cnt = np.bincount(e, minlength=n_env).astype(float)
    P_e = cnt / N
    p_e = np.stack([np.bincount(e, weights=(y[:, l] > 0.5).astype(float),
                                minlength=n_env) / cnt for l in range(L)], axis=1)
    w = tab[e[:, None], np.arange(L)[None, :], (y > 0.5).astype(np.int64)]

    d1 = np.abs(tab[:, :, 1] * m_raw - (1 - p_e) / P_e[:, None]).max()
    d0 = np.abs(tab[:, :, 0] * m_raw - p_e / P_e[:, None]).max()
    check('w(y=1,e) = (1-p_e)/P(e)', d1 < TOL, f'max dev {d1:.2e}')
    check('w(y=0,e) = p_e/P(e)', d0 < TOL, f'max dev {d0:.2e}')

    # the general form, computed independently rather than via the simplification
    dv = 0.0
    for l in range(L):
        var = p_e[:, l] * (1 - p_e[:, l])
        for k, joint in ((1, P_e * p_e[:, l]), (0, P_e * (1 - p_e[:, l]))):
            dv = max(dv, np.abs(tab[:, l, k] * m_raw - var / joint).max())
    check('w = Var(Y|E) / P(Y,E), computed independently', dv < TOL, f'max dev {dv:.2e}')

    check('mean weight over the training index is 1', abs(w.mean() - 1.0) < TOL,
          f'{w.mean():.8f}')

    worst_mass, worst_var = 0.0, 0.0
    for l in range(L):
        for ee in range(n_env):
            s = e == ee
            mp = w[s & (y[:, l] > 0.5), l].sum()
            mn = w[s & (y[:, l] <= 0.5), l].sum()
            worst_mass = max(worst_mass, abs(mp / mn - 1.0))
        tot = np.array([w[e == ee, l].sum() for ee in range(n_env)])
        r = tot / (p_e[:, l] * (1 - p_e[:, l]))
        worst_var = max(worst_var, np.abs(r / r[0] - 1.0).max())
    check('INVARIANT A: equal pos/neg mass inside every environment',
          worst_mass < 1e-4, f'worst dev {worst_mass:.2e}')
    check('INVARIANT B: environment mass proportional to Var(Y|E)',
          worst_var < 1e-4, f'worst dev {worst_var:.2e}')

    # per label, not a mean over labels: label 0 and label 1 have different profiles here, so a
    # collapsed implementation would give them the same table
    same = np.allclose(tab[:, 0, :], tab[:, 1, :])
    check('PER LABEL: nt and nn get different tables', not same,
          'a collapsed version would make these identical')


def test_floor(derm_table, derm_mass) -> None:
    """What a binding floor actually costs -- which is ONE of DERM's two channels, not both.

    The floor clips Var(Y|E=e) from below. When it clips every environment for a label, that
    label's numerator is the same constant in every environment, so:

      LOST      invariant B. Environment mass was meant to track Var(Y|E); it becomes uniform,
                and an environment where the behaviour barely varies stops being downweighted.
      KEPT      invariant A. Positives and negatives still carry equal mass inside every
                environment, because the denominator P(y, e) is untouched -- and that is the
                channel that removes the prevalence-environment association, i.e. the main
                mechanism. The table also stays LABEL-SPECIFIC for the same reason: P(y, e) is
                per label even when Var is not.

    So a binding floor is a partial degradation, not an exact no-op, and the old version of this
    test asserted the opposite -- that every label collapses onto one shared table. It does not,
    and the test failed for that reason rather than because anything was broken.
    """
    print('\n6. floor. It clips Var(Y|E); what survives the clip, and what does not.')
    exp = pd.read_csv(ROOT / 'data' / 'mice' / 'v1' / 'experiment.csv')
    ann = read_truth().merge(exp[['observation_id', 'pool', 'phase']], on='observation_id')
    cfg_p = FRAME / 'res448_k2_frozen_d4photo_dermPhase' / 'config.json'
    if not cfg_p.exists():
        print('     [skip] no dermPhase config on disk')
        return
    val = set(json.load(open(cfg_p))['val_pools'])
    tr = ann[~ann.pool.isin(val)]
    names, ev = np.unique(tr.phase.to_numpy(), return_inverse=True)
    Y = np.stack([(tr.Y_nt > 0.5).to_numpy(), (tr.Y_nn > 0.5).to_numpy()], 1).astype(np.float32)
    raw_p = np.stack([np.bincount(ev, weights=Y[:, l], minlength=len(names))
                      / np.bincount(ev) for l in range(2)], axis=1)
    v_floor = 0.02 * (1 - 0.02)
    raw_var = raw_p * (1 - raw_p)
    for l, nm in enumerate(LABELS):
        clipped = int((raw_var[:, l] < v_floor).sum())
        print(f'     raw per-phase prevalence, {nm}: '
              f'{100 * raw_p[:, l].min():.2f}%-{100 * raw_p[:, l].max():.2f}%  '
              f'Var {raw_var[:, l].min():.5f}-{raw_var[:, l].max():.5f}  '
              f'-> the 2.0% floor ({v_floor:.4f}) clips {clipped}/{len(names)} environments')
    # nt sits below the floor in every phase and nn sits above it in every phase, on the
    # directed-pair union the heads are actually trained on. Asserted per label rather than
    # as one "all below" claim, which was true only of the mutual-only nn column.
    check('nt: the floor clips EVERY environment on unsampled frames',
          (raw_var[:, LABELS.index('nt')] < v_floor).all(),
          'this is why NEG_RATIO=1 is load-bearing, not incidental')
    check('nn: the floor clips NO environment (the union clears 2%, mutual-only did not)',
          (raw_var[:, LABELS.index('nn')] >= v_floor).all(),
          'so --derm-floor 0.02 is only half-binding here, and only on nt')

    # THE SIGNATURE OF A FULLY BINDING FLOOR, on record. Forced with a floor high enough to clip
    # both labels, so the regime is exercised rather than argued about.
    idx = np.arange(len(Y)); e64 = ev.astype(np.int64)
    tab_bad, m_bad = derm_table(Y, e64, idx, len(names), floor=0.25)
    mass_bad = derm_mass(tab_bad, Y, e64, idx, len(names))
    check('a binding floor does NOT collapse the labels onto one table',
          not np.allclose(tab_bad[:, 0, :], tab_bad[:, 1, :]),
          f'max relative gap between the nt and nn tables '
          f'{float(np.abs(tab_bad[:, 0, :] / tab_bad[:, 1, :] - 1).max()):.2f} '
          f'-- P(y,e) stays per label; derm_w_raw would log as {m_bad:.2f}')
    ratio = tab_bad[:, :, 1] / tab_bad[:, :, 0]
    check('a binding floor KEEPS invariant A: positives still outweigh negatives',
          ratio.min() > 10, f'pos/neg weight ratio {ratio.min():.0f}x-{ratio.max():.0f}x '
                            f'across (environment, label)')
    per_env = mass_bad.sum(2)
    check('what the floor DOES cost is invariant B: environment mass goes uniform',
          float(np.abs(per_env * len(names) - 1).max()) < 1e-4,
          f'every environment lands on {1 / len(names):.4f} of the mass regardless of its Var')

    # and what the runs actually logged
    phi = 0.02
    degen = 2 * len(names) * phi * (1 - phi) / 1.0
    print(f'     if the floor bound, the runs would log derm_w_raw = {degen:.3f}; '
          f'they log 1.09 ({1.09 / degen:.1f}x)')
    check('the landed runs are NOT in the degenerate regime', 1.09 > 5 * degen,
          'derm_w_raw = 1.09 inverts to a mean sampled p_e of ~0.24, the 1:1 regime')


def test_population_var(mod) -> None:
    """The `--derm-prevalence population` path: read off the annotations, not off the anchors.

    Never had a test, which is how it went a long time reading the mutual-only `Y_nn` column --
    estimating the confound for a label no head predicts. Three things are checked: the value is
    p(1-p) on the population, it is computed on the SAME union the heads train on, and feeding it
    to derm_table as `var_pop` actually changes the environment masses.
    """
    print('\n9. the population-prevalence path (population_var)')
    exp = pd.read_csv(ROOT / 'data' / 'mice' / 'v1' / 'experiment.csv')
    cfg_p = FRAME / 'res448_k2_frozen_d4photo_dermPhase' / 'config.json'
    if not cfg_p.exists():
        print('     [skip] no dermPhase config on disk')
        return
    val = set(json.load(open(cfg_p))['val_pools'])
    ann_csv = ROOT / 'dataset' / 'mice' / 'v1' / 'annotations.csv'
    lab = exp[exp.annotation_file.notna()]
    train_obs = sorted(lab.loc[~lab.pool.isin(val), 'observation_id'])
    o2e = dict(zip(exp.observation_id, exp.phase))
    env_names = ['H', 'O', 'P']
    vp = mod.population_var(ann_csv, train_obs, o2e, env_names)
    check('population_var returns a finite Var for every (environment, label)',
          bool(np.isfinite(vp).all()), np.array2string(vp, precision=5))

    # independent recomputation, straight off the truth reader
    t = read_truth()
    t = t[t.observation_id.isin(set(train_obs))].assign(_e=lambda d: d.observation_id.map(o2e))
    want = np.array([[float((t.loc[t._e == e, 'Y_' + l] > 0.5).mean()) for l in LABELS]
                     for e in env_names])
    want = want * (1 - want)
    check('population_var == p(1-p) on the training observations',
          float(np.abs(vp - want).max()) < 1e-9, f'max dev {float(np.abs(vp - want).max()):.2e}')

    # and that it is the UNION, not the mutual-only column that used to be read here
    raw = pd.read_csv(ann_csv, usecols=['observation_id', 'Y_nn'],
                      low_memory=False).dropna(subset=['Y_nn'])
    raw = raw[raw.observation_id.isin(set(train_obs))].assign(_e=lambda d:
                                                              d.observation_id.map(o2e))
    mut = np.array([float((raw.loc[raw._e == e, 'Y_nn'] > 0.5).mean()) for e in env_names])
    uni = np.array([float((t.loc[t._e == e, 'Y_nn'] > 0.5).mean()) for e in env_names])
    check('nn prevalence is the DIRECTED-PAIR UNION, not the mutual-only column',
          float(np.abs(uni - mut).min()) > 1e-3,
          'union ' + ', '.join(f'{n} {100 * u:.2f}%' for n, u in zip(env_names, uni))
          + '  vs mutual-only ' + ', '.join(f'{100 * m:.2f}%' for m in mut))

    # does var_pop actually reach the weights? masses must track it, not the sampled variance.
    a = read_truth().merge(exp[['observation_id', 'pool', 'phase']], on='observation_id')
    tr = a[~a.pool.isin(val)]
    names, ev = np.unique(tr.phase.to_numpy(), return_inverse=True)
    Y = np.stack([(tr['Y_' + l] > 0.5).to_numpy() for l in LABELS], 1).astype(np.float32)
    idx = np.arange(len(Y)); e64 = ev.astype(np.int64)
    tab, _ = mod.derm_table(Y, e64, idx, len(names), floor=0.0, var_pop=vp)
    mass = mod.derm_mass(tab, Y, e64, idx, len(names)).sum(2)
    dev = max(float(np.abs((mass[:, l] / mass[0, l]) / (vp[:, l] / vp[0, l]) - 1).max())
              for l in range(len(LABELS)))
    check('var_pop reaches the weights: environment mass is proportional to it',
          dev < 1e-4, f'worst dev {dev:.2e}')

    # the shipped launch guard, evaluated on the real numbers rather than assumed
    for phi in (0.02, 0.25):
        vf = phi * (1 - phi)
        print(f'     --derm-floor {phi:g} -> v_floor {vf:.4f}: below every population Var? '
              f'{bool((vp < vf).all())}   (max Var {vp.max():.5f})')


def test_truth_definition() -> None:
    """The nn TRUTH is the directed-pair union, and it is what the heads are trained on.

    This is the check that was missing while ten read sites compared the model against the
    mutual-only `Y_nn` column. It pins the definition against the two things that fix it: the
    raw columns it is built from, and the pair-label table the training targets come from.
    """
    print('\n10. the nn truth definition (union of the mutual and directional codes)')
    t = read_truth()
    raw = pd.read_csv(ROOT / 'dataset' / 'mice' / 'v1' / 'annotations.csv',
                      usecols=['observation_id', 'frame_idx', 'Y_nn', 'Y_np', 'Y_nt'],
                      low_memory=False).dropna(subset=['Y_nt'])
    check('read_truth drops every unlabelled row (they are NaN, not 0)',
          len(t) == len(raw) and not t[['Y_nt', 'Y_nn']].isna().any().any(),
          f'{len(t):,} rows over {t.observation_id.nunique()} annotated observations')
    union = ((raw.Y_nn > 0.5) | (raw.Y_np > 0.5)).to_numpy()
    check('read_truth().Y_nn == Y_nn OR Y_np',
          bool(((t.Y_nn.to_numpy() > 0.5) == union).all()),
          f'union fires on {100 * union.mean():.3f}% of frames against mutual-only '
          f'{100 * (raw.Y_nn > 0.5).mean():.3f}%')

    # the definition that matters: the head's own target, built independently by
    # build_pair_labels.py off the BORIS exports. has_nn = any directed pair labelled 2.
    pl = pd.read_parquet(ROOT / 'dataset' / 'mice' / 'v1' / 'pair_labels.parquet')
    key = pd.MultiIndex.from_frame(raw[['observation_id', 'frame_idx']])
    tgt = {}
    for lv, nm in ((2, 'nn'), (1, 'nt')):
        pos = pl[pl.label == lv].drop_duplicates(['observation_id', 'frame_idx'])
        tgt[nm] = key.isin(pd.MultiIndex.from_frame(pos[['observation_id', 'frame_idx']]))
    d_union = int((tgt['nn'] != union).sum())
    d_mutual = int((tgt['nn'] != (raw.Y_nn > 0.5).to_numpy()).sum())
    d_nt = int((tgt['nt'] != (raw.Y_nt > 0.5).to_numpy()).sum())
    # nt is the control: whatever residual disagreement the two pipelines have on frame
    # boundaries shows up there too, so nn is only allowed to be as far off as nt is.
    check("the head's nn target IS the union, to within nt's own boundary residual",
          d_union <= max(d_nt, 1) and d_mutual > 1000,
          f'{d_union} frames disagree with the union, {d_mutual:,} with mutual-only; '
          f'nt disagrees with Y_nt on {d_nt} (the shared residual)')


def test_sampler_composition(derm_table) -> None:
    """Does DERM compose correctly with the 1:1 negative subsampling? Yes -- and here is why.

    The pipeline changes the class balance TWICE: the sampler keeps every any-label positive and
    thins the all-negative frames to 1:1, and DERM then reweights the loss. The order matters.

    DERM computes p_e on `order` -- the exact index set that epoch's loss averages over -- and
    rebuilds it every epoch because the negatives are resampled. So it equalises the distribution
    the gradient actually sees, whatever the sampler did. That is the correct composition, and it
    is why the invariants above hold on the real sampled order and not only on synthetic data.

    But the sampler DOES decide how much correction is needed, and it is not label-aware. A frame
    is kept if it is positive for EITHER behaviour, so one label's NEGATIVE class is a mixture:
    all-negative frames (thinned) plus the other label's positives (kept whole). That mixture's
    proportion varies by phase, so the sampled per-label phase gradient is not the population one.

    Measured on the training pools, log odds ratio O against H, on the directed-pair union:

        nose-to-tail   population +0.314  ->  sampled +0.277    88% survives
        nose-to-nose   population +0.121  ->  sampled +0.062    51% survives

    The attenuation is real and is the point. What is NOT on record any more is the tidy version
    of it: these numbers used to read +0.11 -> -0.01 for nose-to-nose, i.e. "collapsed", and that
    was measured on the mutual-only nn column, a label no head predicts. On the union the nn
    gradient is halved, not erased, so the neat correspondence this docstring used to claim --
    a_O - a_H is large on nt where a gradient survives and ~0 on nn where it does not -- no longer
    follows from the sampling alone, even though the measured biases (ERM, 4 pools: +0.167 on nt,
    -0.010 on nn; results/vision/mice/frame/_figures/derm.json, estimand_bias) are unchanged,
    since they are computed from val_probs.npz and never touched this column. Treat the mechanism
    as established and the size of its effect on nn as unexplained. What it means practically is
    unchanged: the STRENGTH of DERM's correction per label is a side effect of the sampler rather
    than a choice, so it should be logged rather than left implicit.
    """
    print('\n8. composition with the 1:1 negative subsampling')
    cfg_p = FRAME / 'res448_k2_frozen_d4photo_dermPhase' / 'config.json'
    if not cfg_p.exists():
        print('     [skip] no dermPhase config on disk')
        return
    val = set(json.load(open(cfg_p))['val_pools'])
    exp = pd.read_csv(ROOT / 'data' / 'mice' / 'v1' / 'experiment.csv')
    ann = read_truth().merge(exp[['observation_id', 'pool', 'phase']], on='observation_id')
    tr = ann[~ann.pool.isin(val)]
    Y = {l: (tr['Y_' + l] > 0.5).to_numpy() for l in LABELS}
    anyp = Y['nt'] | Y['nn']
    ph = tr.phase.to_numpy()
    rng = np.random.default_rng(0)
    neg, pos = np.flatnonzero(~anyp), np.flatnonzero(anyp)
    order = np.concatenate([pos, rng.choice(neg, size=min(len(neg), len(pos)), replace=False)])
    lo = lambda p: float(np.log(p / (1 - p)))

    names, ev = np.unique(ph[order], return_inverse=True)
    Ysamp = np.stack([Y[l][order] for l in LABELS], axis=1).astype(np.float32)
    tab, m_raw = derm_table(Ysamp, ev.astype(np.int64), np.arange(len(order)), len(names))
    w = tab[ev[:, None], np.arange(len(LABELS))[None, :], (Ysamp > 0.5).astype(np.int64)]
    worst = 0.0
    for li in range(len(LABELS)):
        for i in range(len(names)):
            m = ev == i
            mp = w[m & (Ysamp[:, li] > 0.5), li].sum()
            mn = w[m & (Ysamp[:, li] <= 0.5), li].sum()
            worst = max(worst, abs(mp / mn - 1.0))
    check('invariants hold on the REAL 1:1-sampled order, per label', worst < 1e-3,
          f'worst pos/neg mass deviation {worst:.2e}; derm_w_raw {m_raw:.2f} vs 1.09 logged')

    print('     log odds ratio, O against H -- what DERM is actually removing:')
    for l in LABELS:
        P = {k: Y[l][ph == k].mean() for k in 'HOP'}
        S = {k: Y[l][order[ph[order] == k]].mean() for k in 'HOP'}
        print(f'       {l}: population {lo(P["O"]) - lo(P["H"]):+.3f}  ->  '
              f'sampled {lo(S["O"]) - lo(S["H"]):+.3f}')
    print('     the sampler is label-AGNOSTIC (keeps a frame positive for EITHER behaviour) while')
    print('     DERM is per label, so each label\'s negatives are contaminated by the other\'s')
    print('     positives at a phase-varying rate:')
    for l, o in (('nt', 'nn'), ('nn', 'nt')):
        row = []
        for k in 'HOP':
            m = order[ph[order] == k]
            row.append(f'{k} {100 * Y[o][m[~Y[l][m]]].mean():5.2f}%')
        print(f'       {l} negatives that are {o} positives: ' + '  '.join(row))
    print('     Not an error: the model trains on the sampled distribution, so an attenuated')
    print('     gradient is an attenuated prior. See the docstring -- on the union truth nn')
    print('     keeps about half its gradient rather than losing it, so this no longer on its')
    print('     own explains why the measured a_O-a_H is ~0 on nn and large on nt.')


def test_env_mapping() -> None:
    """The sample -> phase map, and the assumption that makes it exact."""
    print('\n7. environment mapping')
    a = pd.read_csv(ROOT / 'dataset' / 'mice' / 'v1' / 'annotations.csv',
                    usecols=['observation_id', 'frame_idx'], low_memory=False).reset_index()
    bad = 0
    for _, g in a.groupby('observation_id', sort=False):
        i = g['index'].to_numpy()
        if not (np.all(np.diff(i) == 1) and np.all(np.diff(g.frame_idx.to_numpy()) == 1)):
            bad += 1
    check('every observation is one contiguous ascending block in annotations.csv', bad == 0,
          f'{bad} of {a.observation_id.nunique()} violate it '
          f'-- searchsorted over start rows needs this')

    # the protocol is H 30 min, O 15, P 15, so anchors must land 50/25/25 if the map is right.
    # Taken from the run's own log rather than recomputed, so this checks the SHIPPED mapping.
    logged = {'H': 38139, 'O': 19971, 'P': 19462}
    tot = sum(logged.values())
    share = {k: v / tot for k, v in logged.items()}
    want = {'H': 0.50, 'O': 0.25, 'P': 0.25}
    dev = max(abs(share[k] - want[k]) for k in want)
    check('logged anchor shares match the 30/15/15-minute protocol', dev < 0.01,
          ', '.join(f'{k} {100 * share[k]:.1f}%' for k in 'HOP') + f'  max dev {dev:.4f}')


def main() -> None:
    print('DERM implementation audit')
    mod = load_module()
    derm_table = mod.derm_table
    test_formula(derm_table)
    test_floor(derm_table, mod.derm_mass)
    test_sampler_composition(derm_table)
    test_env_mapping()
    test_population_var(mod)
    test_truth_definition()
    print('\n' + ('FAILED: ' + '; '.join(FAIL) if FAIL else 'all checks passed'))
    sys.exit(1 if FAIL else 0)


if __name__ == '__main__':
    main()
