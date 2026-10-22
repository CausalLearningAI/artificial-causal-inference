"""
ECI v2 release check, PART 1 ADDITIONS (A) distinct best neurons and (B) beat or roughly match v1 on every label.

==================================================================================================================
PRE-REGISTRATION (user decision 2026-10-22 ~21:00, written BEFORE any step-2 score: no step-2 SAE had finished and no
score / neurons / ab file of step 2 existed when this file was committed). Adds two release criteria to the amended
step-2 rule (scripts/eci/v2_mask2/amendment_2026-10-22_adopt_N2.md). Everything else is unchanged.

FRAMES AND SCORER. The step-2 SAEs exist only on the step-2 token stores, so A and B are scored there:
    mice  the diagnosis set (43,200 frames, every 20th 5 fps frame of the 144 annotated videos), restricted to the
          atlas window (habituation videos: frame_idx >= 4500, i.e. their last 15 min, as scripts/eci/atlas_align.py);
    ants  the step-2 scored frames of v2 (13,200) and v3 (32,100), every 10th 5 fps frame.
    R~ = the normalised R-precision with ties pro rata (src/eci/v2/score.py = atlas_align.rprec, same definition).
    PRIMARY split = the atlas split: units (mice: observation_id without its last two fields, as atlas_align; ants:
    videos) sorted, alternating halves; cf_top1 = mean over the two directions of the held-out R~ of the best
    neuron of the selection half (atlas_align 'cf_top1_rprec'). The mean over the 10 random halves rnd0-9 (the
    step-2 scorer) is reported too.
    DEVIATIONS FROM THE ATLAS PAGE, stated before scoring:
      (1) Frames: the page uses all labelled frames on the 1 fps grid; here the step-2 frame subsets (the page's
          frames would need new N2 token encodes of every labelled 1 fps frame and its neighbours for 20-33 seeds:
          an estimated > 25 GPU-h for mice alone, beyond the agreed budget; needs the user's OK).
      (2) Smoothing: the page's v2 'continuous' input is the 0.6 s smoothed code; these frames are 2-4 s apart,
          so the v2 codes here are UNSMOOTHED frame max. v1 is unsmoothed codes_max on the page too. Earlier
          measurements found that smoothing raised v2 R~ (mice nose_nose +0.014, CRIM13 mean +0.028). So an unsmoothed
          pass is conservative for v2. An unsmoothed fail is reported as a fail, with this caveat.
      (3) Ants v4 has no step-2 token store: its grooming / on-lid cells are NOT MEASURED here.
    BRIDGE (reported, never decides): the frozen deployed v2 SAE (v2_mice_c0_w2048_s7, v2_ants_c0_w2048_s6) on the
    M0 tokens of the same frames, so the page's at-risk cell (mice nose_tail, page v2 0.078 vs v1 0.108 at the
    full dictionaries) can be compared with this frame set.
v1 = the deployed v1 SAEs: mice matryoshka_btk_1024_k16_fg448al_s0 (codes_max rows = the frames' annotations.csv rows),
    ants matryoshka_btk_1024_k16_antsfg_s0 (dataset/ants/eci30 codes_max, rows matched on (observation_id, frame_idx)).
PREFIXES: full (v1 1024, v2 2048), 256, 128 (reported) = the first p neurons of each Matryoshka dictionary.

(A) DISTINCT BEST NEURONS (mice). Per arm, seed and prefix p: best neuron of a label = the modal selected neuron over
    rnd0-9 x 2 directions. 'Distinct' at p iff the best nose_nose (nn | np) neuron differs from the best nose_tail neuron AND
    on the atlas split neither direction selects the same neuron for both labels. nn_mutual and np_directional best
    neurons are reported. Arm PASSES (A) iff its deployment seed (highest held-out FVE among 70-72) is distinct at
    p = 2048 and p = 256 AND >= 80% of its 20 seeds are distinct at both 2048 and 256. p = 128 is reported only.
(B) BEAT OR ROUGHLY MATCH v1. Cells: mice nose_nose, mice nose_tail, ants v2 groom_any, ants v3 groom_any, ants v3
    on_lid_any (ants v4 not measured, see (3)); prefixes full and 256. Delta_s = R~cf_top1(arm, seed s) -
    R~cf_top1(v1) on the atlas split, s = the 20 mice / 33 ants seeds. A cell PASSES iff the lower bound of the 95% t-CI of
    mean Delta is >= -0.01. Arm PASSES (B) iff every measured cell passes at both prefixes. The same for the
    rnd0-9 means is reported.
Release (amended): N2 is released only if (ii), (ii-user), (iii), (iv) of the amendment AND (A) AND (B) pass for N2.
    A and B are reported for M0 as well.
==================================================================================================================

Hooks / steps:
    ab_scores(...)  called by mask2_test.py cmd_score for every arm and seed -> OUT/<dom>/ab/<arm>_s<seed>.json
    refs            (GPU job: SCRIPT=release_ab.py EXTRA=refs) v1 and frozen-v2 bridge -> OUT/ab_refs.json
    decide          (CPU) -> OUT/ab_decision.json, OUT/ab_table.md
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

PREF = (('full', None), ('256', 256), ('128', 128))
MCELLS = ('nose_nose', 'nose_tail')
ACELLS = (('v2', 'groom_any'), ('v3', 'groom_any'), ('v3', 'on_lid_any'))
FROZEN = {'mice': '/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2/mice/sae/v2_mice_c0_w2048_s7/sae_theta.pt',
          'ants': '/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2/ants/sae/v2_ants_c0_w2048_s6/sae_theta.pt'}
V1 = {'mice': 'dataset/mice/v1/eci/codes/matryoshka_btk_1024_k16_fg448al_s0/codes_max.npy',
      'ants': 'dataset/ants/eci30/codes/matryoshka_btk_1024_k16_antsfg_s0/codes_max.npy'}


def mice_window(St, half):
    """-> frame indices of the atlas window, per-frame halves on them (rnd0-9 + 'atlas')."""
    obs, k = St.F.obs.values, St.F.k.values.astype(int)
    keep = np.flatnonzero(~pd.Series(obs).str.endswith('_H').values | (k >= 4500))
    unit = pd.Series(obs).str.rsplit('_', n=2).str[0].values
    units = sorted(set(unit))
    at = pd.Series(unit).map({u: i % 2 for i, u in enumerate(units)}).values
    h = {s: np.asarray(v)[keep] for s, v in half.items() if s.startswith('rnd')}
    h['atlas'] = at[keep]
    return keep, h


def summarise(o, labels, atlas_key):
    out = {}
    for lb in labels:
        m = o['mean'][lb]
        sp = o['splits'][atlas_key][lb]
        out[lb] = {'R_rnd': m['R'], 'R_atlas': sp['R'], 'modal_neuron': m['modal_neuron'],
                   'modal_share': m['modal_share'], 'atlas_neurons': sp['neurons']}
    return out


def ab_mice(X, Yg, half, St, labels):
    import torch
    from src.eci.v2.score import crossfit
    keep, h = mice_window(St, half)
    ix = torch.from_numpy(keep).to(X.device)
    Xw, Yw = X.index_select(0, ix), Yg.index_select(0, ix)
    out = {'n_frames': int(len(keep))}
    for nm, p in PREF:
        out[nm] = summarise(crossfit(Xw[:, :p].contiguous() if p else Xw, Yw, h, labels), labels, 'atlas')
    return out


def ab_ants(X, Yg, hh, labs):
    from src.eci.v2.score import crossfit
    out = {'n_frames': int(X.shape[0])}
    for nm, p in PREF:
        out[nm] = summarise(crossfit(X[:, :p].contiguous() if p else X, Yg, hh, labs), labs, 'r4')
    return out


def distinct(r):
    """one prefix block of a mice ab record -> bool."""
    a, b = r['nose_nose'], r['nose_tail']
    return bool(a['modal_neuron'] != b['modal_neuron'] and
                all(x != y for x, y in zip(a['atlas_neurons'], b['atlas_neurons'])))


# ---------------------------------------------------------------------------------------------- refs (v1, bridge)
def cmd_refs(args):
    import torch
    import mask2_test as T
    from src.eci.sae import load_sae
    MT, FZ, REPO = T.MT, T.FZ, T.REPO
    dev = torch.device('cuda')
    out = {}

    def rows_codes(path, rows):
        cm = np.load(REPO / path, mmap_mode='r')
        oi = np.argsort(rows)
        X = np.empty((len(rows), cm.shape[1]), np.float16)
        X[oi] = cm[rows[oi]]
        return torch.from_numpy(X).to(dev)
    # mice
    St = T.arm_store(T.Union('mice', 'diag'), 'M0', dev, on_gpu=True)
    Y, sets, half, _, _ = MT.mice_sets(St)
    Yg = torch.from_numpy(Y).to(dev)
    X1 = rows_codes(V1['mice'], St.F.row.values.astype(np.int64))
    out['mice'] = {'v1': ab_mice(X1, Yg, half['all'], St, FZ.MLAB)}
    sae, norm, _ = load_sae(FROZEN['mice'], dev)
    out['mice']['frozen_v2'] = ab_mice(St.codes(sae, norm, 0, 'all'), Yg, half['all'], St, FZ.MLAB)
    T.log('refs mice: ' + json.dumps({k: {p: {lb: round(v[p][lb]['R_atlas'], 4) for lb in MCELLS} for p in ('full', '256')}
                                      for k, v in out['mice'].items()}))
    del St
    torch.cuda.empty_cache()
    # ants
    a1 = pd.read_csv(REPO / 'dataset/ants/eci30/annotations.csv', usecols=['observation_id', 'frame_idx'],
                     dtype={'observation_id': str})
    idx = pd.Series(np.arange(len(a1)), index=pd.MultiIndex.from_arrays([a1.observation_id, a1.frame_idx]))
    sae, norm, _ = load_sae(FROZEN['ants'], dev)
    out['ants'] = {}
    for ser in ('v2', 'v3'):
        St = T.arm_store(T.Union('ants', f'scored_{ser}'), 'M0', dev, on_gpu=True)
        Y, hh, labs = MT.ants_set(St, ser)
        Yg = torch.from_numpy(Y).to(dev)
        keys = pd.MultiIndex.from_arrays([St.F.obs.values.astype(str), St.F.k.values.astype(int)])
        r1 = idx.reindex(keys).values
        miss = int(np.isnan(r1).sum())
        assert miss == 0, f'ants {ser}: {miss} frames without v1 codes'
        out['ants'][ser] = {'v1': ab_ants(rows_codes(V1['ants'], r1.astype(np.int64)), Yg, hh, labs),
                            'frozen_v2': ab_ants(St.codes(sae, norm, 0, 'all'), Yg, hh, labs)}
        T.log(f'refs ants {ser}: ' + json.dumps({k: {p: {lb: round(v[p][lb]['R_atlas'], 4) for lb in labs}
                                                      for p in ('full', '256')} for k, v in out['ants'][ser].items()}))
        del St
        torch.cuda.empty_cache()
    (T.OUT / 'ab_refs.json').write_text(json.dumps(out, indent=1, default=float))


# ---------------------------------------------------------------------------------------------- decide
def tci(d):
    from scipy import stats
    d = np.asarray(d, float)
    m, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
    hw = stats.t.ppf(0.975, len(d) - 1) * se
    return float(m), float(m - hw), float(m + hw)


def cmd_decide(args):
    import mask2_test as T
    refs = json.loads((T.OUT / 'ab_refs.json').read_text())
    res = {'rule': 'scripts/eci/v2_mask2/release_ab.py docstring (pre-registered 2026-10-22)', 'A': {}, 'B': {}}
    L = ['| arm | criterion | cell | prefix | v1 | frozen v2 (bridge) | arm mean ± sd | arm - v1 (95% t-CI) | pass |',
         '|---|---|---|---|---|---|---|---|---|']
    for arm in ('M0', 'N2'):
        R = {d: {s: json.loads((T.OUT / d / 'ab' / f'{arm}_s{s}.json').read_text()) for s in T.SEEDS[d]
                 if (T.OUT / d / 'ab' / f'{arm}_s{s}.json').exists()} for d in ('mice', 'ants')}
        fv = {s: T.fve_of('mice', arm, s) for s in T.DEPLOY_FROM}
        dep = max(fv, key=lambda s: fv[s] if np.isfinite(fv[s]) else -1)
        A = {}
        for p in ('full', '256', '128'):
            ds = {s: distinct(r[p]) for s, r in R['mice'].items()}
            best = {s: {lb: r[p][lb]['modal_neuron'] for lb in r[p] if lb in ('nose_nose', 'nose_tail', 'nn_mutual',
                                                                               'np_directional')}
                    for s, r in R['mice'].items()}
            A[p] = {'share_seeds_distinct': float(np.mean(list(ds.values()))) if ds else None,
                    'deploy_seed': dep, 'deploy_distinct': ds.get(dep), 'deploy_best': best.get(dep),
                    'per_seed_best': best}
        A['PASS'] = bool(all(A[p]['deploy_distinct'] and A[p]['share_seeds_distinct'] >= 0.8 for p in ('full', '256')))
        A['v1_distinct'] = {p: distinct(refs['mice']['v1'][p]) for p in ('full', '256', '128')}
        A['frozen_v2_distinct'] = {p: distinct(refs['mice']['frozen_v2'][p]) for p in ('full', '256', '128')}
        res['A'][arm] = A
        for p in ('full', '256', '128'):
            L.append(f'| {arm} | A distinct | nn vs nt best neuron | {p} | {A["v1_distinct"][p]} | '
                     f'{A["frozen_v2_distinct"][p]} | {A[p]["share_seeds_distinct"]} of seeds | deploy s{dep}: '
                     f'{A[p]["deploy_best"]} | {A["PASS"] if p != "128" else "(reported)"} |')
        B, ok = {}, True
        cells = [('mice', None, lb) for lb in MCELLS + ('nn_mutual', 'np_directional')] + \
                [('ants', ser, lb) for ser, lb in ACELLS]
        for d, ser, lb in cells:
            for p in ('full', '256'):
                for key in ('R_atlas', 'R_rnd'):
                    ref = refs[d] if d == 'mice' else refs[d][ser]
                    v1 = ref['v1'][p][lb][key]
                    fz = ref['frozen_v2'][p][lb][key]
                    xs = [(r[p] if d == 'mice' else r[ser][p])[lb][key] for r in R[d].values()]
                    if len(xs) < 2:
                        continue
                    m, lo, hi = tci(np.array(xs) - v1)
                    decides = key == 'R_atlas' and lb in MCELLS + ('groom_any', 'on_lid_any')
                    ps = bool(lo >= -0.01)
                    if decides:
                        ok &= ps
                    nm = f'{d}{" " + ser if ser else ""} {lb}'
                    B[f'{nm} | {p} | {key}'] = {'v1': v1, 'frozen_v2': fz, 'arm_mean': float(np.mean(xs)),
                                                'arm_sd': float(np.std(xs, ddof=1)), 'n': len(xs),
                                                'minus_v1': [m, lo, hi], 'pass': ps, 'decides': decides}
                    L.append(f'| {arm} | B {key} | {nm} | {p} | {v1:.4f} | {fz:.4f} | {np.mean(xs):.4f} ± '
                             f'{np.std(xs, ddof=1):.4f} (n={len(xs)}) | {m:+.4f} [{lo:+.4f}, {hi:+.4f}] | '
                             f'{ps if decides else "(reported: " + str(ps) + ")"} |')
        B['PASS_measured_cells'] = bool(ok)
        B['not_measured'] = 'ants v4 groom_any / on_lid_any (no step-2 token store)'
        res['B'][arm] = B
    (T.OUT / 'ab_decision.json').write_text(json.dumps(res, indent=1, default=float))
    L += ['', 'A PASS: ' + json.dumps({a: res['A'][a]['PASS'] for a in res['A']}),
          'B PASS (measured cells; ants v4 not measured): ' + json.dumps({a: res['B'][a]['PASS_measured_cells']
                                                                         for a in res['B']})]
    (T.OUT / 'ab_table.md').write_text('\n'.join(L))
    print('\n'.join(L))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=('refs', 'decide'))
    a = p.parse_args()
    {'refs': cmd_refs, 'decide': cmd_decide}[a.step](a)


if __name__ == '__main__':
    main()
