#!/usr/bin/env python3
"""PPI on v1 and prediction-powered transport to v2, for the phase-transition contrast.

v1 (PPI++). The estimand is a mean of WITHIN-POOL phase differences, so the unit is the pool and
the paired difference is the observation. With n labelled pools carrying both a true difference
D_Y and an out-of-fold predicted difference D_f, and N unlabelled pools carrying D_f only:

    theta_ppi = mean_unlabelled(lam * D_f) + mean_labelled(D_Y - lam * D_f)

Unbiased for ANY predictor: whatever lam and however wrong f is, the rectifier subtracts exactly
what the first term added. f only moves the VARIANCE. The power-tuned lam is

    lam* = Cov(D_Y, D_f) / (Var(D_f) * (1 + n/N))

which is also why a badly-calibrated model costs nothing here: lam absorbs the scale.

v2 (PPCI). There are no labels anywhere in v2, so there is no rectifier and the estimate is only
as good as the model transfers. We transport the v1 calibration slope
    beta = Cov(D_Y, D_f) / Var(D_f)          (fitted on v1's labelled pools)
and report beta * mean(D_f) on v2. The interval must carry BOTH the v2 sampling error and the
uncertainty in beta, which is done here by a paired bootstrap over v1 pools and v2 pools jointly.
This is an extrapolation, not an estimate with guarantees -- state it as such.

Usage:
    python scripts/mice_behavior/ppi_phase.py --labelled /tmp/xfit_pool_preds.csv \
        --unlabelled-v1 <csv> --v2 <csv>
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def pool_deltas(df, tcol, pcol, odour=None, trans=('H', 'O')):
    """-> (pools, D_Y, D_f), one row per pool, for ONE exposure and ONE transition.

    Stratification is mandatory, not optional. The two exposures move nt in OPPOSITE directions
    and the two transitions are on/off, so pooling any of them together averages a real effect
    against its own negation and returns approximately zero -- which is exactly what happened the
    first time this was run unstratified (classical nn came out -0.07 +/- 0.20 instead of the
    +0.50 the fear cell actually carries).
    """
    x, y = trans
    keys, dy, df_ = [], [], []
    sub = df if odour is None else df[df.odor == odour]
    for pool, g in sub.groupby('pool'):
        m = g.set_index('phase')
        if x not in m.index or y not in m.index:
            continue
        keys.append(pool)
        dy.append((m.loc[y, tcol] - m.loc[x, tcol]) if tcol else np.nan)
        df_.append(m.loc[y, pcol] - m.loc[x, pcol])
    return np.array(keys), np.array(dy, float), np.array(df_, float)


def ppi(dy_lab, df_lab, df_unl):
    n, N = len(dy_lab), len(df_unl)
    v = df_lab.var(ddof=1)
    lam = (np.cov(dy_lab, df_lab)[0, 1] / (v * (1 + n / N))) if v > 0 else 0.0
    theta = (lam * df_unl).mean() + (dy_lab - lam * df_lab).mean()
    var = (dy_lab - lam * df_lab).var(ddof=1) / n + lam ** 2 * df_unl.var(ddof=1) / N
    return theta, np.sqrt(var), lam


def classical(dy_lab):
    return dy_lab.mean(), dy_lab.std(ddof=1) / np.sqrt(len(dy_lab))


def cluster_boot(keys, arrs, fn, reps=4000, seed=0):
    rng = np.random.default_rng(seed)
    pools = np.unique(keys); out = []
    for _ in range(reps):
        pick = rng.choice(pools, len(pools), replace=True)
        sel = np.concatenate([np.flatnonzero(keys == p) for p in pick])
        try:
            v = fn(*[a[sel] for a in arrs])
            if np.isfinite(v): out.append(v)
        except Exception:
            pass
    return np.percentile(out, [2.5, 97.5]) if out else (np.nan, np.nan)


def crossfit_labelled(folds=('xfit_f1', 'xfit_f2', 'xfit_f3')):
    """Per-observation OUT-OF-FOLD outcomes on all 24 annotated pools.

    Each fold held out 8 pools, so concatenating the three val_probs.npz files gives every
    annotated observation exactly once, scored by a model that never saw its pool. That is the
    condition PPI's rectifier needs; the standing 4-pool split cannot supply it.
    """
    import sys as _s
    _s.path.insert(0, str(Path(__file__).parent))
    from event_eval import runs, postprocess
    F = ROOT / 'results' / 'vision' / 'mice' / 'frame'
    exp = pd.read_csv(ROOT / 'data' / 'mice' / 'v1' / 'experiment.csv')[
        ['observation_id', 'pool', 'phase', 'odor']]
    parts = []
    for t in folds:
        d = np.load(F / t / 'val_probs.npz', allow_pickle=True)
        parts.append(pd.DataFrame({
            'obs': d['obs'], 'gi': d['gi'],
            'p_nt': d['probs'][:, 0], 'p_nn': d['probs'][:, 1],
            'y_nt': d['labels'][:, 0], 'y_nn': d['labels'][:, 1]}))
    A = pd.concat(parts).sort_values(['obs', 'gi'])
    rows = []
    for oid, g in A.groupby('obs', sort=False):
        n = len(g); r = {'observation_id': oid}
        for lab in ('nt', 'nn'):
            tm = g['y_' + lab].to_numpy() > 0.5
            pm = postprocess(g['p_' + lab].to_numpy() >= 0.5, 1, 1)
            mins = n / 5 / 60
            r['t_' + lab] = len(runs(tm)) / mins        # true bouts/min
            r['p_' + lab] = len(runs(pm)) / mins        # predicted bouts/min
            r['to_' + lab] = tm.mean() * 100           # true occupancy, pp
            r['po_' + lab] = g['p_' + lab].mean() * 100  # predicted occupancy, pp
        rows.append(r)
    return pd.DataFrame(rows).merge(exp, on='observation_id')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred-run', default='xfit_f2')
    a = ap.parse_args()
    F = ROOT / 'results' / 'vision' / 'mice' / 'frame'
    lab = crossfit_labelled()
    lab.to_csv(F / '_figures' / 'xfit_labelled_pool_preds.csv', index=False)
    u1 = pd.read_csv(F / a.pred_run / 'pred_unannotated_v1.csv')
    u2 = pd.read_csv(F / a.pred_run / 'pred_unannotated_v2.csv')
    print(f'labelled {lab["pool"].nunique()} pools | unlabelled v1 {u1["pool"].nunique()} | '
          f'v2 {u2["pool"].nunique()}\n')
    print(f'{"cell":16}{"classical":>22}{"PPI++":>22}{"CI":>6}{"r":>7}{"v2 transported":>24}')
    res = {}
    for lb in ('nn', 'nt'):
        U1 = u1.assign(**{'po_' + lb: u1['p_' + lb] * 100})
        U2 = u2.assign(**{'po_' + lb: u2['p_' + lb] * 100})
        for od, odn in (('F', 'fear'), ('S', 'social')):
            kl, dy, dfl = pool_deltas(lab, 'to_' + lb, 'po_' + lb, od)
            ku, _, dfu = pool_deltas(U1, None, 'po_' + lb, od)
            kv, _, dfv = pool_deltas(U2, None, 'po_' + lb, od)
            c, cse = classical(dy)
            th, se, lam = ppi(dy, dfl, dfu)
            beta = np.cov(dy, dfl)[0, 1] / dfl.var(ddof=1)
            rng = np.random.default_rng(0); bs = []
            pl, pv = np.unique(kl), np.unique(kv)
            for _ in range(3000):
                ia = np.concatenate([np.flatnonzero(kl == p) for p in rng.choice(pl, len(pl), True)])
                ib = np.concatenate([np.flatnonzero(kv == p) for p in rng.choice(pv, len(pv), True)])
                try:
                    bb = np.cov(dy[ia], dfl[ia])[0, 1] / dfl[ia].var(ddof=1)
                    bs.append(bb * dfv[ib].mean())
                except Exception:
                    pass
            v2 = beta * dfv.mean(); v2lo, v2hi = np.percentile(bs, [2.5, 97.5])
            key = f'{lb}_{odn}'
            res[key] = dict(classical=[c, c - 1.96 * cse, c + 1.96 * cse],
                            ppi=[th, th - 1.96 * se, th + 1.96 * se],
                            v2=[v2, v2lo, v2hi], lam=float(lam), beta=float(beta),
                            n=len(dy), N=len(dfu), Nv2=len(dfv),
                            shrink=float(1 - se / cse), r=float(np.corrcoef(dy, dfl)[0, 1]))
            print(f'{key:16}{c:+7.3f} [{c-1.96*cse:+.2f},{c+1.96*cse:+.2f}]'
                  f'{th:+8.3f} [{th-1.96*se:+.2f},{th+1.96*se:+.2f}]'
                  f'{-100*(1-se/cse):5.0f}%{np.corrcoef(dy,dfl)[0,1]:7.2f}'
                  f'{v2:+9.3f} [{v2lo:+.2f},{v2hi:+.2f}]')
    json.dump(res, open(F / '_figures' / 'ppi_results.json', 'w'), indent=1)
    print(f'\nH->O contrast, occupancy scale (pp). n=24 labelled / N=48 unlabelled v1 / 36 v2 pools.')
