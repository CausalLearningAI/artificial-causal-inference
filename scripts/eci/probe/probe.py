"""
ECI linear-probe ceiling: supervised probes on pooled frame features (scripts/eci/probe/feats.py) and the best single
SAE neuron on the same frames. Design and frame sets: results/vision/eci_probe/LOG.md.

Probe: standardised features (fit on the training frames), L2 logistic regression (lbfgs), C in CGRID chosen by the
  mean inner-CV AUROC over 3 grouped folds of the TRAINING units only (mice pools / ants videos), then refit on all
  training frames. No class weights.
Best single neuron (code reps, 'max' pooling = the ECI frame-max readout, and 'mean'): the neuron with the highest
  AUROC on the training frames, its raw pooled code as the score on the test frames.
Training frames: mice = the 1 fps grid (stride 5, per-video offset: the frames the v1 DINOv2 store holds) of the training pools, for EVERY
  representation (so all columns see the same training frames); ants = the v3 + v4 eval frames (1 fps) of the training half.
Test predictions: mice = the 1 fps grid of the test pools for every rep, and every 5 fps test frame for the v2-store
  reps (d3, V, V_s*); ants = the v3 + v4 eval frames of the test half.
Splits: mice dir 'sg2non' = train on the 11 SG pools, test on the 12 non-SG pools; 'non2sg' = the reverse; the
  unknown-annotator pool (rd41_3) is in neither. Ants (user decision 2026-11-04): v3 + v4 videos only (147; v2 and
  v4_infested excluded), half A / half B by video stratified by series x treatment (OUT/ants_split.csv, rng 20261104);
  'A2B' = train A, test B (primary), 'B2A' = swapped.
Permutation control (reps d2:mean, d3:mean, V_s1001:mm): the same probe (the chosen C) fitted on labels permuted across
  training videos of equal length (whole label sequences swapped, rng 0).

Step: run --dom D --rep R --beh B  ->  BEE/preds/<dom>/<rep>__<beh>.npz + OUT/fits/<dom>/<rep>__<beh>.json
"""
import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

BEE = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/probe')
REPO = Path(__file__).resolve().parents[3]
OUT = REPO / 'results/vision/eci_probe'
CGRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
V_SEEDS = (1001, 1002, 1003)
BEH = {'mice': ('nose_nose', 'nn', 'np', 'nt'), 'ants': ('groom_any', 'Y2F', 'B2F', 'single', 'double')}
PERM_REPS = ('d2:mean', 'd3:mean', 'V_s1001:mm')


def log(s):
    print(time.strftime('%H:%M:%S'), s, flush=True)


def reps(dom):
    r = ['d2:mean', 'd2:mm', 'd3:mean', 'd3:mm', 'V:mean', 'V:mm', 'v1c:mean', 'v1c:mm', 'v1c:max']
    return r + [f'V_s{s}:{p}' for s in V_SEEDS for p in ('mean', 'mm', 'max')]


def is_v1(rep):
    return rep.split(':')[0] in ('d2', 'v1c')


def frames(dom):
    return pd.read_parquet(BEE / dom / 'frames.parquet')


def load_parts(dom, fam, F):
    """-> {name: (len(F), d) float16 memmap-free array} and a bool 'has' (frames with features)."""
    if fam == 'v2':
        od, tags = BEE / dom / 'parts', None
        i, R = 0, []
        while (od / f't{i}_rows.parquet').exists():
            R.append(pd.read_parquet(od / f't{i}_rows.parquet'))
            i += 1
        R = pd.concat(R, ignore_index=True)
        assert len(R) == len(F) and (R.obs.values == F.obs.values).all() and (R.k.values == F.k.values).all(), 'v2 parts != table'
        return od, [f't{j}' for j in range(i)], np.arange(len(F)), R
    od = BEE / dom / 'parts1'
    if dom == 'mice':
        tags = [f's{j}' for j in range(16)]
        R = pd.concat([pd.read_parquet(od / f'{t}_rows.parquet') for t in tags], ignore_index=True)
        idx = pd.Series(np.arange(len(F)), index=F.row.values).reindex(R.row.values).values
    else:
        tags, j = [], 0
        while (od / f't{j}_rows.parquet').exists():
            tags.append(f't{j}')
            j += 1
        R = pd.concat([pd.read_parquet(od / f'{t}_rows.parquet') for t in tags], ignore_index=True)
        idx = R.tab.values
    assert not np.isnan(idx.astype(float)).any()
    idx = idx.astype(np.int64)
    assert (F.obs.values[idx] == R.obs.values).all() and (F.k.values[idx] == R.k.values).all()
    return od, tags, idx, R


def load_rep(dom, rep, F):
    """-> X (len(F), d) float32 (NaN rows where absent), has (bool)."""
    base, pl = rep.split(':')
    od, tags, idx, R = load_parts(dom, 'v1' if is_v1(rep) else 'v2', F)
    names = {'mean': ['mean'], 'max': ['max'], 'mm': ['mean', 'max']}[pl]
    blocks = []
    for nm in names:
        blocks.append(np.concatenate([np.load(od / f'{t}_{base}_{nm}.npy') for t in tags]))
    Xp = np.concatenate(blocks, 1) if len(blocks) > 1 else blocks[0]
    X = np.zeros((len(F), Xp.shape[1]), np.float16)
    X[idx] = Xp
    has = np.zeros(len(F), bool)
    has[idx] = True
    return X, has


def auroc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float('nan')


def col_auroc(X, y):
    """AUROC of every column of X (ties = 0.5) via average ranks."""
    from scipy.stats import rankdata
    r = rankdata(X, axis=0)
    npos, nneg = y.sum(), (~y).sum()
    return (r[y].sum(0) - npos * (npos + 1) / 2) / (npos * nneg)


def fit_lr(X, y, C, w0=None):
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(C=C, max_iter=500, tol=1e-4)
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter('always')
        m.fit(X, y)
    return m, any('converge' in str(w.message).lower() for w in wl)


def probe(Xtr, ytr, gtr, rng):
    """inner grouped 3-fold CV over CGRID on (Xtr, ytr, groups gtr) -> chosen C, inner AUROC per C, scaler, model."""
    from sklearn.preprocessing import StandardScaler
    ug = np.unique(gtr)
    perm = rng.permutation(ug)
    fold = pd.Series(np.arange(len(perm)) % 3, index=perm).reindex(gtr).values
    inner = {}
    for C in CGRID:
        a = []
        for f in range(3):
            tr, te = fold != f, fold == f
            if ytr[te].sum() == 0 or ytr[tr].sum() == 0:
                continue
            sc = StandardScaler().fit(Xtr[tr])
            m, _ = fit_lr(sc.transform(Xtr[tr]), ytr[tr], C)
            a.append(auroc(ytr[te], m.decision_function(sc.transform(Xtr[te]))))
        inner[C] = float(np.mean(a))
    Cb = max(inner, key=inner.get)
    sc = StandardScaler().fit(Xtr)
    m, nc = fit_lr(sc.transform(Xtr), ytr, Cb)
    return Cb, inner, sc, m, nc


def predict(sc, m, X, rows):
    out = np.full(len(rows), np.nan, np.float32)
    for a in range(0, len(rows), 50000):
        r = rows[a:a + 50000]
        out[a:a + len(r)] = m.predict_proba(sc.transform(X[r].astype(np.float32)))[:, 1]
    return out


def perm_labels(y, vid, rng):
    """swap whole per-video label sequences among training videos of equal length."""
    yp = y.copy()
    s = pd.Series(np.arange(len(vid))).groupby(vid)
    groups = {}
    for v, ix in s:
        groups.setdefault(len(ix), []).append((v, ix.values))
    for L, lst in groups.items():
        p = rng.permutation(len(lst))
        for (v, ix), j in zip(lst, p):
            yp[ix] = y[lst[j][1]]
    return yp


def ants_split():
    """v3 + v4 eval videos (147), half A / half B, stratified by series x treatment T, rng 20261104 -> OUT/ants_split.csv."""
    p = OUT / 'ants_split.csv'
    if p.exists():
        return pd.read_csv(p, dtype={'obs': str})
    F = frames('ants')
    V = F[F.experiment.isin(['v3', 'v4'])][['obs', 'experiment']].drop_duplicates()
    T = []
    for f, ex in (('dataset/ants/eci30/annotations.csv', 'v3'), ('dataset/ants/eci_c1v4_noday/annotations.csv', 'v4')):
        a = pd.read_csv(REPO / f, usecols=['observation_id', 'experiment', 'T'], dtype={'observation_id': str})
        T.append(a[a.experiment == ex].drop_duplicates(['observation_id']).set_index('observation_id')['T'])
    T = pd.concat(T)
    V['T'] = V.obs.map(T).values
    assert V['T'].notna().all() and len(V) == 147
    rng = np.random.default_rng(20261104)
    V['half'] = ''
    for (ex, t), g in V.sort_values('obs').groupby(['experiment', 'T']):
        p_ = rng.permutation(g.index.values)
        V.loc[p_[:len(p_) // 2], 'half'] = 'A'
        V.loc[p_[len(p_) // 2:], 'half'] = 'B'
    V = V.sort_values(['experiment', 'T', 'half', 'obs'])
    V.to_csv(p, index=False)
    return pd.read_csv(p, dtype={'obs': str})


def cmd_run(a):
    t0 = time.time()
    dom, rep, beh = a.dom, a.rep, a.beh
    F = frames(dom)
    X, has = load_rep(dom, rep, F)
    y = F[f'y_{beh}'].values.astype(bool)
    log(f'{dom} {rep} {beh}: X {X.shape}, frames with features {has.sum():,} ({time.time() - t0:.0f}s)')
    rng = np.random.default_rng(0)
    res = {'dom': dom, 'rep': rep, 'beh': beh, 'dim': int(X.shape[1]), 'frames_with_features': int(has.sum()), 'splits': {}}
    P = {k: np.full(len(F), np.nan, np.float32) for k in ('probe', 'neuron', 'perm')}
    code_rep = rep.split(':')[0] in ('v1c',) + tuple(f'V_s{s}' for s in V_SEEDS) and rep.split(':')[1] in ('max', 'mean')
    if dom == 'mice':
        grid = F.grid1.values
        unit = F.unit.values
        splits = {'sg2non': (F.sg.values, F.nonsg.values), 'non2sg': (F.nonsg.values, F.sg.values)}
    else:  # ants (user decision 2026-11-04): v3 + v4 only, half / half by video stratified by series x treatment
        unit = F.obs.values
        S = ants_split()
        h = F.obs.map(S.set_index('obs').half).fillna('').values
        grid = np.isin(h, ['A', 'B'])
        splits = {'A2B': (h == 'A', h == 'B'), 'B2A': (h == 'B', h == 'A')}
    for sp, (trm, tem) in splits.items():
        tr = np.flatnonzero(trm & grid & has)
        te_all = np.flatnonzero(tem & has)
        te = te_all if (dom == 'ants' or not is_v1(rep)) else np.flatnonzero(tem & grid & has)
        assert len(tr) and len(te)
        Xtr = X[tr].astype(np.float32)
        Cb, inner, sc, m, nc = probe(Xtr, y[tr], unit[tr], rng)
        P['probe'][te] = predict(sc, m, X, te)
        d = {'train_frames': int(len(tr)), 'train_pos': int(y[tr].sum()), 'test_frames': int(len(te)),
             'test_pos': int(y[te].sum()), 'C': Cb, 'inner_auroc': inner, 'not_converged': bool(nc),
             'test_auroc_probe': auroc(y[te], P['probe'][te])}
        if rep in PERM_REPS:
            yp = perm_labels(y[tr], F.obs.values[tr], np.random.default_rng(1))
            m2, _ = fit_lr(sc.transform(Xtr), yp, Cb)
            P['perm'][te] = predict(sc, m2, X, te)
            d['test_auroc_perm'] = auroc(y[te], P['perm'][te])
            d['perm_label_agreement'] = float((yp == y[tr]).mean())
        if code_rep:
            ca = col_auroc(Xtr, y[tr])
            j = int(np.nanargmax(ca))
            P['neuron'][te] = X[te, j].astype(np.float32)
            d.update({'neuron': j, 'neuron_train_auroc': float(ca[j]), 'test_auroc_neuron': auroc(y[te], P['neuron'][te])})
        res['splits'][sp] = d
        log(f'{sp}: {json.dumps({k: v for k, v in d.items() if k != "inner_auroc"})} ({time.time() - t0:.0f}s)')
    od = BEE / 'preds' / dom
    od.mkdir(parents=True, exist_ok=True)
    np.savez(od / f'{rep.replace(":", "__")}__{beh}.npz', **P)
    of = OUT / 'fits' / dom
    of.mkdir(parents=True, exist_ok=True)
    res['seconds'] = time.time() - t0
    (of / f'{rep.replace(":", "__")}__{beh}.json').write_text(json.dumps(res, indent=1))


def cmd_list(a):
    """print the task list (dom rep beh), one per line, for the array wrapper."""
    for dom in ('mice', 'ants'):
        for rep in reps(dom):
            for beh in BEH[dom]:
                print(dom, rep, beh)


def cmd_task(a):
    """array entry: run line SLURM_ARRAY_TASK_ID (+ offset) of the task list."""
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cmd_list(a)
    lines = [l.split() for l in buf.getvalue().strip().split('\n')]
    if a.only:
        lines = [l for l in lines if l[0] == a.only]
    dom, rep, beh = lines[a.task]
    a.dom, a.rep, a.beh = dom, rep, beh
    cmd_run(a)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('step')
    ap.add_argument('--dom')
    ap.add_argument('--rep')
    ap.add_argument('--beh')
    ap.add_argument('--only', default='')
    ap.add_argument('--task', type=int, default=0)
    a = ap.parse_args()
    globals()[f'cmd_{a.step}'](a)
