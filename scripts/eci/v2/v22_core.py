"""
ECI v2.2 core: mice frame table with annotators, cached readouts (raw frame code + 3-state HMM P(on)) for any model,
all-neuron AUROC kernels, neuron-selection rules and the held-out metrics with a unit x seed bootstrap.

Readouts (identical to v21_ladder.readout / v21_extended): v2-type models = per-neuron 3-state HMM P(on) fitted label-free
on the 144 eval windows (one sequence per window); v1 = raw frame-max code for ranking (AUROC, precision@100, selection),
and the same HMM fitted on each selected v1 neuron's raw code for F1 / time.

Steps (CPU, defaultp):
  cache --model KEY [--codes PATH]   -> BEE22/cache/mice/<KEY>_{raw,pon}.npy (aligned to the eval frame table)
  diag  --model KEY                  -> OUT22/diag/<KEY>.json (Step 0: per-neuron AUROC / discrimination on all pools and
                                        on SG pools, top-20 lists, Matryoshka level, decoder cosines)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'scripts/eci/v2'))
import v21_ladder as L  # noqa: E402

T = L.T
OUT22 = REPO / 'results/vision/eci_v2_2'
BEE22 = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_2')
BEH = ('nose_nose', 'nt')
NAME = {'nt': 'nose-tail (nt)', 'nose_nose': 'nose-nose (nose_nose)', 'nn': 'nose-nose-mutual (nn)',
        'np': 'nose-nose-passive (np)', 'groom_any': 'grooming-to-focal (any)'}
PREFIX = (128, 256, 512, 2048)
log = L.log


# ---------------------------------------------------------------------------------------------- frames
def mice_table():
    """eval frame table (v21_ladder.frames) + annotator per video (experiment.csv, observation_id = obs) + sg flag."""
    F, SP = L.frames('mice')
    ex = pd.read_csv(REPO / 'data/mice/v1/experiment.csv')
    ex = ex[ex.annotation_file.notna()]
    assert ex.observation_id.is_unique
    # rd41_3 (= het_kmt5b_m_3): experiment.csv says SG, but the annotator is recorded as unknown upstream (blank in the
    # lab's tracking sheet; user 2026-11-04) -> treated as UNKNOWN ('NA'): not in the SG selection set, not in the
    # non-SG evaluation set
    ann = ex.set_index('observation_id').annotator.where(ex.set_index('observation_id').pool.astype(str) != 'rd41_3', 'NA')
    F['annot'] = F.obs.map(ann).fillna('NA').values
    F['sg'] = (F.annot.values == 'SG')
    F['nonsg'] = ~np.isin(F.annot.values, ('SG', 'NA'))
    # three-class label for the specificity measures: 1 = nt only, 2 = nose_nose only, 0 = neither, -1 = both
    a, b = F.y_nt.values, F.y_nose_nose.values
    F['cls'] = np.where(a & b, -1, np.where(a, 1, np.where(b, 2, 0)))
    return F, SP


def obs_starts(F):
    o = F.obs.values
    br = np.r_[True, o[1:] != o[:-1]]
    st = np.r_[np.flatnonzero(br), len(F)].astype(np.int64)
    assert len(st) - 1 == F.obs.nunique()
    return st


# ---------------------------------------------------------------------------------------------- readouts
def cache_paths(key):
    d = BEE22 / 'cache' / 'mice'
    return d / f'{key}_raw.npy', d / f'{key}_pon.npy'


def build_cache(key, codes=None, rows=None):
    """raw codes + HMM P(on) of every neuron, aligned to the eval frame table (exactly v21_ladder.readout)."""
    from multiprocessing import get_context
    from src.eci.v2 import hmm as HM
    import v20 as V20
    import eval as EV
    F, _ = mice_table()
    pr, pp = cache_paths(key)
    pr.parent.mkdir(parents=True, exist_ok=True)
    if key.startswith('v1_'):
        X, _ = EV.load_dict('mice', key, F)
        np.save(pr, np.asarray(X, np.float32))
        return
    od = T.BEE / 'mice' / 'codes'
    R = pd.read_parquet(rows or (od / 'eval_rows.parquet'))
    E = T.frame_sets('mice')['eval']
    rix = pd.Series(np.arange(len(R)), index=pd.MultiIndex.from_frame(R)).reindex(pd.MultiIndex.from_frame(E)).values
    assert not np.isnan(rix).any()
    rix = rix.astype(np.int64)
    st = np.r_[0, np.flatnonzero((R.obs.values[1:] != R.obs.values[:-1]) | (R.series.values[1:] != R.series.values[:-1])) + 1, len(R)]
    raw = np.load(codes or (od / f'{key}_eval.npy')).astype(np.float32)
    Xt = np.ascontiguousarray(raw.T)
    V20._G.update(Xc=Xt, sc=st.astype(np.int64))
    with get_context('fork').Pool(int(os.environ.get('SLURM_CPUS_PER_TASK', 8))) as pool:
        prm = [None] * Xt.shape[0]
        for j, p in pool.imap_unordered(V20._fit_one, range(Xt.shape[0]), chunksize=16):
            prm[j] = p
    pon = np.zeros_like(raw)
    for j in range(Xt.shape[0]):
        pon[:, j] = HM.apply(Xt[j], st, prm[j])[0]
    np.save(pr, raw[rix].astype(np.float16))
    np.save(pp, pon[rix])


def load(key):
    """-> (rank readout X, raw code R, P(on) or None (v1)) aligned to the eval frame table, float32."""
    pr, pp = cache_paths(key)
    R = np.load(pr).astype(np.float32)
    if key.startswith('v1_'):
        return R, R, None
    P = np.load(pp)
    return P, R, P


def hmm_cols(R, cols, st):
    from src.eci.v2 import hmm as HM
    out = np.zeros((len(R), len(cols)), np.float32)
    for i, j in enumerate(cols):
        x = np.asarray(R[:, j], np.float64)
        out[:, i] = HM.apply(x, st, HM.fit(x, st, K=3))[0]
    return out


# ---------------------------------------------------------------------------------------------- AUROC kernels
def auroc_cols(X, rows, y):
    """AUROC (ties 0.5) of every column of X on frames rows with labels y -> (W,)."""
    from scipy.stats import rankdata
    y = np.asarray(y, bool)
    P = int(y.sum())
    N = len(y) - P
    if P == 0 or N == 0:
        return np.full(X.shape[1], np.nan)
    out = np.empty(X.shape[1])
    for c0 in range(0, X.shape[1], 256):
        r = rankdata(np.asarray(X[rows, c0:c0 + 256], np.float64), axis=0)
        out[c0:c0 + 256] = (r[y].sum(0) - P * (P + 1) / 2) / (P * N)
    return out


def neuron_stats(X, F, rows):
    """per-neuron AUROC / specificity measures on frames rows -> dict of (W,) arrays."""
    cls = F.cls.values[rows]
    out = {'fire': (X[rows] > 0).mean(0)}
    for b in BEH:
        out[f'auc_{b}'] = auroc_cols(X, rows, F[f'y_{b}'].values[rows])
    m = (cls == 1) | (cls == 2)
    out['disc'] = auroc_cols(X, rows[m], cls[m] == 1)  # discrimination AUROC: nt (positive) vs nose_nose, exclusive frames
    for b, c in (('nt', 1), ('nose_nose', 2)):
        m2 = (cls == c) | (cls == 0)
        out[f'auc0_{b}'] = auroc_cols(X, rows[m2], cls[m2] == c)  # behaviour vs frames with neither label
    out['bal_nt'] = 0.5 * (out['auc0_nt'] + out['disc'])
    out['bal_nose_nose'] = 0.5 * (out['auc0_nose_nose'] + (1 - out['disc']))
    return out


ELIG = 0.002  # analysis.ELIG: a neuron is eligible iff it fires (readout > 0) on >= 0.2% of the selection frames


def select(st, rule):
    """st = neuron_stats on the selection frames -> {behaviour: neuron}. Rules:
       auc     = per behaviour the best AUROC (behaviour vs every other frame), the current rule;
       bal     = per behaviour the best balanced AUROC = 0.5 x [AUROC vs frames with neither label + discrimination AUROC
                 vs the sibling behaviour] (no constraint that the two picks differ);
       balpair = the pair (i != j) maximising bal_nose_nose[i] + bal_nt[j]."""
    el = st['fire'] >= ELIG
    sc = {b: np.where(el, st[('auc_' if rule == 'auc' else 'bal_') + b], np.nan) for b in BEH}
    pick = {b: int(np.nanargmax(sc[b])) for b in BEH}
    if rule == 'balpair' and pick['nt'] == pick['nose_nose']:
        i = pick['nt']
        a = sc['nose_nose'].copy(); a[i] = np.nan
        c = sc['nt'].copy(); c[i] = np.nan
        o1 = sc['nose_nose'][i] + np.nanmax(c)
        o2 = np.nanmax(a) + sc['nt'][i]
        pick = {'nose_nose': i, 'nt': int(np.nanargmax(c))} if o1 >= o2 else {'nose_nose': int(np.nanargmax(a)), 'nt': i}
    return pick


# ---------------------------------------------------------------------------------------------- Step 0
def cmd_diag(a):
    import torch
    key = a.model
    F, SP = mice_table()
    X, R, P = load(key)
    res = {'key': key}
    W = X.shape[1]
    dec = None
    if not key.startswith('v1_'):
        ck = torch.load(T.BEE / 'mice' / 'sae' / key / 'sae_theta.pt', map_location='cpu', weights_only=False)
        sd = ck['state_dict'] if 'state_dict' in ck else ck.get('sae', ck)
        Wd = [v for k, v in sd.items() if k.endswith('W_dec')][0].float().numpy()
        dec = Wd / np.linalg.norm(Wd, axis=1, keepdims=True)
    for part, rows in (('all', np.arange(len(F))), ('sg', np.flatnonzero(F.sg.values)), ('nonsg', np.flatnonzero(~F.sg.values))):
        st = neuron_stats(X, F, rows)
        el = st['fire'] >= ELIG
        D = {'n_eligible': int(el.sum()), 'picks': {r: select(st, r) for r in ('auc', 'bal', 'balpair')}, 'top20': {}}
        for b in BEH:
            o = np.argsort(-np.where(el, st[f'auc_{b}'], -1))[:20]
            top = st[f'auc_{b}'][o[0]]
            D['top20'][b] = [{'neuron': int(j), 'level': next(p for p in PREFIX if j < p), 'auc': float(st[f'auc_{b}'][j]),
                              'disc_nt_vs_nn': float(st['disc'][j]), 'bal': float(st[f'bal_{b}'][j]), 'fire': float(st['fire'][j]),
                              **({'cos_to_top': float(dec[j] @ dec[o[0]])} if dec is not None else {})} for j in o]
            # best "specific" neuron among near-best detectors (AUROC within 0.05 of the best)
            near = np.flatnonzero(el & (st[f'auc_{b}'] >= top - 0.05))
            sgn = 1 if b == 'nt' else -1
            jb = int(near[np.argmax(sgn * st['disc'][near])])
            D[f'most_specific_near_best_{b}'] = {'neuron': jb, 'auc': float(st[f'auc_{b}'][jb]), 'disc_nt_vs_nn': float(st['disc'][jb]),
                                                'n_near': int(len(near))}
            jb2 = int(np.nanargmax(np.where(el, st[f'bal_{b}'], np.nan)))
            D[f'best_bal_{b}'] = {'neuron': jb2, 'auc': float(st[f'auc_{b}'][jb2]), 'disc_nt_vs_nn': float(st['disc'][jb2]),
                                  'level': next(p for p in PREFIX if jb2 < p)}
        if dec is not None:  # children of the top nose_nose neuron: decoder cosine > 0.3, at a deeper Matryoshka level
            j0 = D['top20']['nose_nose'][0]['neuron']
            cs = dec @ dec[j0]
            ch = [j for j in np.argsort(-cs)[1:40] if cs[j] > 0.3]
            D['children_of_top_nose_nose'] = [{'neuron': int(j), 'cos': float(cs[j]), 'level': next(p for p in PREFIX if j < p),
                                               'auc_nt': float(st['auc_nt'][j]), 'auc_nose_nose': float(st['auc_nose_nose'][j]),
                                               'disc_nt_vs_nn': float(st['disc'][j]), 'fire': float(st['fire'][j])} for j in ch]
        # distribution summary: how many eligible neurons are nt-specific (disc > 0.6) with nt AUROC > 0.7
        D['n_nt_specific'] = {f'auc_nt>{t}': int((el & (st['auc_nt'] > t) & (st['disc'] > 0.6)).sum()) for t in (0.6, 0.7, 0.75)}
        D['n_nn_specific'] = {f'auc_nose_nose>{t}': int((el & (st['auc_nose_nose'] > t) & (st['disc'] < 0.4)).sum()) for t in (0.6, 0.7, 0.75)}
        (BEE22 / 'diag').mkdir(parents=True, exist_ok=True)
        np.savez_compressed(BEE22 / 'diag' / f'{key}_{part}.npz', **st)
        res[part] = D
    (OUT22 / 'diag').mkdir(parents=True, exist_ok=True)
    (OUT22 / 'diag' / f'{key}.json').write_text(json.dumps(res, indent=1))
    log(f'diag {key} done')


def cmd_cache(a):
    t0 = time.time()
    build_cache(a.model, a.codes or None, a.rows or None)
    log(f'cache {a.model} ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=('cache', 'diag'))
    p.add_argument('--model', default='')
    p.add_argument('--codes', default='')
    p.add_argument('--rows', default='')
    a = p.parse_args()
    {'cache': cmd_cache, 'diag': cmd_diag}[a.step](a)
