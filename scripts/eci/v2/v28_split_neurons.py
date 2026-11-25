"""
ECI v2.8 split check (POST HOC robustness analysis, CPU only, no training; written 2026-11-25 after the v2.8 audit and the
v2.9 pre-registration). Question: is the annotator-based selection / evaluation split (select on annotator SG's 11 pools,
evaluate on the 12 non-SG pools) biased compared with random POOL-level splits?

Per dictionary, one pass over all eligible neurons computes AUROC (P(on) readout, ties 0.5) and average precision (AP; P(on)
ranked, raw code tie-break, same sort as v28_audit_neurons._ap_multi) for nose-nose, nose-tail and the discrimination
AUROC (nose-tail-only vs nose-nose-only frames) on MANY frame subsets at once. Subsets are unions of "cells" (cell = pool x
annotator; two pools have two annotators). AUROC of any subset comes from the cell-pair U-statistic matrix; AP of every
subset is accumulated in the same descending sweep.
Subsets: the official split (SG / non-SG), R = 50 random annotator-stratified pool splits (select 11 / evaluate 12),
leave-one-annotator-out (SG, SD, MF, MB), per annotator, per pool, all 23 pools.
Rule sgap = v29_prereg.md (candidates = top-20 eligible by select AUROC U top-20 by select AP; nose-nose: select
discrimination AUROC <= 0.4, nose-tail: != nose-nose and discrimination >= 0.6; both raw firing < 10% of all frames;
argmax select AP) applied on every select side. Precision@top-k/video (k = 3, 10) for every picked / fixed neuron per video.
usage: v28_split_neurons.py --key K   (CPU job; -> results/vision/eci_v2_8/split/work/K.pkl)
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit, prange

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v28_audit_neurons as A  # noqa: E402

C = A.C
OUT = A.REPO / 'results/vision/eci_v2_8/split'
WORK = OUT / 'work'
NSPLIT = 50
SEED = 20261125
FIRE_CAP = 0.10
FIXED = {'PREDk_s2703': {'nose_nose': 3593, 'nt': 3009}, 'v1_s0': {'nose_nose': 611, 'nt': 414},
         'V_s1002': {'nose_nose': 436, 'nt': 270}}  # PREDk s2703 = v2.8 sgap picks; v1 = documented picks; V s1002 = v2.7 rule auc
log = A.log


def stratum(ann_set):
    return ann_set if ann_set in ('SG', 'SD', 'MF', 'MB') else 'mixed'


def make_subsets(F):
    """cells, pools and the named subsets (bool masks over cells). Deterministic (same for every dictionary)."""
    m = F.annot.values != 'NA'
    cell_df = pd.DataFrame({'unit': F.unit.values[m], 'annot': F.annot.values[m]}).drop_duplicates().sort_values(['unit', 'annot'])
    cells = list(zip(cell_df.unit, cell_df.annot))
    pools = sorted(set(cell_df.unit))
    pann = {p: '+'.join(sorted({a for u, a in cells if u == p})) for p in pools}
    cpool = np.array([pools.index(u) for u, _ in cells])
    G = len(cells)

    def from_pools(ps):
        ps = set(ps)
        return np.array([pools[i] in ps for i in cpool])
    sub = {}
    sg = [p for p in pools if pann[p] == 'SG']
    sub['official|sel'] = from_pools(sg)
    sub['official|ev'] = ~sub['official|sel']
    strata = {}
    for p in pools:
        strata.setdefault(stratum(pann[p]), []).append(p)
    rng = np.random.default_rng(SEED)
    seen = set()
    r = 0
    while r < NSPLIT:
        odd = [s for s, v in strata.items() if len(v) % 2]
        extra = set(rng.choice(sorted(odd), size=11 - sum(len(v) // 2 for v in strata.values()), replace=False).tolist())
        sel = []
        for s in sorted(strata):
            v = list(rng.permutation(strata[s]))
            sel += v[:len(v) // 2 + (s in extra)]
        key = tuple(sorted(sel))
        if key in seen or set(sel) == set(sg):
            continue
        seen.add(key)
        assert len(sel) == 11
        sub[f'rand{r:02d}|sel'] = from_pools(sel)
        sub[f'rand{r:02d}|ev'] = ~sub[f'rand{r:02d}|sel']
        r += 1
    for a in ('SG', 'SD', 'MF', 'MB'):
        ev = [p for p in pools if a in pann[p].split('+')]
        sub[f'loao_{a}|sel'] = ~from_pools(ev)
        sub[f'loao_{a}|ev'] = from_pools(ev)
    for a in sorted({a for _, a in cells}):
        sub[f'annot_{a}'] = np.array([c[1] == a for c in cells])
    for p in pools:
        sub[f'pool_{p}'] = from_pools([p])
    sub['all'] = np.ones(G, bool)
    return cells, pools, pann, cpool, sub


# ---------------------------------------------------------------------------------------------- kernel
@njit(parallel=True, cache=True)
def _kern(Xt, St, cell, lab, Msub, G):
    """Xt (W, n) float32 P(on), St (W, n) float64 AP score, cell (n,) int64 (-1 = outside), lab (n,) int8 bits
    (1 nose-nose, 2 nose-tail, 4 discrimination positive = nose-tail only, 8 discrimination negative = nose-nose only),
    Msub (Q, G) uint8 -> U (W, 3, G, G) (sum over positives in cell a of #negatives in cell b below + 0.5 #tied),
    APacc (W, 2, Q) (sum over subset positives of the precision at that positive; divide by subset positives)."""
    W, n = Xt.shape
    Q = Msub.shape[0]
    U = np.zeros((W, 3, G, G))
    APacc = np.zeros((W, 2, Q))
    for j in prange(W):
        x = Xt[j]
        o = np.argsort(x)
        negb = np.zeros((3, G))
        gneg = np.zeros((3, G))
        i = 0
        while i < n:
            v = x[o[i]]
            e = i
            while e < n and x[o[e]] == v:
                r = o[e]
                g = cell[r]
                if g >= 0:
                    L = lab[r]
                    if (L & 1) == 0:
                        gneg[0, g] += 1
                    if (L & 2) == 0:
                        gneg[1, g] += 1
                    if (L & 8) != 0:
                        gneg[2, g] += 1
                e += 1
            for t in range(i, e):
                r = o[t]
                g = cell[r]
                if g >= 0:
                    L = lab[r]
                    for c in range(3):
                        if (L & (1 << c)) != 0:
                            for h in range(G):
                                U[j, c, g, h] += negb[c, h] + 0.5 * gneg[c, h]
            for t in range(i, e):
                g = cell[o[t]]
                if g >= 0:
                    for c in range(3):
                        negb[c, g] += gneg[c, g]
                        gneg[c, g] = 0.0
            i = e
        s = St[j]
        o2 = np.argsort(-s)
        cnt = np.zeros(G)
        tp = np.zeros((2, G))
        for i in range(n):
            r = o2[i]
            g = cell[r]
            if g < 0:
                continue
            cnt[g] += 1
            L = lab[r]
            for c in range(2):
                if (L & (1 << c)) != 0:
                    tp[c, g] += 1
                    for q in range(Q):
                        if Msub[q, g]:
                            num = 0.0
                            den = 0.0
                            for h in range(G):
                                if Msub[q, h]:
                                    num += tp[c, h]
                                    den += cnt[h]
                            APacc[j, c, q] += num / den
    return U, APacc


def sgap(st, el, fok):
    """rule sgap (v29_prereg.md) on one select side; st = dict of (W,) arrays auc_nn / auc_nt / disc / ap_nn / ap_nt."""
    def cand(b):
        sa = np.where(el, st[f'auc_{b}'], -np.inf)
        sp = np.where(el, np.nan_to_num(st[f'ap_{b}'], nan=-1.0), -np.inf)
        return sorted(set(np.argsort(-sa)[:20].tolist()) | set(np.argsort(-sp)[:20].tolist()))
    cn = [j for j in cand('nn') if el[j] and fok[j] and st['disc'][j] <= 0.4]
    if not cn:
        return None
    nn = cn[int(np.nanargmax([st['ap_nn'][j] for j in cn]))]
    ct = [j for j in cand('nt') if el[j] and fok[j] and j != nn and st['disc'][j] >= 0.6]
    if not ct:
        return None
    nt = ct[int(np.nanargmax([st['ap_nt'][j] for j in ct]))]
    return int(nn), int(nt)


def main(a):
    t0 = time.time()
    key = a.key
    F, _ = C.mice_table()
    cells, pools, pann, cpool, sub = make_subsets(F)
    G = len(cells)
    names = list(sub)
    Msub = np.stack([sub[k] for k in names]).astype(np.uint8)
    cix = {c: i for i, c in enumerate(cells)}
    cell = np.array([cix.get((u, an), -1) for u, an in zip(F.unit.values, F.annot.values)], np.int64)
    ynn, ynt, cls = F.y_nose_nose.values, F.y_nt.values, F.cls.values
    lab = (ynn * 1 + ynt * 2 + (cls == 1) * 4 + (cls == 2) * 8).astype(np.int8)
    # per-cell positive / negative counts per category (nose-nose, nose-tail, discrimination)
    PN = np.zeros((3, 2, G))
    for c, (pos, neg) in enumerate(((ynn, ~ynn), (ynt, ~ynt), (cls == 1, cls == 2))):
        PN[c, 0] = np.bincount(cell[cell >= 0], weights=pos[cell >= 0], minlength=G)
        PN[c, 1] = np.bincount(cell[cell >= 0], weights=neg[cell >= 0], minlength=G)
    pr, pp = A.mice_paths(key)
    R = np.load(pr)
    X = R.astype(np.float32) if key.startswith('v1_') else np.load(pp)
    n, W = X.shape
    assert n == len(F)
    log(f'{key}: loaded {X.shape}, {G} cells, {len(pools)} pools, {len(names)} subsets ({time.time() - t0:.0f}s)')
    el = np.zeros(W)
    fr = np.zeros(W)
    rmax = np.zeros(W)
    for c0 in range(0, n, 50000):
        el += (X[c0:c0 + 50000] > 0).sum(0)
        rc = np.asarray(R[c0:c0 + 50000], np.float32)
        fr += (rc > 0).sum(0)
        rmax = np.maximum(rmax, rc.max(0))
    el = el / n >= C.ELIG
    fr = fr / n
    fok = fr < FIRE_CAP
    cols = np.flatnonzero(el)
    AUC = np.full((len(names), 3, W), np.nan, np.float32)
    AP = np.full((len(names), 2, W), np.nan, np.float32)
    nb = 256
    for b0 in range(0, len(cols), nb):
        cc = cols[b0:b0 + nb]
        Xt = np.ascontiguousarray(X[:, cc].T, dtype=np.float32)
        St = Xt.astype(np.float64) + 1e-6 * np.ascontiguousarray(R[:, cc].T, dtype=np.float64) / np.maximum(rmax[cc, None], 1e-12)
        U, APacc = _kern(Xt, St, cell, lab, Msub, G)
        for q in range(len(names)):
            m = Msub[q].astype(float)
            for c in range(3):
                P, N = PN[c, 0] @ m, PN[c, 1] @ m
                if P > 0 and N > 0:
                    AUC[q, c, cc] = np.einsum('wgh,g,h->w', U[:, c], m, m) / (P * N)
            for c in range(2):
                P = PN[c, 0] @ m
                if P > 0:
                    AP[q, c, cc] = APacc[:, c, q] / P
        del Xt, St, U, APacc
        log(f'{key}: block {b0 // nb + 1}/{(len(cols) + nb - 1) // nb} ({time.time() - t0:.0f}s)')
    # identity check against the v2.8 audit (official split = SG / non-SG frames)
    chk = {}
    ap_ = A.WORK / f'mice_{key}.pkl'
    if ap_.exists():
        S = pickle.load(open(ap_, 'rb'))['stats']
        qi = {'sg': names.index('official|sel'), 'non': names.index('official|ev')}
        for nm in ('sg', 'non'):
            for c, b in enumerate(('nose_nose', 'nt', 'disc')):
                chk[f'auc|{nm}|{b}'] = float(np.nanmax(np.abs(AUC[qi[nm], c, cols] - S[f'auc|{nm}|{b}'][cols])))
            for c, b in enumerate(('nose_nose', 'nt')):
                chk[f'ap|{nm}|{b}'] = float(np.nanmax(np.abs(AP[qi[nm], c, cols] - S[f'ap|{nm}|{b}'][cols])))
        log(f'{key}: max |diff| vs audit stats: {chk}')
    # picks on every select side: sgap (= v29_prereg, firing cap), sgap_nocap (= the v2.8 audit rule without the cap),
    # bestap (eligible neuron with the best select AP per behaviour, no other constraint; may be the same neuron twice)
    picks = {'sgap': {}, 'sgap_nocap': {}, 'bestap': {}}
    for q, nm in enumerate(names):
        if not nm.endswith('|sel'):
            continue
        st = {'auc_nn': AUC[q, 0], 'auc_nt': AUC[q, 1], 'disc': AUC[q, 2], 'ap_nn': AP[q, 0], 'ap_nt': AP[q, 1]}
        picks['sgap'][nm[:-4]] = sgap(st, el, fok)
        picks['sgap_nocap'][nm[:-4]] = sgap(st, el, np.ones(W, bool))
        picks['bestap'][nm[:-4]] = tuple(int(np.nanargmax(np.where(el, np.nan_to_num(AP[q, c], nan=-1.0), -1.0))) for c in range(2))
    log(f"{key}: official picks " + ', '.join(f"{r} {p['official']}" for r, p in picks.items()))
    # best select-AP neuron per behaviour on the official split (no constraints; reported for v1)
    best_ap = {b: int(np.nanargmax(np.where(el, AP[names.index('official|sel'), c], -1))) for c, b in enumerate(('nose_nose', 'nt'))}
    neurons = {j for pr_ in picks.values() for p in pr_.values() if p for j in p} | set(best_ap.values())
    if key in FIXED:
        neurons |= set(FIXED[key].values())
    neurons = sorted(neurons)
    # per-video top-k (raw code ranking, earliest first, valid frames) for every neuron of interest
    rows = np.flatnonzero(cell >= 0)
    obs = F.obs.values[rows]
    vids = pd.unique(obs)
    vcell = pd.Series(cell[rows]).groupby(obs).first().reindex(vids).values
    topk = {}
    for b in ('nose_nose', 'nt'):
        y, v = F[f'y_{b}'].values[rows], F[f'v_{b}'].values[rows]
        vv = pd.DataFrame({'o': obs, 'y': y & v, 'v': v}).groupby('o', sort=False).agg(pos=('y', 'sum'), nval=('v', 'sum')).reindex(vids)
        topk[('pos', b)] = vv.pos.values.astype(float)
        topk[('nval', b)] = vv.nval.values.astype(float)
        for j in neurons:
            raw = np.asarray(R[rows, j], np.float32)
            for k in (3, 10):
                vi, h, p = A.topk_video(obs, y, v, raw, k)
                assert (vi == vids).all() and np.allclose(p, topk[('pos', b)])
                topk[(j, b, k)] = h
    res = {'key': key, 'W': W, 'cells': cells, 'pools': pools, 'pann': pann, 'cpool': cpool, 'names': names, 'Msub': Msub,
           'PN': PN, 'eligible': el, 'fire_raw': fr, 'picks': picks, 'best_ap_official': best_ap,
           'fixed': FIXED.get(key), 'neurons': neurons,
           'auc': {j: AUC[:, :, j] for j in neurons}, 'ap': {j: AP[:, :, j] for j in neurons},
           'vids': np.asarray(vids), 'vcell': vcell, 'topk': topk, 'check_vs_audit': chk, 'n_rows': n}
    # select-side AP of every pick (for the dictionary ranking) is in res['ap'][pick][q]
    if key == 'PREDk_s2703':  # frame / label description of the split (once)
        d = pd.DataFrame({'unit': F.unit.values[rows], 'annot': F.annot.values[rows], 'obs': obs, 'nn': ynn[rows], 'nt': ynt[rows]})
        res['describe'] = d.groupby(['unit', 'annot']).agg(videos=('obs', 'nunique'), frames=('obs', 'size'), nn=('nn', 'sum'), nt=('nt', 'sum')).reset_index()
        res['n_na_rows'] = int((F.annot.values == 'NA').sum())
    WORK.mkdir(parents=True, exist_ok=True)
    pickle.dump(res, open(WORK / f'{key}.pkl', 'wb'))
    log(f'{key}: done ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--key', required=True)
    main(p.parse_args())
