"""
ECI v2.0 proximity-matched test (pre-registered in scripts/eci/v2/v20_gate.py AMENDMENT 1 (d)) + the ants 5 fps probe
frame table used by Part B. CPU only.

Steps:
    proxy --dom mice|ants|antsprobe  per-frame proximity proxy of the frame table -> OUT/proxy_<dom>.npy
    prox  --dom mice|ants            stratified AUROC of v1 (11 seeds) and v2.0 (P(on), continuous) best neurons
    antsgate                         Part A gate of v1 / v2.0 on the ants 5 fps probe frames (for the DYN3 comparison)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
for p in ('', 'scripts/eci/v2', 'scripts/eci/v2_eval', 'scripts/eci/v2_prob', 'scripts/eci/v2_behav', 'scripts/eci/v2_diag'):
    sys.path.insert(0, str(REPO / p))
EC = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes')
OUT = REPO / 'results/vision/eci_v2_0B/gate'
PROX_CELLS = ('groom_any', 'Y2F', 'B2F', 'nn', 'np', 'nt', 'nose_nose')
log = lambda s: print(time.strftime('%H:%M:%S'), s, flush=True)  # noqa: E731


def grid_proxy(tf, pos, raw, nF, Gs):
    """per frame: min distance (patches) between distinct 8-connected components of tokens with raw >= 0.25."""
    from scipy import ndimage
    out = np.full(nF, np.nan, np.float32)
    m = raw >= 0.25
    tf, pos = tf[m], pos[m]
    o = np.argsort(tf, kind='stable')
    tf, pos = tf[o], pos[o]
    st = np.r_[0, np.cumsum(np.bincount(tf, minlength=nF))]
    S8 = np.ones((3, 3), bool)
    for f in range(nF):
        p = pos[st[f]:st[f + 1]]
        if len(p) == 0:
            continue
        G = int(Gs[f])
        g = np.zeros((G, G), bool)
        g.flat[p] = True
        lab, n = ndimage.label(g, structure=S8)
        if n <= 1:
            out[f] = 0.0
            continue
        best = np.inf
        for c in range(1, n):
            d = ndimage.distance_transform_edt(lab != c)
            best = min(best, float(d[(lab > c)].min()))
        out[f] = best - 1.0  # adjacent cells of different components cannot occur (8-conn); distance - 1 = gap cells
    return out


def store_meta(root, dom):
    """behav-style store (v2_prob / v2_behav): F (vid, k, G), tf, pos, raw."""
    import behav as BH
    BH.BEE = root
    St = BH.Store(dom, 'v2', 'dense', 1024)
    return St.F, St.tf, St.pos, St.raw


def cmd_proxy(a):
    import eval as EV
    OUT.mkdir(parents=True, exist_ok=True)
    if a.dom == 'mice':  # eval frames = v2_prob probe windows (F.code_row -> probe order)
        Fs, tf, pos, raw = store_meta(EC / 'v2_prob', 'mice')
        Pr = pd.read_parquet(REPO / 'results/vision/eci_v2_prob/mice/probe.parquet')
        px = grid_proxy(tf, pos, raw, len(Fs), Fs.G.values)
        key = lambda F_: pd.MultiIndex.from_arrays([F_.vid.values.astype(int), F_.k.values.astype(int)])  # noqa: E731
        o = pd.Series(np.arange(len(Fs)), index=key(Fs)).reindex(key(Pr)).values
        pr = px[o]
        F = EV.frame_table('mice')
        np.save(OUT / 'proxy_mice.npy', pr[F.code_row.values])
    elif a.dom == 'antsprobe':
        Fs, tf, pos, raw = store_meta(EC / 'v2_prob', 'ants')
        Pr = pd.read_parquet(REPO / 'results/vision/eci_v2_prob/ants/probe.parquet')
        px = grid_proxy(tf, pos, raw, len(Fs), Fs.G.values)
        key = lambda F_: pd.MultiIndex.from_arrays([F_.vid.values.astype(int), F_.k.values.astype(int)])  # noqa: E731
        o = pd.Series(np.arange(len(Fs)), index=key(Fs)).reindex(key(Pr)).values
        np.save(OUT / 'proxy_antsprobe.npy', px[o])
    else:  # ants eval frames = full1 1 fps frames (f1_idx)
        import diag as D
        T2 = D.T2
        old = T2.BEE
        T2.BEE = D.BEE
        T2.KINDS['ants'].setdefault('full1', {'series': D.F1_SERIES['ants'][0]})
        try:
            U = T2.Union('ants', 'full1')
        finally:
            T2.BEE = old
        F = EV.frame_table('ants')
        import v20 as V
        c = V.dcfg('ants')
        Gser = {sc['name']: sc['S'] // 16 for sc in c['series']}
        Gs = U.F.series.map(Gser).values
        px = grid_proxy(U.tf, U.pos, U.raw, len(U.F), Gs)
        np.save(OUT / 'proxy_ants.npy', px[F.f1_idx.values])
    log(f'proxy {a.dom} saved')


def auroc(s, y):
    from scipy.stats import rankdata
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def strat(F, X, use, SP, cells_res, proxy):
    out = {}
    for cell, r in cells_res.items():
        ser, b = cell.split('|')
        if b not in PROX_CELLS:
            continue
        vm = F[f'v_{b}'].values & (F.series.values == ser) & use & np.isfinite(proxy)
        idx = np.flatnonzero(vm)
        y = F[f'y_{b}'].values[idx].astype(bool)
        px = proxy[idx]
        edges = np.unique(np.quantile(px, np.linspace(0, 1, 11)))
        sb = np.clip(np.searchsorted(edges, px, side='right') - 1, 0, len(edges) - 2) if len(edges) > 1 else np.zeros(len(px), int)
        units = F.unit.values[idx]
        A = []
        for nm, h, j in r['AUROC_neurons']:
            half = np.array([SP[nm][u] for u in units])
            o = half == 1 - h
            c = np.asarray(X[idx, j], np.float64)
            num = den = 0.0
            for s_ in np.unique(sb[o]):
                m = o & (sb == s_)
                npos, nneg = int(y[m].sum()), int((~y[m]).sum())
                if npos >= 10 and nneg >= 10:
                    num += npos * auroc(c[m], y[m])
                    den += npos
            if den:
                A.append(num / den)
        out[cell] = {'strat_AUROC': float(np.mean(A)) if A else None, 'proxy_AUROC': auroc(-px, y),
                     'AUROC': r['AUROC'], 'n_strata': int(len(edges) - 1)}
    return out


def cmd_prox(a):
    import analysis as AN
    import eval as EV
    import os
    os.environ.setdefault('V2_OUT', str(REPO / 'results/vision/eci_v2_0B'))
    os.environ.setdefault('V2_BEE', str(EC / 'v2_0B'))
    import v20_eval as VE
    import v20_gate as G
    dom = a.dom
    F = EV.frame_table(dom)
    SP = AN.splits(F.unit.values, EV.SPLITS)
    starts = AN.seq_starts(F)
    proxy = np.load(OUT / f'proxy_{dom}.npy')
    S = json.loads((G.OUT / f'{dom}_scores.json').read_text())
    res = {'v1': {}, 'v2': {}}
    for s in range(11):
        X, use = EV.load_dict(dom, f'v1_s{s}', F)
        res['v1'][s] = strat(F, X, use, SP, S['v1'][str(s)], proxy)
    for k, suf in (('hmm', '_hmm'), ('smooth', '_smooth')):
        X, use = VE.deliverable_codes(dom, F, suf)
        res['v2'][k] = strat(F, X, use, SP, S['v2'][k], proxy)
    (OUT / f'prox_{dom}.json').write_text(json.dumps(res, indent=1, default=float))
    for cell in res['v2']['hmm']:
        v1s = [res['v1'][s][cell]['strat_AUROC'] for s in res['v1'] if res['v1'][s].get(cell, {}).get('strat_AUROC') is not None]
        log(f'{dom} {cell}: proxy AUROC {res["v2"]["hmm"][cell]["proxy_AUROC"]:.3f} | v1 AUROC '
            f'{np.mean([res["v1"][s][cell]["AUROC"] for s in res["v1"]]):.3f} strat {np.mean(v1s):.3f} | v2 P(on) '
            f'{res["v2"]["hmm"][cell]["AUROC"]:.3f} strat {res["v2"]["hmm"][cell]["strat_AUROC"]} | v2 cont '
            f'{res["v2"]["smooth"][cell]["AUROC"]:.3f} strat {res["v2"]["smooth"][cell]["strat_AUROC"]}')


def ants_probe_table():
    """eval.py-style frame table of the ants v2_prob Tier B probe frames (labels from the eci30 annotation rows)."""
    import eval as EV
    import v20 as V
    Pr = pd.read_parquet(REPO / 'results/vision/eci_v2_prob/ants/probe.parquet')
    c = V.dcfg('ants')
    ann = pd.read_csv(REPO / 'dataset/ants/eci30/annotations.csv', usecols=lambda x: x == 'observation_id' or x.startswith('Y_'))
    src = {}
    for sc in c['series']:
        for v in json.loads((REPO / sc['plan']).read_text())['videos']:
            src[(sc['name'], v['obs'])] = v['src_lo']
    rows = np.array([src[(a_, b_)] for a_, b_ in zip(Pr.series.values, Pr.obs.values)]) + Pr.k.values
    A_ = ann.iloc[rows].reset_index(drop=True)
    assert (A_.observation_id.values == Pr.obs.values).all()
    F = pd.concat([Pr.reset_index(drop=True), A_.drop(columns='observation_id')], axis=1)  # Pr order kept
    F['experiment'] = F.series
    F['unit'] = F.obs
    for b, cols in EV.BEH['ants'].items():
        F[f'y_{b}'] = (F[cols].fillna(0).to_numpy() > 0).any(1)
        F[f'v_{b}'] = F[cols].notna().all(1).to_numpy()
    return F, Pr


def row_counts_5fps(F, SP, BEH):
    """eval.py row_counts with the bout step of contiguous 5 fps frames (1 grid step, as mice)."""
    out = {}
    for ser in F.series.unique():
        for b in BEH:
            m = F[f'v_{b}'].values & (F.series.values == ser)
            g = F[m]
            y = g[f'y_{b}'].values
            o, k = g.obs.values, g.k.values
            new = y & np.r_[True, (o[1:] != o[:-1]) | (k[1:] != k[:-1] + 1) | ~y[:-1]]
            h = np.array([SP['atlas'][u] for u in g.unit.values])
            r = {'frames': int(len(g)), 'pos': int(y.sum()), 'bouts': int(new.sum())}
            for hh in (0, 1):
                r[f'h{hh}_bouts'] = int((new & (h == hh)).sum())
                r[f'h{hh}_videos_pos'] = int(len(np.unique(o[y & (h == hh)])))
            r['eligible'] = bool(min(r['h0_bouts'], r['h1_bouts']) >= 20 and min(r['h0_videos_pos'], r['h1_videos_pos']) >= 3)
            out[f'{ser}|{b}'] = r
    return out


def cmd_antsgate(a):
    import analysis as AN
    import eval as EV
    import os
    os.environ.setdefault('V2_OUT', str(REPO / 'results/vision/eci_v2_0B'))
    os.environ.setdefault('V2_BEE', str(EC / 'v2_0B'))
    import v20_eval as VE
    import v20_gate as G
    F, Pr = ants_probe_table()
    SP = AN.splits(F.unit.values, EV.SPLITS)
    elig = row_counts_5fps(F, SP, EV.BEH['ants'])
    proxy = np.load(OUT / 'proxy_antsprobe.npy')
    res = {'eligibility': elig, 'v1': {}, 'v2': {}, 'prox': {'v1': {}, 'v2': {}}}
    for s in range(11):
        X = np.load(EC / f'v2_prob/ants/codes/v1_s{s}.npy').astype(np.float32)
        res['v1'][s] = G.score('ants', X, np.ones(len(F), bool), F, SP)
        res['prox']['v1'][s] = strat(F, X, np.ones(len(F), bool), SP, res['v1'][s], proxy)
    for k, suf in (('hmm', '_hmm'), ('smooth', '_smooth')):
        X, use = VE.deliverable_codes('ants', F, suf)
        res['v2'][k] = G.score('ants', X, use, F, SP)
        res['prox']['v2'][k] = strat(F, X, use, SP, res['v2'][k], proxy)
    (OUT / 'antsprobe_scores.json').write_text(json.dumps(res, default=float))
    R = {k: G.gate_rows('ants', {str(s): v for s, v in res['v1'].items()}, res['v2'][k], elig) for k in ('hmm', 'smooth')}
    (OUT / 'antsprobe_gate.json').write_text(json.dumps(R, indent=1, default=float))
    for k in R:
        log(f'antsprobe {k}: ' + ', '.join(f'{c} {"P" if r["PASS"] else "F"}{"" if r["eligible"] else "(d)"}' for c, r in R[k].items()))


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=('proxy', 'prox', 'antsgate'))
    p.add_argument('--dom', default='mice')
    a = p.parse_args()
    {'proxy': cmd_proxy, 'prox': cmd_prox, 'antsgate': cmd_antsgate}[a.step](a)


if __name__ == '__main__':
    main()
