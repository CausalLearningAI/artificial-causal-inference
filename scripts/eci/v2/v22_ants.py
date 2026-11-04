"""
ECI v2.2 ants non-regression (C4 of scripts/eci/v2/v22_prereg.md): grooming-to-focal (any) (cell pooled|groom_any) under
the v21 extended pipeline and rule (v21_extended.table pooled frame table, v21_ladder.score2 cross-fitted best-AUROC pick,
12 directions over videos), for V and for a variant. Readout = per-neuron 3-state HMM P(on) (v21_ladder.readout).
Per direction: held-out video x video AUROC pair counts (for a video bootstrap within series) and per-video sum P(on) /
true on-frames (Pearson r of per-video time).
Steps: tag --model KEY (CPU; KEY = V_s1001 | V_s1001@box3 | VMS_s1004@ms ...) ; report (CPU) -> OUT22/ants_report.json
"""
import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v21_extended as X  # noqa: E402
import v22_eval as E  # noqa: E402

L, T = X.L, X.L.T
OUT22 = E.OUT22
BEE22 = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_2')
CELL = 'pooled|groom_any'
log = L.log


def codes_path(key):
    base, ro = (key.split('@') + [''])[:2]
    if base.startswith('Vall_') and not ro:  # v2.3
        return Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_3') / 'ants' / 'codes' / f'{base}_max_eval.npy'
    if not ro:
        p = BEE22 / 'ants' / 'codes' / f'{base}_max_eval.npy'
        return p if p.exists() else T.BEE / 'ants' / 'codes' / f'{base}_eval.npy'
    if ro == 'top10':
        return T.BEE / 'ants' / 'codes' / f'{base}_eval_top10.npy'
    return BEE22 / 'ants' / 'codes' / f'{base}_{ro}_eval.npy'


def readout(key, F):
    from multiprocessing import get_context
    from src.eci.v2 import hmm as HM
    import v20 as V20
    od = T.BEE / 'ants' / 'codes'
    R = pd.read_parquet(od / 'eval_rows.parquet')
    Ef = T.frame_sets('ants')['eval']
    rix = pd.Series(np.arange(len(R)), index=pd.MultiIndex.from_frame(R)).reindex(pd.MultiIndex.from_frame(Ef)).values
    assert not np.isnan(rix).any()
    rix = rix.astype(np.int64)
    st = np.r_[0, np.flatnonzero((R.obs.values[1:] != R.obs.values[:-1]) | (R.series.values[1:] != R.series.values[:-1])) + 1, len(R)]
    raw = np.load(codes_path(key)).astype(np.float32)
    Xt = np.ascontiguousarray(raw.T)
    V20._G.update(Xc=Xt, sc=st.astype(np.int64))
    with get_context('fork').Pool(int(os.environ.get('SLURM_CPUS_PER_TASK', 8))) as pool:
        prm = [None] * Xt.shape[0]
        for j, p in pool.imap_unordered(V20._fit_one, range(Xt.shape[0]), chunksize=16):
            prm[j] = p
    pon = np.zeros_like(raw)
    for j in range(Xt.shape[0]):
        pon[:, j] = HM.apply(Xt[j], st, prm[j])[0]
    return pon[rix]


def cmd_tag(a):
    key = a.model
    t0 = time.time()
    F, P, SP, elig = X.table('ants')
    keep = [c for c in P.columns if not (c.startswith('y_') or c.startswith('v_'))] + ['y_groom_any', 'v_groom_any']
    P = P[keep]
    Xr = readout(key, F)
    res = L.score2(P, Xr, np.ones(len(P), bool), SP)[CELL]
    D = L.cell_dirs(P, SP, {CELL: res}, 'gate')[CELL]
    y_all = P.y_groom_any.values.astype(bool)
    vids = sorted(set(P.obs.values[P.v_groom_any.values]))
    vix = pd.Series(np.arange(len(vids)), index=vids)
    out = {'key': key, 'AUROC': res['AUROC'], 'videos': vids, 'series': pd.Series(P.series.values, index=P.obs.values).groupby(level=0).first().reindex(vids).tolist(),
           'dirs': []}
    for fr, j, _, _ in D:
        u = vix.reindex(P.obs.values[fr]).values.astype(np.int64)
        s = Xr[fr, j].astype(np.float64)
        y = y_all[fr]
        A, Pu, Nu = E.uu_auc(s, y, u, len(vids))
        out['dirs'].append({'neuron': int(j), 'auc': (A, Pu, Nu), 'S': np.bincount(u, weights=s, minlength=len(vids)),
                            'T': np.bincount(u, weights=y, minlength=len(vids)), 'n': np.bincount(u, minlength=len(vids)).astype(float)})
    mine = float(np.mean([d['auc'][0].sum() / (d['auc'][1].sum() * d['auc'][2].sum()) for d in out['dirs']]))
    out['AUROC_recomputed'] = mine
    ep = X.OUT / 'tags' / 'ants' / f'{key}.pkl'
    if ep.exists():
        Ec = pickle.load(open(ep, 'rb'))['cells'][CELL]
        out['identity_vs_extended'] = {'AUROC_abs_diff': abs(mine - Ec['AUROC']),
                                       'picks_identical': [d['neuron'] for d in Ec['dirs']] == [d['neuron'] for d in out['dirs']]}
        log(f'IDENTITY ants {key}: {out["identity_vs_extended"]}')
        if out['identity_vs_extended']['AUROC_abs_diff'] > 1e-6 or not out['identity_vs_extended']['picks_identical']:
            raise SystemExit('ANTS IDENTITY FAILED')
    (OUT22 / 'ants_tags').mkdir(parents=True, exist_ok=True)
    pickle.dump(out, open(OUT22 / 'ants_tags' / f'{key}.pkl', 'wb'))
    log(f'ants tag {key}: AUROC {mine:.4f} ({time.time() - t0:.0f}s)')


def model_boot(tg, W):
    """W (R, nvid) video weights -> AUROC (R,), Pearson r of per-video sum P(on) vs true (R,), each mean over directions."""
    au, rr = [], []
    for d in tg['dirs']:
        A, Pu, Nu = d['auc']
        au.append(np.einsum('ru,uv,rv->r', W, A, W) / ((W @ Pu) * (W @ Nu)))
        ho = d['n'] > 0
        rr.append(E.wpearson(d['S'][ho], d['T'][ho], W[:, ho]))
    return np.nanmean(au, 0), np.nanmean(rr, 0)


def cmd_report(a):
    NB = 1000
    arms = json.loads((OUT22 / 'ants_arms.json').read_text())
    tags = {k: pickle.load(open(OUT22 / 'ants_tags' / f'{k}.pkl', 'rb')) for ks in arms.values() for k in ks}
    t0 = next(iter(tags.values()))
    vids, ser = np.array(t0['videos']), np.array(t0['series'])
    rng = np.random.default_rng(20261104 + 1)
    W = np.ones((NB + 1, len(vids)))
    for s in np.unique(ser):
        ix = np.flatnonzero(ser == s)
        W[1:, ix] = np.stack([np.bincount(rng.integers(0, len(ix), len(ix)), minlength=len(ix)) for _ in range(NB)])
    per = {arm: np.stack([np.stack(model_boot(tags[k], W)) for k in ks]) for arm, ks in arms.items()}  # (seeds, 2, NB+1)
    res = {'levels': {}, 'contrasts': {}, 'picks': {arm: {k: pd.Series([d['neuron'] for d in tags[k]['dirs']]).value_counts().to_dict()
                                                            for k in ks} for arm, ks in arms.items()}}
    for arm, A in per.items():
        res['levels'][arm] = {M: {'median': float(np.median(A[:, i, 0])), 'min': float(A[:, i, 0].min()), 'max': float(A[:, i, 0].max())}
                              for i, M in enumerate(('AUROC', 'pearson_r_time'))}
    for arm in per:
        if arm == 'V':
            continue
        r = {}
        for i, M in enumerate(('AUROC', 'pearson_r_time')):
            ax, ay = per[arm][:, i, 1:], per['V'][:, i, 1:]
            ix = rng.integers(0, len(ax), (NB, len(ax)))
            iy = rng.integers(0, len(ay), (NB, len(ay)))
            q = np.arange(NB)[:, None]
            dd = np.nanmean(ax[ix, q], 1) - np.nanmean(ay[iy, q], 1)
            r[M] = {'diff': float(per[arm][:, i, 0].mean() - per['V'][:, i, 0].mean()), 'ci': np.nanpercentile(dd, [2.5, 97.5]).tolist()}
        r['C4'] = bool(r['AUROC']['ci'][0] >= -0.02 and r['pearson_r_time']['ci'][0] >= -0.02)
        res['contrasts'][f'{arm} - V'] = r
    res['picks'] = {a_: {k: {int(n): int(c) for n, c in v.items()} for k, v in p.items()} for a_, p in res['picks'].items()}
    (OUT22 / 'ants_report.json').write_text(json.dumps(res, indent=1))
    log(json.dumps({k: v for k, v in res.items() if k != 'picks'}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=('tag', 'report'))
    p.add_argument('--model', default='')
    a = p.parse_args()
    {'tag': cmd_tag, 'report': cmd_report}[a.step](a)
