"""
ECI v2.5 scoring (pre-registration scripts/eci/v2/v25_prereg.md). Reuses the v2.3 / v2.4 code paths (v22_core, v22_eval,
v22_full, v23_ants, v24_top3) with module attributes patched here (the shared scripts are not edited):
  V25_EDGE=1 (default): the mice frame table = the v2.3 ext frames minus the edge frames (first / last L = 2 frames of each
     store video; BEE25/mice/codes/rows.parquet), caches for EVERY model (V, v1 s0 and the v2.5 arms) in BEE25/mice/cache
     aligned to that table (V's P(on) recomputed by the same HMM on the edge-excluded sequences), tags in OUT25/mice.
  V25_EDGE=0: the original ext table and the stored v2.3 caches (scorer identity for V), tags in OUT25/identity/mice.
  ants: the scored frames contain no edge frame (asserted), so V / v1 tags of v2.3 are reused; v2.5 arms tagged here.
Mice rules: auc, dist (nose-tail != nose-nose pick and discrimination AUROC >= 0.6 on the selection set), dist2048
  (escalation 1: dist with the nose-tail pick among latents >= 512 only).
Steps
  cache --dom D --key K   (CPU) -> BEE25/<dom>/cache/K_{raw,pon}.npy
  tag --dom D --key K     (CPU)
  prep --key K            (CPU, mice) precision@top3/video cells
  report --dom D          (CPU) -> OUT25/<dom>/report.json
  fwdcheck --dom D --key K   direct SAE forward pass on 200 frames == assembled codes
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

os.environ.setdefault('V22_FRAMES', 'ext')
sys.path.insert(0, str(Path(__file__).resolve().parent))
import v22_core as C  # noqa: E402
import v22_eval as E  # noqa: E402
import v22_full as VF  # noqa: E402
import v23_ants as AN  # noqa: E402
import v24_top3 as T3  # noqa: E402
import v25_train as TR  # noqa: E402

assert C.FRAMES == 'ext'
REPO, log = C.REPO, C.log
BEE25, OUT25, L = TR.BEE25, TR.OUT25, TR.L
EDGE = os.environ.get('V25_EDGE', '1') == '1'
NB = 1000
RULES = ('auc', 'dist', 'dist2048')
ARMS25 = tuple(TR.ARMS)
SEEDS = TR.SEEDS
VSEEDS = tuple(range(1001, 1007))
MODELS = {'V': [f'V_s{s}' for s in VSEEDS], 'v1s0': ['v1_s0'], **{a: [f'{a}_s{s}' for s in SEEDS] for a in ARMS25}}
MODELS_ANTS = {'V': MODELS['V'], 'v1': [f'v1_s{s}' for s in range(11)], 'v1s0': ['v1_s0'], **{a: MODELS[a] for a in ARMS25}}
MICE_OUT = OUT25 / 'mice' if EDGE else OUT25 / 'identity' / 'mice'
ANTS_OUT = OUT25 / 'ants'
DT = 0.2
EDGE_S = 10  # frames = 2 s at 5 fps (edge-artifact check)
DISC_MIN = 0.6

# ---------------------------------------------------------------------------------------------- patches
_mice_table0, _cache_paths0, _select0 = C.mice_table, C.cache_paths, C.select


def mice_table():
    F, SP = _mice_table0()
    assert len(F) == 864000
    Ra = pd.read_parquet(BEE25 / 'mice' / 'codes' / 'rows_all.parquet')
    assert (Ra.xrow.values == np.arange(len(F))).all() and (Ra.obs.values == F.obs.values).all() and (Ra.k.values == F.k.values).all()
    F['T'] = Ra['T'].values
    F['fi'] = Ra.fi.values
    F['xrow'] = np.arange(len(F))
    if not EDGE:
        return F, SP
    F = F[~Ra.edge.values].reset_index(drop=True)
    assert ((F.fi.values >= L) & (F.fi.values < F['T'].values - L)).all()
    return F, SP


def cache_paths(key):
    if EDGE:
        d = BEE25 / 'mice' / 'cache'
        return d / f'{key}_raw.npy', d / f'{key}_pon.npy'
    return _cache_paths0(key)


def select(st, rule):
    if rule not in ('dist', 'dist2048'):
        return _select0(st, rule)
    el = st['fire'] >= C.ELIG
    nn = int(np.nanargmax(np.where(el, st['auc_nose_nose'], np.nan)))
    cand = el & (np.arange(len(el)) != nn) & (st['disc'] >= DISC_MIN)
    if rule == 'dist2048':
        cand &= np.arange(len(el)) >= 512
    if not cand.any():
        raise SystemExit(f'rule {rule}: no candidate nose-tail neuron')
    return {'nose_nose': nn, 'nt': int(np.nanargmax(np.where(cand, st['auc_nt'], np.nan)))}


def directions(F, SP):
    return {'sg2non': [(F.sg.values, F.nonsg.values)]}, sorted(set(F.unit.values[F.sg.values]))


C.mice_table, C.cache_paths, C.select = mice_table, cache_paths, select
E.OUT22, E.RULES, E.directions = MICE_OUT, RULES, directions
AN.OUTA = ANTS_OUT
_readout0 = AN.readout


def ants_readout(key, F):
    if key.split('_s')[0] not in ARMS25:
        return _readout0(key, F)
    cp = BEE25 / 'ants' / 'cache' / f'{key}_pon.npy'
    return np.load(cp), np.load(cp.with_name(f'{key}_raw.npy')).astype(np.float32), np.load(cp)


AN.readout = ants_readout


def tag_file(dom, key):
    if dom == 'mice':
        return MICE_OUT / 'tags' / f'{key}.pkl'
    mine = ANTS_OUT / 'tags' / f'{key}.pkl'
    if mine.exists() or key.split('_s')[0] in ARMS25:
        return mine
    return REPO / 'results/vision/eci_v2_3/ants/tags' / f'{key}.pkl'


def obs_starts(obs):
    return np.r_[np.flatnonzero(np.r_[True, obs[1:] != obs[:-1]]), len(obs)].astype(np.int64)


# ---------------------------------------------------------------------------------------------- cache
def cmd_cache(a):
    t0 = time.time()
    key, dom = a.key, a.dom
    od = BEE25 / dom / 'cache'
    od.mkdir(parents=True, exist_ok=True)
    if dom == 'mice':
        assert EDGE
        F, _ = mice_table()
        if key.split('_s')[0] in ARMS25:
            raw = np.load(BEE25 / 'mice' / 'codes' / f'{key}_max.npy')
            Rk = pd.read_parquet(BEE25 / 'mice' / 'codes' / 'rows.parquet')
            assert len(raw) == len(F) and (Rk.xrow.values == F.xrow.values).all()
        else:
            pr0, _ = _cache_paths0(key)
            raw = np.load(pr0)[F.xrow.values]
        np.save(od / f'{key}_raw.npy', raw.astype(np.float16))
        if key.startswith('v1_'):
            log(f'cache mice {key}: raw only, {raw.shape} ({time.time() - t0:.0f}s)')
            return
        P, it = VF.pon_fast(raw.astype(np.float32), obs_starts(F.obs.values))
        np.save(od / f'{key}_pon.npy', P)
    else:
        F = AN.table()
        Rk = pd.read_parquet(BEE25 / 'ants' / 'codes' / 'rows.parquet')
        raw = np.load(BEE25 / 'ants' / 'codes' / f'{key}_max.npy').astype(np.float32)
        assert len(raw) == len(Rk)
        st = np.r_[0, np.flatnonzero((Rk.obs.values[1:] != Rk.obs.values[:-1]) | (Rk.series.values[1:] != Rk.series.values[:-1])) + 1, len(Rk)]
        P, it = VF.pon_fast(raw, st)
        rix = pd.Series(np.arange(len(Rk)), index=pd.MultiIndex.from_arrays([Rk.series.values, Rk.obs.values, Rk.k.values])).reindex(
            pd.MultiIndex.from_arrays([F.series.values, F.obs.astype(str).values, F.k.values])).values
        assert not np.isnan(rix).any()
        rix = rix.astype(np.int64)
        np.save(od / f'{key}_pon.npy', P[rix])
        np.save(od / f'{key}_raw.npy', raw[rix].astype(np.float16))
    log(f'cache {dom} {key}: {raw.shape}, HMM iterations median {np.median(it):.0f} max {it.max()} ({time.time() - t0:.0f}s)')


def cmd_tag(a):
    if a.dom == 'mice':
        E.cmd_tag(argparse.Namespace(model=a.key))
    else:
        AN.cmd_tag(argparse.Namespace(model=a.key))


# ---------------------------------------------------------------------------------------------- report helpers
def boot_level(v, rng):
    ix = rng.integers(0, len(v), (NB, len(v)))
    bb = np.nanmean(v[:, 1:][ix, np.arange(NB)[:, None]], 1)
    return {'mean': float(np.nanmean(v[:, 0])), 'ci': np.nanpercentile(bb, [2.5, 97.5]).tolist(), 'seeds': v[:, 0].tolist()}


def boot_diff(vx, vy, rng, paired=False):
    ax, ay = vx[:, 1:], vy[:, 1:]
    q = np.arange(NB)[:, None]
    ix = rng.integers(0, len(ax), (NB, len(ax)))
    if paired:
        assert len(ax) == len(ay)
        dd = np.nanmean(ax[ix, q] - ay[ix, q], 1)
    else:
        iy = rng.integers(0, len(ay), (NB, len(ay)))
        dd = np.nanmean(ax[ix, q], 1) - np.nanmean(ay[iy, q], 1)
    return {'diff': float(np.nanmean(vx[:, 0]) - np.nanmean(vy[:, 0])), 'ci': np.nanpercentile(dd, [2.5, 97.5]).tolist(), 'paired': paired}


def video_ms(tags, setkey, b, scale):
    acc = {}
    for tg in tags:
        for d in tg['sets'][setkey][:1]:
            V = d[b]['video']
            S, Tt, n = V['S'], V['T'], V['n']
            for nm, x in (('raw_bias', S - Tt), ('raw_mae', np.abs(S - Tt)), ('cal_bias', d[b]['c'] * S - Tt),
                          ('cal_mae', np.abs(d[b]['c'] * S - Tt)), ('const_mae', np.abs(d[b]['const_rate'] * n - Tt))):
                x = x * scale(n)
                acc.setdefault(nm, []).append((x.mean(), x.std(ddof=1) / np.sqrt(len(x))))
    return {nm: [float(np.mean([m for m, _ in v])), float(np.mean([s for _, s in v]))] for nm, v in acc.items()}


MK = ('auc', 'p100', 'F1', 'disc', 'r_soft', 'bias_raw_soft_s', 'mae_raw_soft_s', 'mae_cal_soft_s', 'bias_raw_soft_frac',
      'mae_raw_soft_frac', 'mae_cal_soft_frac')


def edge_frac(cell, fi, T):
    """fraction of the top-3/video frames (raw ranking, eligible videos) in the first / last 2 s of the store video."""
    e = cell['pos'] > 0
    rows = [r for ok, tl in zip(e, cell['top_raw']) if ok for (r, _, _, _) in tl]
    if not rows:
        return float('nan')
    rows = np.array(rows)
    return float(((fi[rows] < EDGE_S) | (fi[rows] >= T[rows] - EDGE_S)).mean())


def mice_prep(key, F, tag):
    ev = F.nonsg.values
    rows = np.flatnonzero(ev)
    picks = {r: tag['sets'][f'sg2non|{r}'][0]['picks'] for r in RULES}
    js = sorted({j for p in picks.values() for j in p.values()})
    pr, pp = C.cache_paths(key)
    R = T3.cols_chunked(pr, js)
    X = R if key.startswith('v1_') else T3.cols_chunked(pp, js)
    col = {j: i for i, j in enumerate(js)}
    obs, unit = F.obs.values[rows], F.unit.values[rows]
    fi, Tt = F.fi.values[rows], F['T'].values[rows]
    cells = {}
    for r in RULES:
        for b in C.BEH:
            j = picks[r][b]
            c = T3.top3_videos(obs, F[f'y_{b}'].values[rows], F[f'v_{b}'].values[rows], R[rows, col[j]], X[rows, col[j]], unit)
            c['neuron'] = j
            c['edge_frac'] = edge_frac(c, fi, Tt)
            cells[f'{r}|{b}'] = c
    return cells


def cmd_prep(a):
    F, _ = C.mice_table()
    wp = MICE_OUT / 'work' / f'{a.key}.pkl'
    wp.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(mice_prep(a.key, F, pickle.load(open(tag_file('mice', a.key), 'rb'))), open(wp, 'wb'))
    log(f'prep {a.key} done')


# ---------------------------------------------------------------------------------------------- mice report
def cmd_report_mice(a):
    t0 = time.time()
    F, _ = C.mice_table()
    units = sorted(set(F.unit.values))
    uidx = {u: i for i, u in enumerate(units)}
    non = sorted(set(F.unit.values[F.nonsg.values]))
    rng = np.random.default_rng(20261104)  # = v22_eval.cmd_report sg2non pool weights
    W = np.ones((NB + 1, len(units)))
    ix_ = [units.index(u) for u in non]
    W[1:, ix_] = np.stack([np.bincount(rng.integers(0, len(non), len(non)), minlength=len(non)) for _ in range(NB)])
    wb = lambda ue: W[:, [uidx[u] for u in ue]]  # noqa: E731
    wv = lambda vids, us: W[:, [uidx[u] for u in us]]  # noqa: E731
    models = {k: v for k, v in MODELS.items() if all(tag_file('mice', m).exists() for m in v)}
    tags, prep = {}, {}
    for arm, ks in models.items():
        for k in ks:
            tags[k] = pickle.load(open(tag_file('mice', k), 'rb'))
            wp = MICE_OUT / 'work' / f'{k}.pkl'
            prep[k] = pickle.load(open(wp, 'rb')) if wp.exists() else mice_prep(k, F, tags[k])
    log(f'mice: {len(tags)} tags, prep done ({time.time() - t0:.0f}s)')
    per = {}
    res = {'edge_rule': EDGE, 'models': models, 'levels': {}, 'contrasts': {}, 'picks': {}, 'distinct': {}, 'video': {},
           'counts': {}, 'edge_frac': {}, 'dist_binds': {}, 'sel_disc': {}}
    for arm, ks in models.items():
        for r in RULES:
            M = [E.model_metrics(tags[k], f'sg2non|{r}', wb) for k in ks]
            d = {mk: np.stack([m[mk] for m in M]) for mk in M[0]}
            for b in C.BEH:
                new = [T3.top3_rep(prep[k][f'{r}|{b}'], wv) for k in ks]
                for nm in ('top3', 'chance'):
                    d[(b, nm)] = np.stack([x[nm] for x in new])
            per[f'{arm}|{r}'] = d
            pk = [tags[k]['sets'][f'sg2non|{r}'][0]['picks'] for k in ks]
            res['picks'][f'{arm}|{r}'] = dict(zip(ks, pk))
            res['distinct'][f'{arm}|{r}'] = [int(sum(p['nose_nose'] != p['nt'] for p in pk)), len(pk)]
            res['video'][f'{arm}|{r}'] = {b: video_ms([tags[k] for k in ks], f'sg2non|{r}', b, lambda n: DT) for b in C.BEH}
            res['edge_frac'][f'{arm}|{r}'] = {b: [prep[k][f'{r}|{b}']['edge_frac'] for k in ks] for b in C.BEH}
        pa = [tags[k]['sets']['sg2non|auc'][0]['picks']['nt'] for k in ks]
        pd_ = [tags[k]['sets']['sg2non|dist'][0]['picks']['nt'] for k in ks]
        res['dist_binds'][arm] = f'{sum(x != y for x, y in zip(pa, pd_))}/{len(ks)}'
    rng2 = np.random.default_rng(20261105)
    for an, d in per.items():
        res['levels'][an] = {f'{b}|{m}': boot_level(v, rng2) for (b, m), v in d.items() if m in MK + ('top3', 'chance')}
    rng3 = np.random.default_rng(20261107)
    pairs = []
    for x in ARMS25 + ('v1s0',):
        for r in RULES:
            pairs.append((f'{x}|{r}', 'V|auc', False))
    for r in RULES:
        pairs.append((f'V|{r}', 'V|auc', True)) if r != 'auc' else None
        for x in ('W1', 'Wh', 'C1', 'M2'):
            pairs.append((f'{x}|{r}', f'N0|{r}', True))
        pairs.append((f'M2|{r}', f'C1|{r}', True))
        pairs += [(f'C1|{r}', f'W1|{r}', True), (f'Wh|{r}', f'W1|{r}', True)]
        for x in ARMS25 + ('V',):
            pairs.append((f'{x}|{r}', 'v1s0|auc', False))
    for x, y, paired in pairs:
        if x in per and y in per:
            res['contrasts'][f'{x} - {y}'] = {f'{b}|{m}': boot_diff(per[x][(b, m)], per[y][(b, m)], rng3, paired)
                                              for (b, m) in per[x] if m in MK + ('top3',)}
    # ground truth on the scored frames + what the edge rule removed
    F0 = _mice_table0()[0]
    Ra = pd.read_parquet(BEE25 / 'mice' / 'codes' / 'rows_all.parquet')
    for b in C.BEH:
        c0 = prep['V_s1001'][f'auc|{b}']
        e = c0['pos'] > 0
        G = F[F.nonsg.values]
        y = G[f'y_{b}'].values
        pv = G.groupby('obs')[f'y_{b}'].mean()
        res['counts'][b] = {'held_out_frames': int(len(G)), 'held_out_videos': int(G.obs.nunique()), 'positive_frames': int(y.sum()),
                            'prevalence': float(y.mean()), 'per_video_prevalence_mean_se': [float(pv.mean()), float(pv.std(ddof=1) / np.sqrt(len(pv)))],
                            'top3_eligible_videos': int(e.sum()), 'top3_frames_scored_per_seed': int(3 * e.sum()),
                            'edge_removed_frames_all': int(Ra.edge.sum()), 'edge_removed_positives_all': int(F0[f'y_{b}'].values[Ra.edge.values].sum()),
                            'edge_removed_frames_held_out': int((Ra.edge.values & F0.nonsg.values).sum()),
                            'edge_removed_positives_held_out': int(F0[f'y_{b}'].values[Ra.edge.values & F0.nonsg.values].sum()),
                            'scored_frames_all': int(len(F))}
    MICE_OUT.mkdir(parents=True, exist_ok=True)
    (MICE_OUT / 'report.json').write_text(json.dumps(res, indent=1, default=float))
    log(f'mice report done ({time.time() - t0:.0f}s)')


# ---------------------------------------------------------------------------------------------- ants report
def cmd_report_ants(a):
    t0 = time.time()
    F = AN.table()
    Ra = pd.read_parquet(BEE25 / 'ants' / 'codes' / 'rows_all.parquet')
    assert not Ra.edge.any(), 'ants: an edge frame in the scored spans'
    vids = F.groupby('obs', sort=True).agg(stratum=('stratum', 'first'), half=('half', 'first'))
    sp = json.loads((REPO / 'results/vision/eci_v2_3/ants/split.json').read_text())
    assert sorted(vids.index[vids.half == 'A']) == sorted(sp['A'])
    models = {k: v for k, v in MODELS_ANTS.items() if all(tag_file('ants', m).exists() for m in v)}
    tags = {k: pickle.load(open(tag_file('ants', k), 'rb')) for ks in models.values() for k in ks}
    rng = np.random.default_rng(20261106)
    res = {'models': models, 'dirs': {}, 'tag_files': {k: str(tag_file('ants', k)) for k in tags}}
    b = 'groom_any'
    Tof = {(s_, o): t for s_, o, t in zip(Ra.series.values, Ra.obs.values, Ra['T'].values)}
    fiof = pd.Series(Ra.fi.values, index=pd.MultiIndex.from_arrays([Ra.series.values, Ra.obs.values, Ra.k.values]))
    fi_all = fiof.reindex(pd.MultiIndex.from_arrays([F.series.values, F.obs.astype(str).values, F.k.values])).values.astype(np.int64)
    T_all = np.array([Tof[(s_, str(o))] for s_, o in zip(F.series.values, F.obs.values)])
    for dn, (s_, e_) in AN.DIRS.items():
        ev = vids.index[vids.half == e_].tolist()
        strata = vids.loc[ev, 'stratum'].values
        W = np.ones((NB + 1, len(ev)))
        for s in np.unique(strata):
            ix = np.flatnonzero(strata == s)
            W[1:, ix] = np.stack([np.bincount(rng.integers(0, len(ix), len(ix)), minlength=len(ix)) for _ in range(NB)])
        vix = {v: i for i, v in enumerate(ev)}
        wv = lambda vs, us: W[:, [vix[v] for v in vs]]  # noqa: E731
        rows = np.flatnonzero(F.half.values == e_)
        per, R = {}, {'levels': {}, 'contrasts': {}, 'picks': {}, 'video': {}, 'edge_frac': {}}
        for arm, ks in models.items():
            M, ef = [], []
            for k in ks:
                d = tags[k]['sets'][dn][0]
                assert d['units_ev'] == ev
                m = {(b, m_): v for m_, v in E.metrics(d[b], W).items()}
                X, Rr, _ = AN.readout(k, F)
                j = d['picks'][b]
                c = T3.top3_videos(F.obs.values[rows], F[f'y_{b}'].values[rows], F[f'v_{b}'].values[rows], Rr[rows, j], X[rows, j], F.obs.values[rows])
                t3 = T3.top3_rep(c, wv)
                m[(b, 'top3')], m[(b, 'chance')] = t3['top3'], t3['chance']
                ef.append(edge_frac(c, fi_all[rows], T_all[rows]))
                M.append(m)
                R.setdefault('top3_counts', {'eligible': int((c['pos'] > 0).sum()), 'held_out_videos': len(c['pos'])})
            per[arm] = {mk: np.stack([m[mk] for m in M]) for mk in M[0]}
            R['picks'][arm] = {k: tags[k]['sets'][dn][0]['picks'][b] for k in ks}
            R['video'][arm] = AN.video_stats([tags[k] for k in ks], dn, b)
            R['edge_frac'][arm] = ef
        rng2 = np.random.default_rng(20261108)
        for arm, d in per.items():
            R['levels'][arm] = {f'{bb}|{m_}': boot_level(v, rng2) for (bb, m_), v in d.items() if m_ in MK + ('top3', 'chance')}
        pairs = [(x, 'V', False) for x in ARMS25 + ('v1s0', 'v1')] + [(x, 'N0', True) for x in ('W1', 'Wh', 'C1', 'M2')] + \
                [('C1', 'W1', True), ('Wh', 'W1', True), ('M2', 'C1', True)] + [(x, 'v1s0', False) for x in ARMS25 + ('V',)] + [(x, 'v1', False) for x in ARMS25]
        for x, y, paired in pairs:
            if x in per and y in per:
                R['contrasts'][f'{x} - {y}'] = {f'{bb}|{m_}': boot_diff(per[x][(bb, m_)], per[y][(bb, m_)], rng2, paired)
                                                for (bb, m_) in per[x] if m_ in MK + ('top3',)}
        Fe = F.iloc[rows]
        R['prevalence'] = float(Fe[f'y_{b}'].mean())
        R['held_out'] = {'frames': int(len(Fe)), 'videos': int(Fe.obs.nunique()), 'positive_frames': int(Fe[f'y_{b}'].sum())}
        res['dirs'][dn] = R
    ANTS_OUT.mkdir(parents=True, exist_ok=True)
    (ANTS_OUT / 'report.json').write_text(json.dumps(res, indent=1, default=float))
    log(f'ants report done ({time.time() - t0:.0f}s)')


def cmd_report(a):
    (cmd_report_mice if a.dom == 'mice' else cmd_report_ants)(a)


# ---------------------------------------------------------------------------------------------- forward check
def cmd_fwdcheck(a):
    import torch
    from src.eci.v2 import sae as SA
    dom, key = a.dom, a.key
    arm = key.split('_s')[0]
    alpha = json.loads((OUT25 / f'{dom}_count.json').read_text())['alpha']
    sae, norm = SA.load(BEE25 / dom / 'sae' / key / 'sae_theta.pt', torch.device('cpu'))
    Rk = pd.read_parquet(BEE25 / dom / 'codes' / 'rows.parquet')
    Z = np.load(BEE25 / dom / 'codes' / f'{key}_max.npy', mmap_mode='r')
    st = TR.Store(dom)
    rng = np.random.default_rng(11)
    mx, n = 0.0, 0
    for r in rng.choice(len(Rk), 200, replace=False):
        ser, obs, fi, nf = Rk.series.values[r], Rk.obs.values[r], int(Rk.fi.values[r]), int(Rk['T'].values[r])
        x, d, _, _, _ = st.load_xd((ser, obs, fi, fi + 1, nf), dev=torch.device('cpu'))
        with torch.no_grad():
            ref = sae.encode(norm(TR.feats_t(x, d, arm, alpha, TR.alpha1(dom) if arm == 'M2' else None).float()), mode='threshold').max(0).values.numpy() if len(x) else np.zeros(2048, np.float32)
        mx = max(mx, float(np.abs(ref.astype(np.float16).astype(np.float32) - np.asarray(Z[r], np.float32)).max()))
        n += 1
    res = {'key': key, 'frames': n, 'max_abs_diff_fp16': mx}
    (OUT25 / 'codes').mkdir(parents=True, exist_ok=True)
    (OUT25 / 'codes' / f'fwdcheck_{dom}_{key}.json').write_text(json.dumps(res))
    log(f'fwdcheck {dom} {res}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=('cache', 'tag', 'prep', 'report', 'fwdcheck'))
    p.add_argument('--dom', default='mice', choices=('mice', 'ants'))
    p.add_argument('--key', default='')
    a = p.parse_args()
    {'cache': cmd_cache, 'tag': cmd_tag, 'prep': cmd_prep, 'report': cmd_report, 'fwdcheck': cmd_fwdcheck}[a.step](a)
