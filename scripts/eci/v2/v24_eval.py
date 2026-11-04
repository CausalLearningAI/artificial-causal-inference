"""
ECI v2.4 scoring (pre-registration scripts/eci/v2/v24_prereg.md). Reuses the v2.3 code paths unchanged, with the output
locations redirected for the new P* models only (module attributes patched here; the shared scripts are not edited):
  mice: frame table / selection / held-out quantities = v22_core + v22_eval.cmd_tag with V22_FRAMES=ext (864,000 frames);
        P(on) = v22_full.pon_fast, one sequence per video; precision@top3/video = v24_top3.top3_videos / top3_rep.
  ants: v23_ants.table / cmd_tag (v3 + v4, halves A / B), the same HMM readout from the v24 codes.
Baselines (V s1001-1006, Vall s2001-2003, v1 s0 (mice ext), v1 s0-s10 (ants)) use their stored v2.3 caches; their tags
are read from v2.3 (V is ALSO re-tagged here: scorer identity).
Steps
  cache --dom D --key K   (CPU) raw codes + HMM P(on) of a P* model -> BEE24/<dom>/cache/
  tag --dom D --key K     (CPU) -> OUT24/<dom>/tags/K.pkl
  report --dom D          (CPU) -> OUT24/<dom>/report.json, OUT24/<dom>/tables.md
  fwdcheck --dom D --key K   (CPU) direct SAE forward pass on 200 frames == assembled codes
  sheet --dom D --key K   (CPU) top-12 held-out frames of the selected neurons (rule auc) -> OUT24/sheets/
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
import v24_top3 as T3  # noqa: E402  (precision@top3/video functions; that file and its folder are not modified)
import v24_pool as PL  # noqa: E402

assert C.FRAMES == 'ext'
REPO, T, log = C.REPO, C.T, C.log
BEE24, OUT24 = PL.BEE24, PL.OUT24
NB = 1000
RULES = ('auc', 'bal', 'hyb')
ARMS24 = ('P2W', 'P4W', 'P2N', 'P2C')
SEEDS = (2401, 2402, 2403)
MODELS = {'V': [f'V_s{s}' for s in range(1001, 1007)], 'Vall': [f'Vall_s{s}' for s in (2001, 2002, 2003)], 'v1s0': ['v1_s0'],
          **{a: [f'{a}_s{s}' for s in SEEDS] for a in ARMS24}}
MODELS_ANTS = {**{k: v for k, v in MODELS.items() if k != 'v1s0'}, 'v1': [f'v1_s{s}' for s in range(11)], 'v1s0': ['v1_s0']}
MICE_OUT, ANTS_OUT = OUT24 / 'mice', OUT24 / 'ants'
DT = 0.2

_cache_paths0 = C.cache_paths


def cache_paths(key):
    if key[0] == 'P':
        d = BEE24 / 'mice' / 'cache'
        return d / f'{key}_raw.npy', d / f'{key}_pon.npy'
    return _cache_paths0(key)


C.cache_paths = cache_paths
E.OUT22 = MICE_OUT  # tags of this run
AN.OUTA = ANTS_OUT
_readout0 = AN.readout


def ants_readout(key, F):
    if key[0] != 'P':
        return _readout0(key, F)
    cp = BEE24 / 'ants' / 'cache' / f'{key}_pon.npy'
    cr = cp.with_name(f'{key}_raw.npy')
    return np.load(cp), np.load(cr).astype(np.float32), np.load(cp)


AN.readout = ants_readout


def tag_file(dom, key):
    mine = (MICE_OUT if dom == 'mice' else ANTS_OUT) / 'tags' / f'{key}.pkl'
    if mine.exists() or key[0] == 'P':
        return mine
    return (C.OUT22 if dom == 'mice' else REPO / 'results/vision/eci_v2_3/ants') / 'tags' / f'{key}.pkl'


# ---------------------------------------------------------------------------------------------- cache
def cmd_cache(a):
    t0 = time.time()
    raw = np.load(BEE24 / a.dom / 'codes' / f'{a.key}_max.npy')
    if a.dom == 'mice':
        F = pd.read_parquet(C.BEEF / 'frames.parquet', columns=['obs'])
        assert len(F) == len(raw) == 864000
        o = F.obs.values
        st = np.r_[np.flatnonzero(np.r_[True, o[1:] != o[:-1]]), len(o)].astype(np.int64)
        P, it = VF.pon_fast(raw, st)
        pr, pp = cache_paths(a.key)
        pr.parent.mkdir(parents=True, exist_ok=True)
        np.save(pr, raw)
        np.save(pp, P)
    else:  # = v23_ants.readout: HMM over every 5 fps eval span, then the v3 + v4 frame rows
        F = AN.table()
        od = T.BEE / 'ants' / 'codes'
        Rw = pd.read_parquet(od / 'eval_rows.parquet')
        assert len(Rw) == len(raw)
        Ef = T.frame_sets('ants')['eval']
        rix = pd.Series(np.arange(len(Rw)), index=pd.MultiIndex.from_frame(Rw)).reindex(pd.MultiIndex.from_frame(Ef)).values
        assert not np.isnan(rix).any()
        rix = rix.astype(np.int64)[F.frow.values]
        st = np.r_[0, np.flatnonzero((Rw.obs.values[1:] != Rw.obs.values[:-1]) | (Rw.series.values[1:] != Rw.series.values[:-1])) + 1, len(Rw)]
        P, it = VF.pon_fast(raw.astype(np.float32), st)
        cp = BEE24 / 'ants' / 'cache' / f'{a.key}_pon.npy'
        cp.parent.mkdir(parents=True, exist_ok=True)
        np.save(cp, P[rix])
        np.save(cp.with_name(f'{a.key}_raw.npy'), raw[rix].astype(np.float16))
    log(f'cache {a.dom} {a.key}: {raw.shape}, HMM iterations median {np.median(it):.0f} max {it.max()} ({time.time() - t0:.0f}s)')


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


def boot_diff(vx, vy, rng):
    ax, ay = vx[:, 1:], vy[:, 1:]
    ix = rng.integers(0, len(ax), (NB, len(ax)))
    iy = rng.integers(0, len(ay), (NB, len(ay)))
    q = np.arange(NB)[:, None]
    dd = np.nanmean(ax[ix, q], 1) - np.nanmean(ay[iy, q], 1)
    return {'diff': float(np.nanmean(vx[:, 0]) - np.nanmean(vy[:, 0])), 'ci': np.nanpercentile(dd, [2.5, 97.5]).tolist()}


def video_ms(tags, setkey, b, scale):
    """per-video time: mean over held-out videos +- SE, mean over seeds; raw / calibrated bias and MAE, const MAE."""
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


MK = ('auc', 'p100', 'F1', 'disc', 'r_soft', 'bias_raw_soft_s', 'mae_raw_soft_s', 'mae_cal_soft_s', 'bias_raw_soft_frac', 'mae_raw_soft_frac')


# ---------------------------------------------------------------------------------------------- mice report
def mice_prep(key, F, tag):
    """= v24_top3.cmd_mice_prep (ext branch), returned instead of written."""
    ev = F.nonsg.values
    rows = np.flatnonzero(ev)
    picks = {r: tag['sets'][f'sg2non|{r}'][0]['picks'] for r in RULES}
    js = sorted({j for p in picks.values() for j in p.values()})
    pr, pp = C.cache_paths(key)
    R = T3.cols_chunked(pr, js)
    X = R if key.startswith('v1_') else T3.cols_chunked(pp, js)
    col = {j: i for i, j in enumerate(js)}
    obs, unit = F.obs.values[rows], F.unit.values[rows]
    cells = {}
    for r in RULES:
        for b in C.BEH:
            j = picks[r][b]
            c = T3.top3_videos(obs, F[f'y_{b}'].values[rows], F[f'v_{b}'].values[rows], R[rows, col[j]], X[rows, col[j]], unit)
            c['neuron'] = j
            cells[f'{r}|{b}'] = c
    return cells


def cmd_report_mice(a):
    t0 = time.time()
    F, _ = C.mice_table()
    assert len(F) == 864000
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
    per, res = {}, {'models': models, 'tag_files': {k: str(tag_file('mice', k)) for k in tags}, 'levels': {}, 'contrasts': {},
                    'picks': {}, 'distinct': {}, 'video': {}, 'counts': {}}
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
    rng2 = np.random.default_rng(20261105)
    for an, d in per.items():
        res['levels'][an] = {f'{b}|{m}': boot_level(v, rng2) for (b, m), v in d.items() if m in MK + ('top3', 'chance')}
    rng3 = np.random.default_rng(20261107)
    pairs = [(f'{x}|{r}', f'V|{r}') for x in ('Vall',) + ARMS24 for r in RULES] + [(f'{x}|auc', 'V|auc') for x in ARMS24 for _ in ()] + \
            [('P2W|auc', 'P2N|auc'), ('P2W|bal', 'P2N|bal'), ('P2C|auc', 'P2W|auc'), ('P4W|auc', 'P2W|auc'), ('P2C|auc', 'P2N|auc')] + \
            [(f'{x}|auc', 'Vall|auc') for x in ARMS24] + [(f'{x}|auc', 'v1s0|auc') for x in ARMS24 + ('V',)]
    for x, y in pairs:
        if x in per and y in per:
            res['contrasts'][f'{x} - {y}'] = {f'{b}|{m}': boot_diff(per[x][(b, m)], per[y][(b, m)], rng3)
                                              for (b, m) in per[x] if m in MK + ('top3',)}
    for b in C.BEH:
        c0 = prep['V_s1001'][f'auc|{b}']
        e = c0['pos'] > 0
        G = F[F.nonsg.values]
        y = G[f'y_{b}'].values
        pv = G.groupby('obs')[f'y_{b}'].mean()
        res['counts'][b] = {'held_out_frames': int(len(G)), 'held_out_videos': int(G.obs.nunique()), 'positive_frames': int(y.sum()),
                            'prevalence': float(y.mean()), 'per_video_prevalence_mean_se': [float(pv.mean()), float(pv.std(ddof=1) / np.sqrt(len(pv)))],
                            'top3_eligible_videos': int(e.sum()), 'top3_frames_scored_per_seed': int(3 * e.sum())}
    # B1-B3 per arm (rule auc, vs V|auc)
    res['criteria'] = {}
    for arm in ARMS24:
        c = res['contrasts'].get(f'{arm}|auc - V|auc')
        if not c:
            continue
        dd = res['distinct'][f'{arm}|auc']
        res['criteria'][arm] = {
            'B1 nose-nose AUROC lower CI': c['nose_nose|auc']['ci'][0], 'B1': c['nose_nose|auc']['ci'][0] >= -0.02,
            'B2 nose-tail top3 diff': c['nt|top3']['diff'], 'B2 nose-tail top3 lower CI': c['nt|top3']['ci'][0],
            'B2 nose-tail AUROC lower CI': c['nt|auc']['ci'][0],
            'B2': c['nt|top3']['diff'] > 0 and c['nt|top3']['ci'][0] > 0 and c['nt|auc']['ci'][0] >= -0.02,
            'B3 distinct seeds': f'{dd[0]}/{dd[1]}', 'B3': dd[0] == dd[1] == 3}
    MICE_OUT.mkdir(parents=True, exist_ok=True)
    (MICE_OUT / 'report.json').write_text(json.dumps(res, indent=1, default=float))
    log(f'mice report done ({time.time() - t0:.0f}s): {json.dumps(res["criteria"], default=str)}')


# ---------------------------------------------------------------------------------------------- ants report
def cmd_report_ants(a):
    t0 = time.time()
    F = AN.table()
    vids = F.groupby('obs', sort=True).agg(stratum=('stratum', 'first'), half=('half', 'first'))
    sp = json.loads((REPO / 'results/vision/eci_v2_3/ants/split.json').read_text())
    assert sorted(vids.index[vids.half == 'A']) == sorted(sp['A']) if 'A' in sp else True
    models = {k: v for k, v in MODELS_ANTS.items() if all(tag_file('ants', m).exists() for m in v)}
    tags = {k: pickle.load(open(tag_file('ants', k), 'rb')) for ks in models.values() for k in ks}
    rng = np.random.default_rng(20261106)
    res = {'models': models, 'dirs': {}}
    b = 'groom_any'
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
        per, R = {}, {'levels': {}, 'contrasts': {}, 'picks': {}, 'video': {}}
        for arm, ks in models.items():
            M = []
            for k in ks:
                d = tags[k]['sets'][dn][0]
                assert d['units_ev'] == ev
                m = {(b, m_): v for m_, v in E.metrics(d[b], W).items()}
                X, Rr, _ = AN.readout(k, F)
                j = d['picks'][b]
                c = T3.top3_videos(F.obs.values[rows], F[f'y_{b}'].values[rows], F[f'v_{b}'].values[rows], Rr[rows, j], X[rows, j], F.obs.values[rows])
                t3 = T3.top3_rep(c, wv)
                m[(b, 'top3')], m[(b, 'chance')] = t3['top3'], t3['chance']
                M.append(m)
                R.setdefault('top3_counts', {'eligible': int((c['pos'] > 0).sum()), 'held_out_videos': len(c['pos'])})
            per[arm] = {mk: np.stack([m[mk] for m in M]) for mk in M[0]}
            R['picks'][arm] = {k: tags[k]['sets'][dn][0]['picks'][b] for k in ks}
            R['video'][arm] = AN.video_stats([tags[k] for k in ks], dn, b)
        rng2 = np.random.default_rng(20261108)
        for arm, d in per.items():
            R['levels'][arm] = {f'{bb}|{m_}': boot_level(v, rng2) for (bb, m_), v in d.items() if m_ in MK + ('top3', 'chance')}
        for x, y in [(x, 'V') for x in ('Vall',) + ARMS24] + [('P2W', 'P2N'), ('P2C', 'P2W'), ('P4W', 'P2W'), ('P2C', 'P2N')] + \
                    [(x, 'Vall') for x in ARMS24] + [(x, 'v1') for x in ARMS24 + ('V',)]:
            if x in per and y in per:
                R['contrasts'][f'{x} - {y}'] = {f'{bb}|{m_}': boot_diff(per[x][(bb, m_)], per[y][(bb, m_)], rng2)
                                                for (bb, m_) in per[x] if m_ in MK + ('top3',)}
        Fe = F.iloc[rows]
        R['prevalence'] = float(Fe[f'y_{b}'].mean())
        res['dirs'][dn] = R
    res['B4'] = {arm: {'AUROC lower CI': res['dirs']['A2B']['contrasts'][f'{arm} - V'][f'{b}|auc']['ci'][0],
                       'B4': res['dirs']['A2B']['contrasts'][f'{arm} - V'][f'{b}|auc']['ci'][0] >= -0.02}
                 for arm in ARMS24 if f'{arm} - V' in res['dirs']['A2B']['contrasts']}
    ANTS_OUT.mkdir(parents=True, exist_ok=True)
    (ANTS_OUT / 'report.json').write_text(json.dumps(res, indent=1, default=float))
    log(f'ants report done ({time.time() - t0:.0f}s): {res["B4"]}')


def cmd_prep(a):
    F, _ = C.mice_table()
    wp = MICE_OUT / 'work' / f'{a.key}.pkl'
    wp.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(mice_prep(a.key, F, pickle.load(open(tag_file('mice', a.key), 'rb'))), open(wp, 'wb'))
    log(f'prep {a.key} done')


def cmd_report(a):
    (cmd_report_mice if a.dom == 'mice' else cmd_report_ants)(a)


# ---------------------------------------------------------------------------------------------- forward check
def frame_cells(dom, P, vi, k):
    sh = PL.Shards(dom, P)
    t, r0, r1 = sh.vrow[vi]
    ix = sh.index(t)
    m = np.flatnonzero(ix['k'][r0:r1] == k) + r0
    if not len(m):
        return None, None, None
    return sh.read(t, 'x', int(m[0]), int(m[-1]) + 1), sh.read(t, 'd', int(m[0]), int(m[-1]) + 1), ix['pos'][m].astype(int)


def load_sae(dom, key):
    import torch
    from src.eci.v2 import sae as SA
    return SA.load(BEE24 / dom / 'sae' / key / 'sae_theta.pt', torch.device('cpu'))


def cmd_fwdcheck(a):
    import torch
    import v24_train as TR
    dom, key = a.dom, a.key
    P, arm, _ = TR.parse_key(key)
    alpha = TR.alpha_of(dom, P)
    sae, norm = load_sae(dom, key)
    R = TR.eval_rows(dom)
    Z = np.load(BEE24 / dom / 'codes' / f'{key}_max.npy', mmap_mode='r')
    rng = np.random.default_rng(11)
    mx, n = 0.0, 0
    for r in rng.choice(len(R), 200, replace=False):
        x, d, _ = frame_cells(dom, P, int(R.vid.values[r]), int(R.k.values[r]))
        if x is None:
            ref = np.zeros(Z.shape[1], np.float32)
        else:
            with torch.no_grad():
                ref = sae.encode(norm(TR.feats_t(torch.from_numpy(x), torch.from_numpy(d), arm, alpha).float()), mode='threshold').max(0).values.numpy()
        mx = max(mx, float(np.abs(ref.astype(np.float16).astype(np.float32) - np.asarray(Z[r], np.float32)).max()))
        n += 1
    res = {'key': key, 'frames': n, 'max_abs_diff_fp16': mx}
    (OUT24 / 'codes').mkdir(parents=True, exist_ok=True)
    (OUT24 / 'codes' / f'fwdcheck_{dom}_{key}.json').write_text(json.dumps(res))
    log(f'fwdcheck {res}')


# ---------------------------------------------------------------------------------------------- sheets
def cmd_sheet(a):
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import v21_extended_sheet as XS
    import v24_train as TR
    dom, key = a.dom, a.key
    P, arm, _ = TR.parse_key(key)
    alpha = TR.alpha_of(dom, P)
    sae, norm = load_sae(dom, key)
    vids = PL.videos(dom)
    vix = {(s, o): i for i, (s, o, _) in enumerate(vids)}
    plans = XS.plans_of(dom)
    tag = pickle.load(open(tag_file(dom, key), 'rb'))
    if dom == 'mice':
        from src.eci.foreground import ODOR_ROT90
        F, _ = C.mice_table()
        X, R, _ = C.load(key)
        oc = pd.read_csv(REPO / 'dataset/mice/v1/eci/odor_corner.csv').set_index('observation_id')['odor_corner']
        rot = {o: int(ODOR_ROT90[oc[o]]) for o in F.obs.unique()}
        d = tag['sets']['sg2non|auc'][0]
        ev = np.flatnonzero(F.nonsg.values)
        behs, ser_of, gap = list(C.BEH), (lambda o: 'v1'), 25
    else:
        F = AN.table()
        X, R, _ = AN.readout(key, F)
        rot = {}
        d = tag['sets']['A2B'][0]
        ev = np.flatnonzero(F.half.values == 'B')
        behs, gap = ['groom_any'], 10
        sser = dict(zip(F.obs.values, F.series.values))
        ser_of = lambda o: sser[o]  # noqa: E731
    name = {**C.NAME}
    fig, axs = plt.subplots(2 * len(behs), 6, figsize=(12, 2.45 * 2 * len(behs)), gridspec_kw={'hspace': 0.6, 'wspace': 0.06})
    for ax in axs.ravel():
        ax.axis('off')
    out = {'model': key, 'rule': 'auc' if dom == 'mice' else 'A2B best AUROC', 'behaviours': {}}
    for bi, b in enumerate(behs):
        j = int(d['picks'][b])
        y = F[f'y_{b}'].values[ev]
        got = XS.pick_frames(X[ev, j].astype(np.float64), R[ev, j].astype(np.float64), y, F.obs.values[ev], F.k.values[ev], gap=gap)
        fr = []
        for n_, i in enumerate(got):
            r = int(ev[i])
            obs, k = str(F.obs.values[r]), int(F.k.values[r])
            ser = ser_of(obs)
            vi = vix[('v4' if ser == 'v4_infested' else ser, obs)]
            x, dd, pos = frame_cells(dom, P, vi, k)
            G = int(np.load(PL.STORE / dom / PL.VER / 'videos' / vids[vi][0] / obs / 'index.npz')['G'])
            S = int(np.load(PL.STORE / dom / PL.VER / 'videos' / vids[vi][0] / obs / 'index.npz')['S'])
            g = (G + P - 1) // P
            m = np.zeros(g * g, np.float32)
            with torch.no_grad():
                m[pos] = sae.encode(norm(TR.feats_t(torch.from_numpy(x), torch.from_numpy(dd), arm, alpha).float()), mode='threshold')[:, j].numpy()
            m = m.reshape(g, g)
            img = XS.decode_v2(dom, ser, obs, k, S, plans, rot)
            lab = bool(y[i])
            fr.append({'obs': obs, 'k': k, 'readout': float(X[r, j]), 'code': float(R[r, j]), 'y': lab, 'map_max': float(m.max())})
            ax = axs[2 * bi + n_ // 6, n_ % 6]
            H, Wd = img.shape[:2]
            q = int(np.argmax(m))
            pr, pc = q // g, q % g
            ps = H / g
            hw = (0.36 if dom == 'mice' else 0.42) * H / 2
            cy, cx = (pr + 0.5) * ps, (pc + 0.5) * ps
            y0, x0 = int(np.clip(cy - hw, 0, H - 2 * hw)), int(np.clip(cx - hw, 0, Wd - 2 * hw))
            ax.imshow(img, extent=(0, Wd, H, 0))
            ax.imshow(np.ma.masked_less_equal(m, 0), extent=(0, Wd, H, 0), cmap='inferno', alpha=0.45, interpolation='nearest',
                      vmin=0, vmax=max(float(m.max()), 1e-6))
            ax.add_patch(Rectangle((pc * ps, pr * ps), ps, ps, fill=False, ec='cyan', lw=1.4))
            ax.set_xlim(x0, x0 + 2 * hw)
            ax.set_ylim(y0 + 2 * hw, y0)
            ax.set_title(f'{"yes" if lab else "NO"}\n{obs} k{k}', fontsize=6.5, color='black' if lab else 'red')
        npos = sum(f['y'] for f in fr)
        axs[2 * bi, 0].text(-0.02, 1.32, f'{name.get(b, b)}: neuron #{j}, true label in {npos}/{len(fr)} top frames',
                            transform=axs[2 * bi, 0].transAxes, fontsize=9, weight='bold')
        out['behaviours'][b] = {'neuron': j, 'labelled': npos, 'shown': len(fr), 'frames': fr}
        log(f'sheet {dom} {key} {b}: neuron {j}, labelled {npos}/{len(fr)}')
    held = 'held-out non-SG pools, rule auc' if dom == 'mice' else 'held-out half B, selected on A'
    fig.suptitle(f'{dom} {key}: top frames of the selected neurons ({held}); ranked by P(on), ties by raw code, <= 2 per video\n'
                 f'cyan = max pooled cell ({P}x{P}), heat = per-cell code; caption = true label (red NO = absent)', fontsize=9.5, y=0.995)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88 if dom == 'mice' else 0.8, bottom=0.01)
    od = OUT24 / 'sheets'
    od.mkdir(parents=True, exist_ok=True)
    fig.savefig(od / f'sheet_{dom}_{key}.png', dpi=100)
    (od / f'sheet_{dom}_{key}.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=('cache', 'tag', 'prep', 'report', 'fwdcheck', 'sheet'))
    p.add_argument('--dom', required=True, choices=('mice', 'ants'))
    p.add_argument('--key', default='')
    a = p.parse_args()
    {'cache': cmd_cache, 'tag': cmd_tag, 'prep': cmd_prep, 'report': cmd_report, 'fwdcheck': cmd_fwdcheck, 'sheet': cmd_sheet}[a.step](a)
