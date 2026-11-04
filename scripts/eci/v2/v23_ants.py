"""
ECI v2.3 ants alignment (scripts/eci/v2/v23_prereg.md AMENDMENT 1). Frames: the eval.py ants frame table (human-labelled
first 10 min, 1 fps, 600 frames per video), series v3 + v4 only (147 videos). Video halves A / B stratified by series x
treatment (rng 20261105). Per behaviour (grooming-to-focal (any), Y2F, B2F, focal single, focal double): best-AUROC
eligible neuron on the selection half, held-out metrics on the other half (A -> B primary, B -> A reported), the v22_eval
held-out quantities and bootstrap (videos within series x treatment, and seeds; 1,000 replicates).
Readouts: v2 (V_s*, Vall_s*) = per-neuron 3-state HMM P(on) (v22_full.pon_fast, one sequence per 5 fps eval span of every
ants eval video), read at the scored frames; v1_s* = raw stored frame code for ranking, the same HMM on the picked
neuron's 1 fps code (one sequence per video) for F1 / time.
Steps: tag --model KEY (CPU) -> OUTA/tags/KEY.pkl ; report (CPU) -> OUTA/report.json, OUTA/summary.md
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v22_core as C  # noqa: E402
import v22_eval as E  # noqa: E402
import v22_full as VF  # noqa: E402

REPO, T, log = C.REPO, C.T, C.log
OUTA = REPO / 'results/vision/eci_v2_3/ants'
BEE23 = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_3')
BEH = {'groom_any': None, 'Y2F': None, 'B2F': None, 'single': None, 'double': None}
NAME = {'groom_any': 'grooming-to-focal (any)', 'Y2F': 'Y2F (yellow grooms focal)', 'B2F': 'B2F (blue grooms focal)',
        'single': 'focal single (Y2F xor B2F)', 'double': 'focal double (Y2F and B2F)'}
SPLIT_RNG = 20261105
DIRS = {'A2B': ('A', 'B'), 'B2A': ('B', 'A')}
ARMS = {'v1': [f'v1_s{s}' for s in range(11)], 'v1s0': ['v1_s0'], 'V': [f'V_s{s}' for s in range(1001, 1007)],
        'Vall': [f'Vall_s{s}' for s in (2001, 2002, 2003)]}
CONTRASTS = [('Vall', 'V'), ('Vall', 'v1'), ('V', 'v1'), ('Vall', 'v1s0')]
WIN_S = 600.0  # seconds per scored video (1 fps x 600 frames)


def table():
    import eval as EV
    F0 = EV.frame_table('ants')
    keep = F0.series.isin(['v3', 'v4']).values
    F = F0[keep].copy()
    F['frow'] = np.flatnonzero(keep)
    tr = {}
    for ser in ('v3', 'v4'):
        e = pd.read_csv(REPO / 'data/ants' / ser / 'experiment.csv')
        tr.update({(ser, str(o)): int(t) for o, t in zip(e.observation_id, e.treatment)})
    F['treatment'] = [tr[(s, o)] for s, o in zip(F.series.values, F.obs.values)]
    assert F.v_groom_any.all(), 'unlabelled frames in v3 / v4'
    F['y_single'] = F.y_Y2F.values ^ F.y_B2F.values
    F['y_double'] = F.y_Y2F.values & F.y_B2F.values
    F['unit'] = F.obs.values
    F['cls'] = 0
    v = F.groupby('obs', sort=False).agg(series=('series', 'first'), treatment=('treatment', 'first'), n=('k', 'size'))
    assert len(v) == 147 and (v.n == 600).all(), (len(v), v.n.unique())
    rng = np.random.default_rng(SPLIT_RNG)
    half = {}
    for (s, t), g in v.groupby(['series', 'treatment'], sort=True):
        o = np.array(sorted(g.index))[rng.permutation(len(g))]
        half.update({x: ('A' if i % 2 == 0 else 'B') for i, x in enumerate(o)})
    F['half'] = F.obs.map(half).values
    F['stratum'] = F.series + '_t' + F.treatment.astype(str)
    return F.reset_index(drop=True)


def readout(key, F):
    """-> rank readout X, raw code R (both (len(F), W) float32, aligned to F), P(on) or None (v1)."""
    import eval as EV
    if key.startswith('v1_'):
        F0 = EV.frame_table('ants')
        R = np.asarray(EV.load_dict('ants', key, F0)[0][F.frow.values], np.float32)
        return R, R, None
    cp = BEE23 / 'ants' / 'cache' / f'{key}_pon.npy'
    cr = cp.with_name(f'{key}_raw.npy')
    if cp.exists() and cr.exists():
        return np.load(cp), np.load(cr).astype(np.float32), np.load(cp)
    od = T.BEE / 'ants' / 'codes'
    Rw = pd.read_parquet(od / 'eval_rows.parquet')
    Ef = T.frame_sets('ants')['eval']  # = eval.frame_table order, v4_infested renamed v4
    rix = pd.Series(np.arange(len(Rw)), index=pd.MultiIndex.from_frame(Rw)).reindex(pd.MultiIndex.from_frame(Ef)).values
    assert not np.isnan(rix).any()
    rix = rix.astype(np.int64)[F.frow.values]
    st = np.r_[0, np.flatnonzero((Rw.obs.values[1:] != Rw.obs.values[:-1]) | (Rw.series.values[1:] != Rw.series.values[:-1])) + 1, len(Rw)]
    base = key
    p = (BEE23 / 'ants' / 'codes' / f'{base}_max_eval.npy') if base.startswith('Vall_') else (od / f'{base}_eval.npy')
    raw = np.load(p).astype(np.float32)
    assert len(raw) == len(Rw)
    P, it = VF.pon_fast(raw, st)
    cp.parent.mkdir(parents=True, exist_ok=True)
    np.save(cp, P[rix])
    np.save(cr, raw[rix].astype(np.float16))
    log(f'{key}: HMM on {len(raw):,} span frames (iterations median {np.median(it):.0f}, max {it.max()})')
    return np.load(cp), np.load(cr).astype(np.float32), np.load(cp)


def cmd_tag(a):
    key = a.model
    t0 = time.time()
    F = table()
    X, R, Pn = readout(key, F)
    el = (X > 0).mean(0) >= C.ELIG
    st = np.r_[np.flatnonzero(np.r_[True, F.obs.values[1:] != F.obs.values[:-1]]), len(F)].astype(np.int64)
    picks = {}
    cols = []
    for dn, (s_, e_) in DIRS.items():
        sel = F.half.values == s_
        for b in BEH:
            y = F[f'y_{b}'].values
            cols.append(np.where(sel, np.where(y, 2, 1), 0))
    A = C._auc_multi(np.ascontiguousarray(X.T, dtype=np.float32), np.ascontiguousarray(np.stack(cols, 1).astype(np.int8)))
    q = 0
    for dn in DIRS:
        picks[dn] = {}
        for b in BEH:
            sc = np.where(el, A[q], np.nan)
            picks[dn][b] = (int(np.nanargmax(sc)), float(np.nanmax(sc)))
            q += 1
    js = sorted({j for d in picks.values() for j, _ in d.values()})
    PC = Pn[:, js] if Pn is not None else VF.pon_fast(R[:, js], st)[0]
    col = {j: i for i, j in enumerate(js)}
    rng = np.random.default_rng(0)
    res = {'key': key, 'sets': {}, 'n_eligible': int(el.sum())}
    for dn, (s_, e_) in DIRS.items():
        sel, ev = F.half.values == s_, F.half.values == e_
        units_ev = sorted(set(F.unit.values[ev]))
        d = {'picks': {b: j for b, (j, _) in picks[dn].items()}, 'sel_auc': {b: v for b, (_, v) in picks[dn].items()}, 'units_ev': units_ev}
        for b in BEH:
            j = picks[dn][b][0]
            d[b] = E.held(F, X, R, Pn, j, PC[:, col[j]], sel, ev, b, units_ev, rng)
        res['sets'][dn] = [d]
    OUTA.joinpath('tags').mkdir(parents=True, exist_ok=True)
    pickle.dump(res, open(OUTA / 'tags' / f'{key}.pkl', 'wb'))
    log(f'ants tag {key} ({time.time() - t0:.0f}s): picks A2B {res["sets"]["A2B"][0]["picks"]}')


def video_stats(tags, dn, b):
    acc = {}
    for tg in tags:
        d = tg['sets'][dn][0]
        V = d[b]['video']
        S, Tt, n = V['S'], V['T'], V['n']
        for nm, x in (('raw_bias', S - Tt), ('raw_mae', np.abs(S - Tt)), ('cal_bias', d[b]['c'] * S - Tt),
                      ('cal_mae', np.abs(d[b]['c'] * S - Tt)), ('const_mae', np.abs(d[b]['const_rate'] * n - Tt)), ('true', Tt)):
            x = x * (WIN_S / n)  # seconds of the 600-s window
            acc.setdefault(nm, []).append((x.mean(), x.std(ddof=1) / np.sqrt(len(x))))
    return {nm: (float(np.mean([m for m, _ in v])), float(np.mean([s for _, s in v]))) for nm, v in acc.items()}


def cmd_report(a):
    NB = 1000
    F = table()
    vids = F.groupby('obs', sort=True).agg(stratum=('stratum', 'first'), half=('half', 'first'))
    rng = np.random.default_rng(20261106)
    tags = {k: pickle.load(open(OUTA / 'tags' / f'{k}.pkl', 'rb')) for ks in ARMS.values() for k in ks
            if (OUTA / 'tags' / f'{k}.pkl').exists()}
    arms = {arm: ks for arm, ks in ARMS.items() if all(k in tags for k in ks)}
    res = {'arms': arms, 'split': {'rng': SPLIT_RNG, 'A': sorted(vids.index[vids.half == 'A']), 'B': sorted(vids.index[vids.half == 'B'])},
           'strata': vids.groupby(['stratum', 'half']).size().unstack().to_dict(), 'dirs': {}}
    MK = ('auc', 'p100', 'F1', 'r_soft', 'bias_raw_soft_frac', 'mae_raw_soft_frac', 'bias_cal_soft_frac', 'mae_cal_soft_frac', 'mae_const_frac')
    md = ['# v2.3 ants alignment: v1 vs V vs Vall (v3 + v4, 147 videos, first 10 min at 1 fps)', '',
          f'Split by video, stratified by series x treatment (rng {SPLIT_RNG}): A {int((vids.half == "A").sum())} videos, '
          f'B {int((vids.half == "B").sum())}. Mean over seeds [95% CI, bootstrap over held-out videos within series x treatment and '
          f'seeds, {NB} replicates]. Per-video time in seconds of the 600-s window, mean over held-out videos +- SE (mean over seeds). '
          'v1 = seeds s0-s10, v1s0 = seed 0, V = 1001-1006, Vall = 2001-2003. Rule: best AUROC on the selection half.', '']
    for dn, (s_, e_) in DIRS.items():
        ev = vids.index[vids.half == e_].tolist()
        strata = vids.loc[ev, 'stratum'].values
        W = np.ones((NB + 1, len(ev)))
        for s in np.unique(strata):
            ix = np.flatnonzero(strata == s)
            W[1:, ix] = np.stack([np.bincount(rng.integers(0, len(ix), len(ix)), minlength=len(ix)) for _ in range(NB)])
        per = {}
        for arm, ks in arms.items():
            M = []
            for k in ks:
                d = tags[k]['sets'][dn][0]
                assert d['units_ev'] == ev
                M.append({(b, m_): v for b in BEH for m_, v in E.metrics(d[b], W).items()})
            per[arm] = {mk: np.stack([m[mk] for m in M]) for mk in M[0]}
        R = {'levels': {}, 'contrasts': {}, 'picks': {}, 'video': {}}
        for arm, d in per.items():
            for (b, m_), v in d.items():
                if m_ not in MK:
                    continue
                ix = rng.integers(0, len(v), (NB, len(v)))
                bb = np.nanmean(v[:, 1:][ix, np.arange(NB)[:, None]], 1)
                R['levels'].setdefault(arm, {})[f'{b}|{m_}'] = {'mean': float(np.nanmean(v[:, 0])), 'ci': np.nanpercentile(bb, [2.5, 97.5]).tolist(),
                                                               'seeds': v[:, 0].tolist()}
            R['picks'][arm] = {k: tags[k]['sets'][dn][0]['picks'] for k in arms[arm]}
            R['video'][arm] = {b: video_stats([tags[k] for k in arms[arm]], dn, b) for b in BEH}
        for x, y in CONTRASTS:
            if x not in per or y not in per:
                continue
            r = {}
            for (b, m_) in per[x]:
                if m_ not in MK:
                    continue
                ax, ay = per[x][(b, m_)][:, 1:], per[y][(b, m_)][:, 1:]
                ix = rng.integers(0, len(ax), (NB, len(ax)))
                iy = rng.integers(0, len(ay), (NB, len(ay)))
                q = np.arange(NB)[:, None]
                dd = np.nanmean(ax[ix, q], 1) - np.nanmean(ay[iy, q], 1)
                r[f'{b}|{m_}'] = {'diff': float(np.nanmean(per[x][(b, m_)][:, 0]) - np.nanmean(per[y][(b, m_)][:, 0])),
                                  'ci': np.nanpercentile(dd, [2.5, 97.5]).tolist()}
            R['contrasts'][f'{x} - {y}'] = r
        res['dirs'][dn] = R
        Fe = F[F.half.values == e_]
        md += [f'## select on {s_}, measure on {e_}{" (PRIMARY)" if dn == "A2B" else ""}', '']
        for b in BEH:
            y = Fe[f'y_{b}'].values
            br = np.r_[True, Fe.obs.values[1:] != Fe.obs.values[:-1]]
            nb_ = int((y & (br | ~np.r_[False, y[:-1]])).sum())
            md += [f'### {NAME[b]}', '', f'Held-out: {len(Fe):,} frames, {len(ev)} videos; positive {int(y.sum()):,} frames, {nb_:,} bouts '
                   f'(at 1 fps), {int(Fe.groupby("obs")[f"y_{b}"].any().sum())} videos with >= 1; prevalence {y.mean():.4f}.', '',
                   '| arm | AUROC | precision@100 | F1 (P(on) > 0.5) | Pearson r per-video time | raw time bias, s | raw time MAE, s | '
                   'calibrated time bias, s | calibrated time MAE, s | constant-rate MAE, s | picks |', '|' + '---|' * 11]
            for arm in arms:
                L_ = R['levels'][arm]
                f = lambda m_, d=3: f"{L_[f'{b}|{m_}']['mean']:.{d}f} [{L_[f'{b}|{m_}']['ci'][0]:.{d}f}, {L_[f'{b}|{m_}']['ci'][1]:.{d}f}]"  # noqa: E731
                vs = R['video'][arm][b]
                pm = lambda nm: f'{vs[nm][0]:.1f} +- {vs[nm][1]:.1f}'  # noqa: E731
                pk = sorted({p[b] for p in R['picks'][arm].values()})
                md.append(f"| {arm} | {f('auc')} | {f('p100')} | {f('F1')} | {f('r_soft', 2)} | {pm('raw_bias')} | {pm('raw_mae')} | "
                          f"{pm('cal_bias')} | {pm('cal_mae')} | {pm('const_mae')} | {', '.join(f'#{j}' for j in pk[:6])}{' ...' if len(pk) > 6 else ''} |")
            md.append('')
        md += [f'### Contrasts (select on {s_}, measure on {e_})', '', '| contrast | ' + ' | '.join(f'{NAME[b]} AUROC' for b in BEH) +
               ' | grooming-to-focal (any) precision@100 | grooming-to-focal (any) Pearson r time |', '|---|' + '---|' * (len(BEH) + 2)]
        for cn, r in R['contrasts'].items():
            md.append(f'| {cn} | ' + ' | '.join(f"{r[f'{b}|auc']['diff']:+.3f} [{r[f'{b}|auc']['ci'][0]:+.3f}, {r[f'{b}|auc']['ci'][1]:+.3f}]" for b in BEH) +
                      ''.join(f" | {r[f'groom_any|{m_}']['diff']:+.3f} [{r[f'groom_any|{m_}']['ci'][0]:+.3f}, {r[f'groom_any|{m_}']['ci'][1]:+.3f}]" for m_ in ('p100', 'r_soft')) + ' |')
        md.append('')
    c = res['dirs']['A2B']['contrasts'].get('Vall - V')
    if c:
        res['A4'] = {'AUROC lower CI': c['groom_any|auc']['ci'][0], 'Pearson r lower CI': c['groom_any|r_soft']['ci'][0]}
        res['A4']['pass'] = bool(res['A4']['AUROC lower CI'] >= -0.02 and res['A4']['Pearson r lower CI'] >= -0.02)
        md = md[:2] + ['## A4 (grooming-to-focal (any), A -> B, Vall - V; both lower CIs >= -0.02)', '', '```', json.dumps(res['A4'], indent=1), '```', ''] + md[2:]
    (OUTA / 'report.json').write_text(json.dumps(res, indent=1, default=float))
    (OUTA / 'summary.md').write_text('\n'.join(md) + '\n')
    log('ants report written' + (f": A4 {res['A4']}" if 'A4' in res else ''))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=('tag', 'report', 'split'))
    p.add_argument('--model', default='')
    a = p.parse_args()
    if a.step == 'split':
        F = table()
        v = F.groupby('obs').agg(stratum=('stratum', 'first'), half=('half', 'first'))
        OUTA.mkdir(parents=True, exist_ok=True)
        out = {'rng': SPLIT_RNG, 'videos': len(v), 'frames': len(F), 'by_stratum_half': v.groupby(['stratum', 'half']).size().unstack().to_dict(),
               'A': sorted(v.index[v.half == 'A']), 'B': sorted(v.index[v.half == 'B'])}
        for h in 'AB':
            G = F[F.half == h]
            br = np.r_[True, G.obs.values[1:] != G.obs.values[:-1]]
            out[f'counts_{h}'] = {b: {'frames_on': int(G[f'y_{b}'].sum()), 'bouts': int((G[f'y_{b}'].values & (br | ~np.r_[False, G[f'y_{b}'].values[:-1]])).sum()),
                                      'videos_with_any': int(G.groupby('obs')[f'y_{b}'].any().sum())} for b in BEH}
        (OUTA / 'split.json').write_text(json.dumps(out, indent=1))
        print(json.dumps({k: v_ for k, v_ in out.items() if k not in ('A', 'B')}, indent=1))
    else:
        {'tag': cmd_tag, 'report': cmd_report}[a.step](a)
