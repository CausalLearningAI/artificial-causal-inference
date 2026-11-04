"""
ECI v2.2 mice evaluation: SG-based neuron selection, held-out metrics, unit x seed bootstrap (pre-registration
scripts/eci/v2/v22_prereg.md).

Direction sets (selection frames -> held-out frames):
  sg2non  select on ALL frames of the SG pools, evaluate on the non-SG pools (primary, (a))
  sgcf    within the SG pools: the 6 unit splits of eval.SPLITS (atlas + rnd0-4) x 2 halves = 12 directions ((b))
  old     the v21 extended rule: all 24 pools, the same 12 directions, rule 'auc' (identity check only)
Rules (v22_core.select): auc (current: best AUROC), bal (best balanced AUROC), balpair (best distinct pair under bal).
Eligibility (label-free, all rules): the neuron's readout is > 0 on >= 0.2% of all eval frames.

Per (model, direction set, rule, behaviour) and direction, held-out quantities for the bootstrap:
  AUROC and discrimination AUROC (nt vs nose_nose, exclusive frames) as unit x unit pair-count matrices (weighted AUROC
  = w' A w / (w' P)(w' N)); the top frames for precision@100 (rank readout, ties by raw code, then a fixed random
  draw); per-unit TP / FP / FN of P(on) > 0.5; per-video sum P(on), true on-frames, frames; calibration scalar c =
  sum true / sum P(on) on the selection frames (held fixed in the bootstrap, as the picks).
Steps
  tag --model KEY      (CPU) -> OUT22/tags/<KEY>.pkl (+ identity vs the v21 extended pickle for stored V / v1 models)
  report               (CPU) -> OUT22/report.json, OUT22/tables.md
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
import v22_core as C  # noqa: E402

L = C.L
OUT22, log = C.OUT22, C.log
EXT = C.REPO / 'results/vision/eci_v2_1/extended/tags/mice'
RULES = ('auc', 'bal', 'balpair', 'hyb', 'p100')  # hyb, p100 = POST HOC exploratory
DT = 0.2  # seconds per mice eval frame
KTOP = 4000


def directions(F, SP):
    import eval as EV
    import analysis as AN
    units = F.unit.values
    sgu = sorted(set(units[F.sg.values]))
    D = {'sg2non': [(F.sg.values, F.nonsg.values)]}
    SPs = AN.splits(sgu, EV.SPLITS)
    D['sgcf'] = []
    D['old'] = []
    for nm in EV.SPLITS:
        for h in (0, 1):
            hs = np.array([SPs[nm].get(u, -1) for u in units])
            D['sgcf'].append((hs == h, hs == 1 - h))
            ha = np.array([SP[nm][u] for u in units])
            D['old'].append((ha == h, ha == 1 - h))
    return D, sgu


def uu_auc(s, y, u, U):
    """A (U, U): A[a, b] = sum over positives i in unit a of (#negatives in unit b below s_i + 0.5 #tied); P_u, N_u."""
    A = np.zeros((U, U))
    pos = y.astype(bool)
    for b in range(U):
        ng = np.sort(s[(~pos) & (u == b)])
        if not len(ng):
            continue
        sp = s[pos]
        lo = np.searchsorted(ng, sp, 'left')
        hi = np.searchsorted(ng, sp, 'right')
        A[:, b] = np.bincount(u[pos], weights=lo + 0.5 * (hi - lo), minlength=U)
    return A, np.bincount(u[pos], minlength=U).astype(float), np.bincount(u[~pos], minlength=U).astype(float)


def held(F, X, R, Pn, j, pcol, sel, ev, b, units_ev, rng):
    """held-out quantities of neuron j for behaviour b; pcol = P(on) column (n,) of j."""
    uix = pd.Series(np.arange(len(units_ev)), index=units_ev)
    rows = np.flatnonzero(ev)
    u = uix.reindex(F.unit.values[rows]).values.astype(np.int64)
    U = len(units_ev)
    y = F[f'y_{b}'].values[rows]
    s = X[rows, j].astype(np.float64)
    out = {'auc': uu_auc(s, y, u, U)}
    cls = F.cls.values[rows]
    m = (cls == 1) | (cls == 2)
    out['disc'] = uu_auc(s[m], cls[m] == 1, u[m], U)
    o = np.lexsort((rng.random(len(rows)), -R[rows, j].astype(np.float64), -s))[:KTOP]
    out['top'] = (u[o], y[o])
    p = pcol[rows]
    hd = p > 0.5
    out['f1'] = tuple(np.bincount(u, weights=w, minlength=U) for w in (hd & y, hd & ~y, (~hd) & y))
    vcode, vids = pd.factorize(F.obs.values[rows])
    out['video'] = {'S': np.bincount(vcode, weights=p), 'H': np.bincount(vcode, weights=hd), 'T': np.bincount(vcode, weights=y),
                    'n': np.bincount(vcode).astype(float), 'unit': uix.reindex(pd.Series(F.unit.values[rows]).groupby(vcode).first().values).values,
                    'vid': list(vids)}
    srows = np.flatnonzero(sel)
    ys = F[f'y_{b}'].values[srows]
    out['c'] = float(ys.sum() / pcol[srows].sum()) if pcol[srows].sum() > 0 else np.nan
    out['c_hard'] = float(ys.sum() / (pcol[srows] > 0.5).sum()) if (pcol[srows] > 0.5).sum() > 0 else np.nan
    out['const_rate'] = float(ys.mean())
    return out


def cmd_tag(a):
    key = a.model
    t0 = time.time()
    F, SP = C.mice_table()
    X, R, Pn = C.load(key)
    el = (X > 0).mean(0) >= C.ELIG
    st_ = C.obs_starts(F)
    D, sgu = directions(F, SP)
    res = {'key': key, 'sets': {}}
    picks_all = {}
    stc = {}
    for dset, dirs in D.items():
        rules = ('auc',) if dset == 'old' else RULES
        for rule in rules:
            for di, (sel, ev) in enumerate(dirs):
                if dset != 'old' and (dset, di) not in stc:
                    stc[(dset, di)] = C.neuron_stats(X, F, np.flatnonzero(sel), R)
                st = dict(stc[(dset, di)]) if dset != 'old' else None
                if st is None:
                    rows = np.flatnonzero(sel)
                    st = {'fire': np.ones(X.shape[1])}
                    for b in C.BEH:
                        st[f'auc_{b}'] = C.auroc_cols(X, rows, F[f'y_{b}'].values[rows])
                st['fire'] = np.where(el, 1.0, 0.0)  # eligibility on all eval frames (label-free)
                pk = C.select(st, rule)
                picks_all[(dset, rule, di)] = (pk, sel, ev, {b: float(st[f'auc_{b}'][pk[b]]) for b in C.BEH})
    # P(on) columns of every picked neuron (v1: the 3-state HMM on its raw code, as v21_extended)
    js = sorted({j for pk, *_ in picks_all.values() for j in pk.values()})
    if Pn is None:
        PC = C.hmm_cols(R, js, st_)
    else:
        PC = Pn[:, js]
    col = {j: i for i, j in enumerate(js)}
    rng = np.random.default_rng(0)
    for (dset, rule, di), (pk, sel, ev, sauc) in picks_all.items():
        units_ev = sorted(set(F.unit.values[ev]))
        d = {'picks': pk, 'sel_auc': sauc, 'units_ev': units_ev}
        for b in C.BEH:
            d[b] = held(F, X, R, Pn, pk[b], PC[:, col[pk[b]]], sel, ev, b, units_ev, rng)
        res['sets'].setdefault(f'{dset}|{rule}', []).append(d)
    # identity vs the v21 extended pickle (stored models, old rule)
    ep = EXT / f'{key}.pkl'
    if ep.exists():
        E = pickle.load(open(ep, 'rb'))
        mx, n, same = 0.0, 0, True
        for b in C.BEH:
            Ec = E['cells'][f'v1|{b}']
            mine = res['sets']['old|auc']
            same &= [dd['neuron'] for dd in Ec['dirs']] == [dd['picks'][b] for dd in mine]
            aucs = []
            for dd in mine:
                A, P_, N_ = dd[b]['auc']
                aucs.append(A.sum() / (P_.sum() * N_.sum()))
            mx, n = max(mx, abs(np.mean(aucs) - Ec['AUROC'])), n + 1
            # per-video sum P(on) of every direction (held-out videos only)
            for dd, ed in zip(mine, Ec['dirs']):
                vmap = dict(zip(Ec['videos'], ed['S']))
                for v, sv in zip(dd[b]['video']['vid'], dd[b]['video']['S']):
                    mx, n = max(mx, abs(sv - vmap[v])), n + 1
        res['identity_vs_extended'] = {'picks_identical': bool(same), 'max_abs_diff': float(mx), 'n': n}
        log(f'IDENTITY {key}: {res["identity_vs_extended"]}')
        if not same or mx > 1e-4:
            raise SystemExit(f'IDENTITY CHECK FAILED {key}')
    (OUT22 / 'tags').mkdir(parents=True, exist_ok=True)
    with open(OUT22 / 'tags' / f'{key}.pkl', 'wb') as f:
        pickle.dump(res, f)
    log(f'tag {key} done ({time.time() - t0:.0f}s)')


# ---------------------------------------------------------------------------------------------- metrics on weights
def wpearson(x, y, W):
    sw = W.sum(1)
    mx, my = (W @ x) / sw, (W @ y) / sw
    cxy = (W @ (x * y)) / sw - mx * my
    vx = (W @ (x * x)) / sw - mx ** 2
    vy = (W @ (y * y)) / sw - my ** 2
    with np.errstate(invalid='ignore', divide='ignore'):
        return cxy / np.sqrt(vx * vy)


def metrics(h, w):
    """h = held() dict, w (R, U) unit weights -> {metric: (R,)}."""
    out = {}
    with np.errstate(invalid='ignore', divide='ignore'):
        for nm in ('auc', 'disc'):
            A, P_, N_ = h[nm]
            out[nm] = np.einsum('ru,uv,rv->r', w, A, w) / ((w @ P_) * (w @ N_))
        uu, yy = h['top']
        wf = w[:, uu]
        cum = np.cumsum(wf, 1)
        inc = np.clip(100 - (cum - wf), 0, wf)
        p = (inc * yy).sum(1) / 100
        p[cum[:, -1] < 100] = np.nan
        out['p100'] = p
        tp, fp, fn = (w @ x for x in h['f1'])
        out['F1'] = 2 * tp / (2 * tp + fp + fn)
        V = h['video']
        Wv = w[:, V['unit']]
        sw = Wv.sum(1)
        n = V['n']
        for nm, S in (('soft', V['S']), ('hard', V['H'])):
            c = h['c'] if nm == 'soft' else h['c_hard']
            for kind, Sx in (('raw', S), ('cal', c * S)):
                bv = Sx - V['T']
                out[f'bias_{kind}_{nm}_s'] = (Wv @ bv) / sw * DT
                out[f'mae_{kind}_{nm}_s'] = (Wv @ np.abs(bv)) / sw * DT
                out[f'bias_{kind}_{nm}_frac'] = (Wv @ (bv / n)) / sw
                out[f'mae_{kind}_{nm}_frac'] = (Wv @ (np.abs(bv) / n)) / sw
            out[f'r_{nm}'] = wpearson(S, V['T'], Wv)
        bc = h['const_rate'] * n - V['T']
        out['mae_const_frac'] = (Wv @ (np.abs(bc) / n)) / sw
    return out


def model_metrics(tag, setkey, w_by_units):
    """mean over directions of the per-direction metrics; w_by_units(units_ev) -> (R, U) weights."""
    acc = {}
    for d in tag['sets'][setkey]:
        w = w_by_units(d['units_ev'])
        for b in C.BEH:
            for k, v in metrics(d[b], w).items():
                acc.setdefault((b, k), []).append(v)
    return {k: np.nanmean(np.stack(v), 0) for k, v in acc.items()}


# ---------------------------------------------------------------------------------------------- report
def arms_default():
    return json.loads((OUT22 / 'arms.json').read_text())


def cmd_report(a):
    NB = 1000
    F, _ = C.mice_table()
    units = sorted(set(F.unit.values))
    rng = np.random.default_rng(20261104)
    sgu = sorted(set(F.unit.values[F.sg.values]))
    non = sorted(set(F.unit.values[F.nonsg.values]))
    # replicate weights: multinomial resampling WITHIN the held-out unit pool of each direction set
    Wrep = {}
    for nm, pool in (('sg2non', non), ('sgcf', sgu), ('old', units)):
        W = np.ones((NB + 1, len(units)))
        ix = [units.index(u) for u in pool]
        W[1:, ix] = np.stack([np.bincount(rng.integers(0, len(pool), len(pool)), minlength=len(pool)) for _ in range(NB)])
        Wrep[nm] = W
    arms = arms_default()  # {arm: {'models': [...], 'rule': rule}}
    res = {'arms': arms, 'sets': {}}
    cache = {}
    for dset in ('sg2non', 'sgcf'):
        uidx = {u: i for i, u in enumerate(units)}
        wb = lambda ue: Wrep[dset][:, [uidx[u] for u in ue]]  # noqa: E731
        per = {}
        for arm, spec in arms.items():
            if not all((OUT22 / 'tags' / f'{k}.pkl').exists() for k in spec['models']):
                log(f'{arm}: tags missing, skipped')
                continue
            M = []
            for k in spec['models']:
                if k not in cache:
                    cache[k] = pickle.load(open(OUT22 / 'tags' / f'{k}.pkl', 'rb'))
                M.append(model_metrics(cache[k], f'{dset}|{spec["rule"]}', wb))
            per[arm] = {mk: np.stack([m[mk] for m in M]) for mk in M[0]}  # (seeds, NB+1)
        lev, picks = {}, {}
        for arm, d in per.items():
            spec = arms[arm]
            lev[arm] = {f'{b}|{k}': {'median': float(np.nanmedian(v[:, 0])), 'min': float(np.nanmin(v[:, 0])),
                                     'max': float(np.nanmax(v[:, 0])), 'mean': float(np.nanmean(v[:, 0])), 'seeds': v[:, 0].tolist()}
                        for (b, k), v in d.items()}
            picks[arm] = {k: [dd['picks'] for dd in cache[k]['sets'][f'{dset}|{spec["rule"]}']] for k in spec['models']}
        con = {}
        for x, y in [tuple(c.split(' - ')) for c in json.loads((OUT22 / 'contrasts.json').read_text())]:
            if x not in per or y not in per:
                continue
            r = {}
            for mk in per[x]:
                ax, ay = per[x][mk][:, 1:], per[y][mk][:, 1:]
                ix = rng.integers(0, len(ax), (NB, len(ax)))
                iy = rng.integers(0, len(ay), (NB, len(ay)))
                rr = np.arange(NB)[:, None]
                dd = np.nanmean(ax[ix, rr], 1) - np.nanmean(ay[iy, rr], 1)
                r[f'{mk[0]}|{mk[1]}'] = {'diff': float(np.nanmean(per[x][mk][:, 0]) - np.nanmean(per[y][mk][:, 0])),
                                         'ci': np.nanpercentile(dd, [2.5, 97.5]).tolist()}
            con[f'{x} - {y}'] = r
        res['sets'][dset] = {'levels': lev, 'contrasts': con, 'picks': picks}
    # ground truth / coverage
    gt = {}
    for part, m in (('all', np.ones(len(F), bool)), ('sg', F.sg.values), ('nonsg', F.nonsg.values), ('unknown', F.annot.values == 'NA')):
        G = F[m]
        d = {'pools': int(G.unit.nunique()), 'videos': int(G.obs.nunique()), 'frames': int(len(G))}
        for b in ('nose_nose', 'nt', 'nn', 'np'):
            y = G[f'y_{b}'].values
            pv = G.groupby('obs')[f'y_{b}'].mean()
            on = np.r_[False, y] [1:] & ~np.r_[False, y][:-1]
            # bouts: rising edges within a video
            br = np.r_[True, G.obs.values[1:] != G.obs.values[:-1]]
            bouts = int((y & (br | ~np.r_[False, y[:-1]])).sum())
            d[b] = {'frames_on': int(y.sum()), 'bouts': bouts, 'rate': float(y.mean()), 'per_video_mean': float(pv.mean()),
                    'per_video_se': float(pv.std(ddof=1) / np.sqrt(len(pv))), 'videos_with_any': int((pv > 0).sum())}
            _ = on
        d['exclusive_nt_only'] = int((G.cls.values == 1).sum())
        d['exclusive_nose_nose_only'] = int((G.cls.values == 2).sum())
        d['both'] = int((G.cls.values == -1).sum())
        gt[part] = d
    res['ground_truth'] = gt
    res['annotators'] = F.groupby('annot').agg(pools=('unit', 'nunique'), videos=('obs', 'nunique'), frames=('obs', 'size')).to_dict('index')
    (OUT22 / 'report.json').write_text(json.dumps(res, indent=1, default=float))
    log('report.json written')
    cmd_tables(a)


FMT = [('auc', 'AUROC', 3), ('p100', 'precision@100', 3), ('F1', 'F1 (P(on)>0.5)', 3), ('disc', 'discrimination AUROC nt vs nose_nose', 2),
       ('bias_raw_soft_s', 'raw time bias, s', 1), ('mae_raw_soft_s', 'raw time MAE, s', 1), ('mae_raw_soft_frac', 'raw time MAE, frac', 3),
       ('bias_cal_soft_s', 'cal. time bias, s', 2), ('mae_cal_soft_s', 'cal. time MAE, s', 2), ('mae_cal_soft_frac', 'cal. time MAE, frac', 4),
       ('r_soft', 'Pearson r time', 2)]


def verdict(res):
    out = {}
    S = res['sets']['sg2non']
    for arm in S['levels']:
        if arm.startswith('v1') or arm in ('V|auc', 'Vf|auc'):
            continue
        ref = 'Vf|auc' if res['arms'][arm].get('amendment1') else 'V|auc'
        pk = S['picks'][arm]
        dist = [p[0]['nose_nose'] != p[0]['nt'] for p in pk.values()]
        need = 2 if len(dist) <= 3 else 4
        lv = S['levels'][arm]
        c2 = (lv['nt|disc']['median'] > 0.6, lv['nose_nose|disc']['median'] < 0.4)
        con = S['contrasts'].get(f'{arm} - {ref}', {})
        c3 = {b: con[f'{b}|auc']['ci'][0] for b in C.BEH} if con else {}
        out[arm] = {'C1 distinct seeds': f'{sum(dist)}/{len(dist)}', 'C1': sum(dist) >= need,
                    'C2 nt disc median': lv['nt|disc']['median'], 'C2 nose_nose disc median': lv['nose_nose|disc']['median'],
                    'C2': bool(c2[0] and c2[1]), 'C3 lower CI (AUROC vs V|auc)': c3,
                    'C3': bool(c3) and all(v >= -0.02 for v in c3.values()), 'C3 reference': ref,
                    'tiebreak mean p@100': 0.5 * (lv['nose_nose|p100']['mean'] + lv['nt|p100']['mean'])}
    return out


def cmd_tables(a):
    res = json.loads((OUT22 / 'report.json').read_text())
    md = []
    V = verdict(res)
    res['verdict_C1_C3'] = V
    (OUT22 / 'report.json').write_text(json.dumps(res, indent=1, default=float))
    md += ['## Verdict C1-C3 (design (a): select on SG, evaluate on non-SG pools)', '',
           '| arm | C1 distinct (seeds) | C2 nt disc / nose_nose disc (median) | C3 AUROC lower CI vs V|auc nose_nose / nt | C1 | C2 | C3 | mean p@100 |',
           '|---|---|---|---|---|---|---|---|']
    for arm, v in V.items():
        c3 = v['C3 lower CI (AUROC vs V|auc)']
        tag_ = ' (POST HOC)' if res['arms'][arm].get('post_hoc') else ''
        md.append(f"| {arm}{tag_} | {v['C1 distinct seeds']} | {v['C2 nt disc median']:.2f} / {v['C2 nose_nose disc median']:.2f} | "
                  f"{c3.get('nose_nose', float('nan')):+.3f} / {c3.get('nt', float('nan')):+.3f} | {'yes' if v['C1'] else 'NO'} | "
                  f"{'yes' if v['C2'] else 'NO'} | {'yes' if v['C3'] else 'NO'} | {v['tiebreak mean p@100']:.3f} |")
    for dset, title in (('sg2non', 'design (a): select on SG pools, evaluate on the 12 non-SG pools'),
                        ('sgcf', 'design (b): cross-fitted within the 11 SG pools (12 directions)')):
        S = res['sets'][dset]
        md += ['', f'## Levels, {title}', '', 'seed median [min, max]', '']
        for b in C.BEH:
            md += [f'### {C.NAME[b]}', '', '| arm | ' + ' | '.join(n for _, n, _ in FMT) + ' |', '|---|' + '---|' * len(FMT)]
            for arm, lv in S['levels'].items():
                md.append(f'| {arm} | ' + ' | '.join(f"{lv[f'{b}|{k}']['median']:.{d}f} [{lv[f'{b}|{k}']['min']:.{d}f}, {lv[f'{b}|{k}']['max']:.{d}f}]"
                                                     for k, _, d in FMT) + ' |')
            md.append('')
        md += [f'## Differences, {title}', '', 'mean over seeds [95% CI, pools + seeds bootstrap]', '']
        for b in C.BEH:
            md += [f'### {C.NAME[b]}', '', '| contrast | ' + ' | '.join(n for _, n, _ in FMT) + ' |', '|---|' + '---|' * len(FMT)]
            for cn, r in S['contrasts'].items():
                md.append(f'| {cn} | ' + ' | '.join(f"{r[f'{b}|{k}']['diff']:+.{d}f} [{r[f'{b}|{k}']['ci'][0]:+.{d}f}, {r[f'{b}|{k}']['ci'][1]:+.{d}f}]"
                                                    for k, _, d in FMT) + ' |')
            md.append('')
        md += [f'## Selected neurons, {title}', '', '| arm | model: nose-nose / nose-tail (per direction) |', '|---|---|']
        for arm, pk in S['picks'].items():
            md.append(f'| {arm} | ' + '; '.join(f"{k}: " + ', '.join(f"#{p['nose_nose']}/#{p['nt']}" for p in v) for k, v in pk.items()) + ' |')
    md += ['', '## Ground truth', '', '```', json.dumps(res['ground_truth'], indent=1), '```']
    (OUT22 / 'tables.md').write_text('\n'.join(md) + '\n')
    log('tables.md written')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=('tag', 'report', 'tables'))
    p.add_argument('--model', default='')
    a = p.parse_args()
    {'tag': cmd_tag, 'report': cmd_report, 'tables': cmd_tables}[a.step](a)
