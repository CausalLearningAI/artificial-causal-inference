"""ECI v1 vs v2 (arm V) extended evaluation on the known behaviours, pooled per domain (post hoc, reported view).

Nothing is retrained or re-encoded. Per model tag (v1_s0..10, V_s1001..1006):
  readout   = v21_ladder.readout (v1: raw frame-max code; V: per-neuron 3-state HMM P(on), refitted exactly as the ladder)
  selection = v21_ladder.score2 on the POOLED frame table (mice: the single series v1 = all 72 annotated pools; ants:
              v21_precK.pooled_frame = v2 + v3 + v4 + v4_infested, focal double only v3 + v4): best-AUROC neuron picked on
              one half of the units and evaluated on the other half, 12 directions (6 splits x 2 halves).
  v1 has no P(on) of its own: for the F1 / time metrics the SAME 3-state HMM (src.eci.v2.hmm.fit K=3, label-free) is
  fitted to the raw code of each selected v1 neuron over all eval frames, sequences = one per video (ants: the 1 fps
  eval grid, mice: 5 fps). AUROC and precision@100 for v1 use its raw code (= ladder / precK, identity-checked).
Steps
  tag --dom D --model KEY   (CPU job) -> results/vision/eci_v2_1/extended/tags/<dom>/<KEY>.pkl (+ selected columns on beegfs)
  report                    (CPU job) -> extended/report.json, report.md
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
import v21_ladder as L  # noqa: E402
import v21_precK as PK  # noqa: E402

REPO = L.REPO
OUT = REPO / 'results/vision/eci_v2_1/extended'
BEX = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_1/extended')
DT = {'mice': 0.2, 'ants': 1.0}  # seconds per eval frame (mice 5 fps contiguous, ants 1 fps grid)
CELLS = {'mice': ['nt', 'nose_nose', 'nn', 'np'], 'ants': ['groom_any', 'Y2F', 'B2F', 'single', 'double']}
NAME = {'nt': 'nose-tail', 'nose_nose': 'nose-nose', 'nn': 'nose-nose-mutual', 'np': 'nose-nose-passive',
        'groom_any': 'grooming-to-focal (any)', 'Y2F': 'Y2F', 'B2F': 'B2F', 'single': 'focal single', 'double': 'focal double'}
TAGS = {'v1': [f'v1_s{i}' for i in range(11)], 'v2': [f'V_s{s}' for s in range(1001, 1007)]}
NB = 1000
log = L.log


def table(dom):
    F, SP = L.frames(dom)
    if dom == 'mice':
        return F, F, SP, None
    import v20_gate as G
    import eval as EV
    S0 = json.loads((G.OUT / 'ants_scores.json').read_text())
    gate = [c for c, r in S0['eligibility'].items() if r['eligible'] and c.split('|')[1] not in G.ONLID]
    _beh = dict(EV.BEH)
    fel = L.focal_elig()
    EV.BEH.clear()
    EV.BEH.update(_beh)
    fcells = {'single': [c for c, e in fel.items() if e and c.endswith('|single')],
              'double': [c for c, e in fel.items() if e and c.endswith('|double')]}
    P, elig = PK.pooled_frame(F, gate, fcells)
    return F, P, SP, {b: sorted(s) for b, s in elig.items()}


def obs_starts(F):
    o = F.obs.values
    br = np.r_[True, o[1:] != o[:-1]]
    st = np.r_[np.flatnonzero(br), len(F)].astype(np.int64)
    assert len(st) - 1 == F.obs.nunique(), 'videos not contiguous in the frame table'
    return st


def cmd_tag(a):
    from src.eci.v2 import hmm as HM
    dom, tag = a.dom, a.model
    t0 = time.time()
    F, P, SP, elig = table(dom)
    X, _ = L.readout(dom, tag, '', F)
    res = L.score2(P, X, np.ones(len(P), bool), SP)
    cells = [f'{"v1" if dom == "mice" else "pooled"}|{b}' for b in CELLS[dom]]
    # identity vs the stored ladder (mice) / pooled precK (ants) values
    chk = {'n': 0, 'max_abs_diff': 0.0}
    if dom == 'mice':
        ref = json.loads((L.LO / 'mice' / f'{tag}.json').read_text())['gate']
        pk = np.load(PK.OUTD / 'boot_mice.npz')
        refp = {c: float(pk[f'{tag}|gate|{c}'][0, 0]) for c in cells}
    else:
        ref = json.loads((PK.POOL / f'{tag}.json').read_text())['cells']
        refp = {c: ref[c]['p@100'] for c in cells}
    D = L.cell_dirs(P, SP, res, 'gate')
    out = {'tag': tag, 'dom': dom, 'elig_series': elig, 'cells': {}}
    neurons = sorted({p[2] for c in cells for p in res[c]['AUROC_neurons']})
    raw = np.asarray(X[:, neurons], np.float64)
    st = obs_starts(F)
    if tag.startswith('v1_'):
        pon = np.zeros_like(raw)
        nfb = 0
        for i in range(len(neurons)):
            prm = HM.fit(raw[:, i], st, K=3)
            nfb += prm is None
            pon[:, i] = HM.apply(raw[:, i], st, prm)[0]
        out['hmm_fallbacks'] = int(nfb)
    else:
        pon = raw
    col = {j: i for i, j in enumerate(neurons)}
    vcode, vids = pd.factorize(P.obs.values)
    unit_of_vid = pd.Series(P.unit.values).groupby(vcode).first().values
    for c in cells:
        b = c.split('|')[1]
        r = res[c]
        y_all = P[f'y_{b}'].values.astype(bool)
        idx = np.flatnonzero(P[f'v_{b}'].values)
        vv = vcode[idx]
        cv = np.unique(vv)  # videos of this cell
        vpos = pd.Series(np.arange(len(cv)), index=cv).reindex(vv).values
        nv = np.bincount(vpos, minlength=len(cv)).astype(np.float64)
        Tv = np.bincount(vpos, weights=y_all[idx], minlength=len(cv))
        dirs = []
        p100 = []
        for (fr, j, jn, tb), (nm, h, j2) in zip(D[c], r['AUROC_neurons']):
            assert j == j2
            sc = np.asarray(X[fr, j], np.float64)
            p100.append(PK.direct(sc, y_all[fr], tb, min(100, len(fr))))
            p = pon[idx, col[j]]
            hd = p > 0.5
            yy = y_all[idx]
            d = {'split': nm, 'sel_half': int(h), 'neuron': int(j),
                 'half': np.array([SP[nm][u] for u in unit_of_vid[cv]]),  # unit half of each cell video
                 'S': np.bincount(vpos, weights=p, minlength=len(cv)),
                 'H': np.bincount(vpos, weights=hd, minlength=len(cv)),
                 'TP': np.bincount(vpos, weights=hd & yy, minlength=len(cv)),
                 'FP': np.bincount(vpos, weights=hd & ~yy, minlength=len(cv))}
            if tag.startswith('v1_'):
                fi = raw[idx, col[j]] > 0
                d['TPf'] = np.bincount(vpos, weights=fi & yy, minlength=len(cv))
                d['FPf'] = np.bincount(vpos, weights=fi & ~yy, minlength=len(cv))
            dirs.append(d)
        pm = float(np.mean(p100))
        for M, x, y in (('AUROC', r['AUROC'], ref[c]['AUROC']), ('p@100', pm, refp[c])):
            chk['n'] += 1
            chk['max_abs_diff'] = max(chk['max_abs_diff'], abs(x - y))
        out['cells'][c] = {'videos': vids[cv].tolist(), 'units': unit_of_vid[cv].tolist(), 'n': nv, 'T': Tv,
                           'base_rate': float(y_all[idx].mean()), 'n_frames': int(len(idx)), 'AUROC': r['AUROC'],
                           'AUROC_dirs': r['AUROC_dirs'], 'p@100': pm, 'p@100_dirs': p100, 'dirs': dirs,
                           'best_neuron': r['best_neuron']}
    out['identity'] = chk
    if chk['max_abs_diff'] > 1e-6:
        raise SystemExit(f'IDENTITY CHECK FAILED {tag}: {chk}')
    (OUT / 'tags' / dom).mkdir(parents=True, exist_ok=True)
    with open(OUT / 'tags' / dom / f'{tag}.pkl', 'wb') as f:
        pickle.dump(out, f)
    (BEX / dom).mkdir(parents=True, exist_ok=True)
    np.savez(BEX / dom / f'{tag}.npz', neurons=np.array(neurons), raw=raw.astype(np.float32), pon=pon.astype(np.float32))
    log(f'{dom} {tag}: identity {chk}; {len(neurons)} neurons; {time.time() - t0:.0f}s')


# ---------------------------------------------------------------------------------------------- metrics
def f1(tp, fp, fn):
    with np.errstate(invalid='ignore', divide='ignore'):
        return 2 * tp / (2 * tp + fp + fn)


def cell_metrics(C, Wv=None):
    """C = one tag's cell dict. Wv (R, nvid) video weights (unit multiplicities) or None (weights 1, R = 1).
    -> dict of (R,) arrays (direction means / per-video means as described in the report)."""
    nvid = len(C['n'])
    if Wv is None:
        Wv = np.ones((1, nvid))
    R = Wv.shape[0]
    n, T = C['n'], C['T']
    acc = {k: [] for k in ('F1', 'F1fire', 'r_soft', 'r_hard', 'rho_soft', 'rho_hard')}
    bias = {k: np.zeros((R, nvid)) for k in ('raw_soft', 'raw_hard', 'cal_soft', 'cal_hard', 'const')}
    absb = {k: np.zeros((R, nvid)) for k in bias}
    cnt = np.zeros(nvid)
    for d in C['dirs']:
        ho = d['half'] != d['sel_half']
        sel = ~ho
        W = Wv[:, ho]
        tp, fp = d['TP'][ho], d['FP'][ho]
        fn = T[ho] - tp
        acc['F1'].append(f1(W @ tp, W @ fp, W @ fn))
        if 'TPf' in d:
            acc['F1fire'].append(f1(W @ d['TPf'][ho], W @ d['FPf'][ho], W @ (T[ho] - d['TPf'][ho])))
        for kind, Sx in (('soft', d['S']), ('hard', d['H'])):
            with np.errstate(invalid='ignore', divide='ignore'):
                c = (Wv[:, sel] @ T[sel]) / (Wv[:, sel] @ Sx[sel])  # (R,) calibration on the selection half
            b_raw = np.broadcast_to(Sx[ho] - T[ho], (R, ho.sum()))
            b_cal = c[:, None] * Sx[ho][None] - T[ho][None]
            for key, b in ((f'raw_{kind}', b_raw), (f'cal_{kind}', b_cal)):
                bias[key][:, ho] += b
                absb[key][:, ho] += np.abs(b)
            acc[f'r_{kind}'].append(wpearson(Sx[ho], T[ho], W))
            if R == 1:
                acc[f'rho_{kind}'].append(np.array([spearman(Sx[ho], T[ho])]))
        with np.errstate(invalid='ignore', divide='ignore'):  # reference: constant rate of the selection half, no neuron
            rt = (Wv[:, sel] @ T[sel]) / (Wv[:, sel] @ n[sel])
        b_c = rt[:, None] * n[ho][None] - T[ho][None]
        bias['const'][:, ho] += b_c
        absb['const'][:, ho] += np.abs(b_c)
        cnt[ho] += 1
    out = {k: np.nanmean(np.stack(v), 0) for k, v in acc.items() if v}
    ok = cnt > 0
    for key in bias:  # per-video mean over the directions in which it is held out, then (weighted) mean over videos
        bv, av = bias[key][:, ok] / cnt[ok], absb[key][:, ok] / cnt[ok]
        Wo = Wv[:, ok]
        for unit, sc in (('s', None), ('frac', n[ok])):
            bb, aa = (bv, av) if sc is None else (bv / sc, av / sc)
            out[f'bias_{key}_{unit}'] = (Wo * bb).sum(1) / Wo.sum(1)
            out[f'mae_{key}_{unit}'] = (Wo * aa).sum(1) / Wo.sum(1)
            if R == 1:
                out[f'bias_{key}_{unit}_se'] = np.array([bb[0].std(ddof=1) / np.sqrt(ok.sum())])
                out[f'mae_{key}_{unit}_se'] = np.array([aa[0].std(ddof=1) / np.sqrt(ok.sum())])
    return out


def wpearson(x, y, W):
    sw = W.sum(1)
    mx, my = (W @ x) / sw, (W @ y) / sw
    cxy = (W @ (x * y)) / sw - mx * my
    vx = (W @ (x * x)) / sw - mx ** 2
    vy = (W @ (y * y)) / sw - my ** 2
    with np.errstate(invalid='ignore', divide='ignore'):
        return cxy / np.sqrt(vx * vy)


def spearman(x, y):
    from scipy.stats import spearmanr
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(spearmanr(x, y).statistic)


SECS = ('raw_soft', 'raw_hard', 'cal_soft', 'cal_hard')


def cmd_report(a):
    rng = np.random.default_rng(L.SEED + 7)
    REP, md = {}, []
    for dom in os.environ.get('EXT_DOMS', 'mice,ants').split(','):
        T_ = {g: {t: pickle.load(open(OUT / 'tags' / dom / f'{t}.pkl', 'rb')) for t in ts} for g, ts in TAGS.items()}
        any_ = T_['v1']['v1_s0']
        units = sorted({u for C in any_['cells'].values() for u in C['units']})
        Wu = np.bincount(rng.integers(0, len(units), (NB, len(units))).ravel()
                         + np.repeat(np.arange(NB), len(units)) * len(units), minlength=NB * len(units)
                         ).reshape(NB, len(units)).astype(np.float64)
        uix = {u: i for i, u in enumerate(units)}
        dt = DT[dom]
        D = {'identity_max_abs_diff': max(t['identity']['max_abs_diff'] for g in T_.values() for t in g.values()),
             'identity_n': sum(t['identity']['n'] for g in T_.values() for t in g.values()), 'cells': {}}
        for c in any_['cells']:
            b = c.split('|')[1]
            C0 = any_['cells'][c]
            fr = C0['T'] / C0['n']
            R = {'name': NAME[b], 'n_frames': C0['n_frames'], 'n_videos': len(C0['n']), 'n_units': len(set(C0['units'])),
                 'base_rate': C0['base_rate'], 'rate_per_video_mean': float(fr.mean()),
                 'rate_per_video_se': float(fr.std(ddof=1) / np.sqrt(len(fr))),
                 'true_time_per_video_s_mean': float((C0['T'] * dt).mean()),
                 'true_time_per_video_s_se': float((C0['T'] * dt).std(ddof=1) / np.sqrt(len(fr))),
                 'video_len_s': float(C0['n'].mean() * dt), 'series': (any_['elig_series'] or {}).get(b),
                 'models': {}}
            Wv = Wu[:, [uix[u] for u in C0['units']]]
            boot = {}
            for g, tags in T_.items():
                per, picks, bt = {}, {}, []
                for t, J in tags.items():
                    C = J['cells'][c]
                    assert C['videos'] == C0['videos']
                    m = {k: float(v[0]) for k, v in cell_metrics(C).items()}
                    for k in list(m):
                        if k.startswith(('bias_', 'mae_')) and k.endswith(('_s', '_s_se')):
                            m[k] *= dt
                    m['AUROC'], m['p@100'] = C['AUROC'], C['p@100']
                    per[t] = m
                    picks[t] = pd.Series([d['neuron'] for d in C['dirs']]).value_counts().to_dict()
                    mb = cell_metrics(C, Wv)
                    bt.append(np.stack([mb['F1'], mb['mae_cal_soft_frac'], mb['r_soft'], mb['mae_raw_soft_frac']]))
                boot[g] = np.stack(bt)  # (seeds, 4, NB)
                keys = per[next(iter(per))].keys()
                R['models'][g] = {'per_seed': per, 'picks': {t: {int(k): int(v) for k, v in p.items()} for t, p in picks.items()},
                                  'median': {k: float(np.nanmedian([per[t][k] for t in per])) for k in keys},
                                  'min': {k: float(np.nanmin([per[t][k] for t in per])) for k in keys},
                                  'max': {k: float(np.nanmax([per[t][k] for t in per])) for k in keys}}
            con = {}
            for i, M in enumerate(('F1', 'mae_cal_soft_frac', 'r_soft', 'mae_raw_soft_frac')):
                ax, ay = boot['v2'][:, i], boot['v1'][:, i]
                ix = rng.integers(0, len(ax), (NB, len(ax)))
                iy = rng.integers(0, len(ay), (NB, len(ay)))
                rr = np.arange(NB)[:, None]
                d = np.nanmean(ax[ix, rr], 1) - np.nanmean(ay[iy, rr], 1)
                pt = (np.nanmean([R['models']['v2']['per_seed'][t][M] for t in TAGS['v2']])
                      - np.nanmean([R['models']['v1']['per_seed'][t][M] for t in TAGS['v1']]))
                con[M] = {'diff': float(pt), 'ci': np.nanpercentile(d, [2.5, 97.5]).tolist()}
            con.update(stored_contrasts(dom, c, rng))
            R['v2_minus_v1'] = con
            D['cells'][c] = R
        REP[dom] = D
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'report.json').write_text(json.dumps(REP, indent=1, default=float))
    (OUT / 'tables.md').write_text(tables_md(REP))
    log('report.json + tables.md written')


def fm(R, g, k, nd=3, sc=1.0):
    m = R['models'][g]
    return f'{m["median"][k] * sc:.{nd}f} [{m["min"][k] * sc:.{nd}f}, {m["max"][k] * sc:.{nd}f}]'


def fc(R, k, nd=3, sc=1.0):
    r = R['v2_minus_v1'][k]
    return f'{r["diff"] * sc:+.{nd}f} [{r["ci"][0] * sc:+.{nd}f}, {r["ci"][1] * sc:+.{nd}f}]'


def tables_md(REP):
    md = []
    for dom, D in REP.items():
        md += [f'## {dom}', '', f'identity: AUROC and precision@100 recomputed here vs the stored ladder / precK values, '
               f'max abs diff {D["identity_max_abs_diff"]:.1e} over {D["identity_n"]} (model, behaviour) values', '',
               '### Ground truth', '', '| behaviour | series | videos (units) | frames | frames with behaviour (overall) | '
               'per-video rate, mean ± SE | true time per video, s, mean ± SE (video = ' + f'{next(iter(D["cells"].values()))["video_len_s"]:.0f} s) |',
               '|---|---|---|---|---|---|---|']
        for c, R in D['cells'].items():
            md.append(f'| {R["name"]} | {",".join(R["series"]) if R["series"] else "v1 (all pools)"} | {R["n_videos"]} ({R["n_units"]}) | '
                      f'{R["n_frames"]} | {R["base_rate"]:.4f} | {R["rate_per_video_mean"]:.4f} ± {R["rate_per_video_se"]:.4f} | '
                      f'{R["true_time_per_video_s_mean"]:.1f} ± {R["true_time_per_video_s_se"]:.1f} |')
        md += ['', '### Detection (seed median [min, max]; v2 - v1 = mean over seeds [95% CI])', '',
               '| behaviour | model | AUROC | precision@100 | per-frame F1 (P(on) > 0.5) |', '|---|---|---|---|---|']
        for c, R in D['cells'].items():
            for g in ('v1', 'v2'):
                md.append(f'| {R["name"]} | {g} | {fm(R, g, "AUROC")} | {fm(R, g, "p@100")} | {fm(R, g, "F1")} |')
            md.append(f'| {R["name"]} | **v2 - v1** | {fc(R, "AUROC")} | {fc(R, "p@100")} | {fc(R, "F1")} |')
        md += ['', '### Per-video total time (held-out videos; fraction of video length unless stated)', '',
               '| behaviour | model | raw bias, sum P(on), frac | raw MAE, sum P(on), frac | calibrated bias, frac | '
               'calibrated MAE, frac | calibrated MAE, s | Pearson r (sum P(on) vs true) | Spearman rho |', '|---|---|---|---|---|---|---|---|---|']
        md.insert(len(md) - 2, 'Reference: constant-rate predictor (selection-half mean rate x video length, no neuron) MAE, frac: '
                  + '; '.join(f'{R["name"]} {R["models"]["v1"]["median"]["mae_const_frac"]:.4f}' for R in D['cells'].values()))
        md.insert(len(md) - 2, '')
        for c, R in D['cells'].items():
            for g in ('v1', 'v2'):
                md.append(f'| {R["name"]} | {g} | {fm(R, g, "bias_raw_soft_frac")} | {fm(R, g, "mae_raw_soft_frac")} | '
                          f'{fm(R, g, "bias_cal_soft_frac")} | {fm(R, g, "mae_cal_soft_frac")} | {fm(R, g, "mae_cal_soft_s", 1)} | '
                          f'{fm(R, g, "r_soft")} | {fm(R, g, "rho_soft")} |')
            md.append(f'| {R["name"]} | **v2 - v1** | - | {fc(R, "mae_raw_soft_frac")} | - | {fc(R, "mae_cal_soft_frac")} | - | '
                      f'{fc(R, "r_soft")} | - |')
        md += ['', '### Hard on-frames (P(on) > 0.5) instead of sum P(on)', '',
               '| behaviour | model | raw bias, frac | raw MAE, frac | calibrated MAE, frac | Pearson r | Spearman rho |',
               '|---|---|---|---|---|---|---|']
        for c, R in D['cells'].items():
            for g in ('v1', 'v2'):
                md.append(f'| {R["name"]} | {g} | {fm(R, g, "bias_raw_hard_frac")} | {fm(R, g, "mae_raw_hard_frac")} | '
                          f'{fm(R, g, "mae_cal_hard_frac")} | {fm(R, g, "r_hard")} | {fm(R, g, "rho_hard")} |')
        md += ['', '### Selected neurons per seed (neuron #: number of the 12 directions that picked it)', '',
               '| behaviour | model | seed: picks |', '|---|---|---|']
        for c, R in D['cells'].items():
            for g in ('v1', 'v2'):
                pk = R['models'][g]['picks']
                md.append(f'| {R["name"]} | {g} | ' + '; '.join(f'{t}: ' + ', '.join(f'#{n}x{k}' for n, k in p.items())
                                                            for t, p in pk.items()) + ' |')
        md.append('')
    return '\n'.join(md)


def stored_contrasts(dom, c, rng):
    """AUROC and precision@100 V - v1 CIs from the EXISTING bootstrap arrays (ladder / precK, 2000 replicates)."""
    out = {}
    if dom == 'ants':
        J = json.loads((PK.OUTD / 'pooled_ants.json').read_text())
        sn = {'groom_any': 'any (= focal or)', 'Y2F': 'Y2F', 'B2F': 'B2F', 'single': 'focal single',
              'double': 'focal double'}[c.split('|')[1]]
        for M, k in (('AUROC', 'AUROC'), ('p@100', 'p@100')):
            r = J[sn]['contrasts']['V - v1'][k]
            out[M] = {'diff': r['diff'], 'ci': r['ci'], 'source': 'precK/pooled_ants.json'}
        return out
    zA = np.load(L.LB / 'boot_mice.npz')
    zP = np.load(PK.OUTD / 'boot_mice.npz')
    for M, z, i in (('AUROC', zA, 0), ('p@100', zP, 0)):
        ax = np.stack([z[f'{t}|gate|{c}'][i] for t in TAGS['v2']])
        ay = np.stack([z[f'{t}|gate|{c}'][i] for t in TAGS['v1']])
        n = ax.shape[1] - 1
        ix = rng.integers(0, len(ax), (n, len(ax)))
        iy = rng.integers(0, len(ay), (n, len(ay)))
        rr = np.arange(n)[:, None]
        d = np.nanmean(ax[:, 1:][ix, rr], 1) - np.nanmean(ay[:, 1:][iy, rr], 1)
        out[M] = {'diff': float(ax[:, 0].mean() - ay[:, 0].mean()), 'ci': np.nanpercentile(d, [2.5, 97.5]).tolist(),
                  'source': 'ladder boot_mice.npz (AUROC) / precK boot_mice.npz (p@100), pools + seeds resampled'}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=('tag', 'report'))
    p.add_argument('--dom')
    p.add_argument('--model')
    a = p.parse_args()
    {'tag': cmd_tag, 'report': cmd_report}[a.step](a)


if __name__ == '__main__':
    main()
