"""
ECI linear-probe ceiling: metrics with a paired unit bootstrap, tables and contrasts (scripts/eci/probe/probe.py preds).

Steps
  score --dom D --beh B   (CPU) every prediction vector of (D, B) on every evaluation set -> BEE/boot/<D>__<B>.npz
                          (point estimates + 1000 bootstrap replicates of AUROC / AP / precision@100 / per-video time r)
  table                   (CPU) -> OUT/tables.md, OUT/summary.json

Evaluation sets
  mice  <dir>|1fps : the test pools' frames on the 1 fps grid (the frames the v1 DINOv2 store holds) - every column
        <dir>|5fps : every test frame at 5 fps (v2-store columns only: d3, V, V_s*)
        dir sg2non = trained on the 11 SG pools, tested on the 12 non-SG pools; non2sg = the reverse.
  ants  A2B / B2A : v3 + v4 eval frames (1 fps, first 10 min) of the test half (probe trained on the other half)
Bootstrap: units = mice pools / ants videos (resampled within series x treatment), 1000 replicates, rng 20261104, the SAME draws
  for every column of an evaluation set (paired contrasts). Seeds of v2 (V_s1001-1003) are averaged, not resampled.
Metrics: AUROC (ties 0.5), AP (tied scores = one step), precision@100 (top-100 frames by score, ties pro rata),
  per-video time Pearson r (sum of the score per video vs true positive frames per video; for the probe the score is the
  predicted probability, for a single neuron its raw pooled code).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe as PR  # noqa: E402

BEE, OUT, REPO = PR.BEE, PR.OUT, PR.REPO
NB = 1000
MET = ('AUROC', 'AP', 'prec@100', 'time_r')


class Scorer:
    def __init__(self, s, y, vid, w_unit_of_frame):
        o = np.argsort(-s, kind='stable')
        self.ss, self.y = s[o], y[o].astype(np.float64)
        _, self.gid = np.unique(-self.ss, return_inverse=True)  # groups in descending score order
        self.ng = self.gid.max() + 1
        self.uf = w_unit_of_frame[o]
        self.vid = vid[o]
        self.s = s[o]

    def metrics(self, cu):
        w = cu[self.uf].astype(np.float64)
        wp = np.bincount(self.gid, w * self.y, self.ng)
        wa = np.bincount(self.gid, w, self.ng)
        wn = wa - wp
        Wp, Wn = wp.sum(), wn.sum()
        negbelow = Wn - np.cumsum(wn)  # negatives with a LOWER score (groups after this one)
        auc = float((wp * (negbelow + 0.5 * wn)).sum() / (Wp * Wn)) if Wp > 0 and Wn > 0 else np.nan
        ctp, call = np.cumsum(wp), np.cumsum(wa)
        ap = float((wp * ctp / np.maximum(call, 1e-12)).sum() / Wp) if Wp > 0 else np.nan
        # precision@100
        j = int(np.searchsorted(call, 100.0))
        if j >= self.ng:
            p100 = float(ctp[-1] / call[-1])
        else:
            before_a = call[j - 1] if j else 0.0
            before_p = ctp[j - 1] if j else 0.0
            p100 = float((before_p + (100.0 - before_a) * wp[j] / wa[j]) / 100.0)
        # per-video time r (weighted by unit multiplicity)
        vs = np.bincount(self.vid, self.s * 1.0, self.vid.max() + 1)
        vy = np.bincount(self.vid, self.y, self.vid.max() + 1)
        vw = np.bincount(self.vid, w, self.vid.max() + 1) / np.maximum(np.bincount(self.vid, None, self.vid.max() + 1), 1)
        m = vw > 0
        vs, vy, vw = vs[m], vy[m], vw[m]
        mx, my = np.average(vs, weights=vw), np.average(vy, weights=vw)
        cov = np.average((vs - mx) * (vy - my), weights=vw)
        r = float(cov / np.sqrt(np.average((vs - mx) ** 2, weights=vw) * np.average((vy - my) ** 2, weights=vw)))
        return np.array([auc, ap, p100, r])


def eval_sets(dom, beh, F):
    """-> {set: (frame mask, unit codes per frame, draws (NB, n_units) multiplicities)}"""
    rng = np.random.default_rng(20261104)
    out = {}
    if dom == 'mice':
        for d, tm in (('sg2non', F.nonsg.values), ('non2sg', F.sg.values)):
            units = pd.unique(F.unit.values[tm])
            uc = pd.Series(np.arange(len(units)), index=units)
            draws = np.stack([np.bincount(rng.integers(0, len(units), len(units)), minlength=len(units)) for _ in range(NB)])
            for fs, m in (('1fps', tm & F.grid1.values), ('5fps', tm)):
                u = np.full(len(F), -1)
                u[m] = uc.reindex(F.unit.values[m]).values
                out[f'{d}|{fs}'] = (m, u, draws)
    else:
        S = PR.ants_split().set_index('obs')
        h = F.obs.map(S.half).fillna('').values
        for d, tm in (('A2B', h == 'B'), ('B2A', h == 'A')):
            V = S[S.half == ('B' if d == 'A2B' else 'A')].reset_index()
            uc = pd.Series(np.arange(len(V)), index=V.obs.values)
            draws = np.zeros((NB, len(V)), np.int64)
            for _, g in V.groupby(['experiment', 'T']):
                ix = g.index.values
                for b in range(NB):
                    draws[b, ix] = np.bincount(rng.integers(0, len(ix), len(ix)), minlength=len(ix))
            u = np.full(len(F), -1)
            u[tm] = uc.reindex(F.obs.values[tm]).values
            out[d] = (tm, u, draws)
    return out


def cmd_score(a):
    dom, beh = a.dom, a.beh
    F = PR.frames(dom)
    y = F[f'y_{beh}'].values.astype(bool)
    vid_all = pd.factorize(F.obs.values)[0]
    sets = eval_sets(dom, beh, F)
    res = {}
    for rep in PR.reps(dom):
        p = BEE / 'preds' / dom / f'{rep.replace(":", "__")}__{beh}.npz'
        if not p.exists():
            print('MISSING', p, flush=True)
            continue
        Z = np.load(p)
        for kind in ('probe', 'neuron', 'perm'):
            s = Z[kind]
            for sn, (m, u, draws) in sets.items():
                mm = m & np.isfinite(s)
                if mm.sum() == 0:
                    continue
                if dom == 'mice' and sn.endswith('5fps') and PR.is_v1(rep):
                    continue
                assert mm.sum() == m.sum(), (rep, kind, sn, 'predictions missing on part of the set')
                sc = Scorer(s[mm].astype(np.float64), y[mm], pd.factorize(vid_all[mm])[0], u[mm])
                pt = sc.metrics(np.ones(draws.shape[1]))
                bt = np.stack([sc.metrics(draws[b]) for b in range(NB)])
                res[f'{rep}|{kind}|{sn}'] = (pt, bt, float(y[mm].mean()), int(mm.sum()), int(y[mm].sum()))
        print(rep, 'done', flush=True)
    od = BEE / 'boot'
    od.mkdir(parents=True, exist_ok=True)
    np.savez(od / f'{dom}__{beh}.npz', keys=np.array(list(res)), pt=np.stack([v[0] for v in res.values()]),
             bt=np.stack([v[1] for v in res.values()]), prev=np.array([v[2] for v in res.values()]),
             n=np.array([v[3] for v in res.values()]), npos=np.array([v[4] for v in res.values()]))


# ---------------------------------------------------------------------------------------------- tables
NAMES = {'nose_nose': 'nose-nose (nose_nose)', 'nn': 'nose-nose-mutual (nn)', 'np': 'nose-nose-passive (np)',
         'nt': 'nose-tail (nt)', 'groom_any': 'grooming-to-focal (any)', 'Y2F': 'Y2F', 'B2F': 'B2F',
         'single': 'focal single', 'double': 'focal double'}


def column(D, base, pl, kind, sn):
    """-> (pt (4,), bt (NB, 4)) of a column; V_s* = mean over seeds."""
    bases = [f'V_s{s}' for s in PR.V_SEEDS] if base == 'v2c' else [base]
    ks = [f'{b}:{pl}|{kind}|{sn}' for b in bases]
    if not all(k in D for k in ks):
        return None
    return np.mean([D[k][0] for k in ks], 0), np.mean([D[k][1] for k in ks], 0)


COLS = [('raw DINOv2', 'd2', 'probe'), ('raw DINOv3', 'd3', 'probe'), ('raw V input', 'V', 'probe'),
        ('v1 SAE full code', 'v1c', 'probe'), ('v2 SAE full code', 'v2c', 'probe'),
        ('v1 best neuron', 'v1c', 'neuron'), ('v2 best neuron', 'v2c', 'neuron')]
CONTRASTS = [('raw DINOv2 - v1 SAE full code', ('d2', 'probe'), ('v1c', 'probe')),
             ('raw V input - v2 SAE full code', ('V', 'probe'), ('v2c', 'probe')),
             ('raw DINOv3 - v2 SAE full code', ('d3', 'probe'), ('v2c', 'probe')),
             ('v1 SAE full code - v1 best neuron', ('v1c', 'probe'), ('v1c', 'neuron')),
             ('v2 SAE full code - v2 best neuron', ('v2c', 'probe'), ('v2c', 'neuron')),
             ('raw DINOv3 - raw DINOv2', ('d3', 'probe'), ('d2', 'probe')),
             ('raw V input - raw DINOv3', ('V', 'probe'), ('d3', 'probe'))]


def fmt(pt, bt, i, d=3):
    lo, hi = np.nanpercentile(bt[:, i], [2.5, 97.5])
    return f'{pt[i]:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]'


def cmd_table(a):
    md, summ = ['# ECI linear-probe ceiling - tables', '',
                'Every cell: point estimate [95% CI, bootstrap over test units, 1000 replicates]. v2 = mean over V_s1001-1003. '
                'Neuron columns: neuron pooling = max (the ECI frame-max readout); the full-code probes use the stated pooling. '
                'Probes are trained on the 1 fps grid of the training units for every column.', ''], {}
    for dom in ('mice', 'ants'):
        sets = ['sg2non|1fps', 'non2sg|1fps', 'sg2non|5fps', 'non2sg|5fps'] if dom == 'mice' else ['A2B', 'B2A']
        for sn in sets:
            for pl in ('mm', 'mean'):
                for mi, (mt, other) in enumerate(((('AUROC', 0), ('AP', 1)), (('prec@100', 2), ('time_r', 3)))):
                    md += [f'## {dom} | eval set {sn} | pooling {"mean||max" if pl == "mm" else "mean"} | '
                           f'{mt[0]} and {other[0]}', '',
                           '| behaviour | prevalence (positive frames / frames) | metric | ' + ' | '.join(c[0] for c in COLS) + ' |',
                           '|---|---|---|' + '---|' * len(COLS)]
                    for beh in PR.BEH[dom]:
                        p = BEE / 'boot' / f'{dom}__{beh}.npz'
                        if not p.exists():
                            continue
                        Z = np.load(p)
                        D = {k: (Z['pt'][i], Z['bt'][i], Z['prev'][i], Z['n'][i], Z['npos'][i]) for i, k in enumerate(Z['keys'])}
                        anyk = next((k for k in D if k.endswith(sn)), None)
                        if anyk is None:
                            continue
                        prev = f'{D[anyk][2]:.4f} ({D[anyk][4]:,} / {D[anyk][3]:,})'
                        for (mname, mi_) in (mt, other):
                            cells = []
                            for cname, base, kind in COLS:
                                c = column(D, base, 'max' if kind == 'neuron' else pl, kind, sn)
                                cells.append('n/a' if c is None else fmt(c[0], c[1], mi_))
                                if c is not None:
                                    summ[f'{dom}|{sn}|{pl}|{beh}|{cname}|{mname}'] = [float(c[0][mi_])] + \
                                        list(map(float, np.nanpercentile(c[1][:, mi_], [2.5, 97.5])))
                            md.append(f'| {NAMES[beh]} | {prev} | {mname} | ' + ' | '.join(cells) + ' |')
                    md.append('')
            # contrasts
            for pl in ('mm', 'mean'):
                md += [f'## {dom} | eval set {sn} | pooling {"mean||max" if pl == "mm" else "mean"} | contrasts (paired bootstrap)', '',
                       '| behaviour | contrast | AUROC difference | AP difference | precision@100 difference | time r difference |',
                       '|---|---|---|---|---|---|']
                for beh in PR.BEH[dom]:
                    p = BEE / 'boot' / f'{dom}__{beh}.npz'
                    if not p.exists():
                        continue
                    Z = np.load(p)
                    D = {k: (Z['pt'][i], Z['bt'][i]) for i, k in enumerate(Z['keys'])}
                    for cn, (b1, k1), (b2, k2) in CONTRASTS:
                        c1 = column(D, b1, 'max' if k1 == 'neuron' else pl, k1, sn)
                        c2 = column(D, b2, 'max' if k2 == 'neuron' else pl, k2, sn)
                        if c1 is None or c2 is None:
                            continue
                        dp, db = c1[0] - c2[0], c1[1] - c2[1]
                        md.append(f'| {NAMES[beh]} | {cn} | ' + ' | '.join(fmt(dp, db, i) for i in range(4)) + ' |')
                        for i, mname in enumerate(MET):
                            summ[f'{dom}|{sn}|{pl}|{beh}|{cn}|{mname}'] = [float(dp[i])] + list(map(float, np.nanpercentile(db[:, i], [2.5, 97.5])))
                md.append('')
            # extra rows: permutation control, neuron on mean pooling, full code on max pooling
            md += [f'## {dom} | eval set {sn} | controls and extra readouts (AUROC)', '',
                   '| behaviour | label-permuted probe d2:mean | label-permuted probe d3:mean | label-permuted probe V_s1001:mean||max | '
                   'v1 SAE full code, max pooling | v2 SAE full code, max pooling | v1 best neuron, mean pooling | v2 best neuron, mean pooling |',
                   '|---|---|---|---|---|---|---|---|']
            for beh in PR.BEH[dom]:
                p = BEE / 'boot' / f'{dom}__{beh}.npz'
                if not p.exists():
                    continue
                Z = np.load(p)
                D = {k: (Z['pt'][i], Z['bt'][i]) for i, k in enumerate(Z['keys'])}
                cells = []
                for k in ('d2:mean|perm', 'd3:mean|perm', 'V_s1001:mm|perm'):
                    kk = f'{k}|{sn}'
                    cells.append(fmt(D[kk][0], D[kk][1], 0) if kk in D else 'n/a')
                for base, pl, kind in (('v1c', 'max', 'probe'), ('v2c', 'max', 'probe'), ('v1c', 'mean', 'neuron'), ('v2c', 'mean', 'neuron')):
                    c = column(D, base, pl, kind, sn)
                    cells.append('n/a' if c is None else fmt(c[0], c[1], 0))
                md.append(f'| {NAMES[beh]} | ' + ' | '.join(cells) + ' |')
            md.append('')
    (OUT / 'tables.md').write_text('\n'.join(md) + '\n')
    (OUT / 'summary.json').write_text(json.dumps(summ, indent=0))
    print('tables written', len(summ), 'entries')


def cmd_checks(a):
    """identity of the new pooled max codes vs the stored codes the earlier evaluations used."""
    res = {}
    def cmp(mine, ref):
        mine, ref = mine.astype(np.float32), ref.astype(np.float32)
        return {'frames': int(len(mine)), 'fire_agree': float(((mine > 0) == (ref > 0)).mean()),
                'mean_abs_diff': float(np.abs(mine - ref).mean()), 'max_abs_diff': float(np.abs(mine - ref).max()),
                'share_entries_absdiff_gt_0.01': float((np.abs(mine - ref) > 0.01).mean())}
    F = PR.frames('mice')
    W = pd.read_parquet('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_2/full/frames.parquet', columns=['obs', 'k'])
    idx = pd.Series(np.arange(len(F)), index=pd.MultiIndex.from_arrays([F.obs.values, F.k.values]))
    wi = idx.reindex(pd.MultiIndex.from_arrays([W.obs.values, W.k.values])).values.astype(np.int64)
    rng = np.random.default_rng(0)
    sub = np.sort(rng.choice(len(W), 50000, replace=False))
    X, _ = PR.load_rep('mice', 'V_s1001:max', F)
    ref = np.load('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_2/full/cache/mice/V_s1001_raw.npy', mmap_mode='r')
    res['mice V_s1001 frame max vs v2_2 full cache (50,000 random atlas frames)'] = cmp(X[wi[sub]], np.asarray(ref[sub]))
    del X
    X, has = PR.load_rep('mice', 'v1c:max', F)
    TA = pd.read_parquet('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_prob/mice/tierA/frames.parquet')
    ta = pd.Series(np.arange(len(TA)), index=TA.row.values)
    m = has & (F.part.values == 'atlas')
    r = ta.reindex(F.row.values[m]).values.astype(np.int64)
    o = np.argsort(r)
    refv = np.asarray(np.load('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_prob/mice/tierA/v1.npy', mmap_mode='r')[r[o]])
    res['mice v1c frame max (stored fp16 tokens) vs stored deployed v1 s0 codes (tierA), 1 fps atlas frames'] = cmp(X[np.flatnonzero(m)[o]], refv)
    # best-neuron cross-check vs results/vision/eci_v2_2/full (V|auc, sg2non, P(on) readout, 324,000 non-SG atlas frames)
    from sklearn.metrics import roc_auc_score
    R = json.loads((REPO / 'results/vision/eci_v2_2/full/report.json').read_text())
    picks = R['sets']['sg2non']['picks']['V|auc']
    rep_auc = R['sets']['sg2non']['levels']['V|auc']
    nons = F.nonsg.values[wi]  # atlas rows (v22 order) that are non-SG
    for si, sd in enumerate(PR.V_SEEDS):
        key = f'V_s{sd}'
        X, _ = PR.load_rep('mice', f'{key}:max', F)
        pon = np.load(f'/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_2/full/cache/mice/{key}_pon.npy', mmap_mode='r')
        for beh in ('nose_nose', 'nt'):
            y = F[f'y_{beh}'].values[wi][nons]
            j = picks[key][0][beh]
            fit = json.loads((OUT / 'fits' / 'mice' / f'{key}__max__{beh}.json').read_text())['splits']['sg2non']
            res[f'neuron check {key} {beh}'] = {
                'v22_pick': j, 'v22_reported_auroc_pon': rep_auc[f'{beh}|auc']['seeds'][si],
                'recomputed_auroc_pon_v22_pick': float(roc_auc_score(y, np.asarray(pon[:, j])[nons])),
                'auroc_raw_framemax_v22_pick_this_store': float(roc_auc_score(y, X[wi[nons], j].astype(np.float32))),
                'this_probe_pick': fit['neuron'],
                'auroc_raw_framemax_this_pick_atlas_nonsg': float(roc_auc_score(y, X[wi[nons], fit['neuron']].astype(np.float32))),
                'frames': int(nons.sum())}
        del X
    G = PR.frames('ants')
    X, has = PR.load_rep('ants', 'v1c:max', G)
    ref = np.load('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_eval/ants/codes/v1_s0.npy')
    res['ants v1c frame max (CPU re-encode) vs eval.py v1_s0 codes'] = cmp(X[has], ref[has])
    res['ants frames with v1 features'] = int(has.sum())
    (OUT / 'checks.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('step')
    ap.add_argument('--dom')
    ap.add_argument('--beh')
    ap.add_argument('--task', type=int, default=None)
    a = ap.parse_args()
    if a.step == 'score' and a.task is not None:
        a.dom, a.beh = [(d, b) for d in ('mice', 'ants') for b in PR.BEH[d]][a.task]
    globals()[f'cmd_{a.step}'](a)
