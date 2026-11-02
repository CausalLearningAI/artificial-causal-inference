"""
ECI v2.1 AMENDMENT 4 (pre-registered in v21_train.py, commit 03dbf92): ants focal grooming labels, the v1 -> v2 ladder
and the paired video + seed bootstrap.

Steps
  cols --dom D --model KEY [--agg top10|mean]   (CPU) KEY = v1_s<0..10> | <arm>_s<seed> | PB
      readout (v1 raw frame max; v2 P(on) = per-neuron 3-state HMM fitted on the scoring spans, exactly v21_train.score),
      scorer on the gate cells (IDENTITY CHECK vs the stored score JSON, fail loud) and, for ants, on the focal cells
      single / double (or = groom_any is a gate cell). Saves BEE/ladder/<dom>/<KEY>[_<agg>].npz: the columns of every
      picked neuron + picks per (cell, direction) and the cell metrics -> OUT/ladder/<dom>/<KEY>[_<agg>].json
  boot --dom D [--agg top10]                       (CPU) the bootstrap of AMENDMENT 4 B -> OUT/ladder/boot_<dom>[_agg].json
"""
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
for p in ('', 'scripts/eci/v2', 'scripts/eci/v2_eval', 'scripts/eci/v2_prob', 'scripts/eci/v2_behav'):
    sys.path.insert(0, str(REPO / p))
os.environ.setdefault('V2_OUT', str(REPO / 'results/vision/eci_v2_0B'))
os.environ.setdefault('V2_BEE', '/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_0B')
import v21_train as T  # noqa: E402

LB = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_1/ladder')
LO = T.OUT / 'ladder'
FOCAL = ('single', 'double')
NB, SEED = 2000, 20261102
log = T.log


def frames(dom):
    import analysis as AN
    import eval as EV
    F = EV.frame_table(dom)
    SP = AN.splits(F.unit.values, EV.SPLITS)
    return F, SP


def focal_frame(F):
    """F with ONLY the focal y_ / v_ columns (single, double)."""
    G = F[[c for c in F.columns if not (c.startswith('y_') or c.startswith('v_'))]].copy()
    y = F.Y_Y2F.fillna(0).values > 0
    b = F.Y_B2F.fillna(0).values > 0
    v = F.Y_Y2F.notna().values & F.Y_B2F.notna().values
    G['y_single'], G['v_single'] = y ^ b, v
    G['y_double'], G['v_double'] = y & b, v
    return G


def readout(dom, key, agg, F):
    """-> (n_F, W) float32 readout matrix aligned to F (and the stored score json for the identity check, or None)."""
    import eval as EV
    if key.startswith('v1_'):
        X, _ = EV.load_dict(dom, key, F)
        ref = json.loads((REPO / f'results/vision/eci_v2_0B/gate/{dom}_{key}.json').read_text())
        return np.asarray(X, np.float32), {'cells': ref}
    from multiprocessing import get_context
    from src.eci.v2 import hmm as HM
    import v20 as V20
    arm = key.split('_s')[0]
    od = T.BEE / dom / 'codes'
    sfx = '_n2' if arm == 'SN2' else ''
    R = pd.read_parquet(od / f'eval_rows{sfx}.parquet')
    E = T.frame_sets(dom)['eval']
    rix = pd.Series(np.arange(len(R)), index=pd.MultiIndex.from_frame(R)).reindex(pd.MultiIndex.from_frame(E)).values
    assert not np.isnan(rix).any()
    rix = rix.astype(np.int64)
    st = np.r_[0, np.flatnonzero((R.obs.values[1:] != R.obs.values[:-1]) | (R.series.values[1:] != R.series.values[:-1])) + 1, len(R)]
    if key == 'PB':
        raw = T.ref_codes(dom, R)
        jf = T.OUT / dom / f'PB_s{T.REF_SEED[dom]}.json'
    else:
        raw = np.load(od / (f'{key}_eval_{agg}.npy' if agg else f'{key}_eval.npy')).astype(np.float32)
        jf = T.OUT / dom / (f'{key}_{agg}.json' if agg else f'{key}.json')
    Xt = np.ascontiguousarray(raw.T)
    V20._G.update(Xc=Xt, sc=st.astype(np.int64))
    with get_context('fork').Pool(int(os.environ.get('SLURM_CPUS_PER_TASK', 8))) as pool:
        prm = [None] * Xt.shape[0]
        for j, p in pool.imap_unordered(V20._fit_one, range(Xt.shape[0]), chunksize=16):
            prm[j] = p
    pon = np.zeros_like(raw)
    for j in range(Xt.shape[0]):
        pon[:, j] = HM.apply(Xt[j], st, prm[j])[0]
    ref = json.loads(jf.read_text()) if jf.exists() else None
    return pon[rix], ({'cells': ref['hmm']['cells'], 'gate': ref['hmm']['gate']} if ref else None)


def score2(F, X, use, SP):
    """v20_gate.score with the nAP-picked neuron of every direction recorded too ('nAP_neurons'); the cell values are
    eval.score_cells' (same loop, same kernels)."""
    import analysis as AN
    import eval as EV
    import v20_gate as G
    out = {}
    vcode = pd.factorize(F.obs.values)[0].astype(np.int64)
    nvid = int(vcode.max()) + 1
    starts = AN.seq_starts(F)
    for ser in F.series.unique():
        for b in [c[2:] for c in F.columns if c.startswith('y_')]:
            vmask = F[f'v_{b}'].values & (F.series.values == ser) & use
            if vmask.sum() == 0 or F[f'y_{b}'].values[vmask].sum() == 0:
                continue
            idx_c, st_c = AN.sub_starts(vmask, starts)
            yc = F[f'y_{b}'].values[idx_c]
            Xc0 = X[idx_c]
            elig = np.flatnonzero((Xc0 > 0).mean(0) >= AN.ELIG)
            if len(elig) == 0:
                continue
            Xc = np.asfortranarray(Xc0[:, elig])
            vc = vcode[idx_c]
            per = {}
            halves = {nm: np.array([SP[nm][u] for u in F.unit.values[idx_c]]) for nm in EV.SPLITS}
            for nm in EV.SPLITS:
                for h in (0, 1):
                    sel = np.flatnonzero(halves[nm] == h)
                    if len(sel) == 0 or yc[sel].sum() == 0 or yc[sel].all():
                        per[(nm, h)] = None
                        continue
                    _, st_h = AN.sub_starts(halves[nm] == h, st_c)
                    per[(nm, h)] = AN.metrics_cols(Xc, sel.astype(np.int64), yc[sel], st_h, vc[sel], nvid)[0]
            cf = {M: [] for M in EV.MET}
            pick = {M: [] for M in EV.MET}
            for nm in EV.SPLITS:
                for h in (0, 1):
                    A_, B_ = per[(nm, h)], per[(nm, 1 - h)]
                    if A_ is None or B_ is None:
                        continue
                    for mi, M in enumerate(EV.MET):
                        col = A_[:, mi]
                        if not np.isfinite(col).any():
                            continue
                        q = int(np.nanargmax(col))
                        cf[M].append(float(B_[q, mi]))
                        pick[M].append([nm, h, int(elig[q])])
            out[f'{ser}|{b}'] = {'n_frames': int(len(idx_c)), 'base_rate': float(yc.mean()), 'n_elig': int(len(elig)),
                                 'n_dirs': len(cf['AUROC']), **{M: (float(np.mean(v)) if v else None) for M, v in cf.items()},
                                 'AUROC_dirs': cf['AUROC'], 'AUROC_neurons': pick['AUROC'], 'nAP_neurons': pick['nAP']}
    t = G.top1(F, X, use, SP, out)
    for c in out:
        out[c].update({f'top1_{k}': v for k, v in t[c].items()})
        js = pd.Series([x[2] for x in out[c]['AUROC_neurons']]).value_counts()
        out[c]['best_neuron'] = int(js.index[0]) if len(js) else None
    return out


def cmd_cols(a):
    import v20_gate as G
    dom, key, agg = a.dom, a.model, a.agg
    tag = f'{key}_{agg}' if agg else key
    t0 = time.time()
    F, SP = frames(dom)
    X, ref = readout(dom, key, agg, F)
    use = np.ones(len(F), bool)
    sc = {'gate': score2(F, X, use, SP)}
    if dom == 'ants':
        sc['focal'] = score2(focal_frame(F), X, use, SP)
    # identity check vs the stored score json (gate cells): AUROC, nAP, top-1% precision
    chk = {'n': 0, 'max_abs_diff': 0.0}
    if ref is None:
        chk['note'] = 'no stored score json (fresh model scored here first)'
    else:
        for c, r in sc['gate'].items():
            if c not in ref['cells'] or c.split('|')[1] in G.ONLID:
                continue
            pairs = [(r['AUROC'], ref['cells'][c]['AUROC']), (r['nAP'], ref['cells'][c]['nAP'])]
            if 'gate' in ref and c in ref['gate']:
                pairs.append((r['top1_prec'], ref['gate'][c]['v2_top1_prec']))
            elif 'top1_prec' in ref['cells'][c]:
                pairs.append((r['top1_prec'], ref['cells'][c]['top1_prec']))
            for x, y in pairs:
                if x is None or y is None:
                    continue
                chk['n'] += 1
                chk['max_abs_diff'] = max(chk['max_abs_diff'], abs(x - y))
        if chk['max_abs_diff'] > 1e-6:
            raise SystemExit(f'IDENTITY CHECK FAILED {tag}: {chk}')
    js = sorted({p[2] for part in sc.values() for r in part.values() for p in r['AUROC_neurons'] + r['nAP_neurons']})
    LB.joinpath(dom).mkdir(parents=True, exist_ok=True)
    np.savez(LB / dom / f'{tag}.npz', cols=X[:, js].astype(np.float32), neurons=np.array(js, np.int64))
    keep = ('n_frames', 'base_rate', 'n_dirs', 'AUROC', 'nAP', 'F1max', 'Rt', 'AUROC_neurons', 'nAP_neurons', 'top1_prec', 'top1_lift',
            'top1_base', 'best_neuron')
    out = {'key': key, 'agg': agg, 'identity': chk,
           **{part: {c: {k: v for k, v in r.items() if k in keep} for c, r in d.items()} for part, d in sc.items()}}
    LO.joinpath(dom).mkdir(parents=True, exist_ok=True)
    (LO / dom / f'{tag}.json').write_text(json.dumps(out, default=float))
    log(f'{dom} {tag}: identity {chk}; {len(js)} picked neurons ({time.time() - t0:.0f}s)')


# ---------------------------------------------------------------------------------------------- bootstrap
def cell_dirs(F, SP, cells_res, part):
    """per cell: list of (held-out frame idx (into F), AUROC-picked neuron, nAP-picked neuron, tie-break vector)."""
    out = {}
    rng = np.random.default_rng(0)  # v20_gate.top1 draws one tie-break vector per cell, in this order
    for cell, r in cells_res.items():
        ser, b = cell.split('|')
        vm = F[f'v_{b}'].values & (F.series.values == ser)
        idx = np.flatnonzero(vm)
        tb = rng.random(len(idx))
        units = F.unit.values[idx]
        L = []
        npk = {(nm, h): j for nm, h, j in r['nAP_neurons']}
        assert len(npk) == len(r['AUROC_neurons']), cell
        for nm, h, j in r['AUROC_neurons']:
            half = np.array([SP[nm][u] for u in units])
            o = np.flatnonzero(half == 1 - h)
            L.append((idx[o], int(j), int(npk[(nm, h)]), tb[o]))
        out[cell] = L
    return out


def wmetrics(s, y, u, tb, Wt, cnt_u):
    """weighted AUROC / nAP / top-1% precision for every replicate. s scores (n,), y bool, u unit index (n,), tb tie-break,
    Wt (U, R) unit multiplicities, cnt_u frames per unit on these frames -> (3, R)."""
    from scipy import sparse
    n = len(s)
    U, R = Wt.shape
    s = s.astype(np.float64)
    o = np.argsort(-s, kind='mergesort')
    so = s[o]
    gid = np.r_[0, np.cumsum(so[1:] != so[:-1])]
    G = int(gid[-1]) + 1
    yo, uo = y[o], u[o]
    Pp = sparse.csr_matrix((np.ones(int(yo.sum())), (gid[yo], uo[yo])), shape=(G, U))
    Pn = sparse.csr_matrix((np.ones(int((~yo).sum())), (gid[~yo], uo[~yo])), shape=(G, U))
    wp, wn = np.asarray(Pp @ Wt), np.asarray(Pn @ Wt)
    P, N = wp.sum(0), wn.sum(0)
    ctp, cfp = np.cumsum(wp, 0), np.cumsum(wn, 0)
    with np.errstate(invalid='ignore', divide='ignore'):
        auc = (wp * (N - (cfp - wn) - wn) + 0.5 * wp * wn).sum(0) / (P * N)
        ap = (np.where(wp > 0, wp / P * ctp / (ctp + cfp), 0.0)).sum(0)
        pi = P / (P + N)
        nap = (ap - pi) / (1 - pi)
        bad = (P == 0) | (N == 0)
        auc[bad], nap[bad] = np.nan, np.nan
    # top-1%: scorer order lexsort((tb, -s)), cut = ceil(0.01 x total weight), partial weight at the boundary frame
    o2 = np.lexsort((tb, -s))
    tot = cnt_u @ Wt
    cut = np.maximum(1, np.ceil(0.01 * tot - 1e-9))
    K = min(n, int(0.05 * n) + 50)
    while True:
        wf = Wt[u[o2[:K]]]
        cum = np.cumsum(wf, 0)
        if (cum[-1] >= cut).all() or K == n:
            break
        K = min(n, 2 * K)
    inc = np.clip(cut - (cum - wf), 0, wf)
    prec = (inc * y[o2[:K]][:, None]).sum(0) / cut
    prec[tot == 0] = np.nan
    return np.stack([auc, nap, prec])


def model_boot(args):
    """-> {cell: (3, NB + 1) array}; column 0 = all weights 1 (identity), 1.. = the replicates."""
    dom, tag, part, Wser = args
    F, SP = frames(dom)
    if part == 'focal':
        F = focal_frame(F)
    J = json.loads((LO / dom / f'{tag}.json').read_text())[part]
    z = np.load(LB / dom / f'{tag}.npz')
    col = {int(j): i for i, j in enumerate(z['neurons'])}
    C = z['cols']
    D = cell_dirs(F, SP, J, part)
    out = {}
    for cell, L in D.items():
        ser = cell.split('|')[0]
        units, W = Wser[ser]
        uix = pd.Series(np.arange(len(units)), index=units)
        Wt = np.concatenate([np.ones((len(units), 1)), W.T], 1)
        y_all = F[f'y_{cell.split("|")[1]}'].values
        acc = []
        for fr, j, jn, tb in L:
            u = uix.reindex(F.unit.values[fr]).values.astype(np.int64)
            cnt_u = np.bincount(u, minlength=len(units)).astype(np.float64)
            m = wmetrics(C[fr, col[j]], y_all[fr], u, tb, Wt, cnt_u)
            if jn != j:
                m[1] = wmetrics(C[fr, col[jn]], y_all[fr], u, tb, Wt, cnt_u)[1]
            acc.append(m)
        out[cell] = np.nanmean(np.stack(acc), 0) if acc else None
    return tag, part, out


def cmd_boot(a):
    from multiprocessing import get_context
    dom = a.dom
    F, SP = frames(dom)
    rng = np.random.default_rng(SEED)
    Wser = {}
    for ser in sorted(F.series.unique()):
        units = np.array(sorted(set(F.unit.values[F.series.values == ser])))
        W = np.stack([np.bincount(rng.integers(0, len(units), len(units)), minlength=len(units)) for _ in range(NB)])
        Wser[ser] = (units, W.astype(np.float64))
    tags = sorted(p.stem for p in (LO / dom).glob('*.json'))
    parts = ['gate'] + (['focal'] if dom == 'ants' else [])
    jobs = [(dom, t, p, Wser) for t in tags for p in parts]
    res = {}
    with get_context('fork').Pool(int(os.environ.get('SLURM_CPUS_PER_TASK', 8))) as pool:
        for tag, part, out in pool.imap_unordered(model_boot, jobs):
            res.setdefault(tag, {})[part] = out
            log(f'boot {dom} {tag} {part} done')
    # identity: column 0 vs the cols-step metrics
    mx = 0.0
    for tag, d in res.items():
        J = json.loads((LO / dom / f'{tag}.json').read_text())
        for part, cells in d.items():
            for c, m in cells.items():
                if m is None:
                    continue
                for i, M in enumerate(('AUROC', 'nAP', 'top1_prec')):
                    ref = J[part][c][M]
                    if ref is not None:
                        mx = max(mx, abs(m[i, 0] - ref))
    log(f'boot identity (weights 1 vs scorer): max abs diff {mx:.2e}')
    if mx > 1e-6:
        raise SystemExit('BOOT IDENTITY CHECK FAILED')
    np.savez_compressed(LB / f'boot_{dom}.npz', **{f'{t}|{p}|{c}': m for t, d in res.items() for p, cs in d.items()
                                                     for c, m in cs.items() if m is not None})
    (LO / f'boot_{dom}_meta.json').write_text(json.dumps({'NB': NB, 'seed': SEED, 'identity_max_abs_diff': mx,
                                                          'models': sorted(res)}, indent=1))
    log(f'boot {dom}: {len(res)} models saved')


# ---------------------------------------------------------------------------------------------- report
RUNGS = {'R0 v1': [f'v1_s{i}' for i in range(11)], 'R1 PB': ['PB'], 'R1 SN2': [f'SN2_s{i}' for i in (1001, 1002, 1003)],
         'R2 S': [f'S_s{i}' for i in (1001, 1002, 1003)], 'R3 V': [f'V_s{i}' for i in (1001, 1002, 1003)],
         'R3 W': [f'W_s{i}' for i in (1001, 1002, 1003)], 'ref B': [f'B_s{i}' for i in (1001, 1002, 1003)],
         'S fresh': [f'S_s{i}' for i in (1004, 1005, 1006)], 'V fresh': [f'V_s{i}' for i in (1004, 1005, 1006)],
         'W fresh': [f'W_s{i}' for i in (1004, 1005, 1006)]}
NO_AGG = ('R0 v1', 'R1 PB')  # stored frame-max codes only (AMENDMENT 4 R)
CONTRASTS = [('R1 PB', 'R0 v1'), ('R1 SN2', 'R0 v1'), ('R2 S', 'R0 v1'), ('R3 V', 'R0 v1'), ('R3 W', 'R0 v1'),
             ('ref B', 'R0 v1'), ('R2 S', 'R1 SN2'), ('R2 S', 'R1 PB'), ('R3 V', 'R2 S'), ('R3 W', 'R2 S'),
             ('R3 W', 'R3 V'), ('S fresh', 'R0 v1'), ('V fresh', 'R0 v1'), ('W fresh', 'R0 v1'),
             ('V fresh', 'S fresh'), ('W fresh', 'S fresh'), ('W fresh', 'V fresh')]
MN = ('AUROC', 'nAP', 'top1_prec')


def focal_elig():
    import eval as EV
    F, SP = frames('ants')
    G = focal_frame(F)
    EV.BEH['ants'] = {'single': [], 'double': []}
    return {k: v['eligible'] for k, v in EV.row_counts('ants', G, SP).items()}


def cmd_report(a):
    import v20_gate as G
    agg = a.agg
    out, md = {}, []
    for dom in ('mice', 'ants'):
        if not (LB / f'boot_{dom}.npz').exists():
            log(f'no bootstrap for {dom} yet; skipped')
            continue
        S0 = json.loads((G.OUT / f'{dom}_scores.json').read_text())
        gcells = [c for c, r in S0['eligibility'].items() if r['eligible'] and c.split('|')[1] not in G.ONLID]
        fel = focal_elig() if dom == 'ants' else {}
        fcells = {'single': [c for c, e in fel.items() if e and c.endswith('|single')],
                  'double': [c for c, e in fel.items() if e and c.endswith('|double')],
                  'or': [c for c in gcells if c.endswith('|groom_any')]}
        z = np.load(LB / f'boot_{dom}.npz')
        tg = lambda k, rung: k if (not agg or rung in NO_AGG) else f'{k}_{agg}'  # noqa: E731
        dm = {}  # (rung, key) -> (3, NB+1) domain means over gate cells; plus focal
        rows = {}
        for rung, keys in RUNGS.items():
            ks = [k for k in keys if all(f'{tg(k, rung)}|gate|{c}' in z for c in gcells)]
            if not ks:
                continue
            per = []
            for k in ks:
                t = tg(k, rung)
                per.append(np.mean([z[f'{t}|gate|{c}'] for c in gcells], 0))
            dm[rung] = np.stack(per)  # (n_seeds, 3, NB+1)
            # descriptive columns from the score / ladder jsons
            R = {'models': [tg(k, rung) for k in ks], 'n': len(ks)}
            pas, f1, dyn, dist, foc = [], [], [], [], {f: [] for f in fcells}
            for k in ks:
                t = tg(k, rung)
                J = json.loads((LO / dom / f'{t}.json').read_text())
                f1.append(float(np.mean([J['gate'][c]['F1max'] for c in gcells])))
                for f, cs in fcells.items():
                    if cs:
                        src = 'gate' if f == 'or' else 'focal'
                        foc[f].append(float(np.mean([J[src][c]['AUROC'] for c in cs])))
                if dom == 'mice':
                    dist.append(J['gate']['v1|nn']['best_neuron'] != J['gate']['v1|nt']['best_neuron'])
                if k.startswith('v1_'):
                    continue
                sj = T.OUT / dom / (f'PB_s{T.REF_SEED[dom]}.json' if k == 'PB' else f'{t}.json')
                SJ = json.loads(sj.read_text())
                pas.append(sum(r['PASS_final'] for c, r in SJ['hmm']['gate'].items() if c in gcells))
                dyn.append(SJ['dyn256']['share'])
            obs = dm[rung][:, :, 0]
            R.update({'mean_AUROC': obs[:, 0].tolist(), 'mean_nAP': obs[:, 1].tolist(), 'mean_top1_prec': obs[:, 2].tolist(),
                      'mean_F1max': f1, 'gate_pass': pas, 'dyn256': dyn, 'distinct_nn_nt': dist,
                      **{f'focal_{f}_AUROC': v for f, v in foc.items()}})
            rows[rung] = R
        rng = np.random.default_rng(SEED + 1)
        con = {}
        for x, y in CONTRASTS:
            if x not in dm or y not in dm:
                continue
            r = {}
            for mi, M in enumerate(MN):
                ax, ay = dm[x][:, mi, 1:], dm[y][:, mi, 1:]
                pt = float(dm[x][:, mi, 0].mean() - dm[y][:, mi, 0].mean())
                fx = ax.mean(0) - ay.mean(0)  # seeds fixed
                ix = rng.integers(0, len(ax), (NB, len(ax)))
                iy = rng.integers(0, len(ay), (NB, len(ay)))
                rr = np.arange(NB)[:, None]
                sx = np.nanmean(ax[ix, rr], 1)
                sy = np.nanmean(ay[iy, rr], 1)
                d = sx - sy
                r[M] = {'diff': pt, 'ci_units_seeds': np.nanpercentile(d, [2.5, 97.5]).tolist(),
                        'ci_units_only': np.nanpercentile(fx, [2.5, 97.5]).tolist()}
            con[f'{x} - {y}'] = r
        # focal forms (AMENDMENT 4 F, reported cells): same bootstrap on the mean over each form's eligible cells
        fcon = {}
        for f, cs in fcells.items():
            if not cs:
                continue
            src = 'gate' if f == 'or' else 'focal'
            dmf = {}
            for rung, keys in RUNGS.items():
                ks = [k for k in keys if all(f'{tg(k, rung)}|{src}|{c}' in z for c in cs)]
                if ks:
                    dmf[rung] = np.stack([np.mean([z[f'{tg(k, rung)}|{src}|{c}'] for c in cs], 0) for k in ks])
            for x, y in CONTRASTS:
                if x not in dmf or y not in dmf:
                    continue
                r = {}
                for mi, M in enumerate(MN):
                    ax, ay = dmf[x][:, mi, 1:], dmf[y][:, mi, 1:]
                    ix = rng.integers(0, len(ax), (NB, len(ax)))
                    iy = rng.integers(0, len(ay), (NB, len(ay)))
                    rr = np.arange(NB)[:, None]
                    d = np.nanmean(ax[ix, rr], 1) - np.nanmean(ay[iy, rr], 1)
                    r[M] = {'diff': float(dmf[x][:, mi, 0].mean() - dmf[y][:, mi, 0].mean()),
                            'ci_units_seeds': np.nanpercentile(d, [2.5, 97.5]).tolist(),
                            'ci_units_only': np.nanpercentile(ax.mean(0) - ay.mean(0), [2.5, 97.5]).tolist()}
                fcon.setdefault(f, {})[f'{x} - {y}'] = r
            fcon.setdefault(f, {})['_levels'] = {rg: [float(np.median(v[:, 0, 0])), float(np.median(v[:, 2, 0]))] for rg, v in dmf.items()}
        out[dom] = {'gate_cells': gcells, 'focal_cells': fcells, 'rows': rows, 'contrasts': con, 'focal_contrasts': fcon}
        md.append(f'## {dom}' + (f' (readout {agg} for the v2 token-code arms; v1 / PB frame max)' if agg else ' (readout frame max)'))
        fmr = lambda v: '-' if not v else (f'{np.median(v):.3f} [{min(v):.3f}, {max(v):.3f}]' if len(v) > 1 else f'{v[0]:.3f}')  # noqa
        hd = ('| rung | n | mean AUROC P(on) | top-1% prec P(on) | nAP P(on) | F1max P(on) | gate passes / seed | dyn share @256 |'
              + (' distinct nn/nt | ' if dom == 'mice' else ' focal single AUROC | focal double AUROC | focal or AUROC |'))
        md += ['', hd, '|' + '---|' * (hd.count('|') - 1)]
        for rung, R in rows.items():
            tail = (f' {sum(R["distinct_nn_nt"])}/{len(R["distinct_nn_nt"])} |' if dom == 'mice' else
                    f' {fmr(R.get("focal_single_AUROC"))} | {fmr(R.get("focal_double_AUROC"))} | {fmr(R.get("focal_or_AUROC"))} |')
            md.append(f'| {rung} | {R["n"]} | {fmr(R["mean_AUROC"])} | {fmr(R["mean_top1_prec"])} | {fmr(R["mean_nAP"])} | '
                      f'{fmr(R["mean_F1max"])} | {R["gate_pass"] or "-"} | {fmr(R["dyn256"])} |' + tail)
        md += ['', '| contrast | d mean AUROC [95% CI videos+seeds] (videos only) | d top-1% prec | d nAP |', '|---|---|---|---|']
        for k, r in con.items():
            cell = lambda m: f'{r[m]["diff"]:+.3f} [{r[m]["ci_units_seeds"][0]:+.3f}, {r[m]["ci_units_seeds"][1]:+.3f}] ([{r[m]["ci_units_only"][0]:+.3f}, {r[m]["ci_units_only"][1]:+.3f}])'  # noqa
            md.append(f'| {k} | {cell("AUROC")} | {cell("top1_prec")} | {cell("nAP")} |')
        for f, cc in fcon.items():
            md += ['', f'focal {f} (cells {fcells[f]}): levels AUROC / top-1% prec P(on), median over seeds: '
                   + ', '.join(f'{k} {v[0]:.3f} / {v[1]:.3f}' for k, v in cc['_levels'].items()), '',
                   f'| contrast (focal {f}) | d mean AUROC [95% CI videos+seeds] | d top-1% prec |', '|---|---|---|']
            for k, r in cc.items():
                if k == '_levels':
                    continue
                md.append(f'| {k} | ' + ' | '.join(f'{r[m]["diff"]:+.3f} [{r[m]["ci_units_seeds"][0]:+.3f}, {r[m]["ci_units_seeds"][1]:+.3f}]'
                                                  for m in ('AUROC', 'top1_prec')) + ' |')
        md.append('')
    tag = f'_{agg}' if agg else ''
    (LO / f'ladder{tag}.json').write_text(json.dumps(out, indent=1, default=float))
    (LO / f'ladder{tag}.md').write_text('\n'.join(md))
    print('\n'.join(md))


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=('cols', 'boot', 'report'))
    p.add_argument('--dom', choices=('mice', 'ants'))
    p.add_argument('--model', default='')
    p.add_argument('--agg', default='')
    a = p.parse_args()
    {'cols': cmd_cols, 'boot': cmd_boot, 'report': cmd_report}[a.step](a)


if __name__ == '__main__':
    main()
