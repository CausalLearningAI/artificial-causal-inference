"""precision@K (K = 100, 1000) on top of the v2.1 ladder (AMENDMENT 4) artifacts; nothing existing is modified.

Same per-(split, half) held-out directions, same AUROC-picked neuron per direction, same tie-break vector and same
bootstrap weights (rng SEED, pools/videos) as v21_ladder.py. K is FIXED per cell per replicate (cut = min(K, replicate
frame total)), the 1% cut of the ladder is not used for K. Identity: top-1% precision recomputed here vs the stored value.
steps: boot --dom D (writes precK/boot_D.npz, precK/cells_D.json)   report (writes precK/precK.json, precK.md)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v21_ladder as L  # noqa: E402

REPO = L.REPO
OUTD = REPO / 'results/vision/eci_v2_1/precK'
KS = (100, 1000)
NB, SEED = L.NB, L.SEED
NAMES = {'nn': 'nose-nose-mutual (nn)', 'np': 'nose-nose-passive (np)', 'nose_nose': 'nose-nose (nose_nose)', 'nt': 'nose-tail (nt)'}


def nm(c):
    a, b = c.split('|')
    return NAMES.get(b, b) if a == 'v1' else c


TAGS = [f'v1_s{i}' for i in range(11)] + [f'{a}_s{s}' for a in 'SV' for s in range(1001, 1007)]


def wprec(s, y, u, tb, Wt, cnt_u, cuts):
    """weighted precision of the top-`cut` frames (scorer order lexsort((tb, -s)), partial weight at the boundary).
    cuts: (R,) arrays list. Returns list of (R,) precisions."""
    n = len(s)
    o2 = np.lexsort((tb, -s.astype(np.float64)))
    tot = cnt_u @ Wt
    K = min(n, 4096)
    cl = [np.minimum(c, tot) for c in cuts]
    while True:
        wf = Wt[u[o2[:K]]]
        cum = np.cumsum(wf, 0)
        if all((cum[-1] >= c - 1e-9).all() for c in cl) or K == n:
            break
        K = min(n, 2 * K)
    yy = y[o2[:K]][:, None]
    out = []
    for c in cl:
        inc = np.clip(c - (cum - wf), 0, wf)
        with np.errstate(invalid='ignore', divide='ignore'):
            p = (inc * yy).sum(0) / c
        p[tot == 0] = np.nan
        out.append(p)
    return out


def direct(c, y, tb, cut):
    top = np.lexsort((tb, -c))[:cut]
    return float(y[top].mean())


def job(args):
    dom, tag, part, Wser, gcells = args
    F, SP = L.frames(dom)
    if part == 'focal':
        F = L.focal_frame(F)
    J = json.loads((L.LO / dom / f'{tag}.json').read_text())[part]
    z = np.load(L.LB / dom / f'{tag}.npz')
    col = {int(j): i for i, j in enumerate(z['neurons'])}
    C = z['cols']
    D = L.cell_dirs(F, SP, J, part)
    res, info, idchk = {}, {}, 0.0
    for cell, Ld in D.items():  # tie-break vectors were drawn over ALL cells of the json, in order (cell_dirs); filter after
        if cell not in gcells:
            continue
        ser, b = cell.split('|')
        units, W = Wser[ser]
        uix = pd.Series(np.arange(len(units)), index=units)
        Wt = np.concatenate([np.ones((len(units), 1)), W.T], 1)
        y_all = F[f'y_{b}'].values
        acc, ns, t1, n_short = [], [], [], 0
        for fr, j, jn, tb in Ld:
            y = y_all[fr]
            if len(fr) == 0 or y.sum() == 0:
                continue  # skipped by v20_gate.top1 as well
            sc = np.asarray(C[fr, col[j]], np.float64)
            u = uix.reindex(F.unit.values[fr]).values.astype(np.int64)
            cnt_u = np.bincount(u, minlength=len(units)).astype(np.float64)
            m = np.stack(wprec(sc, y, u, tb, Wt, cnt_u, [np.full(Wt.shape[1], float(k)) for k in KS]))
            for i, k in enumerate(KS):  # column 0 (all weights 1) vs a direct unweighted computation
                idchk = max(idchk, abs(m[i, 0] - direct(sc, y, tb, min(k, len(fr)))))
            n_short += int(len(fr) < 1000)
            acc.append(m)
            ns.append(len(fr))
            t1.append(direct(sc, y, tb, max(1, int(np.ceil(0.01 * len(fr))))))
        res[cell] = np.nanmean(np.stack(acc), 0) if acc else None
        info[cell] = {'n_frames': int(len(np.concatenate([x[0] for x in Ld]))) if False else int(J[cell]['n_frames']),
                      'base_rate': float(J[cell]['base_rate']), 'heldout_median': float(np.median(ns)) if ns else None,
                      'n_dirs': len(ns), 'dirs_with_heldout_lt_1000': n_short,
                      'top1_prec_here': float(np.mean(t1)) if t1 else None, 'top1_prec_stored': J[cell]['top1_prec']}
    return tag, part, res, info, idchk


def cmd_boot(dom):
    from multiprocessing import get_context
    import v20_gate as G
    F, SP = L.frames(dom)
    rng = np.random.default_rng(SEED)
    Wser = {}
    for ser in sorted(F.series.unique()):  # same draws as v21_ladder.cmd_boot
        units = np.array(sorted(set(F.unit.values[F.series.values == ser])))
        W = np.stack([np.bincount(rng.integers(0, len(units), len(units)), minlength=len(units)) for _ in range(NB)])
        Wser[ser] = (units, W.astype(np.float64))
    S0 = json.loads((G.OUT / f'{dom}_scores.json').read_text())
    gate = [c for c, r in S0['eligibility'].items() if r['eligible'] and c.split('|')[1] not in G.ONLID]
    fcells = {}
    if dom == 'ants':
        import eval as EV
        _beh = dict(EV.BEH)
        fel = L.focal_elig()  # mutates EV.BEH; restored so the workers build the full frame table
        EV.BEH.clear()
        EV.BEH.update(_beh)
        fcells = {'single': [c for c, e in fel.items() if e and c.endswith('|single')],
                  'double': [c for c, e in fel.items() if e and c.endswith('|double')],
                  'or': [c for c in gate if c.endswith('|groom_any')]}
    parts = {'gate': gate, **({'focal': fcells['single'] + fcells['double']} if dom == 'ants' else {})}
    jobs = [(dom, t, p, Wser, cs) for t in TAGS for p, cs in parts.items()]
    out, info, mx = {}, {}, 0.0
    with get_context('fork').Pool(int(os.environ.get('SLURM_CPUS_PER_TASK', 8))) as pool:
        for tag, part, res, inf, idc in pool.imap_unordered(job, jobs):
            mx = max(mx, idc)
            for c, m in res.items():
                if m is not None:
                    out[f'{tag}|{part}|{c}'] = m
            info[f'{tag}|{part}'] = inf
            L.log(f'precK {dom} {tag} {part} done')
    OUTD.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTD / f'boot_{dom}.npz', **out)
    (OUTD / f'cells_{dom}.json').write_text(json.dumps({'gate_cells': gate, 'focal_cells': fcells, 'info': info, 'Ks': KS,
                                                        'weights_vs_direct_max_abs_diff': mx}, default=float))
    L.log(f'precK boot {dom} saved; weights-vs-direct max diff {mx:.2e}')


def cmd_report():
    rng = np.random.default_rng(SEED + 1)
    out, md = {}, []
    for dom in ('mice', 'ants'):
        z = np.load(OUTD / f'boot_{dom}.npz')
        meta = json.loads((OUTD / f'cells_{dom}.json').read_text())
        gcells, fcells, info = meta['gate_cells'], meta['focal_cells'], meta['info']
        groups = {'v1': [f'v1_s{i}' for i in range(11)], 'S': [f'S_s{s}' for s in range(1001, 1007)],
                  'V': [f'V_s{s}' for s in range(1001, 1007)]}
        # identity: top-1% precision recomputed here vs stored
        dif = [abs(i['top1_prec_here'] - i['top1_prec_stored']) for k, i_ in info.items() for i in i_.values()
               if i['top1_prec_here'] is not None and i['top1_prec_stored'] is not None]
        sets = {'gate': (gcells, 'gate')}
        if dom == 'ants':
            for f, cs in fcells.items():
                sets[f'focal_{f}'] = (cs, 'gate' if f == 'or' else 'focal')
        R = {'identity_top1_max_abs_diff': max(dif), 'identity_n': len(dif),
             'weights_vs_direct_max_abs_diff': meta['weights_vs_direct_max_abs_diff'], 'sets': {}}
        # per-cell descriptives
        cd = {}
        for c in gcells:
            r = [info[f'{t}|gate'][c] for t in groups['v1'][:1]][0]
            cd[c] = {'n_frames': r['n_frames'], 'base_rate': r['base_rate'], 'heldout_median': r['heldout_median'],
                     'n_dirs': r['n_dirs'], 'dirs_with_heldout_lt_1000': r['dirs_with_heldout_lt_1000'],
                     'n_per_1pct_heldout': 0.01 * r['heldout_median']}
        R['cells'] = cd
        for sname, (cs, part) in sets.items():
            dm = {}
            for g, tags in groups.items():
                dm[g] = np.stack([np.mean([z[f'{t}|{part}|{c}'] for c in cs], 0) for t in tags])  # (seeds, 2, NB+1)
            lv = {g: {f'p@{k}': float(np.median(dm[g][:, i, 0])) for i, k in enumerate(KS)} for g in dm}
            lv_seeds = {g: {f'p@{k}': dm[g][:, i, 0].tolist() for i, k in enumerate(KS)} for g in dm}
            # seed-median top-1% (stored, same cells)
            for g, tags in groups.items():
                v = []
                for t in tags:
                    pj = json.loads((L.LO / dom / f'{t}.json').read_text())[part]
                    v.append(float(np.mean([pj[c]['top1_prec'] for c in cs])))
                lv[g]['top1pct'] = float(np.median(v))
            con = {}
            for x, y in (('V', 'v1'), ('S', 'v1'), ('V', 'S')):
                r = {}
                for i, k in enumerate(KS):
                    ax, ay = dm[x][:, i, 1:], dm[y][:, i, 1:]
                    ix = rng.integers(0, len(ax), (NB, len(ax)))
                    iy = rng.integers(0, len(ay), (NB, len(ay)))
                    rr = np.arange(NB)[:, None]
                    d = np.nanmean(ax[ix, rr], 1) - np.nanmean(ay[iy, rr], 1)
                    r[f'p@{k}'] = {'diff': float(dm[x][:, i, 0].mean() - dm[y][:, i, 0].mean()),
                                   'ci': np.nanpercentile(d, [2.5, 97.5]).tolist(),
                                   'ci_units_only': np.nanpercentile(ax.mean(0) - ay.mean(0), [2.5, 97.5]).tolist()}
                con[f'{x} - {y}'] = r
            R['sets'][sname] = {'cells': cs, 'levels_median': lv, 'levels_seeds': lv_seeds, 'contrasts': con}
        out[dom] = R
        md += [f'## {dom}', '', f'identity: top-1% precision recomputed vs stored, max abs diff {R["identity_top1_max_abs_diff"]:.2e} '
               f'over {R["identity_n"]} (model, cell) values; weights-vs-direct precK diff {R["weights_vs_direct_max_abs_diff"]:.2e}', '',
               '| cell | eval frames | base rate | held-out frames/direction (median) | 1% of held-out | dirs with held-out < 1000 |', '|---|---|---|---|---|---|']
        for c, r in cd.items():
            md.append(f'| {nm(c)} | {r["n_frames"]} | {r["base_rate"]:.4f} | {r["heldout_median"]:.0f} | {r["n_per_1pct_heldout"]:.0f} | {r["dirs_with_heldout_lt_1000"]}/{r["n_dirs"]} |')
        for sname, S in R['sets'].items():
            md += ['', f'### {sname} (mean over {len(S["cells"])} cells; levels = seed median)', '',
                   '| arm | precision@100 | precision@1000 | top-1% precision |', '|---|---|---|---|']
            for g, v in S['levels_median'].items():
                md.append(f'| {g} | {v["p@100"]:.3f} | {v["p@1000"]:.3f} | {v["top1pct"]:.3f} |')
            md += ['', '| contrast | d precision@100 [95% CI pools/videos+seeds] | d precision@1000 [95% CI] |', '|---|---|---|']
            for k, r in S['contrasts'].items():
                md.append(f'| {k} | ' + ' | '.join(f'{r[m]["diff"]:+.3f} [{r[m]["ci"][0]:+.3f}, {r[m]["ci"][1]:+.3f}]' for m in ('p@100', 'p@1000')) + ' |')
        md.append('')
    (OUTD / 'precK.json').write_text(json.dumps(out, indent=1, default=float))
    (OUTD / 'precK.md').write_text('\n'.join(md))
    print('\n'.join(md))




# ---------------------------------------------------------------------------------------------- pooled ants (post hoc)
PM = ('AUROC', 'nAP', 'top1_prec', 'p@100', 'p@1000')
PCELLS = ('Y2F', 'B2F', 'groom_any', 'single', 'double')  # 'or' == groom_any
POOL = OUTD / 'pooled'


def pooled_frame(F, gate, fcells):
    """ants frame table with ONE series 'pooled' and only the 5 pooled behaviours; each behaviour restricted to the series
    in which its per-series cell is eligible (same eligibility as the gate / focal cells)."""
    G = L.focal_frame(F)
    P = F[[c for c in F.columns if not (c.startswith('y_') or c.startswith('v_'))]].copy()
    elig = {b: {c.split('|')[0] for c in gate if c.endswith('|' + b)} for b in ('Y2F', 'B2F', 'groom_any')}
    elig['single'] = {c.split('|')[0] for c in fcells['single']}
    elig['double'] = {c.split('|')[0] for c in fcells['double']}
    for b in PCELLS:
        src = G if b in ('single', 'double') else F
        P[f'y_{b}'] = src[f'y_{b}'].values
        P[f'v_{b}'] = src[f'v_{b}'].values & F.series.isin(elig[b]).values
    P['series'] = 'pooled'
    return P, elig


def pooled_task(tag):
    import v20_gate as G
    import eval as EV
    F, SP = L.frames('ants')
    S0 = json.loads((G.OUT / 'ants_scores.json').read_text())
    gate = [c for c, r in S0['eligibility'].items() if r['eligible'] and c.split('|')[1] not in G.ONLID]
    _beh = dict(EV.BEH)
    fel = L.focal_elig()
    EV.BEH.clear()
    EV.BEH.update(_beh)
    fcells = {'single': [c for c, e in fel.items() if e and c.endswith('|single')],
              'double': [c for c, e in fel.items() if e and c.endswith('|double')]}
    X, _ = L.readout('ants', tag, '', F)
    P, elig = pooled_frame(F, gate, fcells)
    use = np.ones(len(P), bool)
    res = L.score2(P, X, use, SP)  # same picking rule per cell: AUROC-picked neuron per (split, half) direction
    units = np.array(sorted(set(F.unit.values)))
    rng = np.random.default_rng(SEED + 2)
    W = np.bincount(rng.integers(0, len(units), (NB, len(units))).ravel() + np.repeat(np.arange(NB), len(units)) * len(units),
                    minlength=NB * len(units)).reshape(NB, len(units)).astype(np.float64)
    Wt = np.concatenate([np.ones((len(units), 1)), W.T], 1)
    uix = pd.Series(np.arange(len(units)), index=units)
    D = L.cell_dirs(P, SP, res, 'gate')
    out, js, chk = {}, {}, 0.0
    for cell, Ld in D.items():
        b = cell.split('|')[1]
        y_all = P[f'y_{b}'].values
        acc = []
        for fr, j, jn, tb in Ld:
            y = y_all[fr]
            u = uix.reindex(P.unit.values[fr]).values.astype(np.int64)
            cnt_u = np.bincount(u, minlength=len(units)).astype(np.float64)
            s = np.asarray(X[fr, j], np.float64)
            m = L.wmetrics(s, y, u, tb, Wt, cnt_u)
            if jn != j:
                m[1] = L.wmetrics(np.asarray(X[fr, jn], np.float64), y, u, tb, Wt, cnt_u)[1]
            pk = np.stack(wprec(s, y, u, tb, Wt, cnt_u, [np.full(Wt.shape[1], float(k)) for k in KS]))
            acc.append(np.concatenate([m, pk]))
        M = np.nanmean(np.stack(acc), 0)
        out[cell] = M
        for i, name in enumerate(('AUROC', 'nAP', 'top1_prec')):
            if res[cell][name] is not None:
                chk = max(chk, abs(M[i, 0] - res[cell][name]))
        js[cell] = {'n_frames': res[cell]['n_frames'], 'base_rate': res[cell]['base_rate'], 'n_dirs': res[cell]['n_dirs'],
                    'series': sorted(elig[b]), 'AUROC': res[cell]['AUROC'], 'nAP': res[cell]['nAP'],
                    'top1_prec': res[cell]['top1_prec'], 'p@100': float(M[3, 0]), 'p@1000': float(M[4, 0])}
    if chk > 1e-6:
        raise SystemExit(f'POOLED IDENTITY FAILED {tag}: {chk}')
    POOL.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(POOL / f'{tag}.npz', **{c.split('|')[1]: m for c, m in out.items()})
    (POOL / f'{tag}.json').write_text(json.dumps({'identity_max_abs_diff': chk, 'cells': js}, default=float))
    L.log(f'pooled {tag}: identity {chk:.2e}')


def pooled_report():
    rng = np.random.default_rng(SEED + 3)
    groups = {'v1': [f'v1_s{i}' for i in range(11)], 'S': [f'S_s{s}' for s in range(1001, 1007)],
              'V': [f'V_s{s}' for s in range(1001, 1007)]}
    Z = {t: np.load(POOL / f'{t}.npz') for ts in groups.values() for t in ts}
    J = {t: json.loads((POOL / f'{t}.json').read_text()) for t in Z}
    sets = {'Y2F': ['Y2F'], 'B2F': ['B2F'], 'any (= focal or)': ['groom_any'], 'focal single': ['single'],
            'focal double': ['double'], 'MEAN of Y2F, B2F, any': ['Y2F', 'B2F', 'groom_any']}
    idm = max(j['identity_max_abs_diff'] for j in J.values())
    out, md = {'identity_max_abs_diff': idm}, [
        '# Ants POOLED cells (post hoc, reported view; NOT the pre-registered gate)', '',
        'All eval frames of the eligible series (v2, v3, v4, v4_infested) in one ranking per behaviour; best neuron per '
        'direction picked by AUROC on the other half (ladder rule); videos resampled across all series (2000 replicates) '
        'and seeds. K fixed (100 / 1000) per replicate.', '',
        f'identity (weights-1 column vs scorer, AUROC/nAP/top-1%): max abs diff {idm:.2e}', '']
    c0 = J['v1_s0']['cells']
    md += ['| pooled cell | series pooled | eval frames | base rate |', '|---|---|---|---|']
    for c, r in c0.items():
        md.append(f'| {c.split("|")[1]} | {",".join(r["series"])} | {r["n_frames"]} | {r["base_rate"]:.4f} |')
    for sn, cs in sets.items():
        dm = {g: np.stack([np.mean([Z[t][c] for c in cs], 0) for t in ts]) for g, ts in groups.items()}  # (seeds,5,NB+1)
        lv = {g: [float(np.median(dm[g][:, i, 0])) for i in range(5)] for g in dm}
        md += ['', f'## {sn}', '', '| arm | mean AUROC | nAP | top-1% prec | prec@100 | prec@1000 |', '|---|---|---|---|---|---|']
        for g, v in lv.items():
            md.append(f'| {g} (seed median) | ' + ' | '.join(f'{x:.3f}' for x in v) + ' |')
        con = {}
        md += ['', '| contrast | ' + ' | '.join(PM) + ' |', '|---|' + '---|' * 5]
        for x, y in (('V', 'v1'), ('S', 'v1'), ('V', 'S')):
            r = {}
            for i, M in enumerate(PM):
                ax, ay = dm[x][:, i, 1:], dm[y][:, i, 1:]
                ix = rng.integers(0, len(ax), (NB, len(ax)))
                iy = rng.integers(0, len(ay), (NB, len(ay)))
                rr = np.arange(NB)[:, None]
                d = np.nanmean(ax[ix, rr], 1) - np.nanmean(ay[iy, rr], 1)
                r[M] = {'diff': float(dm[x][:, i, 0].mean() - dm[y][:, i, 0].mean()), 'ci': np.nanpercentile(d, [2.5, 97.5]).tolist()}
            con[f'{x} - {y}'] = r
            md.append(f'| {x} - {y} | ' + ' | '.join(f'{r[M]["diff"]:+.3f} [{r[M]["ci"][0]:+.3f}, {r[M]["ci"][1]:+.3f}]' for M in PM) + ' |')
        out[sn] = {'levels_median': dict(zip(groups, lv.values())), 'contrasts': con}
    (OUTD / 'pooled_ants.json').write_text(json.dumps(out, indent=1, default=float))
    (OUTD / 'pooled_ants.md').write_text('\n'.join(md))
    print('\n'.join(md))


if __name__ == '__main__':
    if sys.argv[1] == 'boot':
        cmd_boot(sys.argv[2])
    elif sys.argv[1] == 'pooled':
        pooled_task(sys.argv[2])
    elif sys.argv[1] == 'pooled_report':
        pooled_report()
    else:
        cmd_report()
