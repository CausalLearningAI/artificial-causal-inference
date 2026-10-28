"""
ECI v1 vs v2.0 on the same footing (CPU only; no SAE trained, nothing published).

==================================================================================================================
PRE-REGISTRATION (written 2026-10-28 BEFORE any number of steps 'dyn' / 'ants2' below was computed; steps 'align',
'disc', 'cap' only re-read existing scores with rules fixed earlier: eval.py scorer / eligibility, discoveries.py count()
(43402a9), behaviour suite amendment 2.)
ALIGN: per (series, behaviour) cell and the domain mean over eval.py-eligible cells: AUROC (primary), nAP, F1max of the
    cross-fitted best neuron. v1 = s0 (deployed) + s1-10 retrained, 11 seeds, its own readout (raw frame max, eval audit);
    v2.0 = the 5 PASS B seeds V20F5 950-954 (eval readout: mice 0.6 s smooth, ants raw) and the final deployed model on
    its codes sets (_hmm P(on), _smooth). Delta = v2.0 - v1: seeds vs seeds Welch 95% CI; final model vs v1 seeds =
    final - mean(v1), CI = t interval of the v1 seed mean (df 10). 'v2.0 better' on a row iff the CI lower bound > 0.
DISC: NES and NESS selected neurons (count(): Bonferroni, t-test rows, full / common window, max pooling), per series and
    outcome (mean; bouts / latency where both models have them), at prefix 256 and the full dictionary (v1 1024, v2.0
    2048); v1 = the published online model (fg448al / antsfg eci30 + eci_c1v4_noday for v4 / frogsfg), v2.0 = final _hmm
    and _smooth. Descriptive (no test across models).
CAP: behaviour-suite amendment 2 (C1-C8) capacity full / @256, v1 5 seeds (existing eci_v2_behav run) vs V20F5 5 seeds
    (eci_v2_0 run), same probe set and code; Welch 95% CI.
DYN (new label-free measure, mice and ants: the probe sets of the suite, 72 videos x 1500 contiguous 5 fps frames, v1 and
    V20F5 frame codes 'D' of the suite, unsmoothed frame max):
    motion energy ME(t) = mean over the pixels that are animal (frozen tight dark cue, src/eci/v2/mask.py, at the v2
    encoder input side S, canonical frame) in frame t OR t-1 of |grey(t) - grey(t-1)| (consecutive 5 fps frames; the
    first frame of each window has none); static appearance A(t) = number of animal pixels in frame t.
    Per neuron j eligible iff c_j > 0 on >= 1% of the frames: r_M = Spearman(c_j, ME), r_A = Spearman(c_j, A) over the
    frames with ME. DYNAMIC iff r_M >= 0.10 AND r_M > |r_A|. Bout duration = median length (s) of the maximal runs of
    c_j > 0 inside a window. Reported per arm: n eligible, n dynamic (full and among j < 256), quantiles of r_M, r_A and
    bout duration; v2.0 - v1 dynamic count Welch 95% CI over the 5 + 5 seeds.
ANTS2 (ISTAnt, ants v2, 44 videos, treatment T 1 vs 2): per-video mean of the labelled grooming (groom_any = Y2F | B2F,
    annotated frames) and of the v2.0 final grooming neuron (the AUROC best neuron of cell v2|groom_any picked most often
    over the cross-fit directions) as P(on) and continuous; Welch t test 2 vs 1 (effect = mean difference, p) and OLS
    with recording_date as a covariate; same for v1 s0's grooming neuron (raw frame max). Reading: if the LABEL itself
    shows no significant arm effect, the null NES result is a power / effect-size question, not the representation.
==================================================================================================================
Steps: align | disc | cap (login, small json / csv reads) ; me --dom D (CPU job: motion energy) ; dyn | ants2 (CPU job)
Output: results/vision/eci_v2_0B/v1_vs_v2/*.json, table.md
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
for p in ('', 'scripts/eci/v2', 'scripts/eci/v2_eval', 'scripts/eci/v2_prob'):
    sys.path.insert(0, str(REPO / p))
V = REPO / 'results/vision'
OUT = V / 'eci_v2_0B/v1_vs_v2'
EC = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes')
MET = ('AUROC', 'nAP', 'F1max')
FINAL = {'mice': 'v20_mice_N2f5_s950', 'ants': 'v20_ants_N2f5_s951', 'frogs': 'v20_frogs_N2f5_s952'}
log = lambda s: print(time.strftime('%H:%M:%S'), s, flush=True)  # noqa: E731


def welch(a, b):
    from scipy import stats
    a, b = np.asarray(a, float), np.asarray(b, float)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    d = a.mean() - b.mean()
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    h = stats.t.ppf(0.975, df) * np.sqrt(va + vb)
    return [float(d), float(d - h), float(d + h)]


def one_vs(x, b):
    from scipy import stats
    b = np.asarray(b, float)
    h = stats.t.ppf(0.975, len(b) - 1) * b.std(ddof=1) / np.sqrt(len(b))
    d = float(x - b.mean())
    return [d, d - h, d + h]


# ---------------------------------------------------------------------------------------------- align
def cmd_align(a):
    res = {}
    for dom in ('mice', 'ants'):
        elig = json.loads((V / f'eci_v2_0/eval/{dom}_summary.json').read_text())['eligibility']
        ok = [c for c, r in elig.items() if r.get('eligible')]
        rd = 'raw'
        v1 = {s: json.loads((V / f'eci_v2_eval/{dom}/v1_s{s}__{rd}.json').read_text())['cells'] for s in range(11)}
        f5 = {s: json.loads((V / f'eci_v2_0/eval/{dom}/V20F5_s{s}.json').read_text())['cells'] for s in range(950, 955)}
        fin = {k: json.loads((V / f'eci_v2_0B/eval/{dom}/deliv_{k}.json').read_text())['cells'] for k in ('hmm', 'smooth')}
        rows = {}
        for cell in ok + ['MEAN']:
            r = {}
            for M in MET:
                get = (lambda C: float(np.mean([C[c][M] for c in ok if C.get(c, {}).get(M) is not None]))) if cell == 'MEAN' \
                    else (lambda C: C[cell][M] if C.get(cell, {}).get(M) is not None else np.nan)
                b = np.array([get(C) for C in v1.values()])
                x5 = np.array([get(C) for C in f5.values()])
                r[M] = {'v1_mean': float(np.nanmean(b)), 'v1_sd': float(np.nanstd(b, ddof=1)), 'V20F5_mean': float(np.nanmean(x5)),
                        'F5_minus_v1': welch(x5[np.isfinite(x5)], b[np.isfinite(b)]),
                        **{f'final_{k}': get(C) for k, C in fin.items()},
                        **{f'final_{k}_minus_v1': one_vs(get(C), b[np.isfinite(b)]) for k, C in fin.items()}}
            rows[cell] = r
        res[dom] = rows
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'align.json').write_text(json.dumps(res, indent=1))
    L = ['| domain | cell | metric | v1 (11 seeds) | V20F5 (5) | F5 - v1 [95% CI] | final P(on) | P(on) - v1 [CI] | final continuous | cont - v1 [CI] |',
         '|---|---|---|---|---|---|---|---|---|---|']
    f = lambda v: f'{v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]'  # noqa: E731
    for dom, rows in res.items():
        for cell, r in rows.items():
            for M in (MET if cell == 'MEAN' else ('AUROC',)):
                x = r[M]
                L.append(f'| {dom} | {cell} | {M} | {x["v1_mean"]:.3f} ± {x["v1_sd"]:.3f} | {x["V20F5_mean"]:.3f} | {f(x["F5_minus_v1"])} | '
                         f'{x["final_hmm"]:.3f} | {f(x["final_hmm_minus_v1"])} | {x["final_smooth"]:.3f} | {f(x["final_smooth_minus_v1"])} |')
    (OUT / 'align.md').write_text('\n'.join(L))
    print('\n'.join(L))


# ---------------------------------------------------------------------------------------------- discoveries
def cmd_disc(a):
    import discoveries as DI
    V1 = {'mice': [('mice', V / 'mice/eci/nes/matryoshka_btk_1024_k16_fg448al_s0', 'mice', None)],
          'ants': [('ants_v2', V / 'ants/eci30/nes/matryoshka_btk_1024_k16_antsfg_s0', 'ants_eci30', 'v2_'),
                   ('ants_v3', V / 'ants/eci30/nes/matryoshka_btk_1024_k16_antsfg_s0', 'ants_eci30', 'v3_'),
                   ('ants_v4', V / 'ants/eci30/nes/matryoshka_btk_1024_k16_antsfg_s0', 'ants_eci30', ('v4_', 'infested'))],
          'frogs': [('frogs_juv', V / 'frogs/eci/nes/matryoshka_btk_1024_k16_frogsfg_s0', 'frogs', None)]}
    V2 = {'mice': [('mice', 'mice/eci/nes', 'mice')],
          'ants': [('ants_v2', 'ants/eci_r4v2p/nes', 'ants_eci_r4v2p'), ('ants_v3', 'ants/eci_r4v3p/nes', 'ants_eci_r4v3p'),
                   ('ants_v4', 'ants/eci_r4v4p/nes', 'ants_eci_r4v4p')],
          'frogs': [('frogs_juv', 'frogs/eci/nes', 'frogs'), ('frogs_tad', 'frogs/eci_tadpole_r4/nes', 'frogs_tadpole')]}

    def tot(c, keep=None):
        if c is None:
            return None
        out = {}
        for an, d in c.items():
            if keep and not an.startswith(keep):
                continue
            for p, n in d.items():
                out[p] = out.get(p, 0) + n
        return out
    res = {}
    for dom in V1:
        for ser, base, key, keep in V1[dom]:
            sae = base.name
            r = {}
            for o, sub in DI.SUB.items():
                r[f'NES_{o}'] = tot(DI.count(base / sub / 'summary.csv', o), keep)
                r[f'NESS_{o}'] = tot(DI.count(V / 'eci_atlas_v37/methods' / key / sae / 'dines' / sub / 'summary.csv', o), keep)
            res[f'v1|{ser}'] = r
        for ser, root, key in V2[dom]:
            for suf in ('_hmm', '_smooth'):
                nm = FINAL[dom] + suf
                r = {}
                for o, sub in DI.SUB.items():
                    r[f'NES_{o}'] = tot(DI.count(V / root / nm / sub / 'summary.csv', o))
                    r[f'NESS_{o}'] = tot(DI.count(V / 'eci_v2/ness' / key / nm / 'dines' / sub / 'summary.csv', o))
                res[f'v2.0{suf}|{ser}'] = r
    (OUT / 'disc.json').write_text(json.dumps(res, indent=1))
    L = ['| model | series | NES mean @256 / full | NESS mean @256 / full | NES bouts @256 / full | NESS bouts @256 / full | NES latency @256 / full |',
         '|---|---|---|---|---|---|---|']
    g = lambda d, full: '-' if d is None else f'{d.get(256, 0)} / {d.get(full, 0)}'  # noqa: E731
    for k, r in res.items():
        m, ser = k.split('|')
        full = 1024 if m == 'v1' else 2048
        L.append(f'| {m} | {ser} | {g(r["NES_mean"], full)} | {g(r["NESS_mean"], full)} | {g(r["NES_bouts"], full)} | '
                 f'{g(r["NESS_bouts"], full)} | {g(r["NES_latency"], full)} |')
    (OUT / 'disc.md').write_text('\n'.join(L))
    print('\n'.join(L))


# ---------------------------------------------------------------------------------------------- capacity
def cmd_cap(a):
    res = {}
    for dom in ('mice', 'ants'):
        v1 = [json.loads(f.read_text()) for f in sorted((V / f'eci_v2_behav/{dom}/suite_am2').glob('v1_s*.json'))]
        v2 = [json.loads(f.read_text()) for f in sorted((V / f'eci_v2_0/behav/{dom}/suite_am2').glob('V20F5_s*.json'))]
        r = {}
        for q in ('capacity_full', 'capacity_256'):
            A_, B_ = [x[q] for x in v2], [x[q] for x in v1]
            r[q] = {'v1': B_, 'V20F5': A_, 'F5_minus_v1': welch(A_, B_)}
        res[dom] = r
        log(f'{dom}: ' + json.dumps({q: [np.mean(v['v1']), np.mean(v['V20F5']), v['F5_minus_v1']] for q, v in r.items()}))
    (OUT / 'cap.json').write_text(json.dumps(res, indent=1))


# ---------------------------------------------------------------------------------------------- motion energy (CPU job)
def cmd_me(a):
    import torch
    sys.path.insert(0, str(REPO / 'scripts/eci/v2_behav'))
    import behav as BH
    from src.eci.v2.pipeline import Series
    from src.eci.v2.mask import grey_of
    from concurrent.futures import ThreadPoolExecutor
    dom = a.dom
    torch.set_num_threads(2)
    Pr = pd.read_parquet(V / f'eci_v2_behav/{dom}/probe.parquet')
    SE = {s: Series(BH.series_cfg(dom, s), torch.device('cpu'), load_encoder=False) for s in Pr.series.unique()}
    groups = list(Pr.groupby('vid', sort=True))

    def one(item):
        vid, g = item
        se = SE[g.series.iloc[0]]
        v = se.V[g.obs.iloc[0]]
        ks = g.k.values
        assert np.all(np.diff(ks) == 1)
        me, ar = np.full(len(ks), np.nan, np.float32), np.zeros(len(ks), np.float32)
        prev_g = prev_m = None
        for c0 in range(0, len(ks), 300):
            kk = ks[c0:c0 + 300]
            fr = se.run_job(**se.job(v, int(kk[0]), len(kk), threads=2))
            for b in range(0, len(kk), 50):
                x, _ = se.frames(v, fr[b:b + 50], kk[b:b + 50])
                gr = grey_of(x).float()
                m = se.mask.pixels(x, v['src_lo'] + kk[b:b + 50])
                for i in range(len(gr)):
                    t = c0 + b + i
                    ar[t] = float(m[i].sum())
                    if prev_g is not None:
                        u = m[i] | prev_m
                        me[t] = float((gr[i] - prev_g).abs()[u].mean()) if u.any() else 0.0
                    prev_g, prev_m = gr[i], m[i]
        return vid, me, ar
    ME = np.full(len(Pr), np.nan, np.float32)
    AR = np.zeros(len(Pr), np.float32)
    idx = {vid: g.index.values for vid, g in groups}
    t0 = time.time()
    with ThreadPoolExecutor(max(1, int(os.environ.get('SLURM_CPUS_PER_TASK', 8)) // 2)) as ex:
        for i, (vid, me, ar) in enumerate(ex.map(one, groups)):
            ME[idx[vid]], AR[idx[vid]] = me, ar
            log(f'me {dom}: {i + 1}/{len(groups)} videos ({time.time() - t0:.0f}s)')
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f'me_{dom}.npz', me=ME, area=AR, vid=Pr.vid.values, k=Pr.k.values)


# ---------------------------------------------------------------------------------------------- dyn (CPU job)
def cmd_dyn(a):
    from scipy.stats import rankdata
    res = {}
    for dom in ('mice', 'ants'):
        z = np.load(OUT / f'me_{dom}.npz')
        ME, AR, vid = z['me'], z['area'], z['vid']
        ok = np.isfinite(ME)
        rM, rA = rankdata(ME[ok]), rankdata(AR[ok])
        zs = lambda r: (r - r.mean()) / r.std()  # noqa: E731
        zM, zA = zs(rM), zs(rA)
        st = np.r_[0, np.flatnonzero(np.diff(vid) != 0) + 1, len(vid)]
        arms = {'v1': [(s, EC / f'v2_behav/{dom}/codes/v1_s{s}_D.npy') for s in range(5)],
                'V20F5': [(s, EC / f'v2_0/behav/{dom}/codes/V20F5_s{s}_D.npy') for s in range(950, 955)]}
        R = {}
        for arm, lst in arms.items():
            for s, p in lst:
                X = np.load(p, mmap_mode='r')
                W = X.shape[1]
                out = {'eligible': 0, 'dynamic': 0, 'dynamic_256': 0, 'eligible_256': 0, 'rM': [], 'rA': [], 'bout_s': []}
                for j0 in range(0, W, 128):
                    C = np.asarray(X[:, j0:j0 + 128], np.float32)
                    for jj in range(C.shape[1]):
                        c = C[:, jj]
                        if (c > 0).mean() < 0.01:
                            continue
                        j = j0 + jj
                        rc = zs(rankdata(c[ok]))
                        r_m, r_a = float((rc * zM).mean()), float((rc * zA).mean())
                        on = c > 0
                        runs = []
                        for a0, a1 in zip(st[:-1], st[1:]):
                            o = on[a0:a1].astype(np.int8)
                            d = np.diff(np.r_[0, o, 0])
                            runs += list(np.flatnonzero(d == -1) - np.flatnonzero(d == 1))
                        dyn = r_m >= 0.10 and r_m > abs(r_a)
                        out['eligible'] += 1
                        out['dynamic'] += int(dyn)
                        out['eligible_256'] += int(j < 256)
                        out['dynamic_256'] += int(dyn and j < 256)
                        out['rM'].append(r_m)
                        out['rA'].append(r_a)
                        out['bout_s'].append(float(np.median(runs)) / 5.0 if runs else 0.0)
                q = lambda v: [float(x) for x in np.quantile(v, [0.1, 0.5, 0.9])] if v else None  # noqa: E731
                R[f'{arm}_s{s}'] = {k: (q(v) if isinstance(v, list) else v) for k, v in out.items()}
                log(f'dyn {dom} {arm} s{s}: ' + json.dumps(R[f'{arm}_s{s}']))
        cmp = {}
        for q_ in ('dynamic', 'dynamic_256', 'eligible'):
            A_ = [R[k][q_] for k in R if k.startswith('V20F5')]
            B_ = [R[k][q_] for k in R if k.startswith('v1')]
            cmp[q_] = {'v1': B_, 'V20F5': A_, 'F5_minus_v1': welch(A_, B_)}
        res[dom] = {'seeds': R, 'compare': cmp}
    (OUT / 'dyn.json').write_text(json.dumps(res, indent=1))
    log(json.dumps({d: r['compare'] for d, r in res.items()}))


# ---------------------------------------------------------------------------------------------- ants v2 (ISTAnt)
def cmd_ants2(a):
    import statsmodels.formula.api as smf
    from scipy import stats
    import eval as EV
    F = EV.frame_table('ants')
    F = F[F.series == 'v2']
    ex = pd.read_csv(REPO / 'dataset/ants/eci_r4v2p/experiment.csv').set_index('observation_id')
    lab = F.groupby('obs').y_groom_any.mean()
    res = {}

    def test(y, name):
        d = pd.DataFrame({'y': y}).join(ex[['T', 'recording_date']])
        d = d.dropna()
        g1, g2 = d[d['T'] == 1].y, d[d['T'] == 2].y
        t = stats.ttest_ind(g2, g1, equal_var=False)
        m = smf.ols('y ~ C(T) + C(recording_date)', data=d).fit()
        k = [c for c in m.params.index if c.startswith('C(T)')][0]
        res[name] = {'n': [int(len(g1)), int(len(g2))], 'mean_T1': float(g1.mean()), 'mean_T2': float(g2.mean()),
                     'effect_T2_minus_T1': float(g2.mean() - g1.mean()), 'welch_p': float(t.pvalue),
                     'ols_day_effect': float(m.params[k]), 'ols_day_p': float(m.pvalues[k]),
                     'cohen_d': float((g2.mean() - g1.mean()) / np.sqrt((g1.var() + g2.var()) / 2))}
        log(f'{name}: ' + json.dumps(res[name]))
    test(lab, 'label groom_any (annotated frames)')
    # v2.0 final grooming neuron: AUROC best neuron of v2|groom_any, most frequent over directions
    cell = json.loads((V / 'eci_v2_0B/eval/ants/deliv_hmm.json').read_text())['cells']['v2|groom_any']
    js = pd.Series([x[2] for x in cell['AUROC_neurons']]).value_counts()
    j = int(js.index[0])
    res['v2.0_neuron'] = {'neuron': j, 'picked_in_dirs': int(js.iloc[0]), 'n_dirs': int(len(cell['AUROC_neurons'])),
                          'cell_AUROC_hmm': cell['AUROC']}
    S = np.load(V / 'eci_v2_0B/ants/summaries_v2.npz')
    obs = S['obs'].astype(str)
    for q in ('mean_p_on', 'mean_continuous', 'bout_rate_per_min'):
        test(pd.Series(S[q][:, j], index=obs), f'v2.0 n{j} {q}')
    c1 = json.loads((V / 'eci_v2_eval/ants/v1_s0__raw.json').read_text())['cells']['v2|groom_any']
    j1 = int(pd.Series([x[2] for x in c1['AUROC_neurons']]).value_counts().index[0])
    X1 = np.load(REPO / 'dataset/ants/eci/codes/matryoshka_btk_1024_k16_antsfg_s0/codes_max.npy', mmap_mode='r')
    A1 = pd.read_csv(REPO / 'dataset/ants/eci/annotations.csv', usecols=['observation_id'])
    v2obs = set(obs)
    m = A1.observation_id.isin(v2obs).values
    y1 = pd.Series(np.asarray(X1[np.flatnonzero(m), j1], np.float32), index=A1.observation_id.values[m]).groupby(level=0).mean()
    res['v1_neuron'] = {'neuron': j1, 'cell_AUROC': c1['AUROC']}
    test(y1, f'v1 s0 n{j1} mean raw code (all frames)')
    (OUT / 'ants2.json').write_text(json.dumps(res, indent=1))


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=('align', 'disc', 'cap', 'me', 'dyn', 'ants2'))
    p.add_argument('--dom', default='mice')
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {'align': cmd_align, 'disc': cmd_disc, 'cap': cmd_cap, 'me': cmd_me, 'dyn': cmd_dyn, 'ants2': cmd_ants2}[a.step](a)


if __name__ == '__main__':
    main()
