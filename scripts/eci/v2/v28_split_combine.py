"""
ECI v2.8 split check (POST HOC): combine the per-dictionary pickles of v28_split_neurons.py.
-> results/vision/eci_v2_8/split/{tables.md, split.json}
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / 'results/vision/eci_v2_8/split'
WORK = OUT / 'work'
BEH = ('nose_nose', 'nt')
BN = {'nose_nose': 'nose-nose', 'nt': 'nose-tail'}
V2 = ['PREDk_s2703', 'PREDk_s2701', 'PREDk_s2702', 'PRED26_s2601', 'PRED26_s2602', 'PRED26_s2603'] + [f'V_s{s}' for s in range(1001, 1007)]
ALL = V2 + ['v1_s0']
MET = ('auc', 'ap', 'top3', 'top10')
MN = {'auc': 'AUROC', 'ap': 'AP', 'top3': 'precision@top3/video', 'top10': 'precision@top10/video'}


def f3(x):
    return 'n/a' if x is None or not np.isfinite(x) else f'{x:.3f}'


D = {k: pickle.load(open(WORK / f'{k}.pkl', 'rb')) for k in ALL if (WORK / f'{k}.pkl').exists()}
D0 = D['PREDk_s2703']
names, Msub, PN, cells, pools, pann = D0['names'], D0['Msub'], D0['PN'], D0['cells'], D0['pools'], D0['pann']
for k, r in D.items():
    assert r['names'] == names and (r['Msub'] == Msub).all() and np.allclose(r['PN'], PN) and list(r['vids']) == list(D0['vids']), k
QI = {n: i for i, n in enumerate(names)}
RAND = sorted({n.split('|')[0] for n in names if n.startswith('rand')})
LOAO = ['loao_SG', 'loao_SD', 'loao_MF', 'loao_MB']


def metrics(r, j, b, q):
    """AUROC, AP, precision@top3/10/video of neuron j of dictionary r on subset q, + chance levels."""
    c = BEH.index(b)
    qi = QI[q]
    m = Msub[qi].astype(float)
    vm = Msub[qi][r['vcell']].astype(bool)
    pos, nval = r['topk'][('pos', b)], r['topk'][('nval', b)]
    e = vm & (pos > 0)
    out = {'auc': float(r['auc'][j][qi, c]), 'ap': float(r['ap'][j][qi, c])}
    for k in (3, 10):
        out[f'top{k}'] = float(r['topk'][(j, b, k)][e].sum() / (k * e.sum())) if e.any() else float('nan')
    out['chance_ap'] = float(PN[c, 0] @ m / ((PN[c, 0] + PN[c, 1]) @ m))
    out['chance_topk'] = float(np.mean(pos[e] / nval[e])) if e.any() else float('nan')
    out['videos_pos'] = int(e.sum())
    out['videos'] = int(vm.sum())
    out['frames'] = int((PN[c, 0] + PN[c, 1]) @ m)
    return out


def sel_ap_sum(r, split, rule='sgap'):
    p = r['picks'][rule][split]
    if p is None:
        return 0.0
    return float(r['ap'][p[0]][QI[f'{split}|sel'], 0] + r['ap'][p[1]][QI[f'{split}|sel'], 1])


def summ(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return {'mean': float(v.mean()), 'sd': float(v.std(ddof=1)), 'p2.5': float(np.percentile(v, 2.5)), 'p97.5': float(np.percentile(v, 97.5)), 'n': int(len(v))}


res = {'post_hoc': True, 'dictionaries': list(D), 'n_random_splits': len(RAND), 'check_vs_audit': {k: r['check_vs_audit'] for k, r in D.items()}}
md = ['# ECI v2.8 split check: annotator split vs random pool splits (POST HOC robustness analysis, CPU only, no training)', '',
      'Scripts: scripts/eci/v2/v28_split_neurons.py (per dictionary), v28_split_combine.py. Frames: the v2.7 edge-excluded ext '
      'frame table; scored frames = the 23 annotated pools (rd41_3, annotator unknown, excluded as in v2.7). Readout and rule '
      'sgap exactly as scripts/eci/v2/v29_prereg.md (firing cap < 10%). AUROC / AP on the P(on) readout (AP raw-code tie-break; '
      'v1 s0: raw code); precision@top-k/video = raw-code ranking, k valid frames per video, averaged over videos with >= 1 '
      'positive; chance = subset prevalence (AP) and mean per-video prevalence over videos with >= 1 positive (precision@top-k). '
      'All splits are by POOL. Written after the v2.8 audit and the v2.9 pre-registration were seen.', '']
# identity check
md += ['Identity check vs the v2.8 audit (official split, all eligible neurons; max |difference| over AUROC / AP, SG and non-SG): ' +
       ', '.join(f"{k} {max(v.values()):.1e}" for k, v in res['check_vs_audit'].items() if v), '']

# ---------------------------------------------------------------- 1. split description
ex = pd.read_csv(REPO / 'data/mice/v1/experiment.csv')
ex = ex[ex.annotation_file.notna()].copy()
ex['unit'] = ex.observation_id.str.rsplit('_', n=2).str[0]
meta = ex.groupby('unit').agg(pool=('pool', 'first'), line=('line', 'first'), sex=('sex', 'first'), genotype=('genotype', 'first'),
                              date=('date', 'min'))
d = D0['describe'].groupby('unit').agg(videos=('videos', 'sum'), frames=('frames', 'sum'), nn=('nn', 'sum'), nt=('nt', 'sum')).join(meta)
d['annot'] = [pann[u] for u in d.index]
d['side'] = np.where(d.annot == 'SG', 'select (SG)', 'evaluate (non-SG)')
d = d.sort_values(['side', 'annot', 'date'], ascending=[False, True, True])
md += ['## 1. The current split (per pool; select = annotator SG, evaluate = the other annotators)', '',
       f"rd41_3 (6 videos, {D0['n_na_rows']:,} frames, annotator unknown upstream) is in neither side.", '',
       '| pool | side | annotator (videos) | line | sex | genotype | first date | videos | frames | nose-nose positives | nose-nose prevalence | nose-tail positives | nose-tail prevalence |',
       '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
ann_v = ex.groupby('unit').annotator.apply(lambda s: ', '.join(f'{a} {n}' for a, n in s.value_counts().items()))
for u, x in d.iterrows():
    md.append(f"| {x.pool} | {x.side} | {ann_v[u]} | {x.line} | {x.sex} | {x.genotype} | {x.date} | {x.videos} | {x.frames:,} | {int(x.nn):,} | "
              f"{100 * x.nn / x.frames:.2f}% | {int(x.nt):,} | {100 * x.nt / x.frames:.2f}% |")
dd = D0['describe'].copy()
md += ['', '| group | pools | videos | frames | nose-nose prevalence | nose-tail prevalence | pool range nose-nose | pool range nose-tail |', '|---|---|---|---|---|---|---|---|']
desc = {}
for nm, msk in [('select (SG)', d.annot == 'SG'), ('evaluate (non-SG)', d.annot != 'SG')] + [(f'annotator {a} (by video)', None) for a in sorted(dd.annot.unique())]:
    if msk is not None:
        x = d[msk]
        pv = (x.nn / x.frames, x.nt / x.frames)
        row = dict(pools=len(x), videos=int(x.videos.sum()), frames=int(x.frames.sum()), nn=float(x.nn.sum() / x.frames.sum()), nt=float(x.nt.sum() / x.frames.sum()),
                   nn_range=[float(pv[0].min()), float(pv[0].max())], nt_range=[float(pv[1].min()), float(pv[1].max())])
    else:
        a = nm.split()[1]
        x = dd[dd.annot == a]
        pv = (x.nn / x.frames, x.nt / x.frames)
        row = dict(pools=int(x.unit.nunique()), videos=int(x.videos.sum()), frames=int(x.frames.sum()), nn=float(x.nn.sum() / x.frames.sum()), nt=float(x.nt.sum() / x.frames.sum()),
                   nn_range=[float(pv[0].min()), float(pv[0].max())], nt_range=[float(pv[1].min()), float(pv[1].max())])
    desc[nm] = row
    md.append(f"| {nm} | {row['pools']} | {row['videos']} | {row['frames']:,} | {100 * row['nn']:.2f}% | {100 * row['nt']:.2f}% | "
              f"{100 * row['nn_range'][0]:.2f}-{100 * row['nn_range'][1]:.2f}% | {100 * row['nt_range'][0]:.2f}-{100 * row['nt_range'][1]:.2f}% |")
cf = {}
for col in ('line', 'sex', 'genotype'):
    cf[col] = pd.crosstab(d[col], d.side).to_dict()
md += ['', 'Composition (pools per side): ' + '; '.join(f"{col}: " + ', '.join(f"{v} {int((d[col][d.annot == 'SG'] == v).sum())} vs {int((d[col][d.annot != 'SG'] == v).sum())}"
                                                                       for v in sorted(d[col].unique())) for col in ('line', 'sex', 'genotype')) +
       ' (select vs evaluate). Every pool has both odours x 3 phases (H / O / P), so phase and odour are balanced by construction. '
       f"Recording dates: select {d.date[d.annot == 'SG'].min()} to {d.date[d.annot == 'SG'].max()} (median {sorted(d.date[d.annot == 'SG'])[5]}), "
       f"evaluate {d.date[d.annot != 'SG'].min()} to {d.date[d.annot != 'SG'].max()}.", '']
res['describe'] = {'pools': d.reset_index().to_dict('records'), 'groups': desc, 'composition': cf}

# ---------------------------------------------------------------- 2. random splits
RULES = ('sgap', 'sgap_nocap', 'bestap')
RULEDOC = {'sgap': 'rule sgap as pre-registered (v29_prereg.md, raw firing < 10%)',
           'sgap_nocap': 'rule sgap WITHOUT the firing cap (= the v2.8 audit rule)',
           'bestap': 'best select-side AP per behaviour, no other constraint (may pick one neuron for both behaviours)'}
md += ['## 2. Random pool splits (POST HOC): pick on the select side (11 pools), score on the evaluate side (12 pools)', '',
       f'{len(RAND)} random splits, stratified by annotator (SG 11 pools -> 5 or 6 select, SD 5 -> 2 or 3, MF 3 -> 1 or 2, MB 2 -> 1, '
       'the two mixed-annotator pools (MB+CP, MB+MR) -> 1; exactly 11 select pools), the official split excluded. Each cell: '
       'mean +- SD [2.5-97.5 percentile] over the splits where the rule found a pick; "official" = the SG -> non-SG split; '
       '"pct" = fraction of random splits with a LOWER value than the official one. Three pick rules: ' +
       '; '.join(f'{k} = {v}' for k, v in RULEDOC.items()) + '.', '']
rand = {}
for rule in RULES:
    rand[rule] = {}
    for k, r in D.items():
        rand[rule][k] = {}
        for b in BEH:
            c = BEH.index(b)
            vals = {m: [] for m in MET + ('chance_ap', 'chance_topk', 'sel_ap')}
            ids = []
            for s in RAND:
                p = r['picks'][rule][s]
                if p is None:
                    ids.append(None)
                    for m in vals:
                        vals[m].append(float('nan'))
                    continue
                j = p[c]
                ids.append(int(j))
                mt = metrics(r, j, b, f'{s}|ev')
                for m in MET + ('chance_ap', 'chance_topk'):
                    vals[m].append(mt[m])
                vals['sel_ap'].append(float(r['ap'][j][QI[f'{s}|sel'], c]))
            po = r['picks'][rule]['official']
            off = metrics(r, po[c], b, 'official|ev') if po else None
            ids_c = pd.Series([i for i in ids if i is not None], dtype=float).value_counts()
            x = {'summary': {m: summ(v) for m, v in vals.items() if np.isfinite(v).sum() > 1}, 'values': vals, 'ids': ids,
                 'official_pick': int(po[c]) if po else None, 'official': off,
                 'official_sel_ap': float(r['ap'][po[c]][QI['official|sel'], c]) if po else None,
                 'frac_same_as_official': float(np.mean([i is not None and po is not None and i == po[c] for i in ids])),
                 'top_ids': {int(i): int(n) for i, n in ids_c.head(3).items()}, 'n_no_pick': int(sum(i is None for i in ids))}
            if off:
                x['official_pct'] = {m: float(np.nanmean(np.where(np.isfinite(vals[m]), np.asarray(vals[m]) < off[m], np.nan))) for m in MET}
            rand[rule][k][b] = x
res['random'] = rand


def cellfmt(s, m):
    if m not in s:
        return 'n/a'
    s = s[m]
    return f"{s['mean']:.3f} +- {s['sd']:.3f} [{s['p2.5']:.3f}, {s['p97.5']:.3f}]"


for rule in RULES:
    for b in BEH:
        md += [f'### {BN[b]}, {rule}', '',
               '| dictionary | splits without a pick | official pick | splits picking it | most frequent picks (count) | ' +
               ' | '.join(f'{MN[m]} random' for m in MET) + ' | ' + ' | '.join(f'{MN[m]} official (pct)' for m in MET) +
               ' | select-side AP random | select-side AP official |', '|' + '---|' * (7 + 2 * len(MET))]
        for k in ALL:
            if k not in D:
                continue
            x = rand[rule][k][b]
            s = x['summary']
            offc = [f"{x['official'][m]:.3f} ({x['official_pct'][m]:.2f})" if x['official'] else 'no pick' for m in MET]
            md.append(f"| {k} | {x['n_no_pick']}/{len(RAND)} | {'#' + str(x['official_pick']) if x['official_pick'] is not None else 'none'} | "
                      f"{x['frac_same_as_official']:.2f} | " + ', '.join(f'#{i} ({n})' for i, n in x['top_ids'].items()) +
                      f" | {' | '.join(cellfmt(s, m) for m in MET)} | {' | '.join(offc)} | {cellfmt(s, 'sel_ap')} | {f3(x['official_sel_ap'])} |")
        md.append('')
for b in BEH:
    x = rand['bestap']['PREDk_s2703'][b]
    md += [f"Chance on the evaluate side ({BN[b]}): AP (= prevalence) random {cellfmt(x['summary'], 'chance_ap')}, official {x['official']['chance_ap']:.4f}; "
           f"precision@top-k/video chance random {cellfmt(x['summary'], 'chance_topk')}, official {x['official']['chance_topk']:.4f}."]
md.append('')

# optimism: select-side minus evaluate-side AP of the pick, official vs random (sgap; PREDk s2703 and the v2 mean)
md += ['### Selection optimism (select-side AP minus evaluate-side AP of the same pick; rule sgap)', '',
       '| dictionary | behaviour | random: mean +- SD | official |', '|---|---|---|---|']
opt = {}
for k in V2:
    for b in BEH:
        x = rand['sgap'][k][b]
        dl = np.asarray(x['values']['sel_ap']) - np.asarray(x['values']['ap'])
        o = (x['official_sel_ap'] - x['official']['ap']) if x['official'] else float('nan')
        if np.isfinite(dl).sum() > 1:
            opt[f'{k}|{b}'] = {**summ(dl), 'official': o}
            md.append(f"| {k} | {BN[b]} | {np.nanmean(dl):+.3f} +- {np.nanstd(dl, ddof=1):.3f} | {o:+.3f} |")
res['optimism'] = opt
md.append('')

# dictionary ranking by select-side AP sum (sgap, as pre-registered: no pair -> 0)
md += ['### Choosing one dictionary per split by the select-side (nose-nose AP + nose-tail AP) of its sgap pair (no pair -> 0)', '']
rank = {}
ks = [k for k in V2 if k in D]
ch = [max(ks, key=lambda k: sel_ap_sum(D[k], s)) for s in RAND]
cnt = pd.Series(ch).value_counts()
offc = max(ks, key=lambda k: sel_ap_sum(D[k], 'official'))
ev = [{b: metrics(D[k], D[k]['picks']['sgap'][s][BEH.index(b)], b, f'{s}|ev') for b in BEH} for s, k in zip(RAND, ch)]
rank = {'chosen_counts': {k: int(n) for k, n in cnt.items()}, 'official_choice': offc, 'chosen': ch,
        'chosen_eval': {f'{b}|{m}': summ([e[b][m] for e in ev]) for b in BEH for m in MET}}
md.append('Among the 12 v2 dictionaries: chosen in the random splits: ' + ', '.join(f'{k} {n}/{len(RAND)}' for k, n in cnt.items()) +
          f'; the official split chooses {offc}. Evaluate-side score of the chosen dictionary\'s pair: ' +
          '; '.join(f"{BN[b]} AP {cellfmt(rank['chosen_eval'], f'{b}|ap')}, precision@top3/video {cellfmt(rank['chosen_eval'], f'{b}|top3')}" for b in BEH) + '.')
md.append('')
res['ranking'] = rank

# verdict PREDk s2703 vs v1 s0 per split
md += ['### Verdict "v2 PREDk s2703 beats v1 s0" per random split (same select pools, same evaluate pools)', '',
       'PREDk s2703 re-picks with rule sgap on every split. v1 s0 has no sgap pair (its candidates all fire on > 10% of frames), so it '
       'is compared three ways: v1 re-picked with sgap without the cap, v1 re-picked by best select AP, and v1\'s documented neurons '
       '#611 / #414 (fixed; picked on SG pools in v2.5, so SG pools on the evaluate side favour v1 slightly). Difference = PREDk s2703 minus v1.', '',
       '| behaviour | metric | v1 comparator | difference mean +- SD [2.5-97.5] | fraction of splits PREDk s2703 > v1 | official split difference |',
       '|---|---|---|---|---|---|']
verdict = {}
if 'v1_s0' in D:
    rv = D['v1_s0']
    for b in BEH:
        c = BEH.index(b)
        a_ = rand['sgap']['PREDk_s2703'][b]
        for m in MET:
            comps = {'sgap_nocap': (np.asarray(rand['sgap_nocap']['v1_s0'][b]['values'][m]), rand['sgap_nocap']['v1_s0'][b]['official']),
                     'bestap': (np.asarray(rand['bestap']['v1_s0'][b]['values'][m]), rand['bestap']['v1_s0'][b]['official']),
                     'fixed #611/#414': (np.asarray([metrics(rv, rv['fixed'][b], b, f'{s}|ev')[m] for s in RAND]), metrics(rv, rv['fixed'][b], b, 'official|ev'))}
            for cn, (v_, vo) in comps.items():
                dlt = np.asarray(a_['values'][m]) - v_
                od = a_['official'][m] - vo[m] if (a_['official'] and vo) else float('nan')
                ok = np.isfinite(dlt)
                s = summ(dlt)
                verdict[f'{b}|{m}|{cn}'] = {**s, 'frac_pos': float(np.mean(dlt[ok] > 0)), 'n': int(ok.sum()), 'official': float(od)}
                md.append(f"| {BN[b]} | {MN[m]} | {cn} | {s['mean']:+.3f} +- {s['sd']:.3f} [{s['p2.5']:+.3f}, {s['p97.5']:+.3f}] | "
                          f"{np.mean(dlt[ok] > 0):.2f} ({ok.sum()} splits) | {od:+.3f} |")
    md.append('')
res['verdict_v2_vs_v1'] = verdict

# ---------------------------------------------------------------- 3. fixed neurons per annotator / pool
md += ['## 3. Fixed neurons, metrics per annotator and per pool on all 23 annotated pools', '',
       'CAUTION: PREDk s2703 #3593 / #3009 and V s1002 #436 / #270 were PICKED on the SG pools; v1 #611 / #414 were picked on SG '
       'pools too (v2.5 rule auc). Their SG rows re-use selection data and are optimistic by construction; only the non-SG rows are held-out.', '']
fixed = []
for k in ('PREDk_s2703', 'V_s1002', 'v1_s0'):
    if k not in D:
        continue
    r = D[k]
    fixed.append((k, 'nose_nose', r['fixed']['nose_nose'], 'fixed'))
    fixed.append((k, 'nt', r['fixed']['nt'], 'fixed'))
    if k == 'v1_s0':
        po = r['picks']['sgap_nocap']['official']
        if po:
            fixed += [(k, 'nose_nose', po[0], 'sgap no cap'), (k, 'nt', po[1], 'sgap no cap')]
        fixed += [(k, b, r['best_ap_official'][b], 'best SG AP') for b in BEH]
seen, fx = set(), []
for t in fixed:
    if (t[0], t[1], t[2]) not in seen:
        seen.add((t[0], t[1], t[2]))
        fx.append(t)
ann_sub = [n for n in names if n.startswith('annot_')]
pool_sub = [n for n in names if n.startswith('pool_')]
fixres = {}
for b in BEH:
    md += [f'### {BN[b]}: per annotator (videos annotated by that person; AUROC / AP / precision@top3/video)', '',
           '| neuron | ' + ' | '.join(f"{a.split('_')[1]} ({metrics(D0, fx[0][2], b, a)['videos']} videos)" for a in ann_sub) + ' | non-SG pools | all 23 pools |',
           '|' + '---|' * (len(ann_sub) + 3)]
    md.append('| prevalence (= AP chance) | ' + ' | '.join(f"{100 * metrics(D0, fx[0][2], b, a)['chance_ap']:.2f}%" for a in ann_sub + ['official|ev', 'all']) + ' |')
    for k, bb, j, why in fx:
        if bb != b:
            continue
        r = D[k]
        row = []
        for a in ann_sub + ['official|ev', 'all']:
            mt = metrics(r, j, b, a)
            fixres[f'{k}|#{j}|{b}|{a}'] = mt
            row.append(f"{mt['auc']:.3f} / {mt['ap']:.3f} / {f3(mt['top3'])}")
        md.append(f"| {k} #{j} ({why}) | " + ' | '.join(row) + ' |')
    md += ['', f'### {BN[b]}: per pool (AP / precision@top3/video; AP lift = AP / prevalence in brackets)', '',
           '| pool | annotator | prevalence | ' + ' | '.join(f'{k} #{j} ({why})' for k, bb, j, why in fx if bb == b) + ' |',
           '|' + '---|' * (3 + sum(1 for t in fx if t[1] == b))]
    for ps in pool_sub:
        u = ps[5:]
        row = []
        for k, bb, j, why in fx:
            if bb != b:
                continue
            mt = metrics(D[k], j, b, ps)
            fixres[f'{k}|#{j}|{b}|{ps}'] = mt
            row.append(f"{mt['auc']:.3f} / {mt['ap']:.3f} ({mt['ap'] / mt['chance_ap']:.1f}x) / {f3(mt['top3'])}" if mt['chance_ap'] > 0 else 'n/a')
        md.append(f"| {d.pool.get(u, u)} | {pann[u]} | {100 * metrics(D0, fx[0][2], b, ps)['chance_ap']:.2f}% | " + ' | '.join(row) + ' |')
    md += ['(per-pool cells: AUROC / AP (AP lift) / precision@top3/video)', '']
res['fixed'] = fixres

# ---------------------------------------------------------------- 4. leave one annotator out
md += ['## 4. Leave one annotator out (select = all pools without that annotator, evaluate = that annotator\'s pools)', '',
       'MB evaluate side = rd24, rd28 and the two mixed pools rd11_2 (MB + CP) and rd14 (MB + MR). Cells: pick / AUROC / AP / '
       'precision@top3/video / precision@top10/video. Rows: every dictionary with rule sgap (pre-registered), then v1 s0 with the '
       'fallback rules and its fixed documented neurons.', '']
lo = {}
rows_ = [(k, 'sgap') for k in ALL if k in D] + [('v1_s0', 'sgap_nocap'), ('v1_s0', 'bestap'), ('v1_s0', 'fixed')]
for b in BEH:
    md += [f'### {BN[b]}', '', '| dictionary (rule) | ' + ' | '.join(f"leave out {s.split('_')[1]}" for s in LOAO) + ' |', '|' + '---|' * (1 + len(LOAO))]
    mt0 = [metrics(D0, D0['neurons'][0], b, f'{s}|ev') for s in LOAO]
    md.append('| evaluate side: pools, videos with >= 1 positive, frames, chance AP / precision@top-k/video | ' + ' | '.join(
        f"{len({D0['cpool'][g] for g in np.flatnonzero(Msub[QI[s + '|ev']])})} pools, {m['videos_pos']} videos, {m['frames']:,} frames, {m['chance_ap']:.4f} / {m['chance_topk']:.4f}"
        for s, m in zip(LOAO, mt0)) + ' |')
    for k, rule in rows_:
        if k not in D:
            continue
        r = D[k]
        row = []
        for s in LOAO:
            j = r['fixed'][b] if rule == 'fixed' else (r['picks'][rule][s][BEH.index(b)] if r['picks'][rule][s] else None)
            if j is None:
                row.append('no pick')
                continue
            mt = metrics(r, j, b, f'{s}|ev')
            lo[f'{k}|{rule}|{b}|{s}'] = {'pick': int(j), **mt}
            row.append(f"#{j} / {mt['auc']:.3f} / {mt['ap']:.3f} / {mt['top3']:.3f} / {mt['top10']:.3f}")
        md.append(f'| {k} ({rule}) | ' + ' | '.join(row) + ' |')
    md.append('')
ks = [k for k in V2 if k in D]
for s in LOAO:
    lo[f'chosen|{s}'] = max(ks, key=lambda kk: sel_ap_sum(D[kk], s))
md += ['v2 dictionary chosen by select-side sgap AP sum: ' + ', '.join(f"leave out {s.split('_')[1]}: {lo[f'chosen|{s}']}" for s in LOAO) + '.', '']
res['loao'] = lo
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'split.json').write_text(json.dumps(res, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)))
(OUT / 'tables.md').write_text('\n'.join(md) + '\n')
print('\n'.join(md))
