"""
ECI v2-temporary NES: v1 vs v2 side by side per contrast (reads result.json of both runs; no compute beyond JSON).
v1 = the published NES result sets (mice results/vision/mice/eci/nes/matryoshka_btk_1024_k16_fg448al_s0, ants
results/vision/ants/eci30/nes/matryoshka_btk_1024_k16_antsfg_s0); v2 = results/vision/eci_v2_nes/<dom>/nes/<sae>.
Per analysis x outcome (per-video mean activation / bout rate / latency; the primary setting of each runner) x prefix:
the selected neurons in round order (tau, 95% CI = tau +- t(df, 0.975) se, p, Bonferroni threshold), and the
behaviour neurons' FIRST-ROUND statistics at the full dictionary whether selected or not.
Writes results/vision/eci_v2_nes/compare.json and tables.md.
  python scripts/eci/v2/v2nes_tables.py [--check-v1]   (--check-v1: v1check/ rerun == published, every setting)
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path('/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference')
RV = REPO / 'results/vision'
OUT = RV / 'eci_v2_nes'
RUNS = {
    'mice': {'v1': (RV / 'mice/eci/nes/matryoshka_btk_1024_k16_fg448al_s0', (128, 256, 1024)),
             'v2': (OUT / 'mice/nes/v2tmp_PREDk16_s2901', (128, 256, 512, 2048, 4096))},
    'ants': {'v1': (RV / 'ants/eci30/nes/matryoshka_btk_1024_k16_antsfg_s0', (128, 256, 1024)),
             'v2': (OUT / 'ants/nes/v2tmp_PREDk_s2703', (128, 256, 512, 2048, 4096))}}
# behaviour neurons (user-facing names): v2 = the v2.9 / v2.7 picks; v1 = the v1 s0 neurons of the v2.9 alignment cards
BEH = {'mice': {'v1': {611: 'nose-nose', 414: 'nose-tail'}, 'v2': {3461: 'nose-nose', 1780: 'nose-tail'}},
       'ants': {'v1': {90: 'grooming-to-focal'}, 'v2': {33: 'grooming-to-focal'}}}
ANALYSES = {
    'mice': [f'A_{g}_{t}' for g in ('het', 'wt') for t in ('1to2', '2to3', '4to5', '5to6')] + [f'B_stage{s}' for s in range(1, 7)],
    # v3_2_vs_6_8 (t = 6 and 8 pooled) is computed by the runner's core set but not reported (user: never pool 6 + 8)
    'ants': ['v2_1_vs_2', 'v3_2_vs_6', 'v3_2_vs_8', 'v4_2_vs_6', 'v4_infested_2_vs_6', 'infested_vs_healthy_t2',
             'infested_vs_healthy_t6', 'infested_vs_healthy_t2_6']}
OUTCOMES = {'mean': ('.', 'p{P}_max_mean_t_bonferroni_full', 'per-video mean activation'),
            'bouts': ('maxpool_bouts', 'p{P}_bout_rate_q0.95_g0_t_bonferroni_full', 'bout rate (bouts/min)'),
            'latency': ('maxpool_latency', 'p{P}_latency_q0.95_t_bonferroni_common', 'latency to first bout (s)')}
LABEL = {'A_het_1to2': 'het, social habituation -> odour', 'A_het_2to3': 'het, social odour -> post',
         'A_het_4to5': 'het, fear habituation -> odour', 'A_het_5to6': 'het, fear odour -> post',
         'A_wt_1to2': 'wt, social habituation -> odour', 'A_wt_2to3': 'wt, social odour -> post',
         'A_wt_4to5': 'wt, fear habituation -> odour', 'A_wt_5to6': 'wt, fear odour -> post',
         **{f'B_stage{s}': f'het vs wt, stage {s}' for s in range(1, 7)},
         'v2_1_vs_2': 'v2 (ISTAnt) treatment 1 vs 2', 'v3_2_vs_6': 'v3 t=2 vs t=6',
         'v3_2_vs_8': 'v3 t=2 vs t=8 (confounded with recording day)', 'v4_2_vs_6': 'v4 t=2 vs t=6',
         'v4_infested_2_vs_6': 'v4_infested t=2 vs t=6', 'infested_vs_healthy_t2': 'infested vs healthy, t=2 (not day-matched)',
         'infested_vs_healthy_t6': 'infested vs healthy, t=6 (not day-matched)',
         'infested_vs_healthy_t2_6': 'infested vs healthy, t=2+6 (not day-matched)'}


def ci(tau, se, df):
    h = float(stats.t.ppf(0.975, df)) * se
    return [tau - h, tau + h]


def one(res, beh):
    r = res['rounds']
    sel = [{'round': int(r['round'][i]), 'neuron': int(r['neuron'][i]), 'tau': r['tau'][i], 'se': r['se'][i],
            'ci': ci(r['tau'][i], r['se'][i], r['df'][i]), 'p': r['p'][i], 'threshold': r['threshold'][i]}
           for i in range(len(r.get('neuron', [])))]
    fr = res['first_round'] or {}
    idx = {int(j): i for i, j in enumerate(fr.get('neuron', []))}
    b = {}
    for j, name in beh.items():
        if j not in idx:
            b[name] = {'neuron': j, 'tested': False}
            continue
        i = idx[j]
        b[name] = {'neuron': j, 'tested': True, 'tau': fr['tau'][i], 'se': fr['se'][i], 'ci': ci(fr['tau'][i], fr['se'][i], fr['df'][i]),
                   'p': fr['p'][i], 'bonf_threshold_round1': 0.05 / res['n_tested'], 'selected': j in [s['neuron'] for s in sel],
                   'rank_by_abs_tau_round1': int(1 + np.sum(np.abs(np.array(fr['tau'], float)) > abs(fr['tau'][i])))}
    return {'n_units': res.get('n_units'), 'n_tested': res['n_tested'], 'stopped': res.get('stopped', ''), 'selected': sel, 'behaviour': b}


def collect():
    out = {}
    for dom, runs in RUNS.items():
        out[dom] = {}
        for ver, (root, prefs) in runs.items():
            for oc, (sub, pat, _) in OUTCOMES.items():
                for aid in ANALYSES[dom]:
                    f = root / sub / aid / 'result.json'
                    if not f.exists():
                        continue
                    R = json.loads(f.read_text())
                    for P in prefs:
                        k = pat.format(P=P)
                        if k not in R:
                            continue
                        beh = {j: n for j, n in BEH[dom][ver].items() if j < P}
                        out[dom].setdefault(aid, {}).setdefault(oc, {}).setdefault(ver, {})[str(P)] = one(R[k], beh)
    return out


def fmt_sel(x):
    if not x['selected']:
        return 'none'
    return '; '.join(f"#{s['neuron']} {s['tau']:+.3g} [{s['ci'][0]:+.3g}, {s['ci'][1]:+.3g}] p={s['p']:.1e}" for s in x['selected'])


def fmt_beh(x):
    parts = []
    for name, b in x['behaviour'].items():
        if not b.get('tested'):
            parts.append(f"{name} #{b['neuron']}: not tested")
            continue
        parts.append(f"{name} #{b['neuron']}: {b['tau']:+.3g} [{b['ci'][0]:+.3g}, {b['ci'][1]:+.3g}] p={b['p']:.1e}"
                     f"{' SELECTED' if b['selected'] else ''} (rank {b['rank_by_abs_tau_round1']} by |tau|)")
    return '; '.join(parts) or '-'


def tables(C):
    L = ['# ECI v2-temporary NES: v1 vs v2 per contrast', '',
         'Primary setting of each runner (codes_max pooling, t-test, Bonferroni alpha 0.05, full window; latency: common '
         'window), nuisance none, selection by |tau| (the v1 settings). Cells: selected neurons in round order, tau '
         '[95% CI] and p; behaviour neurons: FIRST-ROUND tau [95% CI], p and rank by |tau| among the tested neurons, '
         'at the full dictionary (v1 1024, v2 4096), selected or not. tau units: mean activation = code units of each '
         'dictionary (not comparable across v1 / v2); bout rate = bouts/min; latency = s.', '',
         'Dictionaries: v1 mice matryoshka_btk_1024_k16_fg448al_s0, v1 ants matryoshka_btk_1024_k16_antsfg_s0; v2 mice '
         'PREDk16_s2901 (nose-nose #3461, nose-tail #1780), v2 ants PREDk_s2703 (grooming-to-focal #33). v1 behaviour '
         'neurons for reference: mice nose-nose #611, nose-tail #414; ants grooming #90.', '']
    for dom in C:
        L += [f'## {dom}', '']
        for oc, (_, _, oname) in OUTCOMES.items():
            rows = [(aid, C[dom][aid][oc]) for aid in ANALYSES[dom] if aid in C[dom] and oc in C[dom][aid]]
            if not rows:
                continue
            L += [f'### {oname}', '', '| contrast | n | v1 selected (full 1024) | v2 selected (full 4096) | v1 behaviour neurons | v2 behaviour neurons |',
                  '|---|---|---|---|---|---|']
            for aid, d in rows:
                v1 = d.get('v1', {}).get('1024')
                v2 = d.get('v2', {}).get('4096')
                n = (v2 or v1 or {}).get('n_units', '')
                L.append(f"| {aid} ({LABEL.get(aid, '')}) | {n} | {fmt_sel(v1) if v1 else 'n/a'} | {fmt_sel(v2) if v2 else 'pending'} | "
                         f"{fmt_beh(v1) if v1 else 'n/a'} | {fmt_beh(v2) if v2 else 'pending'} |")
            L += ['', 'Smaller prefixes (selected neurons): ', '']
            L += ['| contrast | ' + ' | '.join([f'v1 p{P}' for P in (128, 256)] + [f'v2 p{P}' for P in (128, 256, 512, 2048)]) + ' |',
                  '|---|' + '---|' * 6]
            for aid, d in rows:
                cells = [fmt_sel(d['v1'][str(P)]) if str(P) in d.get('v1', {}) else '-' for P in (128, 256)]
                cells += [fmt_sel(d['v2'][str(P)]) if str(P) in d.get('v2', {}) else '-' for P in (128, 256, 512, 2048)]
                L.append(f'| {aid} | ' + ' | '.join(cells) + ' |')
            L.append('')
    return '\n'.join(L) + '\n'


def check_v1():
    """v1check/<dom>/<sae>/summary.csv (rerun with today's code) vs the published summary.csv: identical rows?"""
    import pandas as pd
    rep = {}
    for dom, sub in (('mice', 'mice/eci/nes/matryoshka_btk_1024_k16_fg448al_s0'),
                     ('ants', 'ants/eci30/nes/matryoshka_btk_1024_k16_antsfg_s0')):
        a = pd.read_csv(RV / sub / 'summary.csv')
        f = OUT / 'v1check' / dom / Path(sub).name / 'summary.csv'
        if not f.exists():
            rep[dom] = 'missing'
            continue
        b = pd.read_csv(f)
        # 'confound' is a free-text label reworded after the v1 runs (commit a781b20); not part of the result
        key = [c for c in a.columns if c not in ('tau', 'se', 't', 'df', 'p', 'threshold', 'confound')]
        a[key], b[key] = a[key].astype(str), b[key].astype(str)  # NaN-safe exact key match
        m = a.merge(b, on=key, how='outer', suffixes=('_pub', '_new'), indicator=True)
        num = {c: float(np.nanmax(np.abs(m[f'{c}_pub'] - m[f'{c}_new']))) if (m['_merge'] == 'both').any() else None
               for c in ('tau', 'se', 'p')}
        prim = m[(m.pooling == 'max') & (m.test == 't') & (m.correction == 'bonferroni') & (m.window == 'full')
                 & (m.outcome_type == 'mean')] if 'pooling' in m else m
        rep[dom] = {'rows_published': len(a), 'rows_rerun': len(b), 'only_published': int((m['_merge'] == 'left_only').sum()),
                    'only_rerun': int((m['_merge'] == 'right_only').sum()), 'max_abs_diff': num,
                    'primary_rows_matched': int((prim['_merge'] == 'both').sum()), 'primary_rows': len(prim)}
    return rep


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--check-v1', action='store_true')
    a = ap.parse_args()
    C = collect()
    if a.check_v1:
        C['_v1check'] = check_v1()
        print(json.dumps(C['_v1check'], indent=1))
    (OUT / 'compare.json').write_text(json.dumps(C, indent=1, default=float))
    (OUT / 'tables.md').write_text(tables({k: v for k, v in C.items() if not k.startswith('_')}))
    print('wrote', OUT / 'compare.json', OUT / 'tables.md')
