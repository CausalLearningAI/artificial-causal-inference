"""
ECI v2.8 audit: combine the per-dictionary pickles of v28_audit_neurons.py into tables (POST HOC, analysis only).
-> results/vision/eci_v2_8/audit/{audit.json, tables.md}
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / 'results/vision/eci_v2_8/audit'
WORK = OUT / 'work'
PREF = (128, 256, 512, 2048, 4096, 8192)
BEH = ('nose_nose', 'nt')
BN = {'nose_nose': 'nose-nose', 'nt': 'nose-tail'}
RULES = ('auc', 'dist', 'dist2')
KS = (3, 10, 30)


def level(key, j):
    if key.startswith('v1_'):
        return '-'
    return str(1 + int(np.searchsorted(PREF, j, side='right')))


def arm_of(k):
    return k.rsplit('_s', 1)[0]


def f3(x):
    return 'n/a' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{x:.3f}'


def ci(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))], float(v.std(ddof=1))


def mice():
    D = {}
    for p in sorted(WORK.glob('mice_*.pkl')):
        r = pickle.load(open(p, 'rb'))
        D[r['key']] = r
    rows = []
    for k, r in D.items():
        S, T = r['stats'], r['topk']
        for rule in RULES:
            pk = r['picks'][rule]
            row = {'key': k, 'arm': arm_of(k), 'rule': rule, 'distinct': pk['nose_nose'] != pk['nt']}
            for b in BEH:
                j = pk[b]
                row[f'{b}|neuron'] = j
                row[f'{b}|level'] = level(k, j)
                for nm in ('sg', 'non'):
                    row[f'{b}|{nm}|auc'] = float(S[f'auc|{nm}|{b}'][j])
                    row[f'{b}|{nm}|ap'] = float(S[f'ap|{nm}|{b}'][j])
                    for kk in KS:
                        row[f'{b}|{nm}|top{kk}'] = T[(nm, b, j, kk)]
                row[f'{b}|non|disc'] = float(S['auc|non|disc'][j])
                bt = r['boot'][(j, b)]
                for kind in ('video', 'pool'):
                    for m in ('auc', 'ap') + tuple(f'top{kk}' for kk in KS):
                        c, sd = ci(bt[kind][m])
                        row[f'{b}|non|{m}|{kind}_ci'] = c
                        row[f'{b}|non|{m}|{kind}_sd'] = sd
            rows.append(row)
    # 1. per-dictionary table
    md = ['# ECI v2.8 audit (POST HOC, analysis only; scripts/eci/v2/v28_audit_neurons.py, v28_audit_combine.py)', '',
          'Mice frames: edge-excluded ext frames. SG = selection pools (annotator SG, 11 pools); held-out = 12 non-SG pools, '
          '72 videos, 431,712 frames (nose-nose prevalence 1.57%, 72 videos with >= 1 positive; nose-tail 0.85%, 63 videos). '
          'AUROC / AP on the P(on) readout (AP ties broken by raw code); precision@top-k/video = raw-code ranking, k frames '
          'per video with >= 1 positive (k = 3 is the official precision@top3/video).', '']
    md.append('## 1. Per dictionary, picks and scores (SG = selection pools, HO = held-out)')
    for rule in RULES:
        md += ['', f'### rule {rule}', '',
               '| dictionary | nose-nose neuron (level) | SG nose-nose AUROC | SG nose-nose precision@top3/video | HO nose-nose AUROC | HO nose-nose precision@top3/video | HO nose-nose AP | '
               'nose-tail neuron (level) | SG nose-tail AUROC | SG nose-tail precision@top3/video | HO nose-tail AUROC | HO nose-tail precision@top3/video | HO nose-tail AP | HO nose-tail precision@top10/video |',
               '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
        for row in rows:
            if row['rule'] != rule:
                continue
            md.append(f"| {row['key']} | #{row['nose_nose|neuron']} (L{row['nose_nose|level']}) | {f3(row['nose_nose|sg|auc'])} | {f3(row['nose_nose|sg|top3'])} | "
                      f"{f3(row['nose_nose|non|auc'])} | {f3(row['nose_nose|non|top3'])} | {f3(row['nose_nose|non|ap'])} | "
                      f"#{row['nt|neuron']} (L{row['nt|level']}) | {f3(row['nt|sg|auc'])} | {f3(row['nt|sg|top3'])} | {f3(row['nt|non|auc'])} | "
                      f"{f3(row['nt|non|top3'])} | {f3(row['nt|non|ap'])} | {f3(row['nt|non|top10'])} |")
    # 2. does the SG score predict the held-out score across dictionaries?
    corr = {}
    md += ['', '## 2. SG score vs held-out score across dictionaries (Spearman rho, picked neuron)', '',
           '| rule | behaviour | metric | rho | p | n dictionaries |', '|---|---|---|---|---|---|']
    for rule in RULES:
        R_ = [x for x in rows if x['rule'] == rule and not x['key'].startswith('v1_')]
        for b in BEH:
            for m in ('auc', 'ap', 'top3', 'top10', 'top30'):
                a = np.array([x[f'{b}|sg|{m}'] for x in R_])
                h = np.array([x[f'{b}|non|{m}'] for x in R_])
                rho, pv = spearmanr(a, h)
                corr[f'{rule}|{b}|{m}'] = [float(rho), float(pv), len(R_)]
                md.append(f'| {rule} | {BN[b]} | {m} | {rho:+.2f} | {pv:.3f} | {len(R_)} |')
    # within-dictionary: SG AUROC vs held-out AUROC across eligible neurons
    md += ['', 'Within a dictionary, across eligible neurons: Spearman of SG AUROC vs held-out AUROC (median over dictionaries): ']
    wc = {b: [] for b in BEH}
    for k, r in D.items():
        el = r['eligible']
        for b in BEH:
            wc[b].append(spearmanr(r['stats'][f'auc|sg|{b}'][el], r['stats'][f'auc|non|{b}'][el])[0])
    md[-1] += ', '.join(f'{BN[b]} {np.nanmedian(wc[b]):.2f} (range {np.nanmin(wc[b]):.2f} to {np.nanmax(wc[b]):.2f})' for b in BEH)
    # 3. choosing ONE dictionary from SG only
    md += ['', '## 3. Choosing one dictionary from SG pools only (rule dist picks; v1 excluded)', '',
           '| SG criterion | chosen | HO nose-nose AUROC | HO nose-nose precision@top3/video | HO nose-tail AUROC | HO nose-tail precision@top3/video | HO nose-tail AP | rank of chosen by HO nose-tail precision@top3/video |',
           '|---|---|---|---|---|---|---|---|']
    R_ = [x for x in rows if x['rule'] == 'dist' and not x['key'].startswith('v1_')]
    crit = {
        'max SG nose-tail precision@top3/video': lambda x: x['nt|sg|top3'],
        'max SG nose-tail AUROC': lambda x: x['nt|sg|auc'],
        'max SG (nose-nose AUROC + nose-tail AUROC)': lambda x: x['nose_nose|sg|auc'] + x['nt|sg|auc'],
        'max SG (nose-nose AP + nose-tail AP)': lambda x: x['nose_nose|sg|ap'] + x['nt|sg|ap'],
        'max SG min(nose-nose, nose-tail) precision@top10/video': lambda x: min(x['nose_nose|sg|top10'], x['nt|sg|top10']),
        'ORACLE: max HO nose-tail precision@top3/video': lambda x: x['nt|non|top3'],
        'ORACLE: max HO (nose-nose AUROC + nose-tail AUROC)': lambda x: x['nose_nose|non|auc'] + x['nt|non|auc'],
    }
    order = sorted(R_, key=lambda x: -x['nt|non|top3'])
    chosen = {}
    for nm, fn in crit.items():
        c = max(R_, key=fn)
        chosen[nm] = c['key']
        md.append(f"| {nm} | {c['key']} | {f3(c['nose_nose|non|auc'])} | {f3(c['nose_nose|non|top3'])} | {f3(c['nt|non|auc'])} | {f3(c['nt|non|top3'])} | "
                  f"{f3(c['nt|non|ap'])} | {1 + order.index(c)} / {len(order)} |")
    # 4. noise floor: one dictionary's bootstrap vs between-seed spread
    md += ['', '## 4. Noise floor (rule dist picks): bootstrap SD of ONE dictionary vs SD across seeds of an arm', '',
           '| behaviour | metric | median video-bootstrap SD (one dictionary) | median pool-bootstrap SD | median 95% CI width (video) | between-seed SD PREDc (3) | PRED26 (3) | V (6) | all v2.6+v2.7 PRED-type (12+3) |',
           '|---|---|---|---|---|---|---|---|---|']
    noise = {}
    for b in BEH:
        for m in ('auc', 'ap', 'top3', 'top10', 'top30'):
            vs = np.median([x[f'{b}|non|{m}|video_sd'] for x in R_])
            ps = np.median([x[f'{b}|non|{m}|pool_sd'] for x in R_])
            cw = np.median([x[f'{b}|non|{m}|video_ci'][1] - x[f'{b}|non|{m}|video_ci'][0] for x in R_])

            def sd_of(arms):
                v = [x[f'{b}|non|{m}'] for x in R_ if x['arm'] in arms]
                return float(np.std(v, ddof=1)) if len(v) > 1 else float('nan')
            s = {a: sd_of((a,)) for a in ('PREDc', 'PRED26', 'V')}
            s['pred_all'] = sd_of(('PREDc', 'PREDd', 'PREDk', 'PREDh', 'PRED26'))
            noise[f'{b}|{m}'] = {'video_sd': vs, 'pool_sd': ps, 'video_ci_width': cw, **s}
            md.append(f"| {BN[b]} | {m} | {vs:.3f} | {ps:.3f} | {cw:.3f} | {s['PREDc']:.3f} | {s['PRED26']:.3f} | {s['V']:.3f} | {s['pred_all']:.3f} |")
    # 5. is the pick noisy? top-5 candidates by SG AUROC
    md += ['', '## 5. Pick stability: top-5 neurons by SG AUROC per dictionary (held-out spread)', '',
           '| dictionary | behaviour | SG AUROC of #1..#5 | HO AUROC of #1..#5 | HO precision@top3/video of #1..#5 | HO AP of #1..#5 |', '|---|---|---|---|---|---|']
    pickstab = {}
    for k, r in D.items():
        el = r['eligible']
        for b in BEH:
            sa = np.where(el, r['stats'][f'auc|sg|{b}'], -np.inf)
            top5 = np.argsort(-sa)[:5]
            T = r['topk']
            hs = [T.get(('non', b, int(j), 3), np.nan) for j in top5]
            pickstab[f'{k}|{b}'] = {'neurons': top5.tolist(), 'sg_auc': [float(r['stats'][f'auc|sg|{b}'][j]) for j in top5],
                                    'ho_auc': [float(r['stats'][f'auc|non|{b}'][j]) for j in top5], 'ho_top3': hs,
                                    'ho_ap': [float(r['stats'][f'ap|non|{b}'][j]) for j in top5]}
            q = pickstab[f'{k}|{b}']
            md.append(f"| {k} | {BN[b]} | {' / '.join(f'{v:.3f}' for v in q['sg_auc'])} | {' / '.join(f'{v:.3f}' for v in q['ho_auc'])} | "
                      f"{' / '.join(f3(v) for v in q['ho_top3'])} | {' / '.join(f'{v:.3f}' for v in q['ho_ap'])} |")
    # 6. alternative SG-only pick rules (within the candidate set), dist-style distinctness for nose-tail
    md += ['', '## 6. Alternative SG-only pick rules (POST HOC), mean held-out over dictionaries (v1 excluded)', '',
           'Candidates per behaviour = top-20 by SG AUROC and top-20 by SG AP (plus the stored picks). Nose-tail must differ from the nose-nose pick '
           'and have SG discrimination AUROC >= 0.6 (as rule dist).', '',
           '| rule | HO nose-nose AUROC | HO nose-nose AP | HO nose-nose precision@top3/video | HO nose-tail AUROC | HO nose-tail AP | HO nose-tail precision@top3/video | HO nose-tail precision@top10/video | distinct |',
           '|---|---|---|---|---|---|---|---|---|']
    alt = {}
    for rn in ('sg_auc', 'sg_ap', 'sg_top10', 'rank_auc_ap', 'rank_auc_ap_top10', 'oracle_ho_auc'):
        vals = {f'{b}|{m}': [] for b in BEH for m in ('auc', 'ap', 'top3', 'top10')}
        nd, per = 0, {}
        for k, r in D.items():
            if k.startswith('v1_'):
                continue
            S, T, el = r['stats'], r['topk'], r['eligible']
            pk = {}
            for b in BEH:
                cand = [j for j in r['cand'][b] if el[j] and ('non', b, j, 3) in T]
                if b == 'nt':
                    cand = [j for j in cand if j != pk['nose_nose'] and S['auc|sg|disc'][j] >= 0.6] or cand
                sc_auc = np.array([S[f'auc|sg|{b}'][j] for j in cand])
                sc_ap = np.array([S[f'ap|sg|{b}'][j] for j in cand])
                sc_t10 = np.array([T[('sg', b, j, 10)] for j in cand])
                rk = lambda v: np.argsort(np.argsort(-np.nan_to_num(v, nan=-1)))  # noqa: E731
                score = {'sg_auc': sc_auc, 'sg_ap': sc_ap, 'sg_top10': sc_t10, 'rank_auc_ap': -(rk(sc_auc) + rk(sc_ap)),
                         'rank_auc_ap_top10': -(rk(sc_auc) + rk(sc_ap) + rk(sc_t10)),
                         'oracle_ho_auc': np.array([S[f'auc|non|{b}'][j] for j in cand])}[rn]
                pk[b] = cand[int(np.nanargmax(score))]
                vals[f'{b}|auc'].append(S[f'auc|non|{b}'][pk[b]])
                vals[f'{b}|ap'].append(S[f'ap|non|{b}'][pk[b]])
                vals[f'{b}|top3'].append(T[('non', b, pk[b], 3)])
                vals[f'{b}|top10'].append(T[('non', b, pk[b], 10)])
            nd += pk['nose_nose'] != pk['nt']
            per[k] = {b: [int(pk[b]), float(S[f'auc|non|{b}'][pk[b]]), T[('non', b, pk[b], 3)], float(S[f'ap|non|{b}'][pk[b]])] for b in BEH}
        alt[rn] = {'mean': {kk: float(np.nanmean(v)) for kk, v in vals.items()}, 'per_dict': per, 'distinct': [int(nd), len(per)]}
        a = alt[rn]['mean']
        md.append(f"| {rn} | {a['nose_nose|auc']:.3f} | {a['nose_nose|ap']:.3f} | {a['nose_nose|top3']:.3f} | {a['nt|auc']:.3f} | {a['nt|ap']:.3f} | "
                  f"{a['nt|top3']:.3f} | {a['nt|top10']:.3f} | {nd}/{len(per)} |")
    md += ['', 'Reference: rule dist (official), same dictionaries: ' + ', '.join(
        f"{BN[b]} HO AUROC {np.mean([x[f'{b}|non|auc'] for x in R_]):.3f}, AP {np.mean([x[f'{b}|non|ap'] for x in R_]):.3f}, "
        f"precision@top3/video {np.mean([x[f'{b}|non|top3'] for x in R_]):.3f}" for b in BEH)]
    # 6b. rule "sgap" (POST HOC, devised after the held-out numbers above were seen): per behaviour the candidate with the best
    # SG AP; nose-nose restricted to SG discrimination AUROC <= 0.4, nose-tail != nose-nose with SG discrimination >= 0.6;
    # dictionaries ranked by SG (nose-nose AP + nose-tail AP)
    md += ['', '## 6b. Rule sgap (POST HOC) per dictionary, ranked by SG (nose-nose AP + nose-tail AP)', '',
           '| dictionary | nose-nose neuron | SG nose-nose AUROC / AP / precision@top3/video | HO nose-nose AUROC / AP / precision@top3/video | '
           'nose-tail neuron | SG nose-tail AUROC / AP / precision@top3/video | HO nose-tail AUROC / AP / precision@top3/video |', '|---|---|---|---|---|---|---|']
    sgap = []
    for k, r in D.items():
        if k.startswith('v1_'):
            continue
        S, T, el = r['stats'], r['topk'], r['eligible']
        cn = [j for j in r['cand']['nose_nose'] if el[j] and ('sg', 'nose_nose', j, 3) in T and S['auc|sg|disc'][j] <= 0.4]
        nn = cn[int(np.nanargmax([S['ap|sg|nose_nose'][j] for j in cn]))]
        ct = [j for j in r['cand']['nt'] if el[j] and ('sg', 'nt', j, 3) in T and j != nn and S['auc|sg|disc'][j] >= 0.6]
        nt = ct[int(np.nanargmax([S['ap|sg|nt'][j] for j in ct]))]
        g = lambda b, j, nm: [float(S[f'auc|{nm}|{b}'][j]), float(S[f'ap|{nm}|{b}'][j]), T[(nm, b, j, 3)]]  # noqa: E731
        sgap.append({'key': k, 'nose_nose': nn, 'nt': nt, 'sg_nn': g('nose_nose', nn, 'sg'), 'sg_nt': g('nt', nt, 'sg'),
                     'ho_nn': g('nose_nose', nn, 'non'), 'ho_nt': g('nt', nt, 'non')})
    sgap.sort(key=lambda x: -(x['sg_nn'][1] + x['sg_nt'][1]))
    for x in sgap:
        md.append(f"| {x['key']} | #{x['nose_nose']} | {' / '.join(f3(v) for v in x['sg_nn'])} | {' / '.join(f3(v) for v in x['ho_nn'])} | "
                  f"#{x['nt']} | {' / '.join(f3(v) for v in x['sg_nt'])} | {' / '.join(f3(v) for v in x['ho_nt'])} |")
    rho = spearmanr([x['sg_nn'][1] + x['sg_nt'][1] for x in sgap], [x['ho_nn'][1] + x['ho_nt'][1] for x in sgap])
    md += ['', f'Spearman (SG AP sum vs held-out AP sum) over {len(sgap)} dictionaries: rho {rho[0]:+.2f} (p {rho[1]:.1e}).']
    res = {'rows': rows, 'corr_across_dicts': corr, 'sgap': sgap, 'within_dict_corr': {b: [float(x) for x in v] for b, v in wc.items()},
           'chosen': chosen, 'noise': noise, 'pick_stability': pickstab, 'alt_rules': alt,
           'picks_match_tag': {k: r['picks_match_tag'] for k, r in D.items()}}
    return res, md


def ants():
    D = {}
    for p in sorted(WORK.glob('ants_*.pkl')):
        r = pickle.load(open(p, 'rb'))
        D[r['key']] = r
    md = ['', '## 7. Ants grooming-to-focal: per dictionary (pick = best selection-half AUROC for grooming-to-focal (any); v3 + v4, 1 fps)', '',
          '| dictionary | direction | neuron (level) | stored pick | sel AUROC any | HO AUROC any [video 95% CI] | HO AUROC single | HO AUROC double | HO precision@top3/video any [CI] | single | double | HO AP any |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|']
    rows = []
    for k, r in D.items():
        S = r['stats']
        for dn, d in r['dirs'].items():
            s_, e_ = ('A', 'B') if dn == 'A2B' else ('B', 'A')
            j = d['pick']
            row = {'key': k, 'arm': arm_of(k), 'dir': dn, 'neuron': j, 'level': level(k, j), 'stored': d['stored_pick'],
                   'sel_auc': float(S[f'auc|{s_}|groom_any'][j]), 'sel_top3': d['top3'][('sel', 'groom_any', j)]}
            for b in ('groom_any', 'single', 'double'):
                row[f'{b}|auc'] = float(S[f'auc|{e_}|{b}'][j])
                row[f'{b}|ap'] = float(S[f'ap|{e_}|{b}'][j])
                row[f'{b}|top3'] = d['top3'][('held', b, j)]
                for m in ('auc', 'top3', 'ap'):
                    row[f'{b}|{m}|ci'], row[f'{b}|{m}|sd'] = ci(d['boot'][b][m])
            rows.append(row)
            md.append(f"| {k} | {dn} | #{j} (L{row['level']}) | {d['stored_pick']} | {row['sel_auc']:.3f} | {row['groom_any|auc']:.3f} [{row['groom_any|auc|ci'][0]:.3f}, {row['groom_any|auc|ci'][1]:.3f}] | "
                      f"{row['single|auc']:.3f} | {row['double|auc']:.3f} | {row['groom_any|top3']:.3f} [{row['groom_any|top3|ci'][0]:.3f}, {row['groom_any|top3|ci'][1]:.3f}] | "
                      f"{row['single|top3']:.3f} | {row['double|top3']:.3f} | {row['groom_any|ap']:.3f} |")
    corr = {}
    for dn in ('A2B', 'B2A'):
        R_ = [x for x in rows if x['dir'] == dn]
        if len(R_) > 2:
            corr[dn] = {'auc': list(map(float, spearmanr([x['sel_auc'] for x in R_], [x['groom_any|auc'] for x in R_]))),
                        'top3': list(map(float, spearmanr([x['sel_top3'] for x in R_], [x['groom_any|top3'] for x in R_]))), 'n': len(R_)}
    md += ['', 'Selection-half vs held-out half across dictionaries (Spearman rho, p): ' + json.dumps(corr)]
    md += ['', 'Ants noise: median video-bootstrap SD of HO AUROC any ' + f"{np.median([x['groom_any|auc|sd'] for x in rows]):.3f}" +
           ', of HO precision@top3/video any ' + f"{np.median([x['groom_any|top3|sd'] for x in rows]):.3f}"]
    return {'rows': rows, 'corr': corr}, md


if __name__ == '__main__':
    rm, mdm = mice()
    ra, mda = ants()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'audit.json').write_text(json.dumps({'mice': rm, 'ants': ra}, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)))
    (OUT / 'tables.md').write_text('\n'.join(mdm + mda) + '\n')
    print('\n'.join(mdm + mda))
