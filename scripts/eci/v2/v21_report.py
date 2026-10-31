"""
ECI v2.1 four-arm report (rules: scripts/eci/v2/v21_train.py pre-registration). CPU, reads small json files only.
-> results/vision/eci_v2_1/report.md, report.json
"""
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / 'results/vision/eci_v2_1'
G = REPO / 'results/vision/eci_v2_0B/gate'
ARMS, SEEDS = ('S', 'A', 'B', 'V'), (1001, 1002, 1003)
CELLS = {'mice': ('v1|nn', 'v1|nt', 'v1|np', 'v1|nose_nose'), 'ants': None}
ONLID = ('YOL', 'BOL', 'FOL', 'onlid_any', 'lid_any3')


def mr(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return f'{np.median(v):.3f} [{min(v):.3f}, {max(v):.3f}]' if v else '-'


def main():
    rep, L = {}, []
    gate = json.loads((G / 'gate.json').read_text())
    dynold = json.loads((REPO / 'results/vision/eci_v2_0B/v1_vs_v2/dyn.json').read_text())
    for dom in ('mice', 'ants'):
        R = {}
        for arm in ARMS:
            for s in SEEDS:
                f = OUT / dom / f'{arm}_s{s}.json'
                if f.exists():
                    R[(arm, s)] = json.loads(f.read_text())
        ref = gate[dom]
        el = lambda g: {c: r for c, r in g.items() if r['eligible'] and c.split('|')[1] not in ONLID}  # noqa: E731
        cells = sorted(el(ref['hmm']))
        L += [f'## {dom}', '', '| arm | gate PASS_final P(on) (median [range] of eligible cells) | gate cont | mean AUROC P(on) | '
              'mean AUROC cont | mean top-1% prec P(on) | mean nAP P(on) / cont | mean F1max P(on) / cont | prox strat AUROC (ants) | '
              'dyn share @256 | FVE |', '|' + '---|' * 11]
        # reference rows
        v1a = np.mean([ref['hmm'][c]['v1_AUROC'] for c in cells])
        v1t = np.mean([ref['hmm'][c]['v1_top1_prec'] for c in cells])
        v1n = np.mean([ref['hmm'][c]['v1_nAP'] for c in cells])
        L.append(f'| v1 (11 seeds, mean) | - | - | {v1a:.3f} | - | {v1t:.3f} | {v1n:.3f} | - | - | '
                 f'{np.mean([x["dynamic_256"] / x["eligible_256"] for k, x in dynold[dom]["seeds"].items() if k.startswith("v1")]):.3f} | - |')
        rows = {}
        pm = {k: sum(r['PASS_final'] for r in el(ref[k]).values()) for k in ('hmm', 'smooth')}
        L.append(f'| v2.0 PASS B (old store, N2) | {pm["hmm"]} / {len(cells)} | {pm["smooth"]} / {len(cells)} | '
                 f'{np.mean([ref["hmm"][c]["v2_AUROC"] for c in cells]):.3f} | {np.mean([ref["smooth"][c]["v2_AUROC"] for c in cells]):.3f} | '
                 f'{np.mean([ref["hmm"][c]["v2_top1_prec"] for c in cells]):.3f} | {np.mean([ref["hmm"][c]["v2_nAP"] for c in cells]):.3f} / '
                 f'{np.mean([ref["smooth"][c]["v2_nAP"] for c in cells]):.3f} | - | - | '
                 f'{np.mean([x["dynamic_256"] / x["eligible_256"] for k, x in dynold[dom]["seeds"].items() if k.startswith("V20F5")]):.3f} (5 fps seeds) | - |')
        for arm in ARMS:
            S = [R[(arm, s)] for s in SEEDS if (arm, s) in R]
            if not S:
                continue
            g = lambda x, rd: el(x[rd]['gate'])  # noqa: E731
            pas = {rd: [sum(r['PASS_final'] for r in g(x, rd).values()) for x in S] for rd in ('hmm', 'smooth')}
            au = {rd: [np.mean([r['v2_AUROC'] for r in g(x, rd).values()]) for x in S] for rd in ('hmm', 'smooth')}
            t1 = [np.mean([r['v2_top1_prec'] for r in g(x, 'hmm').values()]) for x in S]
            nap = {rd: [np.mean([x[rd]['cells'][c]['nAP'] for c in g(x, rd)]) for x in S] for rd in ('hmm', 'smooth')}
            f1 = {rd: [np.mean([x[rd]['cells'][c]['F1max'] for c in g(x, rd)]) for x in S] for rd in ('hmm', 'smooth')}
            px = [np.mean([v['strat_AUROC'] for v in x['hmm'].get('prox', {}).values() if v.get('strat_AUROC') is not None])
                  if x['hmm'].get('prox') else np.nan for x in S]
            dy = [x['dyn256']['share'] for x in S]
            fv = [x['fve'] for x in S]
            rows[arm] = {'pass_hmm': pas['hmm'], 'pass_smooth': pas['smooth'], 'auroc_hmm': au['hmm'], 'auroc_smooth': au['smooth'],
                         'top1': t1, 'nap': nap, 'f1': f1, 'prox': px, 'dyn': dy, 'fve': fv,
                         'best_neurons': {c: [x['hmm']['cells'][c]['best_neuron'] for x in S] for c in cells},
                         'per_cell_hmm': {c: [{'AUROC': g(x, 'hmm')[c]['v2_AUROC'], 'dAUROC': g(x, 'hmm')[c]['dAUROC'],
                                               'top1': g(x, 'hmm')[c]['v2_top1_prec'], 'PASS_final': g(x, 'hmm')[c]['PASS_final']}
                                              for x in S] for c in cells}}
            L.append(f'| {arm} ({len(S)} seeds) | {pas["hmm"]} | {pas["smooth"]} | {mr(au["hmm"])} | {mr(au["smooth"])} | {mr(t1)} | '
                     f'{mr(nap["hmm"])} / {mr(nap["smooth"])} | {mr(f1["hmm"])} / {mr(f1["smooth"])} | {mr(px)} | {mr(dy)} | {mr(fv)} |')
        L += ['', '| cell | v1 AUROC | v1 top-1% | ' + ' | '.join(f'{a} PASS_final P(on) (seeds) / AUROC median' for a in rows) + ' |',
              '|' + '---|' * (3 + len(rows))]
        for c in cells:
            L.append(f'| {c} | {ref["hmm"][c]["v1_AUROC"]:.3f} | {ref["hmm"][c]["v1_top1_prec"]:.3f} | ' + ' | '.join(
                ''.join('P' if d['PASS_final'] else 'F' for d in rows[a]['per_cell_hmm'][c]) + f' / {np.median([d["AUROC"] for d in rows[a]["per_cell_hmm"][c]]):.3f}'
                for a in rows) + ' |')
        rep[dom] = rows
        L.append('')
    # pick rule (mice + ants)
    tot = {}
    for arm in ARMS:
        if all(arm in rep[d] for d in rep):
            tot[arm] = (float(np.median([a + b for a, b in zip(rep['mice'][arm]['pass_hmm'], rep['ants'][arm]['pass_hmm'])])),
                        float(np.median([(a + b) / 2 for a, b in zip(rep['mice'][arm]['auroc_hmm'], rep['ants'][arm]['auroc_hmm'])])),
                        float(np.median(rep['mice'][arm]['dyn'] + rep['ants'][arm]['dyn'])))
    if tot:
        best = max(tot, key=lambda a: (tot[a][0], tot[a][1]))
        if best != 'S' and 'S' in tot and not (tot[best][0] >= tot['S'][0] and tot[best][2] > tot['S'][2]):
            best = 'S'
        L += ['## pick (pre-registered rule)', '', json.dumps({a: {'median_pass': v[0], 'median_AUROC': v[1], 'median_dyn': v[2]}
                                                              for a, v in tot.items()}), '', f'picked arm: {best}']
        rep['pick'] = {'totals': tot, 'picked': best}
    (OUT / 'report.md').write_text('\n'.join(L))
    (OUT / 'report.json').write_text(json.dumps(rep, indent=1, default=float))
    print('\n'.join(L))


if __name__ == '__main__':
    main()
