"""
ECI v2.9 ADDITION (POST HOC; not part of the pre-registered decisions): per PREDk dictionary (15 v2.9 + 3 v2.7), the
distribution over the 50 annotator-stratified random pool splits of the v2.8 split check (rule sgap with the firing cap,
re-picked on every select side): evaluate-side AUROC, AP (+ chance) and precision@top3/video of the picks (mean and
2.5-97.5 percentiles); fraction of splits picking the same neurons as the official SG split; fraction of splits in which the
dictionary is chosen by select-side sgap AP sum among all 15 PREDk dictionaries (12 v2.9 + 3 v2.7); fraction of splits meeting the
pre-registered POINT thresholds (nose-nose AP >= 0.10, precision@top3/video >= 0.35, AUROC >= 0.75; nose-tail distinct,
AP >= 0.10, precision@top3/video >= 0.30; firing cap built into sgap; the top-3 overlap criterion is not computable from the
split pickles and is omitted). Plus the leave-one-annotator-out table for one dictionary (--loao KEY).
-> results/vision/eci_v2_9/split/{tables.md, split.json}
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / 'results/vision/eci_v2_9/split'
W9, W8 = OUT / 'work', REPO / 'results/vision/eci_v2_8/split/work'
BEH = ('nose_nose', 'nt')
BN = {'nose_nose': 'nose-nose', 'nt': 'nose-tail'}
KEYS = [f'PREDk16_s{s}' for s in range(2901, 2907)] + [f'PREDk8_s{s}' for s in (2901, 2902, 2903)] + \
    [f'PREDk32_s{s}' for s in (2901, 2902, 2903)] + [f'PREDk_s{s}' for s in (2701, 2702, 2703)]
LOAO = ['loao_SG', 'loao_SD', 'loao_MF', 'loao_MB']


def load():
    D = {}
    for k in KEYS:
        p = W9 / f'{k}.pkl' if '_s29' in k else W8 / f'{k}.pkl'
        if p.exists():
            D[k] = pickle.load(open(p, 'rb'))
    return D


def metrics(r, QI, j, b, q):
    """= v28_split_combine.metrics."""
    c = BEH.index(b)
    qi = QI[q]
    Msub, PN = r['Msub'], r['PN']
    m = Msub[qi].astype(float)
    vm = Msub[qi][r['vcell']].astype(bool)
    pos, nval = r['topk'][('pos', b)], r['topk'][('nval', b)]
    e = vm & (pos > 0)
    out = {'auc': float(r['auc'][j][qi, c]), 'ap': float(r['ap'][j][qi, c])}
    for k in (3, 10):
        out[f'top{k}'] = float(r['topk'][(j, b, k)][e].sum() / (k * e.sum())) if e.any() else float('nan')
    out['chance_ap'] = float(PN[c, 0] @ m / ((PN[c, 0] + PN[c, 1]) @ m))
    out['chance_topk'] = float(np.mean(pos[e] / nval[e])) if e.any() else float('nan')
    return out


def sel_sum(r, QI, s):
    p = r['picks']['sgap'][s]
    return 0.0 if p is None else float(r['ap'][p[0]][QI[f'{s}|sel'], 0] + r['ap'][p[1]][QI[f'{s}|sel'], 1])


def passes(mn, mt):
    return (mn['ap'] >= 0.10 and mn['top3'] >= 0.35 and mn['auc'] >= 0.75 and mt['ap'] >= 0.10 and mt['top3'] >= 0.30)


def summ(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return None
    return {'mean': float(v.mean()), 'sd': float(v.std(ddof=1)) if len(v) > 1 else float('nan'), 'p2.5': float(np.percentile(v, 2.5)), 'p97.5': float(np.percentile(v, 97.5)), 'n': int(len(v))}


def cf(s):
    return 'n/a' if s is None else f"{s['mean']:.3f} +- {s['sd']:.3f} [{s['p2.5']:.3f}, {s['p97.5']:.3f}]"


def main(a):
    D = load()
    r0 = next(iter(D.values()))
    names = r0['names']
    for k, r in D.items():
        assert r['names'] == names and (r['Msub'] == r0['Msub']).all(), k
    QI = {n: i for i, n in enumerate(names)}
    RAND = sorted({n.split('|')[0] for n in names if n.startswith('rand')})
    chosen = [max(D, key=lambda k: sel_sum(D[k], QI, s)) for s in RAND]
    res = {'post_hoc': True, 'dictionaries': list(D), 'n_random_splits': len(RAND), 'per_dict': {},
           'chosen_official': max(D, key=lambda k: sel_sum(D[k], QI, 'official'))}
    md = ['# ECI v2.9 ADDITION (POST HOC): random pool splits for the PREDk dictionaries (v28_split_neurons via v29_split.py, v29_split_combine.py)', '',
          f'{len(RAND)} annotator-stratified random pool splits (select 11 pools / evaluate 12; the v2.8 split check, seed 20261125); rule sgap '
          '(firing cap) re-picked on each select side. Cells: mean +- SD [2.5, 97.5 percentile] over splits with a pick. "same picks" = fraction of '
          'splits whose sgap pair equals the official (SG-selected) pair; "chosen" = fraction of splits in which the dictionary has the '
          f'maximum select-side sgap AP sum among these {len(D)} PREDk dictionaries; "meets point thresholds" = nose-nose AP >= 0.10, '
          'precision@top3/video >= 0.35, AUROC >= 0.75 and nose-tail AP >= 0.10, precision@top3/video >= 0.30 (point values on the '
          'evaluate side; the top-3-overlap criterion is not computable here). Not part of the pre-registered decisions.', '',
          '| dictionary | no pair | same picks | chosen | meets point thresholds | nose-nose AUROC | nose-nose AP (chance) | nose-nose precision@top3/video | '
          'nose-tail AUROC | nose-tail AP (chance) | nose-tail precision@top3/video |', '|---|---|---|---|---|---|---|---|---|---|---|']
    for k, r in D.items():
        po = r['picks']['sgap']['official']
        vals = {f'{b}|{m}': [] for b in BEH for m in ('auc', 'ap', 'top3', 'chance_ap')}
        same, ok, nop = [], [], 0
        for s in RAND:
            p = r['picks']['sgap'][s]
            if p is None:
                nop += 1
                same.append(False)
                ok.append(False)
                continue
            same.append(po is not None and tuple(p) == tuple(po))
            mt = {b: metrics(r, QI, p[i], b, f'{s}|ev') for i, b in enumerate(BEH)}
            ok.append(bool(p[0] != p[1] and passes(mt['nose_nose'], mt['nt'])))
            for b in BEH:
                for m in ('auc', 'ap', 'top3', 'chance_ap'):
                    vals[f'{b}|{m}'].append(mt[b][m])
        x = {'official_pick': list(po) if po else None, 'no_pair': nop, 'frac_same': float(np.mean(same)), 'frac_chosen': float(np.mean([c == k for c in chosen])),
             'frac_meet_point': float(np.mean(ok)), 'summary': {kk: summ(v) for kk, v in vals.items()}}
        res['per_dict'][k] = x
        s_ = x['summary']
        md.append(f"| {k} | {nop}/{len(RAND)} | {x['frac_same']:.2f} | {x['frac_chosen']:.2f} | {x['frac_meet_point']:.2f} | " + ' | '.join(
            f"{cf(s_[f'{b}|auc'])} | {cf(s_[f'{b}|ap'])} ({s_[f'{b}|chance_ap']['mean']:.4f}) | {cf(s_[f'{b}|top3'])}" if s_[f'{b}|auc'] else 'n/a | n/a | n/a' for b in BEH) + ' |')
    md += ['', f"Official (SG) split chooses {res['chosen_official']} among these {len(D)} dictionaries.", '']
    if a.loao and a.loao in D:
        r = D[a.loao]
        md += [f'## Leave one annotator out: {a.loao} (select = pools without that annotator, evaluate = its pools; rule sgap re-picked)', '',
               '| left out | chosen among the PREDk dictionaries | nose-nose pick / AUROC / AP (chance) / precision@top3/video | nose-tail pick / AUROC / AP (chance) / precision@top3/video |', '|---|---|---|---|']
        lo = {}
        for s in LOAO:
            ch = max(D, key=lambda kk: sel_sum(D[kk], QI, s))
            p = r['picks']['sgap'][s]
            cells = []
            for i, b in enumerate(BEH):
                if p is None:
                    cells.append('no pair')
                    continue
                mt = metrics(r, QI, p[i], b, f'{s}|ev')
                lo[f'{s}|{b}'] = {'pick': int(p[i]), **mt}
                cells.append(f"#{p[i]} / {mt['auc']:.3f} / {mt['ap']:.3f} ({mt['chance_ap']:.4f}) / {mt['top3']:.3f}")
            lo[f'{s}|chosen'] = ch
            md.append(f"| {s.split('_')[1]} | {ch} | " + ' | '.join(cells) + ' |')
        res['loao'] = {'key': a.loao, **lo}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'split.json').write_text(json.dumps(res, indent=1, default=float))
    (OUT / 'tables.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--loao', default='')
    main(p.parse_args())
