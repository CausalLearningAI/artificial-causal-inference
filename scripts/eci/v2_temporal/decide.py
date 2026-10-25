"""
Decision tables for the temporal-handling study (rule: temporal.py docstring, pre-registered 2026-10-25).
Reads OUT/<bench>/scores/<arm>_s<seed>.json -> OUT/decision.json, OUT/table.md. CPU only.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
OUT = Path(os.environ.get('TMP_OUT') or REPO / 'results/vision/eci_v2_temporal')
VARIANTS = {'T0': ('T0', 'sm'), 'T0u': ('T0', 'raw'), 'T1': ('T1', 'raw'), 'T1s': ('T1', 'sm'),
            'T2': ('T2', 'raw'), 'T2s': ('T2', 'sm'), 'T3': ('T3', 'raw'), 'T3s': ('T3', 'sm')}
ORDER = list(VARIANTS)
BENCH = ('mabe22', 'calms21', 'crim13')
METRICS = (('R', 'R~'), ('AP', 'AP'), ('vid_r', 'video r'), ('A', 'AUROC'))


def welch(a, b):
    from scipy import stats
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a.mean() - b.mean()
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    if se == 0:
        return float(d), float(d), float(d)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    h = stats.t.ppf(0.975, df) * se
    return float(d), float(d - h), float(d + h)


def load(bench):
    d = OUT / bench / 'scores'
    out = {}
    for f in sorted(d.glob('*_s*.json')) if d.exists() else []:
        r = json.loads(f.read_text())
        out[(r['arm'], r['seed'])] = r
    return out


def bench_vals(S):
    """-> {variant: {'seeds': [...], metric: [per-seed primary], 'labels': {label: {metric: [per seed]}}}}"""
    V = {}
    for vn, (arm, ro) in VARIANTS.items():
        rs = [r for (a, s), r in sorted(S.items()) if a == arm and ro in r['summary']]
        if not rs:
            continue
        v = {'seeds': [r['seed'] for r in rs]}
        for m, _ in METRICS:
            v[m] = [r['summary'][ro]['_primary'][m] for r in rs]
        labs = [k for k in rs[0]['summary'][ro] if not k.startswith('_')]
        v['labels'] = {lb: {m: [r['summary'][ro][lb][m] for r in rs] for m, _ in METRICS} for lb in labs}
        v['fve'] = [r['train']['eval_fve'] for r in rs]
        V[vn] = v
    return V


def lab_vals(S, dom):
    """lab primary per seed: mice = cell 'mice' primary (nn, nt) on the atlas split; ants = mean of v2 / v3 groom_any
    (split r4). Also the rnd-split means and per-cell labels."""
    V = {}
    for vn, (arm, ro) in VARIANTS.items():
        rs = [r for (a, s), r in sorted(S.items()) if a == arm and ro in next(iter(r['cells'].values()))['summary']]
        if not rs:
            continue
        v = {'seeds': [r['seed'] for r in rs], 'cells': {}}
        for m, _ in METRICS:
            for key in ('summary', 'summary_rnd'):
                v[f'{m}{"" if key == "summary" else "_rnd"}'] = [
                    float(np.mean([c[key][ro]['_primary'][m] for c in r['cells'].values()])) for r in rs]
        for cell in rs[0]['cells']:
            labs = [k for k in rs[0]['cells'][cell]['summary'][ro] if not k.startswith('_')]
            v['cells'][cell] = {lb: {m: [r['cells'][cell]['summary'][ro][lb][m] for r in rs] for m, _ in METRICS}
                                for lb in labs}
        V[vn] = v
    return V


def fmt(x):
    x = np.asarray(x, float)
    return f'{x.mean():.4f} ± {x.std(ddof=1):.4f}' if len(x) > 1 else f'{x.mean():.4f}'


def fci(d):
    return f'{d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]'


def verdict(d):
    if d[0] >= 0.01 and d[1] > 0:
        return 'WIN'
    if d[2] < 0:
        return 'LOSS'
    return 'tie'


def main():
    res, L = {'rule': 'scripts/eci/v2_temporal/temporal.py docstring (pre-registered 2026-10-25)', 'bench': {}}, []
    L += ['# Temporal handling - results', '',
          'Metric names: R~ = cross-fitted normalised R-precision (IPW on benchmarks); AP = average precision of the '
          'AP-picked neuron; video r = cross-fitted Pearson r between per-video mean neuron value and annotated '
          'time-in-behaviour; AUROC of the AUROC-picked neuron. Cells: mean ± sd over SAE seeds; differences vs T0: '
          'Welch 95% CI. Primary = mean over the benchmark\'s behaviours.', '']
    contrasts = {}
    for b in BENCH:
        V = bench_vals(load(b))
        if not V:
            continue
        res['bench'][b] = {'values': V}
        L += [f'## {b}', '', '| variant | seeds | R~ | AP | video r | AUROC | held-out FVE | R~ - T0 | AP - T0 | '
              'video r - T0 | R~ verdict |', '|---|---|---|---|---|---|---|---|---|---|---|']
        for vn in ORDER:
            if vn not in V:
                continue
            v = V[vn]
            c = {m: welch(v[m], V['T0'][m]) for m, _ in METRICS} if 'T0' in V and vn != 'T0' else None
            if c:
                contrasts.setdefault(vn, {})[b] = c
            L.append(f'| {vn} | {len(v["seeds"])} | {fmt(v["R"])} | {fmt(v["AP"])} | {fmt(v["vid_r"])} | '
                     f'{fmt(v["A"])} | {fmt(v["fve"])} | ' +
                     (f'{fci(c["R"])} | {fci(c["AP"])} | {fci(c["vid_r"])} | {verdict(c["R"])} |' if c else
                      '- | - | - | reference |'))
        L += ['', f'Per behaviour ({b}): R~ / AP / video r, mean over seeds', '',
              '| variant | ' + ' | '.join(V['T0']['labels']) + ' |', '|---|' + '---|' * len(V['T0']['labels'])]
        for vn in ORDER:
            if vn in V:
                L.append(f'| {vn} | ' + ' | '.join(
                    f'{np.mean(x["R"]):.3f} / {np.mean(x["AP"]):.3f} / {np.mean(x["vid_r"]):.2f}'
                    for x in V[vn]['labels'].values()) + ' |')
        L.append('')
    # rule
    dec = {}
    for vn, cb in contrasts.items():
        wins = [b for b, c in cb.items() if verdict(c['R']) == 'WIN']
        loss = [b for b, c in cb.items() if verdict(c['R']) == 'LOSS']
        vloss = [b for b, c in cb.items() if c['vid_r'][2] < 0]
        complete = len(cb) == len(BENCH)
        dec[vn] = {'wins': wins, 'losses': loss, 'video_r_losses': vloss, 'complete': complete,
                   'adopted': bool(complete and len(wins) >= 2 and not loss and not vloss),
                   'mean_R_gain': float(np.mean([c['R'][0] for c in cb.values()]))}
    ad = [v for v in ORDER if dec.get(v, {}).get('adopted')]
    if ad:
        best = max(ad, key=lambda v: dec[v]['mean_R_gain'])
        tied = [v for v in ad if dec[best]['mean_R_gain'] - dec[v]['mean_R_gain'] <= 0.01]
        choice = min(tied, key=ORDER.index)
    else:
        choice = 'T0'
    res['decision'] = {'per_variant': dec, 'adopted': ad, 'choice': choice}
    q1 = [v for v in ('T1', 'T1s', 'T2', 'T2s') if v in ad]
    q2 = [v for v in ('T3', 'T3s') if v in ad]
    ran = {v: dec.get(v, {}).get('complete', False) for v in ORDER[1:]}
    q1_ran = [v for v in ('T1', 'T1s', 'T2', 'T2s') if ran[v]]
    q2_ran = [v for v in ('T3', 'T3s') if ran[v]]
    part = lambda vs: '; '.join(f"{v}: WIN on {', '.join(dec[v]['wins']) or 'none'}, LOSS on {', '.join(dec[v]['losses']) or 'none'}" for v in vs)  # noqa: E731
    res['verdicts'] = {
        'Q1': ('NOT RUN' if not q1_ran else
               ('local temporal information beats post-hoc smoothing: ' + ', '.join(q1)) if q1 else
               'post-hoc smoothing (T0) is not beaten by any tested local temporal variant (none adopted by the rule; '
               + part(q1_ran) + ')') + ('' if len(q1_ran) == 4 else f" [not run: {', '.join(v for v in ('T1', 'T1s', 'T2', 'T2s') if not ran[v])}]"),
        'Q2': ('NOT RUN (no T3 scores)' if not q2_ran else
               ('V-JEPA adopted: ' + ', '.join(q2)) if q2 else
               'V-JEPA 2-L is not better than T0 by the rule (not adopted; ' + part(q2_ran) + ')')
        + ('' if len(q2_ran) == 2 or not q2_ran else f" [not run: {', '.join(v for v in ('T3', 'T3s') if not ran[v])}]")}
    L += ['## Decision (pre-registered rule)', '', '| variant | WIN on | LOSS on | video-r LOSS on | adopted |',
          '|---|---|---|---|---|']
    for vn in ORDER:
        if vn in dec:
            d = dec[vn]
            L.append(f'| {vn} | {", ".join(d["wins"]) or "-"} | {", ".join(d["losses"]) or "-"} | '
                     f'{", ".join(d["video_r_losses"]) or "-"} | {d["adopted"]}{"" if d["complete"] else " (incomplete)"} |')
    L += ['', f'Choice: **{choice}**', '', f'Q1: {res["verdicts"]["Q1"]}', '', f'Q2: {res["verdicts"]["Q2"]}', '']
    # lab
    res['lab'] = {}
    for dom in ('mice', 'ants'):
        S = load(dom)
        if not S:
            continue
        V = lab_vals(S, dom)
        res['lab'][dom] = {'values': V}
        prim = 'mean(nose_nose, nose_tail) R~ (atlas split)' if dom == 'mice' else 'mean(v2, v3 groom_any) R~ (r4 split)'
        L += [f'## Lab {dom} - primary = {prim}', '',
              '| variant | seeds | R~ | R~ (rnd splits) | AP | video r | R~ - T0 | AP - T0 | video r - T0 | verdict |',
              '|---|---|---|---|---|---|---|---|---|---|']
        for vn in ORDER:
            if vn not in V:
                continue
            v = V[vn]
            c = {m: welch(v[m], V['T0'][m]) for m in ('R', 'AP', 'vid_r')} if vn != 'T0' and 'T0' in V else None
            res['lab'][dom].setdefault('contrasts', {})[vn] = c
            L.append(f'| {vn} | {len(v["seeds"])} | {fmt(v["R"])} | {fmt(v["R_rnd"])} | {fmt(v["AP"])} | '
                     f'{fmt(v["vid_r"])} | ' + (f'{fci(c["R"])} | {fci(c["AP"])} | {fci(c["vid_r"])} | '
                                               f'{verdict(c["R"])} |' if c else '- | - | - | reference |'))
        L += ['', '| variant | cell | label | R~ | AP | video r | AUROC |', '|---|---|---|---|---|---|---|']
        for vn in ORDER:
            if vn in V:
                for cell, labs in V[vn]['cells'].items():
                    for lb, x in labs.items():
                        L.append(f'| {vn} | {cell} | {lb} | {fmt(x["R"])} | {fmt(x["AP"])} | {fmt(x["vid_r"])} | '
                                 f'{fmt(x["A"])} |')
        L.append('')
    (OUT / 'decision.json').write_text(json.dumps(res, indent=1, default=float))
    (OUT / 'table.md').write_text('\n'.join(L))
    print('\n'.join(L))


if __name__ == '__main__':
    main()
