"""
CalMS21 ECI - scoring of the PRE-REGISTERED resolution x encoder test (rules verbatim in eci_res.sh; login safe, reads
the eval / probe / encode JSONs of results/vision/benchmark_calms21/).

Per arm (config) and SAE seed: primary = mean cross-fitted best-neuron AUROC (AUROC-picked neuron) over the scored
labels of plan/eval_plan.json (each label's AUROC averaged over split seeds 0 and 1 first). Contrast A vs B: delta =
mean(A per-seed primaries) - mean(B per-seed primaries), 95% CI = Welch t interval (Welch-Satterthwaite df).
    WIN    delta >= +0.010 AND CI excludes 0;   LOSES   delta <= -0.010 AND CI excludes 0;   otherwise NO DIFFERENCE
    (a) 64 vs 32 within each encoder: d2_896 vs d2_448, d3_1024 vs d3_512, vj_1024 vs vj_512
    (b) at 64: d3_1024 vs d2_896
    (c) V-JEPA's ranking: verdicts of vj vs d2 and vj vs d3 at 32 (vj_512 vs d2_448, d3_512) and at 64 (vj_1024 vs
        d2_896, d3_1024); the ranking HOLDS at 64 iff both verdicts at 64 equal those at 32.
Also: per-label AUROC / AP / honest top-1% (mean over seeds x splits), ceilings (probe JSONs), throughput (encode JSONs).
Outputs: OUT/res/res.md, OUT/res/res.json
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calms_common import OUT, PLAN  # noqa: E402

ARMS = ['d2_448', 'd2_896', 'd3_512', 'd3_1024', 'vj_512', 'vj_1024']
NAME = {'d2_448': 'D2-32', 'd2_896': 'D2-64', 'd3_512': 'D3-32', 'd3_1024': 'D3-64', 'vj_512': 'VJ-32',
        'vj_1024': 'VJ-64'}
RD = OUT / 'res'


def load(arm):
    R = {}
    for f in sorted((OUT / 'eval').glob(f'{arm}_s*.json')):
        m = re.fullmatch(rf'{re.escape(arm)}_s(\d+)\.json', f.name)
        if m:
            R[int(m.group(1))] = json.loads(f.read_text())['m4096']
    return R


def per_seed(R, labs, q='cf_auroc'):
    return {s: float(np.mean([np.mean([r[sp][b][q] for sp in r]) for b in labs])) for s, r in sorted(R.items())}


def mean_q(R, b, q):
    return float(np.mean([r[sp][b][q] for r in R.values() for sp in r]))


def welch(a, b):
    a, b = np.asarray(a), np.asarray(b)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    d = a.mean() - b.mean()
    h = stats.t.ppf(0.975, df) * se
    return float(d), float(d - h), float(d + h), float(se), float(df)


def verdict(d, lo, hi):
    if d >= 0.010 and lo > 0:
        return 'WIN'
    if d <= -0.010 and hi < 0:
        return 'LOSES'
    return 'NO DIFFERENCE'


def main():
    labs = json.loads((PLAN / 'eval_plan.json').read_text())['scored_labels']
    D = {a: load(a) for a in ARMS}
    D = {a: v for a, v in D.items() if v}
    S = {a: per_seed(v, labs) for a, v in D.items()}
    out = {'labels': labs, 'arms': {}, 'contrasts': {}, 'missing': [a for a in ARMS if a not in D]}
    L = ['# CalMS21: resolution x encoder (pre-registered, eci_res.sh)', '',
         f'primary = mean cross-fitted best-neuron AUROC over {", ".join(labs)} (AUROC-picked neuron; split seeds 0, 1 '
         'averaged); per-seed values; Welch 95% CI', '',
         '| arm | seeds | per-seed primary AUROC | mean ± SE |', '|---|---|---|---|']
    for a, R in D.items():
        v = np.array(list(S[a].values()))
        se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float('nan')
        out['arms'][a] = {'seeds': list(S[a]), 'per_seed_primary_auroc': list(S[a].values()), 'mean': float(v.mean()),
                          'se': float(se), 'auroc': {b: mean_q(R, b, 'cf_auroc') for b in labs},
                          'ap': {b: mean_q(R, b, 'cf_ap') for b in labs},
                          'top1': {b: mean_q(R, b, 'cf_top1') for b in labs},
                          'rate_auc_pick': {b: mean_q(R, b, 'cf_rate_auc_pick') for b in labs},
                          'top1_rate': {b: mean_q(R, b, 'cf_top1_rate') for b in labs}}
        L.append(f"| {NAME[a]} ({a}) | {len(v)} | {', '.join(f'{x:.3f}' for x in v)} | {v.mean():.4f} ± {se:.4f} |")
    C = {'(a) D2: 64 vs 32': ('d2_896', 'd2_448'), '(a) D3: 64 vs 32': ('d3_1024', 'd3_512'),
         '(a) VJ: 64 vs 32': ('vj_1024', 'vj_512'), '(b) at 64: D3 vs D2': ('d3_1024', 'd2_896'),
         '(c) at 32: VJ vs D2': ('vj_512', 'd2_448'), '(c) at 32: VJ vs D3': ('vj_512', 'd3_512'),
         '(c) at 64: VJ vs D2': ('vj_1024', 'd2_896'), '(c) at 64: VJ vs D3': ('vj_1024', 'd3_1024'),
         '(descriptive) at 32: D3 vs D2': ('d3_512', 'd2_448')}
    L += ['', '| contrast | delta primary AUROC [95% CI] | df | verdict |', '|---|---|---|---|']
    for k, (a, b) in C.items():
        if a in S and b in S and len(S[a]) > 1 and len(S[b]) > 1:
            d, lo, hi, se, df = welch(list(S[a].values()), list(S[b].values()))
            v = verdict(d, lo, hi)
            out['contrasts'][k] = {'a': a, 'b': b, 'delta': d, 'ci': [lo, hi], 'se': se, 'df': df, 'verdict': v}
            L.append(f'| {k} ({NAME[a]} - {NAME[b]}) | {d:+.4f} [{lo:+.4f}, {hi:+.4f}] | {df:.1f} | {v} |')
        else:
            L.append(f'| {k} | missing arm | | |')
    c = out['contrasts']
    if all(k in c for k in ('(c) at 32: VJ vs D2', '(c) at 32: VJ vs D3', '(c) at 64: VJ vs D2', '(c) at 64: VJ vs D3')):
        holds = (c['(c) at 32: VJ vs D2']['verdict'] == c['(c) at 64: VJ vs D2']['verdict']
                 and c['(c) at 32: VJ vs D3']['verdict'] == c['(c) at 64: VJ vs D3']['verdict'])
        out['vjepa_ranking_holds_at_64'] = holds
        L += ['', f"(c) V-JEPA ranking holds at 64: **{'YES' if holds else 'NO'}**"]
    L += ['', '## per label (mean over seeds x splits): cf AUROC / cf AP / honest top-1% (rate of the AUROC pick)', '',
          '| arm | ' + ' | '.join(labs) + ' |', '|---|' + '---|' * len(labs)]
    for a, r in out['arms'].items():
        L.append(f'| {NAME[a]} | ' + ' | '.join(f"{r['auroc'][b]:.3f} / {r['ap'][b]:.4f} / {r['top1'][b]:.3f} "
                                                f"({r['rate_auc_pick'][b]:.2f})" for b in labs) + ' |')
    # ceilings
    P = {}
    for a in ARMS:
        f = OUT / 'probe' / f'probe_{a}.json'
        if f.exists():
            P[a] = {(s['probe'], s['behaviour']): s for s in json.loads(f.read_text())['summary']}
    if P:
        L += ['', '## supervised ceiling (DIAGNOSTIC; 5 video-grouped folds): AUROC / AP', '',
              '| arm | probe | ' + ' | '.join(labs) + ' | mean AUROC |', '|---|---|' + '---|' * (len(labs) + 1)]
        for a, p in P.items():
            for pr in ('mean_lin', 'mil_ctx'):
                if all((pr, b) in p for b in labs):
                    L.append(f'| {NAME[a]} | {pr} | ' + ' | '.join(
                        f"{p[(pr, b)]['auroc_mean']:.3f} / {p[(pr, b)]['ap_mean']:.3f}" for b in labs)
                        + f" | {np.mean([p[(pr, b)]['auroc_mean'] for b in labs]):.3f} |")
            out.setdefault('ceiling', {})[a] = {f'{pr}/{b}': {'auroc': p[(pr, b)]['auroc_mean'],
                                                              'ap': p[(pr, b)]['ap_mean']}
                                                for (pr, b) in p if pr in ('mean_lin', 'mil_ctx')}
    RD.mkdir(parents=True, exist_ok=True)
    (RD / 'res.md').write_text('\n'.join(L) + '\n')
    (RD / 'res.json').write_text(json.dumps(out, indent=1))
    print('\n'.join(L))


if __name__ == '__main__':
    main()
