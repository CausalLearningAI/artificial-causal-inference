"""
MABe22 mouse ECI - scoring of the PRE-REGISTERED CONFIRMATION RUN (rule verbatim in eci_confirm.sh; login safe, reads
the eval JSONs of results/vision/benchmark_mabe22/eci/confirm/eval/).

Per arm (A0, D2, D4 = X_D4, A7, A7D2 = X_A7D2) and SAE seed (100-105): primary = mean cross-fitted AUROC over
oral_oral, oral_genital, approaches, huddles (each label's AUROC averaged over the 2 split seeds first). Delta vs A0 =
mean(arm seeds) - mean(A0 seeds); 95% CI = Welch t interval on the 6 vs 6 per-seed primary means (Welch-Satterthwaite
df). Win iff delta >= +0.010 AND CI excludes 0 AND cf AP (mean over seeds x splits) on oral_oral and oral_genital each
>= 0.9 x A0's. Decision: if both A7 and D2 win, A7D2 is preferred only if it also beats each by >= 0.01 with a CI
excluding 0 (same Welch interval, A7D2 vs that arm); otherwise the cheaper winner (D2 cheaper than A7).
Outputs: OUT/confirm/confirm.md, OUT/confirm/confirm.json

RESULTS (2026-10-15 07:01, seeds 100-105, new token store, rule fixed in eci_confirm.sh at commit 5b7df89 before any
number): per-seed primary means A0 0.750 0.717 0.760 0.714 0.723 0.710 (0.7288 +- 0.0085 SE); delta vs A0 [Welch 95% CI]
D2 +0.030 [+0.004, +0.056], D4 +0.034 [+0.009, +0.059], A7 +0.033 [+0.011, +0.055], A7D2 +0.060 [+0.033, +0.086];
AP ratios oo / og D2 1.89 / 1.57, D4 1.40 / 0.95, A7 1.35 / 1.48, A7D2 1.55 / 1.58 -> all four WIN. A7D2 vs A7 +0.027
[+0.005, +0.048], vs D2 +0.030 [+0.004, +0.055] -> pre-registered decision: A7D2 (D4 orbit tokens + 2 s delta).
Honest top-1% oral_oral / oral_genital stays 0.023-0.035 / 0.036-0.048 in every arm (sniffing bar 0.25 not reached).
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eci_common import OUT, SHORT  # noqa: E402

P4 = ['oral_oral_contact', 'oral_genital_contact', 'approaches', 'huddles']
ORAL = ['oral_oral_contact', 'oral_genital_contact', 'oral_ear_contact']
ARMS = {'A0': 'A0', 'D2': 'D2', 'D4': 'X_D4', 'A7': 'A7', 'A7D2': 'X_A7D2'}
CD = OUT / 'confirm'


def load(arm):
    R = {}
    for f in sorted((CD / 'eval').glob(f'{arm}_s*.json')):
        m = re.fullmatch(rf'{re.escape(arm)}_s(\d+)\.json', f.name)
        if m and int(m.group(1)) >= 100:
            R[int(m.group(1))] = json.loads(f.read_text())['m4096']
    return R


def per_seed(R, labels=P4, q='cf_auroc'):
    return {s: float(np.mean([np.mean([r[sp][b][q] for sp in r]) for b in labels])) for s, r in sorted(R.items())}


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


def main():
    D = {k: load(v) for k, v in ARMS.items()}
    D = {k: v for k, v in D.items() if v}
    S = {k: per_seed(v) for k, v in D.items()}
    out = {'arms': {}, 'missing': [k for k in ARMS if k not in D]}
    L = ['# MABe22 confirmation run (pre-registered, eci_confirm.sh)', '',
         'primary = mean cf AUROC over oral_oral, oral_genital, approaches, huddles; SAE seeds 100-105; Welch 95% CI', '',
         '| arm | seeds | per-seed primary means | mean ± SE | delta vs A0 [95% CI] | AP ratio oo / og (oe) | '
         'honest top-1% oo / og / oe | verdict |', '|---|---|---|---|---|---|---|---|']
    win = {}
    a0 = D.get('A0')
    for k, R in D.items():
        v = np.array(list(S[k].values()))
        se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float('nan')
        row = {'seeds': list(S[k]), 'per_seed': list(S[k].values()), 'mean': float(v.mean()), 'se': float(se),
               'auroc': {b: mean_q(R, b, 'cf_auroc') for b in P4 + ['oral_ear_contact']},
               'ap': {b: mean_q(R, b, 'cf_ap') for b in P4 + ['oral_ear_contact']},
               'top1': {b: mean_q(R, b, 'cf_top1') for b in ORAL},
               'max_top1': {b: max(d['top1']['test_top1'] for r in R.values() for sp in r for d in r[sp][b]['dirs'])
                            for b in ORAL}}
        verdict = 'control'
        if k != 'A0' and a0:
            d, lo, hi, sed, df = welch(list(S[k].values()), list(S['A0'].values()))
            apr = {b: row['ap'][b] / out['arms']['A0']['ap'][b] for b in ORAL}
            w = d >= 0.010 and (lo > 0 or hi < 0) and apr['oral_oral_contact'] >= 0.9 and apr['oral_genital_contact'] >= 0.9
            win[k] = bool(w)
            row.update(delta=d, ci=[lo, hi], se_delta=sed, df=df, ap_ratio=apr, win=bool(w))
            verdict = 'WIN' if w else 'no win'
        row['verdict'] = verdict
        out['arms'][k] = row
        ds = (f"{row['delta']:+.4f} [{row['ci'][0]:+.4f}, {row['ci'][1]:+.4f}]" if 'delta' in row else '-')
        ar = (f"{row['ap_ratio']['oral_oral_contact']:.2f} / {row['ap_ratio']['oral_genital_contact']:.2f} "
              f"({row['ap_ratio']['oral_ear_contact']:.2f})" if 'ap_ratio' in row else '-')
        t1 = ' / '.join(f"{row['top1'][b]:.3f}" for b in ORAL)
        L.append(f"| {k} | {len(v)} | {', '.join(f'{x:.3f}' for x in v)} | {v.mean():.4f} ± {se:.4f} | {ds} | {ar} | "
                 f"{t1} | {verdict} |")
    # decision
    dec = None
    if 'A0' in D:
        winners = [k for k in ('D2', 'D4', 'A7', 'A7D2') if win.get(k)]
        if win.get('A7') and win.get('D2') and 'A7D2' in D:
            ok = []
            for k in ('A7', 'D2'):
                d, lo, hi, _, _ = welch(list(S['A7D2'].values()), list(S[k].values()))
                ok.append(d >= 0.01 and lo > 0)
                out['arms']['A7D2'][f'vs_{k}'] = {'delta': d, 'ci': [lo, hi]}
            dec = 'A7D2' if all(ok) else 'D2'
        elif winners:
            cost = {'D2': 1, 'D4': 1, 'A7': 8, 'A7D2': 8}
            dec = sorted(winners, key=lambda k: (cost[k], -out['arms'][k]['delta']))[0]
        else:
            dec = 'A0 (no arm wins)'
        out['winners'] = winners
        out['decision'] = dec
    L += ['', f"Winners: {out.get('winners')}; pre-registered decision: **{dec}**",
          f"Missing arms: {out['missing'] or 'none'}", '',
          '## per-label (mean over seeds x splits): AUROC / AP', '',
          '| arm | ' + ' | '.join(SHORT[b] for b in P4 + ['oral_ear_contact']) + ' |', '|---|' + '---|' * 5]
    for k, r in out['arms'].items():
        L.append(f'| {k} | ' + ' | '.join(f"{r['auroc'][b]:.3f} / {r['ap'][b]:.4f}" for b in P4 + ['oral_ear_contact'])
                 + ' |')
    if 'A7D2' in out['arms'] and 'vs_A7' in out['arms']['A7D2']:
        L += ['', 'A7D2 vs A7 / D2: ' + '; '.join(
            f"{k}: {out['arms']['A7D2'][f'vs_{k}']['delta']:+.4f} [{out['arms']['A7D2'][f'vs_{k}']['ci'][0]:+.4f}, "
            f"{out['arms']['A7D2'][f'vs_{k}']['ci'][1]:+.4f}]" for k in ('A7', 'D2'))]
    CD.mkdir(parents=True, exist_ok=True)
    (CD / 'confirm.md').write_text('\n'.join(L) + '\n')
    (CD / 'confirm.json').write_text(json.dumps(out, indent=1))
    print('\n'.join(L))


if __name__ == '__main__':
    main()
