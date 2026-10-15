"""
MABe22 mouse ECI - EXPERIMENT P2: scoring under the PRE-REGISTERED rules of eci_p2.sh (login safe: reads JSONs).
Arms P2D (x = [z_t, z_t - z_{t-60}]) and P2N (x = z_t), DINOv3-L/16 512 px block 20, SAE seeds 320-325, full width
m4096. Per arm: per-seed primary AUROC (mean cf AUROC over oral_oral, oral_genital, approaches, huddles; split seeds
averaged first), cf AUROC / AP / honest top-1% for oral_oral_contact and oral_genital_contact (mean over seeds x
splits), weighted base rates. Contrast P2D - P2N, Welch 95% CI on 6 vs 6 per-seed values: WIN = delta >= +0.010 and
CI excludes 0; LOSES = delta <= -0.010 and CI excludes 0; else EQUIVALENT (primary AUROC = decision; oo / og AP =
descriptive). Answer rule (DINOv2-B AP loss holds): no-Delta / Delta AP ratio <= 0.70 and the AP CI excludes 0
(Delta higher) on both sniffing labels = HOLDS; on one = PARTIAL; else DOES NOT HOLD.
Outputs: OUT/p2/p2.md, OUT/p2/p2.json
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eci_common import BEHAVIOURS, OUT, PLAN, SHORT  # noqa: E402
from eci_confirm import welch  # noqa: E402

P2 = OUT / 'p2'
EV = P2 / 'eval'
SEEDS = list(range(320, 326))
P4 = ['oral_oral_contact', 'oral_genital_contact', 'approaches', 'huddles']
SN = ['oral_oral_contact', 'oral_genital_contact']
RUNS = {'P2D': 'P2D_p2d3l_512_L20', 'P2N': 'P2N_p2d3l_512_L20'}
LABEL = {'P2D': 'Delta 2 s [z_t, z_t - z_t-60]', 'P2N': 'no Delta (z_t only)'}


def load(run):
    R = {}
    for f in sorted(EV.glob(f'{run}_s*.json')):
        m = re.fullmatch(rf'{re.escape(run)}_s(\d+)\.json', f.name)
        if m and int(m.group(1)) in SEEDS:
            j = json.loads(f.read_text())
            R[int(m.group(1))] = j[j['full']]
    return R


def per_seed(R, b, k):
    return [float(np.mean([r[sp][b][k] for sp in r])) for _, r in sorted(R.items())]


def summ(R):
    prim = [float(np.mean([np.mean([r[sp][b]['cf_auroc'] for sp in r]) for b in P4])) for _, r in sorted(R.items())]
    v = np.array(prim)
    out = {'seeds': sorted(R), 'primary': prim, 'mean': float(v.mean()), 'se': float(v.std(ddof=1) / np.sqrt(len(v)))}
    for k in ('cf_auroc', 'cf_ap', 'cf_top1'):
        out[k] = {b: float(np.mean(per_seed(R, b, k))) for b in BEHAVIOURS}
        out[k + '_seed'] = {b: per_seed(R, b, k) for b in SN}
    return out


def contrast(a, c):
    d, lo, hi, se, df = welch(a, c)
    v = 'WIN' if d >= 0.010 and lo > 0 else ('LOSES' if d <= -0.010 and hi < 0 else 'EQUIVALENT')
    return {'delta': d, 'ci': [lo, hi], 'se': se, 'df': df, 'verdict': v}


def main():
    E = pd.read_parquet(PLAN / 'eval_frames.parquet')
    base = {b: float((E[b] * E.w).sum() / E.w.sum()) for b in BEHAVIOURS}
    S = {a: summ(load(r)) for a, r in RUNS.items() if load(r)}
    out = {'base_rate_weighted': base, 'arms': S}
    L = ['# MABe22 ECI P2: Delta 2 s vs no Delta on DINOv3-L/16 512 px block 20 (rules: scripts/benchmark/mabe22/'
         'eci_p2.sh)', '', 'primary AUROC = mean cf AUROC over oral_oral, oral_genital, approaches, huddles; SAE seeds '
         '320-325; full width 4096; weighted base rate oral_oral {:.4f}, oral_genital {:.4f}.'.format(
             base['oral_oral_contact'], base['oral_genital_contact']), '',
         '| arm | seeds | per-seed primary AUROC | primary AUROC mean ± SE | oral_oral cf AUROC / AP / honest top-1% | '
         'oral_genital cf AUROC / AP / honest top-1% |', '|---|---|---|---|---|---|']
    for a, s in S.items():
        cell = lambda b: f"{s['cf_auroc'][b]:.3f} / {s['cf_ap'][b]:.4f} / {s['cf_top1'][b]:.3f}"  # noqa: E731
        L.append(f"| {a} {LABEL[a]} | {len(s['seeds'])} | {', '.join(f'{x:.3f}' for x in s['primary'])} | "
                 f"{s['mean']:.4f} ± {s['se']:.4f} | {cell(SN[0])} | {cell(SN[1])} |")
    L += ['', 'per label, mean over seeds x splits: cf AUROC / cf AP', '',
          '| arm | ' + ' | '.join(SHORT[b] for b in BEHAVIOURS) + ' |', '|---|' + '---|' * len(BEHAVIOURS)]
    for a, s in S.items():
        L.append(f'| {a} | ' + ' | '.join(f"{s['cf_auroc'][b]:.3f} / {s['cf_ap'][b]:.4f}" for b in BEHAVIOURS) + ' |')
    if len(S) == 2 and all(len(s['seeds']) == 6 for s in S.values()):
        d, n = S['P2D'], S['P2N']
        C = {'primary AUROC': contrast(d['primary'], n['primary'])}
        for b in SN:
            C[f'{SHORT[b]} cf AP (descriptive)'] = contrast(d['cf_ap_seed'][b], n['cf_ap_seed'][b])
            C[f'{SHORT[b]} cf AP (descriptive)']['ratio_noDelta_over_Delta'] = n['cf_ap'][b] / d['cf_ap'][b]
        out['contrast'] = C
        L += ['', '## Contrast P2D - P2N (Delta minus no Delta), Welch 95% CI on 6 vs 6 per-seed values', '',
              '| metric | delta [95% CI] | no-Delta / Delta ratio | verdict |', '|---|---|---|---|']
        for k, c in C.items():
            r = f"{c['ratio_noDelta_over_Delta']:.2f}x" if 'ratio_noDelta_over_Delta' in c else '-'
            L.append(f"| {k} | {c['delta']:+.4f} [{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}] | {r} | {c['verdict']} |")
        hold = [b for b in SN if C[f'{SHORT[b]} cf AP (descriptive)']['ratio_noDelta_over_Delta'] <= 0.70
                and C[f'{SHORT[b]} cf AP (descriptive)']['ci'][0] > 0]
        ans = 'HOLDS' if len(hold) == 2 else ('PARTIAL' if hold else 'DOES NOT HOLD')
        out['answer'] = {'labels_holding': hold, 'answer': ans}
        L += ['', f'ANSWER (pre-registered rule): the DINOv2-B sniffing-AP loss without Delta {ans} on DINOv3-L '
              f'(labels meeting ratio <= 0.70 and CI excluding 0: {[SHORT[b] for b in hold] or "none"}). DINOv2-B '
              'reference (round 1, 1 s Delta, 3 graded seeds): cf AP oral_oral 0.0089 vs 0.0180 (0.49x), oral_genital '
              '0.0127 vs 0.0183 (0.69x).']
    P2.mkdir(parents=True, exist_ok=True)
    (P2 / 'p2.md').write_text('\n'.join(L) + '\n')
    (P2 / 'p2.json').write_text(json.dumps(out, indent=1, default=float))
    print('\n'.join(L))


if __name__ == '__main__':
    main()
