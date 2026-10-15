"""
CalMS21 ECI - ROLE SWAP (rules: eci_swap.sh): tables from OUT/eval/<arm>_s<seed>.json (login safe, small JSONs).
-> results/vision/benchmark_calms21/swap/swap.{md,json}
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / 'results/vision/benchmark_calms21/swap'
T1 = ['attack', 'investigation', 'mount']
T3 = ['sniff_face', 'approach', 'mount_attempt', 'intromission']
ARMS = {'S1': 'D3L-64, 2 s (reference)', 'S2': 'D3L-64, no Delta', 'S3': 'D2L-64, 2 s',
        'S1r': 'D3L-64, 2 s, reduced pool (S4 control)', 'S4': 'D3L-64, 2 s, orbit, reduced pool',
        'S5': 'D3L-64, 2 s, NO MASK (EXPLORATORY)'}
CON = [('S2', 'S1', 'no Delta vs 2 s'), ('S3', 'S1', 'DINOv2-L vs DINOv3-L'),
       ('S4', 'S1r', 'orbit vs none, pool-matched (the orbit verdict)'),
       ('S4', 'S1', 'orbit + reduced pool vs S1 (requested; confounded)'),
       ('S5', 'S1', 'no mask vs mask (EXPLORATORY, outside the rule)')]
BAR = 0.25


def welch(a, b):
    a, b = np.asarray(a), np.asarray(b)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    d = a.mean() - b.mean()
    h = stats.t.ppf(0.975, df) * se
    return float(d), float(d - h), float(d + h), float(df)


def verdict(d, lo, hi):
    if d >= 0.010 and lo > 0:
        return 'WIN'
    if d <= -0.010 and hi < 0:
        return 'LOSES'
    return 'EQUIVALENT'


def main():
    keep = json.loads((OUT / 'plan' / 'halves.json').read_text())['pair_primary_labels']
    R = {}
    for a in ARMS:
        fs = sorted((OUT / 'eval').glob(f'{a}_s*.json'))
        if fs:
            R[a] = [json.loads(f.read_text()) for f in fs]
    S = {}
    for a, rr in R.items():
        S[a] = {'seeds': [r['seed'] for r in rr],
                'pair': [float(np.mean([r['scores'][b]['cf_auroc'] for b in keep])) for r in rr],
                'task1': [float(np.mean([r['scores'][b]['cf_auroc'] for b in T1])) for r in rr],
                'per_label': {}}
        for b in T1 + T3:
            sc = [r['scores'][b] for r in rr]
            br = sc[0]['weighted_base_rate']
            S[a]['per_label'][b] = {
                'cf_auroc': float(np.mean([s['cf_auroc'] for s in sc])),
                'cf_auroc_se': float(np.std([s['cf_auroc'] for s in sc], ddof=1) / np.sqrt(len(sc))) if len(sc) > 1 else None,
                'cf_ap': float(np.mean([s['cf_ap'] for s in sc])), 'base_rate': br,
                'ap_lift': float(np.mean([s['cf_ap'] for s in sc]) / br),
                'cf_top1': float(np.mean([s['cf_top1'] for s in sc])),
                'cf_top1_max_seed': float(np.max([s['cf_top1'] for s in sc])),
                'rate_auc_pick': float(np.mean([s['cf_rate_auc_pick'] for s in sc])),
                'rate_top1_pick': float(np.mean([s['cf_top1_rate'] for s in sc])),
                'n_frames': sc[0]['n_frames'], 'n_pos': sc[0]['n_pos'],
                **{k: float(np.mean([s_[k] for s_ in sc])) for k in ('cf_rprec', 'cf_rprec_auc_pick', 'cf_f1max',
                                                                    'cf_f1max_auc_pick') if k in sc[0]}}
    L = ['# CalMS21 role swap (eci_swap.sh): single neurons vs PAIR behaviours', '',
         'SAE pool = 19 task1 TEST videos (labels unread); scoring = 70 task1 TRAIN videos; video-level cross-fitting, '
         '10 split seeds x 2 directions; 6 SAE seeds (400-405) per arm.',
         f'pair primary AUROC = mean cross-fitted best-neuron AUROC over {", ".join(keep)} (each on its own sequences); '
         'task-1 primary AUROC = the same over attack, investigation, mount (sanity).', '',
         '| arm | what | seeds | per-seed pair primary AUROC | pair primary AUROC mean ± SE | task-1 primary AUROC mean ± SE |',
         '|---|---|---|---|---|---|']
    for a, s in S.items():
        p, t = np.array(s['pair']), np.array(s['task1'])
        se = lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float('nan')  # noqa: E731
        L.append(f"| {a} | {ARMS[a]} | {len(p)} | {', '.join(f'{x:.3f}' for x in p)} | {p.mean():.4f} ± {se(p):.4f} | "
                 f"{t.mean():.4f} ± {se(t):.4f} |")
    L += ['', '| contrast | delta pair primary AUROC [Welch 95% CI] | df | verdict | delta task-1 primary AUROC [95% CI] '
          '(descriptive) |', '|---|---|---|---|---|']
    C = []
    for x, y, nm in CON:
        if x in S and y in S and len(S[x]['pair']) > 1 and len(S[y]['pair']) > 1:
            d, lo, hi, df = welch(S[x]['pair'], S[y]['pair'])
            d1, lo1, hi1, _ = welch(S[x]['task1'], S[y]['task1'])
            v = verdict(d, lo, hi)
            C.append({'contrast': f'{x} vs {y}', 'what': nm, 'delta': d, 'lo': lo, 'hi': hi, 'df': df, 'verdict': v,
                      'task1_delta': d1, 'task1_lo': lo1, 'task1_hi': hi1})
            L.append(f'| {x} vs {y}: {nm} | {d:+.4f} [{lo:+.4f}, {hi:+.4f}] | {df:.1f} | {v} | {d1:+.4f} [{lo1:+.4f}, {hi1:+.4f}] |')
    L += ['', '## per label (mean over seeds of the 10-split means)', '',
          'cf AUROC (AUROC pick) / cf AP (AP pick) / AP lift = cf AP / weighted base rate / honest top-1% (top-1% pick, '
          'neurons firing on >= 1% of frames) / weighted firing rate of the AUROC pick / of the top-1% pick', '']
    labs = T3 + T1
    L += ['| arm | ' + ' | '.join(labs) + ' |', '|---|' + '---|' * len(labs)]
    L.append('| base rate (frames, positives) | ' + ' | '.join(
        f"{S[next(iter(S))]['per_label'][b]['base_rate']:.4f} ({S[next(iter(S))]['per_label'][b]['n_frames']}, "
        f"{S[next(iter(S))]['per_label'][b]['n_pos']})" for b in labs) + ' |' if S else '')
    for a, s in S.items():
        L.append(f'| {a} | ' + ' | '.join(
            f"{s['per_label'][b]['cf_auroc']:.3f} / {s['per_label'][b]['cf_ap']:.4f} / x{s['per_label'][b]['ap_lift']:.1f} / "
            f"{s['per_label'][b]['cf_top1']:.3f} / {s['per_label'][b]['rate_auc_pick']:.3f} / {s['per_label'][b]['rate_top1_pick']:.3f}"
            for b in labs) + ' |')
    L += ['', '## addendum 2 (secondary, not part of any verdict): R-precision and F1-max', '',
          'R-precision = weighted precision among the top frames holding the positives\' weight mass; F1-max = best F1 '
          'over thresholds. (b) = neuron picked by the same metric on the selection half, (a) = the AUROC-picked neuron; '
          'both scored on the other half, 10 splits x 2 directions, mean over seeds. Base rate = weighted positive rate.', '',
          '| arm | ' + ' | '.join(labs) + ' |', '|---|' + '---|' * len(labs)]
    if S:
        a0 = next(iter(S))
        L.append('| base rate | ' + ' | '.join(f"{S[a0]['per_label'][b]['base_rate']:.4f}" for b in labs) + ' |')
    for a, s in S.items():
        pl = s['per_label']
        if 'cf_rprec' not in pl[labs[0]]:
            continue
        L.append(f'| {a} R-prec (b) / R-prec (a) / F1-max (b) / F1-max (a) | ' + ' | '.join(
            f"{pl[b]['cf_rprec']:.3f} / {pl[b]['cf_rprec_auc_pick']:.3f} / {pl[b]['cf_f1max']:.3f} / "
            f"{pl[b]['cf_f1max_auc_pick']:.3f}" for b in labs) + ' |')
    L += ['', f'## sniffing bar: honest top-1% on sniff_face >= {BAR}', '']
    for a, s in S.items():
        t = s['per_label']['sniff_face']['cf_top1']
        L.append(f"- {a}: sniff_face honest top-1% {t:.3f} (best seed {s['per_label']['sniff_face']['cf_top1_max_seed']:.3f})"
                 f" -> {'PASSES' if t >= BAR else 'does not pass'}")
    (OUT / 'swap.md').write_text('\n'.join(L) + '\n')
    (OUT / 'swap.json').write_text(json.dumps({'arms': S, 'contrasts': C, 'pair_labels': keep}, indent=1))
    print('\n'.join(L))


if __name__ == '__main__':
    sys.exit(main())
