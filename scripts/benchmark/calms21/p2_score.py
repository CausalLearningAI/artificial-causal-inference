"""
CalMS21 ECI phase 2 scoring (rules verbatim in eci_recheck.sh; login safe). Per arm: per-seed primary AUROC (mean
cross-fitted best-neuron AUROC over the scored labels, AUROC-picked neuron, split seeds averaged first) for frame value
= max (the arm's 'full' read-out) and = mean ('mean_' read-out); contrasts with Welch 95% CI and the round-3 verdict
words; the MABe22 comparison table. Outputs OUT/p2/p2.md, OUT/p2/p2.json
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calms_common import OUT, PLAN  # noqa: E402
from eci_res_score import welch  # noqa: E402


def verdict(d, lo, hi):
    return 'WIN' if d >= 0.010 and lo > 0 else ('LOSES' if d <= -0.010 and hi < 0 else 'EQUIVALENT')


def load(run, mean=False):
    R = {}
    for f in sorted((OUT / 'eval').glob(f'{run}_s*.json')):
        m = re.fullmatch(rf'{re.escape(run)}_s(\d+)\.json', f.name)
        if m:
            j = json.loads(f.read_text())
            k = ('mean_' if mean else '') + j.get('full', 'm4096')
            if k in j:
                R[int(m.group(1))] = j[k]
    return R


def ps(R, labs, q='cf_auroc'):
    return {s: float(np.mean([np.mean([r[sp][b][q] for sp in r]) for b in labs])) for s, r in sorted(R.items())}


def mq(R, b, q):
    return float(np.mean([r[sp][b][q] for r in R.values() for sp in r])) if R else float('nan')


def main():
    labs = json.loads((PLAN / 'eval_plan.json').read_text())['scored_labels']
    B = (OUT / 'p2' / 'base.txt').read_text().split()[0]
    runs = {'base (2 s)': B, 'P_D0 (none)': f'P_D0_{B}', 'P_D1 (1 s)': f'P_D1_{B}', 'P_D4 (4 s)': f'P_D4_{B}',
            'P_W512': f'P_W512_{B}', 'P_N512': f'P_N512_{B}', 'P_K8': f'P_K8_{B}', 'P_NOMASK': f'P_NOMASK_{B}',
            'P_ORB': f'P_ORB_{B}'}
    D = {k: load(v) for k, v in runs.items()}
    Dm = {k: load(v, True) for k, v in runs.items()}
    S = {k: ps(v, labs) for k, v in D.items() if v}
    Sm = {k: ps(v, labs) for k, v in Dm.items() if v}
    out = {'base': B, 'labels': labs, 'arms': {}, 'contrasts': {}}
    L = [f'# CalMS21 phase 2 (eci_recheck.sh): base {B}', '',
         'primary AUROC = mean cross-fitted best-neuron AUROC over ' + ', '.join(labs) + '; 6 seeds; Welch 95% CI', '',
         '| arm | seeds | primary AUROC, max (mean ± SE) | primary AUROC, mean pooling | ' + ' | '.join(
             f'{b} AUROC / AP / top-1%' for b in labs) + ' |', '|---|---|---|---|' + '---|' * len(labs)]
    for k in runs:
        if k not in S:
            L.append(f'| {k} | missing |')
            continue
        v = np.array(list(S[k].values()))
        vm = np.array(list(Sm.get(k, {}).values()))
        out['arms'][k] = {'run': runs[k], 'per_seed_max': S[k], 'per_seed_mean': Sm.get(k),
                          'auroc': {b: mq(D[k], b, 'cf_auroc') for b in labs}, 'ap': {b: mq(D[k], b, 'cf_ap') for b in labs},
                          'top1': {b: mq(D[k], b, 'cf_top1') for b in labs}}
        a = out['arms'][k]
        L.append(f'| {k} | {len(v)} | {v.mean():.4f} ± {v.std(ddof=1) / np.sqrt(len(v)):.4f} | '
                 + (f'{vm.mean():.4f} ± {vm.std(ddof=1) / np.sqrt(len(vm)):.4f}' if len(vm) > 1 else '-') + ' | '
                 + ' | '.join(f"{a['auroc'][b]:.3f} / {a['ap'][b]:.3f} / {a['top1'][b]:.3f}" for b in labs) + ' |')
    C = [('2 s vs 1 s', 'base (2 s)', 'P_D1 (1 s)', S, 'WIN'),
         ('none vs 1 s', 'P_D0 (none)', 'P_D1 (1 s)', S, 'EQUIVALENT (sniffing AP x0.5)'),
         ('4 s vs 1 s', 'P_D4 (4 s)', 'P_D1 (1 s)', S, 'WIN'),
         ('none vs 2 s (descriptive)', 'P_D0 (none)', 'base (2 s)', S, None),
         ('4 s vs 2 s (descriptive)', 'P_D4 (4 s)', 'base (2 s)', S, None),
         ('plain 512 vs 4096', 'P_W512', 'base (2 s)', S, 'EQUIVALENT'),
         ('nested 128/256/512 vs plain 512', 'P_N512', 'P_W512', S, 'EQUIVALENT (weak)'),
         ('k 8 vs 16', 'P_K8', 'base (2 s)', S, 'LOSES'),
         ('no mask vs mask (mask misses the intruder in 8.3% of frames)', 'P_NOMASK', 'base (2 s)', S, 'LOSES'),
         ('orbit vs none', 'P_ORB', 'base (2 s)', S, 'WIN')]
    L += ['', '| choice | delta primary AUROC [95% CI] | CalMS21 verdict | MABe22 verdict | agree |', '|---|---|---|---|---|']
    for name, a, b, SS, mv in C:
        if a in SS and b in SS:
            d, lo, hi, se, df = welch(list(SS[a].values()), list(SS[b].values()))
            v = verdict(d, lo, hi)
            ag = '-' if mv is None else ('yes' if mv.split()[0] == v else 'no -> UNSURE (keep the simpler option)')
            out['contrasts'][name] = {'a': a, 'b': b, 'delta': d, 'ci': [lo, hi], 'verdict': v, 'mabe22': mv, 'agree': ag}
            L.append(f'| {name} | {d:+.4f} [{lo:+.4f}, {hi:+.4f}] | {v} | {mv or "-"} | {ag} |')
        else:
            L.append(f'| {name} | missing | | {mv or "-"} | |')
    # mean vs max on every arm (paired by seed: same SAE, same codes)
    L += ['', '## frame value: mean vs max pooling (same SAEs; Welch on per-seed primaries)', '',
          '| arm | delta mean - max [95% CI] | verdict |', '|---|---|---|']
    for k in runs:
        if k in S and k in Sm and len(Sm[k]) > 1:
            d, lo, hi, _, _ = welch(list(Sm[k].values()), list(S[k].values()))
            out['contrasts'][f'mean vs max: {k}'] = {'delta': d, 'ci': [lo, hi], 'verdict': verdict(d, lo, hi)}
            L.append(f'| {k} | {d:+.4f} [{lo:+.4f}, {hi:+.4f}] | {verdict(d, lo, hi)} |')
    (OUT / 'p2').mkdir(parents=True, exist_ok=True)
    (OUT / 'p2' / 'p2.md').write_text('\n'.join(L) + '\n')
    (OUT / 'p2' / 'p2.json').write_text(json.dumps(out, indent=1))
    print('\n'.join(L))


if __name__ == '__main__':
    main()
