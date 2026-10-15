"""
MABe22 mouse ECI round 3: scoring of E1 / E2 / E3 / E5 / E6 under the PRE-REGISTERED rules of eci_round3.sh (login
safe: reads the eval JSONs of OUT/r3/eval/). E4 is scored by eci_r3_ethogram.py.

primary = mean cf AUROC over oral_oral, oral_genital, approaches, huddles (each label averaged over the 2 split seeds
first) at the dictionary's full width unless a read-out prefix is named; per-seed primaries 6 vs 6; Welch 95% CI
(eci_confirm.welch). Verdict: WIN iff primary-AUROC delta >= +0.010 and CI excludes 0 and cf AP oral_oral / oral_genital
>= 0.9 x control; LOSES iff delta <= -0.010 and CI excludes 0; otherwise EQUIVALENT (delta >= 0.010 with CI excluding 0
but an AP guard failing = "not a win (AP guard)"). Outputs OUT/r3/r3.md, OUT/r3/r3.json.

RESULTS (2026-10-15, rules committed 4fed43a before any job; E1b rule 9c86cad before any E1b job). Control R re-trained
on the round-3 store (masks reproduce the confirmation store: same kept patches per shard), seeds 200-205: per-seed
primary AUROC 0.754 0.791 0.795 0.795 0.815 0.817 (0.794 +- 0.009 SE). Deltas = primary AUROC [Welch 95% CI]:
  E1  R512 (Matryoshka 64-512) -0.039 [-0.062, -0.015] LOSES; R512plain +0.013 [-0.012, +0.037] EQUIVALENT (simpler,
      preferred; oo / og AP 0.98x / 0.96x); R512plain vs R512 +0.051 [+0.038, +0.064] (Matryoshka hurts at 512); R's
      512 / 1024 / 2048 prefixes = R exactly (+0.000); R512 prefixes 64 / 128 / 256: -0.069 / -0.054 / -0.045 vs R.
      E2 width rule keys on R512 -> W = 4096.
  E1b (seeds 210-215, control R512plain 0.802 +- 0.005): R512m128 (128/256/512) -0.024 [-0.055, +0.007] EQUIVALENT
      (og AP 0.71x; 2 of 6 seeds at 0.746); R512m128b (128/512) -0.049 [-0.066, -0.031] LOSES -> by the rule nesting
      from 128 is kept (one arm equivalent).
  E2  per-lag union (0.5/1/2/4 s) vs matched pool of 4 R seeds -0.004 [-0.020, +0.011] EQUIVALENT (no win); matched
      pool vs R +0.017 [-0.007, +0.040]; single lags 0.5 s -0.056 / 1 s -0.033 (lose) / 4 s +0.003; context-mean -0.027
      [-0.059, +0.006] EQUIVALENT (og AP 0.40x); multi-lag concat -0.009 [-0.039, +0.020] EQUIVALENT (R simpler).
      Union picks (AUROC, 24 per label, 0.5/1/2/4 s): oral_oral 0/0/8/16, oral_genital 0/0/8/16, approaches 4/3/6/11,
      huddles 0/2/4/18; contact 4/10/6/4, close 6/13/2/3.
  E3  token density (mouse length 2.2 / 3.0 / 4.5 patches at 336 / 448 / 672, eci_r3_bodylen): R336 -0.018 [-0.047,
      +0.011] EQUIVALENT (oo AP 0.59x); R672 +0.001 [-0.028, +0.031] EQUIVALENT (oo AP 0.71x). Orbit encoding of t and
      t-60 ~0.07 / 0.13 / 0.37 L40S GPU-s per frame.
  E5  shared mouse + beetle dictionary (equal token counts 784,816 + 784,816) -0.024 [-0.050, +0.003] EQUIVALENT ->
      shared preferred by the rule (oo / og AP 0.97x / 0.94x).
  E6  adapted DINOv2 (ssl_dapt objective, 10,866 steps, 2.0 L40S-h, student cos to stock 0.86) -0.010 [-0.044, +0.025]
      EQUIVALENT -> frozen preferred.
  Sniffing bar (honest top-1% >= 0.25 on oral_oral / oral_genital): not reached; best single value 0.061.
  E4: see eci_r3_ethogram.py / OUT/r3/e4/ethogram.md.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eci_common import BEHAVIOURS, OUT, SHORT  # noqa: E402
from eci_confirm import welch  # noqa: E402

P4 = ['oral_oral_contact', 'oral_genital_contact', 'approaches', 'huddles']
SN = ['oral_oral_contact', 'oral_genital_contact']
EV = OUT / 'r3' / 'eval'
SEEDS = list(range(200, 206))


def load(run, ro=None, seeds=None):
    """-> {seed: {split: per-behaviour}} at read-out ro (default: the run's full width)"""
    seeds = seeds or SEEDS
    R = {}
    for f in sorted(EV.glob(f'{run}_s*.json')):
        m = re.fullmatch(rf'{re.escape(run)}_s(\d+)\.json', f.name)
        if m and int(m.group(1)) in seeds:
            j = json.loads(f.read_text())
            R[int(m.group(1))] = j[ro or j['full']]
    return R


def summ(R):
    ps = {s: float(np.mean([np.mean([r[sp][b]['cf_auroc'] for sp in r]) for b in P4])) for s, r in sorted(R.items())}
    q = lambda b, k: float(np.mean([r[sp][b][k] for r in R.values() for sp in r]))  # noqa: E731
    v = np.array(list(ps.values()))
    return {'seeds': list(ps), 'per_seed': list(ps.values()), 'mean': float(v.mean()),
            'se': float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float('nan'),
            'auroc': {b: q(b, 'cf_auroc') for b in BEHAVIOURS}, 'ap': {b: q(b, 'cf_ap') for b in BEHAVIOURS},
            'top1': {b: q(b, 'cf_top1') for b in BEHAVIOURS},
            'max_top1': {b: max(d['top1']['test_top1'] for r in R.values() for sp in r for d in r[sp][b]['dirs'])
                         for b in SN}}


def verdict(a, c):
    d, lo, hi, se, df = welch(a['per_seed'], c['per_seed'])
    apr = {b: a['ap'][b] / c['ap'][b] for b in SN}
    if d >= 0.010 and lo > 0:
        v = 'WIN' if min(apr.values()) >= 0.9 else 'not a win (AP guard)'
    elif d <= -0.010 and hi < 0:
        v = 'LOSES'
    else:
        v = 'EQUIVALENT'
    return {'delta': d, 'ci': [lo, hi], 'se': se, 'df': df, 'ap_ratio': apr, 'verdict': v}


def row(name, a, c=None, note=''):
    vd = verdict(a, c) if c is not None and len(a['per_seed']) > 1 else None
    ds = (f"primary AUROC {vd['delta']:+.4f} [{vd['ci'][0]:+.4f}, {vd['ci'][1]:+.4f}]" if vd else '-')
    ar = (f"oo AP {vd['ap_ratio']['oral_oral_contact']:.2f}x / og AP {vd['ap_ratio']['oral_genital_contact']:.2f}x"
          if vd else '-')
    t1 = f"{a['top1']['oral_oral_contact']:.3f} / {a['top1']['oral_genital_contact']:.3f}"
    v = (vd['verdict'] + (f' ({note})' if note else '')) if vd else (note or 'control')
    line = (f"| {name} | {len(a['per_seed'])} | {', '.join(f'{x:.3f}' for x in a['per_seed'])} | "
            f"{a['mean']:.4f} ± {a['se']:.4f} | {ds} | {ar} | {t1} | {v} |")
    return line, vd


HDR = ['| arm | seeds | per-seed primary AUROC | primary AUROC mean ± SE | delta vs control [Welch 95% CI] | '
       'AP ratio vs control | honest top-1% oo / og | verdict |', '|---|---|---|---|---|---|---|---|']


def per_label(rows):
    L = ['', 'per label, mean over seeds x splits: cf AUROC / cf AP', '',
         '| arm | ' + ' | '.join(SHORT[b] for b in BEHAVIOURS) + ' |', '|---|' + '---|' * len(BEHAVIOURS)]
    for n, s in rows:
        L.append(f'| {n} | ' + ' | '.join(f"{s['auroc'][b]:.3f} / {s['ap'][b]:.4f}" for b in BEHAVIOURS) + ' |')
    return L


def section(title, ctrl_name, ctrl, arms, out, key):
    """arms: list of (name, summ, control summ or None, note)"""
    L = [f'## {title}', ''] + HDR
    ln, _ = row(ctrl_name, ctrl)
    L.append(ln)
    out[key] = {'control': {ctrl_name: ctrl}, 'arms': {}}
    rows = [(ctrl_name, ctrl)]
    for n, s, c, note in arms:
        if s is None or not s['per_seed']:
            L.append(f'| {n} | 0 | - | - | - | - | - | missing |')
            continue
        ln, vd = row(n, s, c, note)
        L.append(ln)
        out[key]['arms'][n] = {'summary': s, 'vs': vd}
        rows.append((n, s))
    return L + per_label(rows) + ['']


def S(run, ro=None, seeds=None):
    R = load(run, ro, seeds)
    return summ(R) if R else None


def main():
    out, L = {}, ['# MABe22 ECI round 3 (pre-registered rules: scripts/benchmark/mabe22/eci_round3.sh)', '',
                  'primary AUROC = mean cross-fitted AUROC over oral_oral, oral_genital, approaches, huddles; '
                  'SAE seeds 200-205; Welch 95% CI on 6 vs 6 per-seed primaries; AP ratio = cf AP vs control.', '']
    R = S('R')
    W = 4096
    if R:
        # E1
        r512, r512p = S('R512'), S('R512plain')
        arms = [('R512 (d 512, Matryoshka)', r512, R, ''), ('R512plain (d 512, no Matryoshka)', r512p, R, ''),
                ('R512plain vs R512', r512p, r512, 'contrast: Matryoshka at 512')]
        for m in (512, 1024, 2048):
            arms.append((f'R prefix {m}', S('R', f'm{m}'), R, 'descriptive'))
        for m in (64, 128, 256):
            arms.append((f'R512 prefix {m} (vs R)', S('R512', f'm{m}'), R, 'descriptive'))
            arms.append((f'R512 prefix {m} (vs R512)', S('R512', f'm{m}'), r512, 'descriptive'))
        L += section('E1 width (control R, d 4096)', 'R', R, arms, out, 'E1')
        if r512 and len(r512['per_seed']) == 6:
            v = verdict(r512, R)['verdict']
            W = 1024 if v in ('EQUIVALENT', 'WIN') else 4096
            L += [f'E1 -> E2 width decision (pre-registered): R512 vs R is {v} -> W = {W}', '']
            out['E2_width'] = W
        # E1b (addendum): seeds 210-215, control R512plain at the same seeds
        B = list(range(210, 216))
        cp = S('R512plain', seeds=B)
        if cp:
            m1, m2, R2 = S('R512m128', seeds=B), S('R512m128b', seeds=B), S('R', seeds=B)
            arms = [('R512m128 (prefixes 128/256/512)', m1, cp, ''), ('R512m128b (prefixes 128/512)', m2, cp, ''),
                    ('R512m128 vs R (seeds 210-215)', m1, R2, 'vs R'), ('R512m128b vs R (seeds 210-215)', m2, R2, 'vs R'),
                    ('R512plain s210-215 vs R s210-215', cp, R2, 'descriptive'),
                    ('R512m128 vs R512plain s200-205', m1, r512p, 'descriptive, unmatched seeds'),
                    ('R512m128b vs R512plain s200-205', m2, r512p, 'descriptive, unmatched seeds')]
            for nm, run in (('R512m128', 'R512m128'), ('R512m128b', 'R512m128b')):
                arms.append((f'{nm} prefix 128 (vs R512plain)', S(run, 'm128', B), cp, 'descriptive'))
            L += section('E1b Matryoshka from 128 at d 512 (control R512plain, seeds 210-215)', 'R512plain (s210-215)',
                         cp, arms, out, 'E1b')
            vs = [verdict(a, cp)['verdict'] for a in (m1, m2) if a and len(a['per_seed']) == 6]
            if len(vs) == 2:
                keep = any(v in ('EQUIVALENT', 'WIN') for v in vs)
                L += [f'E1b decision (pre-registered): verdicts {vs} -> ' +
                      ('nesting from 128 is KEPT' if keep else 'plain 512 stays'), '']
        # E2
        suf = '' if W == 4096 else f'_w{W}'
        RW = S(f'R{suf}')
        if RW:
            U, MP = S(f'U{suf}'), S(f'MP{suf}')
            arms = [('per-lag union vs matched pool', U, MP, 'THE E2 rule'), ('per-lag union vs R_W', U, RW, ''),
                    ('matched pool vs R_W', MP, RW, 'descriptive'),
                    ('lag 0.5 s (L15)', S(f'L15{suf}'), RW, 'descriptive'),
                    ('lag 1 s (L30)', S(f'L30{suf}'), RW, 'descriptive'),
                    ('lag 4 s (L120)', S(f'L120{suf}'), RW, 'descriptive'),
                    ('context-mean (CM)', S(f'CM{suf}'), RW, ''), ('multi-lag concat (ML)', S(f'ML{suf}'), RW, '')]
            L += section(f'E2 time scales (width W = {W}; control R_W = R{suf})', f'R{suf}', RW, arms, out, 'E2')
            # lag attribution of the union's picks
            lagname = ['0.5 s', '1 s', '2 s', '4 s']
            cnt = {}
            for f in sorted(EV.glob(f'U{suf}_s*.json')):
                j = json.loads(f.read_text())
                for b, d in j['member_of_pick'].items():
                    for sp in d.values():
                        for q in ('auc', 'ap'):
                            for mbr in sp[q]:
                                cnt.setdefault(b, {}).setdefault(q, np.zeros(4, int))[mbr] += 1
            if cnt:
                L += ['### E2: lag of the selected neuron in the per-lag union (counts over runs x splits x directions)',
                      '', '| label | AUROC pick: 0.5 / 1 / 2 / 4 s | AP pick: 0.5 / 1 / 2 / 4 s | modal lag (AUROC pick) |',
                      '|---|---|---|---|']
                for b in BEHAVIOURS:
                    if b in cnt:
                        a_, p_ = cnt[b]['auc'], cnt[b]['ap']
                        L.append(f"| {SHORT[b]} | {' / '.join(map(str, a_))} | {' / '.join(map(str, p_))} | "
                                 f"{lagname[int(np.argmax(a_))]} |")
                out['E2_lag_picks'] = {b: {q: v.tolist() for q, v in d.items()} for b, d in cnt.items()}
                L.append('')
        # E3 / E5 / E6
        L += section('E3 token density (control R, 448 px)', 'R', R,
                     [('R336 (24 x 24 grid)', S('R336'), R, ''), ('R672 (48 x 48 grid)', S('R672'), R, '')], out, 'E3')
        L += section('E5 shared mouse + beetle dictionary (control R)', 'R', R,
                     [('RMIX (shared)', S('RMIX'), R, 'shared preferred if equivalent')], out, 'E5')
        L += section('E6 adapted encoder (control R, frozen)', 'R', R,
                     [('RA (adapted DINOv2)', S('RA'), R, 'frozen preferred if equivalent')], out, 'E6')
    # sniffing bar
    best = []
    for f in sorted(EV.glob('*_s*.json')):
        j = json.loads(f.read_text())
        r = j[j['full']]
        for b in SN:
            mx = max(d['top1']['test_top1'] for sp in r for d in r[sp][b]['dirs'])
            best.append((mx, f.stem, b))
    best.sort(reverse=True)
    L += ['## Sniffing-neuron bar (honest top-1% >= 0.25 on oral_oral or oral_genital, any single cross-fitted test '
          'value)', '', f"- reached: {any(m >= 0.25 for m, _, _ in best)}; best: " +
          '; '.join(f'{n} {SHORT[b]} honest top-1% {m:.3f}' for m, n, b in best[:5]), '']
    out['sniff_best'] = best[:20]
    (OUT / 'r3' / 'r3.md').write_text('\n'.join(L) + '\n')
    (OUT / 'r3' / 'r3.json').write_text(json.dumps(out, indent=1, default=float))
    print('\n'.join(L))


if __name__ == '__main__':
    main()
