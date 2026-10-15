"""
Mask-U test - scoring of the PRE-REGISTERED rule (verbatim in scripts/benchmark/eci_mask_u.sh; login safe, reads JSONs).

Per benchmark and arm (cur, U) and SAE seed: primary AUROC = mean cross-fitted best-neuron AUROC (AUROC-picked neuron,
full width 4096) over the primary labels (MABe22: oral_oral, oral_genital, approaches, huddles; CalMS21: attack,
investigation, mount), each label averaged over split seeds 0 and 1 first. delta = mean(U per-seed) - mean(cur
per-seed), Welch 95% CI on the 6 vs 6 per-seed values (mabe22 eci_confirm.welch).
    (i)   equivalent or better on BOTH benchmarks: delta > -0.010 OR the CI contains 0
    (ii)  MABe22 cf AP (AP-picked neuron, mean over seeds x splits) of U >= 0.9 x cur on oral_oral AND oral_genital
    (iii) keypoint recall per animal of U >= cur - 0.01 on MABe22 mouse, MABe22 beetle and CalMS21 (every animal)
    U is ADOPTED iff (i), (ii) and (iii) all hold.
Also reported: per-label cf AUROC / AP / honest top-1%, kept tokens per frame, theta*, firing.
Outputs: OUT/score.md, OUT/score.json (`python score.py [mabe]` = the MABe22 part only, for the interim report).

RESULTS (2026-10-15, seeds 400-405, rule frozen at commit 5447648 before any store / SAE job; OUT/score.md, score.json):
    MABe22 mouse primary AUROC: cur 0.8022 +- 0.0023, U 0.8016 +- 0.0046; delta -0.0007 [Welch 95% CI -0.0128, +0.0114]
        EQUIVALENT -> (i) holds; cf AP ratio U / cur oral_oral 0.77x, oral_genital 0.94x -> (ii) FAILS.
    CalMS21 primary AUROC: cur 0.8811 +- 0.0063, U 0.8835 +- 0.0061; delta +0.0024 [-0.0171, +0.0219] EQUIVALENT.
    Keypoint recall per animal (cur / U): MABe22 mouse 0.972 / 0.975, 0.973 / 0.973, 0.973 / 0.973 (all eval frames, IPW);
        CalMS21 resident 0.998 / 0.995, intruder (white) 0.643 / 0.738 (intruder missed in 19.1% / 14.6% of frames);
        MABe22 beetle 0.917 / 0.849, 1.000 / 0.966 -> (iii) FAILS (beetle).
    Decision: current per-domain rules STAY ((ii) and (iii) fail).
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / 'mabe22'))
from common import OUT  # noqa: E402
from eci_confirm import welch  # noqa: E402

PRIM = {'mabe': ['oral_oral_contact', 'oral_genital_contact', 'approaches', 'huddles'],
        'calms': ['attack', 'investigation', 'mount']}
NAME = {'mabe': 'MABe22 mouse', 'calms': 'CalMS21'}


def load(bench, arm):
    R = {}
    for f in sorted((OUT / bench / 'eval').glob(f'{arm}_s*.json')):
        m = re.fullmatch(rf'{arm}_s(\d+)\.json', f.name)
        if m:
            R[int(m.group(1))] = json.loads(f.read_text())
    return R


def per_seed(R, labels):
    return {s: float(np.mean([np.mean([r['m4096'][sp][b]['cf_auroc'] for sp in r['m4096']]) for b in labels]))
            for s, r in sorted(R.items())}


def mean_q(R, b, q):
    return float(np.mean([r['m4096'][sp][b][q] for r in R.values() for sp in r['m4096']]))


def bench_part(bench):
    C, U = load(bench, 'cur'), load(bench, 'U')
    if not C or not U:
        return None
    labs = list(next(iter(C.values()))['m4096']['split0'].keys())
    pc, pu = per_seed(C, PRIM[bench]), per_seed(U, PRIM[bench])
    d, lo, hi = welch(list(pu.values()), list(pc.values()))[:3]
    ci0 = lo <= 0 <= hi
    ok = d > -0.010 or ci0
    word = ('WIN' if d >= 0.010 and not ci0 else 'LOSES' if d <= -0.010 and not ci0 else 'EQUIVALENT')
    out = {'n_seeds': {'cur': len(C), 'U': len(U)}, 'per_seed_primary_auroc': {'cur': pc, 'U': pu},
           'primary_auroc_mean': {'cur': float(np.mean(list(pc.values()))), 'U': float(np.mean(list(pu.values())))},
           'primary_auroc_se': {a: float(np.std(list(v.values()), ddof=1) / np.sqrt(len(v))) for a, v in (('cur', pc), ('U', pu))},
           'delta_U_minus_cur': d, 'ci95': [lo, hi], 'verdict_word': word, 'cond_i_ok': bool(ok),
           'per_label': {a: {b: {q: mean_q(R, b, q) for q in ('cf_auroc', 'cf_ap', 'cf_top1')} for b in labs}
                         for a, R in (('cur', C), ('U', U))},
           'kept_per_eval_frame_w': {a: float(np.mean([r['kept_per_eval_frame_w'] for r in R.values()])) for a, R in (('cur', C), ('U', U))},
           'eval_frames_0_tokens': {a: int(np.mean([r['eval_frames_0_tokens'] for r in R.values()])) for a, R in (('cur', C), ('U', U))},
           'theta_star': {a: [r['calib'].get('theta_star', r['calib'].get('theta')) for r in R.values()] for a, R in (('cur', C), ('U', U))}}
    if bench == 'mabe':
        out['ap_ratio'] = {b: out['per_label']['U'][b]['cf_ap'] / out['per_label']['cur'][b]['cf_ap']
                           for b in ('oral_oral_contact', 'oral_genital_contact')}
        out['cond_ii_ok'] = bool(all(v >= 0.9 for v in out['ap_ratio'].values()))
    return out


def recall_part():
    """(iii) from the mask QA JSONs: MABe22 mouse + CalMS21 = every eval frame of the stores; beetle = 24 clips x 16 frames."""
    Q = {}
    for nm, f, key in (('MABe22 mouse', 'mabe_store.json', 'recall_per_animal_w'), ('CalMS21', 'calms_store.json', 'recall_per_animal_w'),
                       ('MABe22 beetle', 'beetle_final.json', None)):
        p = OUT / 'maskqa' / f
        if not p.exists():
            Q[nm] = None
            continue
        J = json.loads(p.read_text())
        if key:
            c, u = J['cur'][key], J['U'][key]
        else:
            c, u = J['summary']['cur']['recall_per_animal'], J['summary']['U']['recall_per_animal']
        Q[nm] = {a: {'cur': c[a], 'U': u[a], 'diff': u[a] - c[a], 'ok': bool(u[a] >= c[a] - 0.01)} for a in c}
    ok = all(v is not None and all(x['ok'] for x in v.values()) for v in Q.values())
    return Q, ok


def main():
    only = sys.argv[1:] or ['mabe', 'calms']
    S = {b: bench_part(b) for b in only}
    Q, ok3 = recall_part()
    L = ['# Mask-U test: one animal-mask rule (U) vs the current per-domain rules', '']
    for b, s in S.items():
        if s is None:
            L += [f'## {NAME[b]}: not scored yet', '']
            continue
        L += [f'## {NAME[b]} (primary AUROC = mean cf best-neuron AUROC over {", ".join(PRIM[b])})', '',
              '| arm | seeds | per-seed primary AUROC | primary AUROC mean ± SE | kept tokens / eval frame (w) |', '|---|---|---|---|---|']
        for a in ('cur', 'U'):
            L.append(f"| {a} | {s['n_seeds'][a]} | {', '.join(f'{v:.3f}' for v in s['per_seed_primary_auroc'][a].values())} | "
                     f"{s['primary_auroc_mean'][a]:.4f} ± {s['primary_auroc_se'][a]:.4f} | {s['kept_per_eval_frame_w'][a]:.1f} |")
        L += ['', f"primary AUROC delta U - cur = {s['delta_U_minus_cur']:+.4f} [Welch 95% CI {s['ci95'][0]:+.4f}, "
              f"{s['ci95'][1]:+.4f}] -> {s['verdict_word']}; condition (i) on this benchmark: {'holds' if s['cond_i_ok'] else 'FAILS'}"]
        if 'ap_ratio' in s:
            L.append(f"cf AP ratio U / cur: oral_oral {s['ap_ratio']['oral_oral_contact']:.2f}x, oral_genital "
                     f"{s['ap_ratio']['oral_genital_contact']:.2f}x -> condition (ii): {'holds' if s['cond_ii_ok'] else 'FAILS'}")
        L += ['', 'per label, mean over seeds x splits: cf AUROC / cf AP / honest top-1%', '',
              '| arm | ' + ' | '.join(s['per_label']['cur']) + ' |', '|---|' + '---|' * len(s['per_label']['cur'])]
        for a in ('cur', 'U'):
            L.append(f'| {a} | ' + ' | '.join(f"{v['cf_auroc']:.3f} / {v['cf_ap']:.4f} / {v['cf_top1']:.3f}"
                                              for v in s['per_label'][a].values()) + ' |')
        L.append('')
    L += ['## Keypoint recall per animal (condition iii: U >= cur - 0.01)', '', '| domain | animal | cur | U | U - cur | ok |',
          '|---|---|---|---|---|---|']
    for nm, v in Q.items():
        if v is None:
            L.append(f'| {nm} | (not measured yet) | | | | |')
            continue
        for a, x in v.items():
            L.append(f"| {nm} | {a} | {x['cur']:.3f} | {x['U']:.3f} | {x['diff']:+.3f} | {'yes' if x['ok'] else 'NO'} |")
    full = all(S.get(b) is not None for b in ('mabe', 'calms'))
    dec = None
    if full:
        c1 = S['mabe']['cond_i_ok'] and S['calms']['cond_i_ok']
        c2 = S['mabe']['cond_ii_ok']
        dec = {'cond_i': c1, 'cond_ii': c2, 'cond_iii': ok3, 'adopt_U': bool(c1 and c2 and ok3)}
        L += ['', f"## Decision: (i) {'holds' if c1 else 'FAILS'}, (ii) {'holds' if c2 else 'FAILS'}, (iii) "
              f"{'holds' if ok3 else 'FAILS'} -> {'U ADOPTED' if dec['adopt_U'] else 'current per-domain rules STAY'}"]
    tag = '' if full else '_' + '_'.join(only)
    (OUT / f'score{tag}.md').write_text('\n'.join(L) + '\n')
    (OUT / f'score{tag}.json').write_text(json.dumps({'bench': S, 'recall': Q, 'decision': dec}, indent=1))
    print('\n'.join(L))


if __name__ == '__main__':
    main()
