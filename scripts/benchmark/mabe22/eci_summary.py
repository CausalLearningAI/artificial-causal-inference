"""
MABe22 mouse ECI evaluation: tables + pre-registered verdicts from the eval / probe JSONs (login safe, reads JSONs).

Per arm and behaviour: cross-fitted AUROC (AUROC-picked neuron), AP (AP-picked), honest top-1% (top-1%-picked among
neurons firing on >= 1% of the selection half), weighted firing rate of the AUROC- and top-1%-picked neurons on the test
half; mean +- sd over SAE seeds x split seeds (x 2 directions averaged inside each run).
PRE-REGISTERED RULE (fixed in the brief before any result):
    primary = oral_oral, oral_genital, oral_ear, approaches, huddles; secondary = contact, close, watching; chases
    reported only. An arm WINS vs A0 iff mean primary cf AUROC rises by >= 0.01 AND by > 2 x the pooled seed sd
    (sd over SAE seeds of the per-seed primary mean, split seeds averaged; pooled = sqrt((sd_arm^2 + sd_A0^2) / 2)),
    AND no primary label's cf AP falls below 0.9 x A0's. EQUIVALENT iff |delta| <= 0.01 (then the simpler arm is
    preferred: no Matryoshka (A3), no mask (A6), no delta (A4), no theta (A2)).
    Sniffing-neuron bar: any neuron with honest top-1% >= 0.25 on oral_oral or oral_genital (cross-fitted test value).
Outputs: OUT/table.md, OUT/summary.json

RESULTS (2026-10-15; OUT/table.md, summary.json). Data: 18,496 submission eval frames from 1,828 clips (5 fps grid, label
index 30..1794; strata oral 4,496 (all) / approach-watch-chase 2,500 / huddle 1,500 / rest 10,000, IPW weights 1 / 6.33 /
28.3 / 47.6, weighted base rates = population rates); SAE pool 10,192 userTrain frames (13 / clip), theta* on 2,352
held-out userTrain frames. Label rows verified from keypoint geometry; video frame = label index + 1 (grey under moving-
mouse keypoints: 48.3 at +1 vs 49.1 at 0, 55.1 at -1). Mask: the lab rule missed still mice (keypoint recall 0.735,
18.6% of mice missed, 57% of huddle frames miss a mouse); + the arena-quadrilateral dark cue: recall 0.973, 0 of 18,428
frames miss a mouse, 77 kept patches / frame (7.5% of the 32 x 32 grid).
A0 (3 seeds x 2 splits, m4096): AUROC / AP / honest top-1% oral_oral 0.764 / 0.018 / 0.028, oral_genital 0.805 / 0.018 /
0.039, oral_ear 0.757 / 0.002 / 0.006, approaches 0.725 / 0.008 / 0.025, huddles 0.650 / 0.119 / 0.119; the AUROC-picked
oral neurons fire on 26-33% of frames (broad "two mice touching" units). Prefixes 512 / 1024 equal within noise.
Ceiling (mil_ctx, A0 tokens): oral_oral 0.883 / 0.053, oral_genital 0.869 / 0.037, huddles 0.809 / 0.260.
Pose reference (-min nose-nose): oral_oral AUROC 0.902, AP 0.016, top-1% 0.005; (-min nose-tailbase): oral_genital
0.939 / 0.033 / 0.039.
Pre-registered verdicts vs A0 (primary mean AUROC 0.7404): no arm WINS. Equivalent and simpler: A3 (no Matryoshka,
+0.008) and A4 (no delta, -0.002; but sniffing AP x0.49 / 0.69). Not a win: D2 (2 s delta, +0.022 < 2 x pooled sd 0.027;
oral_ear AP x0.44), A7 (D4 orbit, +0.013). Worse: D02, D05, A6 (no mask, -0.019), A2k8 / A2k16 (frame budget, -0.030),
A5 (224, -0.064), A1 (k 8, -0.049). Sniffing bar (top-1% >= 0.25): not reached by any neuron (best 0.064).
Robustness, 6 SAE seeds (not the verdict): D2 +0.020 (se 0.009), A7 +0.024 (se 0.009), X_D4 (4 s) +0.023 (se 0.012),
A3 +0.005, A4 -0.002.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eci_common import BEHAVIOURS, OUT, PLAN, PRIMARY, SHORT  # noqa: E402

ORDER = ['A0', 'A0_labmask', 'A4', 'D02', 'D05', 'D2', 'A6', 'A3', 'A2k8', 'A2k16', 'A5', 'A1', 'A7', 'X_D4', 'X_A7D2', 'X_A3A4', 'X_D2A3']
SIMPLER = {'A3': 'no Matryoshka', 'A6': 'no mask', 'A4': 'no delta', 'A2k8': 'no theta', 'A2k16': 'no theta'}
ms = lambda a, n=3: f'{np.mean(a):.{n}f} ± {np.std(a):.{n}f}' if len(a) > 1 else (f'{np.mean(a):.{n}f}' if a else '-')  # noqa


def load():
    R = {}
    for d, suf in ((OUT / 'eval', ''), (OUT / 'labmask' / 'eval', '_labmask')):
        for f in sorted(d.glob('*_s*.json')) if d.exists() else []:
            arm, seed = re.fullmatch(r'(.+)_s(\d+)\.json', f.name).groups()
            R.setdefault(arm + suf, {})[int(seed)] = json.loads(f.read_text())
    return R


def runs(R, arm, ro='m4096'):
    """-> list of (seed, split, per-behaviour dict)"""
    return [(s, sp, r[ro][sp]) for s, r in sorted(R[arm].items()) if ro in r for sp in sorted(r[ro])]


def main():
    R = load()
    E = json.loads((PLAN / 'eval_plan.json').read_text())
    arms = [a for a in ORDER if a in R]
    S, L = {}, ['# MABe22 mouse triplets: ECI SAE neurons vs behaviour labels', '',
                f"Eval: {E['n_frames']:,} submission frames from {E['n_clips']} clips (stratified, inverse-probability "
                'weighted); SAE trained on userTrain only; cross-fitted at the clip level (2 halves x 2 split seeds); '
                'mean ± sd over SAE seeds x split seeds.', '',
                '| label | weighted base rate | positives in sample | clips with a positive |', '|---|---|---|---|']
    for b in BEHAVIOURS:
        L.append(f"| {SHORT[b]} | {E['weighted_base_rate'][b]:.4f} | {E['n_pos'][b]} | {E['n_clips_with_pos'][b]} |")
    L.append('')
    for arm in arms:
        for ro in ('m512', 'm1024', 'm4096'):
            rr = runs(R, arm, ro)
            if not rr:
                continue
            s = {}
            for b in BEHAVIOURS:
                s[b] = {q: [r[b][q] for _, _, r in rr] for q in ('cf_auroc', 'cf_ap', 'cf_top1', 'cf_rate_auc_pick',
                                                                 'cf_top1_rate')}
                s[b]['max_test_top1'] = max(d['top1']['test_top1'] for _, _, r in rr for d in r[b]['dirs'])
            seeds = sorted({sd for sd, _, _ in rr})
            per_seed = [np.mean([np.mean([r[b]['cf_auroc'] for b in PRIMARY]) for sd, _, r in rr if sd == x])
                        for x in seeds]
            s['primary_mean_auroc'] = float(np.mean(per_seed))
            s['primary_seed_means'] = per_seed
            s['primary_seed_sd'] = float(np.std(per_seed)) if len(per_seed) > 1 else float('nan')
            s['n_runs'] = len(rr)
            s['seeds'] = seeds
            st = [R[arm][x]['eval_stats'] for x in seeds]
            s['eval_l0'] = [x['l0'] for x in st]
            s['dead_frac_eval'] = [R[arm][x]['firing']['dead_frac_eval'] for x in seeds]
            S[f'{arm}/{ro}'] = s
            L += [f'## {arm} ({ro}; runs {len(rr)}: SAE seeds {seeds} x 2 splits)', '',
                  '| label | AUROC | AP | honest top-1% | rate of AUROC pick | rate of top-1% pick | best single top-1% |',
                  '|---|---|---|---|---|---|---|']
            for b in BEHAVIOURS:
                x = s[b]
                L.append(f"| {SHORT[b]}{' (flagged: 30 clips)' if b == 'chases' else ''} | {ms(x['cf_auroc'])} | "
                         f"{ms(x['cf_ap'], 4)} | {ms(x['cf_top1'])} | {ms(x['cf_rate_auc_pick'])} | "
                         f"{ms(x['cf_top1_rate'])} | {x['max_test_top1']:.3f} |")
            L += ['', f"primary mean AUROC {s['primary_mean_auroc']:.4f} (per seed {np.round(per_seed, 4).tolist()}); "
                  f"eval L0 {np.round(s['eval_l0'], 2).tolist()}; dead frac {np.round(s['dead_frac_eval'], 4).tolist()}", '']
    # verdicts
    V = {}
    if 'A0/m4096' in S:
        a0 = S['A0/m4096']
        L += ['## Grid vs A0 (pre-registered rule, prefix 4096)', '',
              '| arm | primary mean AUROC | delta | pooled seed sd | AP ratio oo / og / oe / appr / hud | verdict |',
              '|---|---|---|---|---|---|']
        for arm in arms:
            if arm == 'A0':
                continue
            s = S[f'{arm}/m4096']
            dlt = s['primary_mean_auroc'] - a0['primary_mean_auroc']
            psd = float(np.sqrt((s['primary_seed_sd'] ** 2 + a0['primary_seed_sd'] ** 2) / 2))
            apr = {b: float(np.mean(s[b]['cf_ap']) / np.mean(a0[b]['cf_ap'])) for b in PRIMARY}
            win = dlt >= 0.01 and dlt > 2 * psd and min(apr.values()) >= 0.9
            eq = abs(dlt) <= 0.01
            verdict = 'WINS' if win else ('equivalent' + (f' (simpler: {SIMPLER[arm]} -> preferred)' if arm in SIMPLER
                                                          else '') if eq else ('worse' if dlt < 0 else 'not a win'))
            V[arm] = {'delta': dlt, 'pooled_sd': psd, 'ap_ratio': apr, 'win': bool(win), 'equivalent': bool(eq),
                      'verdict': verdict}
            if arm.startswith('X_'):
                verdict = 'exploratory (post hoc): ' + verdict
                V[arm]['verdict'] = verdict
            L.append(f"| {arm} | {s['primary_mean_auroc']:.4f} | {dlt:+.4f} | {psd:.4f} | "
                     f"{' / '.join(f'{apr[b]:.2f}' for b in PRIMARY)} | {verdict} |")
        L.append('')
    # sniffing neuron bar
    sn = {}
    for k, s in S.items():
        for b in ('oral_oral_contact', 'oral_genital_contact'):
            sn[f'{k}/{b}'] = {'mean_cf_top1': float(np.mean(s[b]['cf_top1'])), 'max_single': s[b]['max_test_top1']}
    any_bar = any(v['max_single'] >= 0.25 for v in sn.values())
    any_mean = any(v['mean_cf_top1'] >= 0.25 for v in sn.values())
    L += ['## Sniffing-neuron bar (honest top-1% >= 0.25 on oral_oral or oral_genital)', '',
          f'- any single cross-fitted test value >= 0.25: {any_bar}; any arm mean >= 0.25: {any_mean}',
          '- best per arm: ' + '; '.join(f"{k.replace('_contact', '')} mean {v['mean_cf_top1']:.3f} max {v['max_single']:.3f}"
                                         for k, v in sn.items() if k.split('/')[1] == 'm4096'), '']
    # supervised ceiling
    C = {}
    for f in sorted((OUT / 'probe').glob('probe_*.json')) if (OUT / 'probe').exists() else []:
        for x in json.loads(f.read_text())['summary']:
            C[(x['behaviour'], x['probe'] if x['res'] == 'A0' else f"{x['probe']}_{x['res']}")] = x
    if C:
        L += ['## Supervised ceiling on the A0 eval tokens z_t (clip-grouped 5-fold CV, weighted): AUROC / AP', '',
              '| label | n_fg only | mean_lin | mil_ctx | mean_lin (A7 orbit tokens) | mil_ctx (A7 orbit tokens) |',
              '|---|---|---|---|---|---|']
        for b in BEHAVIOURS:
            f_ = lambda p: (f"{C[(b, p)]['auroc_mean']:.3f} / {C[(b, p)]['ap_mean']:.4f}" if (b, p) in C else '-')  # noqa
            L.append(f'| {SHORT[b]} | {f_("n_fg_only")} | {f_("mean_lin")} | {f_("mil_ctx")} | {f_("mean_lin_A7")} | '
                     f'{f_("mil_ctx_A7")} |')
        L.append('')
    # robustness: 3 extra SAE seeds (3 4 5) for some arms, OUT/extra_seeds (NOT part of the pre-registered verdict)
    xd = OUT / 'extra_seeds' / 'eval'
    if xd.exists() and 'A0' in R:
        X = {}
        for f in sorted(xd.glob('*_s*.json')):
            arm, seed = re.fullmatch(r'(.+)_s(\d+)\.json', f.name).groups()
            X.setdefault(arm, {})[int(seed)] = json.loads(f.read_text())
        def seed_means(D):
            return [float(np.mean([np.mean([r['m4096'][sp][b]['cf_auroc'] for b in PRIMARY]) for sp in r['m4096']]))
                    for _, r in sorted(D.items())]
        def ap_mean(D, b):
            return float(np.mean([r['m4096'][sp][b]['cf_ap'] for r in D.values() for sp in r['m4096']]))
        if 'A0' in X:
            A0all = {**R['A0'], **X['A0']}
            m0 = seed_means(A0all)
            L += ['## Robustness (NOT the pre-registered verdict): 6 SAE seeds = the 3 graded + 3 extra (3, 4, 5)', '',
                  '| arm | seeds | primary mean AUROC | sd over seeds | delta vs A0 (6 seeds) | se of delta | '
                  'AP ratio oo / og / oe / appr / hud |', '|---|---|---|---|---|---|---|',
                  f'| A0 | {len(m0)} | {np.mean(m0):.4f} | {np.std(m0):.4f} | - | - | - |']
            for arm in X:
                if arm == 'A0' or arm not in R:
                    continue
                Dall = {**R[arm], **X[arm]}
                m = seed_means(Dall)
                se = float(np.sqrt(np.var(m, ddof=1) / len(m) + np.var(m0, ddof=1) / len(m0)))
                apr = [ap_mean(Dall, b) / ap_mean(A0all, b) for b in PRIMARY]
                L.append(f'| {arm} | {len(m)} | {np.mean(m):.4f} | {np.std(m):.4f} | {np.mean(m) - np.mean(m0):+.4f} | '
                         f"{se:.4f} | {' / '.join(f'{a:.2f}' for a in apr)} |")
            L.append('')
    # pose reference
    pf = OUT / 'pose' / 'pose.json'
    if pf.exists():
        P = json.loads(pf.read_text())
        L += ['## Pose reference (keypoints, label-free scalar scored as one neuron; analysis only): AUROC / AP / top-1%',
              '', '| label | -min nose-nose | -min nose-tailbase |', '|---|---|---|']
        for b in BEHAVIOURS:
            f_ = lambda n: '{auroc:.3f} / {ap:.4f} / {top1:.3f}'.format(**P['scores'][n][b])  # noqa: E731
            L.append(f'| {SHORT[b]} | {f_("neg_min_nose_nose")} | {f_("neg_min_nose_tailbase")} |')
        L.append('')
    qf = OUT / 'masks' / 'mask_qa.json'
    if qf.exists():
        Q = json.loads(qf.read_text())
        L += ['## Mask QA vs keypoints (weighted over eval frames)', '', '```', json.dumps(Q, indent=1), '```', '']
    (OUT / 'table.md').write_text('\n'.join(L) + '\n')
    (OUT / 'summary.json').write_text(json.dumps({'arms': S, 'verdict': V, 'sniffing': sn,
                                                  'ceiling': {f'{b}/{p}': v for (b, p), v in C.items()}},
                                                 indent=1, default=float))
    print('\n'.join(L))


if __name__ == '__main__':
    main()
