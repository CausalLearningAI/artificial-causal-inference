"""
CalMS21 swap knob sweep (eci_knobs.py, rules in its docstring): per-arm table (mean +- sd over SAE seeds) and the
pre-registered contrasts vs B0 (Welch 95% CI). Reads OUT/knobs/eval/*.json -> OUT/knobs/knobs.{md,json}. Light (CPU,
seconds). Run with CALMS_OUT=results/vision/benchmark_calms21/swap.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calms_common import OUT  # noqa: E402

ARMS = ['B0', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']
WHAT = {'B0': 'plain 4096 k16 6k (= S2)', 'B1': 'Matryoshka 512 k16 6k', 'B2': 'Matryoshka 2048 k16 30k',
        'B3': 'Matryoshka 8192 k16 30k', 'B4': 'Matryoshka 2048 k8 30k', 'B5': 'Matryoshka 2048 k32 30k',
        'B6': 'Matryoshka 2048 k16 30k, centred', 'B7': 'plain 2048 k16 30k'}
COLS = [('sf AUROC', lambda r: r['scores']['sniff_face']['cf_auroc']),
        ('sf AP', lambda r: r['scores']['sniff_face']['cf_ap']),
        ('sf AP lift', lambda r: r['scores']['sniff_face']['ap_lift']),
        ('sf honest top-1%', lambda r: r['scores']['sniff_face']['cf_top1']),
        ('sf R-prec (R-prec pick)', lambda r: r['scores']['sniff_face']['cf_rprec']),
        ('sf R-prec (AUROC pick)', lambda r: r['scores']['sniff_face']['cf_rprec_auc_pick']),
        ('inv AUROC', lambda r: r['scores']['investigation']['cf_auroc']),
        ('inv R-prec (R-prec pick)', lambda r: r['scores']['investigation']['cf_rprec']),
        ('inv R-prec (AUROC pick)', lambda r: r['scores']['investigation']['cf_rprec_auc_pick']),
        ('FVE calib', lambda r: r['fve_calib']), ('FVE scoring', lambda r: r['fve_eval']),
        ('dead frac (scoring)', lambda r: r['dead_frac_eval']), ('dead frac (train end)', lambda r: r['dead_frac_train'])]
PRIM = {'P1 sf R-prec (R-prec pick)': 4, 'P2 sf honest top-1%': 3}


def welch(a, b):
    a, b = np.asarray(a), np.asarray(b)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    d = a.mean() - b.mean()
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    h = stats.t.ppf(0.975, df) * se
    return d, d - h, d + h, df


def verdict1(d, lo, hi):
    if d >= 0.010 and lo > 0:
        return 'WIN'
    if d <= -0.010 and hi < 0:
        return 'LOSES'
    return 'EQUIVALENT'


def main():
    R = {}
    for a in ARMS:
        fs = sorted((OUT / 'knobs' / 'eval').glob(f'{a}_s*.json'))
        if fs:
            R[a] = [json.loads(f.read_text()) for f in fs]
    V = {a: np.array([[f(r) for _, f in COLS] for r in rs]) for a, rs in R.items()}
    L = ['# CalMS21 swap: SAE knob sweep for a sniffing neuron (eci_knobs.py)', '',
         'SAE pool = 19 task1 TEST videos (labels unread); scoring = 70 task1 TRAIN videos, video-level cross-fitting '
         '(10 split seeds x 2 directions). D3L-64 tokens, z_t only, mask as stored, frame value = max over kept tokens. '
         'sf = sniff_face (its own sequences, weighted base rate '
         f"{R['B0'][0]['scores']['sniff_face']['weighted_base_rate']:.4f}); inv = investigation (all frames, base rate "
         f"{R['B0'][0]['scores']['investigation']['weighted_base_rate']:.4f}). Mean ± sd over SAE seeds 600-602.", '',
         '| arm | what | seeds | ' + ' | '.join(c for c, _ in COLS) + ' |', '|' + '---|' * (len(COLS) + 3)]
    for a, v in V.items():
        L.append(f'| {a} | {WHAT[a]} | {len(v)} | ' + ' | '.join(
            f'{m:.3f} ± {s:.3f}' if j != 1 else f'{m:.4f} ± {s:.4f}'
            for j, (m, s) in enumerate(zip(v.mean(0), v.std(0, ddof=1) if len(v) > 1 else 0 * v[0]))) + ' |')
    L += ['', '## contrasts vs B0 (pre-registered; Welch 95% CI over 3 vs 3 seeds = a WEAK test)', '',
          '| arm | ' + ' | '.join(f'delta {p} [95% CI]' for p in PRIM) + ' | verdict |', '|---|---|---|---|']
    out = {'table': {a: {c: v[:, j].tolist() for j, (c, _) in enumerate(COLS)} for a, v in V.items()}, 'contrasts': {}}
    for a in V:
        if a == 'B0' or len(V[a]) < 2:
            continue
        vs, cells = [], []
        for p, j in PRIM.items():
            d, lo, hi, df = welch(V[a][:, j], V['B0'][:, j])
            v1 = verdict1(d, lo, hi)
            vs.append(v1)
            cells.append(f'{d:+.3f} [{lo:+.3f}, {hi:+.3f}] df {df:.1f} {v1}')
            out['contrasts'].setdefault(a, {})[p] = dict(delta=d, lo=lo, hi=hi, df=df, verdict=v1)
        if 'WIN' in vs and 'LOSES' in vs:
            v = 'MIXED'
        elif 'WIN' in vs:
            v = 'BETTER'
        elif 'LOSES' in vs:
            v = 'WORSE'
        else:
            v = 'EQUIVALENT'
        out['contrasts'][a]['verdict'] = v
        L.append(f'| {a} | ' + ' | '.join(cells) + f' | {v} |')
    (OUT / 'knobs' / 'knobs.md').write_text('\n'.join(L) + '\n')
    (OUT / 'knobs' / 'knobs.json').write_text(json.dumps(out, indent=1, default=float))
    print('\n'.join(L))


if __name__ == '__main__':
    main()
