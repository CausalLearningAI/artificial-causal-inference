"""
MABe22 mouse ECI - ENCODER STUDY: scoring under the PRE-REGISTERED rules of eci_encoders.sh (login safe: reads JSONs).

1. Ceiling tables: OUT/encoders/probe/probe_<tag>_L<layer>.json (eci_enc_probe.py) -> per tag x layer, mean_lin and
   mil_ctx AUROC / AP on oral_oral and oral_genital (mean over folds x seeds), ceiling score = mil_ctx AUROC mean over
   the two labels; the V-JEPA config = argmax of the ceiling score over the vj_* tags x layers (rule 2, VJ / VJD).
2. SAE table: OUT/encoders/eval/<run>_s<seed>.json, seeds 300-305, full width m4096; the round-3 default rule
   (eci_r3_score.summ / verdict: Welch 95% CI on 6 vs 6 per-seed primary AUROC, AP guards) against D2.
Outputs: OUT/encoders/encoders.md, OUT/encoders/encoders.json
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eci_common import OUT  # noqa: E402
import eci_r3_score as S3  # noqa: E402

E = OUT / 'encoders'
S3.EV = E / 'eval'
S3.SEEDS = list(range(300, 306))
OO, OG = 'oral_oral_contact', 'oral_genital_contact'
NAMES = {'r3': 'DINOv2-B 448 (round-3 store m448_t)', 'd2_448': 'DINOv2-B 448', 'd3b_512': 'DINOv3-B/16 512',
         'd3l_512': 'DINOv3-L/16 512', 'vj_256': 'V-JEPA 2-L 256 (native RoPE grid)',
         'vj_512x': 'V-JEPA 2-L 512 (RoPE extrapolated)', 'vj_512i': 'V-JEPA 2-L 512 (RoPE interpolated)'}


def ceilings():
    C = {}
    for f in sorted((E / 'probe').glob('probe_*_L*.json')):
        m = re.fullmatch(r'probe_(.+)_L(\d+)\.json', f.name)
        j = json.loads(f.read_text())
        d = {}
        for s in j['summary']:
            d[(s['probe'], s['behaviour'])] = (s['auroc_mean'], s['ap_mean'])
        C.setdefault(m.group(1), {})[int(m.group(2))] = {
            'mean_lin': {b: d[('mean_lin', b)] for b in (OO, OG)}, 'mil_ctx': {b: d[('mil_ctx', b)] for b in (OO, OG)},
            'score': float(np.mean([d[('mil_ctx', b)][0] for b in (OO, OG)])), 'dim': j['dim'], 'grid': j['grid']}
    return C


def main():
    out, L = {}, ['# MABe22 ECI encoder study (pre-registered rules: scripts/benchmark/mabe22/eci_encoders.sh)', '']
    C = ceilings()
    out['ceiling'] = {t: {str(k): v for k, v in d.items()} for t, d in C.items()}
    L += ['## 1. Supervised ceiling per layer (DIAGNOSTIC; eval frames, clip-grouped 5-fold CV, 3 seeds)', '',
          'ceiling score = mil_ctx AUROC mean over oral_oral and oral_genital', '',
          '| encoder | layer | mean_lin AUROC oo / og | mean_lin AP oo / og | mil_ctx AUROC oo / og | mil_ctx AP oo / og '
          '| ceiling score |', '|---|---|---|---|---|---|---|']
    for t in [t for t in NAMES if t in C] + [t for t in C if t not in NAMES]:
        for l_, v in sorted(C[t].items()):
            ml, mc = v['mean_lin'], v['mil_ctx']
            L.append(f"| {NAMES.get(t, t)} | {l_} | {ml[OO][0]:.3f} / {ml[OG][0]:.3f} | {ml[OO][1]:.4f} / {ml[OG][1]:.4f} "
                     f"| {mc[OO][0]:.3f} / {mc[OG][0]:.3f} | {mc[OO][1]:.4f} / {mc[OG][1]:.4f} | {v['score']:.4f} |")
    best = {t: max(d, key=lambda k: d[k]['score']) for t, d in C.items()}
    out['best_layer'] = best
    vj = [(t, l_, C[t][l_]['score']) for t in C if t.startswith('vj_') for l_ in C[t]]
    if vj:
        t, l_, sc = max(vj, key=lambda x: x[2])
        out['vj_config'] = {'tag': t, 'layer': l_, 'score': sc, 'n_cells': len(vj)}
        L += ['', f'V-JEPA config by the pre-registered rule (argmax over {len(vj)} cells): {t} layer {l_} '
              f'(ceiling score {sc:.4f}; label-chosen).']
    L += ['', 'best-ceiling layer per encoder: ' + ', '.join(f'{NAMES.get(t, t)} L{l_}' for t, l_ in best.items()), '']
    runs = sorted({re.sub(r'_s\d+\.json$', '', f.name) for f in S3.EV.glob('*_s*.json')})
    sm = {r: S3.S(r) for r in runs}
    sm = {r: s for r, s in sm.items() if s}
    out['runs'] = {r: s for r, s in sm.items()}
    if 'D2' in sm:
        arms = [(r, sm[r], sm['D2'], '') for r in runs if r != 'D2' and r in sm]
        desc = []
        for r in runs:
            if r.startswith('VJD_') and 'VJ_' + r[4:] in sm and r in sm:
                desc.append((f'{r} vs {"VJ_" + r[4:]} (descriptive)', sm[r], sm['VJ_' + r[4:]], 'descriptive'))
        d3b = [r for r in runs if r.startswith('D3B_') and r.endswith('_L12')]
        d3l = [r for r in runs if r.startswith('D3L_') and r.endswith('_L24')]
        if d3b and d3l:
            desc.append((f'{d3l[0]} vs {d3b[0]} (descriptive)', sm[d3l[0]], sm[d3b[0]], 'descriptive'))
        L += S3.section('2. SAE arms vs D2 (DINOv2-B 448, last layer, no orbit, x = [z_t, z_t - z_t-60])', 'D2',
                        sm['D2'], arms + desc, out, 'sae')
    # ---- addendum 2 (C1-C3; seeds 310-315) and the label-free 5/6-depth layer rule
    C2S = list(range(310, 316))
    sc = lambda r: S3.S(r, None, C2S)  # noqa: E731
    depth = {'d2_448': 12, 'd3b_512': 12, 'd3l_512': 24, 'd2l_448': 24, 'vj_256': 24, 'vj_512x': 24, 'vj_512i': 24}
    L += ['## Label-free layer rule (block round(5 n / 6)) vs the screens', '',
          '| encoder | rule block | its ceiling score | best block | best ceiling score | consistent (<= 0.005) |',
          '|---|---|---|---|---|---|']
    out['layer_rule'] = {}
    for t, n in depth.items():
        if t not in C:
            continue
        rb, bb = int(round(5 * n / 6)), best[t]
        if rb in C[t]:
            ok = C[t][bb]['score'] - C[t][rb]['score'] <= 0.005
            L.append(f"| {NAMES.get(t, t)} | {rb} | {C[t][rb]['score']:.4f} | {bb} | {C[t][bb]['score']:.4f} | {ok} |")
            out['layer_rule'][t] = {'rule': rb, 'best': bb, 'consistent': bool(ok)}
    L.append('')
    d3l20, d2c = sc('D3L_d3l_512_L20'), sc('D2_d2_448_L12')
    if d3l20 and d2c:
        L += S3.section('C1. D3L-L20 confirmation, fresh seeds 310-315 vs fresh D2 (seeds 310-315)', 'D2_d2_448_L12',
                        d2c, [('D3L_d3l_512_L20', d3l20, d2c, '')], out, 'C1')
    if 'd2l_448' in C:
        bl = best['d2l_448']
        arms = sorted({bl, 20})
        for l_ in arms:
            r = sc(f'D2L_d2l_448_L{l_}')
            if r and d3l20:
                L += S3.section(f'C2. D3L-L20 (arm) vs D2L block {l_} (control; {"best-ceiling" if l_ == bl else "block 20"}),'
                                ' seeds 310-315', f'D2L_d2l_448_L{l_}', r,
                                [('D3L_d3l_512_L20', d3l20, r, '')] + ([(f'D2L_d2l_448_L{l_} vs D2 (descriptive)', r,
                                                                       d2c, 'descriptive')] if d2c else []),
                                out, f'C2_L{l_}')
    d3lo = sc('D3LO_d3l_512o_L20')
    if d3lo and d3l20:
        L += S3.section('C3. D3L-L20 with D4 orbit (arm) vs D3L-L20 without (control), seeds 310-315', 'D3L_d3l_512_L20',
                        d3l20, [('D3LO_d3l_512o_L20', d3lo, d3l20, '')], out, 'C3')
    if d3l20:
        vj = [(r, sm[r], d3l20, '') for r in runs if r.startswith(('VJ_', 'VJD_')) and r in sm]
        if vj:
            L += S3.section('V-JEPA arms vs D3L-L20 (seeds 310-315)', 'D3L_d3l_512_L20', d3l20, vj, out, 'VJ_vs_D3L')
    (E / 'encoders.md').write_text('\n'.join(L) + '\n')
    (E / 'encoders.json').write_text(json.dumps(out, indent=1, default=float))
    print('\n'.join(L))


if __name__ == '__main__':
    main()
