"""
ECI v2.3 summary (pre-registration scripts/eci/v2/v23_prereg.md): focus tables v1 s0 vs V vs Vall on the ext frames
(864,000), the A1-A4 verdict, per-video time stats (mean over held-out videos +- SE, mean over seeds).
Inputs: results/vision/eci_v2_3/ext/{report.json, tags/, frames.json} (V22_FRAMES=ext), ext/ants_report.json if present.
-> results/vision/eci_v2_3/ext/summary.md, summary.json   (CPU, seconds)
"""
import json
import pickle
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
OE = REPO / 'results/vision/eci_v2_3/ext'
DT = 0.2
NAME = {'nose_nose': 'nose-nose (nose_nose)', 'nt': 'nose-tail (nt)'}
FOCUS = ['v1s0|auc', 'V|auc', 'Vall|auc', 'V|bal', 'Vall|bal', 'V|hyb', 'Vall|hyb']
DSET = {'sg2non': 'design (a), PRIMARY: select on the 11 SG pools, evaluate on the 12 non-SG pools',
        'sgcf': 'design (b): cross-fitted within the 11 SG pools (12 directions)'}


def video_stats(tags, rule, dset, b):
    """per-video time over the held-out videos of each direction: mean +- SE over videos, then mean over seeds."""
    acc = {}
    for tg in tags:
        per = {}
        for d in tg['sets'][f'{dset}|{rule}']:
            V = d[b]['video']
            S, T, n = V['S'], V['T'], V['n']
            for nm, x in (('raw_bias', (S - T) * DT), ('raw_mae', np.abs(S - T) * DT), ('cal_bias', (d[b]['c'] * S - T) * DT),
                          ('cal_mae', np.abs(d[b]['c'] * S - T) * DT), ('const_mae', np.abs(d[b]['const_rate'] * n - T) * DT),
                          ('true', T * DT), ('mae_frac', np.abs(S - T) / n), ('cal_mae_frac', np.abs(d[b]['c'] * S - T) / n)):
                per.setdefault(nm, []).append(x)
        for nm, xs in per.items():
            x = np.concatenate(xs)
            acc.setdefault(nm, []).append((x.mean(), x.std(ddof=1) / np.sqrt(len(x))))
    return {nm: (float(np.mean([m for m, _ in v])), float(np.mean([s for _, s in v]))) for nm, v in acc.items()}


def main():
    R = json.loads((OE / 'report.json').read_text())
    arms = R['arms']
    tags = {}
    out = {'verdict': {}, 'focus': {}}
    md = ['# v2.3: V retrained on all M_t tokens (Vall) vs V vs v1 s0, mice ext frames (864,000)', '',
          'Numbers: mean over seeds [95% CI, bootstrap over held-out pools and seeds, 1,000 replicates]. Per-video time: mean '
          'over held-out videos +- SE (mean over seeds). V = seeds 1001-1006, Vall = 2001-2003, v1s0 = deployed v1 seed 0. '
          'Rule auc = best AUROC on the selection frames (primary); bal = best balanced AUROC; hyb = nose-nose by auc, '
          'nose-tail by bal (post hoc in v2.2).', '']
    for dset, title in DSET.items():
        S = R['sets'][dset]
        md += [f'## {title}', '']
        gt = R['ground_truth']['nonsg' if dset == 'sg2non' else 'sg']
        for b in ('nose_nose', 'nt'):
            g = gt[b]
            md += [f'### {NAME[b]}', '', f"Held-out frames: {gt['frames']:,}; positive {g['frames_on']:,} frames, {g['bouts']:,} bouts, "
                   f"{g['videos_with_any']} videos; prevalence {g['rate']:.4f} (per video {g['per_video_mean']:.4f} +- {g['per_video_se']:.4f}).", '']
            hdr = ('| arm | distinct picks (seeds) | AUROC | precision@100 | F1 (P(on) > 0.5) | discrimination AUROC nt vs nose_nose | '
                   'raw time bias, s | raw time MAE, s | calibrated time bias, s | calibrated time MAE, s (frac) | constant-rate MAE, s | Pearson r per-video time |')
            md += [hdr, '|' + '---|' * 12]
            for arm in FOCUS:
                if arm not in S['levels']:
                    continue
                ci = S['level_ci'][arm]
                f = lambda k, d=3: f"{ci[f'{b}|{k}']['mean']:.{d}f} [{ci[f'{b}|{k}']['ci'][0]:.{d}f}, {ci[f'{b}|{k}']['ci'][1]:.{d}f}]"  # noqa: E731
                ms = arms[arm]['models']
                for k in ms:
                    if k not in tags:
                        tags[k] = pickle.load(open(OE / 'tags' / f'{k}.pkl', 'rb'))
                vs = video_stats([tags[k] for k in ms], arms[arm]['rule'], dset, b)
                pk = S['picks'][arm]
                dist = [all(p['nose_nose'] != p['nt'] for p in v) for v in pk.values()]
                pm = lambda nm, d=1: f'{vs[nm][0]:.{d}f} +- {vs[nm][1]:.{d}f}'  # noqa: E731
                md.append(f"| {arm} | {sum(dist)}/{len(dist)} | {f('auc')} | {f('p100')} | {f('F1')} | {f('disc', 2)} | {pm('raw_bias')} | "
                          f"{pm('raw_mae')} | {pm('cal_bias')} | {pm('cal_mae')} ({vs['cal_mae_frac'][0]:.4f}) | {pm('const_mae')} | {f('r_soft', 2)} |")
                out['focus'][f'{dset}|{b}|{arm}'] = {'levels': {k: ci[f'{b}|{k}'] for k in ('auc', 'p100', 'F1', 'disc', 'r_soft')},
                                                     'video': vs, 'distinct': f'{sum(dist)}/{len(dist)}'}
            md.append('')
        md += [f'### Contrasts, {title}', '', '| contrast | ' + ' | '.join(f'{NAME[b]} {m}' for b in ('nose_nose', 'nt')
                                                                          for m in ('AUROC', 'precision@100', 'discrimination AUROC')) + ' |',
               '|---|' + '---|' * 6]
        for cn, r in S['contrasts'].items():
            md.append(f'| {cn} | ' + ' | '.join(f"{r[f'{b}|{k}']['diff']:+.3f} [{r[f'{b}|{k}']['ci'][0]:+.3f}, {r[f'{b}|{k}']['ci'][1]:+.3f}]"
                                                for b in ('nose_nose', 'nt') for k in ('auc', 'p100', 'disc')) + ' |')
        md += ['', f'### Selected neurons (nose-nose / nose-tail), {title}', '']
        for arm in FOCUS:
            if arm in S['picks']:
                md.append(f'- {arm}: ' + '; '.join(f"{k}: " + ', '.join(sorted({f"#{p['nose_nose']}/#{p['nt']}" for p in v})) for k, v in S['picks'][arm].items()))
        md.append('')
    # verdict A1-A3 (design (a), rule auc), A4 ants
    S = R['sets']['sg2non']
    con = S['contrasts']['Vall|auc - V|auc']
    pk = S['picks']['Vall|auc']
    nd = sum(all(p['nose_nose'] != p['nt'] for p in v) for v in pk.values())
    v = {'A1 nose-nose AUROC lower CI': con['nose_nose|auc']['ci'][0], 'A2 nose-tail AUROC lower CI': con['nt|auc']['ci'][0],
         'A3 distinct seeds': f'{nd}/{len(pk)}'}
    v['A1'] = v['A1 nose-nose AUROC lower CI'] >= -0.02
    v['A2'] = v['A2 nose-tail AUROC lower CI'] >= -0.02
    v['A3'] = nd >= 2
    ap = OE / 'ants_report.json'
    if ap.exists():
        A = json.loads(ap.read_text())
        c = A['contrasts'].get('Vall - V')
        if c:
            v['A4 ants AUROC lower CI'] = c['AUROC']['ci'][0]
            v['A4 ants Pearson r lower CI'] = c['pearson_r_time']['ci'][0]
            v['A4'] = bool(c['C4'])
    v['ADOPT'] = bool(v['A1'] and v['A2'] and v['A3'] and v.get('A4', False)) if 'A4' in v else 'mice criteria only: ' + (
        'all met' if v['A1'] and v['A2'] and v['A3'] else 'NOT met')
    out['verdict'] = v
    md = md[:2] + ['## Verdict (pre-registered A1-A4; design (a), rule auc, Vall - V)', '', '```', json.dumps(v, indent=1), '```', ''] + md[2:]
    (OE / 'summary.md').write_text('\n'.join(md) + '\n')
    (OE / 'summary.json').write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(v, indent=1, default=float))


if __name__ == '__main__':
    main()
