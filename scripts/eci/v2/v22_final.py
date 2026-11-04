"""v2.2 compact summary table (login-node light: reads JSONs only) -> results/vision/eci_v2_2/summary_table.md"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
O = REPO / 'results/vision/eci_v2_2'
R = json.loads((O / 'report.json').read_text())
EXT = json.loads((REPO / 'results/vision/eci_v2_1/extended/report.json').read_text())
ANTS = json.loads((O / 'ants_report.json').read_text()) if (O / 'ants_report.json').exists() else None
ARMS = ['v1|auc', 'V|auc', 'V|bal', 'V@top10|bal', 'V@box3|bal', 'V@coarse|bal', 'V@ms|bal', 'VP2|bal', 'VMS|bal', 'VW8|bal',
        'V|hyb', 'VW8|hyb', 'VMS|auc', 'VW8|auc']
NAME = {'nose_nose': 'nose-nose (nose_nose)', 'nt': 'nose-tail (nt)'}


def f(lv, k, d=3):
    x = lv[k]
    return f"{x['median']:.{d}f} [{x['min']:.{d}f}, {x['max']:.{d}f}]"


def c(con, k, d=3):
    if con is None:
        return '-'
    x = con[k]
    return f"{x['diff']:+.{d}f} [{x['ci'][0]:+.{d}f}, {x['ci'][1]:+.{d}f}]"


md = []
for dset, title in (('sg2non', '(a) select on SG (11 pools), evaluate on the 12 non-SG pools'),
                    ('sgcf', '(b) cross-fitted within the 11 SG pools, 12 directions')):
    S = R['sets'][dset]
    gt = R['ground_truth']['nonsg' if dset == 'sg2non' else 'sg']
    md += [f'## Mice, design {title}', '']
    for b in ('nose_nose', 'nt'):
        g = gt[b]
        md += [f"### {NAME[b]}: held-out prevalence {g['rate']:.4f} of frames ({g['frames_on']} frames, {g['bouts']} bouts, "
               f"{g['videos_with_any']}/{gt['videos']} videos); per-video rate {g['per_video_mean']:.4f} ± {g['per_video_se']:.4f} (mean ± SE); "
               f"true time per 120-s video {g['per_video_mean'] * 120:.2f} s", '',
               '| arm (rule) | distinct picks (seeds) | AUROC | AUROC - V|auc [95% CI] | precision@100 | F1 (P(on) > 0.5) | discrimination AUROC nt vs nose_nose | '
               'raw time bias, s | raw time MAE, s (frac) | calibrated time bias, s | calibrated time MAE, s (frac) | Pearson r, per-video time |',
               '|---|---|---|---|---|---|---|---|---|---|---|---|']
        for a in ARMS:
            if a not in S['levels']:
                continue
            lv = S['levels'][a]
            pk = S['picks'][a]
            dist = sum(all(p['nose_nose'] != p['nt'] for p in v) for v in pk.values())
            con = S['contrasts'].get(f'{a} - V|auc')
            ph = ' (post hoc)' if R['arms'][a].get('post_hoc') else ''
            md.append(f"| {a}{ph} | {dist}/{len(pk)} | {f(lv, b + '|auc')} | {c(con, b + '|auc')} | {f(lv, b + '|p100')} | {f(lv, b + '|F1')} | "
                      f"{f(lv, b + '|disc', 2)} | {f(lv, b + '|bias_raw_soft_s', 1)} | {f(lv, b + '|mae_raw_soft_s', 1)} ({lv[b + '|mae_raw_soft_frac']['median']:.3f}) | "
                      f"{f(lv, b + '|bias_cal_soft_s', 2)} | {f(lv, b + '|mae_cal_soft_s', 2)} ({lv[b + '|mae_cal_soft_frac']['median']:.4f}) | {f(lv, b + '|r_soft', 2)} |")
        cr = S['levels']['V|auc'][b + '|mae_const_frac']['median']
        md += ['', f'Constant-rate reference (no neuron), calibrated-MAE scale: {cr:.4f} of the video ({cr * 120:.2f} s).', '']
    md += ['Differences v2 - v1 (design ' + dset + '): ' + '; '.join(f"{NAME[b]} AUROC {c(S['contrasts']['V|auc - v1|auc'], b + '|auc')}, "
                                                                    f"precision@100 {c(S['contrasts']['V|auc - v1|auc'], b + '|p100')}"
                                                                    for b in ('nose_nose', 'nt')), '']
# ants
E = EXT['ants']['cells']['pooled|groom_any']
md += ['## Ants, grooming-to-focal (any) (v21 extended pipeline and rule; pooled v2, v3, v4, v4_infested; 212 videos)', '',
       f"Prevalence {E['base_rate']:.4f} of frames; per-video rate {E['rate_per_video_mean']:.4f} ± {E['rate_per_video_se']:.4f}; "
       f"true time per 600-s video {E['true_time_per_video_s_mean']:.1f} ± {E['true_time_per_video_s_se']:.1f} s.", '',
       '| model | AUROC | precision@100 | F1 | raw time bias, s | raw time MAE, frac | calibrated time MAE, frac (s) | Pearson r per-video time |',
       '|---|---|---|---|---|---|---|---|']
for g in ('v1', 'v2'):
    m = E['models'][g]
    q = lambda k, d=3, s=1.0: f"{m['median'][k] * s:.{d}f} [{m['min'][k] * s:.{d}f}, {m['max'][k] * s:.{d}f}]"  # noqa: E731
    md.append(f"| {'v1' if g == 'v1' else 'V (v2)'} | {q('AUROC')} | {q('p@100')} | {q('F1')} | {q('bias_raw_soft_s', 1)} | {q('mae_raw_soft_frac')} | "
              f"{q('mae_cal_soft_frac')} ({m['median']['mae_cal_soft_s']:.0f} s) | {q('r_soft', 2)} |")
if ANTS:
    md += ['', 'C4 check (v22_ants.py; video bootstrap within series x seeds, 1000 replicates):', '',
           '| arm | AUROC | Pearson r per-video time | AUROC - V [95% CI] | r - V [95% CI] | C4 |', '|---|---|---|---|---|---|']
    for a, lv in ANTS['levels'].items():
        con = ANTS['contrasts'].get(f'{a} - V')
        md.append(f"| {a} | {f(lv, 'AUROC')} | {f(lv, 'pearson_r_time', 3)} | {c(con, 'AUROC') if con else '-'} | "
                  f"{c(con, 'pearson_r_time') if con else '-'} | {('yes' if con['C4'] else 'NO') if con else '-'} |")
(O / 'summary_table.md').write_text('\n'.join(md) + '\n')
print('\n'.join(md[:30]))
