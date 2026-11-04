"""ECI v2.4 summary tables: OUT24/{mice,ants}/report.json + train jsons + pool jsons -> OUT24/summary.md, OUT24/summary.json."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
REPO = Path(__file__).resolve().parents[3]
OUT = REPO / 'results/vision/eci_v2_4'
ARMS = ('P2W', 'P4W', 'P2N', 'P2C')
NAME = {'nose_nose': 'nose-nose (nose_nose)', 'nt': 'nose-tail (nt)', 'groom_any': 'grooming-to-focal (any)'}


def f(x, d=3):
    return f"{x['mean']:.{d}f} [{x['ci'][0]:.{d}f}, {x['ci'][1]:.{d}f}]"


def df(x, d=3):
    return f"{x['diff']:+.{d}f} [{x['ci'][0]:+.{d}f}, {x['ci'][1]:+.{d}f}]"


def main():
    md = ['# ECI v2.4: SAEs on average-pooled DINOv3 tokens with weighted motion', '',
          'Pre-registration: scripts/eci/v2/v24_prereg.md. Numbers: mean over seeds [95% CI, bootstrap 1,000 replicates: mice = '
          'held-out pools + seeds, ants = held-out videos within series x treatment + seeds]. Differences: mean over seeds, CI of '
          'the replicate differences. Per-video time: mean over held-out videos +- SE (mean over seeds).', '']
    # pooling / training
    md += ['## Pooled tokens, alpha, training', '', '| domain | P | pooled tokens | native M_t tokens per pooled token | alpha | E||x - mean||^2 | E||Delta||^2 | passes | steps |', '|---|---|---|---|---|---|---|---|---|']
    for dom in ('mice', 'ants'):
        pj = json.loads((OUT / f'{dom}_pool.json').read_text())
        for P, r in pj['P'].items():
            md.append(f"| {dom} | {P}x{P} | {r['pooled_tokens']:,} | {r['native_per_pooled']:.2f} | {r['alpha']:.4f} | {r['E_x_minus_mean_sq']:.1f} | "
                      f"{r['E_d_sq']:.1f} | {r['passes']} | {r['steps']:,} |")
    md += ['', f"Lag L = {pj['lag_frames']} frames ({pj['lag_seconds']} s).", '',
           '| domain | model | eval FVE | L0 | dead latents at end (frac) | never fire on eval sample (frac) | steps | passes | GPU | train s |', '|---|---|---|---|---|---|---|---|---|---|']
    for dom in ('mice', 'ants'):
        for a in ARMS:
            for s in (2401, 2402, 2403):
                p = OUT / 'train' / f'{dom}_{a}_s{s}.json'
                if not p.exists():
                    md.append(f'| {dom} | {a}_s{s} | MISSING | | | | | | | |')
                    continue
                r = json.loads(p.read_text())
                md.append(f"| {dom} | {a}_s{s} | {r['eval_fve']:.4f} | {r['eval_l0']:.2f} | {r['dead_frac_at_end_train']:.4f} | "
                          f"{r['never_fire_on_eval_frac']:.4f} | {r['steps_run']:,} | {r['passes']:.2f} | {r['gpu']} | {r['train_time_s']:.0f} |")
    md.append('')
    out = {}
    m = json.loads((OUT / 'mice' / 'report.json').read_text())
    out['mice_criteria'] = m['criteria']
    md += ['## Mice (ext frames, select on SG 11 pools, measure on non-SG 12 pools)', '']
    for b in ('nose_nose', 'nt'):
        c = m['counts'][b]
        md += [f'### {NAME[b]}', '', f"Held-out: {c['held_out_frames']:,} frames, {c['held_out_videos']} videos, {c['positive_frames']:,} positive "
               f"frames; prevalence {c['prevalence']:.4f} (per video {c['per_video_prevalence_mean_se'][0]:.4f} +- {c['per_video_prevalence_mean_se'][1]:.4f}). "
               f"precision@top3/video: {c['top3_eligible_videos']} eligible videos, {c['top3_frames_scored_per_seed']} frames per seed.", '',
               '| arm \\| rule | distinct picks | AUROC | precision@top3/video | chance (top3) | precision@100 | discrimination AUROC nt vs nose_nose | F1 | Pearson r per-video time | raw time bias, s | raw time MAE, s | calibrated time MAE, s |',
               '|' + '---|' * 12]
        for an, L in m['levels'].items():
            v = m['video'][an][b]
            dd = m['distinct'][an]
            md.append(f"| {an} | {dd[0]}/{dd[1]} | {f(L[b + '|auc'])} | {f(L[b + '|top3'])} | {f(L[b + '|chance'], 4)} | {f(L[b + '|p100'])} | "
                      f"{f(L[b + '|disc'], 2)} | {f(L[b + '|F1'])} | {f(L[b + '|r_soft'], 2)} | {v['raw_bias'][0]:.1f} +- {v['raw_bias'][1]:.1f} | "
                      f"{v['raw_mae'][0]:.1f} +- {v['raw_mae'][1]:.1f} | {v['cal_mae'][0]:.1f} +- {v['cal_mae'][1]:.1f} |")
        md.append('')
    md += ['### Mice contrasts', '', '| contrast | nose-nose AUROC | nose-nose precision@top3/video | nose-tail AUROC | nose-tail precision@top3/video | '
           'nose-tail discrimination AUROC | nose-nose Pearson r time | nose-tail Pearson r time |', '|' + '---|' * 8]
    for cn, r in m['contrasts'].items():
        md.append(f"| {cn} | {df(r['nose_nose|auc'])} | {df(r['nose_nose|top3'])} | {df(r['nt|auc'])} | {df(r['nt|top3'])} | {df(r['nt|disc'], 2)} | "
                  f"{df(r['nose_nose|r_soft'], 2)} | {df(r['nt|r_soft'], 2)} |")
    md.append('')
    a = json.loads((OUT / 'ants' / 'report.json').read_text())
    out['ants_B4'] = a['B4']
    for dn, R in a['dirs'].items():
        md += [f"## Ants grooming-to-focal (any), {'select A, measure B (PRIMARY)' if dn == 'A2B' else 'select B, measure A'}", '',
               f"Prevalence {R['prevalence']:.4f}; precision@top3/video: {R['top3_counts']['eligible']} of {R['top3_counts']['held_out_videos']} held-out videos eligible.", '',
               '| arm | AUROC | precision@top3/video | chance (top3) | precision@100 | F1 | Pearson r per-video time | raw time MAE, s | calibrated time MAE, s | picks |', '|' + '---|' * 10]
        for an, L in R['levels'].items():
            v = R['video'][an]
            md.append(f"| {an} | {f(L['groom_any|auc'])} | {f(L['groom_any|top3'])} | {f(L['groom_any|chance'], 4)} | {f(L['groom_any|p100'])} | "
                      f"{f(L['groom_any|F1'])} | {f(L['groom_any|r_soft'], 2)} | {v['raw_mae'][0]:.1f} +- {v['raw_mae'][1]:.1f} | {v['cal_mae'][0]:.1f} +- {v['cal_mae'][1]:.1f} | "
                      f"{', '.join('#' + str(j) for j in R['picks'][an].values())} |")
        md += ['', '| contrast | AUROC | precision@top3/video | Pearson r time | F1 |', '|---|---|---|---|---|']
        for cn, r in R['contrasts'].items():
            md.append(f"| {cn} | {df(r['groom_any|auc'])} | {df(r['groom_any|top3'])} | {df(r['groom_any|r_soft'], 2)} | {df(r['groom_any|F1'])} |")
        md.append('')
    ver = {}
    for arm in ARMS:
        c = dict(m['criteria'].get(arm, {}))
        c.update({k: v for k, v in a['B4'].get(arm, {}).items()})
        c['ADOPT'] = bool(c.get('B1') and c.get('B2') and c.get('B3') and c.get('B4'))
        ver[arm] = c
    out['verdict'] = ver
    md = md[:3] + ['## Verdict (B1-B4 vs V|auc)', '', '```', json.dumps(ver, indent=1, default=str), '```', ''] + md[3:]
    (OUT / 'summary.md').write_text('\n'.join(md) + '\n')
    (OUT / 'summary.json').write_text(json.dumps(out, indent=1, default=str))
    print('\n'.join(md[:12]))


if __name__ == '__main__':
    main()
