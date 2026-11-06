"""ECI v2.6 decisions (scripts/eci/v2/v26_prereg.md) + tables -> results/vision/eci_v2_6/{summary.json, tables.md}."""
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / 'results/vision/eci_v2_6'
NEW = ('N0wc', 'PRED', 'SMOOTH', 'RW', 'N0x')
NAME = {'N0wc': 'N0w-c'}


def f3(lv, d=3):
    return f"{lv['mean']:.{d}f} [{lv['ci'][0]:.{d}f}, {lv['ci'][1]:.{d}f}]"


def c3(c, d=3):
    return f"{c['diff']:+.{d}f} [{c['ci'][0]:+.{d}f}, {c['ci'][1]:+.{d}f}]"


def main():
    M = json.loads((OUT / 'mice' / 'report.json').read_text())
    A = json.loads((OUT / 'ants' / 'report.json').read_text())
    a2b = A['dirs']['A2B']
    MC, AC = M['contrasts'], a2b['contrasts']
    lo = lambda c: c['ci'][0]  # noqa: E731
    S = {'confirm_N0w': {}, 'B': {}, 'gain': {}, 'noninf_vs_V': {}}
    for r in ('dist', 'auc'):
        c, ca = MC[f'N0wc|{r} - V|auc'], AC['N0wc - V']
        d = M['distinct'][f'N0wc|{r}']
        S['confirm_N0w'][r] = {'nt top3 diff': c['nt|top3']['diff'], 'nt top3 LCI': lo(c['nt|top3']), 'nn AUROC LCI': lo(c['nose_nose|auc']),
                               'ants AUROC LCI': lo(ca['groom_any|auc']), 'distinct': f'{d[0]}/{d[1]}'}
        S['confirm_N0w'][r]['pass'] = bool(lo(c['nt|top3']) > 0 and lo(c['nose_nose|auc']) >= -0.02 and lo(ca['groom_any|auc']) >= -0.02 and d[0] == d[1] == 3)
    for x in NEW[1:]:
        if f'{x}|dist - N0wc|dist' not in MC or f'{x} - N0wc' not in AC:
            continue
        ca, cav = AC[f'{x} - N0wc'], AC[f'{x} - V']
        for r in ('dist', 'auc'):
            cm, cmv = MC[f'{x}|{r} - N0wc|{r}'], MC[f'{x}|{r} - V|auc']
            mets = {'mice nose-tail AUROC': cm['nt|auc'], 'mice nose-tail top3': cm['nt|top3'], 'ants AUROC': ca['groom_any|auc'], 'ants top3': ca['groom_any|top3'],
                    'mice nose-nose AUROC': cm['nose_nose|auc'], 'mice nose-nose top3': cm['nose_nose|top3']}
            ni = lo(cm['nose_nose|auc']) >= -0.02 and lo(ca['groom_any|auc']) >= -0.02
            niV = lo(cmv['nose_nose|auc']) >= -0.02 and lo(cav['groom_any|auc']) >= -0.02
            rec = {'contrasts_vs_N0wc': {m: [c['diff'], c['ci'][0], c['ci'][1]] for m, c in mets.items()}, 'noninf_vs_N0wc': ni, 'noninf_vs_V': niV,
                   'distinct': M['distinct'][f'{x}|{r}']}
            if x in ('PRED', 'SMOOTH'):
                win = [m for m in ('mice nose-tail AUROC', 'mice nose-tail top3', 'ants AUROC', 'ants top3') if lo(mets[m]) > 0]
                rec.update(winning=win, pass_=bool(win and ni and niV))
                S['B'][f'{x}|{r}'] = rec
            else:
                win = [m for m in ('mice nose-tail top3', 'mice nose-nose AUROC') if lo(mets[m]) > 0]
                rec.update(winning=win, pass_=bool(win and ni))
                S['gain'][f'{x}|{r}'] = rec
    tb = [(k, v) for k, v in S['B'].items() if v['pass_']]
    tg = [(k, v) for k, v in S['gain'].items() if v['pass_']]
    nt = lambda k: M['levels'][k]['nt|top3']['mean']  # noqa: E731
    if tb:
        S['deploy'] = max(tb, key=lambda kv: nt(kv[0]))[0]
    elif tg:
        S['deploy'] = max(tg, key=lambda kv: nt(kv[0]))[0]
    elif S['confirm_N0w']['dist']['pass']:
        S['deploy'] = 'N0wc|dist'
    else:
        S['deploy'] = None
    S['best_temporal'] = max([k for k in S['B'] if k.endswith('|dist')], key=nt)
    tr = {}
    for p in sorted((OUT / 'train').glob('*_s26*.json')):
        j = json.loads(p.read_text())
        tr[p.stem] = {'fve': j['eval_fve'], 'dead': j['dead_at_end_train'], 'never_fire': j['never_fire_on_eval_frac'], 'steps': j['steps_run'], 'passes': j['passes']}
    S['train'] = tr
    (OUT / 'summary.json').write_text(json.dumps(S, indent=1, default=float))
    # tables
    md = ['# ECI v2.6 tables (scripts/eci/v2/v26_summary.py)', '', '## Mice, held-out non-SG pools (edge-excluded ext frames)', '',
          '| model rule | nose-nose AUROC | nose-nose precision@top3/video (chance) | nose-tail AUROC | nose-tail precision@top3/video (chance) | distinct | nose-tail discrimination AUROC | nose-nose per-video MAE, s | nose-tail per-video MAE, s |',
          '|' + '---|' * 9]
    for k, lv in M['levels'].items():
        if k.endswith('dist2048'):
            continue
        vd = M['video'][k]
        md.append(f"| {k} | {f3(lv['nose_nose|auc'])} | {f3(lv['nose_nose|top3'])} ({lv['nose_nose|chance']['mean']:.3f}) | {f3(lv['nt|auc'])} | {f3(lv['nt|top3'])} "
                  f"({lv['nt|chance']['mean']:.3f}) | {M['distinct'][k][0]}/{M['distinct'][k][1]} | {lv['nt|disc']['mean']:.3f} | "
                  f"{vd['nose_nose']['raw_mae'][0]:.1f} +- {vd['nose_nose']['raw_mae'][1]:.1f} | {vd['nt']['raw_mae'][0]:.1f} +- {vd['nt']['raw_mae'][1]:.1f} |")
    md += ['', '### Mice contrasts', '', '| contrast | nose-nose AUROC | nose-nose precision@top3/video | nose-tail AUROC | nose-tail precision@top3/video | paired |', '|---|---|---|---|---|---|']
    for k, c in MC.items():
        if 'dist2048' in k or not any(k.startswith(x + '|') for x in NEW):
            continue
        md.append(f"| {k} | {c3(c['nose_nose|auc'])} | {c3(c['nose_nose|top3'])} | {c3(c['nt|auc'])} | {c3(c['nt|top3'])} | {c['nt|auc']['paired']} |")
    for dn, R in A['dirs'].items():
        md += ['', f'## Ants grooming-to-focal (any), {dn}', '', '| model | AUROC | precision@top3/video (chance) | per-video MAE, s | Pearson r time |', '|---|---|---|---|---|']
        for k, lv in R['levels'].items():
            vd = R['video'][k]
            md.append(f"| {k} | {f3(lv['groom_any|auc'])} | {f3(lv['groom_any|top3'])} ({lv['groom_any|chance']['mean']:.3f}) | {vd['raw_mae'][0]:.1f} +- {vd['raw_mae'][1]:.1f} | {lv['groom_any|r_soft']['mean']:.2f} |")
        md += ['', '| contrast | AUROC | precision@top3/video | paired |', '|---|---|---|---|']
        for k, c in R['contrasts'].items():
            if any(k.startswith(x + ' ') for x in NEW):
                md.append(f"| {k} | {c3(c['groom_any|auc'])} | {c3(c['groom_any|top3'])} | {c['groom_any|auc']['paired']} |")
    vf = OUT / 'v2_vs_v1' / 'mice_atlas.json'
    if vf.exists():
        V1 = json.loads(vf.read_text())
        md += ['', '## v2 vs v1 (mice: the atlas subset, all models rescored on it; ' + V1['frame_set'] + ')', '',
               '| model | nose-nose AUROC | nose-nose precision@top3/video | nose-tail AUROC | nose-tail precision@top3/video |', '|---|---|---|---|---|']
        for n, lv in V1['levels'].items():
            md.append(f"| {n} | {f3(lv['nose_nose|auc'])} | {f3(lv['nose_nose|top3'])} | {f3(lv['nt|auc'])} | {f3(lv['nt|top3'])} |")
        md += ['', '| contrast | nose-nose AUROC | nose-nose precision@top3/video | nose-tail AUROC | nose-tail precision@top3/video |', '|---|---|---|---|---|']
        for k, c in V1['contrasts'].items():
            md.append(f"| {k} | {c3(c['nose_nose|auc'])} | {c3(c['nose_nose|top3'])} | {c3(c['nt|auc'])} | {c3(c['nt|top3'])} |")
    md += ['', '## Decisions', '', '```', json.dumps({k: v for k, v in S.items() if k != 'train'}, indent=1, default=float), '```', '',
           '## SAE health', '', '| key | eval FVE | dead at end | never fire (eval) | steps | passes |', '|---|---|---|---|---|---|']
    for k, t in tr.items():
        md.append(f"| {k} | {t['fve']:.4f} | {t['dead']} | {t['never_fire']:.4f} | {t['steps']:,} | {t['passes']:.2f} |")
    (OUT / 'tables.md').write_text('\n'.join(md) + '\n')
    print(json.dumps({k: v for k, v in S.items() if k != 'train'}, indent=1, default=float))


if __name__ == '__main__':
    main()
