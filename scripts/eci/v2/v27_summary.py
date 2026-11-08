"""ECI v2.7 decisions (scripts/eci/v2/v27_prereg.md) + tables -> results/vision/eci_v2_7/{summary.json, tables.md}."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / 'results/vision/eci_v2_7'
NEW = ('PREDc', 'PREDd', 'PREDk', 'PREDh')
EMERGE = ('PREDd', 'PREDk', 'PREDh')
PREF = (128, 256, 512, 2048, 4096)


def level(j):
    return next(i + 1 for i, p in enumerate(PREF) if j < p)


def f3(lv, d=3):
    return f"{lv['mean']:.{d}f} [{lv['ci'][0]:.{d}f}, {lv['ci'][1]:.{d}f}]"


def c3(c, d=3):
    return f"{c['diff']:+.{d}f} [{c['ci'][0]:+.{d}f}, {c['ci'][1]:+.{d}f}]"


def main():
    M = json.loads((OUT / 'mice' / 'report.json').read_text())
    A = json.loads((OUT / 'ants' / 'report.json').read_text())
    MC, AC = M['contrasts'], A['dirs']['A2B']['contrasts']
    lo = lambda c: c['ci'][0]  # noqa: E731
    levels = {k: {key: {b: level(j) for b, j in p.items()} for key, p in v.items()} for k, v in M['picks'].items()}
    S = {'levels': levels, 'confirm_PRED': {}, 'emerge': {}}
    for r in ('dist', 'dist2', 'auc'):
        c, ca = MC[f'PREDc|{r} - V|auc'], AC['PREDc - V']
        d = M['distinct'][f'PREDc|{r}']
        e = {'nt top3 diff': c['nt|top3']['diff'], 'nt top3 LCI': lo(c['nt|top3']), 'nt AUROC LCI': lo(c['nt|auc']), 'nn AUROC LCI': lo(c['nose_nose|auc']),
             'ants AUROC LCI': lo(ca['groom_any|auc']), 'distinct': f'{d[0]}/{d[1]}'}
        e['pass'] = bool(e['nt top3 LCI'] > 0 and e['nt AUROC LCI'] >= -0.02 and e['nn AUROC LCI'] >= -0.02 and e['ants AUROC LCI'] >= -0.02 and d[0] == d[1] == 3)
        S['confirm_PRED'][r] = e
    for x in EMERGE:
        ca = AC.get(f'{x} - PREDc')
        for r in ('dist', 'dist2', 'auc'):
            cm = MC.get(f'{x}|{r} - PREDc|{r}')
            if cm is None or ca is None:
                continue
            ni = lo(cm['nose_nose|auc']) >= -0.02 and lo(cm['nt|auc']) >= -0.02 and lo(ca['groom_any|auc']) >= -0.02 and lo(cm['nt|top3']) >= -0.05
            lx = [levels[f'{x}|{r}'][f'{x}_s{s}']['nt'] for s in (2701, 2702, 2703)]
            lc = [levels[f'PREDc|{r}'][f'PREDc_s{s}']['nt'] for s in (2701, 2702, 2703)]
            earlier = sum(a_ < b_ for a_, b_ in zip(lx, lc))
            gain = lo(cm['nose_nose|top3']) > 0
            S['emerge'][f'{x}|{r}'] = {'noninf': ni, 'nt levels arm / PREDc': [lx, lc], 'earlier seeds': earlier, 'nn top3 gain LCI': lo(cm['nose_nose|top3']),
                                       'replace': bool(ni and (earlier >= 2 or gain)),
                                       'contrasts_vs_PREDc': {'nn AUROC': [cm['nose_nose|auc']['diff'], *cm['nose_nose|auc']['ci']], 'nt AUROC': [cm['nt|auc']['diff'], *cm['nt|auc']['ci']],
                                                              'nt top3': [cm['nt|top3']['diff'], *cm['nt|top3']['ci']], 'nn top3': [cm['nose_nose|top3']['diff'], *cm['nose_nose|top3']['ci']],
                                                              'ants AUROC': [ca['groom_any|auc']['diff'], *ca['groom_any|auc']['ci']], 'ants top3': [ca['groom_any|top3']['diff'], *ca['groom_any|top3']['ci']]}}
    rep = [(k, v) for k, v in S['emerge'].items() if v['replace']]
    nt = lambda k: M['levels'][k]['nt|top3']['mean']  # noqa: E731
    if rep:
        S['deploy'] = max(rep, key=lambda kv: nt(kv[0]))[0]
    elif S['confirm_PRED']['dist2']['pass']:
        S['deploy'] = 'PREDc|dist2'
    elif S['confirm_PRED']['dist']['pass']:
        S['deploy'] = 'PREDc|dist'
    else:
        S['deploy'] = None
    S['best_emergence'] = max([f'{x}|dist' for x in EMERGE if f'{x}|dist' in M['levels']], key=nt)
    S['train'] = {p.stem: {k: json.loads(p.read_text())[k] for k in ('eval_fve', 'dead_at_end_train', 'never_fire_on_eval_frac', 'steps_run', 'passes')}
                  for p in sorted((OUT / 'train').glob('*_s27*.json'))}
    (OUT / 'summary.json').write_text(json.dumps(S, indent=1, default=float))
    md = ['# ECI v2.7 tables (scripts/eci/v2/v27_summary.py)', '', '## Mice, held-out non-SG pools (edge-excluded ext frames)', '',
          '| model rule | nose-nose AUROC | nose-nose precision@top3/video (chance) | nose-tail AUROC | nose-tail precision@top3/video (chance) | distinct | nose-tail discrimination AUROC | nose-nose per-video MAE, s | nose-tail per-video MAE, s | levels nose-nose / nose-tail per seed |',
          '|' + '---|' * 10]
    for k, lv in M['levels'].items():
        if k.endswith('dist2048'):
            continue
        vd = M['video'][k]
        lvs = ' '.join(f"{v['nose_nose']}/{v['nt']}" for v in levels[k].values())
        md.append(f"| {k} | {f3(lv['nose_nose|auc'])} | {f3(lv['nose_nose|top3'])} ({lv['nose_nose|chance']['mean']:.3f}) | {f3(lv['nt|auc'])} | {f3(lv['nt|top3'])} "
                  f"({lv['nt|chance']['mean']:.3f}) | {M['distinct'][k][0]}/{M['distinct'][k][1]} | {lv['nt|disc']['mean']:.3f} | "
                  f"{vd['nose_nose']['raw_mae'][0]:.1f} +- {vd['nose_nose']['raw_mae'][1]:.1f} | {vd['nt']['raw_mae'][0]:.1f} +- {vd['nt']['raw_mae'][1]:.1f} | {lvs} |")
    md += ['', '### Mice contrasts', '', '| contrast | nose-nose AUROC | nose-nose precision@top3/video | nose-tail AUROC | nose-tail precision@top3/video | paired |', '|---|---|---|---|---|---|']
    for k, c in MC.items():
        if 'dist2048' in k or not any(k.startswith(x + '|') for x in NEW + ('PRED26', 'V')):
            continue
        md.append(f"| {k} | {c3(c['nose_nose|auc'])} | {c3(c['nose_nose|top3'])} | {c3(c['nt|auc'])} | {c3(c['nt|top3'])} | {c['nt|auc']['paired']} |")
    for dn, R in A['dirs'].items():
        md += ['', f'## Ants grooming-to-focal (any), {dn}', '', '| model | AUROC | precision@top3/video (chance) | per-video MAE, s | picks |', '|---|---|---|---|---|']
        for k, lv in R['levels'].items():
            vd = R['video'][k]
            pk = list(R['picks'][k].values())
            md.append(f"| {k} | {f3(lv['groom_any|auc'])} | {f3(lv['groom_any|top3'])} ({lv['groom_any|chance']['mean']:.3f}) | {vd['raw_mae'][0]:.1f} +- {vd['raw_mae'][1]:.1f} | "
                      f"{', '.join(f'#{j} (L{level(j)})' for j in pk[:6])} |")
        md += ['', '| contrast | AUROC | precision@top3/video | paired |', '|---|---|---|---|']
        for k, c in R['contrasts'].items():
            if any(k.startswith(x + ' ') for x in NEW):
                md.append(f"| {k} | {c3(c['groom_any|auc'])} | {c3(c['groom_any|top3'])} | {c['groom_any|auc']['paired']} |")
    for p in sorted((OUT / 'levels').glob('*.json')) if (OUT / 'levels').exists() else []:
        Lj = json.loads(p.read_text())
        md += ['', f"## Level curve {Lj['arm']} (held-out AUROC of the best selection-set neuron within levels 1..m, mean over seeds)", '',
               '| behaviour | m=1 (128) | m=2 (256) | m=3 (512) | m=4 (2048) | m=5 (4096) |', '|---|---|---|---|---|---|']
        for b, v in Lj['mean'].items():
            md.append(f"| {b} | " + ' | '.join(f'{x:.3f}' for x in v) + ' |')
    vf = OUT / 'v2_vs_v1' / 'mice_atlas.json'
    if vf.exists():
        V1 = json.loads(vf.read_text())
        md += ['', '## v2 vs v1 (mice atlas subset; ' + V1['frame_set'] + ')', '', '| contrast | nose-nose AUROC | nose-nose precision@top3/video | nose-tail AUROC | nose-tail precision@top3/video |', '|---|---|---|---|---|']
        for k, c in V1['contrasts'].items():
            md.append(f"| {k} | {c3(c['nose_nose|auc'])} | {c3(c['nose_nose|top3'])} | {c3(c['nt|auc'])} | {c3(c['nt|top3'])} |")
    md += ['', '## Decisions', '', '```', json.dumps({k: v for k, v in S.items() if k not in ('train', 'levels')}, indent=1, default=float), '```', '',
           '## SAE health', '', '| key | eval FVE | dead at end | never fire (eval) | steps | passes |', '|---|---|---|---|---|---|']
    for k, t in S['train'].items():
        md.append(f"| {k} | {t['eval_fve']:.4f} | {t['dead_at_end_train']} | {t['never_fire_on_eval_frac']:.4f} | {t['steps_run']:,} | {t['passes']:.2f} |")
    (OUT / 'tables.md').write_text('\n'.join(md) + '\n')
    print(json.dumps({k: v for k, v in S.items() if k not in ('train', 'levels')}, indent=1, default=float))


if __name__ == '__main__':
    main()
