import json, sys, numpy as np
dom = sys.argv[1]
suf = sys.argv[2] if len(sys.argv) > 2 else ''
d = json.load(open(f'results/vision/eci_v2_0B/dyn2/{dom}_score{suf}.json'))
el = lambda G: {c: r for c, r in G.items() if r['eligible'] and c.split('|')[1] not in ('YOL', 'BOL', 'FOL', 'onlid_any', 'lid_any3')}
rows = {}
for k in ('hmm', 'smooth'):
    G = el(d['v2.0'][k]); rows[f'v2.0 {k}'] = G
v1 = {c: {'v2_AUROC': r['v1_AUROC'], 'v2_nAP': r['v1_nAP'], 'v2_top1_prec': r['v1_top1_prec']} for c, r in el(d['v2.0']['hmm']).items()}
print(f'{dom}: cells', list(v1))
print('v1 (11 seeds mean): AUROC', round(np.mean([r['v2_AUROC'] for r in v1.values()]), 3), 'top1', round(np.mean([r['v2_top1_prec'] for r in v1.values()]), 3))
for k, G in rows.items():
    print(k, 'PASS orig/final', sum(r['PASS'] for r in G.values()), sum(r['PASS_final'] for r in G.values()), '/', len(G), 'AUROC', round(np.mean([r['v2_AUROC'] for r in G.values()]), 3), 'top1', round(np.mean([r['v2_top1_prec'] for r in G.values()]), 3))
for arm in (('A10', 'A20', 'B10', 'B20') if suf == 'w' else ('X0', 'A', 'B')):
    S = [d[f'{arm}_s{s}'] for s in (990, 991, 992)]
    Gs = [el(x['gate']) for x in S]
    pf = [sum(r['PASS_final'] for r in G.values()) for G in Gs]
    po = [sum(r['PASS'] for r in G.values()) for G in Gs]
    au = [np.mean([r['v2_AUROC'] for r in G.values()]) for G in Gs]
    t1 = [np.mean([r['v2_top1_prec'] for r in G.values()]) for G in Gs]
    pr = [np.nanmean([v['strat_AUROC'] for v in x['prox'].values() if v.get('strat_AUROC') is not None]) for x in S]
    print(arm, 'PASS orig', po, 'final', pf, 'AUROC', np.round(au, 3), 'top1', np.round(t1, 3), 'stratAUROC', np.round(pr, 3),
          'dyn256', [x['dyn256']['share'] for x in S], 'dec>0.25', [x['n_dec_share_gt_0.25'] for x in S], '@256', [x['n_dec_share_gt_0.25_256'] for x in S])
    for c in Gs[0]:
        print('   ', c, 'final', [('P' if G[c]['PASS_final'] else 'F') for G in Gs], 'AUROC', [round(G[c]['v2_AUROC'], 3) for G in Gs])
