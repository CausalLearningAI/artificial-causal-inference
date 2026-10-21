"""Round-2 summary: results/vision/ants/eci_v0_audit/{audit2_*,orig_*}.json + .npz -> summary2_*.csv + printout."""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

D = Path(__file__).resolve().parents[3] / 'results/vision/ants/eci_v0_audit'
runs = []  # (path, key, encoding, result dict, npz path)
for f in sorted(D.glob('audit2_*.json')):
    r = json.loads(f.read_text())
    for enc in ['mean', 'tok0', 'patchavg']:
        if enc in r:
            runs.append(('ours', r['key'], enc, r[enc], D / f"audit2_{r['key']}_{enc}.npz"))
for f in sorted(D.glob('orig_*.json')):
    m = re.match(r'orig_(.+)_(MEAN|CLS)\.json', f.name)
    if m:
        runs.append(('original', m.group(1), {'MEAN': 'mean', 'CLS': 'tok0'}[m.group(2)], json.loads(f.read_text()), f.with_suffix('.npz')))

rng = np.random.default_rng(0)
T1, T2 = [], []
pooled = {}  # (path, enc, agg) -> list of (alive pct array, n hits, observed hit pcts)
for path, key, enc, r, npz in runs:
    per = np.load(npz)
    alive = per['alive']
    auc = per['auroc'][alive]
    pcts = 100 * (np.argsort(np.argsort(auc, kind='stable'), kind='stable') + 0.5) / len(auc)  # rank percentile
    dev = np.abs(auc - 0.5)  # POST HOC (not pre-registered): two-sided alignment strength
    f1a = per['f1'][alive]
    jbest = f'N_{alive[int(np.argmax(f1a))] + 1}'
    for agg in ['mean', 'rate']:
        a = r[agg]
        sp = [h for h in a['hits'] if h['single_pass_hit']]
        T1.append(dict(path=path, sae=key, encoding=enc, agg=agg, alive=r['n_alive'], hits=a['n_hits'], expected_fp=a['expected_fp'],
                       bonf=a['n_bonf'], recursive_hits=len(a['recursive_hits']), best_auroc_any=r['best_auroc'],
                       auroc_q95=r['alive_auroc_quantiles']['0.95'], hitset_mean_pct_auroc=a.get('hitset_mean_pct_auroc'),
                       hitset_p=a.get('hitset_p_more_aligned'), n_hits_pct95=sum(h['pct_auroc'] >= 95 for h in sp),
                       best_f1_code=jbest, best_f1_code_is_hit=jbest in [h['neuron'] for h in sp],
                       notebook_ttest_nes=(r.get('notebook_nes', {}).get('t-testNES', {}).get('neurons'))))
        for h in a['hits']:
            j = int(h['neuron'][2:]) - 1
            d = abs(h['auroc'] - 0.5)
            T2.append(dict(path=path, sae=key, encoding=enc, agg=agg, posthoc_pct_absdev=float(100 * ((dev < d).sum() + 0.5 * (dev == d).sum()) / len(dev)),
                           **{k: h[k] for k in
                      ['neuron', 'single_pass_hit', 'recursive_hit', 'p', 'auroc', 'pct_auroc', 'r_tilde', 'pct_r_tilde', 'f1', 'pct_f1',
                       'f1_pos1ad', 'eta2_position', 'eta2_position_perm_p']}))
        pooled.setdefault((path, enc, agg), []).append((pcts, len(sp), [h['pct_auroc'] for h in sp]))
t1, t2 = pd.DataFrame(T1), pd.DataFrame(T2)
t1.to_csv(D / 'summary2_runs.csv', index=False)
t2.to_csv(D / 'summary2_hits.csv', index=False)
P = []
for (path, enc, agg), lst in pooled.items():
    obs = [x for _, _, hp in lst for x in hp]
    if not obs:
        P.append(dict(path=path, encoding=enc, agg=agg, n_runs=len(lst), n_hits=0))
        continue
    null = np.zeros(20000)
    for pc, n, _ in lst:
        if n:
            null += np.array([pc[rng.choice(len(pc), n, replace=False)].sum() for _ in range(20000)])
    null /= len(obs)
    P.append(dict(path=path, encoding=enc, agg=agg, n_runs=len(lst), n_hits=len(obs), mean_pct_auroc=float(np.mean(obs)),
                  frac_pct95=float(np.mean(np.array(obs) >= 95)), p_more_aligned=float((1 + (null >= np.mean(obs)).sum()) / 20001)))
tp = pd.DataFrame(P)
tp.to_csv(D / 'summary2_pooled.csv', index=False)
pd.set_option('display.width', 250)
pd.set_option('display.max_rows', 500)
print(t1.round(3).to_string())
print(t2[t2['agg'] == 'mean'].round(3).to_string())
print(tp.round(3).to_string())
