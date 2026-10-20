"""Collect results/vision/ants/eci_v0_audit/audit_*.json into one table (summary.csv) + summary.json."""
import json
from pathlib import Path

import pandas as pd

D = Path(__file__).resolve().parents[3] / 'results/vision/ants/eci_v0_audit'
rows = []
for f in sorted(D.glob('audit_*.json')):
    for key, v in json.loads(f.read_text()).items():
        for mode in ['mean', 'tok0', 'patchavg']:
            r = v[mode]
            hits = r['hits']
            rows.append(dict(
                sae=key, encoding=mode, fve_on_input=r['fve_on_input'], alive=r['n_alive'],
                hits_raw=r['n_hits_raw'], expected_fp=r['expected_fp'], hits_bonf=r['n_hits_bonf'], min_p=r['min_p'],
                recursive_matches=sorted(r['recursive_nes_hits']) == sorted(int(h['neuron'][2:]) for h in hits),
                always_on_f1=r['always_on_f1'],
                best_f1=r['best_f1_code']['f1'], best_f1_auroc=r['best_f1_code']['auroc'],
                best_f1_rtilde=r['best_f1_code']['r_tilde'], best_f1_is_hit=r['best_f1_code']['is_nes_hit'],
                best_auroc=r['best_auroc_code']['auroc'],
                cf_f1=r['crossfit_best_f1']['f1'], cf_auroc=r['crossfit_best_f1']['auroc'], cf_rtilde=r['crossfit_best_f1']['r_tilde'],
                hit_best_groom_f1=max([h['groom_f1'] for h in hits], default=None),
                hit_best_groom_rank=min([h['groom_f1_rank'] for h in hits], default=None),
                hit_best_groom_auroc=max([h['groom_auroc'] for h in hits], default=None),
                hits_in_groom_top1=sum(h['groom_f1_rank'] == 1 for h in hits),
                best_pos1ad_f1=r['best_pos1_batch_ad_code']['f1'], best_pos1ad_is_hit=r['best_pos1_batch_ad_code']['is_nes_hit'],
                hit_best_pos1ad_f1=max([h['pos1_batch_ad_f1'] for h in hits], default=None),
                hits_position_perm_p_lt_0p05=sum(h['eta2_position_perm_p'] < 0.05 for h in hits),
            ))
df = pd.DataFrame(rows)
df.to_csv(D / 'summary.csv', index=False)
agg = df.groupby('encoding').agg(n=('sae', 'size'), hits_raw=('hits_raw', 'mean'), expected_fp=('expected_fp', 'mean'),
                                 hits_bonf=('hits_bonf', 'sum'), best_f1=('best_f1', 'mean'), cf_f1=('cf_f1', 'mean'),
                                 cf_auroc=('cf_auroc', 'mean'), cf_rtilde=('cf_rtilde', 'mean'),
                                 best_f1_is_hit=('best_f1_is_hit', 'sum'), best_pos1ad_f1=('best_pos1ad_f1', 'mean'),
                                 best_pos1ad_is_hit=('best_pos1ad_is_hit', 'sum'))
(D / 'summary.json').write_text(agg.to_json(indent=1))
pd.set_option('display.width', 250)
print(df.drop(columns=['recursive_matches']).round(3).to_string())
print(agg.round(3).to_string())
print('recursive NES == single pass in all rows:', bool(df.recursive_matches.all()))
