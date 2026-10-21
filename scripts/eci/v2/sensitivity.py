"""
ECI v2 NES sensitivity index: 'stable in k of n checks' for every neuron NES selects in the PRIMARY setting.

FIXED GRID (written 2026-10-21 before the index was computed; every setting is one the NES runners already compute and
store per analysis in result.json, so each check has its own per-setting result.json):
  primary   input <sae>_smooth (continuous), Bonferroni, window 'full' (mice habituation = its last 15 min), test t;
            mean: per-video mean; bouts: bout rate q 0.95, merge gap 0; latency: q 0.95.
  checks (same prefix, same analysis):
    mean     window hwhole (mice: habituation on its whole 30 min), window trim30, sign-flip test (family A only),
             input prob (P(on): mean = expected fraction of time active), input binary (fnorm-thresholded)
    bouts    threshold q 0.90, q 0.99, merge gap 2, window hwhole, window trim30, input prob, input binary
    latency  threshold q 0.90, q 0.99, rank-based t test, window hwhole, window trim30, input prob, input binary
  A check a setting does not exist for (e.g. hwhole outside mice habituation analyses, sign-flip outside family A)
  is not counted in n. k = number of checks whose selected set contains the neuron.
Not in the runners (reported as not run): rank-based tests for the per-video mean and the bout rate.
Output: results/vision/eci_v2/sensitivity.json {result set: {analysis: {aggregation|prefix: {neuron: {k, n, checks}}}}}
plus the per-setting result.json paths.
"""
import glob
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
V = REPO / 'results/vision'
SUB = {'mean': '', 'bouts': 'maxpool_bouts', 'latency': 'maxpool_latency'}
PRIM = {'mean': 'max_mean_t_bonferroni_full', 'bouts': 'bout_rate_q0.95_g0_t_bonferroni_full',
        'latency': 'latency_q0.95_t_bonferroni_common'}
CHECKS = {'mean': [('window_hwhole', 'smooth', 'max_mean_t_bonferroni_hwhole'),
                   ('window_trim30', 'smooth', 'max_mean_t_bonferroni_trim30'),
                   ('test_signflip', 'smooth', 'max_mean_signflip_bonferroni_full'),
                   ('input_prob', 'prob', 'max_mean_t_bonferroni_full'),
                   ('input_binary', 'binary', 'max_mean_t_bonferroni_full')],
          'bouts': [('q0.90', 'smooth', 'bout_rate_q0.90_g0_t_bonferroni_full'),
                    ('q0.99', 'smooth', 'bout_rate_q0.99_g0_t_bonferroni_full'),
                    ('gap2', 'smooth', 'bout_rate_q0.95_g2_t_bonferroni_full'),
                    ('window_hwhole', 'smooth', 'bout_rate_q0.95_g0_t_bonferroni_hwhole'),
                    ('window_trim30', 'smooth', 'bout_rate_q0.95_g0_t_bonferroni_trim30'),
                    ('input_prob', 'prob', 'bout_rate_q0.95_g0_t_bonferroni_full'),
                    ('input_binary', 'binary', 'bout_rate_q0.95_g0_t_bonferroni_full')],
          'latency': [('q0.90', 'smooth', 'latency_q0.90_t_bonferroni_common'),
                      ('q0.99', 'smooth', 'latency_q0.99_t_bonferroni_common'),
                      ('test_rank', 'smooth', 'latency_q0.95_rank_t_bonferroni_common'),
                      ('window_hwhole', 'smooth', 'latency_q0.95_t_bonferroni_hwhole'),
                      ('window_trim30', 'smooth', 'latency_q0.95_t_bonferroni_trim30'),
                      ('input_prob', 'prob', 'latency_q0.95_t_bonferroni_common'),
                      ('input_binary', 'binary', 'latency_q0.95_t_bonferroni_common')]}
SETS = {'mice': ('mice/eci/nes', 'v2_mice_c0_w2048_s7'),
        'ants_v2': ('ants/eci_r4v2p/nes', 'v2_ants_c0_w2048_s6'), 'ants_v3': ('ants/eci_r4v3p/nes', 'v2_ants_c0_w2048_s6'),
        'ants_v4': ('ants/eci_r4v4p/nes', 'v2_ants_c0_w2048_s6'), 'frogs_juv': ('frogs/eci/nes', 'v2_frogs_c0_w2048_s6'),
        'frogs_tad': ('frogs/eci_tadpole_r4/nes', 'v2_frogs_c0_w2048_s6')}
EXTRA = {'mice': ['family_S', 'subsets/ash1l_all', 'subsets/kdm6b_all', 'subsets/kmt5b_all', 'subsets/all_f',
                  'subsets/all_m'], 'ants_v3': ['v3_2_vs_8_noday']}


def sel(rec):
    return {int(x['neuron'] if isinstance(x, dict) else x) for x in (rec.get('selected') or [])}


def load(path, cache={}):
    if path not in cache:
        cache[path] = json.loads(Path(path).read_text()) if Path(path).exists() else None
    return cache[path]


def main():
    out, files, tot = {}, {}, {'primary_selected': 0, 'stable_all': 0, 'stable_ge_half': 0}
    for key, (root, sae) in SETS.items():
        names = {'smooth': f'{sae}_smooth', 'binary': sae, 'prob': f'{sae}_prob'}
        for sub0 in [''] + EXTRA.get(key, []):
            tag = f'{key}/{sub0}' if sub0 else key
            base = {v: V / root / n / sub0 for v, n in names.items()}
            for agg, sub in SUB.items():
                for rj in sorted(glob.glob(str(base['smooth'] / sub / '*' / 'result.json'))):
                    aid = Path(rj).parent.name
                    P = load(rj)
                    for k, rec in P.items():
                        if not k.endswith(PRIM[agg]) or not isinstance(rec, dict):
                            continue
                        pre = k.split('_')[0]
                        prim = sel(rec)
                        files.setdefault(tag, {}).setdefault(aid, {}).setdefault(agg, {})['primary'] = rj
                        if not prim:
                            continue
                        res = {j: {'k': 0, 'n': 0, 'checks': {}} for j in prim}
                        for name, var, setting in CHECKS[agg]:
                            f = base[var] / sub / aid / 'result.json'
                            R = load(str(f))
                            r2 = R.get(f'{pre}_{setting}') if R else None
                            if r2 is None:
                                continue
                            files[tag][aid][agg][name] = str(f)
                            s2 = sel(r2)
                            for j in prim:
                                res[j]['n'] += 1
                                res[j]['checks'][name] = j in s2
                                res[j]['k'] += int(j in s2)
                        out.setdefault(tag, {}).setdefault(aid, {})[f'{agg}|{pre}'] = res
                        for j, r in res.items():
                            tot['primary_selected'] += 1
                            tot['stable_all'] += int(r['n'] > 0 and r['k'] == r['n'])
                            tot['stable_ge_half'] += int(r['n'] > 0 and r['k'] >= r['n'] / 2)
    (V / 'eci_v2/sensitivity.json').write_text(json.dumps({'grid': __doc__, 'totals': tot, 'index': out,
                                                           'files': files}, indent=1))
    summ = {}
    for tag, d in out.items():
        n = [r for a in d.values() for s in a.values() for r in s.values()]
        summ[tag] = {'primary_neuron_settings': len(n), 'stable_all_checks': sum(r['n'] > 0 and r['k'] == r['n'] for r in n),
                     'stable_ge_half': sum(r['n'] > 0 and r['k'] >= r['n'] / 2 for r in n)}
    print(json.dumps({'totals': tot, 'by_set': summ}, indent=1))
    (V / 'eci_v2/sensitivity_summary.json').write_text(json.dumps({'totals': tot, 'by_set': summ}, indent=1))


if __name__ == '__main__':
    main()
