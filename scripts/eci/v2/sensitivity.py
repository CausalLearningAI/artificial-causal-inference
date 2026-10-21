"""
ECI v2 NES sensitivity index: 'stable in k of n checks' for every neuron NES selects in the PRIMARY setting.

GRID (REDEFINED 2026-10-21 by the user, before this index was recomputed). One test throughout: Welch t on the
per-video outcome (paired t for mice family A); rank tests and the sign-flip test are NOT checks; outcome variants
are NOT checks (the index is computed separately for each variant).
  primary   Bonferroni, window 'full' (mice: habituation = its LAST 15 min, the default; odor / post their whole 15 min),
            mean = per-video mean; bouts = bout rate, threshold q 0.95, merge gap 0; latency = first frame above q 0.95.
  checks    mean     window hwhole (mice only: habituation on its whole 30 min), window trim30 (the first n_trim = 150
                     frames = 30 s of every video's window dropped: the start-of-video handling)
            bouts    threshold q 0.90, q 0.99, merge gap 2 frames, window hwhole, window trim30
            latency  threshold q 0.90, q 0.99, window hwhole, window trim30
            binary outcomes (the fnorm set and the MAP set): NO threshold checks (the outcome is already binary).
  A check whose setting does not exist for an analysis (hwhole outside mice habituation analyses) is not counted in n.
  variants  continuous (<sae>_smooth, primary input), prob (<sae>_prob), binary_fnorm (<sae>), binary_map (<sae>_map).
Output: results/vision/eci_v2/sensitivity.json {variant: {result set: {analysis: {aggregation|prefix: {neuron: {k, n,
checks}}}}}, files: per-setting result.json paths} + sensitivity_summary.json.
"""

import glob
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
V = REPO / 'results/vision'
SUB = {'mean': '', 'bouts': 'maxpool_bouts', 'latency': 'maxpool_latency'}
PRIM = {'mean': 'max_mean_t_bonferroni_full', 'bouts': 'bout_rate_q0.95_g0_t_bonferroni_full',
        'latency': 'latency_q0.95_t_bonferroni_common'}
CHECKS = {'mean': [('window_hwhole', 'max_mean_t_bonferroni_hwhole'), ('window_trim30', 'max_mean_t_bonferroni_trim30')],
          'bouts': [('q0.90', 'bout_rate_q0.90_g0_t_bonferroni_full'), ('q0.99', 'bout_rate_q0.99_g0_t_bonferroni_full'),
                    ('gap2', 'bout_rate_q0.95_g2_t_bonferroni_full'),
                    ('window_hwhole', 'bout_rate_q0.95_g0_t_bonferroni_hwhole'),
                    ('window_trim30', 'bout_rate_q0.95_g0_t_bonferroni_trim30')],
          'latency': [('q0.90', 'latency_q0.90_t_bonferroni_common'), ('q0.99', 'latency_q0.99_t_bonferroni_common'),
                      ('window_hwhole', 'latency_q0.95_t_bonferroni_hwhole'),
                      ('window_trim30', 'latency_q0.95_t_bonferroni_trim30')]}
VARIANTS = {'continuous': '_smooth', 'prob': '_prob', 'binary_fnorm': '', 'binary_map': '_map'}
BINARY = ('binary_fnorm', 'binary_map')
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
    out, files, summ = {}, {}, {}
    for var, suf in VARIANTS.items():
        for key, (root, sae) in SETS.items():
            nm = sae + suf
            for sub0 in [''] + EXTRA.get(key, []):
                tag = f'{key}/{sub0}' if sub0 else key
                base = V / root / nm / sub0
                for agg, sub in SUB.items():
                    for rj in sorted(glob.glob(str(base / sub / '*' / 'result.json'))):
                        aid = Path(rj).parent.name
                        P = load(rj)
                        for k, rec in P.items():
                            if not k.endswith(PRIM[agg]) or not isinstance(rec, dict):
                                continue
                            pre = k.split('_')[0]
                            prim = sel(rec)
                            files.setdefault(var, {}).setdefault(tag, {}).setdefault(aid, {}).setdefault(agg, {})['primary'] = rj
                            if not prim:
                                continue
                            res = {j: {'k': 0, 'n': 0, 'checks': {}} for j in prim}
                            for name, setting in CHECKS[agg]:
                                if var in BINARY and name.startswith('q0'):
                                    continue
                                r2 = P.get(f'{pre}_{setting}')
                                if r2 is None:
                                    continue
                                s2 = sel(r2)
                                for j in prim:
                                    res[j]['n'] += 1
                                    res[j]['checks'][name] = j in s2
                                    res[j]['k'] += int(j in s2)
                            out.setdefault(var, {}).setdefault(tag, {}).setdefault(aid, {})[f'{agg}|{pre}'] = res
        for tag, d in out.get(var, {}).items():
            n = [r for a in d.values() for s_ in a.values() for r in s_.values()]
            summ.setdefault(var, {})[tag] = {
                'primary_neuron_settings': len(n), 'stable_all_checks': sum(r['n'] > 0 and r['k'] == r['n'] for r in n),
                'stable_ge_half': sum(r['n'] > 0 and r['k'] >= r['n'] / 2 for r in n),
                'no_checks': sum(r['n'] == 0 for r in n)}
    (V / 'eci_v2/sensitivity.json').write_text(json.dumps({'grid': __doc__, 'index': out, 'files': files}, indent=1))
    (V / 'eci_v2/sensitivity_summary.json').write_text(json.dumps({'grid': __doc__, 'by_variant': summ}, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()
