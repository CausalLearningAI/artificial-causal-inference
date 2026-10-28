"""
ECI v2: discovery counts per contrast from the NES runners' and NESS (atlas_v37_methods.py) tidy summaries. CPU, reads
small csv files only.

For every v2 codes set (the NES input '<sae>' = smoothed + fnorm thresholded, and the sensitivity '<sae>_smooth' =
smoothed, no threshold) and every analysis: number of neurons selected (rows with round >= 1) per temporal aggregation
(mean = per-video mean of codes_max, bouts = bout rate q 0.95, latency = first frame above q 0.95), prefix, for the
primary settings: Bonferroni, window 'full' (latency 'common'; mice NES family A uses its own matched window only
as a secondary). NESS: 'selected' atoms (kept + undetermined) of dines/ (+ familyC_dines for mice).
Writes results/vision/eci_v2/discoveries.json and discoveries.md.
"""
import glob
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
V = REPO / 'results/vision'
SETS = {'mice': [('mice/eci/nes', 'v2_mice_c0_w2048_s7', 'mice')],
        'ants': [(f'ants/{e}/nes', 'v2_ants_c0_w2048_s6', f'ants_{e}') for e in ('eci_r4v2p', 'eci_r4v3p', 'eci_r4v4p')],
        'frogs': [('frogs/eci/nes', 'v2_frogs_c0_w2048_s6', 'frogs'),
                  ('frogs/eci_tadpole_r4/nes', 'v2_frogs_c0_w2048_s6', 'frogs_tadpole')]}
SUB = {'mean': '', 'bouts': 'maxpool_bouts', 'latency': 'maxpool_latency'}
STRATA = ('ash1l_all', 'kdm6b_all', 'kmt5b_all', 'all_f', 'all_m')


def count(f, outcome):
    if not Path(f).exists():
        return None
    s = pd.read_csv(f)
    if 'correction' in s:
        s = s[s.correction == 'bonferroni']
    if 'test' in s:  # the sign-flip rows are a sensitivity check on the same neurons
        s = s[s.test == 't']
    if 'window' in s:
        s = s[s.window.isin(['full', 'common'])]
    if outcome == 'mean' and 'outcome_type' in s:
        s = s[s.outcome_type == 'mean']
    if 'pooling' in s:
        s = s[s.pooling == 'max']
    out = {}
    for (a, p), g in s.groupby(['analysis_id', 'prefix']):
        out.setdefault(a, {})[int(p)] = int((g['round'] >= 1).sum())
    return out


def main():
    res = {}
    for dom, items in SETS.items():
        for root, sae, key in items:
            for nm in (sae, f'{sae}_smooth', f'{sae}_prob', f'{sae}_map'):
                r = {}
                base = V / root / nm
                for o, sub in SUB.items():
                    r[f'NES_{o}'] = count(base / sub / 'summary.csv', o)
                    if (base / 'v3_2_vs_8_noday' / sub / 'summary.csv').exists():
                        r[f'NES_{o}_v3_2_vs_8_noday'] = count(base / 'v3_2_vs_8_noday' / sub / 'summary.csv', o)
                    nk = key if key != 'frogs_tadpole' else 'frogs_tadpole'
                    for root2 in (f'eci_v2/ness/{nk}/{nm}', f'eci_rollout_r4/ness/{nk}/{nm}'):
                        f = V / root2 / 'dines' / sub / 'summary.csv'
                        if f.exists():
                            r[f'NESS_{o}'] = count(f, o)
                            fc = V / root2 / 'familyC_dines' / sub / 'C_summary.csv'
                            if fc.exists():
                                r[f'NESS_{o}_familyC'] = count(fc, o)
                            fcn = V / root2 / 'familyC_nes' / sub / 'C_summary.csv'
                            if fcn.exists():
                                r[f'NES_{o}_familyC'] = count(fcn, o)
                    f = V / f'eci_v2/ness_v3_2v8/{key}/{nm}' / 'dines' / sub / 'summary.csv'
                    if f.exists():
                        r[f'NESS_{o}_v3_2_vs_8_noday'] = count(f, o)
                if dom == 'mice':  # family S and the line / sex strata (scripts/eci/v2/nes_mice.sh)
                    extra = [('S', base / 'family_S', V / f'eci_v2/ness_S/mice/{nm}')]
                    for st in STRATA:
                        extra.append((f'strata_{st}', base / 'subsets' / st, V / f'eci_v2/ness_strata/{st}/mice/{nm}'))
                    for tag, nd, sd in extra:
                        for o, sub in SUB.items():
                            r[f'{tag}_NES_{o}'] = count(nd / sub / 'summary.csv', o)
                            r[f'{tag}_NESS_{o}'] = count(sd / 'dines' / sub / 'summary.csv', o)
                res[f'{key}|{nm}'] = {'nes_dir': str(base), **r}
    (V / 'eci_v2/discoveries.json').write_text(json.dumps(res, indent=1))
    L = ['# ECI v2 discoveries (number of selected neurons; Bonferroni, primary window; per prefix)', '',
         'NES = Neural Effect Search (src/eci/nes.py runners); NESS = src/eci/dines.py (atlas_v37_methods.py). Input sets: '
         '`<sae>` = 0.6 s smoothed + fnorm firing threshold (pre-specified); `<sae>_smooth` = 0.6 s smoothed, no '
         'threshold (PRIMARY, user decision 2026-10-21); `<sae>_prob` = per-neuron P(on) (src/eci/v2/prob.py). Mice extra: '
         'S_* = family S (female vs male pools), strata_<line>_<sex>_* = families A/B within a line or sex.', '']
    for k, r in res.items():
        L += [f'## {k}', '', '| method / aggregation | ' + 'analysis: counts per prefix |', '|---|---|']
        for m, d in r.items():
            if m == 'nes_dir' or d is None:
                continue
            L.append(f'| {m} | ' + '; '.join(f'{a}: ' + ' / '.join(f'{p}:{n}' for p, n in sorted(v.items()))
                                             for a, v in d.items()) + ' |')
        L.append('')
    (V / 'eci_v2/discoveries.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
