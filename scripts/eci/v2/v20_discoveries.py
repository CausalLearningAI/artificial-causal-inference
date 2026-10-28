"""
ECI v2.0 discovery counts (final PASS B models) with scripts/eci/v2/discoveries.py count() unchanged: neurons selected
(round >= 1) per analysis and prefix, Bonferroni, full / common window, max pooling; NES aggregations mean / bouts /
latency and NESS (dines) 'selected'. Sets: <sae>_hmm (P(on), primary for the mean), <sae>_smooth (continuous),
<sae>_hmmv (Viterbi on-state: bouts / latency of the HMM). -> results/vision/eci_v2_0B/discoveries.json
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'scripts/eci/v2'))
import discoveries as DI  # noqa: E402

V = REPO / 'results/vision'
B = REPO / 'results/vision/eci_v2_0B'


def name(dom):
    return json.loads((B / dom / 'best.json').read_text())['name']


SETS = {'mice': [('mice/eci/nes', 'mice', 'mice')],
        'ants': [(f'ants/{e}/nes', f'ants_{e}', 'ants') for e in ('eci_r4v2p', 'eci_r4v3p', 'eci_r4v4p')],
        'frogs': [('frogs/eci/nes', 'frogs', 'frogs'), ('frogs/eci_tadpole_r4/nes', 'frogs_tadpole', 'frogs')]}


def main():
    res = {}
    for dom, items in SETS.items():
        sae = name(dom)
        for root, key, _ in items:
            for suf in ('_hmm', '_smooth', '_hmmv'):
                nm = sae + suf
                r = {}
                for o, sub in DI.SUB.items():
                    r[f'NES_{o}'] = DI.count(V / root / nm / sub / 'summary.csv', o)
                    f = V / 'eci_v2/ness' / key / nm / 'dines' / sub / 'summary.csv'
                    r[f'NESS_{o}'] = DI.count(f, o) if f.exists() else None
                res[f'{key}|{nm}'] = r
    (B / 'discoveries.json').write_text(json.dumps(res, indent=1))
    for k, r in res.items():
        tot = {m: ({p: sum(a.get(p, 0) for a in v.values()) for p in (128, 256, 2048)} if v else None) for m, v in r.items()}
        print(k, json.dumps(tot))


if __name__ == '__main__':
    main()
