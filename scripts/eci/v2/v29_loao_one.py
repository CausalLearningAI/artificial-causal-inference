"""
ECI v2.9 ADDITION (POST HOC): leave-one-annotator-out table for ONE dictionary, written to its own file so the
existing results/vision/eci_v2_9/split/{tables.md, split.json} (LOAO for PREDk8_s2901) stay untouched.
Same code path as v29_split_combine.py --loao KEY (reuses its load / metrics / sel_sum).
Usage: python scripts/eci/v2/v29_loao_one.py PREDk16_s2901
-> results/vision/eci_v2_9/split/loao_<KEY>.{json,md}
"""
import json
import sys

import v29_split_combine as C


def main(key):
    D = C.load()
    r0 = next(iter(D.values()))
    QI = {n: i for i, n in enumerate(r0['names'])}
    r = D[key]
    lo, md = {'post_hoc': True, 'key': key}, [
        f'# Leave one annotator out: {key} (POST HOC; select = pools without that annotator, evaluate = its pools; rule sgap re-picked)', '',
        '| left out | chosen among the PREDk dictionaries | nose-nose pick / AUROC / AP (chance) / precision@top3/video | '
        'nose-tail pick / AUROC / AP (chance) / precision@top3/video |', '|---|---|---|---|']
    for s in C.LOAO:
        ch = max(D, key=lambda kk: C.sel_sum(D[kk], QI, s))
        p = r['picks']['sgap'][s]
        cells = []
        for i, b in enumerate(C.BEH):
            if p is None:
                cells.append('no pair')
                continue
            mt = C.metrics(r, QI, p[i], b, f'{s}|ev')
            lo[f'{s}|{b}'] = {'pick': int(p[i]), **mt}
            cells.append(f"#{p[i]} / {mt['auc']:.3f} / {mt['ap']:.3f} ({mt['chance_ap']:.4f}) / {mt['top3']:.3f}")
        lo[f'{s}|chosen'] = ch
        md.append(f"| {s.split('_')[1]} | {ch} | " + ' | '.join(cells) + ' |')
    (C.OUT / f'loao_{key}.json').write_text(json.dumps(lo, indent=1, default=float))
    (C.OUT / f'loao_{key}.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main(sys.argv[1])
