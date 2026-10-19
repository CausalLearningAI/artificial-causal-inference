"""Score the blind visual audit (audit_codes.txt) against the label key (audit_key.json): per set (top 48 held-out
frames of the A3 seed-2 neuron and of the supervised probe, nose_nose and nose_tail) label precision, visual
precision, missed-label rate, corrected precision; per annotator; contact-event length (5 fps tight mask, style
step) of each audited frame for the pass-by question. -> results/vision/eci_nnnt_analysis/q1b_audit.json"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
OUTA = REPO / 'results/vision/eci_nnnt_analysis'
BEEA = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/nnnt_analysis')
codes = dict(l.split() for l in (Path(__file__).parent / 'audit_codes.txt').read_text().splitlines()
             if l.strip() and not l.startswith('#'))
key = json.loads((OUTA / 'audit_key.json').read_text())
assert set(codes) == set(key), (len(codes), len(key))
ann = pd.read_csv(REPO / 'data/mice/v1/experiment.csv').set_index('observation_id').annotator
# contact event length of each audited frame (style arrays; A1 / 1.5 rule as cmd_style)
evlen = {}
for tid, k in key.items():
    z = np.load(BEEA / 'style' / f'{k["obs"]}.npz')
    nc, mx = z['ncomp'], z['maxc'].astype(float)
    four = mx[nc == 4]
    A1 = np.median(four) if len(four) >= 20 else 46.0
    ct = mx >= 1.5 * A1
    i = k['k']
    if not ct[i]:
        evlen[tid] = 0.0
        continue
    a = i
    while a > 0 and ct[a - 1]:
        a -= 1
    b = i
    while b + 1 < len(ct) and ct[b + 1]:
        b += 1
    evlen[tid] = (b - a + 1) / 5
target = {'nose_nose': 'NN', 'nose_tail': 'NT'}
labname = {'nose_nose': ('nn_mutual', 'np_directional'), 'nose_tail': ('nose_tail',)}
rows = []
for tid, k in key.items():
    for sn, rank in k['sets'].items():
        lb = 'nose_nose' if sn.endswith('nose_nose') else 'nose_tail'
        rows.append({'tid': tid, 'set': sn, 'label': lb, 'rank': rank, 'code': codes[tid],
                     'labelled': any(x in k['labels'] for x in labname[lb] + (lb,)),
                     'other_labelled': bool(k['labels']) and not any(x in k['labels'] for x in labname[lb] + (lb,)),
                     'annotator': ann[k['obs']], 'contact_event_s': evlen[tid], 'mask_contact': k['contact']})
D = pd.DataFrame(rows)
D['visual_target'] = D.code == D.label.map(target)
D['missed'] = D.visual_target & ~D.labelled


def summ(g):
    n = len(g)
    return {'n': n, 'label_precision': float(g.labelled.mean()), 'visual_target_share': float(g.visual_target.mean()),
            'labelled_and_visual': float((g.labelled & g.visual_target).mean()),
            'missed_label_share_of_frames': float(g.missed.mean()),
            'missed_over_visual_target': float(g.missed.sum() / max(1, g.visual_target.sum())),
            'corrected_precision': float((g.labelled | g.visual_target).mean()),
            'labelled_but_visually_not_target': float((g.labelled & ~g.visual_target).mean()),
            'codes': g.code.value_counts().to_dict(),
            'labelled_frames_contact_event_lt_0.5s': float(((g.contact_event_s < 0.5) & g.labelled).sum() /
                                                           max(1, g.labelled.sum()))}
out = {'sets': {sn: summ(g) for sn, g in D.groupby('set')},
       'per_annotator_by_label': {lb: {an: summ(g) for an, g in D[D.label == lb].groupby('annotator')}
                                  for lb in ('nose_nose', 'nose_tail')},
       'unique_frames': len(key), 'code_counts_unique': pd.Series(codes).value_counts().to_dict()}
# frame-level visual NT / NN among all unique frames, label crosstab
U = pd.DataFrame([{'tid': t, 'code': codes[t], 'labels': '+'.join(sorted(k['labels'])) or '-',
                   'annotator': ann[k['obs']]} for t, k in key.items()])
out['crosstab_code_x_labels'] = pd.crosstab(U.code, U.labels).to_dict()
out['per_annotator_unique'] = {an: {'n': int(len(g)), 'visual_NT': int((g.code == 'NT').sum()),
                                    'visual_NT_unlabelled_nt': int(((g.code == 'NT') & ~g['labels'].str.contains('nose_tail')).sum()),
                                    'visual_NN': int((g.code == 'NN').sum()),
                                    'visual_NN_unlabelled': int(((g.code == 'NN') & ~g['labels'].str.contains('nn_mutual|np_directional|nose_nose')).sum())}
                               for an, g in U.groupby('annotator')}
(OUTA / 'q1b_audit.json').write_text(json.dumps(out, indent=1, default=str))
for sn, s in out['sets'].items():
    print(sn, {k_: (round(v, 3) if isinstance(v, float) else v) for k_, v in s.items()})
print(json.dumps(out['per_annotator_unique']))
print(json.dumps(out['crosstab_code_x_labels']))
