"""build gold/prompts.json from the N2 seed body points (seed_*.json) + hand-placed thin points (thin_manual.json).
Each thin point is a positive SAM prompt of the seed object nearest to it; the crop is grown to contain all its points."""
import json, sys
from pathlib import Path
G = Path(__file__).parent
seeds = {}
for f in sorted(G.glob('seed_*.json')):
    seeds.update(json.loads(f.read_text()))
thin = json.loads((G / 'thin_manual.json').read_text())
S = {'mice': 1024, 'ants_v2': 768, 'ants_v3': 832, 'ants_v4': 816, 'frogs_juv': 800, 'frogs_tad': 384}
old = json.loads((G / 'prompts.json').read_text()) if (G / 'prompts.json').exists() else {}
out = {}
for key, sd in seeds.items():
    side_max = next(v for k, v in S.items() if key.startswith(k))
    t = thin.get(key, {})
    if 'mice' in t:  # every mouse placed by hand: body points + tail points (positives), square crop around them
        an = []
        for m in t['mice']:
            P_ = m['body'] + m['tail']
            xs, ys = [q[0] for q in P_], [q[1] for q in P_]
            s2 = int(min(side_max, max(160, max(xs) - min(xs) + 120, max(ys) - min(ys) + 120)))
            cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
            an.append({'pts': P_, 'labels': [1] * len(P_), 'crop': [int(max(0, min(cx - s2 / 2, side_max - s2))),
                                                                   int(max(0, min(cy - s2 / 2, side_max - s2))), s2]})
        out[key] = {'animals': an, 'thin': [q for m in t['mice'] for q in m['tail']]}
        if key in old and old[key].get('manual'):
            out[key] = old[key]
        continue
    if 'override' in t:
        out[key] = {'animals': t['override'], 'thin': t.get('thin', [])}
        continue
    pts = [p for v in t.values() for p in v]
    an = [{'pts': [o['pts'][0]], 'labels': [1], 'crop': list(o['crop'])} for o in sd]
    for p in pts:
        j = min(range(len(sd)), key=lambda i: (sd[i]['pts'][0][0] - p[0]) ** 2 + (sd[i]['pts'][0][1] - p[1]) ** 2)
        an[j]['pts'].append(p)
        an[j]['labels'].append(1)
    for a in an:
        x0, y0, s = a['crop']
        xs, ys = [p[0] for p in a['pts']], [p[1] for p in a['pts']]
        lo_x, lo_y = min(min(xs) - 12, x0), min(min(ys) - 12, y0)
        hi_x, hi_y = max(max(xs) + 12, x0 + s), max(max(ys) + 12, y0 + s)
        s2 = int(min(side_max, max(hi_x - lo_x, hi_y - lo_y)))
        a['crop'] = [int(max(0, min(lo_x, side_max - s2))), int(max(0, min(lo_y, side_max - s2))), s2]
    out[key] = {'animals': an, 'thin': pts}
    if key in old and old[key].get('manual'):  # hand-corrected entries are kept as they are
        out[key] = old[key]
(G / 'prompts.json').write_text(json.dumps(out, indent=1))
print(len(out), 'frames', sum(len(v['thin']) for v in out.values()), 'thin points')
