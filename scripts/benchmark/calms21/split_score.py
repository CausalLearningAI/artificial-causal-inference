"""CalMS21 - DESCRIPTIVE split robustness (coordinator request; changes no pre-registered verdict). Reads
OUT/eval_splits10/<run>_s<seed>.json (same SAEs and theta*, re-scored with video-level split seeds 0-9; halves deal the
videos holding positives of each label alternately, rarest label = attack first, so each half gets 2-3 of the 5 attack
videos). Per run: primary AUROC (mean cross-fitted best-neuron AUROC over attack, investigation, mount) per split =
mean over the 6 SAE seeds; mean +- SE over the 10 splits next to the pre-registered splits-0-1 value; per-label AUROC
mean and SD across splits; positive videos per half. -> OUT/res/splits.md, splits.json. Login safe."""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calms_common import OUT, PLAN  # noqa: E402

D = OUT / 'eval_splits10'
labs = json.loads((PLAN / 'eval_plan.json').read_text())['scored_labels']
runs = sorted({re.sub(r'_s\d+\.json$', '', f.name) for f in D.glob('*_s*.json')})
L = ['# CalMS21 split robustness (DESCRIPTIVE): 10 video-level split seeds, same SAEs', '',
     '| run | primary AUROC, pre-registered splits 0-1 | primary AUROC, 10 splits (mean ± SE over splits) | '
     + ' | '.join(f'{b} AUROC mean (SD over splits)' for b in labs) + ' | seeds |', '|---|---|---|' + '---|' * (len(labs) + 1)]
out = {}
comp = None
for r in runs:
    J = [json.loads(f.read_text()) for f in sorted(D.glob(f'{r}_s*.json')) if re.fullmatch(rf'{re.escape(r)}_s\d+\.json', f.name)]
    k = J[0]['full']
    sp = sorted(J[0][k], key=lambda x: int(x[5:]))
    per_split = {s: [np.mean([np.mean([j[k][s][b]['cf_auroc']]) for j in J]) for b in labs] for s in sp}
    prim = np.array([np.mean(v) for v in per_split.values()])
    pre = float(np.mean([np.mean([np.mean([j[k][s][b]['cf_auroc'] for s in ('split0', 'split1')]) for b in labs]) for j in J]))
    lab = {b: np.array([per_split[s][i] for s in sp]) for i, b in enumerate(labs)}
    out[r] = {'pre_registered_primary': pre, 'primary_10_mean': float(prim.mean()),
              'primary_10_se': float(prim.std(ddof=1) / np.sqrt(len(prim))), 'per_split_primary': prim.tolist(),
              'label_mean': {b: float(v.mean()) for b, v in lab.items()}, 'label_sd': {b: float(v.std(ddof=1)) for b, v in lab.items()},
              'n_seeds': len(J)}
    comp = comp or J[0]['halves']
    L.append(f"| {r} | {pre:.4f} | {prim.mean():.4f} ± {out[r]['primary_10_se']:.4f} | "
             + ' | '.join(f'{v.mean():.3f} ({v.std(ddof=1):.3f})' for v in lab.values()) + f' | {len(J)} |')
if comp:
    L += ['', '## test videos with a positive, per half (half 0 / half 1)', '', '| split | ' + ' | '.join(labs) + ' |',
          '|---|' + '---|' * len(labs)]
    for s, h in sorted(comp.items(), key=lambda x: int(x[0][5:])):
        L.append(f'| {s} | ' + ' | '.join(f"{h['pos_videos_per_half'][b][0]} / {h['pos_videos_per_half'][b][1]}" for b in labs) + ' |')
(OUT / 'res').mkdir(parents=True, exist_ok=True)
(OUT / 'res' / 'splits.md').write_text('\n'.join(L) + '\n')
(OUT / 'res' / 'splits.json').write_text(json.dumps({'runs': out, 'halves': comp}, indent=1))
print('\n'.join(L))
