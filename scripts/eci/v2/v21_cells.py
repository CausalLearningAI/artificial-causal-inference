"""AMENDMENT 4 (d): per-cell ants / mice table, median over seeds per rung, AUROC P(on) and top-1% precision P(on), frame max
and top10 (v1 / PB frame max) -> results/vision/eci_v2_1/ladder/cells_<dom>.md / .json. Small JSON reads."""
import json
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).parent))
O = Path(__file__).resolve().parents[3] / 'results/vision/eci_v2_1/ladder'
RUNGS = {'v1': [f'v1_s{i}' for i in range(11)], 'PB': ['PB'], 'SN2': [f'SN2_s{i}' for i in (1001, 1002, 1003)],
         'S': [f'S_s{i}' for i in (1001, 1002, 1003)], 'V': [f'V_s{i}' for i in (1001, 1002, 1003)],
         'W': [f'W_s{i}' for i in (1001, 1002, 1003)], 'B': [f'B_s{i}' for i in (1001, 1002, 1003)],
         'S fr': [f'S_s{i}' for i in (1004, 1005, 1006)], 'V fr': [f'V_s{i}' for i in (1004, 1005, 1006)],
         'W fr': [f'W_s{i}' for i in (1004, 1005, 1006)]}
res, L = {}, []
for dom in ('ants', 'mice'):
    for agg in ('', 'top10'):
        cells = None
        tab = {}
        for r, ks in RUNGS.items():
            J = []
            for k in ks:
                t = k if (not agg or r in ('v1', 'PB')) else f'{k}_{agg}'
                f = O / dom / f'{t}.json'
                if f.exists():
                    J.append(json.loads(f.read_text()))
            if not J:
                continue
            cs = [('gate', c) for c in J[0]['gate'] if c.split('|')[1] not in ('YOL', 'BOL', 'FOL', 'onlid_any', 'lid_any3')]
            cs += [('focal', c) for c in J[0].get('focal', {})]
            cells = cells or cs
            tab[r] = {f'{p}:{c}': [float(np.median([j[p][c][m] for j in J if j[p][c][m] is not None])) for m in ('AUROC', 'top1_prec')]
                      for p, c in cells}
        res[f'{dom}{"_" + agg if agg else ""}'] = tab
        L += [f'## {dom}, readout {agg or "frame max"} (v1 / PB frame max); cell: AUROC P(on) / top-1% prec P(on), median over seeds', '',
              '| cell | ' + ' | '.join(tab) + ' |', '|---|' + '---|' * len(tab)]
        for p, c in cells:
            L.append(f'| {c}{" (focal, reported)" if p == "focal" else ""} | ' + ' | '.join(
                f'{tab[r][f"{p}:{c}"][0]:.3f} / {tab[r][f"{p}:{c}"][1]:.3f}' for r in tab) + ' |')
        L.append('')
(O / 'cells.json').write_text(json.dumps(res, indent=1))
(O / 'cells.md').write_text('\n'.join(L))
print('\n'.join(L))
