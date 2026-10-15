"""Phase 2 base (rule in eci_recheck.sh): highest mean primary AUROC among the eight pre-registered grid arms; if it is
V-JEPA, the best image-encoder arm. Writes OUT/p2/base.txt. Login safe (reads eval JSONs)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calms_common import OUT, PLAN  # noqa: E402
from eci_res_score import load, per_seed  # noqa: E402

GRID8 = ['d2_448', 'd2_896', 'd3_512_L10', 'd3_1024_L10', 'd3l_512_L20', 'd3l_1024_L20', 'vj_512', 'vj_1024']
labs = json.loads((PLAN / 'eval_plan.json').read_text())['scored_labels']
M = {}
for a in GRID8:
    R = load(a)
    assert len(R) == 6, (a, len(R))
    M[a] = float(np.mean(list(per_seed(R, labs).values())))
best = max(M, key=M.get)
img = max((a for a in M if not a.startswith('vj')), key=M.get)
base = img if best.startswith('vj') else best
(OUT / 'p2').mkdir(parents=True, exist_ok=True)
(OUT / 'p2' / 'base.txt').write_text(f'{base}\n# p2_base.py: mean primary AUROC {json.dumps({a: round(v, 4) for a, v in M.items()})}; '
                                     f'best overall {best}\n')
print((OUT / 'p2' / 'base.txt').read_text())
