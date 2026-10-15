"""
CalMS21 ECI - ROLE SWAP (rules: eci_swap.sh): the pre-scoring label-drop decision. Label-only, no SAE output is read.
For split seeds 0-9 the video-level halves of the eval frames (calms21 eci_train.halves over all 7 labels, the exact
function the scorer uses) and, per label, the number of source videos with a positive in each half (on that label's own
frames). Rule: a pair label is DROPPED from the pair primary iff in any split seed either half holds < 3 source videos
with a positive. -> OUT/plan/halves.json (CPU; CALMS_OUT=results/vision/benchmark_calms21/swap).
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from calms_common import OUT, PLAN  # noqa: E402

_spec = importlib.util.spec_from_file_location('calms_eci_train', HERE / 'eci_train.py')
CT = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CT)
T1 = ['attack', 'investigation', 'mount']
T3 = ['sniff_face', 'approach', 'mount_attempt', 'intromission']

assert str(OUT).endswith('benchmark_calms21/swap'), OUT
E = pd.read_parquet(PLAN / 'train_frames.parquet')
res = {}
for s in range(10):
    h = CT.halves(E, s, T1 + T3)
    r = {}
    for b in T1 + T3:
        sub = np.ones(len(E), bool) if b in T1 else E['in_' + b].values.astype(bool)
        r[b] = [int(E[b][sub & (h == x)].groupby(E.video[sub & (h == x)]).any().sum()) for x in (0, 1)]
    res[f'split{s}'] = r
minc = {b: int(min(min(v[b]) for v in res.values())) for b in T1 + T3}
drop = [b for b in T3 if minc[b] < 3]
out = {'rule': 'drop a pair label iff any split seed has a half with < 3 source videos holding a positive',
       'min_pos_videos_per_half': minc, 'dropped': drop, 'pair_primary_labels': [b for b in T3 if b not in drop],
       'per_split': res}
(PLAN / 'halves.json').write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k != 'per_split'}, indent=1))
