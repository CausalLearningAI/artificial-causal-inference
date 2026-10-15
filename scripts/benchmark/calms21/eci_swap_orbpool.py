"""
CalMS21 ECI - ROLE SWAP (rules: eci_swap.sh): the reduced SAE pool of the orbit arm. After the base encode of the SAE
pool (store set 'eval' = task1 TEST videos), copy the frames parquet / 32-grid mask rows of the plan's orbit_pool frames
(4,000 train + 1,000 calib) into a separate store, so eci_p2_encode.py --parts orbit encodes only those frames.
CPU, seconds. CALMS_OUT=results/vision/benchmark_calms21/swap, CALMS_TOK=<main swap store>; argv[1] = the orbit store.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calms_common import OUT, PLAN, TOK  # noqa: E402

DST = Path(sys.argv[1])
CFG = 'd3l_1024_L20'
assert str(OUT).endswith('benchmark_calms21/swap') and 'pairpush' in str(TOK) and DST != TOK
DST.mkdir(parents=True, exist_ok=True)
P = pd.read_parquet(PLAN / 'eval_frames.parquet')
keep = set(map(tuple, P.loc[P.orbit_pool, ['video', 'i']].values.tolist()))
tot = 0
for f in sorted(TOK.glob('eval_*of*_frames.parquet')):
    base = f.name[:-len('_frames.parquet')]
    RF = pd.read_parquet(f)
    pos = np.load(TOK / f'{base}_pos32.npy')
    st = np.r_[0, np.cumsum(RF.n32.values)]
    m = np.array([(q, i) in keep for q, i in zip(RF.video, RF.i)])
    R2 = RF[m].reset_index(drop=True)
    p2 = np.concatenate([pos[st[r]:st[r + 1]] for r in np.flatnonzero(m)]) if m.any() else pos[:0]
    R2.to_parquet(DST / f.name)
    np.save(DST / f'{base}_pos32.npy', p2)
    (DST / f'{base}_{CFG}.json').write_text(json.dumps({'note': 'orbit-pool subset index (eci_swap_orbpool.py)',
                                                         'frames': int(m.sum()), 'of': int(len(RF))}))
    tot += int(m.sum())
    print(base, int(m.sum()), 'of', len(RF))
assert tot == int(P.orbit_pool.sum()), tot
print('total', tot)
