"""Identity check of the ants V2F_N2 codes computed by eval.py codes_ants (full1 N2 tokens) against the v2_prob Tier B
codes of the same dictionaries (an independent encode of the same frames by scripts/eci/v2_prob/study.py), on the
(obs, k) frames both sets hold. -> results/vision/eci_v2_eval/ants_v2_identity.json"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
EC = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes')
F = pd.read_parquet(EC / 'v2_eval/ants/frames.parquet')
P = pd.read_parquet(REPO / 'results/vision/eci_v2_prob/ants/probe.parquet')
idx = pd.Series(np.arange(len(P)), index=pd.MultiIndex.from_arrays([P.obs.astype(str), P.k.astype(int)]))
r = idx.reindex(pd.MultiIndex.from_arrays([F.obs.values, F.k.values])).values
ok = np.isfinite(r)
mine_rows, ref_rows = np.flatnonzero(ok), r[ok].astype(np.int64)
out = {'n_common_frames': int(ok.sum())}
for d in ('V2F_N2_s900', 'V2F_N2_s905', 'V2F_N2_s919', 'v1_s0', 'v1_s5'):
    a = np.load(EC / f'v2_eval/ants/codes/{d}.npy', mmap_mode='r')[mine_rows].astype(np.float32)
    b = np.load(EC / f'v2_prob/ants/codes/{d}.npy', mmap_mode='r')[ref_rows].astype(np.float32)
    out[d] = {'fire_agree': float(((a > 0) == (b > 0)).mean()), 'mean_abs_diff': float(np.abs(a - b).mean()),
              'max_abs_diff': float(np.abs(a - b).max()), 'active_share_mine': float((a > 0).mean()),
              'active_share_ref': float((b > 0).mean())}
    print(d, out[d], flush=True)
(REPO / 'results/vision/eci_v2_eval/ants_v2_identity.json').write_text(json.dumps(out, indent=1))
