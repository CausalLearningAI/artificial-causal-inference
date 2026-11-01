"""
v2.1 AMENDMENT 2 measurement: share of the REAL N2 tokens (old all-token probe stores, which hold every N2 token of their
frames) that fall outside U_t (the v2.1 store's token set = union of mask v5 over t-2..t+2) at the same frame and
position. Frames: the v2_prob probe stores (mice = the eval windows; ants = Tier B v2 / v3) and the behaviour-suite probe
stores (v2_behav, mice / ants). Also the N2 and U token counts per frame. CPU. -> results/vision/eci_v2_1/n2cover.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
for p in ('', 'scripts/eci/v2', 'scripts/eci/v2_behav'):
    sys.path.insert(0, str(REPO / p))
EC = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes')
STORE = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_tokens/v2_full')


def main():
    import behav as BH
    res = {}
    for study, root, pq in (('v2_prob', EC / 'v2_prob', 'results/vision/eci_v2_prob/{d}/probe.parquet'),
                            ('v2_behav', EC / 'v2_behav', 'results/vision/eci_v2_behav/{d}/probe.parquet')):
        for dom in ('mice', 'ants'):
            BH.BEE = root
            St = BH.Store(dom, 'v2', 'dense', 1024)
            Pr = pd.read_parquet(REPO / pq.format(d=dom))
            vmap = Pr.groupby('vid').agg(series=('series', 'first'), obs=('obs', 'first'))
            keep = St.keep('N2')
            tf, pos = St.tf[keep], St.pos[keep]
            o = np.argsort(tf, kind='stable')
            tf, pos = tf[o], pos[o]
            st = np.r_[0, np.cumsum(np.bincount(tf, minlength=len(St.F)))]
            n_n2 = n_out = n_u = nfr = 0
            per = []
            for vid, g in St.F.groupby('vid', sort=False):
                ser, obs = vmap.loc[vid, 'series'], vmap.loc[vid, 'obs']
                ser = 'v4' if ser == 'v4_infested' else ser
                d = STORE / dom / 'v2.1_m5' / 'videos' / ser / obs
                z = np.load(d / 'index.npz')
                kpos = {int(k): i for i, k in enumerate(z['k'])}
                ptr, idx = z['ptr'], z['idx'].astype(np.int64)
                for f, k in zip(g.index.values[::5], g.k.values[::5]):
                    i = kpos.get(int(k))
                    if i is None:
                        continue
                    u = set(idx[ptr[i]:ptr[i + 1]].tolist())
                    p = pos[st[f]:st[f + 1]]
                    out = int(sum(1 for q in p.tolist() if q not in u))
                    n_n2 += len(p)
                    n_out += out
                    n_u += len(u)
                    nfr += 1
                    if len(p):
                        per.append(out / len(p))
            res[f'{study}/{dom}'] = {'frames': nfr, 'N2_tokens_per_frame': n_n2 / max(nfr, 1),
                                     'U_tokens_per_frame': n_u / max(nfr, 1), 'N2_outside_U_share': n_out / max(n_n2, 1),
                                     'per_frame_outside_share_q50_q90': [float(np.quantile(per, q)) for q in (0.5, 0.9)] if per else None}
            print(study, dom, json.dumps(res[f'{study}/{dom}']), flush=True)
    (REPO / 'results/vision/eci_v2_1/n2cover.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
