"""
Expected size of a v2.0 token store whose token set = N2 dilated by r patches (r = 1 = the CTX / v1-style one-token
ring, 2, 4, 8), from the N2 token positions of existing stores (no encode): per frame, count of grid cells within
Chebyshev distance r of an N2 token (square (2r+1) dilation, clipped to the grid). Frames: mice = v2_prob probe store
(86k frames), ants = full1 1 fps store (all videos), frogs = the 1 fps N2 store (all videos). Size at 5 fps =
mean tokens / frame x every 5 fps frame of the domain x 2048 bytes.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
for p in ('', 'scripts/eci/v2', 'scripts/eci/v2_behav', 'scripts/eci/v2_diag'):
    sys.path.insert(0, str(REPO / p))
EC = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes')
FRAMES5 = {'mice': 2_592_000 + 1_296_000, 'ants': 132_000 + 963_000 + 729_000, 'frogs': 630_154 + 2_125_860}


def dil_counts(tf, pos, G, nF, rs=(1, 2, 4, 8), every=1):
    from scipy import ndimage
    o = np.argsort(tf, kind='stable')
    tf, pos = tf[o], pos[o]
    st = np.r_[0, np.cumsum(np.bincount(tf, minlength=nF))]
    out = {r: [] for r in (0,) + tuple(rs)}
    for f in range(0, nF, every):
        p = pos[st[f]:st[f + 1]]
        g = np.zeros((int(G[f]), int(G[f])), bool)
        g.flat[p] = True
        out[0].append(int(g.sum()))
        for r in rs:
            out[r].append(int(ndimage.maximum_filter(g, size=2 * r + 1).sum()))
    return {r: float(np.mean(v)) for r, v in out.items()}


def main():
    res = {}
    import behav as BH
    BH.BEE = EC / 'v2_prob'
    St = BH.Store('mice', 'v2', 'dense', 1024)
    k = St.keep('N2')
    res['mice'] = dil_counts(St.tf[k], St.pos[k], St.F.G.values, len(St.F), every=4)
    import diag as D
    T2 = D.T2
    import os
    os.environ.setdefault('LOCAL_DIR', '/tmp')
    old = T2.BEE
    T2.BEE = D.BEE
    T2.KINDS['ants'].setdefault('full1', {'series': D.F1_SERIES['ants'][0]})
    U = T2.Union('ants', 'full1')
    T2.BEE = old
    m = U.arm('N2')
    Gs = U.F.series.map({'v2': 48, 'v3': 52, 'v4': 51}).values
    res['ants'] = dil_counts(U.tf[m], U.pos[m], Gs, len(U.F), every=5)
    sh = sorted((EC / 'v2_0/frogs/store').glob('s[0-9]*'))
    tfs, poss, Fs, o = [], [], [], 0
    for p in sh:
        F_ = pd.read_parquet(p / 'frames.parquet')
        tfs.append(np.load(p / 'tok_frame.npy') + o)
        poss.append(np.load(p / 'tok_pos.npy').astype(np.int64))
        Fs.append(F_)
        o += len(F_)
    F = pd.concat(Fs, ignore_index=True)
    res['frogs'] = dil_counts(np.concatenate(tfs), np.concatenate(poss), F.series.map({'juv': 50, 'tad': 24}).values, len(F), every=5)
    out = {}
    for dom, r in res.items():
        out[dom] = {f'r{k}': {'tokens_per_frame': v, 'TB_at_5fps': v * FRAMES5[dom] * 2048 / 1e12} for k, v in r.items()}
    out['total_TB'] = {f'r{k}': sum(out[d][f'r{k}']['TB_at_5fps'] for d in res) for k in (0, 1, 2, 4, 8)}
    (REPO / 'results/vision/eci_v2_0B/dilsize.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
