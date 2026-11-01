"""
v2.1 AMENDMENT 3 D1 - mice gap test (pre-registered in scripts/eci/v2/v21_train.py). CPU.
Per mice eval frame: N2 = the PASS B kept token positions (eci_codes/v2_0B/mice/videos/v1/<obs>.npz tok_pos, the full
N2 set), v5 = flag bit0 of the v2.1 store index at the same frame. N2-only = N2 minus v5. BRIDGE = an N2-only patch whose
Chebyshev distance is <= BW (= 2h / 16 = 4 patches, h = 32 px) to two distinct 8-connected v5 components.
-> results/vision/eci_v2_1/gap_mice.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
for p in ('', 'scripts/eci/v2', 'scripts/eci/v2_eval', 'scripts/eci/v2_prob'):
    sys.path.insert(0, str(REPO / p))
STORE = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_tokens/v2_full/mice/v2.1_m5/videos/v1')
PB = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_0B/mice/videos/v1')
G, BW, NEUR = 64, 4, {'nn': 803, 'nt': 1402}


def frame_sets(n2pos, v5pos):
    from scipy import ndimage
    g5 = np.zeros((G, G), bool)
    g5.flat[v5pos] = True
    lab, n = ndimage.label(g5, structure=np.ones((3, 3), bool))
    only = np.setdiff1d(n2pos, v5pos)
    bridge = np.zeros(G * G, bool)
    if n >= 2 and len(only):
        near = np.zeros((n, G * G), bool)
        for c in range(1, n + 1):
            d = ndimage.distance_transform_cdt(lab != c, metric='chessboard').ravel()
            near[c - 1] = d <= BW
        cnt = near[:, only].sum(0)
        bridge[only[cnt >= 2]] = True
    return only, bridge


def main():
    import eval as EV
    F = EV.frame_table('mice')
    proxy = np.load(REPO / 'results/vision/eci_v2_0B/gate/proxy_mice.npy')
    rows = []
    for obs, g in F.groupby('obs', sort=False):
        z = np.load(PB / f'{obs}.npz')
        nk = z['n_kept'].astype(np.int64)
        st = np.r_[0, np.cumsum(nk)]
        tp = z['tok_pos'].astype(np.int64)
        am = {j: z['argmax'][:, j] for j in NEUR.values()}
        cm = {j: z['codes_max'][:, j].astype(np.float32) for j in NEUR.values()}
        ix = np.load(STORE / obs / 'index.npz')
        kpos = {int(k): i for i, k in enumerate(ix['k'])}
        ptr, idx, flag = ix['ptr'], ix['idx'].astype(np.int64), ix['flag']
        for fi, k in zip(g.index.values, g.k.values):
            i = kpos[int(k)]
            v5 = idx[ptr[i]:ptr[i + 1]][(flag[ptr[i]:ptr[i + 1]] & 1) > 0]
            n2 = tp[st[k]:st[k + 1]]
            only, bridge = frame_sets(n2, v5)
            v5set = np.zeros(G * G, bool)
            v5set[v5] = True
            r = {'fi': int(fi), 'n_only': int(len(only)), 'n_bridge': int(bridge.sum()), 'n_v5': int(len(v5)), 'n_n2': int(len(n2))}
            for nm, j in NEUR.items():
                p = int(am[j][k])
                r[f'{nm}_code'] = float(cm[j][k])
                r[f'{nm}_peak_out'] = bool(p >= 0 and not v5set[p])
                r[f'{nm}_peak_bridge'] = bool(p >= 0 and bridge[p])
                r[f'{nm}_peak_valid'] = p >= 0
            rows.append(r)
    D = pd.DataFrame(rows).set_index('fi').loc[F.index.values]
    out = {}
    rng = np.random.default_rng(0)
    pxv = proxy
    edges = np.unique(np.nanquantile(pxv, np.linspace(0, 1, 11)))
    dec = np.clip(np.searchsorted(edges, np.nan_to_num(pxv, nan=-1), side='right') - 1, 0, len(edges) - 2)
    for lab in ('nn', 'nt'):
        y = (F[f'y_{lab}'].values > 0) & F[f'v_{lab}'].values
        neg_pool = (~(F[f'y_{lab}'].values > 0)) & F[f'v_{lab}'].values & np.isfinite(pxv)
        pos = np.flatnonzero(y & np.isfinite(pxv))
        negs = []
        for d in np.unique(dec[pos]):
            n_ = int((dec[pos] == d).sum())
            cand = np.flatnonzero(neg_pool & (dec == d))
            negs += list(rng.choice(cand, min(n_, len(cand)), replace=False))
        negs = np.array(negs)
        P_, N_ = D.iloc[pos], D.iloc[negs]
        r = {'n_pos': int(len(pos)), 'n_neg_matched': int(len(negs))}
        for q in ('n_only', 'n_bridge'):
            r[f'{q}_median_pos'] = float(P_[q].median())
            r[f'{q}_median_neg'] = float(N_[q].median())
        r['share_ge1_bridge_pos'] = float((P_.n_bridge >= 1).mean())
        r['share_ge1_bridge_neg'] = float((N_.n_bridge >= 1).mean())
        # (b) the PASS B best neuron of this label
        c = D[f'{lab}_code'].values
        top = np.argsort(-c)[:max(1, int(np.ceil(0.01 * len(c))))]
        for nm_, sel in (('top1pct', top), ('positives', pos)):
            S_ = D.iloc[sel]
            S_ = S_[S_[f'{lab}_peak_valid']]
            r[f'peak_outside_v5_share_{nm_}'] = float(S_[f'{lab}_peak_out'].mean()) if len(S_) else None
            r[f'peak_on_bridge_share_{nm_}'] = float(S_[f'{lab}_peak_bridge'].mean()) if len(S_) else None
            r[f'n_frames_{nm_}'] = int(len(S_))
        ok_a = r['share_ge1_bridge_pos'] - r['share_ge1_bridge_neg'] >= 0.10
        ok_b = (r['peak_outside_v5_share_top1pct'] or 0) >= 0.30
        r['verdict'] = 'SUPPORTED' if (ok_a and ok_b) else ('REFUTED' if (r['peak_outside_v5_share_top1pct'] or 0) < 0.10 else 'NOT SUPPORTED')
        out[lab] = r
        print(lab, json.dumps(r), flush=True)
    out['all_frames'] = {'n_v5_median': float(D.n_v5.median()), 'n_n2_median': float(D.n_n2.median()),
                         'n_only_median': float(D.n_only.median())}
    (REPO / 'results/vision/eci_v2_1/gap_mice.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
