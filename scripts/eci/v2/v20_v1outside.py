"""
What are v1's exact-mask (V1M_d2) tokens that fall OUTSIDE the v2.0 store (N2 dilated by r = 1, and r = 4)? CPU, the V1M
stores (mice diag frames; ants step-2 scored v2 / v3 frames), every 3rd frame. Per outside token:
  (a) off-animal (lab leak measure): fr_any < 0.05, i.e. < 5% of its pixels in the frozen cue or the raw change map
      (mask2 d > T_LO);
  (b) contains animal pixels N2 misses: fr_new > 0 (>= 1 mask2-cue animal pixel) although the token is outside the
      dilated N2 set (an N2 false negative of the token rule);
  (c) Chebyshev distance (tokens) to the nearest N2 token.
Example frames: v1 mask (blue), N2 (green), v1 tokens outside N2 dilated by 1 (red) -> results/vision/eci_v2_0B/ghost/.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
for p in ('', 'scripts/eci/v2', 'scripts/eci/v2_mask', 'scripts/eci/v2_mask2', 'scripts/eci/nnnt', 'scripts/eci'):
    sys.path.insert(0, str(REPO / p))
V1M = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_v1mask')
OUT = REPO / 'results/vision/eci_v2_0B/ghost'
SETS = (('mice', 'diag', 64, 'v1'), ('ants', 'scored_v2', 48, 'v2'), ('ants', 'scored_v3', 52, 'v3'))


def main():
    from scipy import ndimage
    res, examples = {}, {}
    for dom, kind, G, ser in SETS:
        acc = {r: {'n': 0, 'off': 0, 'miss': 0, 'dist': []} for r in (1, 4)}
        ex = []
        for sp in sorted((V1M / dom / kind).glob('s[0-9]*')):
            F = pd.read_parquet(sp / 'frames.parquet')
            tf, pos = np.load(sp / 'tok_frame.npy'), np.load(sp / 'tok_pos.npy').astype(np.int64)
            fl = np.load(sp / 'flags.npy')
            fnew = np.load(sp / 'fr_new.npy').astype(np.float32)
            fany = np.load(sp / 'fr_any.npy').astype(np.float32)
            n2 = (fnew >= 0.25) | (np.load(sp / 'n_app.npy') >= 16)
            o = np.argsort(tf, kind='stable')
            st = np.r_[0, np.cumsum(np.bincount(tf, minlength=len(F)))]
            pos_, n2_, d2_, fn_, fa_ = pos[o], n2[o], ((fl & 2) > 0)[o], fnew[o], fany[o]
            for f in range(0, len(F), 3):
                a, b = st[f], st[f + 1]
                g = np.zeros((G, G), bool)
                g.flat[pos_[a:b][n2_[a:b]]] = True
                if not g.any():
                    continue
                dist = ndimage.distance_transform_cdt(~g, metric='chessboard').ravel()
                sel = d2_[a:b]
                p, fnw, fan = pos_[a:b][sel], fn_[a:b][sel], fa_[a:b][sel]
                for r in (1, 4):
                    out = dist[p] > r
                    A = acc[r]
                    A['n'] += int(out.sum())
                    A['off'] += int((out & (fan < 0.05)).sum())
                    A['miss'] += int((out & (fnw > 0)).sum())
                    A['dist'] += dist[p][out].tolist()[:200]
                if len(ex) < 4 and f % 300 == 0:
                    ex.append({'obs': str(F.obs.iloc[f]), 'k': int(F.k.iloc[f]), 'v1': p.tolist(),
                               'n2': pos_[a:b][n2_[a:b]].tolist(), 'out1': p[dist[p] > 1].tolist()})
        res[f'{dom}/{kind}'] = {f'r{r}': {'outside_tokens': A['n'], 'off_animal_share': A['off'] / max(A['n'], 1),
                                          'animal_px_missed_by_N2_share': A['miss'] / max(A['n'], 1),
                                          'dist_q50_q90_max': [float(np.quantile(A['dist'], q)) for q in (0.5, 0.9)] + [float(np.max(A['dist']))] if A['dist'] else None}
                                for r, A in acc.items()}
        examples[(dom, kind, G, ser)] = ex
        print(dom, kind, json.dumps(res[f'{dom}/{kind}']), flush=True)
    # renders
    import torch
    from PIL import Image
    import v20 as V
    from src.eci.v2.pipeline import Series
    paths = []
    for (dom, kind, G, ser), ex in examples.items():
        se = Series(V.R.series_of(V.dcfg(dom), ser), torch.device('cpu'), load_encoder=False)
        for e in ex:
            v = se.V[e['obs']]
            x, _ = se.frames(v, se.run_job(**se.job(v, e['k'], 1, threads=2)), [e['k']])
            im = x[0].permute(1, 2, 0).numpy().astype(np.float32)
            S = im.shape[0]
            ps = S // G
            lay = np.zeros_like(im)
            al = np.zeros(im.shape[:2], np.float32)
            for key, col in (('v1', (0, 90, 255)), ('n2', (0, 220, 0)), ('out1', (255, 0, 0))):
                for q in e[key]:
                    y0, x0 = (q // G) * ps, (q % G) * ps
                    lay[y0:y0 + ps, x0:x0 + ps] = col
                    al[y0:y0 + ps, x0:x0 + ps] = 0.45
            comp = (im * (1 - al[..., None]) + lay * al[..., None]).astype(np.uint8)
            pth = OUT / f'v1out_{dom}_{kind}_{e["obs"]}_{e["k"]}.png'
            Image.fromarray(np.concatenate([im.astype(np.uint8), comp], 1)).resize((1024, 512)).save(pth)
            paths.append(str(pth))
    res['examples'] = paths
    (OUT / 'v1outside.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(paths))


if __name__ == '__main__':
    main()
