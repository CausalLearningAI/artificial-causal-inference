"""
(a) Does the v2.0 token store (N2 dilated by one token) contain v1's EXACT dilated mask (V1M_d2 of scripts/eci/v2_diag/
v1mask.py: v1's fg448 / ants rule with its DINOv2 backgrounds, 1-patch dilation, mapped to the native DINOv3 grid)?
Frames = the V1M stores (mice diag frames; ants step-2 scored v2 / v3 frames). Per frame: N2 = stored tokens with
fr_new >= 0.25 or n_app >= 16 (the store holds M0 | d2 | d3 tokens only, so N2 tokens outside them are missing and the
reported share is an UPPER bound), superset = N2 dilated by one token; reported = share of d2 tokens outside the superset.
(b) Images of mice median backgrounds with the ghost region (dark rule on the median) for the 4 highest-ghost videos.
CPU.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'scripts/eci/v2'))
V1M = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_v1mask')
OUT = REPO / 'results/vision/eci_v2_0B/ghost'


def contain():
    from scipy import ndimage
    res = {}
    for dom, kinds in (('mice', {'diag': 64}), ('ants', {'scored_v2': 48, 'scored_v3': 52})):
        for kind, G in kinds.items():
            out_n = tot = 0
            per = []
            for sp in sorted((V1M / dom / kind).glob('s[0-9]*')):
                F = pd.read_parquet(sp / 'frames.parquet')
                tf, pos = np.load(sp / 'tok_frame.npy'), np.load(sp / 'tok_pos.npy').astype(np.int64)
                fl = np.load(sp / 'flags.npy')
                n2 = (np.load(sp / 'fr_new.npy').astype(np.float32) >= 0.25) | (np.load(sp / 'n_app.npy') >= 16)
                d2 = (fl & 2) > 0
                o = np.argsort(tf, kind='stable')
                st = np.r_[0, np.cumsum(np.bincount(tf, minlength=len(F)))]
                tf_, pos_, n2_, d2_ = tf[o], pos[o], n2[o], d2[o]
                for f in range(0, len(F), 3):
                    a, b = st[f], st[f + 1]
                    g = np.zeros(G * G, bool)
                    g[pos_[a:b][n2_[a:b]]] = True
                    sup = ndimage.maximum_filter(g.reshape(G, G), size=3).ravel()
                    dp = pos_[a:b][d2_[a:b]]
                    if len(dp):
                        k = int((~sup[dp]).sum())
                        out_n += k
                        tot += len(dp)
                        per.append(k / len(dp))
            res[f'{dom}/{kind}'] = {'v1_tokens': tot, 'outside_share': out_n / max(tot, 1),
                                    'frames': len(per), 'frames_outside_gt_5pct': float(np.mean(np.array(per) > 0.05))}
            print(dom, kind, json.dumps(res[f'{dom}/{kind}']), flush=True)
    return res


def images():
    from PIL import Image
    import v20 as V
    G = json.loads((OUT / 'ghost.json').read_text())
    paths = []
    for key in ('mice/v1', 'mice/v2'):
        dom, ser = key.split('/')
        for o, ga, ov in G[key]['videos_gt_2pct'][:2]:
            z = np.load(V.bg_file(dom, ser, o, 0))
            bg = z['bg'].transpose(1, 2, 0).astype(np.uint8)
            t = z['bg'].astype(np.float64)
            gb = (t[0] * 19595 + t[1] * 38470 + t[2] * 7471 + 32768) / 65536
            gh = (gb < 60) & ((z['q98'].astype(np.float64) - gb) > 40)
            im = bg.copy()
            im[gh] = (0.4 * im[gh] + 0.6 * np.array([255, 0, 128])).astype(np.uint8)
            p = OUT / f'ghost_{ser}_{o}.png'
            Image.fromarray(np.concatenate([bg, im], 1)).resize((1024, 512)).save(p)
            paths.append(str(p))
    return paths


def main():
    res = {'v1_contain': contain(), 'images': images()}
    (OUT / 'v1contain.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res['images']))


if __name__ == '__main__':
    main()
