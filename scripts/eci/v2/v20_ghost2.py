"""
Pooled-background check for the lab videos with ghost area > 2% (v20_ghost.py). CPU, no encode, nothing changed in the
deployed mask. Group = videos of one recording session sharing camera and arena: mice = the same pool and odour session
(observation id without its phase suffix: Habituation / Test / Post); ants = the same video's background windows; frogs
juveniles = the same video. Pooled background = per-pixel median of the group's per-video median backgrounds (q98, mad:
median too); ghost rule recomputed on it. Per video: 8 frames (evenly spread, rng-free), the N2 token set with the
per-video pack vs the pooled pack (same frozen core cue): share of tokens added / removed, and the share of the pooled-N2
tokens OUTSIDE the stored superset (per-video N2 dilated by one token) - must be 0 for the store to contain the new mask.
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts/eci/v2'))
OUT = REPO / 'results/vision/eci_v2_0B/ghost'


def main():
    import os
    import torch
    import torch.nn.functional as F
    os.environ.setdefault('V2_OUT', str(REPO / 'results/vision/eci_v2_0B'))
    os.environ.setdefault('V2_BEE', '/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_0B')
    import v20 as V
    from src.eci.v2 import mask2 as M2
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    dev = torch.device('cpu')
    G0 = json.loads((OUT / 'ghost.json').read_text())
    res = {}
    for key, r in G0.items():
        dom, ser = key.split('/')
        bad = [x[0] for x in r['videos_gt_2pct']]
        if not bad or ser == 'tad':
            continue
        c = V.dcfg(dom)
        sc = V.R.series_of(c, ser)
        NS = V.make_series_class(dom)
        se = NS(sc, dev, load_encoder=False)
        group = (lambda o: o.rsplit('_', 1)[0]) if dom == 'mice' else (lambda o: o)
        allobs = [v['obs'] for v in se.videos]
        rows = []
        for o in bad:
            v = se.V[o]
            mem = [x for x in allobs if group(x) == group(o)]
            packs = [np.load(V.bg_file(dom, ser, x, 0)) for x in mem]
            pool = {k: np.median(np.stack([p[k].astype(np.float32) for p in packs]), 0) for k in ('bg', 'q98', 'mad_s15')}
            own = se.pack(o, 0)
            m = sc['mask']
            bgp = torch.from_numpy(pool['bg']).float()
            q98 = torch.from_numpy(pool['q98']).float()
            gb = M2.grey_f(bgp[None])[0]
            pp = {'bg': bgp, 'mad': torch.from_numpy(pool['mad_s15']).float(), 'q98': q98,
                  'ghost': (gb < m.get('dark_abs', 60)) & ((q98 - gb) > m.get('dark_rel', 40))}
            ks = np.linspace(0, v['n'] - 1, 8).round().astype(int)
            add = rem = n_old = n_new = outside = 0
            for k in ks:
                fr = se.run_job(**se.job(v, int(k), 1, threads=2))
                import src.eci.v2.pipeline as PL
                x, _ = PL.Series.frames(se, v, fr, [int(k)])
                core0 = se.mask.pixels(x, v['src_lo'] + np.array([int(k)]))
                k_old, _m = se.n2(x, core0, own)
                k_new, _m = se.n2(x, core0, pp)
                G = se.G
                dil = F.max_pool2d(k_old.float().view(1, 1, G, G), 3, 1, 1).flatten(1) > 0.5
                add += int((k_new & ~k_old).sum())
                rem += int((k_old & ~k_new).sum())
                n_old += int(k_old.sum())
                n_new += int(k_new.sum())
                outside += int((k_new & ~dil).sum())
            rows.append({'obs': o, 'group_size': len(mem), 'ghost_area_own': next(x[1] for x in r['videos_gt_2pct'] if x[0] == o),
                         'ghost_area_pooled': float(pp['ghost'].float().mean()), 'tokens_old': n_old, 'tokens_new': n_new,
                         'added_share': add / max(n_old, 1), 'removed_share': rem / max(n_old, 1),
                         'new_outside_superset_share': outside / max(n_new, 1)})
            print(key, json.dumps(rows[-1]), flush=True)
        res[key] = {'videos': rows, 'mean_added_share': float(np.mean([x['added_share'] for x in rows])),
                    'mean_removed_share': float(np.mean([x['removed_share'] for x in rows])),
                    'max_outside_superset_share': float(np.max([x['new_outside_superset_share'] for x in rows])),
                    'mean_ghost_area_own': float(np.mean([x['ghost_area_own'] for x in rows])),
                    'mean_ghost_area_pooled': float(np.mean([x['ghost_area_pooled'] for x in rows]))}
        print(key, json.dumps({k: v for k, v in res[key].items() if k != 'videos'}), flush=True)
    (OUT / 'pooled.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
