"""
Static-animal edge case in the lab data (CPU, no encode): an animal that rests for most of a video leaks into the per-video
median background. Per video (ants: per 3000-frame background window, aggregated per video):
  (a) ghost area = share of the frame where the frozen dark rule is TRUE ON THE MEDIAN BACKGROUND (mask2 pack: grey(bg) <
      dark_abs AND q98 - grey(bg) > dark_rel, the series' own thresholds) - i.e. where the median is an animal;
  (b) overlap = share of 5 fps frames in which >= 1 N2-kept token (PASS B per-video token positions) covers a grid cell
      with >= 1 ghost pixel.
Frogs tadpoles: ghost area only (moving crops). Output results/vision/eci_v2_0B/ghost/ghost.json + videos > 2% ghost area.
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'scripts/eci/v2'))
OUT = REPO / 'results/vision/eci_v2_0B/ghost'
B = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_0B')


def grey(bg):
    t = bg.astype(np.float64)
    return (t[0] * 19595 + t[1] * 38470 + t[2] * 7471 + 32768) / 65536


def main():
    import os
    os.environ.setdefault('V2_OUT', str(REPO / 'results/vision/eci_v2_0B'))
    os.environ.setdefault('V2_BEE', str(B))
    import v20 as V
    res = {}
    for dom in ('mice', 'ants', 'frogs'):
        c = V.dcfg(dom)
        for sc in c['series']:
            m = sc['mask']
            da, dr = m.get('dark_abs', 60), m.get('dark_rel', 40)
            P = json.loads((REPO / sc['plan']).read_text())
            G = sc['S'] // 16
            rows = []
            for v in P['videos']:
                nw = int(np.ceil(v['n'] / 3000)) if dom == 'ants' else 1
                ga, gcells = [], []
                for w in range(nw):
                    f = V.bg_file(dom, sc['name'], v['obs'], w)
                    if not f.exists():
                        continue
                    z = np.load(f)
                    gb = grey(z['bg'])
                    gh = (gb < da) & ((z['q98'].astype(np.float64) - gb) > dr)
                    ga.append(float(gh.mean()))
                    if sc.get('geometry') != 'crop':
                        S = gh.shape[0]
                        gcells.append(gh.reshape(G, S // G, G, S // G).any((1, 3)).ravel())
                if not ga:
                    continue
                r = {'obs': v['obs'], 'ghost_area': float(np.mean(ga)), 'ghost_area_max_window': float(np.max(ga))}
                vf = B / dom / 'videos' / sc['name'] / f'{v["obs"]}.npz'
                if gcells and vf.exists():
                    zz = np.load(vf)
                    nk = zz['n_kept'].astype(np.int64)
                    st = np.r_[0, np.cumsum(nk)]
                    pos = zz['tok_pos'].astype(np.int64)
                    hit = np.zeros(len(nk), bool)
                    for k in range(len(nk)):
                        gc = gcells[min(k // 3000, len(gcells) - 1)] if dom == 'ants' else gcells[0]
                        hit[k] = gc[pos[st[k]:st[k + 1]]].any()
                    r['overlap_frames'] = float(hit.mean())
                rows.append(r)
            ga = np.array([r['ghost_area'] for r in rows])
            ov = np.array([r.get('overlap_frames', np.nan) for r in rows])
            res[f'{dom}/{sc["name"]}'] = {
                'n_videos': len(rows), 'ghost_area_median': float(np.median(ga)), 'ghost_area_p90': float(np.quantile(ga, 0.9)),
                'ghost_area_max': float(ga.max()), 'n_ghost_gt_2pct': int((ga > 0.02).sum()),
                'overlap_median': float(np.nanmedian(ov)) if np.isfinite(ov).any() else None,
                'overlap_p90': float(np.nanquantile(ov, 0.9)) if np.isfinite(ov).any() else None,
                'videos_gt_2pct': sorted([(r['obs'], round(r['ghost_area'], 4), round(r.get('overlap_frames', float('nan')), 3))
                                          for r in rows if r['ghost_area'] > 0.02], key=lambda x: -x[1])}
            print(sc['name'], dom, json.dumps({k: v for k, v in res[f'{dom}/{sc["name"]}'].items() if k != 'videos_gt_2pct'}), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'ghost.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
