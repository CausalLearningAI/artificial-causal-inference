"""
Why does N2 miss pale tails on light bedding? (CPU diagnostic, no encode.) For a lab frame, recompute every N2
intermediate exactly as scripts/eci/v2/v20.py N2Series.n2 / src/eci/v2/mask2.hyst_pixels do, and render zoomed panels:
frame | d = max_c |blur(x) - blur(bg)| | per-pixel low threshold tl = max(per-frame thr, K_MAD * mad) | ghost guard
(dilated) | cand (d > tl, minus ghost) | core | N2 pixels after the REACH-capped growth | appendage px | kept tokens.
Also prints, inside a hand-given ROI box (the tail), the share of pixels with d > T_LO, d > per-frame thr, d > K_MAD mad,
cand, grown, and the appendage pixel count per token.
    python mask3_diag.py --dom mice --ser v1 --obs wt_ash1l_m_1_S_H --k 6001 --crop y0,y1,x0,x1 [--roi y0,y1,x0,x1]
-> results/vision/eci_v2_0B/mask_approval/diag/<obs>_<k>.png + .json
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts/eci/v2'))
OUT = REPO / 'results/vision/eci_v2_0B/mask_approval/diag'


def load_lab(dom, ser, obs, k, dev):
    os.environ.setdefault('V2_OUT', str(REPO / 'results/vision/eci_v2_0B'))
    os.environ.setdefault('V2_BEE', '/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_0B')
    import v20 as V
    import src.eci.v2.pipeline as PL
    sc = V.R.series_of(V.dcfg(dom), ser)
    se = V.make_series_class(dom)(sc, dev, load_encoder=False)
    v = se.V[obs]
    fr = se.run_job(**se.job(v, int(k), 1, threads=2))
    x, _ = PL.Series.frames(se, v, fr, [int(k)])
    core0 = se.mask.pixels(x, v['src_lo'] + np.array([int(k)]))
    return se, v, x, core0, se.pack(obs, V.window_of(dom, k))


def main():
    import torch
    import torch.nn.functional as F
    from src.eci.v2 import mask2 as M2
    ap = argparse.ArgumentParser()
    ap.add_argument('--dom', default='mice')
    ap.add_argument('--ser', default='v1')
    ap.add_argument('--obs', default='wt_ash1l_m_1_S_H')
    ap.add_argument('--k', type=int, default=6001)
    ap.add_argument('--crop', default='680,940,440,840')
    ap.add_argument('--roi', default='')
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    dev = torch.device('cpu')
    se, v, x, core0, P = load_lab(a.dom, a.ser, a.obs, a.k, dev)
    h = se.h
    bg, mad, ghost = P['bg'], P['mad'], P['ghost']
    xf = M2.gauss_blur(x.float(), M2.SIGMA)
    bf = M2.gauss_blur(bg[None].float(), M2.SIGMA)
    d = (xf - bf).abs().amax(1)
    far = M2._maxpool(core0.float()[:, None], max(1, int(round(2 * h))))[:, 0] < 0.5
    vv = d[0][far[0]][::7]
    med = vv.median()
    thr = torch.clamp(M2.K_NOISE * 1.4826 * (vv - med).abs().median(), min=M2.T_LO)
    tmad = M2.K_MAD * mad
    tl = torch.maximum(thr, tmad)
    bad = M2._maxpool(ghost.float()[None, None], max(1, int(round(h))))[0, 0] > 0.5
    cand = (d[0] > tl) & ~bad
    r = se.n2(x, core0, P)
    full = M2.hyst_pixels(x, bg, core0, h, t_lo=M2.T_LO, t_hi=M2.T_HI, k_noise=M2.K_NOISE, reach_h=M2.REACH_H,
                          sigma=M2.SIGMA, strong=True, mad=mad, k_mad=M2.K_MAD, bad=ghost)
    body, app = M2.appendage(full['pix'], h)
    keep = r[0][0].view(se.G, se.G).numpy()
    y0, y1, x0, x1 = [int(t) for t in a.crop.split(',')]
    S = x.shape[-1]
    info = {'obs': a.obs, 'k': a.k, 'h': h, 'S': S, 'per_frame_thr': float(thr), 'noise_median_d': float(med),
            'tmad_quantiles_far_5_50_95': np.quantile(tmad[far[0]].numpy(), [.05, .5, .95]).tolist(),
            'ghost_area': float(ghost.float().mean()), 'kept_tokens': int(keep.sum())}
    if a.roi:
        ry0, ry1, rx0, rx1 = [int(t) for t in a.roi.split(',')]
        sl = (slice(ry0, ry1), slice(rx0, rx1))
        dd = d[0][sl]
        g = M2.grey_f(x)[0][sl]
        info['roi'] = {'px': int(dd.numel()), 'grey_q50_q90': np.quantile(g.numpy(), [.5, .9]).tolist(),
                       'd_q50_q90_q99': np.quantile(dd.numpy(), [.5, .9, .99]).tolist(),
                       'd_gt_TLO': float((dd > M2.T_LO).float().mean()), 'd_gt_thr': float((dd > thr).float().mean()),
                       'd_gt_tmad': float((dd > tmad[sl]).float().mean()), 'tmad_q50': float(tmad[sl].median()),
                       'cand': float(cand[sl].float().mean()), 'ghost_guard': float(bad[sl].float().mean()),
                       'pix': float(full['pix'][0][sl].float().mean()), 'grown': float(full['grown'][0][sl].float().mean()),
                       'app_px': int(app[0][sl].sum()), 'body_px': int(body[0][sl].sum()),
                       'core0_px': int(core0[0][sl].sum())}
    print(json.dumps(info, indent=1), flush=True)
    from PIL import Image, ImageDraw

    def tile(arr, name, cmap=None):
        a_ = arr[y0:y1, x0:x1]
        if a_.dtype == bool:
            im = np.repeat((a_ * 255).astype(np.uint8)[..., None], 3, -1)
        elif a_.ndim == 2:
            im = np.repeat(np.clip(a_ / max(float(np.quantile(a_, 0.995)), 1e-6) * 255, 0, 255).astype(np.uint8)[..., None], 3, -1)
        else:
            im = a_.astype(np.uint8)
        t = Image.fromarray(im)
        ImageDraw.Draw(t).text((4, 4), name, fill=(255, 255, 0))
        if a.roi:
            ImageDraw.Draw(t).rectangle([rx0 - x0, ry0 - y0, rx1 - x0, ry1 - y0], outline=(255, 0, 0))
        return t
    fr = x[0].permute(1, 2, 0).numpy()
    K = np.kron(keep, np.ones((16, 16), bool))
    ov = fr.astype(np.float32).copy()
    ov[K] = 0.6 * ov[K] + 0.4 * np.array([0, 255, 0])
    tiles = [tile(fr, 'frame'), tile(d[0].numpy(), 'd'), tile(tl.numpy(), f'tl (thr {float(thr):.0f})'),
             tile((d[0] > M2.T_LO).numpy(), 'd > T_LO 35'), tile((d[0] > tl).numpy(), 'd > tl'),
             tile(bad.numpy(), 'ghost guard'), tile(cand.numpy(), 'cand'), tile(full['core'][0].numpy(), 'core'),
             tile(full['pix'][0].numpy(), 'N2 pix'), tile(app[0].numpy(), 'appendage'), tile(body[0].numpy(), 'body'),
             tile(ov, f'N2 tokens {int(keep.sum())}')]
    w, hh = tiles[0].size
    sh = Image.new('RGB', (4 * w, 3 * hh))
    for i, t in enumerate(tiles):
        sh.paste(t, ((i % 4) * w, (i // 4) * hh))
    OUT.mkdir(parents=True, exist_ok=True)
    sh.save(OUT / f'{a.obs}_{a.k}.png')
    (OUT / f'{a.obs}_{a.k}.json').write_text(json.dumps(info, indent=1))


if __name__ == '__main__':
    main()
