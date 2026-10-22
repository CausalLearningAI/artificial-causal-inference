"""
DEVELOPMENT ONLY (no score of this file decides anything): visual / numeric look at the thin-part cue
(src/eci/v2/mask2.py) on DEV data: lab training videos (mice v1, ants v2, ants v3), CalMS21 task-1 TRAIN videos,
MABe22 mouse_userTrain clips. Benchmark keypoints are drawn only for looking. Held-out benchmark data (CalMS21 test,
MABe22 submission) is never touched here.

    python scripts/eci/v2_mask2/explore.py --src mice,ants_v2,ants_v3,calms21,mabe22 [--tlo 20 --khi 40 ...]
-> OUT/explore/<tag>_<src>.jpg, OUT/explore/<tag>.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (BEE, CAL_PARTS, N_BG, OUT, CalKP, cal_decode, dark_cue, log, mab_decode, mab_keypoints,  # noqa: E402
                    mab_kp, q98, series_cfg)
from src.eci.v2 import mask2 as M2  # noqa: E402


def body_h(cores):
    from scipy import ndimage
    hm = []
    for px in cores:
        lab, n = ndimage.label(px)
        if n == 0:
            continue
        edt = ndimage.distance_transform_edt(px)
        area = np.bincount(lab.ravel(), minlength=n + 1)
        mx = ndimage.maximum(edt, lab, np.arange(1, n + 1))
        big = area[1:].max()
        hm += [float(m) for m, a in zip(mx, area[1:]) if a >= max(30, 0.25 * big)]  # animal-sized components only
    return float(np.median(hm)) if hm else 4.0


def lab_items(dom, ser, n_vid, n_fr, dev):
    from src.eci.v2.pipeline import Series
    se = Series(series_cfg(dom, ser), dev, load_encoder=False)
    vids = [v for v in se.videos if v.get('role') == 'train']
    g = np.random.default_rng(1)
    out = []
    for v in g.choice(vids, n_vid, replace=False):
        ks_bg = np.unique(np.linspace(0, v['n'] - 1, N_BG).round().astype(int))
        xs = []
        for k in ks_bg:
            x, _ = se.frames(v, se.run_job(**se.job(v, int(k), 1)), [int(k)])
            xs.append(x)
        xb = torch.cat(xs)
        bg = M2.median_background(xb)
        ks = np.sort(g.choice(v['n'], n_fr, replace=False))
        x = torch.cat([se.frames(v, se.run_job(**se.job(v, int(k), 1)), [int(k)])[0] for k in ks])
        core = se.mask.pixels(x, v['src_lo'] + ks)
        out.append(dict(name=f'{v["obs"]}', x=x, bg=bg, core=core, kp=None, ks=ks, xb=xb))
    return out


def cal_items(n_vid, n_fr, dev):
    K = CalKP('train')
    g = np.random.default_rng(1)
    out = []
    for j in g.choice(len(K.names), n_vid, replace=False):
        nm, T = K.names[j], int(K.lens[j])
        xb = torch.from_numpy(cal_decode(nm, 'train', np.linspace(0, T - 1, N_BG).round().astype(int))).to(dev)
        bg, pb = M2.median_background(xb), q98(xb)
        ks = np.sort(g.choice(T, n_fr, replace=False))
        x = torch.from_numpy(cal_decode(nm, 'train', ks)).to(dev)
        out.append(dict(name=nm, x=x, bg=bg, core=dark_cue(x, pb), kp=[K.get(nm, int(k)) for k in ks], ks=ks, xb=xb))
    return out


def mab_items(n_vid, n_fr, dev):
    seqs = mab_keypoints('mouse_userTrain')
    g = np.random.default_rng(1)
    out = []
    for c in g.choice(sorted(seqs), n_vid, replace=False):
        a = torch.from_numpy(mab_decode('mouse_userTrain', c)).to(dev)
        bg = M2.median_background(a[np.linspace(0, len(a) - 1, N_BG).round().astype(int)])
        pb = q98(a[np.linspace(0, len(a) - 1, 30).round().astype(int)])
        ks = np.sort(g.choice(np.arange(30, 1790), n_fr, replace=False))
        x = a[ks + 1]
        out.append(dict(name=c, x=x, bg=bg, core=dark_cue(x, pb), kp=[mab_kp(seqs[c], int(k)) for k in ks], ks=ks,
                        xb=a[np.linspace(0, len(a) - 1, N_BG).round().astype(int)]))
    return out


def panel(it, res, i, h, tile=448):
    from PIL import Image, ImageDraw
    x = it['x'][i].cpu().numpy().transpose(1, 2, 0).astype(np.float32)
    S = x.shape[0]
    core, pix, grown = (res[k][i].cpu().numpy() for k in ('core', 'pix', 'grown'))
    d = res['d'][i].cpu().numpy()
    a = x.copy()
    b = x.copy()
    b[it['core'][i].cpu().numpy()] = 0.5 * b[it['core'][i].cpu().numpy()] + 0.5 * np.array([0, 255, 0])
    c = np.stack([np.clip(d / 60, 0, 1) * 255] * 3, -1)
    e = x.copy()
    e[core] = 0.5 * e[core] + 0.5 * np.array([0, 255, 0])
    e[grown] = np.array([255, 0, 255])
    ys, xs = np.nonzero(core)
    if len(ys):
        cy, cx = int(np.median(ys)), int(np.median(xs))
    else:
        cy = cx = S // 2
    w = max(48, S // 4)
    y0, x0 = int(np.clip(cy - w // 2, 0, S - w)), int(np.clip(cx - w // 2, 0, S - w))
    z = e[y0:y0 + w, x0:x0 + w]
    za = a[y0:y0 + w, x0:x0 + w]
    tiles = []
    for im, title in ((a, f'{it["name"]} k{it["ks"][i]}'), (b, 'frozen dark cue'), (c, f'd thr {float(res["thr"][i]):.1f}'),
                      (e, f'new core g / grown m h{h:.1f}'), (za, 'zoom raw'), (z, 'zoom new')):
        t = Image.fromarray(im.clip(0, 255).astype(np.uint8)).resize((tile, tile), Image.NEAREST if im is z or im is za
                                                                   else Image.BILINEAR)
        dr = ImageDraw.Draw(t)
        if it['kp'] is not None and im is a:
            for m in it['kp'][i]:
                for p in m:
                    if np.isfinite(p).all():
                        q = p * tile / S
                        dr.ellipse([q[0] - 2, q[1] - 2, q[0] + 2, q[1] + 2], fill=(255, 0, 0))
        dr.text((4, 4), title, fill=(255, 255, 0))
        tiles.append(t)
    row = Image.new('RGB', (tile * len(tiles), tile))
    for j, t in enumerate(tiles):
        row.paste(t, (j * tile, 0))
    return row


def load_src(src, dev):
    cf = BEE / 'explore_cache' / f'{src}.pt'
    if cf.exists():
        items = torch.load(cf)
    else:
        if src == 'mice':
            items = lab_items('mice', 'v1', 3, 3, torch.device('cpu'))
        elif src.startswith('ants'):
            items = lab_items('ants', src.split('_')[1], 2, 3, torch.device('cpu'))
        elif src == 'calms21':
            items = cal_items(3, 3, torch.device('cpu'))
        else:
            items = mab_items(4, 2, torch.device('cpu'))
        items = [{k: (v.cpu() if torch.is_tensor(v) else v) for k, v in it.items()} for it in items]
        cf.parent.mkdir(parents=True, exist_ok=True)
        torch.save(items, cf)
    return [{k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in it.items()} for it in items]


def main():
    from PIL import Image
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='mice,ants_v2,ants_v3,calms21,mabe22')
    ap.add_argument('--configs', default='{"e0": {}}', help='json {tag: hyst_pixels kwargs}')
    a = ap.parse_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    (OUT / 'explore').mkdir(parents=True, exist_ok=True)
    cfgs = json.loads(Path(a.configs).read_text() if a.configs.endswith('.json') else a.configs)
    stats = {}
    for src in a.src.split(','):
        items = load_src(src, dev)
        h = body_h([c.cpu().numpy() for it in items for c in it['core']])
        for it in items:
            it['mads'] = {sg: M2.background_mad(it['xb'], it['bg'], sigma=sg) for sg in (1.0, 1.5, 2.0)}
            it['q02'], it['q98'] = M2.grey_quantiles(it['xb'])
            gb = M2.grey_f(it['bg'][None])[0]
            da = 256 if src.startswith('ants') else 60
            it['ghost'] = (gb < da) & ((it['q98'] - gb) > 40)
        for tag, kw in cfgs.items():
            kw = dict(kw)
            use_mad = kw.pop('use_mad', False)
            use_q = kw.pop('use_q', 'none')
            rows, st = [], []
            for it in items:
                res = M2.hyst_pixels(it['x'], it['bg'], it['core'], h,
                                     mad=it['mads'][kw.get('sigma', M2.SIGMA)] if use_mad else None,
                                     q_lo=it['q02'] if use_q in ('lo', 'both') else None,
                                     q_hi=it['q98'] if use_q in ('hi', 'both') else None,
                                     bad=it['ghost'] if use_q == 'dark' else None, **kw)
                for i in range(len(it['x'])):
                    rows.append(panel(it, res, i, h))
                    st.append({'name': it['name'], 'k': int(it['ks'][i]), 'thr': float(res['thr'][i]),
                               'core_px': int(res['core'][i].sum()), 'frozen_px': int(it['core'][i].sum()),
                               'grown_px': int(res['grown'][i].sum())})
            sh = Image.new('RGB', (rows[0].width, rows[0].height * len(rows)))
            for j, r in enumerate(rows):
                sh.paste(r, (0, j * r.height))
            sh = sh.resize((sh.width // 2, sh.height // 2))
            sh.save(OUT / 'explore' / f'{tag}_{src}.jpg', quality=85)
            stats[f'{tag}/{src}'] = {'h': h, 'kw': kw, 'use_mad': use_mad, 'frames': st}
            log(f'{tag} {src}: h {h:.1f}; grown/frozen px ' + ' '.join(f"{x['grown_px']}/{x['frozen_px']}" for x in st))
        (OUT / 'explore' / 'stats.json').write_text(json.dumps(stats, indent=1))


if __name__ == '__main__':
    main()
