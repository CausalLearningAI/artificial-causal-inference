"""
Contact sheets of the tadpole body crops (scripts/eci/tadpoles_crops.py) and of the crop mask (rule 'tadpoles',
src/eci/foreground.py), for looking at them.

--what crops: one row per video, n_times crops evenly spread over the video (5 fps frame index i), each labelled with
    i, the crop track status (T tracked, I interpolated, H held = no SLEAP centre within 1 s) and the SLEAP nodes of
    that frame (red dots = head + trunk nodes, cyan = tail nodes). Reads dataset/frogs/v1_tadpole/{crops,sleap,frames}.
--what masks: the same frames with the mask patches tinted red (the exact FgBackgrounds.mask path of the chain) and
    the tadpole pixels (before patch pooling) in yellow; needs the tables and scripts/eci/tadpoles_background.py
    (the domain's eci dir, TADPOLE_TAG respected). --dark-rel / --near-px / --dark-frac override the rule (tuning).
    Prints the mean mask patches per frame.

Usage: python scripts/eci/tadpoles_sheets.py --what crops --obs WT_100_1,FoxP1_169_3 --out sheet.png
       python scripts/eci/tadpoles_sheets.py --what masks --obs WT_100_1 --dark-rel 20 --out masks.png
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
V1T = REPO / 'dataset/frogs/v1_tadpole'
HEAD = ('Left_Eye', 'Right_Eye', 'Heart_Center', 'Tail_Stem')


def times(n, k, seed=None):
    if seed is None:
        return np.round(np.linspace(0, n - 1, k + 2)[1:-1]).astype(int)
    return np.sort(np.random.default_rng(seed).choice(n, k, replace=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--what', required=True, choices=('crops', 'masks'))
    p.add_argument('--obs', required=True)
    p.add_argument('--n-times', type=int, default=10)
    p.add_argument('--seed', type=int, default=None, help='random frames instead of evenly spread')
    p.add_argument('--tile', type=int, default=192)
    p.add_argument('--out', required=True)
    p.add_argument('--dark-rel', type=float, default=None)
    p.add_argument('--near-px', type=float, default=None)
    p.add_argument('--dark-frac', type=float, default=None)
    args = p.parse_args()
    ids = args.obs.split(',')
    T = args.tile
    sheet = Image.new('RGB', (T * args.n_times + 150, T * len(ids)), 'white')
    dr = ImageDraw.Draw(sheet)
    stats = []
    if args.what == 'masks':
        import torch
        import torch.nn.functional as F
        from src.eci.domain import get_domain
        from src.eci.foreground import (GRID, PATCH_PX, RULES, FgBackgrounds, align_tag, crop_planes, obs_rows,
                                        tadpole_pixels)
        D = get_domain('tadpoles')
        rule = dict(RULES['tadpoles'])
        for k, v in (('dark_rel', args.dark_rel), ('near_px', args.near_px), ('dark_frac', args.dark_frac)):
            if v is not None:
                rule[k] = v
        ranges = obs_rows(D.ann_path)
        bgs = FgBackgrounds(D.eci_dir / align_tag('none') / 'background', ranges, rule, 'cpu')
        print(f'rule {rule}')
    for r, o in enumerate(ids):
        c = np.load(V1T / 'crops' / f'{o}.npz')
        z = np.load(V1T / 'sleap' / f'{o}.npz')
        names = [str(n) for n in z['node_names']]
        st = c['status']
        n = len(st)
        ts = times(n, args.n_times, args.seed)
        label = f'{o}\ntracked {np.mean(st == 0):.3f}\ninterp {np.mean(st == 1):.3f}\nheld {np.mean(st == 2):.3f}'
        dr.multiline_text((4, r * T + 4), label, fill='black')
        imgs = [Image.open(V1T / 'frames/crop' / o / f'frame_{i:06d}.jpg').convert('RGB') for i in ts]
        if args.what == 'masks':
            lo = ranges[o][0]
            grey = torch.from_numpy(np.stack([np.asarray(im.convert('L')) for im in imgs]))
            rows = lo + ts
            mask, _ = bgs.mask(torch.zeros(len(ts), GRID * GRID, 1), grey, rows)
            b = bgs.get(o)
            rel = torch.from_numpy(ts)
            pbg, roi = crop_planes(b, rel)
            px = tadpole_pixels(grey, pbg, roi, b['nodes'][rel], rule)
            stats.append((o, float(mask.sum(1).float().mean()), float((mask.sum(1) == 0).float().mean())))
        for j, (i, im) in enumerate(zip(ts, imgs)):
            if args.what == 'masks':
                a = np.asarray(im).astype(np.float32)
                m = np.kron(mask[j].view(GRID, GRID).numpy(), np.ones((PATCH_PX, PATCH_PX), bool))
                a[m] = 0.55 * a[m] + 0.45 * np.array([255, 0, 0])
                a[px[j].numpy()] = [255, 230, 0]
                im = Image.fromarray(a.clip(0, 255).astype(np.uint8))
            im = im.resize((T, T))
            d = ImageDraw.Draw(im)
            if args.what == 'crops':
                s = T / float(c['side'])
                for k, (x, y) in enumerate(z['nodes'][i]):
                    if np.isfinite(x):
                        u, v = (x - c['x0'][i]) * s, (y - c['y0'][i]) * s
                        col = (255, 0, 0) if names[k] in HEAD else (0, 200, 255)
                        d.ellipse([u - 1.5, v - 1.5, u + 1.5, v + 1.5], fill=col)
            d.text((3, 3), f'{i} {"TIH"[st[i]]}', fill=(255, 0, 0) if st[i] else (0, 120, 0))
            sheet.paste(im, (150 + j * T, r * T))
    sheet.save(args.out)
    for o, m, z0 in stats:
        print(f'{o}: mask patches per frame (sheet frames) {m:.1f}, empty-mask frames {z0:.2f}')
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
