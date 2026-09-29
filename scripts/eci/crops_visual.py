"""
Crop prototype visual check (label-free): for random frames of the training videos, draw the blob
outlines and crop squares on 6 full frames, and show 24 crops (8 single, 8 pair, 8 merged) as fed
to DINOv2 (224 x 224, rotated).

Output: dataset/mice/v1/eci/crops/visual_check.jpg
Usage: python scripts/eci/crops_visual.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.eci.crops import frame_units, sample_crops  # noqa: E402
from src.eci.foreground import obs_rows  # noqa: E402

CROPS = REPO / 'dataset/mice/v1/eci/crops'
COL = {0: (0, 200, 0), 1: (255, 0, 0), 2: (0, 120, 255)}
NAME = {0: 'single', 1: 'pair', 2: 'merged'}


def square(spec):
    _, cy, cx, a, side = spec
    h = side / 2
    c, s = np.cos(a), np.sin(a)
    return [(cx + c * u - s * v, cy + s * u + c * v) for u, v in ((-h, -h), (h, -h), (h, h), (-h, h))]


def main():
    cfg = json.loads((CROPS / 'config.json').read_text())
    p = cfg['params']
    ann = REPO / 'dataset/mice/v1/annotations.csv'
    meta = pd.read_csv(ann, usecols=['observation_id', 'frame_path'])
    ranges = obs_rows(ann)
    rng = np.random.default_rng(1)
    pool = {0: [], 1: [], 2: []}
    frames = []
    for it in range(400):
        o = cfg['train_videos'][rng.integers(20)]
        r = int(rng.integers(*ranges[o]))
        rgb = np.asarray(Image.open(REPO / 'dataset' / meta.frame_path[r]).convert('RGB'))
        pix_bg = np.load(REPO / 'dataset/mice/v1/eci/fg448/background' / f'{o}.npz')['pix_bg']
        blobs, crops, _, g = frame_units(np.asarray(Image.fromarray(rgb).convert('L')), pix_bg, p)
        if len(frames) < 6 and (len(frames) < 3 or g['n_merged'] or (crops[:, 0] == 1).any()):
            im = Image.fromarray(rgb.copy()); d = ImageDraw.Draw(im)
            for b in blobs:
                for y, x in b['edge'][::3]:
                    d.point((int(x), int(y)), fill=(255, 255, 0))
            for c in crops:
                d.polygon(square(c), outline=COL[int(c[0])])
            d.text((5, 5), f"{o} r{r} blobs {g['n_blobs']} s{g['n_single']} m{g['n_merged']} f{g['n_fragment']} "
                           f"dmin {g['min_dist']:.0f}", fill=(255, 0, 0))
            frames.append(im.resize((336, 336)))
        for c in crops:
            if len(pool[int(c[0])]) < 8 and rng.random() < 0.3:
                pool[int(c[0])].append((rgb, c))
        if len(frames) >= 6 and all(len(v) >= 8 for v in pool.values()):
            break
    W = Image.new('RGB', (8 * 224, 2 * 336 + 3 * 224), 'white')
    for k, im in enumerate(frames):
        W.paste(im, ((k % 3) * 336 + (k // 3) * 0, (k // 3) * 336))
    # frames occupy 3 columns x 2 rows at 336 px; crops below
    for t in range(3):
        for k, (rgb, c) in enumerate(pool[t]):
            x = sample_crops(torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None], torch.from_numpy(c[None]),
                             torch.zeros(1, dtype=torch.long))[0]
            im = Image.fromarray((x.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
            ImageDraw.Draw(im).text((4, 4), NAME[t], fill=COL[t])
            W.paste(im, (k * 224, 2 * 336 + t * 224))
    W.save(CROPS / 'visual_check.jpg', quality=85)
    print('saved', CROPS / 'visual_check.jpg', {NAME[t]: len(v) for t, v in pool.items()}, len(frames))


if __name__ == '__main__':
    main()
