"""
Mask-U test (rules: scripts/benchmark/eci_mask_u.sh): shared helpers - output paths, keypoint -> cell maps, keypoint
recall per animal, keypoint-hull precision, mask sheets, the GPU-utilisation-friendly logging. Keypoints are used ONLY
to measure masks, never in a mask.
"""
import os
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / 'results/vision/mask_u'
TOK = Path(os.environ.get('MASKU_TOK') or '/mnt/beegfs/locatgrp/shared/rcadei/eci_tok/mask_u')
log = lambda s: print(time.strftime('%H:%M:%S'), s, flush=True)  # noqa: E731


def stage_dir():
    d = Path(os.environ.get('LOCAL_DIR') or '')
    if not str(d).startswith('/localhome') and not str(d).startswith(os.environ.get('TMPDIR') or '/nonexistent'):
        raise RuntimeError(f'LOCAL_DIR must be under /localhome or $TMPDIR, got {d!r}')
    d.mkdir(parents=True, exist_ok=True)
    return d


def recall_cells(mask, cells, valid):
    """mask (n, 1024) bool, cells (n, A, K) int cell index, valid (n, A, K) bool -> (n, A) recall (NaN = no valid kp),
    (n, A) bool missed (valid keypoints but none in a kept cell)."""
    n, A, K = cells.shape
    hit = mask[np.arange(n)[:, None, None], np.clip(cells, 0, 1023)]
    with np.errstate(invalid='ignore', divide='ignore'):
        rec = np.where(valid.any(2), (hit & valid).sum(2) / np.maximum(valid.sum(2), 1), np.nan)
    missed = valid.any(2) & ~(hit & valid).any(2)
    return rec, missed


def hull_cells(xy_cells, valid, grid=32, up=4, tol=1):
    """xy_cells (n, A, K, 2) keypoint (x, y) in CELL units, valid (n, A, K) -> (n, grid*grid) bool: cells within tol
    cells of the convex hull of some animal's valid keypoints (rasterised at up x the grid)."""
    import cv2
    from scipy import ndimage as ndi
    n = len(xy_cells)
    out = np.zeros((n, grid * grid), bool)
    st = np.ones((2 * tol + 1, 2 * tol + 1), bool)
    for f in range(n):
        im = np.zeros((grid * up, grid * up), np.uint8)
        for a in range(xy_cells.shape[1]):
            p = xy_cells[f, a][valid[f, a]] * up
            if len(p) == 0:
                continue
            if len(p) < 3:
                for q in p:
                    cv2.circle(im, (int(q[0]), int(q[1])), up // 2, 1, -1)
                continue
            h = cv2.convexHull(p.astype(np.float32)).astype(np.int32)
            cv2.fillConvexPoly(im, h, 1)
        c = im.reshape(grid, up, grid, up).any((1, 3))
        out[f] = ndi.binary_dilation(c, st).ravel() if tol else c.ravel()
    return out


def sheet(tiles, path, ncol=4):
    """tiles: list of (rgb uint8 (h, w, 3), {name: (1024,) bool}, title, kp_xy_px (m, 2) or None). Each tile is drawn
    once per mask (current = red tint, U = green tint), side by side."""
    from PIL import Image, ImageDraw
    ims = []
    for rgb, masks, title, kp in tiles:
        row = []
        for nm, m in masks.items():
            h, w = rgb.shape[:2]
            g = int(np.sqrt(m.size))
            M = np.kron(m.reshape(g, g), np.ones((h // g, w // g))).astype(bool)
            im = rgb.astype(np.float32).copy()
            ch = 0 if nm != 'U' else 1
            im[M, ch] = im[M, ch] * 0.5 + 127
            im[M] *= np.where(np.arange(3) == ch, 1.0, 0.6)
            pil = Image.fromarray(im.clip(0, 255).astype(np.uint8))
            sc = max(1, 320 // max(h, w))
            if sc > 1:
                pil = pil.resize((w * sc, h * sc), Image.NEAREST)
            elif max(h, w) > 512:
                pil = pil.resize((w // 2, h // 2))
            d = ImageDraw.Draw(pil)
            fx = pil.width / w
            if kp is not None:
                for x, y in kp:
                    d.ellipse([x * fx - 3, y * fx - 3, x * fx + 3, y * fx + 3], outline=(0, 160, 255), width=2)
            d.text((4, 4), f'{nm}: {int(m.sum())} kept | {title}', fill=(255, 255, 0))
            row.append(pil)
        W_ = sum(p.width for p in row)
        r = Image.new('RGB', (W_, row[0].height))
        x = 0
        for p in row:
            r.paste(p, (x, 0))
            x += p.width
        ims.append(r)
    ncol = max(1, ncol // max(1, len(tiles[0][1])))
    Wt = max(i.width for i in ims)
    Ht = max(i.height for i in ims)
    nrow = (len(ims) + ncol - 1) // ncol
    S = Image.new('RGB', (ncol * Wt, nrow * Ht))
    for k, i in enumerate(ims):
        S.paste(i, ((k % ncol) * Wt, (k // ncol) * Ht))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    S.save(path, quality=85)
