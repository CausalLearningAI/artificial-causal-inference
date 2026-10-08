"""
Shared pieces of the per-frame instance-split GATE test (scripts/eci/split_test_*.py): can each 512 px frame be split
into its N animals (ants N = 3, mice N = 4) even while they touch, per frame, without long-term tracking?

Everything here is label-free except the evaluation (contact labels only choose which frames are scored).

Dark mask and blobs (the existing baseline, src/eci/crops.py): dark = grey < dark_abs AND pix_bg - grey > dark_rel
(pix_bg = the video's 0.98-quantile pixel background, dataset/<..>/eci/fg448/background), binary opening with a disk
of radius open_r, holes filled, 8-connected components >= min_area pixels.
    mice  open_r 2, min_area 150 (crops config); single-mouse dark area A1_dark and single range [0.5, 1.5] A1_dark
          from dataset/mice/v1/eci/crops/config.json (calibrated label-free on training videos)
    ants  open_r 1 (ants are ~40 px long, legs 1-2 px), min_area 40; A1_dark calibrated the same way by
          split_test_select.py on training videos

Every method outputs, per frame, a label map (0 = not an animal, 1..M = instance); patch-level maps (32 x 32 slots)
are upsampled to 512 px blocks of 16 x 16. Scoring (pre-registered, identical for every method; score_frame):
    A1_m       the method's own single-animal mask area = median area of its masks on CLEAN non-contact eval frames
               (the dark-blob detector sees exactly N blobs, each in the single range) - masks of different methods
               differ in extent (SAM keeps legs / tails, slots are 16 px blocks), so each is judged in its own unit
    debris     masks < 0.2 A1_m are dropped before counting (leftover slots, specks)
    count_ok   exactly N masks remain
    area_ok    every mask in [0.4, 2.5] A1_m
    dark_ok    every mask holds between 0.3 and 1.6 A1_dark dark pixels: it sits on an animal (not on bedding) and
               does not hold two animals' worth of dark body (two touching mice ~ 2 A1_dark; the single range of the
               blob calibration tops out at 1.5 A1_dark). This is the "no mask spans two animals" rule.
    ants      + dots_ok: both raw dots (tracking raw_yellow / raw_blue, scored only where both were detected) lie in
               two DIFFERENT masks (dot pixel's label, else the nearest labelled pixel within 6 px); with exactly
               3 masks the third holds neither dot.
    correct  = count_ok & area_ok & dark_ok (& dots_ok for ants)
Pair check (contact frames):
    ants  Y2F: the yellow-dot mask exists and a mask holding neither dot lies within 8 px of it (the groomed focal ant
          has its own mask); B2F likewise for blue; both when both are labelled
    mice  the pair cannot be named, so the contact regions are checked: every merged dark blob (> 1.5 A1_dark) must be
          covered by >= 2 different masks each holding >= 0.3 A1_dark of it, and every two dark blobs of at least
          single-mouse size (>= 0.5 A1_dark; smaller ones are tail / body fragments) closer than pair_d (17 px) must
          have different dominant masks; N/A when a frame has neither.
"""
import io
import json
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / 'results/vision/eci_split_test'
DOMAINS = {
    'mice': dict(N=4, ann=REPO / 'dataset/mice/v1/annotations.csv', bg=REPO / 'dataset/mice/v1/eci/fg448/background',
                 blob=dict(dark_abs=60, dark_rel=40, open_r=2, min_area=150), n_train=100, pair_d=17.37),
    'ants': dict(N=3, ann=REPO / 'dataset/ants/eci/annotations.csv', bg=REPO / 'dataset/ants/eci/fg448/background',
                 blob=dict(dark_abs=60, dark_rel=40, open_r=1, min_area=40), n_train=128, pair_d=8.0),
}
GRID, PATCH = 32, 16
COLORS = np.array([[0, 0, 0], [230, 25, 75], [60, 180, 75], [0, 130, 200], [255, 225, 25], [145, 30, 180],
                   [70, 240, 240], [245, 130, 48], [240, 50, 230], [128, 128, 0]], np.uint8)


def _disk(r):
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def dark_clean(grey, pix_bg, p):
    """grey, pix_bg (512, 512) uint8 -> opened, hole-filled dark mask (bool)."""
    g = grey.astype(np.int16)
    m = (g < p['dark_abs']) & ((pix_bg.astype(np.int16) - g) > p['dark_rel'])
    if p['open_r'] > 0:
        m = ndi.binary_opening(m, structure=_disk(p['open_r']))
    return ndi.binary_fill_holes(m)


def blob_labels(dark, p):
    """dark mask -> (label map int32 with components >= min_area renumbered 1..n, areas (n,))."""
    lab, n = ndi.label(dark, structure=np.ones((3, 3), bool))
    if n == 0:
        return lab, np.zeros(0, np.int64)
    areas = np.bincount(lab.ravel(), minlength=n + 1)
    keep = np.flatnonzero(areas >= p['min_area'])
    keep = keep[keep > 0]
    remap = np.zeros(n + 1, np.int32)
    remap[keep] = np.arange(1, len(keep) + 1)
    return remap[lab], areas[keep]


def is_clean(areas, N, lo, hi):
    return len(areas) == N and bool(np.all((areas >= lo) & (areas <= hi)))


def load_calib(domain):
    return json.loads((OUT / domain / 'calib.json').read_text())


def read_tar_frames(path):
    """tar of <frame_path> jpgs -> {frame_path: bytes}."""
    out = {}
    with tarfile.open(path) as t:
        for m in t.getmembers():
            if m.isfile():
                out[m.name] = t.extractfile(m).read()
    return out


def decode(b, mode='RGB'):
    return np.asarray(Image.open(io.BytesIO(b)).convert(mode))


def upsample_patch_labels(lab32):
    return np.kron(lab32.astype(np.int32), np.ones((PATCH, PATCH), np.int32))


def _label_at(L, y, x, r=6):
    y, x = int(round(y)), int(round(x))
    y, x = min(max(y, 0), L.shape[0] - 1), min(max(x, 0), L.shape[1] - 1)
    if L[y, x] > 0:
        return int(L[y, x])
    y0, y1, x0, x1 = max(y - r, 0), min(y + r + 1, L.shape[0]), max(x - r, 0), min(x + r + 1, L.shape[1])
    sub = L[y0:y1, x0:x1]
    yy, xx = np.nonzero(sub)
    if len(yy) == 0:
        return 0
    d = (yy + y0 - y) ** 2 + (xx + x0 - x) ** 2
    i = int(np.argmin(d))
    return int(sub[yy[i], xx[i]]) if d[i] <= r * r else 0


def clean_labels(L, a1_m):
    """drop masks < 0.2 A1_m; -> (relabelled map, kept ids list, areas)."""
    ids, cnt = np.unique(L[L > 0], return_counts=True)
    keep = ids[cnt >= 0.2 * a1_m]
    remap = np.zeros(int(L.max()) + 1 if L.size and L.max() > 0 else 1, np.int32)
    remap[keep] = np.arange(1, len(keep) + 1)
    L2 = remap[L]
    return L2, cnt[cnt >= 0.2 * a1_m]


def score_frame(L, dark, blobs, blob_areas, cal, domain, dots=None, contact=None):
    """L label map (512, 512) of one method (already upsampled); dark, blobs: the dark mask and its components;
    cal: dict with a1_m (this method), a1_dark, single_hi; dots: (yx_yellow or None, yx_blue or None);
    contact: ants (y2f, b2f) bools, mice any truthy. -> dict of booleans / numbers (see module docstring)."""
    N = DOMAINS[domain]['N']
    a1m, a1d = cal['a1_m'], cal['a1_dark']
    L2, areas = clean_labels(L, a1m)
    k = len(areas)
    r = {'n_masks': k, 'count_ok': k == N}
    r['area_ok'] = bool(k > 0 and np.all((areas >= 0.4 * a1m) & (areas <= 2.5 * a1m)))
    darkpix = np.bincount(L2[dark], minlength=k + 1)[1:] if k else np.zeros(0)
    r['dark_ok'] = bool(k > 0 and np.all((darkpix >= 0.3 * a1d) & (darkpix <= 1.6 * a1d)))
    r['max_dark_frac'] = float(darkpix.max() / a1d) if k else 0.0
    ok = r['count_ok'] and r['area_ok'] and r['dark_ok']
    if domain == 'ants':
        yel, blu = dots
        r['both_dots'] = yel is not None and blu is not None
        if r['both_dots']:
            my, mb = _label_at(L2, *yel), _label_at(L2, *blu)
            r['dots_ok'] = bool(my > 0 and mb > 0 and my != mb)
            ok = ok and r['dots_ok']
            if contact is not None and (contact[0] or contact[1]):
                pair = True
                for flag, m_dot in ((contact[0], my), (contact[1], mb)):
                    if not flag:
                        continue
                    if m_dot == 0:
                        pair = False
                        continue
                    others = [j for j in range(1, k + 1) if j not in (my, mb)]
                    near = False
                    if others:
                        dist = ndi.distance_transform_edt(L2 != m_dot)
                        near = any(dist[L2 == j].min() <= 8 for j in others)
                    pair = pair and near
                r['pair_ok'] = bool(pair)
        else:
            r['dots_ok'] = None
            ok = None
    else:
        if contact:
            nb = len(blob_areas)
            checks = []
            for b in range(1, nb + 1):
                if blob_areas[b - 1] > cal['single_hi']:
                    inb = L2[blobs == b]
                    c = np.bincount(inb, minlength=k + 1)[1:]
                    checks.append(int((c >= 0.3 * a1d).sum()) >= 2)
            if nb >= 2:
                dt_idx = {}
                for b in range(1, nb + 1):
                    dt_idx[b] = ndi.distance_transform_edt(blobs != b) if nb <= 8 else None
                for b in range(1, nb + 1):
                    if dt_idx[b] is None or blob_areas[b - 1] < cal['single_lo']:
                        continue
                    for c2 in range(b + 1, nb + 1):
                        if blob_areas[c2 - 1] < cal['single_lo']:
                            continue
                        if dt_idx[b][blobs == c2].min() < DOMAINS['mice']['pair_d']:
                            db = np.bincount(L2[blobs == b], minlength=k + 1)
                            dc = np.bincount(L2[blobs == c2], minlength=k + 1)
                            db[0] = dc[0] = 0
                            checks.append(bool(db.max() > 0 and dc.max() > 0 and db.argmax() != dc.argmax()))
            r['pair_ok'] = bool(all(checks)) if checks else None
    r['correct'] = ok
    return r


def overlay(rgb, L, alpha=0.5, outline=True):
    """rgb (H, W, 3) uint8, L label map -> coloured overlay uint8."""
    out = rgb.astype(np.float32).copy()
    col = COLORS[1 + (np.arange(int(L.max()) + 1) - 1) % (len(COLORS) - 1)]
    col[0] = 0
    m = L > 0
    out[m] = (1 - alpha) * out[m] + alpha * col[L[m]]
    if outline:
        edge = (L != ndi.grey_erosion(L, size=3)) | (L != ndi.grey_dilation(L, size=3))
        edge &= ndi.binary_dilation(m)
        out[edge] = col[ndi.grey_dilation(L, size=3)[edge]]
    return out.clip(0, 255).astype(np.uint8)


def crop_box(domain, dark, dots=None, size_min=150):
    """-> (y0, x0, side) of the tile crop: a square around all given animal pixels (+ 20 px each side), at least
    size_min (ants 150, mice 300 px)."""
    if domain == 'mice':
        size_min = 300
    yy, xx = np.nonzero(dark)
    pts_y, pts_x = list(yy), list(xx)
    if dots:
        for d in dots:
            if d is not None:
                pts_y.append(d[0]); pts_x.append(d[1])
    if not pts_y:
        return 0, 0, 512
    y0, y1, x0, x1 = min(pts_y), max(pts_y), min(pts_x), max(pts_x)
    side = int(max(y1 - y0, x1 - x0, size_min) + 40)
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    side = min(side, 512)
    y0 = int(min(max(cy - side // 2, 0), 512 - side))
    x0 = int(min(max(cx - side // 2, 0), 512 - side))
    return y0, x0, side


def tile(rgb, L, box, px=300, dots=None, text=None):
    from PIL import ImageDraw
    y0, x0, s = box
    ov = overlay(rgb, L)
    im = Image.fromarray(np.ascontiguousarray(ov[y0:y0 + s, x0:x0 + s])).resize((px, px), Image.NEAREST)
    d = ImageDraw.Draw(im)
    f = px / s
    if dots:
        for dt, c in zip(dots, ((255, 200, 0), (0, 90, 255))):
            if dt is not None:
                yy, xx = (dt[0] - y0) * f, (dt[1] - x0) * f
                d.ellipse([xx - 4, yy - 4, xx + 4, yy + 4], outline=(255, 255, 255), width=2)
                d.ellipse([xx - 2, yy - 2, xx + 2, yy + 2], fill=c)
    if text:
        d.rectangle([0, 0, 9 * len(text) + 6, 16], fill=(0, 0, 0))
        d.text((3, 2), text, fill=(255, 255, 255))
    return im


def sheet(tiles, cols, path, title=None):
    px = tiles[0].size[0]
    rows = (len(tiles) + cols - 1) // cols
    top = 24 if title else 0
    S = Image.new('RGB', (cols * px + (cols - 1) * 4, top + rows * px + (rows - 1) * 4), (255, 255, 255))
    if title:
        from PIL import ImageDraw
        ImageDraw.Draw(S).text((4, 5), title, fill=(0, 0, 0))
    for i, t in enumerate(tiles):
        S.paste(t, ((i % cols) * (px + 4), top + (i // cols) * (px + 4)))
    S.save(path, quality=90)
