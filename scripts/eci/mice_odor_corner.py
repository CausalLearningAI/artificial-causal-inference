"""
Where is the odor source in each mice v1 video? -> dataset/mice/v1/eci/odor_corner.csv (+ a contact sheet).

The odor is presented in a white bag that sits OUTSIDE the arena (the bedding square), next to one corner of the
cage (src/mice_behavior/masking.py: the O-vs-not-O probe peaks there). The bag is often only partly in the
camera's field of view, or not at all, so it cannot be found reliably in every video. The water spout (a dark bar
on the cage wall, just outside the arena) is always visible, and the bag corner is fixed relative to it: the
camera is mounted in one of a few rotations, and rotating the image moves both together.

Per video, from its per-pixel background image pix_bg (0.98 quantile of the grey level over 200 frames spread over
the video, written by scripts/eci/fg_background.py; mice moving around are mostly removed, a mouse huddle that stays
for > 98% of the video is not):
  arena     rows / columns where > 50% of the pixels are brighter than 0.85 x the median grey level of the frame
            centre (the bedding) -> bounding box (r0, r1, c0, c1) in 512 px frame coordinates
  spout     dark pixels (pix_bg < 72) in four windows just outside the arena wall (4 to 45 px out), each at the place
            the spout sits in one camera rotation: right wall 8-42% down, left wall 58-92% down, top wall 8-42%
            across, bottom wall 58-92% across. The window with the most dark pixels gives the rotation. A recording
            session (one pool, one odor: its H, O and P videos, recorded back to back) has one rotation: the session
            rotation is the weighted majority (weights = dark pixel counts) of its 3 videos, and it is used for all 3.
  corner    image corner of the bag from the rotation: spout right -> BL, left -> TR, top -> BR, bottom -> TL
            (the user's prior: usually bottom-left, sometimes top-right).
  bag       white pixels (pix_bg > max(0.95 x centre median, 170)) outside the arena box grown by 12 px, counted per
            image quadrant; bag_visible = > 1500 such pixels in the predicted corner's quadrant.
  confidence high   = spout clear in this video (>= 50 dark pixels, the same rotation as the session) and bag visible
                      at the predicted corner
             medium = spout clear, bag not visible (out of frame) or another quadrant is whiter (in the checked cases:
                      white nesting tissue at the arena edge, not a second bag)
             low    = this video's spout signal is weak (< 50 dark pixels) or disagrees with its session: the corner
                      comes from the session vote
Zone used by the zone pooling (scripts/eci/spatial_encode_all.py): 'near odor' = patches whose centre is within
NEAR_RADIUS x the arena side (mean of its height and width) of the arena corner on the bag side; 'rest' = every other
patch. The corner and radius are also written in patch units (32 x 32 grid, 16 px per patch).

Habituation (H) and post (P) videos: the bag holder is often visible there too (the odor itself is presented in O).
The zone is the same physical corner for all 6 stages of a pool (the corner of that video's own rotation), so H and
P measure how much time the mice spend at the same place without / after the odor.

Also prints chi-square tests of the corner against stage, genotype, line, sex, pool and recording date (pool level
where the field is constant within pool).

Usage: python scripts/eci/mice_odor_corner.py [--sheet]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from eci.contrasts import STAGES  # noqa: E402

BG_DIR = ROOT / 'dataset/mice/v1/eci/fg448/background'
OUT = ROOT / 'dataset/mice/v1/eci/odor_corner.csv'
EXPERIMENT = ROOT / 'data/mice/v1/experiment.csv'
SPOUT_TO_CORNER = {'right': 'BL', 'left': 'TR', 'top': 'BR', 'bottom': 'TL'}
NEAR_RADIUS = 0.5  # fraction of the arena side
DARK, SPOUT_MIN, BAG_MIN = 72, 50, 1500
PATCH = 16


def measure(pix):
    p = pix.astype(int)
    cm = float(np.median(p[156:356, 156:356]))
    br = p > 0.85 * cm
    rr, cc = np.flatnonzero(br.mean(1) > 0.5), np.flatnonzero(br.mean(0) > 0.5)
    r0, r1, c0, c1 = int(rr.min()), int(rr.max()), int(cc.min()), int(cc.max())
    H, W, m, w = r1 - r0, c1 - c0, 4, 45
    dark = p < DARK
    win = {'right': dark[r0 + int(.08 * H):r0 + int(.42 * H), c1 + m:c1 + w],
           'left': dark[r0 + int(.58 * H):r0 + int(.92 * H), max(c0 - w, 0):c0 - m],
           'top': dark[max(r0 - w, 0):r0 - m, c0 + int(.08 * W):c0 + int(.42 * W)],
           'bottom': dark[r1 + m:r1 + w, c0 + int(.58 * W):c0 + int(.92 * W)]}
    spout = {k: int(v.sum()) for k, v in win.items()}
    ext = np.ones_like(br)
    g = 12
    ext[max(r0 - g, 0):r1 + g + 1, max(c0 - g, 0):c1 + g + 1] = False
    white = (p > max(0.95 * cm, 170)) & ext
    h = p.shape[0] // 2
    q = {'TL': white[:h, :h].sum(), 'TR': white[:h, h:].sum(), 'BL': white[h:, :h].sum(), 'BR': white[h:, h:].sum()}
    return {'arena_r0': r0, 'arena_r1': r1, 'arena_c0': c0, 'arena_c1': c1, 'centre_grey': cm,
            **{f'spout_dark_{k}': v for k, v in spout.items()}, **{f'white_{k}': int(v) for k, v in q.items()}}


def build():
    e = pd.read_csv(EXPERIMENT)
    rows = []
    for oid in e['observation_id']:
        rows.append({'observation_id': oid, **measure(np.load(BG_DIR / f'{oid}.npz')['pix_bg'])})
    d = e[['observation_id', 'pool', 'line', 'sex', 'genotype', 'odor', 'phase', 'date', 'time']].merge(
        pd.DataFrame(rows), on='observation_id', validate='1:1')
    d['stage'] = [STAGES[(o, p)] for o, p in zip(d['odor'], d['phase'])]
    sides = list(SPOUT_TO_CORNER)
    S = d[[f'spout_dark_{s}' for s in sides]].values
    d['spout_video'] = np.array(sides)[S.argmax(1)]
    d['spout_dark'] = S.max(1)
    d['session'] = d['pool'] + '_' + d['odor']
    vote = d.groupby(['session', 'spout_video'])['spout_dark'].sum().reset_index().sort_values('spout_dark')
    vote = vote.drop_duplicates('session', keep='last').set_index('session')['spout_video']
    d['camera_spout_side'] = d['session'].map(vote)
    d['odor_corner'] = d['camera_spout_side'].map(SPOUT_TO_CORNER)
    clear = (d['spout_dark'] >= SPOUT_MIN) & (d['spout_video'] == d['camera_spout_side'])
    d['corner_source'] = np.where(clear, 'spout (this video)', 'spout (session vote)')
    d['bag_white_px'] = [r[f'white_{r["odor_corner"]}'] for _, r in d.iterrows()]
    d['bag_white_px_other_max'] = [max(r[f'white_{q}'] for q in ('TL', 'TR', 'BL', 'BR') if q != r['odor_corner'])
                                   for _, r in d.iterrows()]
    d['bag_visible'] = d['bag_white_px'] > BAG_MIN
    d['confidence'] = np.where(~clear, 'low', np.where(d['bag_visible'], 'high', 'medium'))
    # arena corner on the bag side (frame px) and the near-odor radius
    top = d['odor_corner'].str[0] == 'T'
    left = d['odor_corner'].str[1] == 'L'
    d['corner_y_px'] = np.where(top, d['arena_r0'], d['arena_r1'])
    d['corner_x_px'] = np.where(left, d['arena_c0'], d['arena_c1'])
    side = ((d['arena_r1'] - d['arena_r0']) + (d['arena_c1'] - d['arena_c0'])) / 2
    d['near_radius_px'] = NEAR_RADIUS * side
    d['corner_y_patch'] = d['corner_y_px'] / PATCH
    d['corner_x_patch'] = d['corner_x_px'] / PATCH
    d['near_radius_patch'] = d['near_radius_px'] / PATCH
    return d


def near_mask(row, grid=32):
    """(grid * grid,) bool: patch centres within the near-odor radius of the arena corner (row-major patches)."""
    c = (np.arange(grid) + 0.5)
    yy, xx = np.meshgrid(c, c, indexing='ij')
    return (np.hypot(yy - row['corner_y_patch'], xx - row['corner_x_patch']) <= row['near_radius_patch']).ravel()


def tests(d):
    from scipy.stats import chi2_contingency
    out = []
    pool = d.drop_duplicates('pool').copy()
    # corner per pool: the pool's sessions can differ (S and F on different days)
    pc = d.groupby('pool')['odor_corner'].agg(lambda s: '/'.join(sorted(set(s))))
    pool['pool_corner'] = pool['pool'].map(pc)
    for field, df, col in (('stage', d, 'odor_corner'), ('odor (S/F)', d, 'odor_corner'),
                           ('genotype (pool level)', pool, 'pool_corner'), ('line (pool level)', pool, 'pool_corner'),
                           ('sex (pool level)', pool, 'pool_corner'), ('recording date (session level)',
                                                                       d.drop_duplicates('session'), 'odor_corner')):
        f = {'stage': 'stage', 'odor (S/F)': 'odor'}.get(field, field.split(' ')[0])
        f = 'date' if f == 'recording' else f
        tab = pd.crosstab(df[f], df[col])
        p = float(chi2_contingency(tab.values)[1]) if min(tab.shape) > 1 else 1.0
        out.append((field, tab, p))
    return out


def sheet(d, path, t=128, nc=12):
    from PIL import Image, ImageDraw
    nr = (len(d) + nc - 1) // nc
    canvas = Image.new('RGB', (nc * t, nr * (t + 12)), (0, 0, 0))
    dr = ImageDraw.Draw(canvas)
    for i, r in enumerate(d.itertuples()):
        im = Image.fromarray(np.load(BG_DIR / f'{r.observation_id}.npz')['pix_bg']).convert('RGB').resize((t, t))
        x, y = (i % nc) * t, (i // nc) * (t + 12)
        canvas.paste(im, (x, y + 12))
        s = t / 512
        cy, cx, rad = r.corner_y_px * s, r.corner_x_px * s, r.near_radius_px * s
        col = {'high': (0, 255, 0), 'medium': (255, 200, 0), 'low': (255, 0, 0)}[r.confidence]
        dr.ellipse((x + cx - rad, y + 12 + cy - rad, x + cx + rad, y + 12 + cy + rad), outline=col)
        dr.text((x + 2, y), f'{r.observation_id.replace("_", "")[:16]} {r.odor_corner}', fill=col)
    canvas.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sheet', default=None, help='write a contact sheet PNG of every video with its zone here')
    args = ap.parse_args()
    d = build()
    cols = ['observation_id', 'pool', 'line', 'sex', 'genotype', 'stage', 'odor', 'phase', 'date', 'time', 'session',
            'camera_spout_side', 'odor_corner', 'confidence', 'corner_source', 'bag_visible', 'bag_white_px',
            'bag_white_px_other_max', 'spout_video', 'spout_dark', 'spout_dark_right', 'spout_dark_left',
            'spout_dark_top', 'spout_dark_bottom', 'white_TL', 'white_TR', 'white_BL', 'white_BR', 'centre_grey',
            'arena_r0', 'arena_r1', 'arena_c0', 'arena_c1', 'corner_y_px', 'corner_x_px', 'near_radius_px',
            'corner_y_patch', 'corner_x_patch', 'near_radius_patch']
    d[cols].to_csv(OUT, index=False)
    print(f'wrote {OUT} ({len(d)} videos)')
    print('corner counts:', d['odor_corner'].value_counts().to_dict())
    print('camera rotation (spout side):', d['camera_spout_side'].value_counts().to_dict())
    print('confidence:', d['confidence'].value_counts().to_dict())
    print('bag visible at the predicted corner:', int(d['bag_visible'].sum()), '; another quadrant whiter:',
          int((d['bag_white_px_other_max'] > np.maximum(d['bag_white_px'], BAG_MIN)).sum()))
    print('bag visible by phase:', d.groupby('phase')['bag_visible'].mean().round(3).to_dict())
    print('sessions:', d['session'].nunique(), '; pools with 2 corners (S and F sessions differ):',
          int((d.groupby('pool')['odor_corner'].nunique() > 1).sum()))
    near = np.stack([near_mask(r) for _, r in d.iterrows()])
    print('near-odor patches per video: min', near.sum(1).min(), 'median', np.median(near.sum(1)), 'max', near.sum(1).max())
    for field, tab, p in tests(d):
        print(f'\n{field}: chi-square p = {p:.3g}\n{tab.to_string()}')
    if args.sheet:
        sheet(d, args.sheet)
        print('sheet ->', args.sheet)


if __name__ == '__main__':
    main()
