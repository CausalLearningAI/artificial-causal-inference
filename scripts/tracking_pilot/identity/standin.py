"""STAND-IN tracks (not AMADEUS): simple blob detectors + a greedy-free Hungarian linker, written
in the tracker contract so the identity layer can be developed before AMADEUS lands.

Mice: dark-blob detector at 1/4 resolution (dark core of black fur, grown into the 'darker than
the per-pixel background' region so shaved patches stay inside the body), adapted from
src/eci/crops.py. Ants: background subtraction at source resolution inside the dish, adapted
from src/tracking/detection.py. Neither detector splits touching animals: a merged blob is one
detection, which is exactly the kind of ambiguity the identity layer must cut at.

Linker: Hungarian matching of consecutive detections on centroid distance with a gate; unmatched
detections start new tracks; a track not matched for `max_gap` frames ends (no filling, so every
row has detected=True).

Usage:
    python standin.py --domain mice --video rd32 [--stride 3]
Writes results/tracking_pilot/{domain}/{video}/standin/tracks.parquet (variant 'standin_blob').
For ants also writes variant 'standin_anttracker' converted from dataset/ants/v3/tracking/*.csv.
"""
import argparse
import time

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment

from common import (FPS, ROOT, annotated_window, iter_frames, out_dir, read_frame, video_info,
                    video_path)

MICE_SCALE = 0.25


def disk(r):
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y <= r * r).astype(np.uint8)


def background(path, scale, start, end, n=60, q=0.98, gray=True):
    """Per-pixel temporal quantile of n frames evenly spread over [start, end]."""
    frames = []
    for f in np.linspace(start + 30, end - 30, n).astype(int):
        im = read_frame(path, int(f), gray=gray)
        frames.append(cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                      if scale != 1 else im)
    return np.quantile(np.stack(frames), q, axis=0).astype(np.uint8)


# ----------------------------------------------------------------------------- mice detector
MICE_P = {'core_abs': 70, 'core_rel': 50, 'cand_abs': 150, 'cand_rel': 60, 'near_r': 6, 'close_r': 3,
          'open_r': 3, 'min_area': 250}


def mice_mask(g, bg, p=MICE_P):
    """g, bg: grey frame and background at MICE_SCALE. Black-fur core, grown by 'clearly darker
    than background' pixels next to it (shaved patches are grey, not black), closed, opened (cuts
    tails), components that contain core, holes filled."""
    gi, bi = g.astype(np.int16), bg.astype(np.int16)
    core = ((gi < p['core_abs']) & (bi - gi > p['core_rel'])).astype(np.uint8)
    core = cv2.morphologyEx(core, cv2.MORPH_OPEN, disk(1))
    near = cv2.dilate(core, disk(p['near_r']))
    cand = ((gi < p['cand_abs']) & (bi - gi > p['cand_rel']) & (near > 0)).astype(np.uint8)
    m = cv2.morphologyEx(core | cand, cv2.MORPH_CLOSE, disk(p['close_r']))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, disk(p['open_r']))
    lab, n = ndi.label(m)
    keep = np.zeros(n + 1, bool)
    keep[np.unique(lab[core > 0])] = True
    keep[0] = False
    return ndi.binary_fill_holes(keep[lab])


def blobs_from_mask(m, min_area, scale):
    """-> list of dicts in SOURCE pixels: cx, cy, w, h (axis-aligned bbox), heading (principal
    axis angle, rad, 180 deg ambiguous), area (source px^2)."""
    lab, n = ndi.label(m, structure=np.ones((3, 3), bool))
    out = []
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        if sl is None:
            continue
        yy, xx = np.nonzero(lab[sl] == i)
        if len(yy) < min_area:
            continue
        yy = yy + sl[0].start
        xx = xx + sl[1].start
        cy, cx = yy.mean(), xx.mean()
        cov = np.cov(np.stack([xx - cx, yy - cy])) if len(xx) > 2 else np.eye(2)
        w_, v = np.linalg.eigh(cov)
        out.append({'cx': (cx + 0.5) / scale - 0.5, 'cy': (cy + 0.5) / scale - 0.5,
                    'w': (sl[1].stop - sl[1].start) / scale, 'h': (sl[0].stop - sl[0].start) / scale,
                    'heading': float(np.arctan2(v[1, 1], v[0, 1])), 'area': len(yy) / scale ** 2})
    return out


# ----------------------------------------------------------------------------- ants detector
ANTS_P = {'diff_thr': 40, 'close_r': 3, 'min_area': 120, 'dish_shrink': 0.94}


def dish_mask(bg_bgr):
    """Largest bright circle = the dish floor (plaster). Fallback: centred circle."""
    g = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2GRAY)
    H, W = g.shape
    c = cv2.HoughCircles(cv2.medianBlur(g, 5), cv2.HOUGH_GRADIENT, dp=2, minDist=W, param1=60,
                         param2=40, minRadius=int(0.35 * W), maxRadius=int(0.5 * W))
    if c is not None:
        x, y, r = c[0, 0]
    else:
        x, y, r = W / 2, H / 2, 0.45 * W
    m = np.zeros_like(g, np.uint8)
    cv2.circle(m, (int(x), int(y)), int(r * ANTS_P['dish_shrink']), 1, -1)
    return m.astype(bool), (float(x), float(y), float(r))


def ants_mask(frame, bg, dish, p=ANTS_P):
    d = cv2.absdiff(frame, bg).max(axis=2)
    m = ((d > p['diff_thr']) & dish).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, disk(p['close_r']))
    return ndi.binary_fill_holes(m)


# ----------------------------------------------------------------------------- linker
def link(dets, gate, max_gap):
    """dets: list over time of (frame_src, list of blob dicts). Returns list of rows with track_id."""
    rows, active, next_id = [], {}, 0  # active: tid -> (last_frame, cx, cy)
    for f, blobs in dets:
        tids = list(active)
        if tids and blobs:
            P = np.array([[active[t][1], active[t][2]] for t in tids])
            Q = np.array([[b['cx'], b['cy']] for b in blobs])
            C = np.linalg.norm(P[:, None] - Q[None], axis=2)
            C2 = np.where(C > gate, 1e6, C)
            r, c = linear_sum_assignment(C2)
            pairs = [(tids[i], j) for i, j in zip(r, c) if C[i, j] <= gate]
        else:
            pairs = []
        used = {j for _, j in pairs}
        for t, j in pairs:
            b = blobs[j]
            active[t] = (f, b['cx'], b['cy'])
            rows.append({'frame_src': f, 'track_id': t, **b})
        for j, b in enumerate(blobs):
            if j not in used:
                active[next_id] = (f, b['cx'], b['cy'])
                rows.append({'frame_src': f, 'track_id': next_id, **b})
                next_id += 1
        for t in list(active):
            if f - active[t][0] > max_gap:
                del active[t]
    return rows


def to_contract(rows, variant):
    df = pd.DataFrame(rows)
    df['t_sec'] = df['frame_src'] / FPS
    df['detected'] = True
    df['conf'] = 1.0
    df['variant'] = variant
    return df


def run_mice(vid, stride):
    path = video_path('mice', vid)
    s, e = annotated_window('mice', vid)
    t0 = time.time()
    bg = background(path, MICE_SCALE, s, e)
    dets = []
    for f, g in iter_frames(path, stride=stride, scale=MICE_SCALE, gray=True, start=s, end=e + 1):
        dets.append((f, blobs_from_mask(mice_mask(g, bg), MICE_P['min_area'], MICE_SCALE)))
        if len(dets) % 3000 == 0:
            print(f'  {vid} frame {f} ({time.time() - t0:.0f}s)', flush=True)
    # gate: a mouse moves < ~1.5 body widths in one step; 0.25 s at stride 3/30 fps
    rows = link(dets, gate=150 * stride / 3, max_gap=stride)
    df = to_contract(rows, 'standin_blob')
    np.save(out_dir('mice', vid, 'standin') / 'background_q.npy', bg)
    return df


def run_ants(vid, stride):
    path = video_path('ants', vid)
    s, e = annotated_window('ants', vid)
    bg = background(path, 1.0, max(s, 1), e, gray=False, q=0.85)
    dish, circ = dish_mask(bg)
    dets = []
    for f, fr in iter_frames(path, stride=stride, start=s, end=e + 1):
        dets.append((f, blobs_from_mask(ants_mask(fr, bg, dish), ANTS_P['min_area'], 1.0)))
    rows = link(dets, gate=60 * stride / 3, max_gap=stride)
    df = to_contract(rows, 'standin_blob')
    d = out_dir('ants', vid, 'standin')
    cv2.imwrite(str(d / 'background.png'), bg)
    pd.Series({'x': circ[0], 'y': circ[1], 'r': circ[2]}).to_json(d / 'dish.json')
    return pd.concat([df, anttracker_tracks(vid)], ignore_index=True)


def anttracker_tracks(vid):
    """The existing AntTracker csv (5 fps, 512 px standardized video) as tracks: one track per
    identity it claims. Its rows are positions the heuristic tracker ASSIGNED (it duplicates
    centroids in merges and holds previous positions when an ant is missing), so detected is set
    to False when n_blobs < 3. track_id 0/1/2 = its focal/blue/yellow."""
    csv = ROOT / f'dataset/ants/v3/tracking/{vid}.csv'
    t = pd.read_csv(csv)
    W, H, _ = video_info(video_path('ants', vid))
    sc = W / 512.0
    s, _ = annotated_window('ants', vid)
    rows = []
    for tid, name in enumerate(['focal', 'blue', 'yellow']):
        rows.append(pd.DataFrame({
            # 5 fps clip of the source starting at start_frame: clip frame k <-> source frame 6k (+s)
            'frame_src': (t['frame_idx'] * 6 + s).astype(int), 'track_id': tid,
            'cx': (t[f'{name}_x'] + 0.5) * sc - 0.5, 'cy': (t[f'{name}_y'] + 0.5) * sc - 0.5,
            'w': 60.0, 'h': 60.0, 'heading': np.nan, 'area': np.nan,
            'detected': t['n_blobs'] >= 3}))
    df = pd.concat(rows, ignore_index=True)
    df['t_sec'] = df['frame_src'] / FPS
    df['conf'] = 1.0
    df['variant'] = 'standin_anttracker'
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--domain', required=True, choices=['mice', 'ants'])
    ap.add_argument('--video', required=True)
    ap.add_argument('--stride', type=int, default=3)
    a = ap.parse_args()
    df = run_mice(a.video, a.stride) if a.domain == 'mice' else run_ants(a.video, a.stride)
    p = out_dir(a.domain, a.video, 'standin') / 'tracks.parquet'
    df.to_parquet(p)
    for v, g in df.groupby('variant'):
        print(f'{a.domain}/{a.video} {v}: {len(g)} rows, {g.track_id.nunique()} tracks, '
              f'{g.frame_src.nunique()} frames -> {p}')


if __name__ == '__main__':
    main()
