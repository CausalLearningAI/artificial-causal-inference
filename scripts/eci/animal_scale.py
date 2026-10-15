"""
Animal scale per domain (label-free): body length / area / width in NATIVE pixels, from a simple pixel rule fitted per
video. Used to choose the encoder input size (patches per body length) for every domain.

    python scripts/eci/animal_scale.py measure --domain ants_v2     (one Slurm CPU job per domain, animal_scale.sh)
    python scripts/eci/animal_scale.py report                       (reads results/vision/animal_scale/<domain>.json)

Method per domain (DOMAINS below): V videos spread evenly over the sorted video list; per video 30 background frames
(evenly spaced in 5-95% of the video) and K measured frames (evenly spaced, shifted by half a step) are decoded at
NATIVE resolution (ffmpeg, grey). Per video:
  background  per-pixel QUANTILE over the background frames (not the median: an animal that rests in one spot for more
              than half of the sample would be absorbed by the median): 0.9 for dark animals, 0.1 for pale animals,
              0.5 for 'abs' (calms21: black and white mice).
  signal      d = bg - blur(grey) (dark), blur(grey) - bg (pale), |.| (abs); dark animals also need grey < 0.7 bg.
  threshold   fitted per video: Otsu over the pooled pixels with d > 15 grey levels (all measured frames), x OTSU_SCALE,
              never below 20.
  mask        threshold, closing (radius 0.3% of the frame side), hole filling, components below MIN_AREA_FRAC of the
              frame area dropped.
  count rule  a frame is KEPT when it has exactly n_animals components and none touches the frame border.
Per kept component: area (px), extent along the principal axis (= BODY LENGTH, max - min of the projection of its pixels,
px), extent along the minor axis (width), and the moment-ellipse axes 4 sqrt(lambda) (PCA). Twice: on the mask as is
('with appendages': tail, legs, antennae) and on the mask after an OPENING with a disk of radius 0.5 x the component's
largest inscribed radius ('body': removes tails / legs / antennae, keeps the largest piece).
Keypoints / trackers are used only as a CHECK (matched to the components by centroid, Hungarian), never to define.
Tokens: for each (patch p, patches per body k) the input side S = ceil(k side / L) p (full frame resized to S; a
non-square frame keeps its aspect), full-frame tokens = grid_w x grid_h, animal tokens = patches of the grid touched by
any kept component (mask with appendages) plus a 1-patch (3 x 3) dilation, median over kept frames.
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / 'results/vision/animal_scale'
FFMPEG = os.environ.get('AS_FFMPEG') or shutil.which('ffmpeg')
FFPROBE = os.environ.get('AS_FFPROBE') or shutil.which('ffprobe')
CAL = Path('/mnt/beegfs/locatgrp/shared/rcadei/benchmark/calms21')
MAB = REPO / 'dataset/benchmark/mabe22'
N_BG = 30
OTSU_SCALE = 0.7
MIN_AREA_FRAC = 3e-4
KS = (4, 6, 8, 12)
PS = (14, 16)

# name -> n animals, polarity, V videos, K measured frames per video, background quantile
DOMAINS = {
    'ants_v2': dict(n=3, pol='dark', V=8, K=30, q=0.9, src='data/ants/v2/observations/source/*.mkv'),
    'ants_v3': dict(n=3, pol='dark', V=8, K=30, q=0.9, src='data/ants/v3/observations/source/*.mkv'),
    'mice_v1': dict(n=4, pol='dark', mode='union_hull', V=8, K=30, q=0.9, src='data/mice/source/*_Test.mp4'),
    'frogs_v1': dict(n=1, pol='dark', mode='union_roi', V=8, K=30, q=0.9, src='data/frogs/v1/observations/source/*.mp4'),
    'mabe22_mouse': dict(n=3, pol='dark', mode='union_hull', V=30, K=30, q=0.9, src='dataset/benchmark/mabe22/videos/mouse_submission/*.avi'),
    'mabe22_beetle': dict(n=2, pol='pale', mode='pale_abs', min_area=2e-3, V=30, K=30, q=0.1, floor=8, pool=5, src='dataset/benchmark/mabe22/videos/beetle_submission/*.avi'),
    'calms21': dict(n=2, pol='abs', floor=70, min_area=2e-3, V=8, K=30, q=0.5, src=str(CAL / 'videos/task1_videos_mp4/train/*.mp4')),
}


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def probe(f):
    o = subprocess.check_output([FFPROBE, '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                                 'stream=width,height,r_frame_rate,duration:format=duration', '-of', 'json', str(f)])
    j = json.loads(o)
    s = j['streams'][0]
    a, b = s['r_frame_rate'].split('/')
    dur = float(s.get('duration') or j['format']['duration'])
    return int(s['width']), int(s['height']), float(a) / float(b), dur


def read_frame(f, t, W, H):
    o = subprocess.run([FFMPEG, '-v', 'error', '-ss', f'{t:.3f}', '-i', str(f), '-frames:v', '1', '-f', 'rawvideo',
                        '-pix_fmt', 'gray', '-'], capture_output=True).stdout
    if len(o) != W * H:
        return None
    return np.frombuffer(o, np.uint8).reshape(H, W)


def otsu(v):
    h = np.bincount(v.astype(np.uint8).ravel(), minlength=256).astype(np.float64)
    p = h / max(h.sum(), 1)
    w = np.cumsum(p)
    m = np.cumsum(p * np.arange(256))
    with np.errstate(all='ignore'):
        s = (m[-1] * w - m) ** 2 / (w * (1 - w))
    return int(np.nanargmax(s))


def disk(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def signal(g, bg, pol, sig):
    gb = cv2.GaussianBlur(g, (0, 0), sig).astype(np.float32)
    if pol == 'dark':
        d = bg - gb
        d[gb > 0.7 * bg] = 0
    elif pol == 'pale':
        d = gb - bg
    else:
        d = np.abs(gb - bg)
    return np.clip(d, 0, 255)



def hull_arena(bg):
    """Bright arena floor of a median background: largest component above Otsu, CONVEX HULL (a static animal in a corner
    is a hole / notch of the bright floor, the hull keeps it inside). Returns (bool mask, floor level)."""
    H, W = bg.shape
    k = 0.25 if W > 1000 else 1.0
    sm = cv2.resize(cv2.GaussianBlur(bg.astype(np.uint8), (0, 0), 2), None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    br = (sm > otsu(sm)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(br)
    i = 1 + int(np.argmax(st[1:, 4]))
    cs, _ = cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(max(cs, key=cv2.contourArea))
    m = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(m, (hull[:, 0, :] / k).astype(np.int32), 1)
    m0 = m = cv2.erode(m, disk(max(1, int(0.004 * W)))).astype(bool)
    fl = float(np.median(bg[m0 & (bg > otsu(sm))]))
    return m, fl


def frog_roi(stem, W, H, shrink=0.06):
    e = pd.read_csv(REPO / 'data/frogs/v1/experiment.csv')
    r = e[e.observation_id == stem].iloc[0]
    cy, cx = (r.roi_top + r.roi_bottom) / 2, (r.roi_left + r.roi_right) / 2
    ay, ax = (r.roi_bottom - r.roi_top) / 2 * (1 - shrink), (r.roi_right - r.roi_left) / 2 * (1 - shrink)
    m = np.zeros((H, W), np.uint8)
    cv2.ellipse(m, (int(cx), int(cy)), (int(ax), int(ay)), 0, 0, 360, 1, -1)
    return m.astype(bool)


def pale_floor(gb):
    """per-frame dish floor of a pale-animal-on-dark-dish video: component of (grey < 20) holding the frame centre, holes
    (the animals) filled, eroded. Returns (mask, floor grey level)."""
    H, W = gb.shape
    n, lab, st, _ = cv2.connectedComponentsWithStats((gb < 20).astype(np.uint8))
    ci = lab[H // 2, W // 2]
    fm = np.zeros((H, W), np.uint8)
    if ci > 0:
        cs, _ = cv2.findContours((lab == ci).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(fm, cs, -1, 1, -1)
    fm = cv2.erode(fm, disk(max(2, int(0.035 * W)))).astype(bool)
    if fm.any():  # only the central disk (0.8 x the floor's equivalent radius) is trusted: the rim glare and the
        # tilted dish wall leak into the floor component near the dish edge
        yy, xx = np.nonzero(fm)
        cy, cx, r = yy.mean(), xx.mean(), 0.8 * math.sqrt(fm.sum() / math.pi)
        Y, X = np.ogrid[:H, :W]
        fm = ((Y - cy) ** 2 + (X - cx) ** 2 < r * r) & fm
    return fm, (float(np.median(gb[fm])) if fm.any() else 0.0)


def components(mask, side, area_px):
    """closing + hole fill + min area; returns list of (bool crop, (y0, x0)), touches-border flag."""
    r = max(1, int(round(0.003 * side)))
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, disk(r))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    m2 = np.zeros_like(m)
    cv2.drawContours(m2, cs, -1, 1, -1)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m2, connectivity=8)
    out, border = [], False
    H, W = m2.shape
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < area_px or max(w, h) > 0.6 * side:  # too small, or frame-sized (dish rim / wall): not an animal
            continue
        out.append((lab[y:y + h, x:x + w] == i, (y, x)))
        border |= x == 0 or y == 0 or x + w == W or y + h == H
    return out, border, m2


def opened(c):
    dist = cv2.distanceTransform(np.pad(c, 1).astype(np.uint8), cv2.DIST_L2, 3)[1:-1, 1:-1]
    r = max(1, int(round(0.5 * dist.max())))
    o = cv2.morphologyEx(np.pad(c, r + 1).astype(np.uint8), cv2.MORPH_OPEN, disk(r))[r + 1:-(r + 1), r + 1:-(r + 1)]
    n, lab, st, _ = cv2.connectedComponentsWithStats(o, connectivity=8)
    if n < 2:
        return c
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return lab == i


def shape_stats(c, off):
    ys, xs = np.nonzero(c)
    ys, xs = ys + off[0], xs + off[1]
    p = np.stack([xs, ys], 1).astype(np.float64)
    mu = p.mean(0)
    cov = np.cov((p - mu).T) if len(p) > 2 else np.eye(2)
    w, v = np.linalg.eigh(cov)
    w, v = w[::-1], v[:, ::-1]
    pr = (p - mu) @ v
    ext = pr.max(0) - pr.min(0) + 1
    return dict(area=float(len(p)), length=float(ext[0]), width=float(ext[1]), ell_major=float(4 * math.sqrt(max(w[0], 0))),
                ell_minor=float(4 * math.sqrt(max(w[1], 0))), cx=float(mu[0]), cy=float(mu[1]), ax=v[:, 0].tolist(),
                pmin=float(pr[:, 0].min()), pmax=float(pr[:, 0].max()))


def tokens(mask, grid_w, grid_h):
    f = cv2.resize(mask.astype(np.float32), (grid_w, grid_h), interpolation=cv2.INTER_AREA) > 0
    d = cv2.dilate(f.astype(np.uint8), np.ones((3, 3), np.uint8))
    return int(f.sum()), int(d.sum())


# ----------------------------------------------------------------------------------------------------------------- checks
def pairs(cent, kp):
    """Hungarian match of component centroids (n, 2) to keypoint points (m, 2); returns index pairs + distances."""
    D = np.linalg.norm(cent[:, None] - kp[None], axis=2)
    r, c = linear_sum_assignment(D)
    return r, c, D[r, c]


class Checks:
    """Reference points per domain (CHECK only). get(video_name, t_sec, frame_idx) -> dict(points (m,2) native px,
    lengths (m,) native px or None) or None."""

    def __init__(self, dom, W):
        self.dom, self.W, self.ok = dom, W, False
        if dom in ('ants_v2', 'ants_v3'):
            self.ok = True
            self.dir = REPO / f"dataset/ants/{dom[-2:]}/tracking"
        elif dom == 'frogs_v1':
            self.ok = True
            self.dir = REPO / 'dataset/frogs/v1/sleap'
        elif dom == 'mabe22_mouse':
            self.ok = True
            self.kp = np.load(MAB / 'raw/mouse_submission_keypoints.npy', allow_pickle=True).item()['sequences']
        elif dom == 'mabe22_beetle':
            self.ok = True
            self.kp = np.load(MAB / 'raw/beetle_submission_keypoints.npy', allow_pickle=True).item()['sequences']
        elif dom == 'calms21':
            self.ok = True
            z = np.load(CAL / 'derived/task1_train.npz', allow_pickle=True)
            self.kp = z['keypoints']
            ln = z['lens']
            self.off = dict(zip([str(n) for n in z['names']], np.concatenate([[0], np.cumsum(ln)[:-1]])))
            self.len = dict(zip([str(n) for n in z['names']], ln))

    def get(self, stem, t, fi):
        d = self.dom
        if d.startswith('ants'):
            f = self.dir / f'{stem}.csv'
            if not f.exists():
                return None
            r = int(round(t * 5))
            tr = pd.read_csv(f)
            tr = tr[tr.frame_idx == r]
            if not len(tr):
                return None
            tr = tr.iloc[0]
            pts = np.array([[tr[f'{c}_x'], tr[f'{c}_y']] for c in ('blue', 'yellow', 'focal')], float) * self.W / 512
            return dict(points=pts, lengths=None)
        if d == 'frogs_v1':
            f = self.dir / f'{stem}.npz'
            if not f.exists():
                return None
            z = np.load(f, allow_pickle=True)
            nm = list(z['node_names'])
            r = int(round(t * 5))
            if r >= len(z['nodes']):
                return None
            raw = z['nodes'][r].astype(float)
            if np.isnan(raw).all():
                return None
            var = {}
            for sc, scn in ((800 / 512, '512->800'), (1.0, 'native')):
                for fx, fn in ((False, 'noflip'), (True, 'xflip')):
                    nd = raw * sc
                    if fx:
                        nd = nd.copy()
                        nd[:, 0] = self.W - 1 - nd[:, 0]
                    eye = np.nanmean(nd[[nm.index('Right_Eye'), nm.index('Left_Eye')]], 0)
                    L = np.linalg.norm(eye - nd[nm.index('Tail_Stem')])
                    var[f'{scn},{fn}'] = (np.nanmean(nd, 0)[None], np.array([L]) if np.isfinite(L) else None, None)
            return dict(variants=var)
        if d == 'mabe22_mouse':
            if stem not in self.kp:
                return None
            i = fi - 1
            k = self.kp[stem]['keypoints']
            if not 0 <= i < len(k):
                return None
            k = k[i].astype(float) * 224 / 512  # (3, 12, 2) (y, x)
            ok = ~(k == 0).all(2).any(1)
            pts = k.mean(1)[:, ::-1]
            L = np.linalg.norm(k[:, 0] - k[:, 9], axis=1)
            return dict(points=pts, lengths=L, valid=ok)
        if d == 'mabe22_beetle':
            if stem not in self.kp:
                return None
            i = fi - 1
            k = self.kp[stem]['keypoints']
            if not 0 <= i < len(k):
                return None
            k = k[i].astype(float).reshape(2, 2) * 224
            return dict(points=k, lengths=None, alt=k[:, ::-1])
        if d == 'calms21':
            if stem not in self.off or not 0 <= fi < self.len[stem]:
                return None
            k = self.kp[self.off[stem] + fi].astype(float)
            var = {}
            for nm_, kk in (('axis0=xy', k), ('axis0=mouse', k.transpose(1, 0, 2))):  # kk: (xy, mouse, 7)
                L = np.linalg.norm(kk[:, :, 0] - kk[:, :, 6], axis=0)
                var[nm_] = (kk.mean(2).T, L, None)
            return dict(variants=var)
        return None


# ------------------------------------------------------------------------------------------------------------------ measure
def pick(files, V):
    files = sorted(files)
    idx = np.unique(np.round(np.linspace(0, len(files) - 1, V)).astype(int))
    return [files[i] for i in idx]


def measure(dom, test=False):
    cfg = DOMAINS[dom]
    n_an = cfg['n']
    files = [Path(p) for p in sorted(REPO.glob(cfg['src']) if not cfg['src'].startswith('/') else Path('/').glob(cfg['src'][1:]))]
    files = [f for f in files if ' ' not in f.name]
    vids = pick(files, 3 if test else cfg['V'])
    K = 8 if test else cfg['K']
    OUT.mkdir(parents=True, exist_ok=True)
    stage = Path('/localhome') / os.environ.get('USER', 'u') / f"{os.environ.get('SLURM_JOB_ID', 'local')}_scale_{dom}"
    stage.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        local = []
        for v in vids:  # stage to local disk (sequential copy), following symlinks
            d = stage / v.name
            shutil.copyfile(v, d)
            local.append(d)
        log(dom, 'staged', len(local), 'videos', round(time.time() - t0), 's')
        rows, qa, ck_rows, mask_pool = [], [], [], []
        qa_rej = []
        ck_by = {}
        chk = None
        for v, lv in zip(vids, local):
            W, H, fps, dur = probe(lv)
            side = float(W)
            if chk is None:
                chk = Checks(dom, W)
            a, b = 0.05 * dur, 0.95 * dur
            tb = np.linspace(a, b, N_BG)
            st = (b - a) / (K - 1) / 2 if K > 1 else 0
            tm = np.linspace(a, b - 2 * st, K) + st
            if dom.startswith(('ants', 'frogs')):  # snap to the 5 fps grid of the trackers
                tm = np.round(tm * 5) / 5
            bgf = [x for x in (read_frame(lv, t, W, H) for t in tb) if x is not None]
            if len(bgf) < 10:
                log('skip', v.name, 'bg frames', len(bgf))
                continue
            bg = np.quantile(np.stack(bgf), cfg['q'], axis=0).astype(np.float32)
            frames = []
            for t in tm:
                g = read_frame(lv, t, W, H)
                if g is not None:
                    frames.append((t, g))
            sig = max(1.0, 0.002 * side)
            mode = cfg.get('mode', 'diff')
            floor_ = cfg.get('floor', 20)
            AR = None
            if mode in ('union_hull', 'union_roi'):
                if mode == 'union_hull':
                    AR, fl = hull_arena(bg)
                else:
                    AR = frog_roi(v.stem, W, H)
                    fl = float(np.median(bg[AR]))
            gbs = [cv2.GaussianBlur(g, (0, 0), sig).astype(np.float32) for _, g in frames]
            if mode == 'pale_abs':
                ds, sg, rings = [], [], []
                for gb in gbs:
                    fm, lv_ = pale_floor(gb)
                    ds.append(np.clip(gb - lv_, 0, 255) * fm)
                    rings.append(fm & ~cv2.erode(fm.astype(np.uint8), disk(max(2, int(0.03 * W)))).astype(bool))
                    x_ = (gb - lv_)[fm][::7]
                    sg.append(1.4826 * np.median(np.abs(x_ - np.median(x_))) if len(x_) else 0.0)
            else:
                ds = [signal(g, bg, cfg['pol'], sig) for _, g in frames]
            pool = np.concatenate([d[::4, ::4][d[::4, ::4] > cfg.get('pool', 15)] for d in ds]).astype(np.uint8)
            thr = max(float(floor_), min(OTSU_SCALE * (otsu(pool) if len(pool) > 100 else 0),
                                         0.45 * (np.percentile(pool, 99.5) if len(pool) > 100 else 0)))
            if mode == 'pale_abs':
                thr = max(10.0, 6 * float(np.median(sg)))  # 6 sigma of the dish-floor noise, at least 10 grey levels
            masks = None
            if mode in ('union_roi', 'union_hull'):
                darks = [(gb < 0.4 * fl) & AR for gb in gbs]
            area_px = cfg.get('min_area', MIN_AREA_FRAC) * W * H
            for frames_i, ((t, g), d) in enumerate(zip(frames, ds)):
                mk = d > thr
                if mode in ('union_roi', 'union_hull'):
                    mk = mk | darks[frames_i]
                comps, border, m2 = components(mk, side, area_px)
                if mode == 'pale_abs':  # components touching the rim band of the floor mask are dish rim, not animals
                    comps = [(c, o) for c, o in comps if not (c & rings[frames_i][o[0]:o[0] + c.shape[0], o[1]:o[1] + c.shape[1]]).any()]
                rec = dict(video=v.stem, t=float(t), n_comp=len(comps), border=bool(border), W=W, H=H, thr=float(thr))
                kept = len(comps) == n_an and not border
                rec['kept'] = kept
                if kept:
                    full = [shape_stats(c, o) for c, o in comps]
                    body = []
                    for c, o in comps:
                        oc = opened(c)
                        body.append(shape_stats(oc, o))
                    rec['full'], rec['body'] = full, body
                    # check
                    if chk.ok:
                        fi = int(round(t * fps))
                        ref = chk.get(v.stem, t, fi)
                        if ref is not None:
                            cent = np.array([[s_['cx'], s_['cy']] for s_ in body])
                            vs = ref['variants'] if 'variants' in ref else {'main': (ref['points'], ref.get('lengths'), ref.get('valid'))}
                            for vn, (pts, lens, okv) in vs.items():
                                if dom == 'mabe22_beetle':
                                    best = None
                                    for key, pp in (('xy', pts), ('yx', ref['alt'])):
                                        r_, c_, dd = pairs(cent, pp)
                                        if best is None or dd.mean() < best[1].mean():
                                            best = (key, dd, r_, c_)
                                    ck_by.setdefault(vn, []).append(dict(video=v.stem, t=float(t), dist=best[1].tolist(), order=best[0],
                                                        body_len=[body[i]['length'] for i in best[2]]))
                                    continue
                                r_, c_, dd = pairs(cent, pts)
                                rr = dict(video=v.stem, t=float(t), dist=dd.tolist(), body_len=[body[i]['length'] for i in r_],
                                          full_len=[full[i]['length'] for i in r_])
                                if lens is not None:
                                    ok = okv if okv is not None else np.ones(len(lens), bool)
                                    rr['ref_len'] = [float(lens[j]) if ok[j] else None for j in c_]
                                ck_by.setdefault(vn, []).append(rr)
                    allm = np.zeros((H, W), np.uint8)
                    for c, o in comps:
                        allm[o[0]:o[0] + c.shape[0], o[1]:o[1] + c.shape[1]] |= c.astype(np.uint8)
                    mask_pool.append((np.packbits(allm), H, W))
                    if len(qa) < 400:
                        qa.append((v.stem, t, g.copy(), comps, [s for s in body]))
                elif len(qa_rej) < 40 and (len(rows) % 3 == 0):
                    qa_rej.append((v.stem + ' REJ n=%d%s' % (len(comps), ' border' if border else ''), t, g.copy(), comps, []))
                rows.append(rec)
            log(dom, v.name, 'frames', len(frames), 'thr', round(thr, 1), 'kept so far',
                sum(r['kept'] for r in rows), '/', len(rows))
        # tokens per (p, k) from the pooled kept masks
        kept = [r for r in rows if r['kept']]
        res = dict(domain=dom, n_animals=n_an, polarity=cfg['pol'], n_videos=len(set(r['video'] for r in rows)),
                   n_frames=len(rows), n_kept=len(kept), videos=sorted(set(r['video'] for r in rows)))
        if not kept:
            qa_image(dom, qa, rows, qa_rej)
            (OUT / f'{dom}.json').write_text(json.dumps(res | dict(failed=True), indent=1))
            return
        W = int(np.median([r['W'] for r in rows]))
        H = int(np.median([r['H'] for r in rows]))
        res['frame_w'], res['frame_h'] = W, H
        for key in ('full', 'body'):
            for m in ('length', 'area', 'width', 'ell_major', 'ell_minor'):
                v = np.array([s[m] for r in kept for s in r[key]])
                res[f'{key}_{m}'] = dict(median=float(np.median(v)), q25=float(np.percentile(v, 25)),
                                         q75=float(np.percentile(v, 75)), p10=float(np.percentile(v, 10)),
                                         p90=float(np.percentile(v, 90)), n=int(len(v)))
        if n_an == 2:
            for nm, fn in (('larger', max), ('smaller', min)):
                v = [fn(r['body'], key=lambda q: q['area']) for r in kept]
                res['by_rank_' + nm] = dict(length_median=float(np.median([q['length'] for q in v])),
                                            area_median=float(np.median([q['area'] for q in v])),
                                            width_median=float(np.median([q['width'] for q in v])))
        L = res['body_length']['median']
        res['kept_frac'] = len(kept) / len(rows)
        res['frames_per_video_kept'] = {k: sum(1 for r in kept if r['video'] == k) for k in res['videos']}
        res['n_comp_hist'] = {str(k): int(sum(1 for r in rows if r['n_comp'] == k)) for k in range(0, 9)}
        res['thr'] = {k: float(np.median([r['thr'] for r in rows if r['video'] == k])) for k in res['videos']}
        # tokens
        tk = {}
        for p in PS:
            for k in KS + (None,):
                S = 448 if k is None else math.ceil(k * side_of(W) / L) * p
                gw = S // p if k is not None else 448 // p
                gh = max(1, round(gw * H / W))
                v = [tokens(np.unpackbits(mp)[:h_ * w_].reshape(h_, w_), gw, max(1, round(gw * h_ / w_))) for mp, h_, w_ in mask_pool[::max(1, len(mask_pool) // 60)]]
                v = np.array(v)
                tk[f'p{p}_' + ('448' if k is None else f'k{k}')] = dict(
                    S=int(S), grid_w=int(gw), grid_h=int(gh), full_tokens=int(gw * gh),
                    animal_tokens_touched=float(np.median(v[:, 0])), animal_tokens_dilated=float(np.median(v[:, 1])),
                    n_masks=len(v))
        res['tokens'] = tk
        # checks: the convention variant (scale / flip / axis layout) with the smallest median centroid distance
        if ck_by:
            vn = min(ck_by, key=lambda q: np.median([x for r in ck_by[q] for x in r['dist']]))
            ck_rows = ck_by[vn]
        if ck_rows:
            dd = np.array([x for r in ck_rows for x in r['dist']])
            ck = dict(variant=(vn if ck_by else ''), variants_tried=list(ck_by), n_frames=len(ck_rows), centroid_dist_px_median=float(np.median(dd)),
                      centroid_dist_over_body_len_median=float(np.median(dd) / L), centroid_dist_p90_px=float(np.percentile(dd, 90)))
            if dom == 'mabe22_beetle':
                ck['order_counts'] = {o: sum(r['order'] == o for r in ck_rows) for o in ('xy', 'yx')}
            if 'ref_len' in ck_rows[0]:
                pr = [(a, b) for r in ck_rows for a, b in zip(r['body_len'], r.get('ref_len', [])) if b]
                pf = [(a, b) for r in ck_rows for a, b in zip(r['full_len'], r.get('ref_len', [])) if b]
                if pr:
                    a_, b_ = np.array(pr).T
                    ck.update(ref_len_median_px=float(np.median(b_)), ref_len_iqr=np.percentile(b_, [25, 75]).tolist(),
                              ref_len_n=int(len(b_)), body_over_ref_median=float(np.median(a_ / b_)),
                              body_len_matched_median_px=float(np.median(a_)),
                              fulllen_over_ref_median=float(np.median(np.array(pf)[:, 0] / np.array(pf)[:, 1])))
            res['check'] = ck
        (OUT / f'{dom}.json').write_text(json.dumps(res, indent=1))
        qa_image(dom, qa, rows, qa_rej)
        arena(dom, vids[0], local[0], cfg, res)
        (OUT / f'{dom}.json').write_text(json.dumps(res, indent=1))
        log(dom, 'done', json.dumps({k: res[k] for k in ('n_frames', 'n_kept', 'body_length', 'full_length')}))
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def side_of(W):
    return float(W)


def qa_image(dom, qa, rows, rej=(), n=6, tile=470):
    if not qa and not rej:
        return
    vs = sorted(set(q[0] for q in qa))
    sel, seen = [], {}
    for i, q in enumerate(qa):
        seen.setdefault(q[0], []).append(i)
    order = []
    j = 0
    while len(order) < n and any(j < len(v) for v in seen.values()):
        for k in vs:
            if j < len(seen[k]) and len(order) < n:
                order.append(seen[k][j * max(1, len(seen[k]) // 3) % len(seen[k])])
        j += 1
    tiles = []
    allq = [qa[i] for i in order[:n]] + list(rej[::max(1, len(rej) // 3)][:3])
    for stem, t, g, comps, body in allq:
        H, W = g.shape
        s = tile / max(H, W)
        img = cv2.cvtColor(cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2BGR)
        for (c, o) in comps:
            cm = np.zeros((H, W), np.uint8)
            cm[o[0]:o[0] + c.shape[0], o[1]:o[1] + c.shape[1]] = c
            cs, _ = cv2.findContours(cv2.resize(cm, img.shape[1::-1], interpolation=cv2.INTER_NEAREST), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(img, cs, -1, (0, 255, 0), 1)
        for b in body:
            ax = np.array(b['ax'])
            mu = np.array([b['cx'], b['cy']])
            p0, p1 = mu + ax * b['pmin'] + 0 * ax, mu + ax * b['pmax']
            # pmin/pmax are projections of pixels relative to the mean: endpoints = mu + ax * pmin / pmax
            cv2.line(img, tuple((p0 * s).astype(int)), tuple((p1 * s).astype(int)), (0, 0, 255), 2)
            ax2 = np.array([-ax[1], ax[0]])
            cv2.line(img, tuple(((mu - ax2 * b['width'] / 2) * s).astype(int)), tuple(((mu + ax2 * b['width'] / 2) * s).astype(int)), (255, 128, 0), 1)
        cv2.putText(img, f'{stem[:18]} t={t:.0f}s L={[round(b["length"]) for b in body]}', (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        canvas = np.zeros((tile, tile, 3), np.uint8)
        canvas[:img.shape[0], :img.shape[1]] = img
        tiles.append(canvas)
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)])
    cv2.imwrite(str(OUT / f'qa_{dom}.png'), grid)


def arena(dom, v, lv, cfg, res):
    """Approximate arena size (native px), label-free, from the background of the first video."""
    try:
        W, H, fps, dur = probe(lv)
        fr = [read_frame(lv, t, W, H) for t in np.linspace(0.1 * dur, 0.9 * dur, 15)]
        fr = [x for x in fr if x is not None]
        bg = np.quantile(np.stack(fr), 0.5, axis=0).astype(np.uint8)
        k = 0.25 if W > 1000 else 1.0
        s = cv2.resize(bg, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        info = {}
        if dom.startswith('ants'):
            c = cv2.HoughCircles(cv2.GaussianBlur(s, (0, 0), 3), cv2.HOUGH_GRADIENT, 1, 100, param1=100, param2=25,
                                 minRadius=int(0.3 * W * k), maxRadius=int(0.52 * W * k))
            if c is not None:
                info = dict(method='Hough circle on the median background', diameter_px=float(2 * c[0, 0, 2] / k))
        elif dom == 'frogs_v1':
            e = pd.read_csv(REPO / 'data/frogs/v1/experiment.csv')
            e = e[e.valid == 1]
            w, h = e.roi_right - e.roi_left, e.roi_bottom - e.roi_top
            info = dict(method='dish oval ROI of data/frogs/v1/experiment.csv (ImageJ), median over videos',
                        diameter_px=float(np.median((w + h) / 2)), width_px=float(w.median()), height_px=float(h.median()))
        elif dom == 'mice_v1':  # bedding = the textured region
            sd = cv2.GaussianBlur(s.astype(np.float32) ** 2, (0, 0), 6) - cv2.GaussianBlur(s.astype(np.float32), (0, 0), 6) ** 2
            tx = np.sqrt(np.clip(sd, 0, None))
            m = (tx > otsu(np.clip(tx * 4, 0, 255)) / 4).astype(np.uint8)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, disk(5))
            n, lab, st, _ = cv2.connectedComponentsWithStats(m)
            i = 1 + int(np.argmax(st[1:, 4]))
            info = dict(method='bbox of the largest textured (local std) region of the median background',
                        width_px=float(st[i, 2] / k), height_px=float(st[i, 3] / k))
        elif dom == 'mabe22_mouse':
            m = (s > otsu(s)).astype(np.uint8)
            n, lab, st, _ = cv2.connectedComponentsWithStats(m)
            i = 1 + int(np.argmax(st[1:, 4]))
            info = dict(method='bbox of the largest bright region of the median background (Otsu)',
                        width_px=float(st[i, 2]), height_px=float(st[i, 3]))
        elif dom == 'mabe22_beetle':
            m = (s > 0.5 * np.percentile(s, 99.5)).astype(np.uint8)
            n, lab, st, _ = cv2.connectedComponentsWithStats(m)
            i = 1 + int(np.argmax(st[1:, 4]))
            info = dict(method='bbox of the bright dish rim (largest bright component, median background)',
                        width_px=float(st[i, 2]), height_px=float(st[i, 3]))
        elif dom == 'calms21':
            sd = cv2.GaussianBlur(s.astype(np.float32) ** 2, (0, 0), 6) - cv2.GaussianBlur(s.astype(np.float32), (0, 0), 6) ** 2
            info = dict(method='not measured (whole frame is the cage view)', width_px=float(W), height_px=float(H))
        res['arena'] = info
    except Exception as e:  # noqa
        res['arena'] = dict(error=repr(e))


# ------------------------------------------------------------------------------------------------------------------ report
ORDER = ['ants_v2', 'ants_v3', 'mice_v1', 'frogs_v1', 'mabe22_mouse', 'mabe22_beetle', 'calms21']


def report():
    R = {d: json.loads((OUT / f'{d}.json').read_text()) for d in ORDER if (OUT / f'{d}.json').exists()}
    summ = {}
    L = []
    L.append('# Animal scale per domain (native pixels, label-free)\n')
    L.append('Script scripts/eci/animal_scale.py. Body length = extent of the OPENED component (tail / legs / antennae removed) along its '
             'principal (PCA) axis; frames kept only when the component count equals the animal count and no component touches the '
             'frame border. Medians over kept animals; IQR = 25-75%, 10-90% = range.\n')
    L.append('| domain | frame (px) | arena (px, approx) | n animals | videos / frames / kept | body length px med [IQR] (10-90%) | with appendages px | body area px | width px | length / frame side | patches per body at 448 (p14) |')
    L.append('|---|---|---|---|---|---|---|---|---|---|---|')
    for d, r in R.items():
        if r.get('failed'):
            L.append(f'| {d} | segmentation FAILED: 0 kept frames of {r["n_frames"]} |' + ' |' * 9)
            continue
        b, f, a, w = r['body_length'], r['full_length'], r['body_area'], r['body_width']
        side = r['frame_w']
        ar = r.get('arena', {})
        arena_s = ar.get('diameter_px') or (f"{ar.get('width_px', 0):.0f} x {ar.get('height_px', 0):.0f}" if ar.get('width_px') else 'n/a')
        arena_s = f'{arena_s:.0f}' if isinstance(arena_s, float) else arena_s
        L.append(f"| {d} | {r['frame_w']} x {r['frame_h']} | {arena_s} | {r['n_animals']} | {r['n_videos']} / {r['n_frames']} / {r['n_kept']} "
                 f"| {b['median']:.0f} [{b['q25']:.0f}-{b['q75']:.0f}] ({b['p10']:.0f}-{b['p90']:.0f}) | {f['median']:.0f} [{f['q25']:.0f}-{f['q75']:.0f}] "
                 f"| {a['median']:.0f} | {w['median']:.0f} | {b['median'] / side:.3f} | {b['median'] / side * 32:.1f} |")
        summ[d] = dict(frame=[r['frame_w'], r['frame_h']], arena=ar, n_animals=r['n_animals'], n_frames=r['n_frames'], n_kept=r['n_kept'],
                       body_length_px=b, length_with_appendages_px=f, body_area_px=a, body_width_px=w, body_ell_major_px=r['body_ell_major'],
                       length_over_frame_side=b['median'] / side, patches_per_body_448_p14=b['median'] / side * 32,
                       patches_per_body_448_p16=b['median'] / side * 28, tokens=r['tokens'], check=r.get('check'))
    L.append('\n## Input side for k patches per body length (side = frame width resized to S; full-frame tokens; animal tokens = patches touched by a kept mask + 1-patch dilation, median)\n')
    L.append('| domain | patch | k | S (px) | full-frame tokens | animal tokens measured | animal tokens analytic |')
    L.append('|---|---|---|---|---|---|---|')
    for d, r in R.items():
        if r.get('failed'):
            continue
        Lb, Wd, side = r['body_length']['median'], r['body_width']['median'], r['frame_w']
        for p in PS:
            for k in KS:
                t = r['tokens'][f'p{p}_k{k}']
                lp, wp = k, Wd / Lb * k
                an = r['n_animals'] * math.pi / 4 * (lp + 2) * (wp + 2)
                L.append(f"| {d} | {p} | {k} | {t['S']} | {t['full_tokens']} ({t['grid_w']}x{t['grid_h']}) | {t['animal_tokens_dilated']:.0f} (touched {t['animal_tokens_touched']:.0f}) | {an:.0f} |")
            t = r['tokens'][f'p{p}_448']
            L.append(f"| {d} | {p} | at 448 | 448 | {t['full_tokens']} ({t['grid_w']}x{t['grid_h']}) | {t['animal_tokens_dilated']:.0f} (touched {t['animal_tokens_touched']:.0f}) | |")
    L.append('\n## Checks against keypoints / trackers (reference only)\n')
    L.append('| domain | n frames | median centroid distance px | in body lengths | reference length px med [IQR] | segmentation body length / reference length | note |')
    L.append('|---|---|---|---|---|---|---|')
    for d, r in R.items():
        c = r.get('check')
        if not c:
            L.append(f'| {d} | no keypoints / tracker available | | | | | |')
            continue
        L.append(f"| {d} | {c['n_frames']} | {c['centroid_dist_px_median']:.1f} | {c['centroid_dist_over_body_len_median']:.2f} | "
                 + (f"{c['ref_len_median_px']:.0f} [{c['ref_len_iqr'][0]:.0f}-{c['ref_len_iqr'][1]:.0f}] (n {c['ref_len_n']}) | {c['body_over_ref_median']:.2f} (matched body {c['body_len_matched_median_px']:.0f} px) |"
                    if 'ref_len_median_px' in c else '| | |') + (f" orders {c['order_counts']}" if 'order_counts' in c else '') + ' |')
    L.append('\n## Two-animal domains: larger / smaller animal of each frame (by component area), body length px median\n')
    for d, r in R.items():
        if 'by_rank_larger' in r:
            L.append(f"- {d}: larger {r['by_rank_larger']['length_median']:.0f} px (area {r['by_rank_larger']['area_median']:.0f}), smaller {r['by_rank_smaller']['length_median']:.0f} px (area {r['by_rank_smaller']['area_median']:.0f})")
            summ[d]['by_rank'] = dict(larger=r['by_rank_larger'], smaller=r['by_rank_smaller'])
    (OUT / 'scale.md').write_text('\n'.join(L) + '\n')
    (OUT / 'scale.json').write_text(json.dumps(summ, indent=1))
    print('\n'.join(L))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['measure', 'report'])
    ap.add_argument('--domain', default='ants_v2')
    ap.add_argument('--test', action='store_true')
    a = ap.parse_args()
    measure(a.domain, a.test) if a.cmd == 'measure' else report()
