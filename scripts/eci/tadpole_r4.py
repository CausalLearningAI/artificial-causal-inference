"""
R4 for the frog TADPOLES (stage 44-48, data/frogs/v1_tadpole/experiment.csv): the r4 recipe (scripts/eci/rollout_r4.py)
on a TRACKER-FREE, label-free body crop, with the tadpoles' own SAE (domain key 'frogs_tadpole').

Why a crop: a tadpole is ~38 source px long and 3-10 wide in the ~800 px frame (scripts/eci/tadpoles_crops.py), i.e.
~2.4 DINOv3 patches per body at native size; the r4 rule is ~10 patches per body length.

Videos: the valid rows of groups WT, FoxP1, En1 (bilateral crispants; Scrambled dropped), every 5 fps frame of the
raw 60 fps video (source frame 12 k + 5, as the juvenile r4 and the SLEAP 5 fps tables).

Steps (scripts/eci/tadpole_r4.sh runs each in Slurm from a `git archive` of a committed revision)
    plan      (CPU) videos, layout (annotations.csv / experiment.csv on BeeGFS, symlinked as dataset/frogs/eci_tadpole_r4
              = src/eci/domain.py TadpolesDomain with TADPOLE_TAG=_r4), held-out split (r4 rule), prepass bursts balanced
              per GROUP (same number of training frames per group), encode shards, the observation-design comparison
              with the juveniles -> OUT/plan.json, OUT/design.json
    locate    (CPU, one array task per video) label-free blob track: ffmpeg rgb24 -> grey (PIL weights) of every 5 fps
              frame; background in two stages (stage 1: per-pixel BG_Q1 quantile of N_BG evenly spread frames; the
              tadpole blob of each sample frame (stage-1 rule) dilated by R_EX px is left out; stage 2: per-pixel BG_Q
              quantile of the remaining samples, pixels with < MIN_FREE samples filled from neighbours); tadpole pixels
              = darker than the background by > DARK_REL inside the dish ROI (experiment.csv roi_*); blob = the largest
              8-connected component with >= MIN_BLOB px; centroid = its pixel mean. Frames without a blob reuse the last
              centroid (before the first blob: the first one). Smoothed centroid = running median over +-SMOOTH frames
              (x and y separately) of that track. NO SLEAP input. -> BEE/locate/<obs>.npz
    validate  (CPU) blob-found fraction, crop centre vs SLEAP (dataset/frogs/v1_tadpole/sleap, ONLY as a check: centre
              distance, fraction of predicted SLEAP nodes inside the crop box), a sheet of 24 random crops ->
              OUT/crop_validation.json, OUT/crop_sheet.jpg
    prepass   (GPU) the bursts: own crops of frames t and the lag crops (frame t - 2 s cut with the box of frame t),
              kept tokens -> BEE/prepass/ (the r4 prepass format), OUT/prepass.json, OUT/mask_sheet.jpg
    train / merge / ftheta / summaries   rollout_r4.py's steps, unchanged, on domain 'frogs_tadpole'
    encode    (GPU, one task per shard) full pass -> BEE/videos/<obs>.npz (the r4 per-video format)
    manifest  (CPU) OUT/manifest.json (paths + commits for the atlas builder)

Crop: SIDE x SIDE source px (edge-padded frame), centred on round(smoothed centroid), not rotated, upsampled to S = 384
(bicubic) = 24 x 24 patches of 16. Mask = the frogs pixel rule on the crop (FG_RULE_TADPOLES: dark_rel 15, core = patch
dark fraction > 0.05, 1-patch dilation onto patches with any dark pixel) WITHOUT the near-SLEAP-node restriction (that
rule's own fallback for frames without nodes), on the 384 px crop (grey of the upsampled crop vs the bilinear-upsampled
background crop, ROI nearest). Delta 2 s: z_t - z'_{t-10}, z' = the tokens of frame t - 10 cut with frame t's box (the
same window, so the difference is appearance, not position); when the two boxes are equal z' is frame t - 10's own
tokens (no second pass).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[2]  # the code tree (a git archive in a job)
REPO = Path('/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference')  # data / results
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'scripts/eci'))
os.environ.setdefault('R4_OUT', str(REPO / 'results/vision/eci_rollout_r4'))
import rollout_r4 as R  # noqa: E402

DOM = 'frogs_tadpole'
OUT = R.OUT / DOM
BEE = R.BEE / DOM
TAG = '_r4'  # TadpolesDomain TADPOLE_TAG -> dataset/frogs/eci_tadpole_r4, results/vision/frogs/eci_tadpole_r4/nes
LAYOUT_LINK = REPO / f'dataset/frogs/eci_tadpole{TAG}'
EXP = REPO / 'data/frogs/v1_tadpole/experiment.csv'
GROUPS = ('WT', 'FoxP1', 'En1')
STEP, OFFSET, FPS = 12, 5, 60
SIDE, S = 96, 384
G = S // 16
N_BG, BG_Q1, BG_Q, R_EX, MIN_FREE = 200, 0.98, 0.9, 10, 5
# v2 (2026-10-16 08:00, after the v1 validation: 6.5% of frames > 48 px from SLEAP): MAX_EXTENT rejects the rim strip /
# speckle components; SMOOTH 5 -> 1 (the +-5 median lagged real fast swims by ~50 px where the raw blob was right)
DARK_REL, MIN_BLOB, SMOOTH, DEADBAND, MAX_EXTENT = 15, 20, 1, 2, 70
LOCATE = 'locate_v2'
DARK_FRAC = 0.05
FRAMES_TARGET = 12_000
log = R.log


def videos_df():
    E = pd.read_csv(EXP, dtype={'session': str, 'substage': str, 'line': str})
    E = E[(E.valid == 1) & E.group.isin(GROUPS)].reset_index(drop=True)
    assert (E.hflip == 0).all() and (E.start_frame == 0).all()
    return E


def n5_of(n_src):
    return (int(n_src) - OFFSET + STEP - 1) // STEP


# ---------------------------------------------------------------------------------------------- plan
def design_match():
    """Observation design of the tadpoles (118 videos) vs the juveniles (data/frogs/v1/experiment.csv)."""
    T = videos_df()
    J = pd.read_csv(REPO / 'data/frogs/v1/experiment.csv', dtype={'session': str})
    J = J[J.valid == 1]
    Tall = pd.read_csv(EXP, dtype={'session': str})
    out = {'tadpole': {}, 'juvenile': {}}
    for k, d in (('tadpole', T), ('juvenile', J)):
        ct = pd.crosstab(d.session, d.group)
        out[k] = {'n_videos': int(len(d)), 'per_group': d.group.value_counts().to_dict(),
                  'n_sessions': int(d.session.nunique()),
                  'sessions_per_group': d.groupby('group').session.nunique().to_dict(),
                  'sessions_with_more_than_one_group': int((ct > 0).sum(1).gt(1).sum()),
                  'dates': sorted(d.date.unique().tolist()), 'videos_per_session': ct.sum(1).describe().to_dict(),
                  'animal_id_recorded': int(d.animal_id.notna().sum())}
    out['tadpole']['arms'] = 'WT (T 0), FoxP1 (T 1), En1 (T 2): whole-embryo (bilateral) CRISPR crispants'
    out['juvenile']['arms'] = 'WT, FoxP1, En1: half-embryo (unilateral) crispants, mutant_side ' + json.dumps(
        {f'{g}|{s}': int(n) for (g, s), n in J.groupby(['group', J.mutant_side.fillna('none')]).size().items()})
    out['tadpole']['substage_by_group'] = {g: d.substage.value_counts().to_dict() for g, d in T.groupby('group')}
    out['tadpole']['dropped'] = {'Scrambled_valid': int(((Tall.group == 'Scrambled') & (Tall.valid == 1)).sum()),
                                 'invalid_WT_FoxP1_En1': int((Tall.group.isin(GROUPS) & (Tall.valid != 1)).sum())}
    ta, ja = set(T.animal_id.dropna().astype(int)), set(J.animal_id.dropna().astype(int))
    out['animal_overlap'] = {'tadpole_ids': len(ta), 'juvenile_ids': len(ja), 'common': sorted(ta & ja)}
    out['session_overlap'] = sorted(set(T.session) & set(J.session))
    out['date_overlap'] = sorted(set(T.date) & set(J.date))
    out['frame'] = {'tadpole_sizes': {f'{w}x{h}': int(n) for (w, h), n in T.groupby(['width', 'height']).size().items()},
                    'source_fps': sorted(T.source_fps.unique().tolist()),
                    'n5_frames': {'min': int(T.source_frames.map(n5_of).min()), 'max': int(T.source_frames.map(n5_of).max())}}
    return out


def cmd_plan(args):
    E = videos_df()
    lay = BEE / 'layout'
    lay.mkdir(parents=True, exist_ok=True)
    videos, rows, lo = [], [], 0
    T_of = {'WT': 0, 'FoxP1': 1, 'En1': 2}
    for _, r in E.iterrows():
        n = n5_of(r.source_frames)
        videos.append({'obs': r.observation_id, 'group': r.group, 'src_lo': lo, 'lay_lo': lo, 'n': n,
                       'path': r.source_file, 'w': int(r.width), 'h': int(r.height), 'hflip': 0, 'offset': OFFSET,
                       'src_frames': int(r.source_frames)})
        k = np.arange(n)
        rows.append(pd.DataFrame({'observation_id': r.observation_id, 'frame_idx': k, 'fps': 5, 'group': r.group,
                                  'T': T_of[r.group], 'source_frame': STEP * k + OFFSET}))
        lo += n
    A = pd.concat(rows, ignore_index=True)
    # held-out split (rollout_r4.cmd_plan rule): every ~6th video, alternating calibration / evaluation
    rng = np.random.default_rng(0)
    order = rng.permutation(len(videos))
    n_held = max(6, int(round(len(videos) / 6)))
    n_held += n_held % 2
    for j, i in enumerate(order):
        videos[i]['role'] = 'calib' if j < n_held // 2 else ('eval' if j < n_held else 'train')
    # prepass bursts, balanced per group: FRAMES_TARGET / 3 frames per group (4 frames per burst)
    ng = E.group.value_counts().to_dict()
    for i, v in enumerate(videos):
        n_b = int(np.ceil(FRAMES_TARGET / len(GROUPS) / (4 * ng[v['group']])))
        g = np.random.default_rng([1, i])
        span = v['n'] - 26
        a = np.floor((np.arange(n_b) + g.random(n_b)) * span / n_b).astype(int)
        assert a.min() >= 0 and a.max() + 25 < v['n']
        v['bursts'] = a.tolist()
    n = np.array([v['n'] for v in videos])
    shards = [list(range(i, min(i + args.per_shard, len(videos)))) for i in range(0, len(videos), args.per_shard)]
    A.to_csv(lay / 'annotations.csv', index=False)
    exp_cols = ['observation_id', 'group', 'line', 'T', 'session', 'date', 'time', 'slot', 'hour', 'animal_id',
                'substage', 'width', 'height', 'hflip']
    Ex = E[exp_cols].copy()
    Ex['T'] = Ex.group.map(T_of)
    Ex.to_csv(lay / 'experiment.csv', index=False)
    (lay / 'codes').mkdir(exist_ok=True)
    (lay / 'sae').mkdir(exist_ok=True)
    if not LAYOUT_LINK.exists():
        os.symlink(lay, LAYOUT_LINK)
    plan = {'domain': DOM, 'G': G, 'S': S, 'side': SIDE, 'n_videos': len(videos), 'n_frames': int(n.sum()),
            'roles': {r: sum(v['role'] == r for v in videos) for r in ('train', 'calib', 'eval')},
            'roles_by_group': {g: {r: sum(v['role'] == r and v['group'] == g for v in videos)
                                   for r in ('train', 'calib', 'eval')} for g in GROUPS},
            'prepass_frames_by_group': {g: 4 * sum(len(v['bursts']) for v in videos if v['group'] == g) for g in GROUPS},
            'shards': shards, 'shard_frames': [int(n[s].sum()) for s in shards], 'layout_dir': str(lay),
            'videos': videos}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'plan.json').write_text(json.dumps(plan, indent=1))
    D = design_match()
    (OUT / 'design.json').write_text(json.dumps(D, indent=1, default=str))
    log(f'{len(videos)} videos, {n.sum():,} frames, roles {plan["roles"]}, prepass frames by group '
        f'{plan["prepass_frames_by_group"]}, {len(shards)} shards')
    log(json.dumps(D, default=str)[:3000])


# ---------------------------------------------------------------------------------------------- decode
def decode_iter(path, w, h, s0=0, n=None, threads=4):
    """rgb24 native frames of the 5 fps grid (source frames 12 k + 5, k >= s0), at most n of them; yields (h, w, 3).
    s0 = 0: no seek, select mod(n, 12) = 5; s0 > 0: accurate input seek to source frame 12 s0 + 5 (rollout_r4.decode:
    -ss (src0 - 0.5) / fps, the first frame kept has pts >= that time), then every 12th frame."""
    src0 = STEP * s0 + OFFSET
    cmd = [str(R.FFMPEG), '-nostdin', '-v', 'error', '-threads', str(threads)]
    if s0 > 0:
        cmd += ['-ss', f'{(src0 - 0.5) / FPS:.6f}']
        vf = f"select='not(mod(n\\,{STEP}))'"
    else:
        vf = f"select='eq(mod(n\\,{STEP})\\,{OFFSET})'"
    cmd += ['-i', str(path), '-an', '-sn', '-vf', vf, '-fps_mode', 'passthrough']
    if n is not None:
        cmd += ['-frames:v', str(n)]
    cmd += ['-pix_fmt', 'rgb24', '-f', 'rawvideo', '-']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=w * h * 3 * 4)
    fb = w * h * 3
    k = 0
    while n is None or k < n:
        b = p.stdout.read(fb)
        if len(b) < fb:
            break
        yield np.frombuffer(b, np.uint8).reshape(h, w, 3)
        k += 1
    p.stdout.close()
    err = p.stderr.read().decode()[-400:]
    rc = p.wait()
    if rc != 0 or (n is not None and k != n):
        raise RuntimeError(f'decode {path} s0 {s0} n {n}: rc {rc}, {k} frames; {err}')


def grey_np(x):
    """(..., 3) uint8 -> PIL convert('L') grey uint8."""
    t = x.astype(np.uint32)
    return ((t[..., 0] * 19595 + t[..., 1] * 38470 + t[..., 2] * 7471 + 32768) >> 16).astype(np.uint8)


# ---------------------------------------------------------------------------------------------- locate
def blob(diff_dark, min_px=MIN_BLOB, max_extent=MAX_EXTENT):
    """bool (h, w) dark-pixel map -> (found, cx, cy, area, component mask or None) of the largest 8-connected
    component with >= min_px pixels whose bounding box is at most max_extent px on its longer side (a tadpole is
    22-60 px long; v2 measured the false blobs of v1 = the dark dish rim strip inside the ROI oval and diffuse
    lighting speckle, both components > 100 px across)."""
    from scipy import ndimage
    lab, nl = ndimage.label(diff_dark, structure=np.ones((3, 3), bool))
    if nl == 0:
        return False, np.nan, np.nan, 0, None
    cnt = np.bincount(lab.ravel())
    cnt[0] = 0
    sl = ndimage.find_objects(lab)
    ext = np.zeros(len(cnt), np.int64)
    for j, s_ in enumerate(sl, start=1):
        if s_ is not None:
            ext[j] = max(s_[0].stop - s_[0].start, s_[1].stop - s_[1].start)
    ok = (cnt >= min_px) & (ext <= max_extent)
    if not ok.any():
        return False, np.nan, np.nan, int(cnt.max()), None
    j = int(np.where(ok, cnt, -1).argmax())
    yy, xx = np.nonzero(lab[sl[j - 1]] == j)
    m = np.zeros(diff_dark.shape, bool)
    m[sl[j - 1]] = lab[sl[j - 1]] == j
    return True, float(xx.mean() + sl[j - 1][1].start + 0.5), float(yy.mean() + sl[j - 1][0].start + 0.5), int(cnt[j]), m


def smooth_track(cx, cy, found, r=SMOOTH):
    """Held track (last centroid where no blob; before the first blob: the first) -> running median over +-r frames."""
    from scipy.ndimage import median_filter
    idx = np.arange(len(cx))
    if not found.any():
        raise ValueError('no blob in any frame')
    good = idx[found]
    last = good[np.clip(np.searchsorted(good, idx, side='right') - 1, 0, len(good) - 1)]
    hx, hy = cx[last], cy[last]
    sx = median_filter(hx, size=2 * r + 1, mode='nearest')
    sy = median_filter(hy, size=2 * r + 1, mode='nearest')
    return hx, hy, sx, sy


def boxes_of(sx, sy, band=DEADBAND):
    """Smoothed centroid -> crop box top-left (source px of the un-padded frame, may be < 0). Dead band: the box
    centre stays where it is until the smoothed centroid is more than `band` px away from it (Chebyshev), then jumps to
    the rounded centroid; a still tadpole keeps one box (no pixel jitter of the crop, and the lag frame's own tokens
    serve the 2 s difference), the body is never more than `band` px off the crop centre by the dead band."""
    cx = np.empty(len(sx), np.int32)
    cy = np.empty(len(sy), np.int32)
    ux, uy = int(np.round(sx[0])), int(np.round(sy[0]))
    for i in range(len(sx)):
        if abs(sx[i] - ux) > band or abs(sy[i] - uy) > band:
            ux, uy = int(np.round(sx[i])), int(np.round(sy[i]))
        cx[i], cy[i] = ux, uy
    return cx - SIDE // 2, cy - SIDE // 2


def cmd_locate(args):
    from tadpoles_crops import roi_mask
    from frogs_background import fill_pixels
    import torch
    from scipy import ndimage
    P = R.load_plan(DOM)
    E = videos_df().set_index('observation_id')
    vids = [args.task] if args.task is not None else range(len(P['videos']))
    outd = BEE / LOCATE
    outd.mkdir(parents=True, exist_ok=True)
    for vi in vids:
        v = P['videos'][vi]
        f = outd / f'{v["obs"]}.npz'
        if f.exists() and not args.force:
            log(f'[SKIP] {f}')
            continue
        t0 = time.time()
        r = E.loc[v['obs']]
        w, h, n = v['w'], v['h'], v['n']
        roi = roi_mask({**r.to_dict(), 'width': w, 'height': h})
        ys, xs = np.nonzero(roi)
        y0b, y1b, x0b, x1b = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        roi_b = roi[y0b:y1b, x0b:x1b]
        grey = np.empty((n, y1b - y0b, x1b - x0b), np.uint8)  # the ROI bounding box of every 5 fps frame
        k = 0
        for fr in decode_iter(v['path'], w, h, 0, None, threads=args.threads):
            if k >= n:
                break
            grey[k] = grey_np(fr[y0b:y1b, x0b:x1b])
            k += 1
        if k != n:
            raise SystemExit(f'{v["obs"]}: decoded {k} frames, expected {n}')
        t_dec = time.time() - t0
        bg_idx = np.unique(np.round(np.linspace(0, n - 1, N_BG)).astype(np.int64))
        smp = grey[bg_idx].astype(np.float32)
        gt = torch.from_numpy(smp.reshape(len(bg_idx), -1))
        bg1 = torch.cat([torch.quantile(gt[:, c:c + 60000], BG_Q1, dim=0)
                         for c in range(0, gt.shape[1], 60000)]).view(smp.shape[1:]).numpy()
        ex = np.zeros(smp.shape, bool)
        disk = np.hypot(*np.mgrid[-R_EX:R_EX + 1, -R_EX:R_EX + 1]) <= R_EX
        n_s1 = 0
        for i in range(len(bg_idx)):
            ok, _, _, _, cm = blob(((bg1 - smp[i]) > DARK_REL) & roi_b)
            if ok:
                ex[i] = ndimage.binary_dilation(cm, structure=disk)
                n_s1 += 1
        s2 = np.where(ex, np.nan, smp)
        n_ok = np.isfinite(s2).sum(0).astype(np.int16)
        g2 = torch.from_numpy(s2.reshape(len(bg_idx), -1))
        bg = torch.cat([torch.nanquantile(g2[:, c:c + 60000], BG_Q, dim=0) for c in range(0, g2.shape[1], 60000)])
        bg = np.nan_to_num(bg.view(smp.shape[1:]).numpy())
        todo = n_ok < MIN_FREE
        bg = fill_pixels(torch.from_numpy(bg), torch.from_numpy(todo)).numpy()
        bg = np.clip(np.round(bg), 0, 255).astype(np.uint8)
        bgi = bg.astype(np.int16)
        cx = np.full(n, np.nan, np.float32)
        cy = np.full(n, np.nan, np.float32)
        area = np.zeros(n, np.int32)
        dark_px = np.zeros(n, np.int32)
        found = np.zeros(n, bool)
        for i in range(n):
            dk = ((bgi - grey[i]) > DARK_REL) & roi_b
            dark_px[i] = dk.sum()
            ok, x, y, a, _ = blob(dk)
            found[i], area[i] = ok, a
            if ok:
                cx[i], cy[i] = x + x0b, y + y0b
        hx, hy, sx, sy = smooth_track(cx.astype(np.float64), cy.astype(np.float64), found)
        bx, by = boxes_of(sx, sy)
        # outside the ROI box: edge values of the box (never used by the mask, which is ROI-restricted)
        full_bg = np.pad(bg, ((y0b, h - y1b), (x0b, w - x1b)), mode='edge')
        tmp = outd / f'{v["obs"]}.tmp.npz'
        np.savez(tmp, obs=v['obs'], n=n, cx=cx, cy=cy, found=found, area=area, dark_px=dark_px, hx=hx, hy=hy, sx=sx,
                 sy=sy, x0=bx, y0=by, side=SIDE, pix_bg=full_bg, roi=roi, n_ok=n_ok, n_filled=int(todo.sum()),
                 bg_idx=bg_idx, roi_box=np.array([y0b, y1b, x0b, x1b]), stage1_blobs=n_s1,
                 params=json.dumps({'N_BG': N_BG, 'BG_Q1': BG_Q1, 'BG_Q': BG_Q, 'R_EX': R_EX, 'MIN_FREE': MIN_FREE,
                                    'DARK_REL': DARK_REL, 'MIN_BLOB': MIN_BLOB, 'SMOOTH': SMOOTH, 'SIDE': SIDE,
                                    'MAX_EXTENT': MAX_EXTENT, 'DEADBAND': DEADBAND}))
        os.replace(tmp, f)
        jump = np.hypot(np.diff(sx), np.diff(sy))
        log(f'{v["obs"]}: {n} frames, blob found {found.mean():.4f}, area median {np.median(area[found]):.0f}, '
            f'stage-1 blobs {n_s1}/{len(bg_idx)}, bg filled {int(todo.sum())} px, |raw - smoothed| p99 '
            f'{np.nanpercentile(np.hypot(cx - sx, cy - sy), 99):.1f} px, smoothed step p99 {np.percentile(jump, 99):.1f} '
            f'px; decode {t_dec:.0f}s, total {time.time() - t0:.0f}s')


# ---------------------------------------------------------------------------------------------- crops (shared)
def crop(frame, x0, y0):
    """(h, w, c) frame, box top-left (may be outside) -> (SIDE, SIDE, c), the frame edge-padded by SIDE / 2."""
    h, w = frame.shape[:2]
    pad = SIDE // 2
    xa, ya = x0 + pad, y0 + pad
    if 0 <= x0 and 0 <= y0 and x0 + SIDE <= w and y0 + SIDE <= h:
        return frame[y0:y0 + SIDE, x0:x0 + SIDE]
    fp = np.pad(frame, ((pad, pad), (pad, pad)) + ((0, 0),) * (frame.ndim - 2), mode='edge')
    xa, ya = int(np.clip(xa, 0, fp.shape[1] - SIDE)), int(np.clip(ya, 0, fp.shape[0] - SIDE))
    return fp[ya:ya + SIDE, xa:xa + SIDE]


def crop_plane(plane, x0, y0, fill):
    """Background / ROI plane crop with constant fill outside the frame (ROI False there)."""
    h, w = plane.shape
    pad = SIDE // 2
    fp = np.pad(plane, pad, mode='edge' if fill is None else 'constant',
                **({} if fill is None else {'constant_values': fill}))
    xa, ya = int(np.clip(x0 + pad, 0, fp.shape[1] - SIDE)), int(np.clip(y0 + pad, 0, fp.shape[0] - SIDE))
    return fp[ya:ya + SIDE, xa:xa + SIDE]


# ---------------------------------------------------------------------------------------------- validate
def cmd_validate(args):
    """Label-free crop checks + the SLEAP agreement (SLEAP read HERE ONLY, never by locate / encode)."""
    from PIL import Image, ImageDraw
    P = R.load_plan(DOM)
    per, nodes_in, dists, dist_raw, foundf = [], [], [], [], []
    rng = np.random.default_rng(0)
    pick_v = rng.choice(len(P['videos']), 24, replace=False)
    picks = []
    tot = {'frames': 0, 'found': 0, 'sleap_frames': 0, 'nodes': 0, 'nodes_in': 0, 'nodes_in_raw_box': 0,
           'nodes_in_120': 0}
    dist_all, raw_all = [], []
    for vi, v in enumerate(P['videos']):
        L = np.load(BEE / LOCATE / f'{v["obs"]}.npz')
        z = np.load(REPO / 'dataset/frogs/v1_tadpole/sleap' / f'{v["obs"]}.npz')
        n = int(L['n'])
        nd = z['nodes'][:n].astype(np.float64)
        c = z['centre'][:n].astype(np.float64)
        if len(z['nodes']) < n:
            raise SystemExit(f'{v["obs"]}: SLEAP {len(z["nodes"])} < {n} frames')
        sx, sy, found = L['sx'], L['sy'], L['found']
        ok = np.isfinite(c).all(1)
        d = np.hypot(sx - c[:, 0], sy - c[:, 1])[ok]
        dr = np.hypot(L['cx'] - c[:, 0], L['cy'] - c[:, 1])[ok & found]
        bx, by = boxes_of(sx, sy)
        x0, y0 = bx.astype(np.float64), by.astype(np.float64)
        fin = np.isfinite(nd[..., 0])
        inside = (nd[..., 0] >= x0[:, None]) & (nd[..., 0] < x0[:, None] + SIDE) & \
                 (nd[..., 1] >= y0[:, None]) & (nd[..., 1] < y0[:, None] + SIDE)
        h120 = 60
        in120 = (np.abs(nd[..., 0] - sx[:, None]) < h120) & (np.abs(nd[..., 1] - sy[:, None]) < h120)
        tot['frames'] += n
        tot['found'] += int(found.sum())
        tot['sleap_frames'] += int(ok.sum())
        tot['nodes'] += int(fin.sum())
        tot['nodes_in'] += int((inside & fin).sum())
        tot['nodes_in_120'] += int((in120 & fin).sum())
        dist_all.append(d)
        raw_all.append(dr)
        per.append({'obs': v['obs'], 'group': v['group'], 'found_frac': float(found.mean()),
                    'sleap_centre_frac': float(ok.mean()), 'dist_median': float(np.median(d)) if len(d) else None,
                    'dist_p90': float(np.percentile(d, 90)) if len(d) else None,
                    'frac_dist_le_10': float((d <= 10).mean()) if len(d) else None,
                    'nodes_in_box_frac': float((inside & fin).sum() / max(fin.sum(), 1)),
                    'frames_all_nodes_in_box_frac': float(((inside | ~fin).all(1) & fin.any(1)).sum() /
                                                          max(fin.any(1).sum(), 1))})
        if vi in pick_v:
            k = int(np.random.default_rng([3, vi]).integers(n))
            picks.append((v, k, x0[k], y0[k], found[k], nd[k], L))
    D, Dr = np.concatenate(dist_all), np.concatenate(raw_all)
    res = {'n_videos': len(per), 'side': SIDE, 'S': S, 'G': G,
           'blob_found_frac': tot['found'] / tot['frames'],
           'blob_found_frac_by_video': {'min': min(p['found_frac'] for p in per),
                                        'p10': float(np.percentile([p['found_frac'] for p in per], 10)),
                                        'median': float(np.median([p['found_frac'] for p in per]))},
           'sleap_centre_frames_frac': tot['sleap_frames'] / tot['frames'],
           'crop_centre_vs_sleap_centre_px': {'median': float(np.median(D)), 'p90': float(np.percentile(D, 90)),
                                              'p99': float(np.percentile(D, 99)), 'frac_le_5': float((D <= 5).mean()),
                                              'frac_le_10': float((D <= 10).mean()),
                                              'frac_le_20': float((D <= 20).mean()),
                                              'frac_gt_48': float((D > 48).mean())},
           'raw_blob_centroid_vs_sleap_centre_px': {'median': float(np.median(Dr)), 'p90': float(np.percentile(Dr, 90)),
                                                    'frac_le_10': float((Dr <= 10).mean())},
           'sleap_nodes_inside_crop_frac': tot['nodes_in'] / tot['nodes'],
           'sleap_nodes_inside_120px_box_frac': tot['nodes_in_120'] / tot['nodes'],
           'note': 'SLEAP (dataset/frogs/v1_tadpole/sleap, centre = mean of the 5 source frames around the 5 fps frame) '
                   'used only here, as a check; the crop is placed by the label-free blob track',
           'per_video': per}
    # 24 random crops (source px, upsampled 4x to 384) with the SLEAP nodes (red dots) and the box centre (cross)
    tiles = []
    for v, k, x0, y0, fd, ndk, L in picks:
        fr = next(decode_iter(v['path'], v['w'], v['h'], k, 1, threads=2))
        cr = crop(fr, int(x0), int(y0))
        im = Image.fromarray(np.ascontiguousarray(cr)).resize((S, S), Image.BICUBIC)
        dr_ = ImageDraw.Draw(im)
        sc = S / SIDE
        for x, y in ndk[np.isfinite(ndk[:, 0])]:
            px, py = (x - x0) * sc, (y - y0) * sc
            dr_.ellipse([px - 3, py - 3, px + 3, py + 3], outline=(255, 0, 0))
        dr_.line([S / 2 - 8, S / 2, S / 2 + 8, S / 2], fill=(0, 255, 0))
        dr_.line([S / 2, S / 2 - 8, S / 2, S / 2 + 8], fill=(0, 255, 0))
        dr_.text((4, 4), f'{v["obs"]} k{k} blob {"yes" if fd else "NO (held)"}', fill=(255, 255, 0))
        tiles.append(im)
    nc = 6
    sheet = Image.new('RGB', (nc * S, int(np.ceil(len(tiles) / nc)) * S))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % nc) * S, (i // nc) * S))
    sheet.save(OUT / 'crop_sheet.jpg', quality=85)
    (OUT / 'crop_validation.json').write_text(json.dumps(res, indent=1))
    log(json.dumps({k_: v_ for k_, v_ in res.items() if k_ != 'per_video'}))


# ---------------------------------------------------------------------------------------------- encoder + mask (GPU)
class Enc:
    """DINOv3-L/16 blocks 1-20 + final LayerNorm (rollout_r4.Pipe.m3, unchanged) on 384 px crops, and the crop mask."""

    def __init__(self, dev):
        import torch
        from transformers import AutoModel
        self.torch, self.dev = torch, dev
        m3 = AutoModel.from_pretrained(R.D3L, attn_implementation='sdpa').to(dev).eval()
        assert len(m3.layer) == 24
        m3.layer = torch.nn.ModuleList(list(m3.layer)[:R.LAYER])
        self.m3 = m3.requires_grad_(False)
        self.mean = torch.tensor(R.MEAN, device=dev)[None, :, None, None]
        self.std = torch.tensor(R.STD, device=dev)[None, :, None, None]
        self.gpu_s = 0.0
        self.n_fwd = 0

    def up(self, crops):
        """(B, SIDE, SIDE, 3) uint8 numpy -> (B, 3, S, S) float 0..255 (bicubic, clamped, rounded = a uint8 image)."""
        F = self.torch.nn.functional
        x = self.torch.from_numpy(np.ascontiguousarray(crops)).to(self.dev).permute(0, 3, 1, 2).float()
        return F.interpolate(x, size=(S, S), mode='bicubic', align_corners=False).round().clamp(0, 255)

    def tokens(self, x):
        torch = self.torch
        t0 = time.time()
        with torch.no_grad():
            xin = (x / 255 - self.mean) / self.std
            with torch.autocast('cuda', dtype=torch.bfloat16):
                h3 = self.m3(pixel_values=xin).last_hidden_state
            z = h3[:, R.N_PRE:].float()
            assert z.shape[1] == G * G and torch.isfinite(z).all(), z.shape
            z = z.half()
        torch.cuda.synchronize()
        self.gpu_s += time.time() - t0
        self.n_fwd += len(x)
        return z

    def mask(self, x, bgc, roic):
        """x (B, 3, S, S) float upsampled crop, bgc (B, SIDE, SIDE) uint8 background crops, roic (B, SIDE, SIDE) bool
        -> (B, G*G) bool: frogs pixel rule (dark_rel DARK_REL, ROI, no node restriction), patch dark fraction > 0.05
        core + 1-patch dilation onto patches with any dark pixel."""
        torch = self.torch
        F = torch.nn.functional
        with torch.no_grad():
            t = x.round().int()
            grey = ((t[:, 0] * 19595 + t[:, 1] * 38470 + t[:, 2] * 7471 + 32768) >> 16).float()
            b = torch.from_numpy(np.ascontiguousarray(bgc)).to(self.dev).float()[:, None]
            b = F.interpolate(b, size=(S, S), mode='bilinear', align_corners=False)[:, 0]
            r = torch.from_numpy(np.ascontiguousarray(roic)).to(self.dev).float()[:, None]
            r = F.interpolate(r, size=(S, S), mode='nearest')[:, 0] > 0.5
            px = ((b - grey) > DARK_REL) & r
            dk = F.avg_pool2d(px.float()[:, None], 16).flatten(1)
            core = dk > DARK_FRAC
            grown = F.max_pool2d(core.float().view(-1, 1, G, G), 3, stride=1, padding=1).flatten(1) > 0
            return core | (grown & (dk > 0)), px.flatten(1).sum(1)


def load_locate(obs):
    """Locate outputs of a video; the crop boxes from the smoothed centroid with the dead band (boxes_of)."""
    L = np.load(BEE / LOCATE / f'{obs}.npz')
    d = {k: L[k] for k in ('sx', 'sy', 'found', 'pix_bg', 'roi')}
    d['x0'], d['y0'] = boxes_of(d['sx'], d['sy'])
    return d


def lag_index(k):
    return max(k - R.LAG, 0)


# ---------------------------------------------------------------------------------------------- prepass
def mask_sheet(items, path):
    """items: [(x (3, S, S) uint8, mask (G*G,) bool, title)] -> one JPEG (red tint = kept tokens)."""
    from PIL import Image, ImageDraw
    tiles = []
    for x, m, t in items:
        im = x.transpose(1, 2, 0).astype(np.float32)
        up = np.kron(m.reshape(G, G), np.ones((16, 16))).astype(bool)
        im[up] = 0.6 * im[up] + 0.4 * np.array([255, 0, 0], np.float32)
        tile = Image.fromarray(im.clip(0, 255).astype(np.uint8))
        ImageDraw.Draw(tile).text((4, 4), t, fill=(255, 255, 0))
        tiles.append(tile)
    nc = 4
    sheet = Image.new('RGB', (nc * S, int(np.ceil(len(tiles) / nc)) * S))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % nc) * S, (i // nc) * S))
    sheet.save(path, quality=85)


def cmd_prepass(args):
    import torch
    P = R.load_plan(DOM)
    dev = torch.device('cuda')
    loc = R.local_dir() / 'prepass'
    loc.mkdir(parents=True, exist_ok=True)
    enc = Enc(dev)
    jobs = []
    for vi, v in enumerate(P['videos']):
        for a in v['bursts']:
            jobs.append((vi, a))
    if args.limit:
        jobs = jobs[::max(1, len(jobs) // args.limit)][:args.limit]
    Ls = {}

    def dec(job):
        vi, a = job
        v = P['videos'][vi]
        t0 = time.time()
        fr = [f for f in decode_iter(v['path'], v['w'], v['h'], a, 26, threads=2)][::5]  # k = a + 0, 5, ..., 25
        return job, np.stack(fr), time.time() - t0

    for v in P['videos']:
        Ls[v['obs']] = load_locate(v['obs'])
    ftok, fprev = open(loc / 'tok.f16', 'wb'), open(loc / 'prev.f16', 'wb')
    tok_frame, tok_pos, frames, sheet = [], [], [], []
    n_tok, dec_s, n_lagpass = 0, 0.0, 0
    t0 = time.time()
    par = max(4, int(os.environ.get('SLURM_CPUS_PER_TASK', 16)) - 2)
    ex = ThreadPoolExecutor(par)
    futs = [ex.submit(dec, j) for j in jobs[:2 * par]]
    nxt = len(futs)
    batch = []

    def flush(batch):
        nonlocal n_tok, n_lagpass
        own, lag, bgc, roic, meta = [], [], [], [], []
        for (vi, a), fr in batch:
            v = P['videos'][vi]
            L = Ls[v['obs']]
            for j in range(2, 6):
                k = a + 5 * j
                x0, y0 = int(L['x0'][k]), int(L['y0'][k])
                own.append(crop(fr[j], x0, y0))
                lag.append(crop(fr[j - 2], x0, y0))
                bgc.append(crop_plane(L['pix_bg'], x0, y0, None))
                roic.append(crop_plane(L['roi'], x0, y0, False))
                meta.append((vi, k))
        xo, xl = enc.up(np.stack(own)), enc.up(np.stack(lag))
        m, npx = enc.mask(xo, np.stack(bgc), np.stack(roic))
        z = enc.tokens(xo)
        zl = enc.tokens(xl)
        n_lagpass += len(xl)
        for i, (vi, k) in enumerate(meta):
            v = P['videos'][vi]
            pos = torch.nonzero(m[i]).flatten()
            ftok.write(z[i, pos].cpu().numpy().tobytes())
            fprev.write(zl[i, pos].cpu().numpy().tobytes())
            tok_frame.append(np.full(len(pos), len(frames), np.int32))
            tok_pos.append(pos.cpu().numpy().astype(np.int16))
            frames.append({'video': vi, 'obs': v['obs'], 'group': v['group'], 'k': k, 'src_row': v['src_lo'] + k,
                           'lay_row': v['lay_lo'] + k, 'role': v['role'], 'n_kept': int(len(pos)),
                           'n_stored': int(len(pos)), 'n32': int(len(pos)), 'dark_px384': int(npx[i]),
                           'blob_found': bool(Ls[v['obs']]['found'][k])})
            n_tok += len(pos)
            if v['role'] == 'eval' and len(sheet) < 16 and (not sheet or sheet[-1][2].split()[0] != v['obs']):
                sheet.append((xo[i].round().clamp(0, 255).byte().cpu().numpy(), m[i].cpu().numpy(),
                              f'{v["obs"]} k{k} kept {len(pos)}'))

    i = 0
    while i < len(jobs):
        job, fr, ds = futs[i].result()
        futs[i] = None
        dec_s += ds
        if nxt < len(jobs):
            futs.append(ex.submit(dec, jobs[nxt]))
            nxt += 1
        batch.append((job, fr))
        if len(batch) == 8:  # 32 own + 32 lag crops
            flush(batch)
            batch = []
        if i % 500 == 0:
            log(f'prepass {i}/{len(jobs)} bursts, {n_tok:,} tokens, {time.time() - t0:.0f}s (GPU {enc.gpu_s:.0f}s, '
                f'decode thread-s {dec_s:.0f})')
        i += 1
    if batch:
        flush(batch)
    ex.shutdown()
    ftok.close()
    fprev.close()
    F = pd.DataFrame(frames)
    F.to_parquet(loc / 'frames.parquet')
    np.save(loc / 'tok_frame.npy', np.concatenate(tok_frame))
    np.save(loc / 'tok_pos.npy', np.concatenate(tok_pos))
    info = {'domain': DOM, 'S': S, 'G': G, 'side': SIDE, 'n_bursts': len(jobs), 'n_frames': len(F), 'n_tokens': n_tok,
            'frames_by_role': F.role.value_counts().to_dict(), 'frames_by_group': F.group.value_counts().to_dict(),
            'frames_by_group_role': {f'{g}|{r}': int(n) for (g, r), n in F.groupby(['group', 'role']).size().items()},
            'tokens_by_role': F.groupby('role').n_stored.sum().to_dict(),
            'kept_per_frame': {'mean': float(F.n_kept.mean()), 'median': float(F.n_kept.median()),
                               'p10': float(F.n_kept.quantile(0.1)), 'p90': float(F.n_kept.quantile(0.9)),
                               'frames_0_kept': int((F.n_kept == 0).sum())},
            'kept_per_frame_blob_found': float(F.n_kept[F.blob_found].mean()),
            'kept_per_frame_blob_missing': float(F.n_kept[~F.blob_found].mean()) if (~F.blob_found).any() else None,
            'wall_s': time.time() - t0, 'gpu_s': enc.gpu_s, 'n_forward': enc.n_fwd,
            'fps_gpu_forward': enc.n_fwd / max(enc.gpu_s, 1e-9), 'decode_thread_s': dec_s,
            'gpu': torch.cuda.get_device_name(0), 'amp': 'bf16', 'code_commit': os.environ.get('CODE_COMMIT')}
    log(json.dumps(info))
    OUT.mkdir(parents=True, exist_ok=True)
    if args.limit:
        log('limit set: test only, nothing written to OUT / BEE')
        return
    if sheet:
        mask_sheet(sheet, OUT / 'mask_sheet.jpg')
    (OUT / 'prepass.json').write_text(json.dumps(info, indent=1, default=float))
    R.atomic_copy(loc, BEE / 'prepass')
    log(f'-> {BEE / "prepass"}')


# ---------------------------------------------------------------------------------------------- encode
def cmd_encode(args):
    import torch
    P = R.load_plan(DOM)
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False
    outdir = BEE / 'videos'
    outdir.mkdir(parents=True, exist_ok=True)
    vids = P['shards'][args.shard] if args.videos is None else [int(x) for x in args.videos.split(',')]
    todo = [vi for vi in vids if not (outdir / f'{P["videos"][vi]["obs"]}.npz').exists()]
    log(f'shard {args.shard}: {len(vids)} videos, {len(todo)} to do -> {outdir}')
    if not todo:
        return
    enc = Enc(dev)
    saes, best = R.load_saes(DOM, dev)
    CH = args.chunk
    jobs = []
    for vi in todo:
        v = P['videos'][vi]
        n = v['n'] if not args.max_frames else min(v['n'], args.max_frames)
        for k0 in range(0, n, CH):
            jobs.append((vi, k0, min(CH, n - k0), n))
    Ls, lk = {}, threading.Lock()

    def L_of(obs):
        with lk:
            if obs not in Ls:
                Ls[obs] = load_locate(obs)
            return Ls[obs]

    def dec(job):
        """Decode frames [max(0, k0 - LAG), k0 + m) and cut the own crops of k0..k0+m-1 and the lag crops (frame
        max(k - LAG, 0) with the box of k); the lag crop is skipped (None) when its box equals the lag frame's own box
        and the lag frame is inside this chunk (its own tokens are reused)."""
        vi, k0, m, n = job
        v = P['videos'][vi]
        L = L_of(v['obs'])
        t0 = time.time()
        a = max(0, k0 - R.LAG)
        fr = list(decode_iter(v['path'], v['w'], v['h'], a, k0 + m - a, threads=args.threads))
        own, bgc, roic, lag, lag_k = [], [], [], [], []
        for k in range(k0, k0 + m):
            x0, y0 = int(L['x0'][k]), int(L['y0'][k])
            own.append(crop(fr[k - a], x0, y0))
            bgc.append(crop_plane(L['pix_bg'], x0, y0, None))
            roic.append(crop_plane(L['roi'], x0, y0, False))
            j = lag_index(k)
            same = int(L['x0'][j]) == x0 and int(L['y0'][j]) == y0
            if not (same and j >= k0):
                lag.append(crop(fr[j - a], x0, y0))
                lag_k.append(k)
        return job, np.stack(own), np.stack(bgc), np.stack(roic), (np.stack(lag) if lag else None), lag_k, \
            time.time() - t0

    par = args.par
    ex = ThreadPoolExecutor(par)
    futs = {i: ex.submit(dec, jobs[i]) for i in range(min(len(jobs), par + 1))}
    nxt = len(futs)
    T0 = time.time()
    dec_s, tot, n_lag = 0.0, 0, 0
    cur, acc, tv = None, None, None
    BS = args.bs

    def finish(vi, acc, tv):
        v = P['videos'][vi]
        arr = {k_: np.concatenate(x) for k_, x in acc.items() if k_ not in ('l0', 'ntok', 'n_lag')}
        nn = len(arr['n_fg'])
        f = R.local_dir() / f'{v["obs"]}.npz'
        np.savez(f, **arr, l0=np.array([acc['l0'][v_] for v_ in R.VARIANTS]),
                 ntok=np.array([acc['ntok'][v_] for v_ in R.VARIANTS]), obs=v['obs'], n=nn, n_lag=acc['n_lag'][0],
                 code_commit=os.environ.get('CODE_COMMIT', ''))
        R.atomic_copy(f, outdir / f.name)
        f.unlink()
        dt = time.time() - tv
        log(f'{v["obs"]} ({v["role"]}): {nn} frames in {dt:.0f}s ({nn / dt:.1f} fps), kept/frame {arr["n_fg"].mean():.1f}, '
            f'0-kept frames {(arr["n_fg"] == 0).sum()}, lag passes {acc["n_lag"][0]} '
            f'({acc["n_lag"][0] / nn:.3f} of the frames)')

    for i in range(len(jobs)):
        job, own, bgc, roic, lag, lag_k, ds = futs.pop(i).result()
        if nxt < len(jobs):
            futs[nxt] = ex.submit(dec, jobs[nxt])
            nxt += 1
        dec_s += ds
        vi, k0, m, n = job
        if vi != cur:
            if cur is not None:
                finish(cur, acc, tv)
            cur, tv = vi, time.time()
            acc = {'n_fg': [], 'n_fg32': [], 'dark_px': [], 'l0': {v_: 0.0 for v_ in R.VARIANTS},
                   'ntok': {v_: 0 for v_ in R.VARIANTS}, 'n_lag': [0],
                   **{f'{p}_{v_}': [] for p in ('max', 'mean', 'arg') for v_ in R.VARIANTS}}
        # own tokens of the chunk, mask
        Z = torch.empty((m, G * G, R.D), dtype=torch.float16, device=dev)
        M = torch.empty((m, G * G), dtype=torch.bool, device=dev)
        NPX = torch.empty(m, dtype=torch.int64, device=dev)
        for b in range(0, m, BS):
            x = enc.up(own[b:b + BS])
            M[b:b + BS], NPX[b:b + BS] = enc.mask(x, bgc[b:b + BS], roic[b:b + BS])
            Z[b:b + BS] = enc.tokens(x)
        # lag tokens: reuse own tokens of frame max(k - LAG, 0) when the boxes agree, else the lag-crop pass
        Pz = torch.empty_like(Z)
        need = set(lag_k)
        for kk in range(k0, k0 + m):
            if kk not in need:
                Pz[kk - k0] = Z[lag_index(kk) - k0]
        if lag is not None:
            for b in range(0, len(lag), BS):
                zl = enc.tokens(enc.up(lag[b:b + BS]))
                idx = torch.tensor([kk - k0 for kk in lag_k[b:b + BS]], device=dev)
                Pz[idx] = zl
            n_lag += len(lag)
            acc['n_lag'][0] += len(lag)
        with torch.no_grad():
            for b in range(0, m, BS):
                out = R.sae_codes(torch, saes, Z[b:b + BS], Pz[b:b + BS], M[b:b + BS])
                for v_, (cm, cmean, am, l0, nt) in out.items():
                    acc[f'max_{v_}'].append(cm.cpu().numpy())
                    acc[f'mean_{v_}'].append(cmean.cpu().numpy())
                    acc[f'arg_{v_}'].append(am.cpu().numpy())
                    acc['l0'][v_] += l0
                    acc['ntok'][v_] += nt
        nf = M.sum(1).to(torch.int16).cpu().numpy()
        acc['n_fg'].append(nf)
        acc['n_fg32'].append(nf)  # no 32-grid here: n_fg32 = n_fg (keeps the r4 merge format)
        acc['dark_px'].append(NPX.to(torch.int32).cpu().numpy())
        tot += m
        del Z, Pz, M
    if cur is not None:
        finish(cur, acc, tv)
    ex.shutdown()
    wall = time.time() - T0
    log(f'shard {args.shard} DONE: {tot:,} frames in {wall:.0f}s ({tot / wall:.1f} fps wall), forward passes '
        f'{enc.n_fwd:,} (lag {n_lag:,} = {n_lag / max(tot, 1):.3f} per frame), GPU busy {enc.gpu_s:.0f}s '
        f'({enc.n_fwd / max(enc.gpu_s, 1e-9):.1f} crops/s), decode thread-s {dec_s:.0f}, {torch.cuda.get_device_name(0)}')


# ---------------------------------------------------------------------------------------------- check
def cmd_check(args):
    """seek decode == sequential decode (2 videos), DINOv3-L crops/s at S = 384 per batch size, decode frames/s."""
    import torch
    P = R.load_plan(DOM)
    res = {'seek': []}
    for vi in (0, len(P['videos']) // 2):
        v = P['videos'][vi]
        t0 = time.time()
        seq = np.stack(list(decode_iter(v['path'], v['w'], v['h'], 0, 40)))
        t_seq = time.time() - t0
        for s0 in (1, 23, 30):
            sk = np.stack(list(decode_iter(v['path'], v['w'], v['h'], s0, 10)))
            res['seek'].append({'obs': v['obs'], 's0': s0, 'equal': bool((seq[s0:s0 + 10] == sk).all()),
                                'max_abs_diff': int(np.abs(seq[s0:s0 + 10].astype(int) - sk.astype(int)).max())})
        res.setdefault('decode_40_frames_s', []).append(t_seq)
    log(json.dumps(res))
    dev = torch.device('cuda')
    enc = Enc(dev)
    x = torch.rand((256, 3, S, S), device=dev) * 255
    thr = {}
    for bs in (32, 64, 128):
        enc.tokens(x[:bs])
        enc.gpu_s, enc.n_fwd = 0.0, 0
        for _ in range(6):
            enc.tokens(x[:bs])
        thr[bs] = enc.n_fwd / enc.gpu_s
    res['crops_per_s'] = thr
    res['gpu'] = torch.cuda.get_device_name(0)
    log(json.dumps(res))
    (OUT / 'check.json').write_text(json.dumps(res, indent=1))
    bad = [r for r in res['seek'] if not r['equal']]
    if bad:
        raise SystemExit(f'SEEK != SEQUENTIAL: {bad}')


# ---------------------------------------------------------------------------------------------- register in rollout_r4
R.DOMS[DOM] = dict(ann=str(LAYOUT_LINK / 'annotations.csv'), exp=str(LAYOUT_LINK / 'experiment.csv'), fps=FPS,
                   step=STEP, offset=OFFSET, S=S, rule='tadpoles_r4_tracker_free', bg=str(BEE / LOCATE),
                   eci=str(LAYOUT_LINK), layout=None, domain='tadpoles', ants_eci=None, native=800, body=38,
                   ref_nfg=None, c1=None, deployed=None, p_store=1.0, chunk=600, par=6, threads=4, bs=32, n_shards=0)


def cmd_manifest(args):
    best = json.loads((OUT / 'sae' / 'best.json').read_text())
    nes_root = REPO / f'results/vision/frogs/eci_tadpole{TAG}/nes'
    M = {'name': 'r4 frogs tadpole (stage 44-48)', 'domain_key': DOM, 'nes_domain': 'tadpoles', 'TADPOLE_TAG': TAG,
         'layout': str(LAYOUT_LINK), 'layout_target': str(BEE / 'layout'),
         'description': 'tracker-free body crop (label-free dark-blob track, 96 source px, upsampled to 384 = 24 x 24 '
                        'patches), frogs pixel mask on the crop (no SLEAP), DINOv3-L/16 block 20, own 512-latent '
                        'Matryoshka BatchTopK SAE per variant (d2 = [z, z - z_{t-2 s} at the same crop window], d0 = z)',
         'contrasts': ['FoxP1_vs_WT', 'En1_vs_WT'], 'primary_rule': 'fnorm',
         'sae': {v_: {'name': best[v_]['name'], 'dir': str(OUT / 'sae' / best[v_]['name']),
                      'theta_star': best[v_]['theta_star']} for v_ in R.VARIANTS},
         'prepass': str(BEE / 'prepass'), 'locate': str(BEE / LOCATE), 'variants': {}, 'commits': {}}
    for v_ in R.VARIANTS:
        nm = best[v_]['name']
        M['variants'][v_] = {
            'codes': {r: str(BEE / 'codes' / (nm + suf)) for r, suf in (('gt0', ''), ('fnorm', '_fnorm'), ('ftheta', '_ftheta'))},
            'codes_exist': {r: (BEE / 'codes' / (nm + suf) / 'DONE').exists()
                            for r, suf in (('gt0', ''), ('fnorm', '_fnorm'), ('ftheta', '_ftheta'))},
            'nes': {r: str(nes_root / (nm + suf)) for r, suf in (('gt0', ''), ('fnorm', '_fnorm'), ('ftheta', '_ftheta'))},
            'ness': {r: str(R.OUT / 'ness' / DOM / (nm + suf)) for r, suf in (('gt0', ''), ('fnorm', '_fnorm'), ('ftheta', '_ftheta'))},
            'summaries': str(OUT / f'summaries_{v_}.npz')}
    for f in sorted((OUT / 'code').glob('*.json')) if (OUT / 'code').exists() else []:
        M['commits'][f.stem] = json.loads(f.read_text())
    M['files'] = {k_: str(OUT / f'{k_}.json') for k_ in ('plan', 'design', 'crop_validation', 'prepass', 'train', 'merge',
                                                          'ftheta', 'summaries') if (OUT / f'{k_}.json').exists()}
    M['crop_sheet'] = str(OUT / 'crop_sheet.jpg')
    M['mask_sheet'] = str(OUT / 'mask_sheet.jpg')
    (OUT / 'manifest.json').write_text(json.dumps(M, indent=1))
    log(json.dumps(M)[:2000])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=('plan', 'check', 'locate', 'validate', 'prepass', 'train', 'encode', 'merge', 'ftheta',
                                    'summaries', 'manifest'))
    p.add_argument('--task', type=int, default=None)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--videos', default=None)
    p.add_argument('--per-shard', type=int, default=4)
    p.add_argument('--max-frames', type=int, default=0)
    p.add_argument('--chunk', type=int, default=600)
    p.add_argument('--par', type=int, default=6)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--bs', type=int, default=64)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--steps', type=int, default=6000)
    p.add_argument('--cap', type=int, default=0)
    p.add_argument('--n-init', type=int, default=500_000)
    p.add_argument('--force', action='store_true')
    args = p.parse_args()
    args.domain = DOM
    {'plan': cmd_plan, 'check': cmd_check, 'locate': cmd_locate, 'validate': cmd_validate, 'prepass': cmd_prepass, 'train': R.cmd_train,
     'encode': cmd_encode, 'merge': R.cmd_merge, 'ftheta': R.cmd_ftheta, 'summaries': R.cmd_summaries,
     'manifest': cmd_manifest}[args.step](args)


if __name__ == '__main__':
    main()
