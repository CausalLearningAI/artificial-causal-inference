"""
MABe22 ECI round 3, E5 (shared mouse + beetle dictionary): the BEETLE token store (label-free; eci_round3.sh E5).

Data: dataset/benchmark/mabe22/videos/beetle_userTrain/<clip>.avi (1,947 clips, 224 x 224, 30 fps, 900 frames). Beetle
keypoints (raw/beetle_user_train.npy, 'sequences'[clip]['keypoints'], (900, 4)) are used ONLY as a check (convention,
mask recall). Behaviour labels are never read.

Scene (contact sheets, 'look'): a dark circular dish seen from above, its bright rim close to the frame edge, black
corners outside it; pale animals on the dark floor. So the mouse 'dark pixel' cue is replaced by a BRIGHT pixel cue:
    pix_bg   per-pixel 2nd LOWEST of 30 evenly spaced grey frames (the mirror of the mouse 2nd highest = 0.98 quantile)
    bright   fraction of the 7 x 7 pixel block of each 448 patch with grey > b_abs and grey - pix_bg > b_rel; in the
             variants with 'hole', the relative condition is waived inside holes of the dish floor (pix_bg < floor_max
             component containing the frame centre, holes filled): an animal that never moves is bright in pix_bg too
             (the analogue of the mouse 'arena' trick)
    bg tok, thr, mask   exactly the mouse rule (src/eci/foreground.py FG_RULE: patch_background excluding bright-core
             frames, video_threshold 'far_q' 0.995 away from bright-core patches, foreground_parts with the bright
             fraction in place of the dark one: dilate_1(drop_isolated(dist > thr) OR bright > 0.15))

Modes (all Slurm, via scripts/benchmark/mabe22/eci.sh):
    look     CPU   contact sheets of a few clips with the keypoints of 4 candidate conventions drawn, and the measured
                   keypoint convention / frame offset (brightness above pix_bg under the keypoints)
    mask     GPU   mask variants on a sample of clips: keypoint recall (fraction of keypoints inside a kept patch, per
                   animal), kept patches per frame, mask sheets (r3/beetle/mask_sheet_<variant>.jpg)
    plan     CPU   label-free frame plan: 5 fps grid i = 60, 66, ..., 894, K train + C calib frames per clip at random
                   (numpy rng 0, disjoint) -> r3/beetle/btrain_frames.parquet
    encode   GPU   per shard: mask (chosen variant) + ORBIT 448 tokens of t and t-60 (clipped to 0) at t's kept positions
                   (eci_encode.Enc.orbit, fp32, TF32 off) -> tok_r3/btrain_<s>of<n>_{m448o_t,m448o_p60,pos448}.npy,
                   _frames.parquet, .json
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts/eci'))
from eci_common import OUT, RAW, VID, decode_clip, log, stage_dir  # noqa: E402

VSET = 'beetle_userTrain'
BOUT = OUT / 'r3/beetle'
TOKR3 = OUT / 'tok_r3'
T_CLIP = 900
LAG = 60  # 2 s at 30 fps (recipe R)
GRID_B = np.arange(60, 900, 6)  # 5 fps label grid 60, 66, ..., 894 (140 per clip)
MOUSE_TRAIN, MOUSE_CALIB = 784_815, 181_143  # mouse round-3 pool sizes (tokens)
# mask variants (bright cue for pale animals on a dark floor); 'tok' = token cue only (reference)
VARIANTS = {
    'tok': {'bright': False},
    'b15': {'bright': True, 'b_abs': 25, 'b_rel': 15, 'hole': False},
    'b15h': {'bright': True, 'b_abs': 25, 'b_rel': 15, 'hole': True, 'floor_max': 20},
    'b10h': {'bright': True, 'b_abs': 20, 'b_rel': 10, 'hole': True, 'floor_max': 20},
    # mask sheet 1: in ~half the clips the floor component leaks round a gap of the rim into the black corners, so the
    # whole bright rim became a 'hole'. Holes are capped at an animal-sized area (hole_max px at 224; a beetle is
    # ~25 x 12 px): larger holes (the rim ring) are not waived.
    'b15hc': {'bright': True, 'b_abs': 25, 'b_rel': 15, 'hole': True, 'floor_max': 20, 'hole_max': 1500},
    'b10hc': {'bright': True, 'b_abs': 20, 'b_rel': 10, 'hole': True, 'floor_max': 20, 'hole_max': 1500},
}


def clips_all():
    c = sorted(p.stem for p in (VID / VSET).glob('*.avi'))
    assert len(c) == 1947, len(c)
    return c


def keypoints(clips):
    """-> {clip: (900, 4) float32}; only the 'keypoints' field is read (never labels)."""
    d = np.load(RAW / 'beetle_user_train.npy', allow_pickle=True).item()
    s = d['sequences']
    return {c: np.asarray(s[c]['keypoints'], np.float32) for c in clips}


def grey_of(fr):
    return (0.299 * fr[..., 0] + 0.587 * fr[..., 1] + 0.114 * fr[..., 2]).round().clip(0, 255).astype(np.uint8)


# 4 candidate keypoint conventions -> animal a (0, 1): (x, y) in [0, 1] frame units
# (the look sheets showed animal 0 under (y, x) = (k0, k1) and animal 1 under (x, y) = (k2, k3): 'a0yx_a1xy')
CONV = {'xy_pairs': ((0, 1), (2, 3)), 'yx_pairs': ((1, 0), (3, 2)), 'x12y12': ((0, 2), (1, 3)), 'y12x12': ((2, 0), (3, 1)),
        'a0yx_a1xy': ((1, 0), (2, 3)), 'a0xy_a1yx': ((0, 1), (3, 2))}


def kp_xy(kp, conv):
    (ax, ay), (bx, by) = CONV[conv]
    return np.stack([kp[:, [ax, ay]], kp[:, [bx, by]]], 1)  # (T, 2 animals, 2 = x, y)


def kp_contrast(grey, pbg, xy, off=0):
    """mean over frames / animals of max(grey - pix_bg) in the 5 x 5 px window at the keypoint (frame t + off)."""
    T = len(grey)
    t = np.arange(max(0, -off), min(T, T - off))
    px = np.clip((xy[t] * 224).round().astype(int), 2, 221)
    d = grey.astype(np.int16) - pbg.astype(np.int16)[None]
    v = []
    for a in range(2):
        w = np.stack([d[t + off, px[:, a, 1] + dy, px[:, a, 0] + dx] for dy in range(-2, 3) for dx in range(-2, 3)], 1)
        v.append(w.max(1).mean())
    return [float(x) for x in v]


def pixbg_low(grey):
    bgi = np.linspace(0, len(grey) - 1, 30).round().astype(int)
    return np.sort(grey[bgi], 0)[1], bgi  # 2nd lowest of 30


def look(args):
    from PIL import Image, ImageDraw
    stage = stage_dir()
    clips = clips_all()
    rng = np.random.default_rng(0)
    pick = list(rng.choice(clips, args.n, replace=False))
    kps = keypoints(pick)
    d0 = np.load(RAW / 'beetle_user_train.npy', allow_pickle=True).item()
    res = {'top_keys': list(d0.keys()), 'seq_keys': list(d0['sequences'][pick[0]].keys()), 'clips': pick, 'per_clip': {}}
    del d0
    rows = []
    for k, c in enumerate(pick):
        fr = decode_clip(VSET, c, stage, 'rgb24')
        assert len(fr) == T_CLIP, (c, len(fr))
        g = grey_of(fr)
        pbg, _ = pixbg_low(g)
        phi = np.sort(g[np.linspace(0, T_CLIP - 1, 30).round().astype(int)], 0)[28]
        kp = kps[c]
        pc = {'kp_min': kp.min(0).round(3).tolist(), 'kp_max': kp.max(0).round(3).tolist(),
              'pixbg_low_q': np.quantile(pbg, [.05, .25, .5, .75, .95]).round(1).tolist(),
              'centre_disc_pixbg_low_q': np.quantile(pbg[56:168, 56:168], [.05, .5, .95]).round(1).tolist(),
              'frame_grey_q': np.quantile(g[::90], [.5, .9, .99, .999]).round(1).tolist(),
              'hi_minus_lo_q': np.quantile(phi.astype(int) - pbg, [.5, .9, .99]).round(1).tolist(),
              'contrast': {cv: {o: kp_contrast(g, pbg, kp_xy(kp, cv), o) for o in (-2, -1, 0, 1, 2)} for cv in CONV}}
        res['per_clip'][c] = pc
        log(f'{c}: ' + json.dumps({cv: np.round(v[0], 1).tolist() for cv, v in pc['contrast'].items()}) + f" pbg {pc['pixbg_low_q']}")
        if k < args.sheet_clips:
            tiles = []
            for t in np.linspace(0, T_CLIP - 1, 6).round().astype(int):
                im = Image.fromarray(fr[t]).resize((448, 448), Image.BICUBIC)
                dr = ImageDraw.Draw(im)
                for cv, col in ((args.conv, ((255, 0, 0), (0, 255, 0))),):
                    xy = kp_xy(kp, cv)[t] * 448
                    for a in range(2):
                        x, y = xy[a]
                        dr.ellipse([x - 6, y - 6, x + 6, y + 6], outline=col[a], width=2)
                dr.text((4, 4), f'{c[:8]} t{t}  {args.conv}: red a0, green a1', fill=(255, 255, 0))
                tiles.append(im)
            rows.append(tiles)
            # bright-vs-background view of the same clip (grey - pix_bg, x4)
            tiles = []
            for t in np.linspace(0, T_CLIP - 1, 6).round().astype(int):
                d = ((g[t].astype(int) - pbg) * 4).clip(0, 255).astype(np.uint8)
                im = Image.fromarray(d).convert('RGB').resize((448, 448), Image.NEAREST)
                ImageDraw.Draw(im).text((4, 4), f'{c[:8]} t{t} 4 x (grey - pix_bg low)', fill=(255, 255, 0))
                tiles.append(im)
            rows.append(tiles)
        del fr, g
    for s in range(0, len(rows), 4):
        R = rows[s:s + 4]
        W = Image.new('RGB', (6 * 448, len(R) * 448))
        for r, tiles in enumerate(R):
            for j, t in enumerate(tiles):
                W.paste(t, (j * 448, r * 448))
        BOUT.mkdir(parents=True, exist_ok=True)
        W.resize((W.width // 2 * 1, W.height // 2 * 1)).save(BOUT / f'look_{s // 4}.jpg', quality=88)
    agg = {cv: {o: np.mean([p['contrast'][cv][o] for p in res['per_clip'].values()], 0).round(2).tolist()
                for o in (-2, -1, 0, 1, 2)} for cv in CONV}
    res['mean_contrast'] = agg
    (BOUT / 'look.json').write_text(json.dumps(res, indent=1))
    log('mean contrast (grey - pix_bg under keypoints, by convention x frame offset): ' + json.dumps(agg))


def floor_holes(pbg, floor_max, hole_max=0):
    """pixels inside the dark dish floor that are NOT floor (bright in pix_bg too, e.g. animals that never move): holes of
    the connected dark component containing the frame centre."""
    from scipy import ndimage as ndi
    fl = pbg < floor_max
    fl = ndi.binary_opening(fl, np.ones((3, 3), bool))
    lab, n = ndi.label(fl)
    if n == 0:
        return np.zeros_like(fl), 0
    sizes = ndi.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    # the dish floor = the largest dark component whose bounding box covers the frame centre
    cand = [k + 1 for k in np.argsort(-sizes) if (lab == k + 1)[100:124, 100:124].any() and sizes[k] > 5000]
    if not cand:
        return np.zeros_like(fl), 0
    floor = lab == cand[0]
    floor_c = ndi.binary_closing(floor, np.ones((3, 3), bool))
    holes = ndi.binary_fill_holes(floor_c) & ~floor
    if hole_max:
        hl, nh = ndi.label(holes)
        if nh:
            hs = ndi.sum(np.ones_like(hl), hl, index=np.arange(1, nh + 1))
            holes = np.isin(hl, 1 + np.flatnonzero(hs <= hole_max))
    return holes, int(floor.sum())


def beetle_mask(enc, frames, ts, var, dev):
    """frames (900, 224, 224, 3) uint8 -> (len(ts), 1024) bool mask on the 448 grid + info + grey of ts."""
    import torch
    import torch.nn.functional as F
    from src.eci.foreground import FG_RULE, cosine_distance, foreground_parts, patch_background, video_threshold
    V = VARIANTS[var]
    grey = grey_of(frames)
    pbg_np, bgi = pixbg_low(grey)
    g = torch.from_numpy(grey).to(dev)
    pbg = torch.from_numpy(pbg_np).to(dev).float()
    hole_np, floor_px = (floor_holes(pbg_np, V['floor_max'], V.get('hole_max', 0)) if V.get('hole') else (np.zeros((224, 224), bool), 0))
    hole = torch.from_numpy(hole_np).to(dev)

    def bright_of(idx):
        gg = g[torch.as_tensor(idx, device=dev)].float()
        if not V['bright']:
            return torch.zeros(len(idx), 1024, device=dev)
        b = (gg > V['b_abs']) & (((gg - pbg[None]) > V['b_rel']) | hole[None])
        return F.avg_pool2d(b.float()[:, None], 7).flatten(1)

    need = np.unique(np.concatenate([bgi, ts]))
    pos_of = {v: j for j, v in enumerate(need)}
    tok = enc(frames[need], 448)
    jb = torch.tensor([pos_of[v] for v in bgi], device=dev)
    br_bg = bright_of(bgi)
    bg, n_ok = patch_background(tok[jb], exclude=br_bg > FG_RULE['dark_frac'], min_frames=8)
    thr = video_threshold(cosine_distance(tok[jb], bg).cpu().numpy(), br_bg.cpu().numpy(), FG_RULE)
    jt = torch.tensor([pos_of[v] for v in ts], device=dev)
    feat, bk, mask = foreground_parts(cosine_distance(tok[jt], bg), bright_of(ts), thr, FG_RULE)
    info = {'thr': thr, 'hole_px': int(hole_np.sum()), 'floor_px': floor_px, 'fallback_positions': int((n_ok < 8).sum()),
            'feat_per_frame': float(feat.sum(1).float().mean()), 'bright_per_frame': float(bk.sum(1).float().mean()),
            'kept_per_frame': float(mask.sum(1).float().mean())}
    return mask, info, grey[ts]


def kp_recall(mask, kp_t, conv):
    """mask (n, 1024) bool numpy, kp_t (n, 4) -> (n, 2) bool: keypoint of animal a inside a kept 448 patch."""
    xy = kp_xy(kp_t, conv)  # (n, 2, 2)
    pr = np.clip((xy[..., 1] * 224 // 7).astype(int), 0, 31)
    pc = np.clip((xy[..., 0] * 224 // 7).astype(int), 0, 31)
    return mask[np.arange(len(mask))[:, None], pr * 32 + pc]


def sheet_kp(rows, kp_rows, conv, path):
    """eci_encode.sheet (kept patches tinted red) + the frame's keypoints (a0 green, a1 cyan) drawn on each tile."""
    from PIL import Image, ImageDraw
    from eci_encode import sheet
    sheet(rows, path)
    W = Image.open(path).convert('RGB')
    dr = ImageDraw.Draw(W)
    for k, kp in enumerate(kp_rows[:16]):
        xy = kp_xy(kp[None], conv)[0] * 448
        for a, col in ((0, (0, 255, 0)), (1, (0, 255, 255))):
            x, y = xy[a] + np.array([(k % 4) * 448, (k // 4) * 448])
            dr.ellipse([x - 5, y - 5, x + 5, y + 5], outline=col, width=2)
    W.save(path, quality=90)


def kp_far(mask, kp_t, conv, r_px=40):
    """mask (n, 1024) bool, kp_t (n, 4) -> (kept patches, kept patches whose centre is > r_px (224 px) from both
    keypoints) per frame: a precision proxy (an ant with legs is ~40 px long at 224)."""
    xy = kp_xy(kp_t, conv) * 224  # (n, 2, 2)
    c = (np.arange(32) * 7 + 3.5)
    cy, cx = np.meshgrid(c, c, indexing='ij')
    cen = np.stack([cx.ravel(), cy.ravel()], 1)  # (1024, 2) x, y
    d = np.sqrt(((cen[None, :, None] - xy[:, None]) ** 2).sum(-1)).min(-1)  # (n, 1024)
    return mask.sum(1), (mask & (d > r_px)).sum(1)


def mask_mode(args):
    import torch
    from eci_encode import Enc
    dev = torch.device('cuda')
    stage = stage_dir()
    clips = clips_all()
    rng = np.random.default_rng(1)
    pick = list(rng.choice(clips, args.n, replace=False))
    kps = keypoints(pick)
    conv = args.conv
    enc = Enc(dev)
    variants = args.variants or list(VARIANTS)
    S = {v: {'rec': [], 'kept': [], 'far': [], 'info': {}, 'rows': [], 'kp': []} for v in variants}
    for ci, c in enumerate(pick):
        fr = decode_clip(VSET, c, stage, 'rgb24')
        assert len(fr) == T_CLIP
        ts = np.sort(np.random.default_rng([2, ci]).choice(GRID_B, 16, replace=False))
        for v in variants:
            m, info, gt = beetle_mask(enc, fr, ts, v, dev)
            m = m.cpu().numpy()
            S[v]['rec'].append(kp_recall(m, kps[c][ts], conv))
            S[v]['kept'].append(m.sum(1))
            S[v]['far'].append(kp_far(m, kps[c][ts], conv)[1])
            S[v]['info'][c] = info
            if ci < 16:
                S[v]['rows'].append((gt[0], m[0], f'{c[:8]} i{ts[0]} n{int(m[0].sum())}'))
                S[v]['kp'].append(kps[c][ts[0]])
        del fr
        if ci % 10 == 0:
            log(f'clip {ci}/{len(pick)} ' + ' '.join(f"{v}:{S[v]['info'][c]['kept_per_frame']:.0f}" for v in variants))
    res = {'clips': pick, 'conv': conv, 'gpu': torch.cuda.get_device_name(0), 'gpu_s': round(enc.gpu_s, 1), 'variants': {}}
    for v in variants:
        rec = np.concatenate(S[v]['rec'])  # (frames, 2)
        kept = np.concatenate(S[v]['kept'])
        r = {'rule': VARIANTS[v], 'frames': int(len(kept)), 'recall_animal0': float(rec[:, 0].mean()),
             'recall_animal1': float(rec[:, 1].mean()), 'recall_both': float(rec.all(1).mean()),
             'kept_mean': float(kept.mean()), 'kept_frac_grid': float(kept.mean() / 1024),
             'kept_q': np.quantile(kept, [0, .1, .5, .9, 1]).tolist(), 'frames_0_kept': int((kept == 0).sum()),
             'far40_frac_of_kept': float(np.concatenate(S[v]['far']).sum() / max(kept.sum(), 1)),
             'far40_per_frame': float(np.concatenate(S[v]['far']).mean()),
             'clip_recall_min': float(min(x.mean() for x in S[v]['rec'])),
             'per_clip_recall': {c: float(S[v]['rec'][k].mean()) for k, c in enumerate(pick)},
             'thr_q': np.quantile([i['thr'] for i in S[v]['info'].values()], [0, .5, 1]).round(4).tolist(),
             'hole_px_mean': float(np.mean([i['hole_px'] for i in S[v]['info'].values()])),
             'feat_per_frame': float(np.mean([i['feat_per_frame'] for i in S[v]['info'].values()])),
             'bright_per_frame': float(np.mean([i['bright_per_frame'] for i in S[v]['info'].values()]))}
        res['variants'][v] = r
        sheet_kp(S[v]['rows'], S[v]['kp'], conv, BOUT / f'mask_sheet_{v}{args.tag}.jpg')
        log(f"{v}: recall a0 {r['recall_animal0']:.3f} a1 {r['recall_animal1']:.3f} both {r['recall_both']:.3f} "
            f"kept/frame {r['kept_mean']:.1f} ({r['kept_frac_grid']:.3f} of grid) q {r['kept_q']} 0-kept {r['frames_0_kept']} "
            f"far>40px {r['far40_per_frame']:.1f}/frame ({r['far40_frac_of_kept']:.3f} of kept)")
    (BOUT / f'mask_check{args.tag}.json').write_text(json.dumps(res, indent=1))


def plan_mode(args):
    clips = clips_all()
    rng = np.random.default_rng(0)
    rows = []
    for c in clips:
        f = rng.choice(GRID_B, args.k + args.calib, replace=False)
        tr, ca = np.sort(f[:args.k]), np.sort(f[args.k:])
        rows += [{'clip': c, 'i': int(i), 'role': 'train'} for i in tr] + [{'clip': c, 'i': int(i), 'role': 'calib'} for i in ca]
    P = pd.DataFrame(rows).sort_values(['clip', 'i']).reset_index(drop=True)
    assert not P.duplicated(['clip', 'i']).any()
    BOUT.mkdir(parents=True, exist_ok=True)
    P.to_parquet(BOUT / 'btrain_frames.parquet')
    log(f"plan: {len(clips)} clips, K {args.k} train + {args.calib} calib, {len(P)} frames "
        f"({(P.role == 'train').sum()} train, {(P.role == 'calib').sum()} calib)")


def encode_mode(args):
    import torch
    from eci_encode import Enc
    t00 = time.time()
    dev = torch.device('cuda')
    stage = stage_dir()
    TOKR3.mkdir(parents=True, exist_ok=True)
    key = f'btrain_{args.shard}of{args.nshard}' + (f'_smoke{args.smoke}' if args.smoke else '')
    if (TOKR3 / f'{key}.json').exists():
        log(f'[SKIP] {key}')
        return
    P = pd.read_parquet(BOUT / 'btrain_frames.parquet')
    clips = np.array_split(np.array(sorted(P['clip'].unique())), args.nshard)[args.shard]
    if args.smoke:
        clips = clips[:args.smoke]
    kps = keypoints(list(clips))  # check only (recall in the meta)
    enc = Enc(dev)
    log(f'{key}: {len(clips)} clips, variant {args.variant}, {torch.cuda.get_device_name(0)}')
    A = {'m448o_t': [], 'm448o_p60': [], 'pos448': []}
    meta, cinfo, rows, rec, kp_rows = [], {}, [], [], []
    tdec = 0.0
    for ci, c in enumerate(clips):
        R = P[P['clip'] == c]
        t0 = time.time()
        fr = decode_clip(VSET, c, stage, 'rgb24')
        tdec += time.time() - t0
        if len(fr) != T_CLIP:
            raise RuntimeError(f'{c}: {len(fr)} frames')
        ts = R.i.values
        mask, info, gt = beetle_mask(enc, fr, ts, args.variant, dev)
        cinfo[c] = info
        ot = enc.orbit(fr[ts])
        op = enc.orbit(fr[np.maximum(ts - LAG, 0)])
        mnp = mask.cpu().numpy()
        rec.append(kp_recall(mnp, kps[c][ts], args.conv))
        for q in range(len(R)):
            p = torch.nonzero(mask[q]).flatten()
            A['m448o_t'].append(ot[q, p].half().cpu().numpy())
            A['m448o_p60'].append(op[q, p].half().cpu().numpy())
            A['pos448'].append(p.cpu().numpy().astype(np.int16))
            meta.append({'clip': c, 'i': int(ts[q]), 'n448': int(len(p)), 'role': R.role.values[q]})
        if args.shard == 0 and len(rows) < 16 and ci % 3 == 0:
            rows.append((gt[0], mnp[0], f'{c[:8]} i{ts[0]} n{int(mnp[0].sum())}'))
            kp_rows.append(kps[c][ts[0]])
        if ci == 0:
            cinfo[c]['orbit_finite'] = bool(torch.isfinite(ot).all() and torch.isfinite(op).all())
        del fr, ot, op
        if ci % 25 == 0:
            log(f'  clip {ci}/{len(clips)}: {enc.n_img} images, gpu {enc.gpu_s:.0f}s, decode {tdec:.0f}s, '
                f'kept/frame {info["kept_per_frame"]:.1f} thr {info["thr"]:.3f}')
    Fm = pd.DataFrame(meta)
    assert len(Fm) == int(P['clip'].isin(clips).sum())
    for k, v in A.items():
        a = np.concatenate(v)
        assert len(a) == Fm.n448.sum()
        if a.dtype == np.float16:
            assert np.isfinite(a).all(), k
        np.save(stage / f'{key}_{k}.npy', a)
        shutil.copyfile(stage / f'{key}_{k}.npy', TOKR3 / f'{key}_{k}.npy')
        (stage / f'{key}_{k}.npy').unlink()
        log(f'  saved {k} {a.shape} {a.dtype}')
    Fm.to_parquet(TOKR3 / f'{key}_frames.parquet')
    if rows:
        sheet_kp(rows, kp_rows, args.conv, BOUT / f'encode_sheet_{key}.jpg')
    rec = np.concatenate(rec)
    kept = Fm.n448.values
    info = {'set': 'btrain', 'shard': args.shard, 'nshard': args.nshard, 'clips': len(clips), 'frames': len(Fm),
            'frames_train': int((Fm.role == 'train').sum()), 'frames_calib': int((Fm.role == 'calib').sum()),
            'tokens_train': int(Fm.n448[Fm.role == 'train'].sum()), 'tokens_calib': int(Fm.n448[Fm.role == 'calib'].sum()),
            'mask_variant': args.variant, 'mask_rule': VARIANTS[args.variant], 'lag': LAG, 'kinds': list(A),
            'orbit': 'eci_encode.Enc.orbit (D4 8-transform mean, 448 bicubic, fp32 TF32 off), stored fp16',
            'kp_conv': args.conv, 'kp_recall_animal0': float(rec[:, 0].mean()), 'kp_recall_animal1': float(rec[:, 1].mean()),
            'images_encoded': enc.n_img, 'gpu_s': round(enc.gpu_s, 1), 'decode_s': round(tdec, 1),
            'kept448_mean': float(kept.mean()), 'kept448_frac_grid': float(kept.mean() / 1024),
            'frames_with_0_kept': int((kept == 0).sum()), 'lag_clipped_frames': int((Fm.i.values - LAG < 0).sum()),
            'thr_quantiles': np.quantile([v['thr'] for v in cinfo.values()], [0, .5, 1]).round(4).tolist(),
            'gpu': torch.cuda.get_device_name(0), 'wall_s': round(time.time() - t00, 1), 'per_clip': cinfo}
    (TOKR3 / f'{key}.json').write_text(json.dumps(info, indent=1))
    log('done: ' + json.dumps({k: v for k, v in info.items() if k != 'per_clip'}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['look', 'mask', 'plan', 'encode'])
    ap.add_argument('--n', type=int, default=24, help='look / mask: number of sample clips')
    ap.add_argument('--sheet-clips', type=int, default=6)
    ap.add_argument('--conv', default='a0yx_a1xy', choices=list(CONV), help='keypoint convention (verified in look)')
    ap.add_argument('--variants', nargs='*', default=[])
    ap.add_argument('--tag', default='')
    ap.add_argument('--k', type=int, default=0, help='plan: train frames per clip')
    ap.add_argument('--calib', type=int, default=2, help='plan: calibration frames per clip')
    ap.add_argument('--variant', default='', help='encode: mask variant')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshard', type=int, default=8)
    ap.add_argument('--smoke', type=int, default=0)
    args = ap.parse_args()
    {'look': look, 'mask': mask_mode, 'plan': plan_mode, 'encode': encode_mode}[args.mode](args)


if __name__ == '__main__':
    main()
