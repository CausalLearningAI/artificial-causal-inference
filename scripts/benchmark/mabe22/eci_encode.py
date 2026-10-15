"""
MABe22 mouse ECI evaluation, step 2 (GPU Slurm job, sharded over clips): local decode, DINOv2-base encode, animal mask.

Per clip (all label-free):
    decode    the .avi copied to local disk, ffmpeg 6.1.1 (ffmpeg611 env) -> rgb24 224 x 224, all 1,801 frames
    encode    DINOv2-base (facebook/dinov2-base, fp32, TF32 off), patch tokens of the last layer:
                448  the 224 frame upsampled bicubically to 448 (torch, on the 0-255 scale, rounded / clipped to uint8
                     as an image resize would be), ImageNet normalisation -> 32 x 32 grid (lab geometry)
                224  the native frame -> 16 x 16 grid (arm A5)
    mask      the lab rule 'fg448' (src/eci/foreground.py FG_RULE / foreground_parts) on the 448 grid, per clip:
                pix_bg   per-pixel 0.98 quantile of 30 evenly spaced grey frames ('clip'), or its per-pixel max with the
                         global userTrain background ('max', --pixbg; keeps mice that never move in a 60 s clip)
                dark     fraction of the 7 x 7 pixel block of each 448 patch with grey < 60 and > 40 below pix_bg
                bg tok   per-position median 448 token of the 30 frames, excluding frames whose patch is dark
                         foreground (dark > 0.15), fallback to the plain median with < 8 usable frames
                thr      0.995 quantile of 1 - cos(token, bg tok) over the 30 frames' patches > 2 patches from any
                         dark-cue patch
                mask     dilate_1(drop_isolated(dist > thr) OR dark > 0.15), every 8-neighbour (fg448)
              the 16 x 16 mask of the 224 arm = a 224 patch is kept when any of its 2 x 2 child 448 patches is kept
    tokens    for each plan frame (label index i -> video frame i + offset) and its 1 s lag (video frame - 30):
                m448   masked 448 tokens of t and lag (lag read at t's positions)
                m224   masked 224 tokens of t and lag
                all448 (eval only) every 448 token of t and lag (arm A6, no mask)
                rnd448 (train / calib only) a uniformly random subset of the 1,024 positions of the SAME size as the
                       frame's mask (arm A6 training pool at a matched token count; rng seeded by clip and i)
Outputs: OUT/tok/<set>_<i>of<n>_<kind>_{t,p}.npy (fp16), _pos<grid>.npy (int16), _<set>_<i>of<n>_frames.parquet,
         <set>_<i>of<n>.json; OUT/masks/sheet_<set>_<i>.jpg (shards 0-1: overlay of the 448 mask on 16 frames).
"""
import argparse
import json
import shutil
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts/eci'))
from eci_common import LAG, OUT, PLAN, TOK, decode_clip, log, offset, stage_dir  # noqa: E402

SETS = {'train': ('mouse_userTrain', 'train_frames.parquet'), 'eval': ('mouse_submission', 'eval_frames.parquet')}
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def plan(set_):
    P = pd.read_parquet(PLAN / SETS[set_][1])
    assert P.equals(P.sort_values(['clip', 'i']).reset_index(drop=True))
    return P


def shard_clips(P, i, n):
    clips = np.array(sorted(P['clip'].unique()))
    return np.array_split(clips, n)[i]


class Enc:
    def __init__(self, dev):
        import torch
        from transformers import AutoModel
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.torch, self.dev = torch, dev
        self.model = AutoModel.from_pretrained('facebook/dinov2-base').to(dev).eval()
        self.model.requires_grad_(False)
        self.mean = torch.tensor(MEAN, device=dev)[None, :, None, None]
        self.std = torch.tensor(STD, device=dev)[None, :, None, None]
        self.n_img = 0
        self.gpu_s = 0.0

    def __call__(self, frames, size, bs=64):
        """frames (n, 224, 224, 3) uint8 numpy -> (n, (size/14)^2, 768) float32 GPU tokens."""
        torch = self.torch
        F = torch.nn.functional
        out = []
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.inference_mode():
            for a in range(0, len(frames), bs):
                x = torch.from_numpy(np.ascontiguousarray(frames[a:a + bs])).to(self.dev).permute(0, 3, 1, 2).float()
                if size != x.shape[-1]:
                    x = F.interpolate(x, size=(size, size), mode='bicubic', align_corners=False).round().clamp(0, 255)
                x = (x / 255 - self.mean) / self.std
                h = self.model(pixel_values=x).last_hidden_state[:, 1:]
                assert h.shape[1] == (size // 14) ** 2 and torch.isfinite(h).all()
                out.append(h.float())
        torch.cuda.synchronize()
        self.gpu_s += time.time() - t0
        self.n_img += len(frames)
        return torch.cat(out)

    def orbit(self, frames, bs=16):
        """arm A7: frames (n, 224, 224, 3) -> (n, 1024, 768) float32 tokens averaged over the 8 D4 transforms of the 448
        input, each transformed grid read back at the original patch positions (invariant_tokens 'orb' kind:
        d4_torch, priors.d4_pos_map; token of original patch p = h_g[:, d4_pos_map(g)[p]])."""
        torch = self.torch
        F = torch.nn.functional
        import invariant_tokens as it
        from src.eci.priors import d4_pos_map
        if not hasattr(self, 'maps'):
            assert it.check_d4_torch()
            self.maps = torch.stack([torch.from_numpy(d4_pos_map(g)) for g in range(8)]).to(self.dev)
        out = []
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.inference_mode():
            for a in range(0, len(frames), bs):
                x = torch.from_numpy(np.ascontiguousarray(frames[a:a + bs])).to(self.dev).permute(0, 3, 1, 2).float()
                x = F.interpolate(x, size=(448, 448), mode='bicubic', align_corners=False).round().clamp(0, 255)
                x = (x / 255 - self.mean) / self.std
                acc = 0
                for g in range(8):
                    h = self.model(pixel_values=it.d4_torch(x, g)).last_hidden_state[:, 1:].float()
                    acc = acc + h[:, self.maps[g]]
                out.append(acc / 8)
        torch.cuda.synchronize()
        self.gpu_s += time.time() - t0
        self.n_img += 8 * len(frames)
        return torch.cat(out)


def _robust_line(u, v):
    a = np.polyfit(u, v, 1)
    for _ in range(2):
        r = np.abs(np.polyval(a, u) - v)
        keep = r <= 3
        if keep.sum() < 10:
            break
        a = np.polyfit(u[keep], v[keep], 1)
    return a


def arena_quad(fl):
    """fl (224, 224) bool floor pixels -> {'inside': bool (224, 224) 2 px inside the fitted floor quadrilateral}."""
    H = fl.shape[0]
    rows = np.flatnonzero(fl.mean(1) > 0.3)
    cols = np.flatnonzero(fl.mean(0) > 0.3)
    if len(rows) < 40 or len(cols) < 40:
        return {'inside': np.zeros_like(fl), 'ok': False}
    rr = np.arange(rows[0] + 10, rows[-1] - 9)
    cc = np.arange(cols[0] + 10, cols[-1] - 9)
    left = np.array([np.flatnonzero(fl[r])[0] if fl[r].any() else -1 for r in rr])
    right = np.array([np.flatnonzero(fl[r])[-1] if fl[r].any() else -1 for r in rr])
    top = np.array([np.flatnonzero(fl[:, c])[0] if fl[:, c].any() else -1 for c in cc])
    bot = np.array([np.flatnonzero(fl[:, c])[-1] if fl[:, c].any() else -1 for c in cc])
    L = _robust_line(rr[left >= 0], left[left >= 0])
    R = _robust_line(rr[right >= 0], right[right >= 0])
    T = _robust_line(cc[top >= 0], top[top >= 0])
    B = _robust_line(cc[bot >= 0], bot[bot >= 0])
    yy, xx = np.mgrid[0:H, 0:H]
    inside = ((xx >= np.polyval(L, yy) + 2) & (xx <= np.polyval(R, yy) - 2) & (yy >= np.polyval(T, xx) + 2)
              & (yy <= np.polyval(B, xx) - 2))
    return {'inside': inside, 'ok': True}


def clip_mask(enc, frames, ts, lags, pixbg_mode, glob_bg, dev, extra_lags=()):
    """-> dict of per-frame masks / tokens for the sample frames (video indices ts, lags) of one clip."""
    import torch
    import torch.nn.functional as F
    from src.eci.foreground import FG_RULE, cosine_distance, foreground_parts, patch_background, video_threshold
    T = len(frames)
    bgi = np.linspace(0, T - 1, 30).round().astype(int)
    grey_all = (0.299 * frames[..., 0] + 0.587 * frames[..., 1] + 0.114 * frames[..., 2]).round().clip(0, 255)
    grey_all = grey_all.astype(np.uint8)  # PIL 'L' conversion (rounded)
    g = torch.from_numpy(grey_all).to(dev)
    pix = g[torch.as_tensor(bgi, device=dev)].float().kthvalue(int(round(0.98 * 30)), dim=0).values.round()
    if pixbg_mode == 'max':
        pix = torch.maximum(pix, torch.from_numpy(glob_bg).to(dev).float())
    hole = torch.zeros_like(pix, dtype=torch.bool)
    quad = None
    if pixbg_mode == 'arena':
        # a mouse that never moves in the 60 s clip is dark in pix_bg too, so 'darker than background' misses it (seen
        # on the first mask sheet: still mice at the arena edges / corners). The arena floor is a bright quadrilateral:
        # its 4 edges are fitted robustly (line fit per side, 2 passes, residual > 3 px dropped, so the notches cut by
        # resting mice are outliers) and, 2 px inside it, a dark pixel counts with the drop condition waived.
        quad = arena_quad(pix.cpu().numpy() > 100)
        hole = torch.from_numpy(quad['inside']).to(dev)
    if pixbg_mode == 'floor':
        # a mouse that never moves in the 60 s clip is dark in pix_bg too; inside the bright arena floor it is a HOLE
        # of the floor region (pix_bg > 100, closed, holes filled): there the drop-below-background condition is waived
        from scipy import ndimage as ndi
        fl = pix.cpu().numpy() > 100
        fl_c = ndi.binary_closing(fl, np.ones((5, 5), bool), iterations=2)
        hole = torch.from_numpy(ndi.binary_fill_holes(fl_c) & ~fl).to(dev)

    def dark_of(idx):
        gg = g[torch.as_tensor(idx, device=dev)].float()
        dk = (gg < FG_RULE['dark_abs']) & (((pix[None] - gg) > FG_RULE['dark_rel']) | hole[None])
        return F.avg_pool2d(dk.float()[:, None], 7).flatten(1)  # (n, 1024)

    xl = {L_: np.maximum(ts - L_, 0) for L_ in extra_lags}
    need = np.unique(np.concatenate([bgi, ts, lags] + list(xl.values())))
    pos_of = {v: j for j, v in enumerate(need)}
    tok = enc(frames[need], 448)  # (n, 1024, 768)
    jb = torch.tensor([pos_of[v] for v in bgi], device=dev)
    dark_bg = dark_of(bgi)
    bg, n_ok = patch_background(tok[jb], exclude=dark_bg > FG_RULE['dark_frac'], min_frames=8)
    dist_bg = cosine_distance(tok[jb], bg)
    thr = video_threshold(dist_bg.cpu().numpy(), dark_bg.cpu().numpy(), FG_RULE)
    jt = torch.tensor([pos_of[v] for v in ts], device=dev)
    jl = torch.tensor([pos_of[v] for v in lags], device=dev)
    dist_t = cosine_distance(tok[jt], bg)
    dark_t = dark_of(ts)
    feat, dk, mask = foreground_parts(dist_t, dark_t, thr, FG_RULE)
    m16 = F.max_pool2d(mask.float().view(-1, 1, 32, 32), 2).flatten(1) > 0
    need224 = np.unique(np.concatenate([ts, lags]))
    p224 = {v: j for j, v in enumerate(need224)}
    tok224 = enc(frames[need224], 224)
    j2t = torch.tensor([p224[v] for v in ts], device=dev)
    j2l = torch.tensor([p224[v] for v in lags], device=dev)
    info = {'thr': thr, 'hole_px': int(hole.sum()), 'fallback_positions': int((n_ok < 8).sum()), 'feat_per_frame': float(feat.sum(1).float().mean()),
            'dark_per_frame': float(dk.sum(1).float().mean()), 'kept_per_frame': float(mask.sum(1).float().mean())}
    info['arena_inside_px'] = int(hole.sum())
    xt = {L_: tok[torch.tensor([pos_of[v] for v in vv], device=dev)] for L_, vv in xl.items()}
    return {'tok_xl': xt, 'tok_t': tok[jt], 'tok_p': tok[jl], 'mask': mask, 'tok224_t': tok224[j2t], 'tok224_p': tok224[j2l],
            'm16': m16, 'grey_t': grey_all[ts], 'info': info}


def sheet(rows, path):
    """rows: list of (grey 224 uint8, mask (1024,) bool, title) -> a 4 x 4 JPG, kept patches tinted red, x2 scale."""
    from PIL import Image, ImageDraw
    tiles = []
    for grey, mask, title in rows[:16]:
        rgb = np.repeat(grey[..., None], 3, 2).astype(np.float32)
        m = np.kron(mask.reshape(32, 32), np.ones((7, 7))).astype(bool)
        rgb[m, 0] = rgb[m, 0] * 0.5 + 127
        rgb[m, 1:] *= 0.6
        im = Image.fromarray(rgb.clip(0, 255).astype(np.uint8)).resize((448, 448), Image.NEAREST)
        ImageDraw.Draw(im).text((4, 4), title, fill=(255, 255, 0))
        tiles.append(im)
    W = Image.new('RGB', (4 * 448, ((len(tiles) + 3) // 4) * 448))
    for k, t in enumerate(tiles):
        W.paste(t, ((k % 4) * 448, (k // 4) * 448))
    path.parent.mkdir(parents=True, exist_ok=True)
    W.save(path, quality=90)


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', choices=list(SETS), required=True)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshard', type=int, default=1)
    ap.add_argument('--pixbg', choices=['clip', 'max', 'floor', 'arena'], default='clip')
    ap.add_argument('--extra-lags', nargs='*', type=int, default=[6, 15, 60],
                    help='extra lag frames (30 fps) stored at the t-frame mask positions (448 only); clipped to frame 0')
    ap.add_argument('--reuse-mask', default='', help='token dir whose pos448 (same set / shard split) is reused')
    ap.add_argument('--orbit', action='store_true', help='arm A7: also store D4 orbit-averaged t / lag tokens')
    ap.add_argument('--only-lags', action='store_true', help='save only the extra-lag tokens (+ pos448, frames)')
    ap.add_argument('--smoke', type=int, default=0, help='only the first N clips of the shard')
    args = ap.parse_args()
    t00 = time.time()
    dev = torch.device('cuda')
    stage = stage_dir()
    TOK.mkdir(parents=True, exist_ok=True)
    key = f'{args.set}_{args.shard}of{args.nshard}' + (f'_smoke{args.smoke}' if args.smoke else '')
    if (TOK / f'{key}.json').exists():
        log(f'[SKIP] {key}')
        return
    P = plan(args.set)
    clips = shard_clips(P, args.shard, args.nshard)
    if args.smoke:
        clips = clips[:args.smoke]
    off = offset()
    vset = SETS[args.set][0]
    glob_bg = np.load(PLAN / 'pixbg_global.npy')
    REF = None
    if args.reuse_mask:
        rk = f'{args.set}_{args.shard}of{args.nshard}'
        RF = pd.read_parquet(Path(args.reuse_mask) / f'{rk}_frames.parquet')
        REF = {'F': RF, 'pos': np.load(Path(args.reuse_mask) / f'{rk}_pos448.npy'), 'st': np.r_[0, np.cumsum(RF.n448.values)]}
    enc = Enc(dev)
    log(f'{key}: {len(clips)} clips, offset {off}, pixbg {args.pixbg}, {torch.cuda.get_device_name(0)}')
    A = {k: [] for k in ('m448_t', 'm448_p', 'pos448', 'm224_t', 'm224_p', 'pos224', 'all448_t', 'all448_p',
                         'rnd448_t', 'rnd448_p', 'posrnd')}
    meta, cinfo, sheet_rows = [], {}, []
    is_eval = args.set == 'eval' and not args.only_lags
    tdec = 0.0
    for ci, c in enumerate(clips):
        R = P[P['clip'] == c]
        t0 = time.time()
        fr = decode_clip(vset, c, stage, 'rgb24')
        tdec += time.time() - t0
        if len(fr) != 1801:
            raise RuntimeError(f'{c}: {len(fr)} frames')
        ts = R.i.values + off
        lags = ts - LAG
        assert lags.min() >= 0 and ts.max() < len(fr)
        M = clip_mask(enc, fr, ts, lags, args.pixbg, glob_bg, dev, tuple(args.extra_lags))
        cinfo[c] = M['info']
        mask, m16 = M['mask'], M['m16']
        if REF is not None:  # --reuse-mask: the t-frame positions of an existing token store (bit-identical masks)
            rr = np.flatnonzero(REF['F']['clip'].values == c)
            assert (REF['F'].i.values[rr] == R.i.values).all()
            mref = torch.zeros_like(mask)
            for q, r_ in enumerate(rr):
                mref[q, torch.from_numpy(REF['pos'][REF['st'][r_]:REF['st'][r_ + 1]].astype(np.int64)).to(dev)] = True
            cinfo[c]['mask_agree_with_ref'] = float((mref == mask).float().mean())
            mask = mref
        if args.orbit:
            ot, op = enc.orbit(fr[ts]), enc.orbit(fr[lags])
        for q in range(len(R)):
            p = torch.nonzero(mask[q]).flatten()
            if args.orbit:
                A.setdefault('m448o_t', []).append(ot[q, p].half().cpu().numpy())
                A.setdefault('m448o_p', []).append(op[q, p].half().cpu().numpy())
                if ci == 0 and q == 0:
                    cinfo[c]['orbit_cos_vs_identity'] = float(torch.nn.functional.cosine_similarity(
                        ot[q, p], M['tok_t'][q, p], dim=-1).mean())
            p2 = torch.nonzero(m16[q]).flatten()
            A['m448_t'].append(M['tok_t'][q, p].half().cpu().numpy())
            A['m448_p'].append(M['tok_p'][q, p].half().cpu().numpy())
            A['pos448'].append(p.cpu().numpy().astype(np.int16))
            for L_, tx in M['tok_xl'].items():
                A.setdefault(f'm448_p{L_}', []).append(tx[q, p].half().cpu().numpy())
            A['m224_t'].append(M['tok224_t'][q, p2].half().cpu().numpy())
            A['m224_p'].append(M['tok224_p'][q, p2].half().cpu().numpy())
            A['pos224'].append(p2.cpu().numpy().astype(np.int16))
            if is_eval:
                A['all448_t'].append(M['tok_t'][q].half().cpu().numpy())
                A['all448_p'].append(M['tok_p'][q].half().cpu().numpy())
            else:
                g = np.random.default_rng([zlib.crc32(c.encode()), int(R.i.values[q])])
                pr = np.sort(g.choice(1024, len(p), replace=False))
                prt = torch.from_numpy(pr).to(dev)
                A['rnd448_t'].append(M['tok_t'][q, prt].half().cpu().numpy())
                A['rnd448_p'].append(M['tok_p'][q, prt].half().cpu().numpy())
                A['posrnd'].append(pr.astype(np.int16))
            meta.append({'clip': c, 'i': int(R.i.values[q]), 'n448': int(len(p)), 'n224': int(len(p2))})
            if args.shard < 2 and len(sheet_rows) < 16 and q == 0 and ci % 3 == 0:
                sheet_rows.append((M['grey_t'][q], mask[q].cpu().numpy(), f'{c[:8]} i{R.i.values[q]} n{len(p)}'))
        del fr, M
        if ci % 50 == 0:
            log(f'  clip {ci}/{len(clips)}: {enc.n_img} images, gpu {enc.gpu_s:.0f}s, decode {tdec:.0f}s, '
                f'kept/frame {cinfo[c]["kept_per_frame"]:.1f} thr {cinfo[c]["thr"]:.3f}')
    Fm = pd.DataFrame(meta)
    assert len(Fm) == int(P['clip'].isin(clips).sum())
    for k, v in A.items():
        if not v or (args.only_lags and not (k.startswith('m448_p') and k != 'm448_p') and k != 'pos448'
                     and not k.startswith('m448o')):
            continue
        a = np.stack(v) if k.startswith('all448') else np.concatenate(v)
        np.save(stage / f'{key}_{k}.npy', a)
        shutil.copyfile(stage / f'{key}_{k}.npy', TOK / f'{key}_{k}.npy')
        (stage / f'{key}_{k}.npy').unlink()
        log(f'  saved {k} {a.shape}')
    Fm.to_parquet(TOK / f'{key}_frames.parquet')
    if sheet_rows:
        sheet(sheet_rows, OUT / 'masks' / f'sheet_{args.set}_{args.shard}.jpg')
    kept = Fm.n448.values
    info = {'set': args.set, 'shard': args.shard, 'nshard': args.nshard, 'clips': len(clips), 'frames': len(Fm),
            'offset': off, 'pixbg': args.pixbg, 'extra_lags': args.extra_lags,
            'extra_lag_clipped_frames': int(sum(int((P[P['clip'] == c].i.values + off - 60 < 0).sum()) for c in clips)), 'images_encoded': enc.n_img, 'gpu_s': round(enc.gpu_s, 1),
            'decode_s': round(tdec, 1), 'img_per_gpu_s': round(enc.n_img / max(enc.gpu_s, 1e-9), 1),
            'kept448_mean': float(kept.mean()), 'kept448_frac_grid': float(kept.mean() / 1024),
            'frames_with_0_kept': int((kept == 0).sum()), 'kept224_mean': float(Fm.n224.mean()),
            'thr_quantiles': np.quantile([v['thr'] for v in cinfo.values()], [0, .5, 1]).round(4).tolist(),
            'gpu': torch.cuda.get_device_name(0), 'wall_s': round(time.time() - t00, 1), 'per_clip': cinfo}
    (TOK / f'{key}.json').write_text(json.dumps(info, indent=1))
    log('done: ' + json.dumps({k: v for k, v in info.items() if k != 'per_clip'}))


if __name__ == '__main__':
    main()
