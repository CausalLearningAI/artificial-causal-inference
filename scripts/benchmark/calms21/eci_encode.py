"""
CalMS21 ECI - token stores for the resolution x encoder test (rules: eci_res.sh). GPU Slurm job, sharded over videos.

Per video (label-free; the plan's frames only):
    decode   the MP4 is copied to local disk and stream-decoded ONCE by ffmpeg 6.1.1 (rgb24, native 1024 x 570); the job
             keeps only the frames it needs (32 background frames evenly spaced over the video, every plan frame t, its
             lag t - 60 (2 s at 30 fps) for the DINO configs, the 16-frame clips of the V-JEPA configs); --workers
             videos are decoded in parallel threads with a bounded look-ahead, overlapping the GPU.
    canvas   the 1024 x 570 frame is PADDED (no crop, no distortion) to a 1024 x 1024 square: 227 rows above and below
             filled with the ImageNet mean colour (124, 116, 104); then resized on the GPU to the encoder input S
             (bicubic, antialias on when downscaling; rounded / clipped to 0-255), ImageNet mean / std.
    mask     on the 32 x 32 grid of the canvas (one cell = 32 x 32 canvas px = one patch of DINOv2 at 448 / DINOv3 and
             V-JEPA at 512), --mask rule (see MASKS); cells with no video pixel (pure padding) are never kept. On a
             64 x 64 grid a patch is kept iff its 32-grid parent cell is kept (the same image region at every grid).
    tokens   last block output after the model's final LayerNorm (Encoder of mabe22/eci_enc_encode.py), prefix tokens
             dropped; masked tokens of t and, for DINO, of t - 60 at t's positions; V-JEPA: clip of 16 frames at a
             stride of 8 video frames, t at clip position 7 (frames t - 56 .. t + 64), token = tubelet 3 = frames
             (t - 8, t); crop_size = S so the RoPE grid is S / 16 wide (integer positions 0..G-1, the lab convention).
             --bf16: bf16 autocast (SDPA attention throughout).
Configs: d2_448 (DINOv2-B/14, 32^2), d2_896 (64^2), d3_512 (DINOv3-B/16 web, 32^2), d3_1024 (64^2), vj_512 (V-JEPA 2-L,
32^2), vj_1024 (64^2). Modes:
    encode   --set train|eval --shard i --nshard n --configs ...: writes TOK/<set>_<i>of<n>_<config>_{t,p60}.npy (fp16),
             _<config>_pos.npy (int16, own grid), <set>_<i>of<n>_frames.parquet (video, i, v, n32), _pos32.npy (mask),
             <set>_<i>of<n>_<configs>.json (timing / throughput); --reuse-mask: read pos32 of an earlier pass instead
             of recomputing it (a V-JEPA-only pass then needs no DINOv2).
    maskqa   --videos: every MASKS rule on 24 plan frames of each listed video: kept cells, keypoint recall per mouse
             (keypoints used for this check only), mask sheets -> OUT/mask/
    bench    random input, tokens / s and peak memory per config (incl. the 129 x 129 stress grid, d3_2064)
"""
import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'mabe22'))
sys.path.insert(0, str(HERE.parents[2]))
from calms_common import (CANVAS, DER, OUT, PADV, PADY, PLAN, TOK, H, W, decode_frames, log, offset,  # noqa: E402
                          stage_dir, video_path)

CONFIGS = {'d2_448': ('dinov2b', 448), 'd2_896': ('dinov2b', 896), 'd3_512': ('dinov3b', 512),
           'd3_1024': ('dinov3b', 1024), 'vj_512': ('vjepal', 512), 'vj_1024': ('vjepal', 1024),
           'd3_2064': ('dinov3b', 2064),
           # addendum 2 (label-free layer rule, block at ~5/6 depth): DINOv3-B L10, DINOv3-L L20; DINOv2-L last layer
           'd3_512_L10': ('dinov3b', 512, 10), 'd3_1024_L10': ('dinov3b', 1024, 10),
           'd3l_512_L20': ('dinov3l', 512, 20), 'd3l_1024_L20': ('dinov3l', 1024, 20), 'd2l_448': ('dinov2l', 448),
           # addendum 3 (size-matched (b)): DINOv2-L at its 5/6-depth block L20
           'd2l_448_L20': ('dinov2l', 448, 20), 'd2l_896_L20': ('dinov2l', 896, 20),
           # detail-vs-density control (addendum 1 (d)): same input size / grid, source first shrunk to plan/lo_side.txt
           'd3_1024lo_L10': ('dinov3b', 1024, 10), 'd3l_1024lo_L20': ('dinov3l', 1024, 20), 'd2_896lo': ('dinov2b', 896)}
LAG = 60
NBG = 32
CLIP, TPOS, VJS = 16, 7, 8
MASKS = {
    # token cue (DINOv2-448 cosine distance to the video's per-position background token > the video threshold,
    # isolated cells dropped) OR pixel cue (fraction of the cell's pixels > 0.15), then a 1-cell dilation
    'dark': 'MABe rule: pixel cue = grey < 60 and > 40 below the per-pixel background',
    'dark_bright': 'pixel cue = dark OR grey > 40 above the per-pixel background (white mouse)',
    'absdiff': 'pixel cue = |grey - background| > 40 (dark or bright change)',
    'absdiff_floor': 'absdiff OR grey < 60 anywhere on the video picture (a black mouse that never moves)',
}


def device():
    import torch
    if torch.cuda.is_available():
        return torch.device('cuda')
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    return torch.device('cpu')


def sync(dev):
    import torch
    if dev.type == 'cuda':
        torch.cuda.synchronize()


def gpu_name(dev):
    import torch
    return torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu'


def canvas_gpu(frames, dev):
    """list / array of (570, 1024, 3) uint8 -> (n, 3, 1024, 1024) float 0-255 GPU tensor."""
    import torch
    x = torch.from_numpy(np.ascontiguousarray(np.stack(frames))).to(dev)
    c = torch.empty((len(frames), CANVAS, CANVAS, 3), dtype=torch.uint8, device=dev)
    c[:] = torch.tensor(PADV, dtype=torch.uint8, device=dev)
    c[:, PADY:PADY + H] = x
    return c.permute(0, 3, 1, 2).float()


class Enc:
    """thin wrapper over mabe22 eci_enc_encode.Encoder (models, hooks, final-LN, prefix drop) with a canvas prep."""

    def __init__(self, cfg, dev, bf16):
        import eci_enc_encode as EE
        name, S, *ly = CONFIGS[cfg]
        if name == 'dinov2l':  # same module layout as dinov2-base: build through the 'dinov2b' branch
            keep = EE.MODELS['dinov2b']
            EE.MODELS['dinov2b'] = ('facebook/dinov2-large', 14, 1, 24)
            self.last = ly[0] if ly else 24
            try:
                self.e = EE.Encoder('dinov2b', S, [self.last], dev, 'extrap', VJS, bf16)
            finally:
                EE.MODELS['dinov2b'] = keep
        else:
            self.last = ly[0] if ly else EE.MODELS[name][3]
            self.e = EE.Encoder(name, S, [self.last], dev, 'extrap', VJS, bf16)
        self.cfg, self.S, self.G, self.video, self.dev = cfg, S, self.e.G, self.e.video, dev
        self.lo = int((PLAN / 'lo_side.txt').read_text().split()[0]) if 'lo' in cfg.split('_')[1] else 0
        self.dim = self.e.dim
        self.gpu_s, self.n_in, self.n_tok = 0.0, 0, 0
        import torch
        self.torch = torch
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=dev)[None, :, None, None]
        self.std = torch.tensor([0.229, 0.224, 0.225], device=dev)[None, :, None, None]

    def prep(self, x):
        F = self.torch.nn.functional
        if self.lo:  # shrink the canvas (antialiased bicubic) so the mouse is ~40 px long, then bicubic back up to S
            x = F.interpolate(x, size=(self.lo, self.lo), mode='bicubic', align_corners=False,
                              antialias=True).round().clamp(0, 255)
            x = F.interpolate(x, size=(self.S, self.S), mode='bicubic', align_corners=False).round().clamp(0, 255)
        elif self.S != CANVAS:
            x = F.interpolate(x, size=(self.S, self.S), mode='bicubic', align_corners=False,
                              antialias=self.S < CANVAS).round().clamp(0, 255)
        return (x / 255 - self.mean) / self.std

    def run(self, fr, bs, pos=None):
        """fr: list of frames (images) or of 16-frame lists (clips) -> (n, G*G, d) fp16 GPU; with pos (list of index
        tensors, one per input) -> list of (n_i, d) fp16 CPU numpy arrays of the kept tokens only (bounded memory)."""
        torch = self.torch
        out = []
        sync(self.dev)
        t0 = time.time()
        with torch.inference_mode():
            for a in range(0, len(fr), bs):
                part = fr[a:a + bs]
                if self.video:
                    x = self.prep(canvas_gpu([f for c in part for f in c], self.dev))
                    x = x.reshape(len(part), CLIP, 3, self.S, self.S)
                    self.n_tok += len(part) * (CLIP // 2) * self.G ** 2
                else:
                    x = self.prep(canvas_gpu(part, self.dev))
                    self.n_tok += len(part) * (self.e.npre + self.G ** 2)
                h = self.e.forward(x)[self.last]
                if pos is None:
                    out.append(h.half())
                else:
                    out += [h[j, pos[a + j]].half().cpu().numpy() for j in range(len(part))]
                del h, x
                self.n_in += len(part)
        sync(self.dev)
        self.gpu_s += time.time() - t0
        return torch.cat(out) if pos is None else out


def video_cells():
    """(1024,) bool: 32-grid cells holding at least one video pixel (rows 227..796 of the canvas)."""
    r = np.arange(32)
    ok = (r * 32 + 31 >= PADY) & (r * 32 < PADY + H)
    return np.repeat(ok, 32)


def to_grid(pos32, G):
    if G == 32:
        return pos32.astype(np.int16)
    assert G == 64
    m = np.zeros((32, 32), bool)
    m.flat[pos32.astype(np.int64)] = True
    return np.flatnonzero(np.kron(m, np.ones((2, 2), bool))).astype(np.int16)


def pixel_cues(grey, bg, rules):
    """grey (n, 1024, 1024) float GPU canvas grey, bg (1024, 1024) -> {rule: (n, 1024) cell fraction}"""
    import torch.nn.functional as F
    vid = grey.new_zeros(grey.shape[1:], dtype=bool)
    vid[PADY:PADY + H] = True
    dark = (grey < 60) & ((bg[None] - grey) > 40)
    bright = (grey - bg[None]) > 40
    out = {}
    for r in rules:
        if r == 'dark':
            c = dark
        elif r == 'dark_bright':
            c = dark | bright
        elif r == 'absdiff':
            c = (grey - bg[None]).abs() > 40
        elif r == 'absdiff_floor':
            c = ((grey - bg[None]).abs() > 40) | (grey < 60)
        else:
            raise KeyError(r)
        c = c & vid[None]
        out[r] = F.avg_pool2d(c.float()[:, None], 32).flatten(1)
    return out


class Masker:
    """per-video mask on the 32 grid: DINOv2-448 token cue + pixel cue (rule), FG_RULE drop-isolated + 1-cell dilation."""

    def __init__(self, enc448, dev):
        self.enc, self.dev = enc448, dev
        self.vc = None

    def video(self, bgframes, rules):
        import torch
        from src.eci.foreground import FG_RULE, cosine_distance, patch_background, video_threshold
        c = canvas_gpu(bgframes, self.dev)
        g = (0.299 * c[:, 0] + 0.587 * c[:, 1] + 0.114 * c[:, 2]).round()
        self.bg = g.median(0).values
        tok = self.enc.run(bgframes, 16)
        cues = pixel_cues(g, self.bg, rules)
        self.state = {}
        vc = torch.from_numpy(video_cells()).to(self.dev)
        for r in rules:
            bgt, _ = patch_background(tok, exclude=cues[r] > FG_RULE['dark_frac'], min_frames=8)
            d = cosine_distance(tok, bgt)
            d[:, ~vc] = 0
            thr = video_threshold(d.cpu().numpy(), cues[r].cpu().numpy(), FG_RULE)
            self.state[r] = (bgt, thr)
        del tok

    def masks(self, frames, tok448, rules):
        """frames: list of t frames, tok448: their (n, 1024, 768) DINOv2-448 tokens -> {rule: (n, 1024) bool GPU}"""
        import torch
        from src.eci.foreground import FG_RULE, cosine_distance, foreground_parts
        c = canvas_gpu(frames, self.dev)
        g = (0.299 * c[:, 0] + 0.587 * c[:, 1] + 0.114 * c[:, 2]).round()
        cues = pixel_cues(g, self.bg, rules)
        vc = torch.from_numpy(video_cells()).to(self.dev)
        out = {}
        for r in rules:
            bgt, thr = self.state[r]
            d = cosine_distance(tok448, bgt)
            d[:, ~vc] = 0
            feat, dk, m = foreground_parts(d, cues[r], thr, FG_RULE)
            out[r] = m & vc[None]
            out[r + '_parts'] = (feat, dk)
        return out


def needed(v, N, dino, vj):
    """video frame indices needed for plan frames v (array) of a video with N frames."""
    s = set(np.linspace(0, N - 1, NBG).round().astype(int).tolist()) | set(v.tolist())
    if dino:
        s |= set((v - LAG).tolist())
    if vj:
        s |= set((v[:, None] + VJS * (np.arange(CLIP)[None] - TPOS)).ravel().tolist())
    return s


def plan_frames(set_):
    P = pd.read_parquet(PLAN / f'{set_}_frames.parquet')
    return P


def shard_videos(P, i, n):
    """videos sorted by plan-frame count, dealt round robin (balanced shards). NOTE: the pass-A shards were cut with an
    unstable sort (tie order differs between CPU types); passes that reuse a mask take the video list from pass A's
    frames parquet instead (cmd_encode), and ties are now broken by name (stable)."""
    c = P.groupby('video').size().reset_index(name='n').sort_values(['n', 'video'], ascending=[False, True], kind='stable')
    return [v for j, v in enumerate(c.video) if j % n == i]


def cmd_encode(args):
    import torch
    dev = device()
    stage = stage_dir()
    TOK.mkdir(parents=True, exist_ok=True)
    t00 = time.time()
    P = plan_frames(args.set)
    split = 'train' if args.set == 'train' else 'test'
    vids = shard_videos(P, args.shard, args.nshard)
    if args.max_per_video:
        P = P.groupby('video', group_keys=False).head(args.max_per_video).reset_index(drop=True)
    if args.smoke:
        vids = vids[:args.smoke]
    base = f'{args.set}_{args.shard}of{args.nshard}' + (f'_smoke{args.smoke}' if args.smoke else '')
    jkey = base + '_' + '_'.join(args.configs)
    if (TOK / f'{jkey}.json').exists():
        log(f'[SKIP] {jkey}')
        return
    off = offset()
    if args.mask == 'file':
        args.mask = (PLAN / 'mask_rule.txt').read_text().split()[0]
        assert args.mask in MASKS, args.mask
    nfr = json.loads((OUT / 'inventory' / 'inventory.json').read_text())['videos'][split]['per_video']
    encs = {c: Enc(c, dev, args.bf16) for c in args.configs}
    dino = any(not e.video for e in encs.values())
    vj = any(e.video for e in encs.values())
    if not args.reuse_mask:
        m448 = encs['d2_448'] if 'd2_448' in encs else Enc('d2_448', dev, args.bf16)
        masker = Masker(m448, dev)
    else:
        RF = pd.read_parquet(TOK / f'{base}_frames.parquet')
        pos_all = np.load(TOK / f'{base}_pos32.npy')
        pst = np.r_[0, np.cumsum(RF.n32.values)]
        vids = list(dict.fromkeys(RF.video.values))  # pass A's own video list and order
    log(f'{jkey}: {len(vids)} videos, {int(P.video.isin(vids).sum())} frames, configs {args.configs}, offset {off}, '
        f'bf16 {args.bf16}, {gpu_name(dev)}')
    pool = ThreadPoolExecutor(args.workers)

    def dec(q):
        t_ = time.time()
        loc = stage / f'{q}.mp4'
        shutil.copyfile(video_path(split, q), loc)
        R = P[P.video == q]
        v = R.i.values + off
        fr, n = decode_frames(loc, needed(v, nfr[q]['frames'], dino, vj), threads=args.threads)
        loc.unlink()
        return fr, time.time() - t_

    fut = {j: pool.submit(dec, vids[j]) for j in range(min(len(vids), args.workers + 1))}
    A = {c: {'t': [], 'p60': [], 'pos': []} for c in args.configs}
    meta, pos32_all = [], []
    tdec = twait = 0.0
    for k, q in enumerate(vids):
        t0 = time.time()
        fr, dt = fut.pop(k).result()
        twait += time.time() - t0
        tdec += dt
        if k + args.workers + 1 < len(vids):
            fut[k + args.workers + 1] = pool.submit(dec, vids[k + args.workers + 1])
        R = P[P.video == q]
        v = R.i.values + off
        N = nfr[q]['frames']
        assert v.min() - max(LAG, VJS * TPOS) >= 0 and v.max() + VJS * (CLIP - 1 - TPOS) <= N - 1, q
        missing = [x for x in needed(v, N, dino, vj) if x not in fr]
        assert not missing, (q, len(missing), missing[:5])
        tf = [fr[x] for x in v]
        if args.reuse_mask:
            rr = np.flatnonzero(RF.video.values == q)
            assert (RF.i.values[rr] == R.i.values).all()
            p32 = [pos_all[pst[r]:pst[r + 1]] for r in rr]
        else:
            bgi = np.linspace(0, N - 1, NBG).round().astype(int)
            masker.video([fr[x] for x in bgi], [args.mask])
            t448 = encs['d2_448'].run(tf, args.bs) if 'd2_448' in encs else m448.run(tf, args.bs)
            mk = torch.cat([masker.masks(tf[a:a + 64], t448[a:a + 64], [args.mask])[args.mask]
                            for a in range(0, len(tf), 64)])
            p32 = [np.flatnonzero(m).astype(np.int16) for m in mk.cpu().numpy()]
        for c, e in encs.items():
            pg = [torch.from_numpy(to_grid(p, e.G).astype(np.int64)).to(dev) for p in p32]
            kinds = [('t', v)] if e.video else [('t', v), ('p60', v - LAG)]
            for kd, vv in kinds:
                if c == 'd2_448' and kd == 't' and not args.reuse_mask:
                    A[c][kd] += [t448[j, pg[j]].cpu().numpy() for j in range(len(vv))]
                elif e.video:
                    A[c][kd] += e.run([[fr[x + VJS * (j - TPOS)] for j in range(CLIP)] for x in vv], args.vj_bs, pg)
                else:
                    A[c][kd] += e.run([fr[x] for x in vv], args.bs if e.G == 32 else max(1, args.bs // 4), pg)
            A[c]['pos'] += [p.cpu().numpy().astype(np.int16) for p in pg]
        for j, r in enumerate(R.itertuples()):
            meta.append({'video': q, 'i': int(r.i), 'v': int(r.i + off), 'n32': int(len(p32[j]))})
        pos32_all += p32
        del fr, tf
        el = time.time() - t00
        log(f'  video {k + 1}/{len(vids)} {q}: {len(v)} frames, kept {np.mean([len(p) for p in p32]):.1f} cells; '
            f'{len(meta) / el:.2f} frames/s wall; gpu ' + ', '.join(f'{c} {e.gpu_s:.0f}s' for c, e in encs.items())
            + f'; decode {tdec:.0f}s summed, waited {twait:.0f}s')
    pool.shutdown()
    Fm = pd.DataFrame(meta)
    saved = {}
    for c, d in A.items():
        for kd, lst in d.items():
            if not lst:
                continue
            a = np.concatenate(lst)
            nm = f'{base}_{c}_{kd}.npy'
            np.save(stage / nm, a)
            shutil.copyfile(stage / nm, TOK / nm)
            (stage / nm).unlink()
            saved[nm] = list(a.shape)
            log(f'  saved {nm} {a.shape}')
    if not args.reuse_mask:
        Fm.to_parquet(TOK / f'{base}_frames.parquet')
        np.save(TOK / f'{base}_pos32.npy', np.concatenate(pos32_all))
    wall = time.time() - t00
    info = {'set': args.set, 'shard': args.shard, 'nshard': args.nshard, 'configs': args.configs, 'videos': len(vids),
            'frames': len(Fm), 'offset': off, 'mask': args.mask if not args.reuse_mask else 'reused pos32',
            'bf16': args.bf16, 'gpu': gpu_name(dev), 'wall_s': round(wall, 1),
            'frames_per_wall_s': round(len(Fm) / wall, 3), 'decode_s_summed': round(tdec, 1),
            'decode_wait_s': round(twait, 1), 'workers': args.workers, 'kept32_mean': float(Fm.n32.mean()),
            'frames_with_0_kept': int((Fm.n32 == 0).sum()), 'saved': saved,
            'per_config': {c: {'grid': e.G, 'size': e.S, 'gpu_s': round(e.gpu_s, 1), 'inputs': e.n_in,
                               'inputs_per_gpu_s': round(e.n_in / max(e.gpu_s, 1e-9), 2),
                               'tokens_per_gpu_s': round(e.n_tok / max(e.gpu_s, 1e-9), 1)} for c, e in encs.items()},
            'peak_mem_gb': round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if dev.type == 'cuda' else None}
    (TOK / f'{jkey}.json').write_text(json.dumps(info, indent=1))
    (OUT / 'encode_json').mkdir(parents=True, exist_ok=True)
    (OUT / 'encode_json' / f'{jkey}.json').write_text(json.dumps(info, indent=1))
    log('done: ' + json.dumps(info))


def cmd_maskqa(args):
    """all MASKS rules on 24 plan frames of each --videos video: keypoint recall, kept cells, sheets."""
    import torch
    from PIL import Image, ImageDraw
    dev = device()
    stage = stage_dir()
    off = offset()
    rules = list(MASKS)
    enc = Enc('d2_448', dev, args.bf16)
    masker = Masker(enc, dev)
    res, rows = {r: [] for r in rules}, []
    od = OUT / 'mask'
    od.mkdir(parents=True, exist_ok=True)
    for spec in args.videos:
        set_, q = spec.split(':')
        set_ = {'test': 'eval'}.get(set_, set_)
        split = 'train' if set_ == 'train' else 'test'
        P = plan_frames(set_)
        R = P[P.video == q]
        R = R.iloc[np.linspace(0, len(R) - 1, min(24, len(R))).round().astype(int)]
        Z = np.load(DER / f'task1_{split}.npz')
        names = list(Z['names'])
        st = np.r_[0, np.cumsum(Z['lens'])][names.index(q)]
        kp, sc = Z['keypoints'][st + R.i.values], Z['scores'][st + R.i.values]
        nfr = json.loads((OUT / 'inventory' / 'inventory.json').read_text())['videos'][split]['per_video'][q]['frames']
        loc = stage / f'{q}.mp4'
        shutil.copyfile(video_path(split, q), loc)
        v = R.i.values + off
        bgi = np.linspace(0, nfr - 1, NBG).round().astype(int)
        fr, _ = decode_frames(loc, set(bgi.tolist()) | set(v.tolist()), threads=8)
        loc.unlink()
        masker.video([fr[x] for x in bgi], rules)
        tf = [fr[x] for x in v]
        tok = enc.run(tf, 16)
        mk = masker.masks(tf, tok, rules)
        # keypoints -> 32-grid cells of the canvas
        cx = np.clip((kp[:, :, 0, :] // 32).astype(int), 0, 31)
        cy = np.clip(((kp[:, :, 1, :] + PADY) // 32).astype(int), 0, 31)
        cell = cy * 32 + cx  # (n, 2, 7)
        for r in rules:
            m = mk[r].cpu().numpy()
            for j in range(len(v)):
                hit = m[j][cell[j]]  # (2, 7)
                good = sc[j] > 0.5
                res[r].append({'video': q, 'set': set_, 'kept': int(m[j].sum()),
                               'recall_res': float(hit[0][good[0]].mean()) if good[0].any() else np.nan,
                               'recall_int': float(hit[1][good[1]].mean()) if good[1].any() else np.nan,
                               'miss_res': bool(not hit[0][good[0]].any()) if good[0].any() else False,
                               'miss_int': bool(not hit[1][good[1]].any()) if good[1].any() else False})
        # sheet for the rule under test: 8 frames of this video
        for r in [args.mask]:
            tiles = []
            m = mk[r].cpu().numpy()
            feat, dk = (x.cpu().numpy() for x in mk[r + '_parts'])
            for j in np.linspace(0, len(v) - 1, 8).round().astype(int):
                im = canvas_gpu([tf[j]], dev)[0].permute(1, 2, 0).cpu().numpy()
                M = np.kron(m[j].reshape(32, 32), np.ones((32, 32))).astype(bool)
                im[M, 0] = im[M, 0] * 0.55 + 115
                im[M, 1:] *= 0.65
                pil = Image.fromarray(im[PADY - 16:PADY + H + 16].clip(0, 255).astype(np.uint8))
                d = ImageDraw.Draw(pil)
                for mm, col in ((0, (0, 255, 0)), (1, (0, 200, 255))):
                    for p_ in range(7):
                        x, y = kp[j, mm, 0, p_], kp[j, mm, 1, p_] + 16
                        d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=col)
                d.text((6, 4), f'{q} v{v[j]} kept {int(m[j].sum())} tok-cue {int(feat[j].sum())} pix-cue '
                               f'{int(dk[j].sum())} ({r})', fill=(255, 255, 0))
                tiles.append(pil.resize((512, (H + 32) // 2)))
            Wt = Image.new('RGB', (2 * 512, 4 * ((H + 32) // 2)))
            for k, t in enumerate(tiles):
                Wt.paste(t, ((k % 2) * 512, (k // 2) * ((H + 32) // 2)))
            Wt.save(od / f'sheet_{r}_{q}.jpg', quality=85)
        log(f'{q}: ' + ', '.join(f"{r} kept {np.mean([x['kept'] for x in res[r][-len(v):]]):.1f} rec res "
                                 f"{np.nanmean([x['recall_res'] for x in res[r][-len(v):]]):.3f} int "
                                 f"{np.nanmean([x['recall_int'] for x in res[r][-len(v):]]):.3f}" for r in rules))
    summ = {}
    for r in rules:
        D = pd.DataFrame(res[r])
        summ[r] = {'desc': MASKS[r], 'frames': len(D), 'kept_mean': float(D.kept.mean()),
                   'kept_frac_of_video_cells': float(D.kept.mean() / video_cells().sum()),
                   'kp_recall_resident': float(D.recall_res.mean()), 'kp_recall_intruder': float(D.recall_int.mean()),
                   'frac_frames_missing_resident': float(D.miss_res.mean()),
                   'frac_frames_missing_intruder': float(D.miss_int.mean()),
                   'per_video': D.groupby('video')[['kept', 'recall_res', 'recall_int']].mean().round(3).to_dict('index')}
        log(f'{r}: ' + json.dumps({k: v for k, v in summ[r].items() if k != 'per_video'}))
    (od / f'maskqa{args.tag and "_" + args.tag}.json').write_text(json.dumps(summ, indent=1))


def cmd_bench(args):
    import torch
    dev = device()
    res = []
    for c in args.configs:
        e = Enc(c, dev, args.bf16)
        n = 1 if e.video or e.G >= 64 else 8
        fr = [np.random.default_rng(0).integers(0, 255, (H, W, 3), dtype=np.uint8)] * (n * (CLIP if e.video else 1))
        inp = [fr[k * CLIP:(k + 1) * CLIP] for k in range(n)] if e.video else fr
        try:
            torch.cuda.reset_peak_memory_stats()
            e.run(inp, n)
            e.gpu_s, e.n_in, e.n_tok = 0.0, 0, 0
            for _ in range(3):
                e.run(inp, n)
            r = {'config': c, 'grid': e.G, 'size': e.S, 'batch': n, 'inputs_per_s': round(e.n_in / e.gpu_s, 3),
                 'tokens_per_s': round(e.n_tok / e.gpu_s, 1),
                 'peak_mem_gb': round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
                 'precision': 'bf16 autocast' if args.bf16 else 'fp32', 'gpu': gpu_name(dev),
                 'note': 'includes the canvas pad + GPU resize from the 1024 canvas'}
        except torch.cuda.OutOfMemoryError as ex:
            r = {'config': c, 'error': 'OOM ' + str(ex)[:120]}
        log(json.dumps(r))
        res.append(r)
        del e
        torch.cuda.empty_cache()
    (OUT / 'bench').mkdir(parents=True, exist_ok=True)
    (OUT / 'bench' / f'bench{args.tag and "_" + args.tag}.json').write_text(json.dumps(res, indent=1))


def cmd_prec(args):
    """bf16 autocast vs fp32 tokens (cosine) on real canvases of 4 plan frames per config."""
    import torch
    dev = device()
    stage = stage_dir()
    off = offset()
    P = plan_frames('eval')
    q = P.video.iloc[0]
    v = P[P.video == q].i.values[:4] + off
    loc = stage / f'{q}.mp4'
    shutil.copyfile(video_path('test', q), loc)
    want = set(v.tolist()) | set((v[:, None] + VJS * (np.arange(CLIP)[None] - TPOS)).ravel().tolist())
    fr, _ = decode_frames(loc, want, threads=8)
    loc.unlink()
    res = {}
    for c in args.configs:
        out = {}
        for bf in (False, True):
            e = Enc(c, dev, bf)
            inp = [[fr[x + VJS * (j - TPOS)] for j in range(CLIP)] for x in v] if e.video else [fr[x] for x in v]
            out[bf] = e.run(inp, 1)
            del e
            torch.cuda.empty_cache()
        cs = torch.nn.functional.cosine_similarity(out[False], out[True], dim=-1).flatten().cpu().numpy()
        res[c] = {'cos_mean': float(cs.mean()), 'cos_p1': float(np.percentile(cs, 1)), 'cos_min': float(cs.min()),
                  'n_tokens': int(cs.size), 'note': 'all tokens of the canvas (incl. padding), 4 frames'}
        log(f'{c}: {res[c]}')
    (OUT / 'bench').mkdir(parents=True, exist_ok=True)
    (OUT / 'bench' / f'prec{args.tag and "_" + args.tag}.json').write_text(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['encode', 'maskqa', 'bench', 'prec'], default='encode')
    ap.add_argument('--set', choices=['train', 'eval'], default='eval')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, nargs='*', default=[])
    ap.add_argument('--nshard', type=int, default=8)
    ap.add_argument('--configs', nargs='+', default=['d2_448'])
    ap.add_argument('--mask', default='absdiff')
    ap.add_argument('--reuse-mask', action='store_true')
    ap.add_argument('--bf16', action='store_true')
    ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--vj-bs', type=int, default=2)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--threads', type=int, default=3)
    ap.add_argument('--smoke', type=int, default=0)
    ap.add_argument('--max-per-video', type=int, default=0, help='smoke only: first N plan frames of each video')
    ap.add_argument('--videos', nargs='*', default=[])
    ap.add_argument('--tag', default='')
    args = ap.parse_args()
    if args.mode == 'encode' and args.shards:  # several shards in one job (fewer, longer jobs)
        for sh in args.shards:
            args.shard = sh
            cmd_encode(args)
        return
    {'encode': cmd_encode, 'maskqa': cmd_maskqa, 'bench': cmd_bench, 'prec': cmd_prec}[args.mode](args)


if __name__ == '__main__':
    main()
