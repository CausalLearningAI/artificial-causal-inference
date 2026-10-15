"""
MABe22 mouse ECI - ENCODER STUDY (rules: eci_encoders.sh): masked patch tokens of other frozen encoders on the SAME
frames and mask positions as round 3 (GPU Slurm job, sharded exactly as the round-3 store).

Frames / mask (READ-ONLY from the round-3 store results/vision/benchmark_mabe22/eci/tok_r3/): <set>_<i>of<n>_frames
.parquet (clip, i, n448) and _pos448.npy (kept flat positions of the 32 x 32 grid of the 448-px DINOv2 input, mask
'arena', computed at frame t). Video frame of label i = i + offset (plan/offset.json); lag frame = max(t - L, 0).
The 224 frame is bicubic-upscaled to S px (torch, 0-255 scale, rounded / clipped, as R), ImageNet mean / std.
Mask on the encoder grid G = S / patch:
    G == 32  identical positions (every encoder grid here at 448 / 512 px tiles the 224 frame into the same 7 px cells)
    G <  32  (G | 32) a G-patch is kept iff ANY of its (32/G)^2 child 448 patches is kept (the round-1 A5 224 rule)
    G >  32  patch-centre rule of eci_r3_encode.transfer_mask
Encoders (fp32; TF32 off for the image encoders as R, on for V-JEPA as the lab pilot; SDPA attention):
    dinov2b  facebook/dinov2-base, patch 14, 1 prefix token (CLS)
    dinov3b  facebook/dinov3-vitb16-pretrain-lvd1689m, patch 16, 1 CLS + 4 register tokens dropped; RoPE on [-1, 1]
             normalised patch-centre coordinates (no augmentation in eval mode), so any S is an interpolation
    dinov3l  facebook/dinov3-vitl16-pretrain-lvd1689m, as dinov3b (1024-d)
    vjepal   facebook/vjepa2-vitl-fpc64-256 (encoder only), tubelet 2 x 16 x 16, 3D RoPE on INTEGER grid positions.
             config.crop_size = S so the RoPE rows are G wide. --vj-pos extrap: positions 0..G-1 (beyond the 0..15 of
             the 256-px training grid when S > 256); interp: height / width positions scaled by 16 / G (fractional,
             inside the trained range). Clip: 16 frames at stride --vj-stride (8 video frames = 3.75 fps, the closest
             integer stride to V-JEPA 2's 4 fps pretraining), frames v + stride (j - 7), j = 0..15, clamped to the clip
             [0, 1800]; the token of frame v = tubelet 3 = clip positions (6, 7) = frames (v - stride, v) (the lab
             pilot's convention). One clip per (target frame, lag).
Layers: --layers l (1-based block index; the LAST block = the model's documented output). Every saved layer is the
block output passed through the model's FINAL LayerNorm (= last_hidden_state for the last block; for intermediate
blocks the convention of DINOv2 / DINOv3 get_intermediate_layers(norm=True) and V-JEPA's out_layers).
Outputs (token stores on BeeGFS, MABE_ENC_TOK, default /mnt/beegfs/locatgrp/shared/rcadei/eci_tok/encoders/ = ETOK):
ETOK/<set>_<i>of<n>_<tag>_L<l>_<t|p60>.npy (fp16), <set>_<i>of<n>_<tag>_pos.npy (int16, G grid),
<set>_<i>of<n>_<tag>_frames.parquet (clip, i, n), <set>_<i>of<n>_<tag>_<kinds>.json (timing, throughput; also copied to
OUT/encoders/encode_json/). Decoding: --workers ffmpeg threads with a bounded look-ahead; each clip decoded once for
all lags of the job. --bf16: bf16 autocast (mode prec measures its token agreement with fp32 first).
--check-r3: for dinov2b at 448, last layer, compares the t tokens with the round-3 store's m448_t (max abs diff).
Mode bench: throughput only (random input, no data): tokens / s per encoder and size -> OUT/encoders/bench_<gpu>.json
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from eci_common import OUT, PLAN, decode_clip, log, offset, stage_dir  # noqa: E402

R3 = OUT / 'tok_r3'
EOUT = OUT / 'encoders'
# token stores on BeeGFS (regenerable data only; NFS is near full): a few large .npy shards per (set, shard, tag, kind)
ETOK = Path(os.environ.get('MABE_ENC_TOK') or '/mnt/beegfs/locatgrp/shared/rcadei/eci_tok/encoders')
SETS = {'train': ('mouse_userTrain', 'train_frames.parquet'), 'eval': ('mouse_submission', 'eval_frames.parquet')}
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
MODELS = {'dinov2b': ('facebook/dinov2-base', 14, 1, 12), 'dinov3b': ('facebook/dinov3-vitb16-pretrain-lvd1689m', 16, 5, 12),
          'dinov3l': ('facebook/dinov3-vitl16-pretrain-lvd1689m', 16, 5, 24),
          'vjepal': ('facebook/vjepa2-vitl-fpc64-256', 16, 0, 24)}
CLIP, TPOS = 16, 7
TUB = TPOS // 2


def map_mask(pos448, G):
    m = np.zeros((32, 32), bool)
    m.flat[pos448.astype(np.int64)] = True
    if G == 32:
        return pos448.astype(np.int16)
    if G < 32:
        assert 32 % G == 0
        f = 32 // G
        return np.flatnonzero(m.reshape(G, f, G, f).any((1, 3))).astype(np.int16)
    idx = np.floor((np.arange(G) + 0.5) * 32 / G).astype(int)
    return np.flatnonzero(m[np.ix_(idx, idx)]).astype(np.int16)


class Encoder:
    def __init__(self, name, S, layers, dev, vj_pos='extrap', vj_stride=8, bf16=False):
        import torch
        from transformers import AutoConfig, AutoModel
        self.torch, self.dev, self.name, self.S, self.bf16 = torch, dev, name, S, bf16
        mid, self.ps, self.npre, self.nblk = MODELS[name]
        self.G = S // self.ps
        assert self.G * self.ps == S, (name, S)
        self.video = name.startswith('vjepa')
        torch.backends.cuda.matmul.allow_tf32 = self.video
        torch.backends.cudnn.allow_tf32 = self.video
        if self.video:
            cfg = AutoConfig.from_pretrained(mid)
            cfg.crop_size = S
            self.model = AutoModel.from_pretrained(mid, config=cfg, attn_implementation='sdpa').to(dev).eval()
            blocks, self.norm = self.model.encoder.layer, self.model.encoder.layernorm
            self.vj_stride = vj_stride
            if vj_pos == 'interp' and self.G != 16:
                sc = 16.0 / self.G
                for blk in blocks:
                    att = blk.attention
                    orig = att.get_position_ids

                    def gp(x, masks=None, _o=orig, _s=sc):
                        f, h, w = _o(x, masks)
                        return f, h.float() * _s, w.float() * _s
                    att.get_position_ids = gp
            self.vj_pos = vj_pos
        else:
            self.model = AutoModel.from_pretrained(mid, attn_implementation='sdpa').to(dev).eval()
            if name == 'dinov2b':
                blocks, self.norm = self.model.encoder.layer, self.model.layernorm
            else:
                blocks, self.norm = self.model.layer, self.model.norm
        self.model.requires_grad_(False)
        assert len(blocks) == self.nblk
        self.layers = list(layers)
        assert all(1 <= l_ <= self.nblk for l_ in self.layers)
        self.cap = {}
        for l_ in self.layers:
            blocks[l_ - 1].register_forward_hook(
                lambda m, i, o, _l=l_: self.cap.__setitem__(_l, o[0] if isinstance(o, tuple) else o))
        self.mean = torch.tensor(MEAN, device=dev)[None, :, None, None]
        self.std = torch.tensor(STD, device=dev)[None, :, None, None]
        self.n_img, self.n_tok, self.gpu_s = 0, 0, 0.0
        self.dim = self.model.config.hidden_size

    def prep(self, frames):
        """(n, 224, 224, 3) uint8 numpy -> (n, 3, S, S) normalised fp32 GPU."""
        F = self.torch.nn.functional
        x = self.torch.from_numpy(np.ascontiguousarray(frames)).to(self.dev).permute(0, 3, 1, 2).float()
        if self.S != x.shape[-1]:
            x = F.interpolate(x, size=(self.S, self.S), mode='bicubic', align_corners=False).round().clamp(0, 255)
        return (x / 255 - self.mean) / self.std

    def forward(self, x):
        """x: (B, 3, S, S) image or (B, 16, 3, S, S) clip -> {layer: (B, G*G, d) fp32 normed tokens}, check dict."""
        torch = self.torch
        self.cap.clear()
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.bf16):
            if self.video:
                out = self.model(pixel_values_videos=x, skip_predictor=True).last_hidden_state
            else:
                out = self.model(pixel_values=x).last_hidden_state
            hs = {l_: self.norm(self.cap[l_]).float() for l_ in self.layers}
        GG = self.G * self.G
        res = {}
        for l_ in self.layers:
            h = hs[l_]
            if l_ == self.nblk:
                assert torch.allclose(h, out.float(), atol=1e-3), 'last-layer hook != last_hidden_state'
            if self.video:
                assert h.shape[1] == (CLIP // 2) * GG, h.shape
                h = h[:, TUB * GG:(TUB + 1) * GG]
            else:
                assert h.shape[1] == self.npre + GG, h.shape
                h = h[:, self.npre:]
            assert torch.isfinite(h).all()
            res[l_] = h.float()
        return res

    def encode(self, frames, vidx, bs):
        """frames: decoded clip (T, 224, 224, 3); vidx: target video frame indices -> {layer: (n, G*G, d) fp32 GPU}."""
        torch = self.torch
        T = len(frames)
        outs = {l_: [] for l_ in self.layers}
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.inference_mode():
            for a in range(0, len(vidx), bs):
                v = np.asarray(vidx[a:a + bs])
                if self.video:
                    idx = np.clip(v[:, None] + self.vj_stride * (np.arange(CLIP)[None] - TPOS), 0, T - 1)
                    x = self.prep(frames[idx.ravel()]).reshape(len(v), CLIP, 3, self.S, self.S)
                    ntok = len(v) * (CLIP // 2) * self.G ** 2
                else:
                    x = self.prep(frames[v])
                    ntok = len(v) * (self.npre + self.G ** 2)
                r = self.forward(x)
                for l_ in self.layers:
                    outs[l_].append(r[l_])
                self.n_img += len(v)
                self.n_tok += ntok
        torch.cuda.synchronize()
        self.gpu_s += time.time() - t0
        return {l_: torch.cat(v) for l_, v in outs.items()}


def plan(set_):
    P = pd.read_parquet(PLAN / SETS[set_][1])
    assert P.equals(P.sort_values(['clip', 'i']).reset_index(drop=True))
    return P


def cmd_encode(args):
    import torch
    dev = torch.device('cuda')
    enc = None
    for sh in args.shard:
        t0 = time.time()
        enc = encode_shard(args, sh, enc, dev)
        log(f'shard {sh} done in {time.time() - t0:.0f}s')


def encode_shard(args, shard, enc, dev):
    import torch
    t00 = time.time()
    stage = stage_dir()
    ETOK.mkdir(parents=True, exist_ok=True)
    base = f'{args.set}_{shard}of{args.nshard}'
    key = f'{base}_{args.tag}' + (f'_smoke{args.smoke}' if args.smoke else '')
    jkey = key + '_' + '_'.join(args.kinds)
    if (ETOK / f'{jkey}.json').exists():
        log(f'[SKIP] {jkey}')
        return enc
    RF = pd.read_parquet(R3 / f'{base}_frames.parquet')
    pos_all = np.load(R3 / f'{base}_pos448.npy')
    st = np.r_[0, np.cumsum(RF.n448.values)]
    assert st[-1] == len(pos_all)
    P = plan(args.set)
    clips = list(dict.fromkeys(RF['clip'].values))
    assert set(clips) == set(P['clip'][P['clip'].isin(clips)]) and len(RF) == int(P['clip'].isin(clips).sum())
    if args.smoke:
        clips = clips[:args.smoke]
    off = offset()
    if enc is None:
        enc = Encoder(args.encoder, args.size, args.layers, dev, args.vj_pos, args.vj_stride, args.bf16)
    enc.n_img, enc.n_tok, enc.gpu_s = 0, 0, 0.0
    log(f'{jkey}: {len(clips)} clips, {args.encoder} S {args.size} grid {enc.G} layers {args.layers} kinds {args.kinds} '
        f'vj_pos {args.vj_pos} stride {args.vj_stride} offset {off} {torch.cuda.get_device_name(0)}')
    A, meta, tdec = {}, [], 0.0
    chk = []
    r3t = None
    if args.check_r3:
        r3t = np.load(R3 / f'{base}_m448_t.npy', mmap_mode='r')
    pool = ThreadPoolExecutor(args.workers)  # parallel ffmpeg decodes, bounded look-ahead (each clip ~270 MB)

    def dec(c_):
        t_ = time.time()
        return decode_clip(SETS[args.set][0], c_, stage, 'rgb24'), time.time() - t_
    fut = {j: pool.submit(dec, clips[j]) for j in range(min(len(clips), 2 * args.workers))}
    twait = 0.0
    for ci, c in enumerate(clips):
        rr = np.flatnonzero(RF['clip'].values == c)
        R = P[P['clip'] == c]
        assert (RF.i.values[rr] == R.i.values).all()
        t0 = time.time()
        fr, dt_ = fut.pop(ci).result()
        twait += time.time() - t0
        tdec += dt_
        nj = ci + 2 * args.workers
        if nj < len(clips):
            fut[nj] = pool.submit(dec, clips[nj])
        if len(fr) != 1801:
            raise RuntimeError(f'{c}: {len(fr)} frames')
        ts = R.i.values + off
        pg = [map_mask(pos_all[st[r_]:st[r_ + 1]], enc.G) for r_ in rr]
        for kd in args.kinds:
            L_ = 0 if kd == 't' else int(kd[1:])
            v = np.maximum(ts - L_, 0)
            out = enc.encode(fr, v, args.bs)
            GG = enc.G * enc.G
            flat = torch.from_numpy(np.concatenate([q * GG + pg[q].astype(np.int64) for q in range(len(rr))])).to(dev)
            for l_, h in out.items():  # one gather + one device-to-host copy per clip and layer
                A.setdefault(f'L{l_}_{kd}', []).append(h.reshape(-1, h.shape[-1])[flat].half().cpu().numpy())
            if r3t is not None and kd == 't' and enc.nblk in out:
                h = out[enc.nblk]
                for q, r_ in enumerate(rr):
                    ref = np.asarray(r3t[st[r_]:st[r_ + 1]], np.float32)
                    mine = h[q, torch.from_numpy(pg[q].astype(np.int64)).to(dev)].half().float().cpu().numpy()
                    chk.append(float(np.abs(ref - mine).max()) if len(ref) else 0.0)
            del out
        for q, r_ in enumerate(rr):
            meta.append({'clip': c, 'i': int(RF.i.values[r_]), 'n': int(len(pg[q]))})
            A.setdefault('pos', []).append(pg[q])
        del fr
        if ci % 25 == 0:
            el = time.time() - t00
            log(f'  clip {ci}/{len(clips)}: {enc.n_img} inputs ({enc.n_img / max(el, 1e-9):.1f} inputs/s wall), '
                f'gpu {enc.gpu_s:.0f}s, decode {tdec:.0f}s (summed over workers), waited for decode {twait:.0f}s')
    pool.shutdown()
    Fm = pd.DataFrame(meta)
    for k, v in A.items():
        if k == 'pos' and 't' not in args.kinds and (ETOK / f'{key}_pos.npy').exists():
            continue
        a = np.concatenate(v)
        nm = f'{key}_{k}.npy'
        np.save(stage / nm, a)
        shutil.copyfile(stage / nm, ETOK / nm)
        (stage / nm).unlink()
        log(f'  saved {nm} {a.shape}')
    Fm.to_parquet(ETOK / f'{key}_frames.parquet')
    info = {'set': args.set, 'shard': shard, 'nshard': args.nshard, 'encoder': args.encoder,
            'model': MODELS[args.encoder][0], 'size': args.size, 'grid': enc.G, 'layers': args.layers, 'kinds': args.kinds,
            'vj_pos': args.vj_pos if enc.video else None, 'vj_stride': args.vj_stride if enc.video else None,
            'clips': len(clips), 'frames': len(Fm), 'offset': off, 'inputs_encoded': enc.n_img,
            'tokens_processed': enc.n_tok, 'gpu_s': round(enc.gpu_s, 1), 'decode_s_summed': round(tdec, 1),
            'decode_wait_s': round(twait, 1), 'decode_workers': args.workers, 'bf16': args.bf16,
            'frames_per_wall_s': round(len(Fm) / max(time.time() - t00, 1e-9), 2),
            'inputs_per_gpu_s': round(enc.n_img / max(enc.gpu_s, 1e-9), 2),
            'tokens_per_gpu_s': round(enc.n_tok / max(enc.gpu_s, 1e-9), 1), 'gpu': torch.cuda.get_device_name(0),
            'wall_s': round(time.time() - t00, 1), 'kept_mean': float(Fm.n.mean()),
            'frames_with_0_kept': int((Fm.n == 0).sum()),
            'check_r3_maxabs': (max(chk) if chk else None)}
    (ETOK / f'{jkey}.json').write_text(json.dumps(info, indent=1, default=float))
    (EOUT / 'encode_json').mkdir(parents=True, exist_ok=True)
    (EOUT / 'encode_json' / f'{jkey}.json').write_text(json.dumps(info, indent=1, default=float))
    log('done: ' + json.dumps(info, default=float))
    return enc


def cmd_bench(args):
    """tokens / s on random inputs, per encoder x size (batch tuned to a fixed token budget), 3 timed iterations."""
    import torch
    dev = torch.device('cuda')
    res = []
    for spec in args.bench:
        name, S, *pr = spec.split(':')
        S = int(S)
        bf = bool(pr and pr[0] == 'bf16')
        enc = Encoder(name, S, [MODELS[name][3]], dev, 'extrap', 8, bf)
        tok_per_input = (CLIP // 2) * enc.G ** 2 if enc.video else enc.npre + enc.G ** 2
        bs = max(1, int(16384 // tok_per_input))
        shape = (bs, CLIP, 3, S, S) if enc.video else (bs, 3, S, S)
        x = torch.randn(shape, device=dev)
        try:
            with torch.inference_mode():
                enc.forward(x)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                t0 = time.time()
                for _ in range(3):
                    enc.forward(x)
                torch.cuda.synchronize()
                dt = (time.time() - t0) / 3
            r = {'encoder': name, 'size': S, 'grid': enc.G, 'batch': bs, 'tokens_per_input': tok_per_input,
                 'inputs_per_s': round(bs / dt, 3), 'tokens_per_s': round(bs * tok_per_input / dt, 1),
                 'peak_mem_gb': round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
                 'precision': 'bf16 autocast' if bf else ('fp32 TF32 on' if enc.video else 'fp32 TF32 off'),
                 'attention': 'sdpa', 'gpu': torch.cuda.get_device_name(0)}
        except torch.cuda.OutOfMemoryError as e:
            r = {'encoder': name, 'size': S, 'precision': 'bf16' if bf else 'fp32', 'error': 'OOM ' + str(e)[:100]}
        log(json.dumps(r))
        res.append(r)
        del enc, x
        torch.cuda.empty_cache()
    g = torch.cuda.get_device_name(0).replace(' ', '_')
    EOUT.mkdir(parents=True, exist_ok=True)
    (EOUT / f'bench_{g}{args.tag and "_" + args.tag}.json').write_text(json.dumps(res, indent=1))


def cmd_prec(args):
    """bf16-autocast vs fp32 tokens on the first --smoke clips of eval shard 0 (masked t tokens, every layer):
    cosine mean / 1st percentile / min -> OUT/encoders/prec_<encoder>_<tag>.json"""
    import torch
    dev = torch.device('cuda')
    stage = stage_dir()
    base = 'eval_0of12'
    RF = pd.read_parquet(R3 / f'{base}_frames.parquet')
    pos_all = np.load(R3 / f'{base}_pos448.npy')
    st = np.r_[0, np.cumsum(RF.n448.values)]
    P = plan('eval')
    clips = list(dict.fromkeys(RF['clip'].values))[:args.smoke or 8]
    off = offset()
    encs = {b: Encoder(args.encoder, args.size, args.layers, dev, args.vj_pos, args.vj_stride, b) for b in (False, True)}
    cos = {l_: [] for l_ in args.layers}
    gs = {False: 0.0, True: 0.0}
    for c in clips:
        rr = np.flatnonzero(RF['clip'].values == c)
        fr = decode_clip('mouse_submission', c, stage, 'rgb24')
        ts = P[P['clip'] == c].i.values + off
        out = {}
        for b, e in encs.items():
            g0 = e.gpu_s
            out[b] = e.encode(fr, ts, args.bs)
            gs[b] += e.gpu_s - g0
        for q, r_ in enumerate(rr):
            p = torch.from_numpy(map_mask(pos_all[st[r_]:st[r_ + 1]], encs[False].G).astype(np.int64)).to(dev)
            for l_ in args.layers:
                a, b = out[False][l_][q, p], out[True][l_][q, p]
                cos[l_].append(torch.nn.functional.cosine_similarity(a, b, dim=1).cpu().numpy())
    res = {'encoder': args.encoder, 'size': args.size, 'tag': args.tag, 'clips': len(clips),
           'gpu_s_fp32': round(gs[False], 2), 'gpu_s_bf16': round(gs[True], 2), 'layers': {}}
    for l_, v in cos.items():
        v = np.concatenate(v)
        res['layers'][l_] = {'n_tokens': int(len(v)), 'cos_mean': float(v.mean()), 'cos_p1': float(np.percentile(v, 1)),
                             'cos_min': float(v.min())}
    log(json.dumps(res))
    (EOUT / f'prec_{args.encoder}_{args.tag}.json').write_text(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['encode', 'bench', 'prec'], default='encode')
    ap.add_argument('--bf16', action='store_true')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--set', choices=list(SETS), default='eval')
    ap.add_argument('--shard', type=int, nargs='+', default=[0])
    ap.add_argument('--nshard', type=int, default=12)
    ap.add_argument('--encoder', choices=list(MODELS), default='dinov3b')
    ap.add_argument('--size', type=int, default=512)
    ap.add_argument('--layers', nargs='+', type=int, default=[12])
    ap.add_argument('--kinds', nargs='+', default=['t'], help="'t' and / or 'p<L>' (lag L frames)")
    ap.add_argument('--tag', default='')
    ap.add_argument('--vj-pos', choices=['extrap', 'interp'], default='extrap')
    ap.add_argument('--vj-stride', type=int, default=8)
    ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--smoke', type=int, default=0)
    ap.add_argument('--check-r3', action='store_true')
    ap.add_argument('--bench', nargs='*', default=[])
    args = ap.parse_args()
    if args.mode == 'bench':
        return cmd_bench(args)
    if args.mode == 'prec':
        return cmd_prec(args)
    assert args.tag
    cmd_encode(args)


if __name__ == '__main__':
    main()
