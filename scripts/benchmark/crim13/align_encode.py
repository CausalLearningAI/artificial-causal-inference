"""
CRIM13 alignment test - token stores (rules: eci_align.sh). GPU Slurm job, one HALF x shard of its videos.

Per video (label-free; the plan's frames of this half, all roles):
    read     seek offsets from <name>_t-seek.mat; only the needed JPEG records are read (offset order, neighbouring
             records coalesced into one sequential read), decoded in threads; videos are read in parallel threads with a
             bounded look-ahead, overlapping the GPU. Needed = 32 background frames evenly spaced over the video, every
             plan frame t and its lag t - 50 (2 s at 25 fps): decoded ONCE for all encoders.
    canvas   pad to a W x W square in the ImageNet mean colour, 656 canvases resized to 640 (align_common.canvas_gpu);
             then resized on the GPU to the encoder input S (bicubic, antialias when downscaling), ImageNet mean / std.
    mask     CalMS21 'dark' rule on the 40 x 40 grid of the 640 canvas (16 px cells): token cue (DINOv2-B/14 at 560 px,
             last layer: cosine distance to the video's per-position background token of the 32 background frames >
             the FG_RULE video threshold, isolated cells dropped) OR pixel cue (> 15% of the cell's pixels have grey < 60
             and > 40 below the per-pixel median background) -> 1-cell dilation; pure-padding cells never kept.
    tokens   DINOv3-L/16 640 px block 20 (d3l_640_L20) and DINOv2-L/14 560 px block 20 (d2l_560_L20), after the final
             LayerNorm, prefix (CLS + registers) dropped, bf16 autocast + SDPA; blocks after 20 are removed (checked
             equal to the hooked full model in --mode check). Kept tokens of t and of t - 50 at t's mask positions.
    recall   per plan frame: is the track point of the resident (track row 1) / intruder (row 2) inside a kept cell
             (and within the 3 x 3 neighbourhood of one) - the mask check, tracks are never used to build the mask.
Outputs (TOK): <half>_<i>of<n>_frames.parquet (video, v, role, n40, recall columns), _pos40.npy (int16),
    _<cfg>_t.npy / _<cfg>_p50.npy (fp16), <base>.json (timing, throughput); copy of the JSON in OUT/encode_json/.
"""
import argparse
import json
import shutil
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent / 'mabe22'))
from align_common import (CANVAS, GRID, LAG, OUT, PLAN, ROOT, TOK, canvas_gpu, load_seek, load_track, log,  # noqa: E402
                          read_frames, stage_dir, track_cell, video_cells)

CONFIGS = {'d2b_560': ('dinov2b', 560, 12), 'd3l_640_L20': ('dinov3l', 640, 20), 'd2l_560_L20': ('dinov2l', 560, 20)}
SAVE = ['d3l_640_L20', 'd2l_560_L20']
NBG = 32


class Enc:
    def __init__(self, cfg, dev, bf16=True, truncate=True):
        import torch
        import eci_enc_encode as EE
        name, S, L = CONFIGS[cfg]
        self.e = EE.Encoder(name, S, [L], dev, 'extrap', 8, bf16)
        blocks = self.e.model.encoder.layer if name.startswith('dinov2') else self.e.model.layer
        if truncate and L < len(blocks):
            del blocks[L:]
        self.n_blocks = len(blocks)
        self.cfg, self.S, self.G, self.L, self.dev, self.torch = cfg, S, self.e.G, L, dev, torch
        assert self.G == GRID, (cfg, self.G)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=dev)[None, :, None, None]
        self.std = torch.tensor([0.229, 0.224, 0.225], device=dev)[None, :, None, None]
        self.gpu_s, self.n_in = 0.0, 0
        self.dim = self.e.dim

    def run(self, canv, bs=32, pos=None, pos2=None):
        """canv (n, 3, 640, 640) float GPU -> (n, G*G, d) fp16 GPU, or with pos (list of index tensors) a list of
        (n_i, d) fp16 numpy arrays of the kept tokens."""
        torch, F = self.torch, self.torch.nn.functional
        out, out2 = [], []
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.inference_mode():
            for a in range(0, len(canv), bs):
                x = canv[a:a + bs]
                if self.S != CANVAS:
                    x = F.interpolate(x, size=(self.S, self.S), mode='bicubic', align_corners=False,
                                      antialias=self.S < CANVAS).round().clamp(0, 255)
                x = (x / 255 - self.mean) / self.std
                h = self.e.forward(x)[self.L]
                if pos is None:
                    out.append(h.half())
                else:
                    out += [h[j, pos[a + j]].half().cpu().numpy() for j in range(len(x))]
                    if pos2 is not None:  # a second gather from the same forward pass (None = skip that input)
                        out2 += [h[j, pos2[a + j]].half().cpu().numpy() for j in range(len(x)) if pos2[a + j] is not None]
                self.n_in += len(x)
                del h, x
        torch.cuda.synchronize()
        self.gpu_s += time.time() - t0
        if pos is None:
            return torch.cat(out)
        return (out, out2) if pos2 is not None else out


def pixel_dark(grey, bg, vc):
    """grey (n, 640, 640), bg (640, 640) -> (n, GRID*GRID) fraction of dark pixels per cell ('dark' rule)."""
    import torch.nn.functional as F
    c = (grey < 60) & ((bg[None] - grey) > 40)
    fr = F.avg_pool2d(c.float()[:, None], CANVAS // GRID).flatten(1)
    return fr * vc[None]


def grey_of(c):
    return (0.299 * c[:, 0] + 0.587 * c[:, 1] + 0.114 * c[:, 2]).round()


def cmd_encode(args):
    import torch
    import src.eci.foreground as FG
    FG.GRID = GRID
    dev = torch.device('cuda')
    stage = stage_dir()
    TOK.mkdir(parents=True, exist_ok=True)
    t00 = time.time()
    P = pd.read_parquet(PLAN / 'frames.parquet')
    P = P[P.half == args.half]
    V = pd.read_parquet(PLAN / 'videos.parquet').set_index('video')
    c = P.groupby('video').size().reset_index(name='n').sort_values(['n', 'video'], ascending=[False, True], kind='stable')
    vids = [v for j, v in enumerate(c.video) if j % args.nshard == args.shard]
    base = f'{args.half}_{args.shard}of{args.nshard}'
    if args.smoke:
        vids = vids[:args.smoke]
        base += f'_smoke{args.smoke}'
        P = P.groupby('video', group_keys=False).head(args.max_per_video)
    if (TOK / f'{base}.json').exists():
        log(f'[SKIP] {base}')
        return
    encs = {k: Enc(k, dev) for k in ['d2b_560'] + SAVE}
    gname = torch.cuda.get_device_name(0)
    log(f'{base}: {len(vids)} videos, {int(P.video.isin(vids).sum())} frames, {gname}')
    pool = ThreadPoolExecutor(args.workers)

    def dec(q):
        t_ = time.time()
        R = P[P.video == q]
        N = int(V.loc[q, 'frames'])
        d = ROOT / V.loc[q, 'half']
        offs = load_seek(d / f'{q}_t-seek.mat')
        assert len(offs) == N, (q, len(offs), N)
        bgi = np.linspace(0, N - 1, NBG).round().astype(int)
        v = R.v.values
        fr, nb = read_frames(d / f'{q}_t.seq', offs, set(bgi.tolist()) | set(v.tolist()) | set((v - LAG).tolist()),
                             threads=args.threads)
        trk = load_track(d / f'{q}_t-track.mat')
        return fr, trk, nb, time.time() - t_

    fut = {j: pool.submit(dec, vids[j]) for j in range(min(len(vids), args.workers + 1))}
    A = {k: {'t': [], 'p50': [], 'rnd_t': [], 'rnd_p50': []} for k in SAVE}
    rnd_pos = []
    meta, pos_all = [], []
    tdec = twait = 0.0
    nbytes = 0
    for k, q in enumerate(vids):
        t0 = time.time()
        fr, (tpos, nm), nb, dt = fut.pop(k).result()
        twait += time.time() - t0
        tdec += dt
        nbytes += nb
        if k + args.workers + 1 < len(vids):
            fut[k + args.workers + 1] = pool.submit(dec, vids[k + args.workers + 1])
        R = P[P.video == q]
        v = R.v.values
        N, W, H = int(V.loc[q, 'frames']), int(V.loc[q, 'W']), int(V.loc[q, 'H'])
        vc = torch.from_numpy(video_cells(W, H)).to(dev)
        bgi = np.linspace(0, N - 1, NBG).round().astype(int)
        with torch.inference_mode():
            cb = canvas_gpu([fr[x] for x in bgi], W, H, dev)
            bg = grey_of(cb).median(0).values
            tokb = encs['d2b_560'].run(cb)
            cue_b = pixel_dark(grey_of(cb), bg, vc)
            bgt, _ = FG.patch_background(tokb, exclude=cue_b > FG.FG_RULE['dark_frac'], min_frames=8)
            dd = FG.cosine_distance(tokb, bgt)
            dd[:, ~vc] = 0
            thr = FG.video_threshold(dd.cpu().numpy(), cue_b.cpu().numpy(), FG.FG_RULE)
            del cb, tokb, dd
            masks = []
            for a in range(0, len(v), 64):
                ct = canvas_gpu([fr[x] for x in v[a:a + 64]], W, H, dev)
                tk = encs['d2b_560'].run(ct)
                d_ = FG.cosine_distance(tk, bgt)
                d_[:, ~vc] = 0
                _, _, m = FG.foreground_parts(d_, pixel_dark(grey_of(ct), bg, vc), thr, FG.FG_RULE)
                masks.append((m & vc[None]).cpu().numpy())
                del ct, tk, d_
        mk = np.concatenate(masks)
        p40 = [np.flatnonzero(m).astype(np.int16) for m in mk]
        pg = [torch.from_numpy(p.astype(np.int64)).to(dev) for p in p40]
        # EXPLORATORY no-mask arm C4 (addendum 1): pool frames (train / calib roles) also keep the tokens of n40 video
        # cells drawn at random (rng seeded by video and frame), the same count as the mask keeps
        vcells = np.flatnonzero(video_cells(W, H))
        ispool = R.role.values != 'eval'
        pr = []
        for j, r in enumerate(R.itertuples()):
            if ispool[j]:
                g = np.random.default_rng([zlib.crc32(q.encode()), int(r.v)])
                pr.append(np.sort(g.choice(vcells, len(p40[j]), replace=False)).astype(np.int16))
        rnd_pos += pr
        pgr = [torch.from_numpy(p.astype(np.int64)).to(dev) for p in pr]
        jpool = np.flatnonzero(ispool)
        for kd, vv in (('t', v), ('p50', v - LAG)):
            for a in range(0, len(vv), 64):
                ct = canvas_gpu([fr[x] for x in vv[a:a + 64]], W, H, dev)
                p2 = [pgr[int(np.searchsorted(jpool, jj))] if ispool[jj] else None for jj in range(a, min(a + 64, len(vv)))]
                for cfg in SAVE:
                    o1, o2 = encs[cfg].run(ct, args.bs, pg[a:a + 64], p2)
                    A[cfg][kd] += o1
                    A[cfg]['rnd_' + kd] += o2
                del ct
        # mask check against the released tracks (never used to build the mask)
        cell = track_cell(tpos[v], W, H)     # (n, 2)
        nb3 = np.zeros_like(mk)
        m2 = mk.reshape(-1, GRID, GRID)
        pad = np.pad(m2, ((0, 0), (1, 1), (1, 1)))
        nb3 = np.zeros_like(m2)
        for dy in range(3):
            for dx in range(3):
                nb3 |= pad[:, dy:dy + GRID, dx:dx + GRID]
        nb3 = nb3.reshape(len(mk), -1)
        for j, r in enumerate(R.itertuples()):
            row = {'video': q, 'v': int(r.v), 'role': r.role, 'n40': int(len(p40[j])), 'nm': int(nm[r.v]),
                   'thr': float(thr)}
            for a_, nmx in ((0, 'res'), (1, 'int')):
                cc = cell[j, a_]
                ok = (nm[r.v] > a_) and cc >= 0
                row[f'has_{nmx}'] = bool(ok)
                row[f'rec_{nmx}'] = bool(mk[j, cc]) if ok else False
                row[f'rec3_{nmx}'] = bool(nb3[j, cc]) if ok else False
            meta.append(row)
        pos_all += p40
        del fr
        el = time.time() - t00
        log(f'  video {k + 1}/{len(vids)} {q}: {len(v)} frames, kept {np.mean([len(p) for p in p40]):.1f} cells, '
            f'{len(meta) / el:.2f} frames/s wall; gpu ' + ', '.join(f'{c_} {e.gpu_s:.0f}s' for c_, e in encs.items())
            + f'; read+decode {tdec:.0f}s summed, waited {twait:.0f}s, {nbytes / 2**30:.2f} GB read')
    pool.shutdown()
    Fm = pd.DataFrame(meta)
    saved = {}
    for cfg, d in A.items():
        for kd, lst in d.items():
            a = np.concatenate(lst)
            assert len(a) == (Fm.n40.sum() if not kd.startswith('rnd') else Fm.n40[Fm.role != 'eval'].sum())
            nm_ = f'{base}_{cfg}_{kd}.npy'
            np.save(stage / nm_, a)
            shutil.copyfile(stage / nm_, TOK / nm_)
            (stage / nm_).unlink()
            saved[nm_] = list(a.shape)
            log(f'  saved {nm_} {a.shape}')
            del a
    Fm.to_parquet(TOK / f'{base}_frames.parquet')
    np.save(TOK / f'{base}_pos40.npy', np.concatenate(pos_all))
    np.save(TOK / f'{base}_rndpos40.npy', np.concatenate(rnd_pos))
    wall = time.time() - t00
    info = {'half': args.half, 'shard': args.shard, 'nshard': args.nshard, 'videos': len(vids), 'frames': len(Fm),
            'gpu': gname, 'wall_s': round(wall, 1), 'frames_per_wall_s': round(len(Fm) / wall, 3),
            'read_decode_s_summed': round(tdec, 1), 'decode_wait_s': round(twait, 1), 'gb_read': nbytes / 2 ** 30,
            'kept40_mean': float(Fm.n40.mean()), 'frames_with_0_kept': int((Fm.n40 == 0).sum()), 'saved': saved,
            'per_config': {c_: {'size': e.S, 'grid': e.G, 'block': e.L, 'blocks_run': e.n_blocks,
                                'gpu_s': round(e.gpu_s, 1), 'inputs': e.n_in,
                                'inputs_per_gpu_s': round(e.n_in / max(e.gpu_s, 1e-9), 2)} for c_, e in encs.items()},
            'peak_mem_gb': round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)}
    (TOK / f'{base}.json').write_text(json.dumps(info, indent=1))
    (OUT / 'encode_json').mkdir(parents=True, exist_ok=True)
    (OUT / 'encode_json' / f'{base}.json').write_text(json.dumps(info, indent=1))
    log('done: ' + json.dumps(info))


def cmd_check(args):
    """truncated (blocks after 20 removed) vs full model with the block-20 hook: cosine of the tokens on 4 frames."""
    import torch
    dev = torch.device('cuda')
    V = pd.read_parquet(PLAN / 'videos.parquet')
    res = {}
    for W in (640, 656):
        r = V[V.W == W].iloc[0]
        d = ROOT / r.half
        offs = load_seek(d / f'{r.video}_t-seek.mat')
        fr, _ = read_frames(d / f'{r.video}_t.seq', offs, [1000, 2000, 3000, 4000])
        c = canvas_gpu([fr[x] for x in sorted(fr)], int(r.W), int(r.H), dev)
        for cfg in SAVE:
            a = Enc(cfg, dev, truncate=True).run(c).float()
            b = Enc(cfg, dev, truncate=False).run(c).float()
            cs = torch.nn.functional.cosine_similarity(a, b, dim=-1).flatten()
            res[f'{cfg}_W{W}'] = {'cos_min': float(cs.min()), 'cos_mean': float(cs.mean()), 'shape': list(a.shape)}
            log(f'{cfg} W{W}: {res[f"{cfg}_W{W}"]}')
            torch.cuda.empty_cache()
    (OUT / 'encode_json').mkdir(parents=True, exist_ok=True)
    (OUT / 'encode_json' / 'check_truncation.json').write_text(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['encode', 'check'], default='encode')
    ap.add_argument('--half', choices=['train', 'test'], default='train')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshard', type=int, default=2)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--threads', type=int, default=3)
    ap.add_argument('--smoke', type=int, default=0)
    ap.add_argument('--max-per-video', type=int, default=24)
    args = ap.parse_args()
    {'encode': cmd_encode, 'check': cmd_check}[args.mode](args)


if __name__ == '__main__':
    main()
