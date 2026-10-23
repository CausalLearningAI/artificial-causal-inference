"""
v1's ANIMAL MASK on the v2 tokens (the last untested v1 / v2 difference). Pre-registration: scripts/eci/v2_diag/diag.py
docstring, block 'V1M'. Masks (both used for the SAE training tokens AND the readout):

  V1M_d2  the EXACT v1 mask: src/eci/foreground.py FgBackgrounds.mask with the v1 rule (mice 'fg448', ants 'ants') and the
          v1 per-video DINOv2 backgrounds (mice fg448al/background, ants eci30/fg448/background): DINOv2-B tokens of the
          canonical frame resized to 512 (grey for the dark cue) and 448 (encoder), feature cue dist > thr_video
          (isolated patches dropped) OR dark cue, 1-patch dilation (ants: only onto patches with a dark pixel), on the
          32 x 32 grid. A native-grid v2 token is kept iff the 32-grid cell holding its centre is kept.
  V1M_d3  the same rule computed on the DINOv3-L block-20 tokens themselves at the native grid G (no second encoder):
          backgrounds from the SAME 200 sample frames and 4 time blocks as each v1 background file (rows, seg_bounds),
          per-position masked median token (excluded where the v1 dark cue > dark_frac, dilated by r; >= 20 frames per
          position else the plain median; >= 10 per block else the masked median), cosine distance, per-window
          threshold = 0.995 quantile of the sample distances over patches farther than far_r from any dark-cue patch.
          Adaptation to the finer grid (physical sizes of v1 kept): r = round(G / 32) (= 2 for mice 64, ants 48 / 52),
          far_r = round(2 G / 32) (= 4 mice, 3 ants), dark cue per native 16 px patch with v1's pixel rule
          (grey < 60 and pix_bg - grey > 40, v1's pix_bg resized to S), drop_isolated on the 8-neighbourhood (unchanged).

Steps (job.sh SCRIPT=v1mask.py):
    bg   --dom D --shard i --nshard n [--selftest]    (GPU) DINOv3 backgrounds -> BEE/<D>/bg/<file id>.npz
    prep --dom D --kind K --shard i --nshard n        (GPU) token store (M0 | d2 | d3 tokens + flags + mask2 fractions)
                                                      -> BEE/<D>/<K>/s<i>/ (mask2_test.Union layout + flags.npy)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'scripts/eci/v2_diag'))
import diag as D  # noqa: E402

FZ, MT, T2, N, R4 = D.FZ, D.MT, D.T2, D.N, D.R4
log = N.log
BEE = Path(os.environ.get('V1M_BEE') or '/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_v1mask')
OUT = D.OUT / 'v1mask'
V1RULE = {'mice': 'fg448', 'ants': 'ants'}
SERIES = {'mice': ('v1',), 'ants': ('v2', 'v3')}
FLAG = {'M0': 1, 'V1M_d2': 2, 'V1M_d3': 4}
DARK_ABS, DARK_REL, DARK_FRAC, THR_Q, MIN_BG, MIN_SEG = 60, 40, 0.15, 0.995, 20, 10


# ---------------------------------------------------------------------------------------------- grid helpers
def dilate_g(m, G, r):
    import torch.nn.functional as F
    if r <= 0:
        return m
    return F.max_pool2d(m.float().view(-1, 1, G, G), 2 * r + 1, 1, r).view(m.shape[0], -1) > 0


def drop_isolated_g(m, G):
    import torch
    import torch.nn.functional as F
    x = m.float().view(-1, 1, G, G)
    nb = F.conv2d(x, torch.ones(1, 1, 3, 3, device=x.device), padding=1) - x
    return m & (nb.view(m.shape[0], -1) > 0)


def rad(G):
    return int(round(G / 32)), int(round(2 * G / 32))


def dark_native(x, pix_bg512, S):
    """x (B, 3, S, S) uint8 cuda, v1 pix_bg (512, 512) uint8 -> (dark fraction (B, G*G), any dark (B, G*G))."""
    import torch.nn.functional as F
    from src.eci.v2.mask import grey_of
    g = grey_of(x).float()
    pb = F.interpolate(pix_bg512.float()[None, None], size=(S, S), mode='bilinear', align_corners=False)[0, 0]
    px = (g < DARK_ABS) & ((pb[None] - g) > DARK_REL)
    fr = F.avg_pool2d(px.float()[:, None], 16).flatten(1)
    return fr, fr > 0


def series_of(dom, ser, dev, enc=None):
    from src.eci.v2.pipeline import Series
    se = Series(T2.series_cfg(dom, ser), dev, load_encoder=False)
    se.enc = enc
    return se


def v1_backgrounds(dom, se, dev):
    from src.eci.foreground import RULES, FgBackgrounds
    from src.eci.v2.pipeline import obs_ranges
    m = se.cfg['mask']
    return FgBackgrounds(REPO / m['bg'], obs_ranges(REPO / se.cfg['ann']), RULES[V1RULE[dom]], dev,
                         align=m.get('align', 'none'), cache=4)


# ---------------------------------------------------------------------------------------------- DINOv3 backgrounds
def cmd_bg(args):
    import torch
    from src.eci.foreground import patch_background
    from src.eci.v2.encode import Encoder
    from src.eci.v2.video import Prefetch
    dev = torch.device('cuda')
    dom = args.dom
    enc = Encoder(dev)
    todo = []
    for ser in SERIES[dom]:
        se = series_of(dom, ser, dev, enc)
        fb = v1_backgrounds(dom, se, dev)
        mine = set(se.V)
        for kk, wid in enumerate(fb.wids):
            obs = fb.ids[fb.obs_index([fb.wstarts[kk]])[0]]
            if obs in mine:
                todo.append((ser, wid, obs))
    todo = sorted(todo)[args.shard::args.nshard]
    if args.selftest:
        todo = todo[:1]
    SE = {ser: series_of(dom, ser, dev, enc) for ser in SERIES[dom]}
    od = BEE / dom / 'bg'
    od.mkdir(parents=True, exist_ok=True)
    t0, rec = time.time(), []
    for ser, wid, obs in todo:
        if (od / f'{wid}.npz').exists() and not args.selftest:
            continue
        se = SE[ser]
        v = se.V[obs]
        z1 = np.load(REPO / se.cfg['mask']['bg'] / f'{wid}.npz')
        rows, sb = z1['rows'], z1['seg_bounds']
        pix_bg = torch.from_numpy(z1['pix_bg']).to(dev)
        ks = rows - v['src_lo']
        jobs = [((i,), se.job(v, int(k), 1, threads=2)) for i, k in enumerate(ks)]
        S, G = se.S, se.G
        r, far_r = rad(G)
        tok = torch.empty((len(ks), G * G, 1024), dtype=torch.float16, device=dev)
        dk = torch.empty((len(ks), G * G), device=dev)
        pf = Prefetch(jobs, max(4, int(os.environ.get('SLURM_CPUS_PER_TASK', 16)) - 4), fn=se.run_job)
        buf = []
        for (i,), fr in pf:
            buf.append((i, fr))
            if len(buf) == 16 or i == len(ks) - 1:
                ii = [b[0] for b in buf]
                with torch.no_grad():
                    x, _ = se.frames(v, np.concatenate([b[1] for b in buf]), ks[ii])
                    tok[ii] = enc(x)
                    dk[ii] = dark_native(x, pix_bg, S)[0]
                buf = []
        excl = dilate_g(dk > DARK_FRAC, G, r)
        bg_msk, n_ok = patch_background(tok, excl, min_frames=MIN_BG)
        bg_seg, d_seg = [], torch.empty((len(ks), G * G), device=dev)
        for a, b in zip(sb[:-1], sb[1:]):
            bb, n_s = patch_background(tok[a:b], excl[a:b], min_frames=MIN_SEG)
            few = n_s < MIN_SEG
            bb[few] = bg_msk[few]
            bg_seg.append(bb)
            d_seg[a:b] = 1 - torch.nn.functional.cosine_similarity(tok[a:b].float(), bb[None], dim=-1)
        bg_seg = torch.stack(bg_seg)
        near = dilate_g(dk > DARK_FRAC, G, far_r)
        thr = float(torch.quantile(d_seg[~near].float()[torch.randperm(int((~near).sum()), device=dev)[:5_000_000]],
                                   THR_Q))
        tmp = od / f'{wid}.tmp.npz'
        np.savez(tmp, rows=rows, seg_bounds=sb, bg_seg=bg_seg.half().cpu().numpy(), thr=np.float32(thr),
                 n_ok=n_ok.short().cpu().numpy(), G=G, r=r, far_r=far_r)
        tmp.rename(od / f'{wid}.npz')
        feat = (d_seg > thr).float().mean().item()
        rec.append({'wid': wid, 'thr': thr, 'feat_share_sample': feat, 'dark_share': float((dk > DARK_FRAC).float().mean()),
                    'min_n_ok': int(n_ok.min())})
        log(f'bg {dom} {wid}: thr {thr:.4f}, feature-cue share of sample patches {feat:.4f}, dark-cue share '
            f'{rec[-1]["dark_share"]:.4f}, min n_ok {rec[-1]["min_n_ok"]}; {len(rec)}/{len(todo)} '
            f'({time.time() - t0:.0f}s; GPU busy {enc.gpu_s:.0f}s); {MT.gpu_line()}')
    o2 = OUT / dom / 'bg'
    o2.mkdir(parents=True, exist_ok=True)
    (o2 / f's{args.shard}{"_selftest" if args.selftest else ""}.json').write_text(
        json.dumps({'n': len(rec), 'wall_s': time.time() - t0, 'gpu_s': enc.gpu_s, 'files': rec}, indent=1))


class D3Bg:
    """DINOv3 backgrounds (cmd_bg) looked up like FgBackgrounds (same windows / time blocks)."""

    def __init__(self, dom, fb, dev):
        self.dom, self.fb, self.dev, self.c = dom, fb, dev, {}

    def mask(self, z, dk, dany, rows, G):
        """z (B, G*G, 1024) fp16, v1 dark fraction / any-dark (B, G*G), rows -> (B, G*G) bool (V1M_d3)."""
        import torch
        from src.eci.foreground import segment_of
        rows = np.asarray(rows)
        k = self.fb.window_index(rows)
        out = torch.zeros(z.shape[:2], dtype=torch.bool, device=z.device)
        r, _ = rad(G)
        for kk in np.unique(k):
            wid = self.fb.wids[kk]
            if wid not in self.c:
                if len(self.c) > 3:
                    self.c.clear()
                b = np.load(BEE / self.dom / 'bg' / f'{wid}.npz')
                self.c[wid] = (torch.from_numpy(b['bg_seg']).to(self.dev), float(b['thr']), b['rows'], b['seg_bounds'])
            bg, thr, brows, sb = self.c[wid]
            sel = np.flatnonzero(k == kk)
            seg = segment_of(rows[sel] - self.fb.wstarts[kk], brows - brows[0], sb)
            st = torch.from_numpy(sel).to(z.device)
            d = 1 - torch.nn.functional.cosine_similarity(z[st].float(), bg[torch.from_numpy(seg).to(z.device)].float(),
                                                          dim=-1)
            feat = drop_isolated_g(d > thr, G)
            core = feat | (dk[st] > DARK_FRAC)
            grown = dilate_g(core, G, r)
            if self.dom == 'ants':  # v1 ants rule: the dilation only adds patches with a dark pixel
                grown = core | (grown & dany[st])
            out[st] = grown
        return out


# ---------------------------------------------------------------------------------------------- prep
def cmd_prep(args):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel
    from src.eci.v2 import mask2 as M2
    from src.eci.v2.encode import Encoder
    from src.eci.v2.mask import grey_of
    from src.eci.v2.video import Prefetch
    dev = torch.device('cuda')
    dom, kind = args.dom, args.kind
    K = MT.KINDS[dom][kind]
    h = T2.h_of(dom, K['series'])
    enc = Encoder(dev)
    se = series_of(dom, K['series'], dev, enc)
    G, S = se.G, se.S
    fb = v1_backgrounds(dom, se, dev)
    d3 = D3Bg(dom, fb, dev)
    m2 = AutoModel.from_pretrained('facebook/dinov2-base', attn_implementation='sdpa').to(dev).eval().requires_grad_(False)
    mean = torch.tensor((0.485, 0.456, 0.406), device=dev)[None, :, None, None]
    std = torch.tensor((0.229, 0.224, 0.225), device=dev)[None, :, None, None]
    cen = ((np.arange(G) + 0.5) * 32 / G).astype(int)
    idx32 = torch.from_numpy((cen[:, None] * 32 + cen[None, :]).ravel()).to(dev)  # native token -> v1 cell
    Fr = MT.frame_list(dom, kind)
    obs_all = list(dict.fromkeys(Fr.obs.values))
    Fr = Fr[Fr.obs.isin(set(obs_all[args.shard::args.nshard]))].reset_index(drop=True)
    items = []
    for o, g in Fr.groupby('obs', sort=False):
        idx = dict(zip(g.k.values, g.index.values))
        for w in sorted({T2.window_of(dom, k) for k in g.k.values}):
            kw = np.array([k for k in g.k.values if T2.window_of(dom, k) == w])
            for k0, n, every, ks in MT.runs_of(kw):
                items.append((o, w, k0, n, every, ks, [idx[k] for k in ks]))
    if args.selftest:
        items = items[:3]
    jobs = [((i,), se.job(se.V[o], k0, n, every)) for i, (o, w, k0, n, every, ks, _) in enumerate(items)]
    loc = N.local_dir() / f'v1m_{dom}_{kind}_s{args.shard}'
    loc.mkdir(parents=True, exist_ok=True)
    ftok = open(loc / 'tok.f16', 'wb')
    meta = {k: [] for k in ('tok_frame', 'tok_pos', 'fr_raw', 'fr_new', 'n_app', 'fr_app', 'fr_any', 'flags')}
    recs, t0, bgc, gpu2 = [], time.time(), {}, 0.0
    bs = 16 if S >= 1000 else 32
    pf = Prefetch(jobs, max(4, int(os.environ.get('SLURM_CPUS_PER_TASK', 16)) - 4), fn=se.run_job)
    for (i,), fr in pf:
        o, w, k0, n, every, ks, fidx = items[i]
        v = se.V[o]
        if (o, w) not in bgc:
            bgc.clear()
            bgc[(o, w)] = T2.load_pack(dom, K['series'], o, w, dev)
        pack = bgc[(o, w)]
        grid = k0 + every * np.arange(n)
        fr = fr[np.isin(grid, ks)]
        assert len(fr) == len(ks)
        for b in range(0, len(ks), bs):
            kb = ks[b:b + bs]
            rows = v['src_lo'] + kb
            with torch.no_grad():
                x, _ = se.frames(v, fr[b:b + bs], kb)
                core0 = se.mask.pixels(x, rows)
                rr = M2.cue(x, pack, core0, h)
                _, app = M2.appendage(rr['pix'], h)
                anyp = core0 | (rr['d'] > M2.T_LO)
                pool = lambda m: F.avg_pool2d(m.float()[:, None], 16).flatten(1)  # noqa: E731
                raw, frn, fapp, fany = pool(core0), pool(rr['pix']), pool(app), pool(anyp)
                ng = fapp * 256
                z = enc(x)
                # V1M_d3 (DINOv3 feature cue)
                dk, dany = dark_native(x, fb.get(fb.wids[fb.window_index(rows[:1])[0]])['pix_bg'], S)
                m_d3 = d3.mask(z, dk, dany, rows, G)
                # V1M_d2 (exact v1 mask, DINOv2-B at 448 of the 512 px frame)
                t1 = time.time()
                x512 = F.interpolate(x.float(), size=(512, 512), mode='bicubic', antialias=True,
                                     align_corners=False).round().clamp(0, 255)
                x448 = F.interpolate(x512, size=(448, 448), mode='bicubic', antialias=True, align_corners=False)
                t2 = m2(pixel_values=((x448 / 255 - mean) / std)).last_hidden_state[:, 1:].float()
                m32, _ = fb.mask(t2, grey_of(x512.to(torch.uint8)), rows)
                m_d2 = m32[:, idx32]
                torch.cuda.synchronize()
                gpu2 += time.time() - t1
            m0 = raw >= 0.25
            store = m0 | m_d2 | m_d3
            flags = m0.to(torch.int8) * 1 + m_d2.to(torch.int8) * 2 + m_d3.to(torch.int8) * 4
            for ii in range(len(kb)):
                pos = torch.nonzero(store[ii]).flatten()
                fid = len(recs)
                ftok.write(z[ii, pos].cpu().numpy().tobytes())
                meta['tok_frame'].append(np.full(len(pos), fid, np.int32))
                meta['tok_pos'].append(pos.cpu().numpy().astype(np.int16))
                for nm, t_ in (('fr_raw', raw), ('fr_new', frn), ('fr_app', fapp), ('fr_any', fany)):
                    meta[nm].append(t_[ii, pos].cpu().numpy().astype(np.float16))
                meta['n_app'].append(ng[ii, pos].round().cpu().numpy().astype(np.int16))
                meta['flags'].append(flags[ii, pos].cpu().numpy())
                r0 = Fr.iloc[fidx[b + ii]]
                rec = {'obs': o, 'k': int(kb[ii]), 'ref_f': int(r0.ref_f), 'n_store': int(len(pos)),
                       'n_M0': int(m0[ii].sum()), 'n_d2': int(m_d2[ii].sum()), 'n_d3': int(m_d3[ii].sum()),
                       'leak_d2': int((m_d2[ii] & (fany[ii] < T2.LEAK)).sum()),
                       'leak_d3': int((m_d3[ii] & (fany[ii] < T2.LEAK)).sum()),
                       'leak_M0': int((m0[ii] & (fany[ii] < T2.LEAK)).sum())}
                for c_ in ('row', 'role', 'series'):
                    if c_ in Fr:
                        rec[c_] = r0[c_]
                recs.append(rec)
        if i % 100 == 0:
            el = time.time() - t0
            log(f'v1m prep {dom} {kind} s{args.shard}: run {i}/{len(items)}, {len(recs):,} frames, {el:.0f}s '
                f'({len(recs) / max(el, 1):.1f} fps; GPU busy v3 {enc.gpu_s:.0f}s + v2 mask {gpu2:.0f}s; decode '
                f'thread-s {pf.dec_s:.0f}); {MT.gpu_line()}')
    ftok.close()
    for k_, x_ in meta.items():
        np.save(loc / ('flags.npy' if k_ == 'flags' else f'{k_}.npy'), np.concatenate(x_))
    Fm = pd.DataFrame(recs)
    Fm.to_parquet(loc / 'frames.parquet')
    el = time.time() - t0
    info = {'dom': dom, 'kind': kind, 'shard': args.shard, 'nshard': args.nshard, 'S': S, 'G': G,
            'n_frames': len(Fm), 'n_stored_tokens': int(Fm.n_store.sum()), 'wall_s': el, 'gpu_s_v3': enc.gpu_s,
            'gpu_s_v2mask': gpu2, 'gpu': torch.cuda.get_device_name(0),
            **{f'tokens_per_frame_{a}': float(Fm[f'n_{a}'].mean()) for a in ('M0', 'd2', 'd3')},
            **{f'leak_share_{a}': float(Fm[f'leak_{a}'].sum() / max(Fm[f'n_{a}'].sum(), 1)) for a in ('M0', 'd2', 'd3')}}
    if args.shard == 0:  # identity: the M0 tokens equal the step-2 store's
        info['check_M0_vs_step2'] = MT.identity_check(loc, K['ref'], Fm, G)
    log(json.dumps(info, default=float))
    od = OUT / dom / 'prep'
    od.mkdir(parents=True, exist_ok=True)
    (od / f'{kind}_s{args.shard}{"_selftest" if args.selftest else ""}.json').write_text(
        json.dumps(info, indent=1, default=float))
    if args.selftest:
        sheet(se, items, loc, Fm, G, od / f'mask_panels_{kind}_selftest.jpg')
    else:
        R4.atomic_copy(loc, BEE / dom / kind / f's{args.shard}')


def sheet(se, items, loc, Fm, G, path):
    """3 frames: M0 / V1M_d2 / V1M_d3 kept tokens (green)."""
    from PIL import Image, ImageDraw
    import torch
    tf, pos, fl = np.load(loc / 'tok_frame.npy'), np.load(loc / 'tok_pos.npy'), np.load(loc / 'flags.npy')
    tile = 384
    picks = np.linspace(0, len(Fm) - 1, 3).astype(int)
    sh = Image.new('RGB', (3 * tile, len(picks) * tile))
    for rI, f in enumerate(picks):
        o, k = Fm.obs.values[f], int(Fm.k.values[f])
        v = se.V[o]
        x, _ = se.frames(v, se.run_job(**se.job(v, k, 1)), [k])
        base = x[0].cpu().numpy().transpose(1, 2, 0).astype(np.float32)
        for c, (nm, bit) in enumerate((('M0', 1), ('V1M_d2', 2), ('V1M_d3', 4))):
            sel = pos[(tf == f) & ((fl & bit) > 0)]
            kept = np.zeros(G * G, bool)
            kept[sel] = True
            Kp = np.kron(kept.reshape(G, G), np.ones((16, 16), bool))
            im = base.copy()
            im[Kp] = 0.6 * im[Kp] + 0.4 * np.array([0, 255, 0])
            t = Image.fromarray(im.clip(0, 255).astype(np.uint8)).resize((tile, tile), Image.BILINEAR)
            ImageDraw.Draw(t).text((4, 4), f'{nm} {kept.sum()} tok | {o} k{k}', fill=(255, 255, 0))
            sh.paste(t, (c * tile, rI * tile))
    sh.save(path, quality=82)
    del torch


# ---------------------------------------------------------------------------------------------- store
def store(dom, kind, which, dev, on_gpu=True):
    """mask2_test.arm_store-like token store of one mask (flags bit) from BEE/<dom>/<kind>."""
    import torch
    old, oldl = T2.BEE, os.environ['LOCAL_DIR']
    T2.BEE, os.environ['LOCAL_DIR'] = BEE, str(Path(oldl) / 'v1m')  # own staging names (no clash with step-2 copies)
    try:
        U = T2.Union(dom, kind)
    finally:
        T2.BEE, os.environ['LOCAL_DIR'] = old, oldl
    fl = np.concatenate([np.load(p['root'] / 'flags.npy') for p in U.parts])
    m = (fl & FLAG[which]) > 0
    St = object.__new__(FZ.TokStore)
    St.F = U.F
    St.tf, St.pos = U.tf[m], U.pos[m]
    St.fd, St.app, St.any, St.raw = U.new[m], U.app[m], U.any[m], U.raw[m]
    St.tok = U.tokens(m)
    St.nf, St.G, St.dev, St.d = len(U.F), U.G, dev, 1024
    St.starts = np.r_[0, np.cumsum(np.bincount(St.tf, minlength=St.nf))]
    St.dist = np.zeros(len(St.tf), np.int8)
    St.T = D.R2.tokens_to(torch, St.tok, dev) if on_gpu else None
    return St


def main():
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=('bg', 'prep'))
    p.add_argument('--dom', choices=('mice', 'ants'))
    p.add_argument('--kind')
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--nshard', type=int, default=1)
    p.add_argument('--selftest', action='store_true')
    a = p.parse_args()
    {'bg': cmd_bg, 'prep': cmd_prep}[a.step](a)


if __name__ == '__main__':
    main()
