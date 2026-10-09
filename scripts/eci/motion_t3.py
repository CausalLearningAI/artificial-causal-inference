"""
T3 motion channel for the ECI behaviour SAE (ants + mice): does a local MOTION block at the same patch position help an
unsupervised SAE spend neurons on behaviour? One change against the levers 'w4096' baseline, everything else fixed.

Arms (same frames, splits, pool, recipe as scripts/eci/sae_levers.py 'w4096'; 3 seeds each)
    BASE   levers w4096: static DINOv2-base foreground tokens (768), results/vision/eci_sae_levers/<domain>/sae/w4096_s*
           (re-evaluated here with the same code as the motion arms; must reproduce the levers numbers).
    M1     [token_t, token_t - token_{t-1}]   D = 1 frame (5 fps) = 0.2 s
    M5     [token_t, token_t - token_{t-5}]   D = 5 frames = 1 s (token_{t-5} = prev.f16 of the existing *_fps1_d5 stores)
    Mpm1   [token_t, token_{t+1} - token_{t-1}]  symmetric +-1 frame (needs the t+1 tokens; optional)
    Neighbour frames are clipped to the video (first / last frame -> zero change), as in fg_extract_train.py.
    Neighbour tokens are taken at the SAME patch positions as the kept foreground patches of frame t (frame t's mask).
Block normalisation: TokenNorm.fit_blocks([768, 768]) (the repo's fg448mot / antsfgmot recipe): one mean per dim and one
    scalar scale per block so each block has average centred norm sqrt(768). The static block is then scaled exactly
    like BASE's input (BASE TokenNorm: average norm sqrt(768)), and the change block carries the same average weight in
    the reconstruction loss.
SAE: src/eci/sae.py Matryoshka BatchTopK, 4096 latents, prefixes 512/1024/2048/4096, k = 16, Adam 5e-4, 500 warmup,
    cosine, 6000 steps x 4096 tokens, AuxK, b_dec = geometric median; training pool = the levers uniform pool (pool seed
    0, the same token indices as BASE).

Steps
    encode   (GPU) tokens of the neighbour frames (row + offset, offset -1 or +1) of every train + eval store frame with
             foreground, at frame t's kept patch positions. Frames are staged from NFS to $LOCAL_DIR with a thread pool
             (JPGs of dataset/<domain>/frames/full, the files the store was built from), encoded with FgEncoder
             (DINOv2-base at 448, fp32 forward, fp16 rounding = the store's), written as two part files per task, in
             the token order of the sorted train frames and of the sorted eval frames. Identity checks on the task's
             first frames: frame t re-encoded vs the store token; frame t-5 vs the d5 store's prev.f16.
    train    (GPU) arms x seeds -> OUT/sae/<arm>_s<seed>/
    evaluate (GPU) frame-max codes, FVE (whole input / static block / change block), sae_levers.score_model (cross-
             fitted best AUROC, AP, honest top-1%, size-controlled AUROC), dynamic share per neuron, contact sheets of 2
             dynamic neurons per arm (frame t | patch crop at t | crop at the neighbour frame).
    video    (CPU, mice) diag_switch_regions.video_metrics of the cross-fitted AUROC / top1 neurons, raw and after
             regressing the per-video mean on fg patches / frame within each half (as scripts/eci/l2sae.py cmd_video).
    summary  (login safe) table + the pre-registered decision -> OUT/summary.json, OUT/table.md
Pre-registered decision: the better motion arm (M1 or M5; mean cross-fitted AUROC over the domain's labels, averaged
    over both domains) is KEPT iff in BOTH domains no label's cross-fitted AUROC (3-seed mean) drops more than 0.02 below
    BASE, AND at least one label per domain gains cf AUROC >= +0.03 or cf AP >= 1.3x, with the 3-seed mean above BASE's
    best seed.
Dynamic share of a neuron: squared norm of its weights on the change block / total, in the normalised input space
    (both blocks have the same average norm); encoder column and decoder row. 'Dynamic' = decoder share >= 0.5.

Outputs: results/vision/eci_t3_motion/<domain>/{nbr,sae,eval,align,codes_best,sheets,video}/, summary.json, table.md
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
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts/eci'))
import multiscale_sae as ms  # noqa: E402
import sae_levers as sl  # noqa: E402
import spatial_sae_pilot as ssp  # noqa: E402
from src.eci.domain import get_domain  # noqa: E402
from src.eci.foreground import FgEncoder, FrameDatasetFG, encode_batch, obs_rows  # noqa: E402
from src.eci.levers import GRID  # noqa: E402
from src.eci.sae import MatryoshkaBatchTopKSAE, TokenNorm, _train_step, geometric_median, load_sae, save_checkpoint  # noqa: E402

D = 768
OUT_ROOT = REPO / 'results/vision/eci_t3_motion'
LEV = REPO / 'results/vision/eci_sae_levers'
D5 = {'mice': 'dataset/mice/v1/eci/train_tokens/dinov2_base_l-1_fg448_fps1_d5',
      'ants': 'dataset/ants/eci/train_tokens/dinov2_base_l-1_antsfg_fps1_d5'}
ARMS = {'M1': {'delta': 1, 'kind': 'back'}, 'M5': {'delta': 5, 'kind': 'back'}, 'Mpm1': {'delta': 1, 'kind': 'sym'}}
LABELS = sl.LABELS
WIDTH, K = 4096, 16
log = lambda s: print(time.strftime('%H:%M:%S'), s, flush=True)  # noqa: E731


# ---------------------------------------------------------------------------------------------- frames / layout
def layout(domain):
    """-> idx (base StoreIndex), tr, ev (sorted store frame indices, frames with >= 1 kept patch for tr; all labelled
    eval frames for ev, as sae_levers), F = sorted union with n_fg > 0."""
    idx = ssp.StoreIndex(domain)
    tr, ev, _, _, train_v, eval_v = idx.split(domain, ms.N_TRAIN[domain], 0)
    tr, ev = np.sort(tr), np.sort(ev)
    F = np.union1d(tr, ev[idx.nfg[ev] > 0])
    return idx, tr, ev, F


def nbr_rows(domain, idx, frames, offset):
    """annotations.csv rows of the neighbour frames (row + offset, clipped to the video), and the frame paths."""
    ann = get_domain(domain).ann_path
    ranges = obs_rows(ann)
    rows = idx.rows[frames]
    obs = idx.obs[frames]
    lo = np.array([ranges[o][0] for o in obs])
    hi = np.array([ranges[o][1] for o in obs])
    nr = np.clip(rows + offset, lo, hi - 1)
    paths = pd.read_csv(ann, usecols=['frame_path'])['frame_path'].values
    return nr, paths


def all_pos(idx):
    """Concatenated pos.i16 of all base-store shards + global token start of every store frame."""
    sizes = idx.store.sizes
    gstart = np.r_[0, np.cumsum(sizes)[:-1]]
    pos = np.concatenate([idx.store.pos(s) for s in range(len(sizes))])
    return pos, gstart[idx.shard] + idx.start


def stage_files(srcs, dst_dir, workers=48):
    """Copy files (NFS -> local) with a thread pool -> local paths (named by index)."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    dsts = [str(dst_dir / f'{i:07d}.jpg') for i in range(len(srcs))]
    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(shutil.copyfile, srcs, dsts, chunksize=64))
    log(f'  staged {len(srcs):,} JPGs to {dst_dir} in {time.time() - t0:.0f}s ({len(srcs) / (time.time() - t0):.0f} files/s)')
    return dsts


@torch.no_grad()
def encode_paths(enc, paths, sel_pos, dev, bs, workers, out_mm=None, starts=None):
    """Encode frames, keep tokens at sel_pos[i] (int array per frame). out_mm/starts: write into a memmap; else return list."""
    ds = FrameDatasetFG(paths, enc.processor)
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, num_workers=workers, shuffle=False, pin_memory=True,
                                     prefetch_factor=4)
    res, i0, t0 = [], 0, time.time()
    for b, (pix, _) in enumerate(dl):
        tok = encode_batch(enc.model, pix, dev)
        for k in range(len(pix)):
            p = sel_pos[i0 + k]
            t = tok[k, torch.from_numpy(p.astype(np.int64)).to(dev)].cpu().numpy()
            if out_mm is None:
                res.append(t)
            else:
                out_mm[starts[i0 + k]:starts[i0 + k] + len(p)] = t
        i0 += len(pix)
        if b % 200 == 0:
            log(f'    batch {b} {i0:,}/{len(paths):,} frames {i0 / (time.time() - t0):.1f} f/s')
    return res, i0 / max(time.time() - t0, 1e-9)


def cmd_encode(args):
    dev = torch.device('cuda')
    loc = ms.local_dir()
    out = Path(args.out_dir) / 'nbr'
    out.mkdir(parents=True, exist_ok=True)
    tag = f'o{args.offset:+d}'
    pfx = out / f'{tag}_task{args.task:02d}of{args.n_tasks:02d}'
    if Path(str(pfx) + '.json').exists():
        log(f'[SKIP] {pfx}.json exists')
        return
    idx, tr, ev, F = layout(args.domain)
    blocks = np.array_split(np.arange(len(F)), args.n_tasks)
    mine = F[blocks[args.task]]
    if args.max_frames:
        mine = mine[:args.max_frames]
    is_tr = np.isin(mine, tr)
    lens = idx.nfg[mine].astype(np.int64)
    pos_all, gst = all_pos(idx)
    sel_pos = [pos_all[gst[f]:gst[f] + idx.nfg[f]] for f in mine]
    nr, paths = nbr_rows(args.domain, idx, mine, args.offset)
    srcs = [str(REPO / 'dataset' / paths[r]) for r in nr]
    log(f'{args.domain} {tag} task {args.task}/{args.n_tasks}: {len(mine):,} frames ({is_tr.sum():,} train, '
        f'{(~is_tr).sum():,} eval), {lens.sum():,} tokens; clipped at the video edge: {(nr == idx.rows[mine]).sum()}')
    enc = FgEncoder('dinov2_base', dev)
    # identity checks on the first frames: frame t vs the store; t-5 vs d5 prev.f16
    n_chk = min(args.n_check, len(mine))
    chk = {}
    if n_chk:
        cf = mine[:n_chk]
        st_tok, _, _ = idx.load(cf)
        rows_t = idx.rows[cf]
        nr5, _ = nbr_rows(args.domain, idx, cf, -5)
        lp = stage_files([str(REPO / 'dataset' / paths[r]) for r in np.r_[rows_t, nr5]], loc / 'chk')
        got, _ = encode_paths(enc, lp, sel_pos[:n_chk] + sel_pos[:n_chk], dev, args.batch_size, args.num_workers)
        g_t, g_5 = np.concatenate(got[:n_chk]).astype(np.float32), np.concatenate(got[n_chk:]).astype(np.float32)
        d5m = D5Matcher(idx, args.domain)
        prev, d5tok, matched = d5m.fetch(cf, with_tokens=True)
        prev, d5tok = prev.astype(np.float32), d5tok.astype(np.float32)
        st = st_tok.astype(np.float32)
        g_t, g_5, prev, d5tok, st = (a[matched] for a in (g_t, g_5, prev, d5tok, st))
        rel = lambda a, b: float(np.linalg.norm(a - b, axis=1).mean() / np.linalg.norm(b, axis=1).mean())  # noqa: E731
        chk = {'n_frames': n_chk, 'n_tokens': int(len(st)), 'd5_unmatched': int((~matched).sum()),
               't_vs_store_max_abs': float(np.abs(g_t - st).max()), 't_vs_store_rel': rel(g_t, st),
               't_vs_store_exact_frac': float((g_t == st).all(1).mean()),
               'tminus5_vs_d5prev_max_abs': float(np.abs(g_5 - prev).max()), 'tminus5_vs_d5prev_rel': rel(g_5, prev),
               'd5tok_vs_store_max_abs': float(np.abs(d5tok - st).max()),
               'scale_true_change_rel_d5': rel(st, prev)}
        log(f'  identity checks: {json.dumps(chk)}')
        shutil.rmtree(loc / 'chk')
    lp = stage_files(srcs, loc / 'jpg')
    starts = np.r_[0, np.cumsum(lens)[:-1]]
    mm = np.lib.format.open_memmap(loc / 'all.npy', 'w+', np.float16, (int(lens.sum()), D))
    t0 = time.time()
    _, fps = encode_paths(enc, lp, sel_pos, dev, args.batch_size, args.num_workers, out_mm=mm, starts=starts)
    mm.flush()
    t_enc = time.time() - t0
    shutil.rmtree(loc / 'jpg')
    tokmask = np.repeat(is_tr, lens)
    for part, m in (('train', tokmask), ('eval', ~tokmask)):
        a = np.empty((int(m.sum()), D), np.float16)
        o = 0
        for c0 in range(0, len(m), 2_000_000):
            mc = m[c0:c0 + 2_000_000]
            n = int(mc.sum())
            a[o:o + n] = mm[c0:c0 + 2_000_000][mc]
            o += n
        a.tofile(f'{pfx}_{part}.f16')  # one sequential write to NFS (no second local copy)
        del a
    info = {'domain': args.domain, 'offset': args.offset, 'task': args.task, 'n_tasks': args.n_tasks,
            'n_frames': int(len(mine)), 'n_train_frames': int(is_tr.sum()), 'n_eval_frames': int((~is_tr).sum()),
            'n_train_tokens': int(tokmask.sum()), 'n_eval_tokens': int((~tokmask).sum()),
            'first_frame': int(mine[0]), 'last_frame': int(mine[-1]), 'n_clipped': int((nr == idx.rows[mine]).sum()),
            'encode_s': t_enc, 'frames_per_s': fps, 'gpu': torch.cuda.get_device_name(0), 'checks': chk,
            'max_frames': args.max_frames}
    Path(str(pfx) + '.json').write_text(json.dumps(info, indent=1))
    log(f'done: {len(mine):,} frames encoded at {fps:.1f} f/s ({t_enc:.0f}s) -> {pfx}_*.f16')


# ---------------------------------------------------------------------------------------------- neighbour loading
def nbr_parts(out, domain, offset, part, n_expected):
    """Concatenate the encode tasks' part files (task order = sorted frame order) into the part's token order."""
    js = sorted((Path(out) / 'nbr').glob(f'o{offset:+d}_task*.json'))
    if not js:
        raise FileNotFoundError(f'no neighbour tokens for offset {offset} in {out}/nbr')
    infos = [json.loads(p.read_text()) for p in js]
    n_tasks = infos[0]['n_tasks']
    if len(infos) != n_tasks or any(i['max_frames'] for i in infos):
        raise RuntimeError(f'offset {offset}: {len(infos)}/{n_tasks} tasks done (or a smoke run)')
    files = [Path(str(p)[:-5] + f'_{part}.f16') for p in js]
    n = sum(i[f'n_{part}_tokens'] for i in infos)
    assert n == n_expected, (offset, part, n, n_expected)
    return files


def stage_concat(files, dst, n):
    t0 = time.time()
    mm = np.lib.format.open_memmap(dst, 'w+', np.float16, (n, D))
    o = 0
    for f in files:
        a = np.fromfile(f, np.float16).reshape(-1, D)
        mm[o:o + len(a)] = a
        o += len(a)
    assert o == n
    mm.flush()
    del mm
    log(f'  staged {len(files)} neighbour part files ({n * D * 2 / 1e9:.1f} GB) to {dst} in {time.time() - t0:.0f}s')
    return np.load(dst, mmap_mode='r')


class D5Matcher:
    """Token 5 frames earlier from the d5 store, matched to the BASE store's tokens by (frame, patch position).
    The d5 store has the same frames (rows asserted equal) but its mask was recomputed from re-encoded tokens, so a few
    patches per shard differ (mice: up to ~110 tokens of ~5M per shard). Base patches with no d5 counterpart get the
    base token itself (zero change); their count is reported."""

    def __init__(self, idx, domain):
        self.idx = idx
        self.d5 = ssp.FgTokenStore(REPO / D5[domain])
        assert len(self.d5.dirs) == len(idx.store.dirs)
        self.d5_start, self.d5_n = [], []
        for s in range(len(self.d5.dirs)):
            z0, z1 = np.load(idx.store.dirs[s] / 'frames.npz'), np.load(self.d5.dirs[s] / 'frames.npz')
            assert np.array_equal(z0['rows'], z1['rows']), s
            n = z1['n_fg'].astype(np.int64)
            self.d5_n.append(n)
            self.d5_start.append(np.r_[0, np.cumsum(n)[:-1]])
        self.f0 = np.r_[0, np.cumsum([len(n) for n in self.d5_n])[:-1]]  # first store frame of every shard
        assert np.array_equal(np.searchsorted(idx.shard, np.arange(len(self.d5_n))), self.f0)
        self._pos = {}

    def pos(self, which, s):
        k = (which, s)
        if k not in self._pos:
            self._pos[k] = (self.idx.store if which == 'base' else self.d5).pos(s)
        return self._pos[k]

    def runs(self, frames):
        idx = self.idx
        for s in np.unique(idx.shard[frames]):
            fs = frames[idx.shard[frames] == s]
            st, nn = idx.start[fs], idx.nfg[fs]
            brk = np.flatnonzero(st[1:] != st[:-1] + nn[:-1]) + 1
            for a, b in zip(np.r_[0, brk], np.r_[brk, len(fs)]):
                yield s, fs[a:b]

    def fetch_run(self, s, fr, with_tokens=False):
        """frames fr (contiguous in shard s) -> prev (n_base_tokens, 768) fp16[, d5 tokens], matched mask."""
        idx = self.idx
        lf = fr - self.f0[s]  # frame indices within the shard
        lo, hi = idx.start[fr[0]], idx.start[fr[-1]] + idx.nfg[fr[-1]]
        dlo = self.d5_start[s][lf[0]]
        dhi = self.d5_start[s][lf[-1]] + self.d5_n[s][lf[-1]]
        kb = np.repeat(lf, idx.nfg[fr]).astype(np.int64) * 1024 + self.pos('base', s)[lo:hi]
        kd = np.repeat(lf, self.d5_n[s][lf]).astype(np.int64) * 1024 + self.pos('d5', s)[dlo:dhi]
        assert (np.diff(kb) > 0).all() and (np.diff(kd) > 0).all()
        j = np.searchsorted(kd, kb).clip(0, max(len(kd) - 1, 0))
        ok = (kd[j] == kb) if len(kd) else np.zeros(len(kb), bool)
        pv = np.asarray(np.memmap(self.d5.dirs[s] / 'prev.f16', np.float16, 'r', shape=(int(self.d5.sizes[s]), D))[dlo:dhi])
        out = pv[j]
        base = None
        if not ok.all() or with_tokens:
            base = np.asarray(idx.store.tokens(s)[lo:hi])
        if not ok.all():
            out[~ok] = base[~ok]
        dt = None
        if with_tokens:
            dt = np.asarray(self.d5.tokens(s)[dlo:dhi])[j]
            dt[~ok] = base[~ok]
        return out, dt, ok

    def fetch(self, frames, with_tokens=False):
        res = [self.fetch_run(s, fr, with_tokens) for s, fr in self.runs(np.sort(frames))]
        return (np.concatenate([r[0] for r in res]), np.concatenate([r[1] for r in res]) if with_tokens else None,
                np.concatenate([r[2] for r in res]))


def stage_prev5(idx, domain, frames, dst):
    """d5 prev tokens (5 frames earlier) for `frames`, in idx.load(frames) token order, written to a local memmap."""
    m = D5Matcher(idx, domain)
    n = int(idx.nfg[frames].sum())
    t0 = time.time()
    mm = np.lib.format.open_memmap(dst, 'w+', np.float16, (n, D))
    o = miss = 0
    for s, fr in m.runs(frames):
        for c0 in range(0, len(fr), 4000):
            pv, _, ok = m.fetch_run(s, fr[c0:c0 + 4000])
            mm[o:o + len(pv)] = pv
            o += len(pv)
            miss += int((~ok).sum())
    assert o == n
    mm.flush()
    del mm
    log(f'  staged d5 prev of {len(frames):,} frames ({n * D * 2 / 1e9:.1f} GB) in {time.time() - t0:.0f}s; '
        f'{miss} of {n:,} base tokens without a d5 counterpart (zero change)')
    STAGE_INFO['d5_unmatched'] = miss
    STAGE_INFO['d5_tokens'] = n
    return np.load(dst, mmap_mode='r')


STAGE_INFO = {}


def needed_offsets(arms):
    o = set()
    for a in arms:
        if ARMS[a]['kind'] == 'sym':
            o |= {-1, +1}
        elif ARMS[a]['delta'] != 5:
            o.add(-ARMS[a]['delta'])
    return sorted(o)


def stage_neighbours(args, idx, frames, part, n, loc):
    """{offset: memmap (n, 768)} for the arms' offsets (-5 from the d5 store, others from OUT/nbr)."""
    nb = {}
    for o in needed_offsets(args.arms):
        nb[o] = stage_concat(nbr_parts(args.out_dir, args.domain, o, part, n), loc / f'nbr{o:+d}_{part}.npy', n)
    if any(ARMS[a]['delta'] == 5 for a in args.arms):
        nb[-5] = stage_prev5(idx, args.domain, frames, loc / f'nbr-5_{part}.npy')
    return nb


def change(arm, tok, nb, a, b):
    """change block rows a:b (fp32 torch on CPU)."""
    A = ARMS[arm]
    t = lambda o: torch.from_numpy(np.ascontiguousarray(nb[o][a:b])).float()  # noqa: E731
    if A['kind'] == 'sym':
        return t(+1) - t(-1)
    return torch.from_numpy(np.ascontiguousarray(tok[a:b])).float() - t(-A['delta'])


# ---------------------------------------------------------------------------------------------- train
def cmd_train(args):
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = True
    out = Path(args.out_dir)
    loc = ms.local_dir()
    idx, tr, ev, F = layout(args.domain)
    log(f'{args.domain}: {len(tr):,} train frames; arms {args.arms}')
    tok, pos, lens = ms.stage(idx, tr, loc / 'train')
    n = len(pos)
    rng = np.random.default_rng(args.pool_seed)
    cap = min(args.cap or sl.CAP[args.domain], n)
    sel = np.sort(rng.choice(n, cap, replace=False))  # identical to sae_levers' uniform pool
    nb = stage_neighbours(args, idx, tr, 'train', n, loc)
    static = torch.from_numpy(sl.gather(tok, sel))
    meta = {'domain': args.domain, 'n_train_frames': len(tr), 'n_train_tokens': n, 'cap': cap, 'pool_seed': args.pool_seed,
            'steps': args.steps, 'arms': {a: ARMS[a] for a in args.arms}, 'pool_first_idx': sel[:5].tolist(),
            'pool_sum_idx': int(sel.sum())}
    for arm in args.arms:
        t0 = time.time()
        ch = torch.empty((cap, D), dtype=torch.float16)
        for c0 in range(0, cap, 500_000):
            s = sel[c0:c0 + 500_000]
            A = ARMS[arm]
            if A['kind'] == 'sym':
                ch[c0:c0 + len(s)] = (torch.from_numpy(sl.gather(nb[1], s)).float()
                                      - torch.from_numpy(sl.gather(nb[-1], s)).float()).half()
            else:
                ch[c0:c0 + len(s)] = (static[c0:c0 + len(s)].float()
                                      - torch.from_numpy(sl.gather(nb[-A['delta']], s)).float()).half()
        X = torch.cat([static, ch], 1).to(dev)
        del ch
        zero = float(sum(int((X[a:a + 500_000, D:].float().abs().sum(1) == 0).sum()) for a in range(0, len(X), 500_000))
                     / len(X))
        rn = float((X[:200_000, D:].float().norm(dim=1) / X[:200_000, :D].float().norm(dim=1)).median())
        meta[arm] = {'zero_change_frac': zero, 'median_change_over_token_norm': rn}
        log(f'{arm}: pool {tuple(X.shape)} built in {time.time() - t0:.0f}s; zero-change tokens {100 * zero:.2f}%, '
            f'median |change| / |token| {rn:.3f}')
        for seed in args.seeds:
            train_one(arm, seed, X, args.steps, out / 'sae' / f'{arm}_s{seed}',
                      {'config': arm, 'arm': ARMS[arm], 'motion_delta': ARMS[arm]['delta'], 'domain': args.domain,
                       'cap': cap, 'pool_seed': args.pool_seed, 'blocks': [D, D]}, dev)
            torch.cuda.empty_cache()
        del X
        torch.cuda.empty_cache()
    meta['d5_match'] = dict(STAGE_INFO)
    (out / f'train_meta_{"_".join(args.arms)}.json').write_text(json.dumps(meta, indent=1))
    shutil.rmtree(loc / 'train', ignore_errors=True)


def train_one(name, seed, X, steps, out, extra, dev, bs=4096, lr=5e-4, warmup=500):
    """sae_levers.train_one (uniform sampler, no centring) with a 1536-dim input and per-block TokenNorm."""
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    g = np.random.default_rng(seed)
    N, d = X.shape
    ii = torch.from_numpy(np.sort(g.choice(N, min(500_000, N), replace=False))).to(dev)
    init = X[ii].float()
    norm = TokenNorm(d).to(dev).fit_blocks(init, [D, D], seed=seed)
    sae = MatryoshkaBatchTopKSAE(d_in=d, n_latents=WIDTH, prefixes=(WIDTH // 8, WIDTH // 4, WIDTH // 2, WIDTH), k=K,
                                 seed=seed).to(dev)
    with torch.no_grad():
        sae.b_dec.copy_(geometric_median(norm(init[:100_000])))
    del init
    opt = torch.optim.Adam(sae.parameters(), lr=lr, betas=(0.9, 0.999))
    gen = torch.Generator(device=dev).manual_seed(seed)
    perm, p = torch.randperm(N, generator=gen, device=dev), 0
    history, t0 = [], time.time()
    for step in range(steps):
        if p + bs > N:
            perm, p = torch.randperm(N, generator=gen, device=dev), 0
        b = perm[p:p + bs]
        p += bs
        logs, gn = _train_step(sae, opt, norm(X[b]), step, steps, lr, warmup, 1.0)
        if step % 1000 == 0 or step == steps - 1:
            logs.update(step=step, elapsed_s=round(time.time() - t0, 1))
            history.append(logs)
            log(f"    step {step}/{steps} fve {logs['fve']:.4f} l0 {logs['l0']:.1f} dead {logs['n_dead']} {logs['elapsed_s']}s")
    sae.eval()
    dead = int((sae.tokens_since_fired >= sae.dead_tokens).sum())
    save_checkpoint(out / 'sae.pt', sae, norm, extra={**extra, 'seed': seed})
    (out / 'train.json').write_text(json.dumps({**extra, 'seed': seed, 'n_pool': N, 'steps': steps,
                                                'token_presentations': steps * bs, 'train_time_s': time.time() - t0,
                                                'dead_at_end_train': dead, 'dead_frac_train': dead / WIDTH,
                                                'norm_scale_blocks': [float(norm.scale[0]), float(norm.scale[D])],
                                                'gpu': torch.cuda.get_device_name(0), 'history': history}, indent=1))
    log(f'    {name} s{seed}: trained in {time.time() - t0:.0f}s, final fve {history[-1]["fve"]:.4f}, dead {dead}/{WIDTH}')


# ---------------------------------------------------------------------------------------------- evaluate
@torch.no_grad()
def encode_codes(sae, norm, arm, tok, nb, lens, dev):
    """Frame-max codes (F, m) fp16 GPU + FVE (whole / static block / change block), L0, firing rates."""
    m = sae.n_latents
    st = np.r_[0, np.cumsum(lens)]
    codes = torch.zeros(len(lens), m, dtype=torch.float16, device=dev)
    dd = norm.mean.shape[0]
    s1 = torch.zeros(dd, dtype=torch.float64, device=dev)
    s2 = torch.zeros(dd, dtype=torch.float64, device=dev)
    sse = torch.zeros(dd, dtype=torch.float64, device=dev)
    fire = torch.zeros(m, dtype=torch.float64, device=dev)
    l0 = ntok = 0
    for f0, f1 in sl.frame_chunks(lens, max(4096, 2 ** 27 // m)):
        lo, hi = st[f0], st[f1]
        if hi == lo:
            continue
        T = torch.from_numpy(np.ascontiguousarray(tok[lo:hi])).to(dev)
        if arm != 'BASE':
            T = torch.cat([T.float(), change(arm, tok, nb, lo, hi).to(dev)], 1)
        x = norm(T)
        z = sae.encode(x, mode='threshold')
        xh = sae.decode(z)
        s1 += x.double().sum(0)
        s2 += x.double().pow(2).sum(0)
        sse += (x - xh).double().pow(2).sum(0)
        act = z > 0
        fire += act.sum(0).double()
        l0 += float(act.sum())
        ntok += len(x)
        fr = torch.repeat_interleave(torch.arange(f1 - f0, device=dev), torch.from_numpy(lens[f0:f1]).to(dev))
        mx = torch.zeros(f1 - f0, m, device=dev)
        mx.index_reduce_(0, fr, z, 'amax', include_self=True)
        codes[f0:f1] = mx.half()
    tss_d = s2 - s1.pow(2) / ntok
    fve = lambda sl_: float(1 - sse[sl_].sum() / tss_d[sl_].sum())  # noqa: E731
    fr_ = (fire / ntok).cpu().numpy()
    stats = {'fve': fve(slice(None)), 'l0_per_token': l0 / ntok, 'dead_frac': float((fr_ == 0).mean()), 'n_tokens': ntok,
             'token_fire_rate': fr_}
    if dd > D:
        stats.update(fve_static=fve(slice(0, D)), fve_change=fve(slice(D, None)),
                     var_share_change=float(tss_d[D:].sum() / tss_d.sum()))
    return codes, stats


@torch.no_grad()
def dyn_share(sae):
    if sae.d_in == D:
        return None, None
    we = sae.W_enc.float().pow(2)  # (d_in, m)
    wd = sae.W_dec.float().pow(2)  # (m, d_in)
    enc = (we[D:].sum(0) / we.sum(0)).cpu().numpy()
    dec = (wd[:, D:].sum(1) / wd.sum(1)).cpu().numpy()
    return enc, dec


@torch.no_grad()
def neuron_frame_max(sae, norm, arm, j, tok, nb, pos, lens, dev):
    """frame max of neuron j + argmax patch per frame."""
    st = np.r_[0, np.cumsum(lens)]
    x = np.zeros(len(lens), np.float32)
    We, be, th = sae.W_enc[:, [j]], sae.b_enc[[j]], sae.threshold
    for f0, f1 in sl.frame_chunks(lens, 1_000_000):
        lo, hi = st[f0], st[f1]
        if hi == lo:
            continue
        T = torch.from_numpy(np.ascontiguousarray(tok[lo:hi])).to(dev)
        if arm != 'BASE':
            T = torch.cat([T.float(), change(arm, tok, nb, lo, hi).to(dev)], 1)
        pre = torch.relu((norm(T) - sae.b_dec) @ We + be)[:, 0]
        z = pre * (pre > th)
        fr = torch.repeat_interleave(torch.arange(f1 - f0, device=dev), torch.from_numpy(lens[f0:f1]).to(dev))
        mx = torch.zeros(f1 - f0, device=dev)
        mx.index_reduce_(0, fr, z, 'amax', include_self=True)
        x[f0:f1] = mx.cpu().numpy()
    return x


def argmax_patch(sae, norm, arm, j, tok, nb, pos, lens, i, dev):
    st = np.r_[0, np.cumsum(lens)]
    lo, hi = st[i], st[i + 1]
    T = torch.from_numpy(np.ascontiguousarray(tok[lo:hi])).to(dev)
    if arm != 'BASE':
        T = torch.cat([T.float(), change(arm, tok, nb, lo, hi).to(dev)], 1)
    pre = ((norm(T) - sae.b_dec) @ sae.W_enc[:, [j]] + sae.b_enc[[j]])[:, 0]
    return int(pos[lo + int(pre.argmax())])


def motion_sheet(path, arm, frames, lab, label, x, boxes, paths_all, ann_rows, ranges_lo_hi, title):
    """Per tile: frame t (patch outlined; green = label positive) | 4x crop around the patch at t | at the neighbour
    frame(s) (t - D, or t-1 and t+1)."""
    from PIL import Image, ImageDraw
    A = ARMS[arm]
    offs = [-1, +1] if A['kind'] == 'sym' else [-A['delta']]
    cw = 128
    tw = 256 + cw * (1 + len(offs))
    cols = 2
    rows_ = (len(frames) + cols - 1) // cols
    sheet = Image.new('RGB', (cols * tw, 18 + rows_ * (256 + 14)), 'white')
    dr = ImageDraw.Draw(sheet)
    dr.text((4, 2), title, fill='black')
    c = 512 // GRID
    for q, (i, p) in enumerate(zip(frames, boxes)):
        r = int(ann_rows[i])
        lo, hi = ranges_lo_hi[q]
        r_, c_ = divmod(int(p), GRID)
        yv = bool(lab[label].iat[i])
        col = (0, 220, 0) if yv else (230, 0, 0)
        ims = []
        for o in [0] + offs:
            rr = int(np.clip(r + o, lo, hi - 1))
            im = Image.open(REPO / 'dataset' / paths_all[rr]).convert('RGB')
            if im.size != (512, 512):
                im = im.resize((512, 512))
            ims.append(im)
        full = ims[0].copy()
        ImageDraw.Draw(full).rectangle([c_ * c - 2, r_ * c - 2, (c_ + 1) * c + 1, (r_ + 1) * c + 1], outline=col, width=3)
        cx, cy = c_ * c + c // 2, r_ * c + c // 2
        box = (max(0, cx - 32), max(0, cy - 32), max(0, cx - 32) + 64, max(0, cy - 32) + 64)
        rr_, cc_ = divmod(q, cols)
        x0, y0 = cc_ * tw, 18 + rr_ * (256 + 14)
        sheet.paste(full.resize((256, 256)), (x0, y0))
        for k, im in enumerate(ims):
            cr = im.crop(box).resize((cw, cw), Image.NEAREST)
            d2 = ImageDraw.Draw(cr)
            px0, py0 = (c_ * c - box[0]) * 2, (r_ * c - box[1]) * 2
            d2.rectangle([px0, py0, px0 + 2 * c, py0 + 2 * c], outline=col, width=1)
            sheet.paste(cr, (x0 + 256 + k * cw, y0 + (256 - cw) // 2))
            dr.text((x0 + 256 + k * cw + 2, y0 + (256 - cw) // 2 - 12), 't' if k == 0 else f't{offs[k - 1]:+d}',
                    fill='black')
        dr.text((x0 + 2, y0 + 256), f'{lab.obs.iat[i]} f{lab.frame_idx.iat[i]} {"POS" if yv else "neg"} a={float(x[i]):.2f}',
                fill='black')
    sheet.save(path, quality=85)


@torch.no_grad()
def cmd_evaluate(args):
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False
    out = Path(args.out_dir)
    loc = ms.local_dir()
    lout = loc / 'out'
    for d in ('eval', 'align', 'codes_best', 'sheets'):
        (lout / d).mkdir(parents=True, exist_ok=True)
    idx, tr, ev, F = layout(args.domain)
    lab, _, valid = ms.eval_labels(args.domain, idx, ev)
    labels = LABELS[args.domain]
    if args.domain == 'mice':
        unit = lab.obs.str.rsplit('_', n=2).str[0].values
    else:
        unit = lab.obs.values
    units = sorted(set(unit))
    half = pd.Series(unit).map({u: i % 2 for i, u in enumerate(units)}).values
    ref_half = np.load(LEV / args.domain / 'half.npy')
    assert np.array_equal(half, ref_half), 'cross-fit halves differ from sae_levers'
    log(f'{args.domain}: {len(ev):,} eval frames, {len(units)} units; halves identical to sae_levers')
    _, pos, lens = ms.stage(idx, ev, loc / 'eval')
    n = int(lens.sum())
    tok = np.load(loc / 'eval' / 'tok.npy')
    shutil.rmtree(loc / 'eval')
    arms_m = [a for a in args.arms if a != 'BASE']
    args.arms, all_arms = arms_m, args.arms
    nb = stage_neighbours(args, idx, ev, 'eval', n, loc) if arms_m else {}
    args.arms = all_arms
    log(f'  eval tokens {n:,} ({tok.nbytes / 1e9:.1f} GB) in RAM + {len(nb)} neighbour sets memory-mapped on local disk')
    rank = np.argsort(np.argsort(lens + np.random.default_rng(0).random(len(lens)) * 1e-3))
    decile = np.minimum(rank * 10 // len(lens), 9)
    Yt = torch.from_numpy(lab[labels].values.astype(bool)).to(dev)
    ann = get_domain(args.domain).ann_path
    ranges = obs_rows(ann)
    paths_all = pd.read_csv(ann, usecols=['frame_path'])['frame_path'].values
    lohi = [ranges[o] for o in lab.obs.values]
    sheet_index = []
    for arm in args.arms:
        for seed in args.seeds:
            t0 = time.time()
            key = f'{arm}_s{seed}'
            pth = (LEV / args.domain / 'sae' / f'w4096_s{seed}' / 'sae.pt') if arm == 'BASE' else out / 'sae' / key / 'sae.pt'
            sae, norm, ck = load_sae(pth, dev)
            codes, st = encode_codes(sae, norm, arm, tok, nb, lens, dev)
            res, arrays = sl.score_model(codes, Yt, labels, valid, half, decile, dev)
            enc_s, dec_s = dyn_share(sae)
            e = {'config': arm, 'seed': seed, 'key': key, 'width': sae.n_latents, 'd_in': sae.d_in,
                 'checkpoint': str(pth.relative_to(REPO)), 'd5_match': dict(STAGE_INFO), **{k: v for k, v in st.items() if k != 'token_fire_rate'},
                 'labels': res}
            fr_ = st['token_fire_rate']
            frame_rate = (codes > 0).float().mean(0).cpu().numpy()
            if dec_s is not None:
                alive = fr_ > 0
                sel_neurons = {b: sorted({d[s]['neuron'] for d in res[b]['dirs'] for s in ('auc', 'ap', 'top1')})
                               for b in labels}
                e['dynamic'] = {
                    'alive': int(alive.sum()),
                    'dec_share_ge_0.5_frac_alive': float((dec_s[alive] >= 0.5).mean()),
                    'enc_share_ge_0.5_frac_alive': float((enc_s[alive] >= 0.5).mean()),
                    'dec_share_quantiles_alive': np.quantile(dec_s[alive], [0.1, 0.25, 0.5, 0.75, 0.9]).tolist(),
                    'enc_share_quantiles_alive': np.quantile(enc_s[alive], [0.1, 0.25, 0.5, 0.75, 0.9]).tolist(),
                    'token_fire_weighted_dec_share': float((dec_s * fr_).sum() / fr_.sum()),
                    'selected': {b: {str(j): {'dec_share': float(dec_s[j]), 'enc_share': float(enc_s[j])}
                                     for j in sel_neurons[b]} for b in labels},
                    'cf_auroc_neuron_dec_share': {b: [float(dec_s[d['auc']['neuron']]) for d in res[b]['dirs']]
                                                  for b in labels}}
            (lout / 'eval' / f'{key}.json').write_text(json.dumps(e, indent=1))
            np.savez_compressed(lout / 'align' / f'{key}.npz', token_fire_rate=fr_.astype(np.float32),
                                frame_rate=frame_rate.astype(np.float32),
                                **({} if dec_s is None else {'dec_share': dec_s, 'enc_share': enc_s}), **arrays)
            js = sorted({d[s]['neuron'] for b in labels for d in res[b]['dirs'] for s in ('auc', 'ap', 'top1')})
            np.savez_compressed(lout / 'codes_best' / f'{key}.npz', neurons=np.array(js),
                                codes=codes[:, torch.tensor(js, device=dev)].cpu().numpy())
            del codes
            torch.cuda.empty_cache()
            if dec_s is not None and seed == args.seeds[0]:
                # two dynamic neurons: (a) best all-frame AUROC on the primary label among dynamic neurons firing on
                # >= 1% of frames; (b) the most dynamic neuron firing on >= 5% of frames
                prim = 'nose_tail' if args.domain == 'mice' else 'groom_any'
                auc_all = arrays[f'{prim}|all|auc']
                dyn = dec_s >= 0.5
                c1 = np.where(dyn & (frame_rate >= 0.01), np.nan_to_num(auc_all, nan=-1), -np.inf)
                c2 = np.where(frame_rate >= 0.05, dec_s, -np.inf)
                picks = []
                if np.isfinite(c1.max()):
                    picks.append(('best_' + prim, int(np.argmax(c1))))
                j2 = int(np.argmax(c2))
                if np.isfinite(c2.max()) and j2 not in [p[1] for p in picks]:
                    picks.append(('most_dynamic_common', j2))
                for why, j in picks:
                    x = neuron_frame_max(sae, norm, arm, j, tok, nb, pos, lens, dev)
                    pick, seen = [], {}
                    for i in np.argsort(-x, kind='stable'):
                        if x[i] <= 0 or len(pick) == 16:
                            break
                        v = lab.obs.iat[i]
                        if seen.get(v, 0) >= 2:
                            continue
                        seen[v] = seen.get(v, 0) + 1
                        pick.append(int(i))
                    boxes = [argmax_patch(sae, norm, arm, j, tok, nb, pos, lens, i, dev) for i in pick]
                    fn = f'{key}__{why}_n{j}.jpg'
                    motion_sheet(lout / 'sheets' / fn, arm, pick, lab, prim, x, boxes, paths_all, lab['row'].values,
                                 [lohi[i] for i in pick],
                                 f'{args.domain} {key} neuron {j} ({why}): dec share {dec_s[j]:.2f}, frame rate '
                                 f'{frame_rate[j]:.3f}, all-frame {prim} AUROC {auc_all[j]:.3f}; green = {prim} positive')
                    sheet_index.append({'key': key, 'why': why, 'neuron': j, 'dec_share': float(dec_s[j]),
                                        'enc_share': float(enc_s[j]), 'frame_rate': float(frame_rate[j]),
                                        f'all_frame_auroc_{prim}': float(auc_all[j]),
                                        'n_pos_shown': int(sum(bool(lab[prim].iat[i]) for i in pick)),
                                        'n_shown': len(pick), 'file': fn})
            f = lambda b: (f"{b}: cf AUROC {res[b]['cf_auroc']:.3f} cf AP {res[b]['cf_ap']:.4f} top1 "  # noqa: E731
                           f"{res[b]['cf_top1']:.3f} sc {res[b]['cf_size_ctrl_auroc']:.3f}")
            log(f'  {key}: FVE {st["fve"]:.4f}' + (f' (static {st["fve_static"]:.4f} change {st["fve_change"]:.4f})'
                                                    if 'fve_static' in st else '')
                + f' L0 {st["l0_per_token"]:.2f} dead {100 * st["dead_frac"]:.1f}% [{time.time() - t0:.0f}s] | '
                + ' | '.join(f(b) for b in labels)
                + (f" | dynamic(dec>=0.5) {e['dynamic']['dec_share_ge_0.5_frac_alive']:.3f}" if 'dynamic' in e else ''))
    (lout / 'sheets' / f'index_{"_".join(args.arms)}.json').write_text(json.dumps(sheet_index, indent=1))
    np.save(lout / 'lens.npy', lens.astype(np.int32))
    lab.drop(columns=['frame_path']).to_parquet(lout / 'labels.parquet')
    np.save(lout / 'half.npy', half)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copytree(lout, out, dirs_exist_ok=True)
    log(f'results copied to {out}')


# ---------------------------------------------------------------------------------------------- video (mice, CPU)
def cmd_video(args):
    """l2sae.cmd_video measurements: per-video in-window means of the cross-fitted AUROC / top1 neurons, raw and
    fg-count controlled (per-video mean regressed on per-video mean fg patches / frame within each half)."""
    import diag_switch_regions as dsr
    from src.eci.domain import MiceDomain
    assert args.domain == 'mice'
    out = Path(args.out_dir)
    vout = out / 'video'
    vout.mkdir(parents=True, exist_ok=True)
    loc = ms.local_dir()
    t0 = time.time()
    A = dsr.design()
    lab = pd.read_parquet(out / 'labels.parquet')
    half_f = np.load(out / 'half.npy')
    lens = np.load(out / 'lens.npy').astype(np.float64)
    assert len(lab) == 172800
    vi = {o: i for i, o in enumerate(A.observation_id)}
    fv = lab.obs.map(vi).values.astype(np.int64)
    shutil.copyfile(dsr.ANN, loc / 'ann.csv')
    nfr = pd.read_csv(loc / 'ann.csv', usecols=['observation_id'])['observation_id'].value_counts()
    (loc / 'ann.csv').unlink()
    wstart = MiceDomain().window_start(nfr.loc[A.observation_id].values, A.stage.values)
    inwin = lab.frame_idx.values >= wstart[fv]
    assert (np.bincount(fv[inwin], minlength=len(A)) == 900).all()
    w = inwin.astype(np.float64)
    vh = pd.Series(half_f).groupby(fv).first()
    assert (vh.values == A.half.values).all(), 'cross-fit halves differ from diag_video_level'
    vmean = lambda fr: np.bincount(fv, weights=np.asarray(fr, np.float64) * w, minlength=len(A)) / 900  # noqa: E731
    fg = vmean(lens)

    def resid(x):
        o = np.empty_like(x)
        for h in (0, 1):
            m = (A.half == h).values
            X_ = np.c_[np.ones(m.sum()), fg[m]]
            o[m] = x[m] - X_ @ np.linalg.lstsq(X_, x[m], rcond=None)[0]
        return o

    R = {'candidates': {}}
    PV = A[['observation_id', 'pool', 'stage', 'half']].copy()
    PV['fg_per_frame'] = fg
    for p in sorted((out / 'eval').glob('*.json')):
        e = json.loads(p.read_text())
        key = e['key']
        z = np.load(out / 'codes_best' / f'{key}.npz')
        neurons, codes = list(z['neurons']), z['codes']
        for sel in ('auc', 'top1'):
            for b in ('nose_nose', 'nose_tail'):
                x = np.full(len(A), np.nan)
                picks = {}
                for d in e['labels'][b]['dirs']:
                    j = d[sel]['neuron']
                    m = (A.half == d['test_half']).values
                    x[m] = vmean(codes[:, neurons.index(j)])[m]
                    picks[d['test_half']] = j
                assert np.isfinite(x).all()
                raw, _, _ = dsr.video_metrics(x, A, b)
                ctl, _, _ = dsr.video_metrics(resid(x), A, b)
                name = f'{key}|{sel}|{b}'
                PV[name] = x
                R['candidates'][name] = {'key': key, 'arm': e['config'], 'seed': e['seed'], 'selection': sel,
                                         'behaviour': b, 'neurons_by_test_half': picks, 'raw': raw, 'fg_controlled': ctl}
    R['truth'] = {b: {q: {sw: dsr.ttest(dsr.plain_deltas(A[f'{b}_{q}'].values, sorted(A.pool.unique()),
                                                          {(p_, s_): i for i, (p_, s_) in enumerate(zip(A.pool, A.stage))})[sw])
                          for sw in dsr.SW} for q in ('rate', 'bpm')} for b in ('nose_nose', 'nose_tail')}
    (vout / 'results.json').write_text(json.dumps(R, indent=1, default=float))
    PV.to_parquet(vout / 'per_video.parquet')
    log(f'video step done in {time.time() - t0:.0f}s -> {vout}')


# ---------------------------------------------------------------------------------------------- summary
def ms_(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return {'mean': float(np.mean(v)), 'sd': float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, 'per_seed': v} if v else None


def domain_summary(out, domain):
    E = [json.loads(p.read_text()) for p in sorted((out / 'eval').glob('*.json'))]
    labels = LABELS[domain]
    agg = {}
    for arm in dict.fromkeys(e['config'] for e in E):
        es = sorted([e for e in E if e['config'] == arm], key=lambda e: e['seed'])
        a = {'seeds': [e['seed'] for e in es]}
        for q in ('fve', 'fve_static', 'fve_change', 'var_share_change', 'l0_per_token', 'dead_frac'):
            if q in es[0]:
                a[q] = ms_([e[q] for e in es])
        for b in labels:
            a[b] = {q: ms_([e['labels'][b][q] for e in es]) for q in
                    ('cf_auroc', 'cf_ap', 'cf_ap_at_auroc', 'cf_top1', 'cf_size_ctrl_auroc', 'base_rate')}
        if 'dynamic' in es[0]:
            a['dynamic'] = {q: ms_([e['dynamic'][q] for e in es]) for q in
                            ('dec_share_ge_0.5_frac_alive', 'enc_share_ge_0.5_frac_alive', 'token_fire_weighted_dec_share')}
            a['dynamic']['median_dec_share'] = ms_([e['dynamic']['dec_share_quantiles_alive'][2] for e in es])
            a['dynamic']['cf_auroc_neuron_dec_share'] = {b: [x for e in es for x in e['dynamic']['cf_auroc_neuron_dec_share'][b]]
                                                         for b in labels}
        agg[arm] = a
    # reproduction of the levers numbers for BASE
    rep = None
    if 'BASE' in agg:
        L = json.loads((LEV / domain / 'summary.json').read_text())['mean_sd']['w4096']
        rep = {b: {q: [agg['BASE'][b][q]['mean'], L[b][q]['mean']] for q in ('cf_auroc', 'cf_ap', 'cf_top1')} for b in labels}
        rep['fve'] = [agg['BASE']['fve']['mean'], L['fve']['mean']]
    return agg, rep


def criteria(agg, labels):
    B = agg['BASE']
    c = {}
    for arm, a in agg.items():
        if arm == 'BASE':
            continue
        x = {}
        for b in labels:
            d = a[b]['cf_auroc']['mean'] - B[b]['cf_auroc']['mean']
            r = a[b]['cf_ap']['mean'] / B[b]['cf_ap']['mean']
            g_auc = d >= 0.03 and a[b]['cf_auroc']['mean'] > max(B[b]['cf_auroc']['per_seed'])
            g_ap = r >= 1.3 and a[b]['cf_ap']['mean'] > max(B[b]['cf_ap']['per_seed'])
            x[b] = {'d_cf_auroc': d, 'ratio_cf_ap': r, 'drop_gt_0.02': bool(d < -0.02), 'auroc_gain': bool(g_auc),
                    'ap_gain': bool(g_ap)}
        x['no_drop'] = not any(x[b]['drop_gt_0.02'] for b in labels)
        x['any_gain'] = any(x[b]['auroc_gain'] or x[b]['ap_gain'] for b in labels)
        x['mean_cf_auroc'] = float(np.mean([a[b]['cf_auroc']['mean'] for b in labels]))
        x['mean_d_cf_auroc'] = float(np.mean([x[b]['d_cf_auroc'] for b in labels]))
        c[arm] = x
    return c


def cmd_summary(args):
    S, L = {}, ['# T3 motion channel: arm x label (cross-fitted, mean +- sd over 3 seeds)\n']
    f = lambda d, k=3: '-' if d is None else f"{d['mean']:.{k}f} +- {d['sd']:.{k}f}"  # noqa: E731
    for domain in ('mice', 'ants'):
        out = OUT_ROOT / domain
        if not (out / 'eval').exists():
            continue
        agg, rep = domain_summary(out, domain)
        labels = LABELS[domain]
        crit = criteria(agg, labels) if 'BASE' in agg else {}
        S[domain] = {'mean_sd': agg, 'base_reproduction_vs_levers': rep, 'criteria': crit}
        L.append(f'\n## {domain}\n')
        if rep:
            L.append('BASE re-evaluated vs levers summary (mine / levers): ' + json.dumps(
                {k: ([round(x, 4) for x in v] if isinstance(v, list) else {q: [round(x, 4) for x in w] for q, w in v.items()})
                 for k, v in rep.items()}) + '\n')
        L.append('| label | arm | cf AUROC | cf AP | honest top1% | size-ctrl AUROC | AP@AUROC | per-seed cf AUROC |')
        L.append('|---|---|---|---|---|---|---|---|')
        for b in labels:
            for arm, a in agg.items():
                d = a[b]
                L.append(f"| {b} | {arm} | {f(d['cf_auroc'])} | {f(d['cf_ap'], 4)} | {f(d['cf_top1'])} | "
                         f"{f(d['cf_size_ctrl_auroc'])} | {f(d['cf_ap_at_auroc'], 4)} | "
                         f"{[round(x, 3) for x in d['cf_auroc']['per_seed']]} |")
        L.append('\n| arm | FVE (input) | FVE static block | FVE change block | change-block var share | dynamic neurons '
                 '(dec share >= 0.5, of alive) | median dec share | firing-weighted dec share | dead |')
        L.append('|---|---|---|---|---|---|---|---|---|')
        for arm, a in agg.items():
            dy = a.get('dynamic', {})
            L.append(f"| {arm} | {f(a['fve'], 4)} | {f(a.get('fve_static'), 4)} | {f(a.get('fve_change'), 4)} | "
                     f"{f(a.get('var_share_change'))} | {f(dy.get('dec_share_ge_0.5_frac_alive'))} | "
                     f"{f(dy.get('median_dec_share'))} | {f(dy.get('token_fire_weighted_dec_share'))} | {f(a['dead_frac'])} |")
        for arm, x in crit.items():
            L.append(f"\n{arm} vs BASE: " + '; '.join(f"{b} dAUROC {x[b]['d_cf_auroc']:+.3f}, AP x{x[b]['ratio_cf_ap']:.2f}"
                                                      f"{' GAIN' if x[b]['auroc_gain'] or x[b]['ap_gain'] else ''}"
                                                      f"{' DROP' if x[b]['drop_gt_0.02'] else ''}" for b in labels)
                     + f" | no drop: {x['no_drop']}, any gain: {x['any_gain']}")
        vr = out / 'video' / 'results.json'
        if vr.exists():
            V = json.loads(vr.read_text())['candidates']
            L.append('\n### video level (mice), held-out halves, raw | fg-count controlled; mean +- sd over seeds\n')
            L.append('| behaviour | arm | sel | video r rate | r bouts/min | pooled d-r | sign agree (of 4 switches) | '
                     'ctl video r rate | ctl r bouts | ctl pooled d-r | ctl sign agree |')
            L.append('|---|---|---|---|---|---|---|---|---|---|---|')
            for b in ('nose_nose', 'nose_tail'):
                for arm in agg:
                    for sel in ('auc', 'top1'):
                        cs = [V[k] for k in V if V[k]['arm'] == arm and V[k]['selection'] == sel and V[k]['behaviour'] == b]
                        if not cs:
                            continue
                        g = lambda q, w: ms_([c[w][q] for c in cs])  # noqa: E731
                        sa = lambda w: np.mean([sum(c[w]['effects'][s]['sign_agree_rate'] for s in c[w]['effects'])  # noqa: E731
                                                for c in cs])
                        L.append(f"| {b} | {arm} | {sel} | {f(g('video_r_rate', 'raw'), 2)} | {f(g('video_r_bpm', 'raw'), 2)} | "
                                 f"{f(g('delta_r_pooled', 'raw'), 2)} | {sa('raw'):.1f} | {f(g('video_r_rate', 'fg_controlled'), 2)} | "
                                 f"{f(g('video_r_bpm', 'fg_controlled'), 2)} | {f(g('delta_r_pooled', 'fg_controlled'), 2)} | "
                                 f"{sa('fg_controlled'):.1f} |")
                        S[domain].setdefault('video', {})[f'{b}|{arm}|{sel}'] = {
                            w: {q: g(q, w) for q in ('video_r_rate', 'video_r_bpm', 'delta_r_pooled')} | {'sign_agree_mean': sa(w)}
                            for w in ('raw', 'fg_controlled')}
            # per-switch dz (seed 0, auc) for each arm
            L.append('\nper-switch dz of the held-out neuron (seed mean, AUROC pick; annotated rate dz in brackets):\n')
            for b in ('nose_nose', 'nose_tail'):
                for arm in agg:
                    cs = [V[k] for k in V if V[k]['arm'] == arm and V[k]['selection'] == 'auc' and V[k]['behaviour'] == b]
                    if cs:
                        sw = list(cs[0]['raw']['effects'])
                        L.append(f'- {b} {arm}: ' + '; '.join(
                            f"{s} {np.mean([c['raw']['effects'][s]['neuron']['dz'] for c in cs]):+.2f} "
                            f"(rate {cs[0]['raw']['effects'][s]['rate']['dz']:+.2f}) agree "
                            f"{sum(c['raw']['effects'][s]['sign_agree_rate'] for c in cs)}/{len(cs)}" for s in sw))
    # decision
    if all(d in S and S[d]['criteria'] for d in ('mice', 'ants')):
        cand = [a for a in ('M1', 'M5') if all(a in S[d]['criteria'] for d in ('mice', 'ants'))]
        best = max(cand, key=lambda a: np.mean([S[d]['criteria'][a]['mean_d_cf_auroc'] for d in ('mice', 'ants')]))
        keep = all(S[d]['criteria'][best]['no_drop'] and S[d]['criteria'][best]['any_gain'] for d in ('mice', 'ants'))
        S['decision'] = {'candidates': cand, 'best_motion_arm': best,
                         'mean_d_cf_auroc': {a: {d: S[d]['criteria'][a]['mean_d_cf_auroc'] for d in ('mice', 'ants')} for a in cand},
                         'per_domain': {d: {k: S[d]['criteria'][best][k] for k in ('no_drop', 'any_gain')} for d in ('mice', 'ants')},
                         'KEPT': bool(keep)}
        L.append(f"\n## Pre-registered decision\nbest motion arm (mean dAUROC over labels, both domains): {best} "
                 f"{json.dumps(S['decision']['mean_d_cf_auroc'])}; per domain {json.dumps(S['decision']['per_domain'])} "
                 f"-> {'KEPT' if keep else 'NOT KEPT'}")
    (OUT_ROOT / 'summary.json').write_text(json.dumps(S, indent=1, default=float))
    (OUT_ROOT / 'table.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['encode', 'train', 'evaluate', 'video', 'summary'])
    p.add_argument('--domain', default='mice', choices=['mice', 'ants'])
    p.add_argument('--out-dir', default=None)
    p.add_argument('--arms', nargs='+', default=['M1', 'M5'])
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--offset', type=int, default=-1)
    p.add_argument('--task', type=int, default=0)
    p.add_argument('--n-tasks', type=int, default=4)
    p.add_argument('--max-frames', type=int, default=0, help='encode: first N frames of the task only (smoke test)')
    p.add_argument('--n-check', type=int, default=256)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--num-workers', type=int, default=14)
    p.add_argument('--cap', type=int, default=0)
    p.add_argument('--steps', type=int, default=6000)
    p.add_argument('--pool-seed', type=int, default=0)
    args = p.parse_args()
    args.out_dir = str(OUT_ROOT / args.domain) if not args.out_dir else str(REPO / args.out_dir) \
        if not Path(args.out_dir).is_absolute() else args.out_dir
    {'encode': cmd_encode, 'train': cmd_train, 'evaluate': cmd_evaluate, 'video': cmd_video,
     'summary': cmd_summary}[args.cmd](args)


if __name__ == '__main__':
    main()
