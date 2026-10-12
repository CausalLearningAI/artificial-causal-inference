"""
Contact-token SAE (ECI, mice v1), steps 2-4 (GPU): train an unsupervised SAE only on the image patches WHERE TWO MICE
TOUCH, so contact becomes a dominant source of variation instead of 1 token in ~2000, and test whether one of its
neurons detects nose-nose / nose-tail contact better than the clean patch pipeline (C1 = 448 px tokens + 1 s motion,
results/vision/eci_t3_motion/mice M5).

Masks: the pre-registered combined splitter (mice_pairs_*.py), label-free automatic rule.
    train  30,000 frames of the 100 UNANNOTATED training videos (contact_sae_split.py), usable frames only
    eval   the 20,000-frame B subset (results/vision/eci_repr_diag/mice/b), its 16,775 usable frames, masks and
           geometry reused from results/vision/eci_mice_pairs/bsub/geom.npz
Patch -> mouse (= mice_pairs_probe.build_tokens): patch (r, c) of the 32 x 32 grid of the 448 px input belongs to the
mouse holding the most of its 16 x 16 frame pixels if that is >= 64 px, else none. A mouse's tokens = its STORED fg448
foreground patches (the store keeps foreground patches only). Tokens = the fg448 store tokens (DINOv2-base, 448 px) and
token_{t-5} at the same patch = prev.f16 of the fg448_fps1_d5 store (motion_t3.D5Matcher; same tokens C1 uses).
Contact tokens (no label used): for every unordered pair {i, j} whose patch masks are within 1 patch (3 x 3 dilation of
j's patch mask meets i's patch mask), A = i's stored patches within 1 patch of j's mask, B = j's within 1 of i's.
If both A and B hold >= 1 stored patch, the pair gives TWO ordered tokens:
    static (i, j)  [mean_A token_t ; mean_B token_t]                                                   1536-d
    motion (i, j)  [mean_A token_t ; mean_B token_t ; mean_A (token_t - token_{t-5}) ; mean_B (...)]  3072-d
and (j, i) with the halves swapped. Pairs that do not touch, or touch without stored patches on both sides, give none.
SAEs (the levers recipe): src/eci/sae.py Matryoshka BatchTopK, widths 1024 / 4096 (prefixes width/8 ... width), k = 16,
    Adam 5e-4, 500 warmup, cosine, 6000 steps x 4096 tokens, AuxK, b_dec = geometric median; TokenNorm one scalar scale
    (static) or one scale per 1536-d block (motion, as motion_t3's per-block norm); seeds 0, 1, 2; all train tokens.
Read-out: per frame, per neuron, the max over the frame's contact tokens; a frame without any contact token gets 0.
Evaluation (usable B-subset frames, weighted back to the natural base rate with the subset's inverse-sampling weights w):
    cross-fitted best neuron as sae_levers.score_model: annotated videos split into two halves by unit (pool-cage id,
    alternating over the sorted ids, = sae_levers half.npy), a neuron is selected on one half and scored on the other,
    both directions averaged. cf AUROC (AUROC-selected), cf AP (AP-selected), honest top-1% (precision in the top 1% of
    the weight mass, selected among neurons firing on >= 1% (weighted) of the selection half's frames). All metrics
    weighted by w (weighted Mann-Whitney with ties averaged; weighted step AP; top-1% ties pro rata).
    C1    the 3 M5 SAEs of results/vision/eci_t3_motion/mice/sae, frame-max over the frame's stored patches, the SAME
          frames and the same scorer.
    ceiling  the supervised linear pair score (mice_pairs_probe P2, out-of-fold scores, 3 seeds), the same halves and
          weighted metrics (pooled out-of-fold scores within each half), plus the probe's own fold-mean AUROC / AP.
    directionality  for the cross-fitted best nose_tail neuron (AUROC selection), on the test half: at the frame's
          argmax contact token (i, j), asym = (a_ij - a_ji) / (a_ij + a_ji) with a_ji the swapped token; median |asym|
          on nose_tail positives vs nose_nose positives vs negatives.
    sheets  seed 0 of every config, per label: neuron selected on half 0 (AUROC and honest top-1%), its top 16 frames
          of half 1 (<= 2 per video), crop around the argmax pair, A patches outlined green, B patches blue, tile border
          green = label positive, red = negative.
Pre-registered reading: "contact SAE clearly helps" iff, for nose_nose or nose_tail, cf AP (3-seed mean) >= 2x C1's on
    the same frames AND honest top-1% >= 0.25; "stretch" = honest top-1% >= 0.40. Also (AP - base) / (ceiling AP - base).

Steps:  tokens | train | eval   (scripts/eci/contact_sae.sh)
Outputs results/vision/eci_contact_sae/: tokens/, c1_codes/, sae/<config>_s<seed>/, eval/, sheets/, summary.json,
table.md
"""
import argparse
import io
import json
import os
import shutil
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import ndimage as ndi

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts/eci'))

OUT = REPO / 'results/vision/eci_contact_sae'
BSUB = REPO / 'results/vision/eci_mice_pairs/bsub'
BDIR = REPO / 'results/vision/eci_repr_diag/mice/b'
C1DIR = REPO / 'results/vision/eci_t3_motion/mice/sae'
D = 768
LABELS = ['nose_nose', 'nose_tail']
CONFIGS = {'static_w1024': ('static', 1024), 'static_w4096': ('static', 4096),
           'motion_w1024': ('motion', 1024), 'motion_w4096': ('motion', 4096)}
MIN_RATE, TOP = 0.01, 0.01
K3 = np.ones((3, 3), bool)
log = lambda s: print(time.strftime('%H:%M:%S'), s, flush=True)  # noqa: E731


def smoke_sel(n, m):
    return np.linspace(0, n - 1, m).astype(int)


def wd(args):
    return OUT / 'smoke' if args.smoke else OUT


# ---------------------------------------------------------------------------------------------- contact tokens
def patch_labels(cnt):
    """cnt (4, 32, 32) mouse pixels per patch -> (1024,) patch label 0 / 1..4 (>= 64 px majority rule)."""
    c = cnt.reshape(4, 1024).astype(np.int32)
    return np.where(c.max(0) >= 64, c.argmax(0) + 1, 0)


def frame_pairs(cnt, P):
    """-> list of (i, j, sel_A, sel_B) for touching pairs i < j with stored patches on both sides (masks over P),
    and the number of touching pairs without (dropped)."""
    lab = patch_labels(cnt)
    g = lab.reshape(32, 32)
    dil = [ndi.binary_dilation(g == m + 1, K3) for m in range(4)]
    own = [lab[P] == m + 1 for m in range(4)]
    near = [dil[m].ravel()[P] for m in range(4)]
    out, n_drop = [], 0
    for i in range(4):
        for j in range(i + 1, 4):
            if not (dil[j] & (g == i + 1)).any():
                continue
            a, b = own[i] & near[j], own[j] & near[i]
            if a.any() and b.any():
                out.append((i, j, a, b))
            else:
                n_drop += 1
    return out, n_drop


class Preloaded:
    """All tokens / positions / t-5 tokens of a set of store frames read once (sorted), served per frame chunk."""

    def __init__(self, idx, d5m, frames):
        self.f = np.sort(frames)
        self.tok, self.pos, self.lens = idx.load(self.f)
        self.prev, _, self.ok = d5m.fetch(self.f)
        self.st = np.r_[0, np.cumsum(self.lens)]

    def __call__(self, ff):
        a, b = np.searchsorted(self.f, ff[0]), np.searchsorted(self.f, ff[-1]) + 1
        assert np.array_equal(self.f[a:b], ff)
        lo, hi = self.st[a], self.st[b]
        return self.tok[lo:hi], self.pos[lo:hi], self.lens[a:b], self.prev[lo:hi]


def build_contact(idx, frames, cnts, d5m, chunk=1500, loader=None):
    """frames: store frame indices (any order), cnts aligned -> dict of arrays (tokens for the ordered pairs)."""
    order = np.argsort(frames)
    fs = frames[order]
    X, meta = [], []
    n_drop = n_touch = 0
    t0 = time.time()
    for c0 in range(0, len(fs), chunk):
        ff = fs[c0:c0 + chunk]
        if loader is None:
            tok, pos, lens = idx.load(ff)
            prev, _, _ = d5m.fetch(ff)
        else:
            tok, pos, lens, prev = loader(ff)
        st = np.r_[0, np.cumsum(lens)]
        for q, f in enumerate(ff):
            k = order[c0 + q]  # position in the caller's order
            T = tok[st[q]:st[q + 1]].astype(np.float32)
            Dl = T - prev[st[q]:st[q + 1]].astype(np.float32)
            P = pos[st[q]:st[q + 1]].astype(np.int64)
            prs, nd = frame_pairs(cnts[k], P)
            n_drop += nd
            n_touch += nd + len(prs)
            for i, j, a, b in prs:
                ta, tb, da, db = T[a].mean(0), T[b].mean(0), Dl[a].mean(0), Dl[b].mean(0)
                X.append(np.r_[ta, tb, da, db].astype(np.float16))
                X.append(np.r_[tb, ta, db, da].astype(np.float16))
                meta.append((k, i, j, int(a.sum()), int(b.sum())))
                meta.append((k, j, i, int(b.sum()), int(a.sum())))
        log(f'    {min(c0 + chunk, len(fs)):,}/{len(fs):,} frames, {len(X):,} tokens [{time.time() - t0:.0f}s]')
    M = np.array(meta, np.int64).reshape(-1, 5)
    return {'X': np.stack(X) if X else np.zeros((0, 4 * D), np.float16), 'frame': M[:, 0], 'i': M[:, 1], 'j': M[:, 2],
            'n_a': M[:, 3], 'n_b': M[:, 4], 'n_touching_pairs': n_touch, 'n_dropped_pairs': n_drop}


def b_frames(idx):
    """usable B-subset frames -> (subset DataFrame of usable frames, store frame indices, cnt)."""
    sub = pd.read_parquet(BDIR / 'subset.parquet')
    gz = np.load(BSUB / 'geom.npz')
    u = gz['idx']
    lab = sub.iloc[u].reset_index(drop=True)
    f_of_row = pd.Series(np.arange(len(idx.rows)), index=idx.rows)
    fr = f_of_row.loc[lab.row.values].values
    assert (idx.rows[fr] == lab.row.values).all() and (idx.nfg[fr] == lab.n_fg.values).all()
    return lab, fr, gz['cnt']


def cmd_tokens(args):
    import motion_t3 as mt
    import spatial_sae_pilot as ssp
    from src.eci.sae import load_sae
    dev = torch.device('cuda')
    out = OUT / ('smoke/tokens' if args.smoke else 'tokens')
    c1out = OUT / ('smoke/c1_codes' if args.smoke else 'c1_codes')
    out.mkdir(parents=True, exist_ok=True)
    c1out.mkdir(parents=True, exist_ok=True)
    idx = ssp.StoreIndex('mice')
    d5m = mt.D5Matcher(idx, 'mice')
    t0 = time.time()
    # ---- train
    T = pd.read_parquet(OUT / 'split/targets.parquet')
    shards = sorted((OUT / 'split').glob('shard_*of*.npz'))
    shards = [s for s in shards if ('smoke' in s.name) == args.smoke]
    n_sh = int(shards[0].stem.split('of')[1].split('_')[0])
    assert args.smoke or len(shards) == n_sh, f'{len(shards)}/{n_sh} split shards'
    ut, cnt, corr = [], [], []
    for s in shards:
        z = np.load(s)
        ut.append(z['u_tid']); cnt.append(z['cnt']); corr.append(z['correct'])
    ut, cnt, corr = np.concatenate(ut), np.concatenate(cnt), np.concatenate(corr)
    assert args.smoke or len(corr) == len(T)
    if args.max_frames:
        ut, cnt = ut[:args.max_frames], cnt[:args.max_frames]
    tf = T.frame.values[ut]
    log(f'train: {len(T):,} targets, {corr.mean():.4f} usable -> {len(ut):,} frames [{time.time() - t0:.0f}s]')
    R = build_contact(idx, tf, cnt, d5m)
    np.save(out / 'train_X.npy', R.pop('X'))
    np.savez(out / 'train_meta.npz', tid=ut[R['frame']], store_frame=tf[R['frame']], **R)
    info = {'train': {'n_targets': len(T), 'n_usable': int(len(ut)), 'frac_usable': float(corr.mean()),
                      'n_tokens': int(len(R['i'])), 'n_frames_with_token': int(len(np.unique(R['frame']))),
                      'n_touching_pairs': R['n_touching_pairs'], 'n_dropped_pairs': R['n_dropped_pairs']}}
    log(json.dumps(info))
    # ---- eval (B subset usable)
    lab, fr, cnt_b = b_frames(idx)
    if args.max_frames:
        k = smoke_sel(len(lab), args.max_frames)
        lab, fr, cnt_b = lab.iloc[k].reset_index(drop=True), fr[k], cnt_b[k]
    t1 = time.time()
    pre = Preloaded(idx, d5m, fr)
    log(f'  B frames: {len(pre.tok):,} tokens + t-5 tokens read in {time.time() - t1:.0f}s')
    R = build_contact(idx, fr, cnt_b, d5m, loader=pre)
    np.save(out / 'eval_X.npy', R.pop('X'))
    np.savez(out / 'eval_meta.npz', store_frame=fr[R['frame']], **R)
    nf = np.bincount(R['frame'], minlength=len(fr))
    info['eval'] = {'n_frames': len(fr), 'n_tokens': int(len(R['i'])), 'frac_frames_with_token': float((nf > 0).mean()),
                    'frac_frames_with_token_weighted': float(np.average(nf > 0, weights=lab.w.values)),
                    'tokens_per_frame_with_token': float(nf[nf > 0].mean()),
                    'n_touching_pairs': R['n_touching_pairs'], 'n_dropped_pairs': R['n_dropped_pairs']}
    for b in LABELS:
        y = lab[b].values.astype(bool)
        info['eval'][f'frac_pos_with_token_{b}'] = float((nf[y] > 0).mean())
    log(json.dumps(info['eval']))
    # ---- C1 M5 codes on the same frames (frame max over the stored patches)
    order = np.argsort(fr)
    tok, lens, prev = pre.tok, pre.lens, pre.prev
    info['eval']['c1_d5_unmatched_tokens'] = int((~pre.ok).sum())
    for seed in (0, 1, 2):
        sae, norm, _ = load_sae(C1DIR / f'M5_s{seed}' / 'sae.pt', dev)
        codes, st = mt.encode_codes(sae, norm, 'M5', tok, {-5: prev}, lens, dev)
        c = np.empty((len(fr), sae.n_latents), np.float16)
        c[order] = codes.cpu().numpy()
        np.save(c1out / f'M5_s{seed}.npy', c)
        info[f'c1_M5_s{seed}'] = {k: v for k, v in st.items() if k != 'token_fire_rate'}
        log(f'  C1 M5 s{seed}: FVE {st["fve"]:.4f} on {st["n_tokens"]:,} tokens')
        del codes
        torch.cuda.empty_cache()
    info['wall_s'] = round(time.time() - t0, 1)
    info['max_frames'] = args.max_frames
    (out / 'tokens.json').write_text(json.dumps(info, indent=1))


# ---------------------------------------------------------------------------------------------- train
def train_one(X, blocks, width, seed, steps, out, extra, dev, bs=4096, lr=5e-4, warmup=500):
    from src.eci.sae import MatryoshkaBatchTopKSAE, TokenNorm, _train_step, geometric_median, save_checkpoint
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    g = np.random.default_rng(seed)
    N, d = X.shape
    ii = torch.from_numpy(np.sort(g.choice(N, min(500_000, N), replace=False))).to(dev)
    init = X[ii].float()
    norm = TokenNorm(d).to(dev)
    norm = norm.fit(init, seed=seed) if len(blocks) == 1 else norm.fit_blocks(init, blocks, seed=seed)
    sae = MatryoshkaBatchTopKSAE(d_in=d, n_latents=width, prefixes=(width // 8, width // 4, width // 2, width), k=16,
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
        logs, _ = _train_step(sae, opt, norm(X[b]), step, steps, lr, warmup, 1.0)
        if step % 1000 == 0 or step == steps - 1:
            logs.update(step=step, elapsed_s=round(time.time() - t0, 1))
            history.append(logs)
            log(f"    step {step}/{steps} fve {logs['fve']:.4f} l0 {logs['l0']:.1f} dead {logs['n_dead']}")
    sae.eval()
    dead = int((sae.tokens_since_fired >= sae.dead_tokens).sum())
    save_checkpoint(out / 'sae.pt', sae, norm, extra={**extra, 'seed': seed})
    (out / 'train.json').write_text(json.dumps({**extra, 'seed': seed, 'n_pool': N, 'steps': steps, 'blocks': blocks,
                                                'token_presentations': steps * bs, 'epochs': steps * bs / N,
                                                'train_time_s': time.time() - t0, 'dead_at_end_train': dead,
                                                'dead_frac_train': dead / width, 'gpu': torch.cuda.get_device_name(0),
                                                'history': history}, indent=1))
    log(f'    trained in {time.time() - t0:.0f}s, final fve {history[-1]["fve"]:.4f}, dead {dead}/{width}')


def cmd_train(args):
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = True
    Xall = torch.from_numpy(np.load(wd(args) / 'tokens' / 'train_X.npy'))
    log(f'train contact tokens {tuple(Xall.shape)}')
    for cfg in args.configs:
        kind, width = CONFIGS[cfg]
        X = (Xall[:, :2 * D] if kind == 'static' else Xall).to(dev)
        blocks = [2 * D] if kind == 'static' else [2 * D, 2 * D]
        for seed in args.seeds:
            o = wd(args) / 'sae' / f'{cfg}_s{seed}'
            if (o / 'sae.pt').exists() and not args.overwrite:
                log(f'[SKIP] {o}')
                continue
            log(f'{cfg} seed {seed}: input {tuple(X.shape)}, blocks {blocks}')
            train_one(X, blocks, width, seed, args.steps, o, {'config': cfg, 'kind': kind, 'width': width}, dev)
            torch.cuda.empty_cache()
        del X
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------------------------- weighted scoring
NB = 0x7C01


@torch.no_grad()
def align_w(X, Y, w, rows, block=512):
    """Weighted per-column AUROC / AP / top-1% precision / firing rate of non-negative fp16 codes X (n, m) against
    bool labels Y (n, L) on `rows`, weights w (n,). Same construction as src/eci/levers.align_gpu (fp16 bit-pattern
    histograms, ties averaged for AUROC, step AP with the zero group last, top-1% ties pro rata) with every count
    replaced by a sum of weights; the top 1% = the highest frames holding 1% of the weight mass."""
    dev = X.device
    Y, w = Y[rows], w[rows].double()
    n, L = Y.shape
    m = X.shape[1]
    W = w.sum()
    P = (Y.double() * w[:, None]).sum(0)
    N = W - P
    k_top = TOP * W
    auc = torch.full((m, L), float('nan'), dtype=torch.float64, device=dev)
    ap, ptop = auc.clone(), auc.clone()
    rate = torch.zeros(m, dtype=torch.float64, device=dev)
    for j0 in range(0, m, block):
        b = min(block, m - j0)
        xb = X[rows, j0:j0 + b]
        bits = (xb.contiguous().view(torch.int16).long() & 0x7FFF) + NB * torch.arange(b, device=dev)[None]
        wb = w[:, None].expand(n, b)
        tot = torch.bincount(bits.flatten(), weights=wb.flatten(), minlength=b * NB).view(b, NB)
        rate[j0:j0 + b] = 1 - tot[:, 0] / W
        tot_d = tot.flip(1)
        ccnt = tot_d.cumsum(1)
        nnz = W - tot[:, 0]
        k = torch.minimum(torch.full_like(nnz, float(k_top)), nnz)
        ib = torch.searchsorted(ccnt.contiguous(), k[:, None] - 1e-9).clamp_max(NB - 1)
        for li in range(L):
            yl = Y[:, li]
            pos = torch.bincount(bits[yl].flatten(), weights=wb[yl].flatten(), minlength=b * NB).view(b, NB).flip(1)
            neg = tot_d - pos
            ctp, cfp = pos.cumsum(1), neg.cumsum(1)
            auc[j0:j0 + b, li] = (pos * (N[li] - cfp) + 0.5 * pos * neg).sum(1) / (P[li] * N[li])
            prec = torch.where(ctp + cfp > 0, ctp / (ctp + cfp).clamp_min(1e-12), torch.zeros_like(ctp))
            ap[j0:j0 + b, li] = (pos / P[li] * prec).sum(1)
            cb = ccnt.gather(1, ib)[:, 0]
            tb = tot_d.gather(1, ib)[:, 0]
            pb = pos.gather(1, ib)[:, 0]
            tp = ctp.gather(1, ib)[:, 0] - pb + pb * (k - (cb - tb)) / tb.clamp_min(1e-12)
            pt = tp / k.clamp_min(1e-12)
            ptop[j0:j0 + b, li] = torch.where(k > 0, pt, torch.full_like(pt, float('nan')))
    return {'auc': auc.cpu().numpy(), 'ap': ap.cpu().numpy(), 'prec_top': ptop.cpu().numpy(),
            'rate': rate.cpu().numpy(), 'base': (P / W).cpu().numpy(), 'n': n}


def cross_fit(codes, Yt, wt, half, dev):
    """codes (F, m) fp16 GPU -> per label the cross-fitted selections (sae_levers.score_model, weighted)."""
    R = {h: align_w(codes, Yt, wt, torch.from_numpy(np.flatnonzero(half == h)).to(dev)) for h in (0, 1)}
    res = {}
    for k, b in enumerate(LABELS):
        dirs = []
        for s, t in ((0, 1), (1, 0)):
            S, T = R[s], R[t]
            e = {}
            for sel, q, honest in (('auc', 'auc', False), ('ap', 'ap', False), ('top1', 'prec_top', True)):
                sc = S[q][:, k].copy()
                if honest:
                    sc[S['rate'] < MIN_RATE] = -np.inf
                sc[~np.isfinite(sc)] = -np.inf
                j = int(np.argmax(sc))
                e[sel] = {'neuron': j, 'select_score': float(sc[j]), 'test_auc': float(T['auc'][j, k]),
                          'test_ap': float(T['ap'][j, k]), 'test_top1': float(T['prec_top'][j, k]),
                          'test_rate': float(T['rate'][j])}
            dirs.append(e)
        res[b] = {'base_half': [float(R[0]['base'][k]), float(R[1]['base'][k])],
                  'cf_auroc': float(np.mean([e['auc']['test_auc'] for e in dirs])),
                  'cf_ap_at_auroc': float(np.mean([e['auc']['test_ap'] for e in dirs])),
                  'cf_ap': float(np.mean([e['ap']['test_ap'] for e in dirs])),
                  'cf_top1': float(np.mean([e['top1']['test_top1'] for e in dirs])),
                  'cf_top1_rate': float(np.mean([e['top1']['test_rate'] for e in dirs])), 'dirs': dirs}
    return res


def wmetrics(s, y, w, cap_nonzero=False):
    """Weighted AUROC (ties averaged) / step AP / top-1% precision (1% of the weight mass, ties pro rata) of one score
    vector, numpy (reference implementation; cap_nonzero: the top set is capped at the frames with s > 0, as align_w)."""
    o = np.argsort(-s, kind='stable')
    s, y, w = s[o], y[o].astype(bool), w[o].astype(np.float64)
    _, start = np.unique(-s, return_index=True)
    wp, wn = np.add.reduceat(w * y, start), np.add.reduceat(w * ~y, start)
    P, N = wp.sum(), wn.sum()
    ctp, cfp = np.cumsum(wp), np.cumsum(wn)
    auc = float((wp * (N - cfp) + 0.5 * wp * wn).sum() / (P * N))
    ap = float((wp / P * ctp / (ctp + cfp)).sum())
    tot = wp + wn
    cc = np.cumsum(tot)
    k = TOP * (P + N)
    if cap_nonzero:
        k = min(k, float(w[s > 0].sum()))
    if k <= 0:
        return auc, ap, float('nan')
    ib = int(np.searchsorted(cc, k - 1e-9))
    tp = ctp[ib] - wp[ib] + wp[ib] * (k - (cc[ib] - tot[ib])) / tot[ib]
    return auc, ap, float(tp / k)


def score_vec(s, y, w, half):
    """A single score (supervised ceiling): weighted AUROC / AP / top-1% per half, averaged over the two halves."""
    m = np.array([wmetrics(s[half == h], y[half == h], w[half == h]) for h in (0, 1)])
    return {'auc': m[:, 0].mean(), 'ap': m[:, 1].mean(), 'prec_top': m[:, 2].mean(), 'per_half': m.tolist()}


# ---------------------------------------------------------------------------------------------- eval
@torch.no_grad()
def frame_max(sae, norm, X, frame, n_frames, dev, chunk=65536):
    """contact-token codes -> frame-max codes (n_frames, m) fp16 GPU, token argmax per frame, FVE, L0."""
    m = sae.n_latents
    codes = torch.zeros(n_frames, m, dtype=torch.float16, device=dev)
    s1 = s2 = sse = 0
    l0 = 0
    for a in range(0, len(X), chunk):
        x = norm(X[a:a + chunk].to(dev))
        z = sae.encode(x, mode='threshold')
        xh = sae.decode(z)
        s1 = s1 + x.double().sum(0); s2 = s2 + x.double().pow(2).sum(0); sse = sse + (x - xh).double().pow(2).sum(0)
        l0 += float((z > 0).sum())
        fr = torch.from_numpy(frame[a:a + chunk]).to(dev)
        mx = torch.zeros(n_frames, m, device=dev)
        mx.index_reduce_(0, fr, z, 'amax', include_self=True)
        codes = torch.maximum(codes, mx.half())
    tss = s2 - s1.pow(2) / len(X)
    return codes, {'fve': float(1 - sse.sum() / tss.sum()), 'l0': l0 / len(X)}


@torch.no_grad()
def neuron_acts(sae, norm, X, j, dev, chunk=65536):
    out = []
    for a in range(0, len(X), chunk):
        z = sae.encode(norm(X[a:a + chunk].to(dev)), mode='threshold')
        out.append(z[:, j].float().cpu().numpy())
    return np.concatenate(out)


def video_units(obs):
    unit = pd.Series(obs).str.rsplit('_', n=2).str[0].values
    import spatial_sae_pilot as ssp
    lab_all, _ = ssp.labelled_frames('mice')
    units = sorted(set(pd.Series(lab_all.obs.unique()).str.rsplit('_', n=2).str[0]))
    hmap = {u: i % 2 for i, u in enumerate(units)}
    return np.array([hmap[u] for u in unit])


def decode_selected(paths, dst, workers=8):
    """frame JPG paths (dataset-relative) -> {path: JPG bytes}, decoded from each video's standardised 5 fps mp4 on
    local disk (get_frames.py's filter chain fps=5,format=rgb24 + a select of the wanted frame numbers, -q:v 2); the
    first frame is compared with the dataset JPG (max abs pixel difference)."""
    import subprocess
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image
    by = {}
    for p in paths:
        by.setdefault(Path(p).parent.name, []).append(p)

    def one(item):
        vdir, ps = item
        ns = sorted(int(Path(p).stem.split('_')[1]) for p in ps)
        od = dst / vdir
        od.mkdir(parents=True, exist_ok=True)
        sel = '+'.join(f'eq(n\\,{n})' for n in ns)
        subprocess.run(['ffmpeg', '-nostdin', '-loglevel', 'error', '-y', '-i',
                        str(REPO / f'data/mice/v1/observations/full/{vdir}.mp4'), '-vf',
                        f'fps=5,format=rgb24,select={sel}', '-vsync', '0', '-start_number', '0',
                        '-q:v', '2', str(od / 'sel_%04d.jpg')], check=True)
        got = sorted(od.glob('sel_*.jpg'))
        assert len(got) == len(ns), (vdir, len(got), len(ns))
        return {str(Path(ps[0]).parent / f'frame_{n:06d}.jpg'): g.read_bytes() for n, g in zip(ns, got)}
    out = {}
    with ThreadPoolExecutor(workers) as ex:
        for d in ex.map(one, by.items()):
            out.update(d)
    p0 = paths[0]
    a = np.asarray(Image.open(REPO / 'dataset' / p0).convert('RGB'), np.int16)
    b = np.asarray(Image.open(io.BytesIO(out[p0])).convert('RGB'), np.int16)
    shutil.rmtree(dst, ignore_errors=True)
    return out, {'checked': p0, 'max_abs_px': int(np.abs(a - b).max()), 'n_frames': len(out), 'n_videos': len(by)}


def contact_sheet(path, picks, lab, meta, a_tok, X_frames, cnt_b, P_of, frames_jpg, title, label):
    from PIL import Image, ImageDraw
    th = 256
    n = len(picks)
    rows_ = (n + 3) // 4
    sheet = Image.new('RGB', (4 * th, rows_ * (th + 14) + 18), 'white')
    dr = ImageDraw.Draw(sheet)
    dr.text((4, 2), title, fill='black')
    for q, (f, t) in enumerate(picks):
        im = Image.open(io.BytesIO(frames_jpg[lab.frame_path.iat[f]])).convert('RGB')
        i, j = int(meta['i'][t]), int(meta['j'][t])
        P = P_of(f)
        prs, _ = frame_pairs(cnt_b[f], P)
        A = B = np.zeros(0, int)
        for pi, pj, a, b in prs:
            if (pi, pj) == (i, j):
                A, B = P[a], P[b]
            elif (pi, pj) == (j, i):
                A, B = P[b], P[a]
        lab_ = patch_labels(cnt_b[f]).reshape(32, 32)
        yy, xx = np.nonzero((lab_ == i + 1) | (lab_ == j + 1))
        cy, cx = (yy.mean() + 0.5) * 16, (xx.mean() + 0.5) * 16
        side = max(192, int(16 * (max(np.ptp(yy), np.ptp(xx)) + 3)))
        x0 = int(np.clip(cx - side / 2, 0, 512 - side)); y0 = int(np.clip(cy - side / 2, 0, 512 - side))
        d2 = ImageDraw.Draw(im)
        for pp, col in ((A, (0, 230, 0)), (B, (40, 120, 255))):
            for p in pp:
                r_, c_ = divmod(int(p), 32)
                d2.rectangle([c_ * 16, r_ * 16, c_ * 16 + 15, r_ * 16 + 15], outline=col, width=1)
        im = im.crop((x0, y0, x0 + side, y0 + side)).resize((th - 6, th - 6))
        yv = bool(lab[label].iat[f])
        tile = Image.new('RGB', (th, th), (0, 200, 0) if yv else (220, 0, 0))
        tile.paste(im, (3, 3))
        rr, cc = divmod(q, 4)
        sheet.paste(tile, (cc * th, 18 + rr * (th + 14)))
        dr.text((cc * th + 2, 18 + rr * (th + 14) + th), f'{lab.obs.iat[f][:22]} r{lab.row.iat[f]} '
                f'{"POS" if yv else "neg"} a={a_tok[t]:.2f}', fill='black')
    sheet.save(path, quality=88)


def cmd_eval(args):
    from src.eci.sae import load_sae
    import spatial_sae_pilot as ssp
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False
    t0 = time.time()
    idx = ssp.StoreIndex('mice')
    lab, fr, cnt_b = b_frames(idx)
    ksel = np.arange(len(lab))
    if args.smoke:
        ksel = smoke_sel(len(lab), args.max_frames)
        lab, fr, cnt_b = lab.iloc[ksel].reset_index(drop=True), fr[ksel], cnt_b[ksel]
    F = len(lab)
    meta = dict(np.load(wd(args) / 'tokens' / 'eval_meta.npz'))
    X = torch.from_numpy(np.load(wd(args) / 'tokens' / 'eval_X.npy'))
    half = video_units(lab.obs.values)
    Y = lab[LABELS].values.astype(bool)
    Yt, wt = torch.from_numpy(Y).to(dev), torch.from_numpy(lab.w.values).to(dev)
    base = {b: float(np.average(Y[:, k], weights=lab.w.values)) for k, b in enumerate(LABELS)}
    _, ev, _, _, _, _ = idx.split('mice', 100, 0)
    ref_half = np.load(REPO / 'results/vision/eci_sae_levers/mice/half.npy')
    pe = pd.Series(np.arange(len(ev)), index=ev).loc[fr].values
    assert np.array_equal(ref_half[pe], half), 'halves differ from sae_levers half.npy'
    log(f'{F:,} usable B frames, {len(X):,} eval contact tokens, halves {np.bincount(half).tolist()}, base {base}')
    (wd(args) / 'eval').mkdir(exist_ok=True)
    allres = {'base_rate': base, 'n_frames': F, 'halves': np.bincount(half).tolist()}
    # ---- C1
    for seed in (0, 1, 2):
        c = torch.from_numpy(np.load(wd(args) / 'c1_codes' / f'M5_s{seed}.npy')).to(dev)
        allres[f'C1_M5_s{seed}'] = cross_fit(c, Yt, wt, half, dev)
        log(f'  C1 M5 s{seed}: ' + ', '.join(f"{b} AUROC {r['cf_auroc']:.3f} AP {r['cf_ap']:.4f} top1 {r['cf_top1']:.3f}"
                                            for b, r in allres[f'C1_M5_s{seed}'].items()))
        del c
    # ---- ceiling (P2 out-of-fold)
    oof = pd.read_parquet(BSUB / 'probe' / 'oof.parquet')
    pj = json.loads((BSUB / 'probe' / 'results.json').read_text())
    for b in LABELS:
        k = LABELS.index(b)
        for seed in (0, 1, 2):
            o = oof[(oof.probe == 'P2') & (oof.behaviour == b) & (oof.seed == seed) & oof.u.isin(ksel)].sort_values('u')
            assert len(o) == F and (o.u.values == ksel).all() and (o.row.values == lab.row.values).all()
            m = score_vec(o.score.values, Y[:, k], lab.w.values, half)
            allres.setdefault('ceiling_P2', {}).setdefault(b, []).append(
                {'auroc': float(m['auc']), 'ap': float(m['ap']), 'top1': float(m['prec_top']), 'per_half': m['per_half']})
        sm = [s for s in pj['summary'] if s['probe'] == 'P2' and s['behaviour'] == b][0]
        allres.setdefault('ceiling_P2_foldmean', {})[b] = {k_: sm[k_] for k_ in sm if k_ in ('auroc', 'ap', 'auroc_sd',
                                                                                             'ap_sd')}
    log(f"  ceiling P2: {json.dumps(allres['ceiling_P2'])}")
    # ---- contact SAEs
    sheets_todo = []
    for cfg in args.configs:
        kind, width = CONFIGS[cfg]
        Xc = X[:, :2 * D] if kind == 'static' else X
        for seed in args.seeds:
            key = f'{cfg}_s{seed}'
            sae, norm, _ = load_sae(wd(args) / 'sae' / key / 'sae.pt', dev)
            codes, st = frame_max(sae, norm, Xc, meta['frame'], F, dev)
            r = cross_fit(codes, Yt, wt, half, dev)
            # check the GPU weighted scorer against the numpy reference on the selected neurons
            chk = []
            for k_, b in enumerate(LABELS):
                for di, e in enumerate(r[b]['dirs']):
                    for sel in ('auc', 'ap', 'top1'):
                        j = e[sel]['neuron']
                        th = half == 1 - di
                        ref = wmetrics(codes[:, j].float().cpu().numpy()[th], Y[th, k_], lab.w.values[th], True)
                        chk.append(max(abs(ref[0] - e[sel]['test_auc']), abs(ref[1] - e[sel]['test_ap']),
                                       abs(ref[2] - e[sel]['test_top1'])))
            r['scorer_check_max_abs_diff'] = float(max(chk))
            # directionality at the cf AUROC nose_tail neuron (per direction)
            dirn = []
            for di, e in enumerate(r['nose_tail']['dirs']):
                test_h = 1 - di  # dirs[0]: selected on half 0, tested on half 1
                j = e['auc']['neuron']
                a = neuron_acts(sae, norm, Xc, j, dev)
                # partner token: same frame, swapped (i, j)
                key_ = meta['frame'] * 16 + meta['i'] * 4 + meta['j']
                sw = pd.Series(np.arange(len(key_)), index=key_).loc[meta['frame'] * 16 + meta['j'] * 4 + meta['i']].values
                ins = half[meta['frame']] == test_h
                best_t = pd.DataFrame({'f': meta['frame'][ins], 'a': a[ins]}, index=np.flatnonzero(ins)).groupby('f').a.idxmax()
                ff = best_t.index.values
                tt = best_t.values
                ai, aj = a[tt], a[sw[tt]]
                ok_ = (ai + aj) > 0
                asym = np.where(ok_, (ai - aj) / np.maximum(ai + aj, 1e-9), np.nan)
                dd = {}
                for nm, msk in (('nose_tail_pos', Y[ff, 1]), ('nose_nose_pos', Y[ff, 0]),
                                ('neg', ~Y[ff, 0] & ~Y[ff, 1])):
                    v = np.abs(asym[msk & ok_])
                    dd[nm] = {'median_abs_asym': float(np.median(v)) if len(v) else None, 'n': int(len(v)),
                              'frac_one_sided': float((v > 0.99).mean()) if len(v) else None}
                dirn.append({'neuron': j, 'test_half': test_h, **dd})
            r['nose_tail']['directionality'] = dirn
            allres[key] = {'fve': st['fve'], 'l0': st['l0'], 'labels': r}
            log(f'  {key}: FVE {st["fve"]:.4f} L0 {st["l0"]:.1f}; ' + ', '.join(
                f"{b} AUROC {r[b]['cf_auroc']:.3f} AP {r[b]['cf_ap']:.4f} top1 {r[b]['cf_top1']:.3f}" for b in LABELS)
                + f"; scorer check {r['scorer_check_max_abs_diff']:.2e}")
            if seed == args.seeds[0]:
                for b in LABELS:
                    for sel in ('auc', 'top1'):
                        e = r[b]['dirs'][0]  # selected on half 0, shown on half 1
                        j = e[sel]['neuron']
                        a = neuron_acts(sae, norm, Xc, j, dev)
                        sheets_todo.append((key, b, sel, j, e[sel], a))
            del codes
            torch.cuda.empty_cache()
    (wd(args) / 'eval' / 'results.json').write_text(json.dumps(allres, indent=1))
    # ---- sheets
    if sheets_todo and not args.no_sheets:
        sd = wd(args) / 'sheets'
        sd.mkdir(exist_ok=True)
        stage = Path(os.environ.get('STAGE', '/tmp'))
        tgt = pd.read_parquet(BSUB / 'targets.parquet')
        lab['frame_path'] = tgt.frame_path.values[np.load(BSUB / 'geom.npz')['idx']][ksel]
        need = set()
        picks_all = []
        for key, b, sel, j, e, a in sheets_todo:
            dfa = pd.DataFrame({'f': meta['frame'], 't': np.arange(len(a)), 'a': a})
            dfa = dfa[(half[dfa.f.values] == 1) & (dfa.a > 0)]
            top = dfa.sort_values('a', ascending=False).drop_duplicates('f')
            top['obs'] = lab.obs.values[top.f.values]
            top = top.groupby('obs', sort=False).head(2).head(16)
            picks = list(zip(top.f.values, top.t.values))
            need.update(lab.frame_path.values[top.f.values])
            picks_all.append((key, b, sel, j, e, a, picks))
        frames_jpg, dchk = decode_selected(sorted(need), stage / 'sheet_frames')
        pos_all = {}
        order = np.argsort(fr)

        def P_of(f):
            if f not in pos_all:
                _, p_, _ = idx.load(np.array([fr[f]]))
                pos_all[f] = p_.astype(np.int64)
            return pos_all[f]
        index = []
        for key, b, sel, j, e, a, picks in picks_all:
            n_pos = int(sum(Y[f, LABELS.index(b)] for f, _ in picks))
            title = (f'{key} neuron {j} ({sel}-selected on half 0 for {b}; test half 1: AUROC {e["test_auc"]:.3f} '
                     f'AP {e["test_ap"]:.3f} top1 {e["test_top1"]:.3f}); top {len(picks)} half-1 frames, {n_pos} {b} '
                     f'positive; green = A patches (token half 1), blue = B')
            contact_sheet(sd / f'{key}_{b}_{sel}_n{j}.jpg', picks, lab, meta, a, None, cnt_b, P_of, frames_jpg, title, b)
            other = [x for x in LABELS if x != b][0]
            index.append({'sheet': f'{key}_{b}_{sel}_n{j}.jpg', 'neuron': j, 'label': b, 'n_pos': n_pos,
                          f'n_pos_{other}': int(sum(Y[f, LABELS.index(other)] for f, _ in picks)),
                          'n_tiles': len(picks), 'obs': [lab.obs.iat[f] for f, _ in picks],
                          'row': [int(lab.row.iat[f]) for f, _ in picks]})
        (sd / 'index.json').write_text(json.dumps({'decode_check': dchk, 'sheets': index}, indent=1))
        log(f'sheets -> {sd}')
    log(f'done [{time.time() - t0:.0f}s]')


# ---------------------------------------------------------------------------------------------- summary
def cmd_summary(args):
    R = json.loads((wd(args) / 'eval' / 'results.json').read_text())
    base = R['base_rate']
    ms = lambda v: f'{np.mean(v):.3f} +- {np.std(v):.3f}'  # noqa: E731
    msa = lambda v: f'{np.mean(v):.4f} +- {np.std(v):.4f}'  # noqa: E731
    rows, summ = [], {'base_rate': base, 'configs': {}}
    lines = ['# Contact-token SAE vs C1 patch SAE vs supervised ceiling (usable B-subset frames, weighted)', '',
             f"{R['n_frames']:,} frames; base rate nose_nose {base['nose_nose']:.4f}, nose_tail {base['nose_tail']:.4f}",
             '', '| label | config | cf AUROC | cf AP | honest top-1% | AP / C1 AP | frac of ceiling AP above base |',
             '|---|---|---|---|---|---|---|']
    c1 = {b: [R[f'C1_M5_s{s}'][b] for s in range(3)] for b in LABELS}
    ceil = R['ceiling_P2']
    for b in LABELS:
        c1ap = np.mean([x['cf_ap'] for x in c1[b]])
        cap = np.mean([x['ap'] for x in ceil[b]])
        for name in ['C1_M5'] + [c for c in CONFIGS]:
            if name == 'C1_M5':
                rr = c1[b]
            else:
                ks = [k for k in R if k.startswith(name + '_s')]
                if not ks:
                    continue
                rr = [R[k]['labels'][b] for k in ks]
            au, ap, t1 = [x['cf_auroc'] for x in rr], [x['cf_ap'] for x in rr], [x['cf_top1'] for x in rr]
            frac = (np.mean(ap) - base[b]) / (cap - base[b])
            lines.append(f'| {b} | {name} | {ms(au)} | {msa(ap)} | {ms(t1)} | {np.mean(ap) / c1ap:.2f} | {frac:.2f} |')
            summ['configs'].setdefault(name, {})[b] = {'cf_auroc': au, 'cf_ap': ap, 'cf_top1': t1,
                                                       'ap_over_c1': float(np.mean(ap) / c1ap), 'frac_ceiling': frac}
        au, ap, t1 = [x['auroc'] for x in ceil[b]], [x['ap'] for x in ceil[b]], [x['top1'] for x in ceil[b]]
        lines.append(f'| {b} | ceiling P2 (supervised) | {ms(au)} | {msa(ap)} | {ms(t1)} | {np.mean(ap) / c1ap:.2f} | 1.00 |')
        summ['ceiling'] = summ.get('ceiling', {})
        summ['ceiling'][b] = {'auroc': au, 'ap': ap, 'top1': t1, 'foldmean': R['ceiling_P2_foldmean'][b]}
    verdict = {}
    for name in CONFIGS:
        if name not in summ['configs']:
            continue
        v = {}
        for b in LABELS:
            s = summ['configs'][name][b]
            v[b] = {'ap_2x': s['ap_over_c1'] >= 2, 'top1_ge_0.25': float(np.mean(s['cf_top1'])) >= 0.25,
                    'stretch_top1_ge_0.40': float(np.mean(s['cf_top1'])) >= 0.40}
            v[b]['helps'] = v[b]['ap_2x'] and v[b]['top1_ge_0.25']
        v['clearly_helps'] = any(v[b]['helps'] for b in LABELS)
        verdict[name] = v
    summ['verdict'] = verdict
    lines += ['', 'Pre-registered: clearly helps iff (cf AP >= 2x C1 AND honest top-1% >= 0.25) for nose_nose or nose_tail; '
              'stretch top-1% >= 0.40', '', '```', json.dumps(verdict, indent=1), '```']
    lines += ['', 'Directionality (cf AUROC nose_tail neuron, test half; median |asym| at the frame argmax token):']
    for k in sorted(R):
        if isinstance(R[k], dict) and 'labels' in R[k]:
            lines.append(f"- {k}: " + json.dumps(R[k]['labels']['nose_tail'].get('directionality')))
    (wd(args) / 'table.md').write_text('\n'.join(lines) + '\n')
    (wd(args) / 'summary.json').write_text(json.dumps(summ, indent=1))
    print('\n'.join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['tokens', 'train', 'eval', 'summary'])
    ap.add_argument('--configs', nargs='+', default=list(CONFIGS))
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--max-frames', type=int, default=0, help='tokens smoke test')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--smoke', action='store_true', help='use the smoke split shard / OUT/smoke for every step')
    ap.add_argument('--no-sheets', action='store_true')
    args = ap.parse_args()
    {'tokens': cmd_tokens, 'train': cmd_train, 'eval': cmd_eval, 'summary': cmd_summary}[args.step](args)


if __name__ == '__main__':
    main()
