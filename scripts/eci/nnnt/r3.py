"""
NN/NT round 3 (2026-10-19): SPECIFICITY. Round 1's best nose_nose neuron (A3 n315) fires on generic body-body contact
between two mice (many unlabelled top frames). Round 3 tests label-free changes aimed at separating head contacts
(nose-nose, nose-tail-base) from generic contact. Labels are used only to score / select recipes, never in training.

PRE-REGISTERED 2026-10-19, written before any round-3 score and before the round-2 S3-S7 / T-max / 2048 scores were
seen (only round-2 S0-S2 scores were known: primary R~ means 0.103 / 0.054 / 0.111).

INPUT: the round-2 1024 px A3 token stream (DINOv3-L block 20, odour rotation, tight mask >= 25 % animal pixels per
16 x 16 patch, no change channel). Train pool = round-2 pool (bursts, all 432 videos, 'train' videos; theta* on
'calib', FVE on 'eval' videos); scoring = the round-2 diagnosis frames (86,400 = every 10th 5 fps frame of the 144
annotated videos, disjoint from the pool), the round-1 scorer (nnnt.score_set) on the same 10 random pool splits.

LABEL-FREE MASK GEOMETRY (step 'geom'; on the 64 x 64 grid of kept tokens of each frame, zero padded)
    dist   chessboard distance of a kept token to the nearest non-kept cell (1 = on the mask boundary = EDGE token)
    n_nb   kept tokens among the 8 neighbours (tip of a blob ~ 3, straight edge ~ 5, interior 8)
    comp   8-connected component size; components < 5 tokens ignored for counting
    single-mouse area A1(video) = median over that video's frames with exactly 4 components (>= 5 tokens each; mice v1 =
    4 mice per arena) of the largest component (~ one whole mouse); fallback the global median if < 20 such frames.
    CONTACT frame (label-free proxy for 'two mice touching') = largest component >= 1.5 x A1(video).

ARMS (base = round-2 S2 recipe: Matryoshka BatchTopK 2048 (128/512/2048), k 16, 30,000 steps x 4096, lr 5e-4; if the
round-2 rule KEEPS a different winner the base width / k / centring follow it - recorded in decisions before scoring)
    R1  S2 checkpoints unchanged, EDGE READOUT: frame value = max over edge tokens only (snouts and tail bases lie on the
        mask boundary; a seam between two touching black bodies is interior).
    R2  SAE trained on edge tokens only (train / calib / eval all edge tokens), edge readout.
    R3  ZCA-whitened tokens (train-pool mean and covariance from 1M random train tokens, eps = 0.05 x mean eigenvalue),
        removes the dominance of high-variance directions (the frequent 'contact' / 'mouse body' directions); readout
        max over all kept tokens.
    R4  cluster-balanced sampling: spherical k-means (K = 1024, 20 iterations, 1M train tokens) on normalised tokens;
        SAE batches drawn with probability ~ cluster_size^-0.5 per token (rare token types get more mass).
    R5  context tokens: x = [z_t, mean of the frame's kept tokens in the 5 x 5 window around t, t excluded] (d_in 2048,
        per-block TokenNorm) so an atom can encode 'head next to a head' vs 'body next to a body'.
    R6  wider Matryoshka: 32,768 latents (128/512/2048/8192/32768), k 16, 30,000 steps (rare concepts get atoms);
        seeds 0-2 (3-5 only if it wins).
    R7  R5 context tokens, trained and read out on edge tokens only (R2 + R5).
    refs scored by this same code (pipeline check vs round 2's sweep.json): S0 (round-1 A3 recipe) and S2, seeds 0-5.
SEEDS 0-5 for R2-R5, R7 (R1 reuses S2 seeds 0-5); identical splits for every arm.

READOUTS / METRICS
    primary = mean(R~ nose_nose, R~ nose_tail), mean of 10 random pool splits (round 1/2), cf AUROC.
    SPECIFICITY (secondary, the aim of this round), for the neuron selected on one pool half, on the held-out half:
        cR~        R~ restricted to CONTACT frames (does it find nose contacts among frames where mice touch?)
        top_label  share of the top-N_pos held-out frames positive for that label (= R-precision P)
        top_generic share of those frames that are CONTACT frames with no nn / np / nt label (generic contact)
        cross      R~ of the nn neuron on nose_tail and of the nt neuron on nose_nose (nn vs nt separation)
    primary_c = mean(cR~ nose_nose, cR~ nose_tail).
RULE
    Winner = highest mean primary over seeds 0-2 among R1-R7. KEEP iff the 6 vs 6 Welch 95% CI of primary (winner -
    reference) excludes 0 and the mean gain >= +0.02, reference = the round-2 decision (S0 if round 2 kept nothing).
    Specificity: the winner's primary_c vs the reference, same Welch test, reported with the result; nn and nt always
    reported separately. Also reported: the arm with the highest primary_c (seeds 0-2), confirmed the same way.
    GOAL CRITERION ("much better than the online model", deployed nn R~ 0.070 / nt 0.108, C1 0.059 / 0.087): for each
    label the 6-seed mean R~ >= 2 x the better online value (nn >= 0.140, nt >= 0.216) with the lower 95% bound above
    the online value, AND >= 50 % of the 24 contact-sheet frames labelled positive for that label.
INTERPRETABILITY (step 'sheets', every winner + refs): neuron selected on pools half 0 of split rnd0, its 24 highest
    held-out (half 1) frames, <= 1 per video, per-token heat map; per frame: label, CONTACT?, peak token dist / n_nb /
    component size relative to A1. Summary: share labelled positive, share generic contact, share of peaks on the mask
    boundary (dist 1) vs interior (dist >= 2), share of peaks at blob tips (n_nb <= 4).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nnnt as N  # noqa: E402
import r2 as R2  # noqa: E402  (1024 px; R2_S must be unset)

assert R2.RES == 1024, 'round 3 runs on the 1024 px stream'
R4, REPO, C = N.R4, N.REPO, N.C
OUT3 = Path(os.environ.get('R3_OUT') or N.OUT / 'r3')
BEE3 = Path(os.environ.get('R3_BEE') or N.BEE / 'r3')
BEE2 = R2.BEE2
log = N.log
G = N.G64
BASE = dict(W=2048, prefixes=(128, 512, 2048), k=16, steps=30000)
ARMS = {  # prep: id | white | ctx ; sel: all | edge (train + readout) ; sampler: uniform | cluster ; reuse: r2 arm
    'S0': dict(reuse='S0', prep='id', sel='all'),
    'S2': dict(reuse='S2', prep='id', sel='all'),
    'R1': dict(reuse='S2', prep='id', sel='edge'),
    'R2': dict(prep='id', sel='edge'),
    'R3': dict(prep='white', sel='all'),
    'R4': dict(prep='id', sel='all', sampler='cluster'),
    'R5': dict(prep='ctx', sel='all'),
    'R6': dict(prep='id', sel='all', W=32768, prefixes=(128, 512, 2048, 8192, 32768)),
    'R7': dict(prep='ctx', sel='edge'),
}


def arm_cfg(a):
    return {**BASE, 'sampler': 'uniform', **ARMS[a]}


# ---------------------------------------------------------------------------------------------- geometry
def geometry(tf, pos, nf):
    """Per token: dist (int8), n_nb (int8), comp size (int16); per frame: n_comp (>= 5 tokens), largest comp."""
    from scipy import ndimage
    st = np.ones((3, 3), bool)
    starts = np.r_[0, np.cumsum(np.bincount(tf, minlength=nf))]
    n = len(tf)
    dist, nnb, csz = np.zeros(n, np.int8), np.zeros(n, np.int8), np.zeros(n, np.int16)
    ncomp, maxc = np.zeros(nf, np.int16), np.zeros(nf, np.int16)
    ker = np.ones((3, 3), np.int16)
    ker[1, 1] = 0
    for f in range(nf):
        a, b = starts[f], starts[f + 1]
        if a == b:
            continue
        p = pos[a:b].astype(np.int64)
        g = np.zeros(G * G, bool)
        g[p] = True
        g = g.reshape(G, G)
        gp = np.pad(g, 1)
        d = ndimage.distance_transform_cdt(gp, metric='chessboard')[1:-1, 1:-1].ravel()
        nb = ndimage.convolve(g.astype(np.int16), ker, mode='constant').ravel()
        lab, _ = ndimage.label(g, structure=st)
        lab = lab.ravel()
        sz = np.bincount(lab)
        sz[0] = 0
        dist[a:b], nnb[a:b], csz[a:b] = d[p], nb[p], np.minimum(sz[lab[p]], 32767)
        big = sz[1:][sz[1:] >= 5]
        ncomp[f], maxc[f] = len(big), (big.max() if len(big) else 0)
    return {'dist': dist, 'n_nb': nnb, 'comp': csz, 'ncomp': ncomp, 'maxc': maxc}


def contact_frames(F, g):
    """A1(video) and the CONTACT flag per frame (largest component >= 1.5 x A1)."""
    vid = F.obs.values
    # component sizes of 4-component frames: their largest is an upper estimate; use median of maxc over such frames
    df = pd.DataFrame({'obs': vid, 'ncomp': g['ncomp'], 'maxc': g['maxc']})
    four = df[df.ncomp == 4]
    glob = float(four.maxc.median())
    cnt = four.groupby('obs').maxc.agg(['median', 'size'])
    A1 = {o: (float(cnt.loc[o, 'median']) if o in cnt.index and cnt.loc[o, 'size'] >= 20 else glob)
          for o in np.unique(vid)}
    a1 = pd.Series(vid).map(A1).values
    return g['maxc'] >= 1.5 * a1, A1, glob


def cmd_geom(args):
    BEE3.mkdir(parents=True, exist_ok=True)
    OUT3.mkdir(parents=True, exist_ok=True)
    info = {}
    for store in ('diag', 'pool'):
        root = BEE2 / store
        F = pd.read_parquet(root / 'frames.parquet')
        tf = np.load(root / 'tok_frame.npy')
        pos = np.load(root / 'tok_pos.npy')
        t0 = time.time()
        g = geometry(tf, pos, len(F))
        np.savez(BEE3 / f'geom_{store}.npz', **g)
        cf, A1, glob = contact_frames(F, g)
        info[store] = {'n_frames': len(F), 'n_tokens': len(tf), 'wall_s': time.time() - t0,
                       'edge_token_share': float((g['dist'] == 1).mean()),
                       'tip_token_share': float((g['n_nb'] <= 4).mean()),
                       'frames_4comp_share': float((g['ncomp'] == 4).mean()),
                       'A1_global_median': glob, 'contact_frame_share': float(cf.mean())}
        if store == 'diag':
            Y, _ = R2.labels_of(F.row.values)
            for li, lb in enumerate(('nose_nose', 'nose_tail', 'nn_mutual', 'np_directional')):
                info[store][f'contact_share_given_{lb}'] = float(cf[Y[:, li]].mean())
            info[store]['contact_share_given_unlabelled'] = float(cf[~Y[:, :2].any(1)].mean())
        log(f'{store}: ' + json.dumps(info[store]))
    (OUT3 / 'geom.json').write_text(json.dumps(info, indent=1))


# ---------------------------------------------------------------------------------------------- inputs
def ctx_of(torch, z, pos, fl, B):
    """Context = mean of the frame's kept tokens in the 5 x 5 window around each token, the token excluded."""
    F_ = torch.nn.functional
    d = z.shape[1]
    Gr = torch.zeros((B, G * G, d), device=z.device)
    Gr[fl, pos] = z
    Cn = torch.zeros((B, G * G), device=z.device)
    Cn[fl, pos] = 1
    S = F_.avg_pool2d(Gr.view(B, G, G, d).permute(0, 3, 1, 2), 5, 1, 2, divisor_override=1)
    Sc = F_.avg_pool2d(Cn.view(B, 1, G, G), 5, 1, 2, divisor_override=1)
    s = S.permute(0, 2, 3, 1).reshape(B, G * G, d)[fl, pos] - z
    c = Sc.view(B, G * G)[fl, pos] - 1
    return s / c.clamp_min(1)[:, None]


class Store:
    """A token store (round-2 diag or pool) with its geometry; iterates frame-aligned chunks on the GPU."""

    def __init__(self, root, geom, dev, torch, on_gpu_tokens=None):
        self.F = pd.read_parquet(root / 'frames.parquet')
        self.tf = np.load(root / 'tok_frame.npy')
        self.pos = np.load(root / 'tok_pos.npy').astype(np.int64)
        self.tok = np.memmap(root / 'tok.f16', np.float16, 'r', shape=(len(self.tf), N.D))
        g = np.load(geom)
        self.dist, self.nnb = g['dist'], g['n_nb']
        self.g = {k: g[k] for k in g.files}
        self.nf = len(self.F)
        self.starts = np.r_[0, np.cumsum(np.bincount(self.tf, minlength=self.nf))]
        self.torch, self.dev = torch, dev
        self.T = on_gpu_tokens

    def chunks(self, frames_per=48):
        torch = self.torch
        for f0 in range(0, self.nf, frames_per):
            f1 = min(self.nf, f0 + frames_per)
            a, b = int(self.starts[f0]), int(self.starts[f1])
            if a == b:
                continue
            if self.T is not None:
                z = R2.chunk_of(self.T, a, b, self.dev)
            else:
                z = torch.from_numpy(np.ascontiguousarray(self.tok[a:b])).to(self.dev)
            yield f0, f1, a, b, z

    def inputs(self, prep, tok_sel, frames_per=48):
        """Concatenated input rows for the tokens where tok_sel (bool, per token) is True: fp16 (n, d) cuda."""
        torch = self.torch
        out = []
        for f0, f1, a, b, z in self.chunks(frames_per):
            m = tok_sel[a:b]
            if not m.any():
                continue
            mt = torch.from_numpy(m).to(self.dev)
            if prep == 'ctx':
                fl = torch.from_numpy(self.tf[a:b] - f0).to(self.dev)
                ps = torch.from_numpy(self.pos[a:b]).to(self.dev)
                zf = z.float()
                x = torch.cat([zf, ctx_of(torch, zf, ps, fl, f1 - f0)], 1)[mt].half()
            else:
                x = z[mt]
            out.append(x)
        return torch.cat(out)


# ---------------------------------------------------------------------------------------------- training
def whitener(torch, X, seed):
    g = torch.Generator(device=X.device).manual_seed(seed)
    ii = torch.randperm(len(X), generator=g, device=X.device)[:1_000_000]
    x = X[ii].double()
    mu = x.mean(0)
    cov = (x - mu).T @ (x - mu) / (len(x) - 1)
    lam, U = torch.linalg.eigh(cov)
    lam = lam.clamp_min(0)
    eps = 0.05 * lam.mean()
    Wz = (U * (lam + eps).rsqrt()) @ U.T
    return mu.float(), Wz.float(), {'eps': float(eps), 'top_eig_share': float(lam[-1] / lam.sum()),
                                    'top10_eig_share': float(lam[-10:].sum() / lam.sum())}


def kmeans_weights(torch, X, norm, seed, K=1024, iters=20):
    g = torch.Generator(device=X.device).manual_seed(seed)
    ii = torch.randperm(len(X), generator=g, device=X.device)[:1_000_000]
    xs = torch.nn.functional.normalize(norm(X[ii]), dim=1)
    Cc = xs[torch.randperm(len(xs), generator=g, device=X.device)[:K]].clone()
    for _ in range(iters):
        lab = torch.cat([(xs[a:a + 65536] @ Cc.T).argmax(1) for a in range(0, len(xs), 65536)])
        Cn = torch.zeros_like(Cc).index_add_(0, lab, xs)
        cnt = torch.bincount(lab, minlength=K)
        Cc = torch.where(cnt[:, None] > 0, torch.nn.functional.normalize(Cn, dim=1), Cc)
    lab = torch.cat([(torch.nn.functional.normalize(norm(X[a:a + 262144]), dim=1) @ Cc.T).argmax(1)
                     for a in range(0, len(X), 262144)])
    size = torch.bincount(lab, minlength=K).double()
    w = size[lab].rsqrt()
    cdf = torch.cumsum(w / w.sum(), 0)
    eff = (size.sqrt() / size.sqrt().sum())  # cluster mass under the sampler
    return cdf, {'K': K, 'size_max_share': float(size.max() / size.sum()), 'n_empty': int((size == 0).sum()),
                 'mass_max_share_sampler': float(eff.max()), 'mass_max_share_uniform': float(size.max() / size.sum())}


def train_arm(torch, cfg, seed, Xtr, Xca, Xev, od, dev):
    from src.eci.sae import MatryoshkaBatchTopKSAE, TokenNorm, _train_step, geometric_median, save_checkpoint
    from src.eci.sae_calibrate import calibrate_threshold, set_threshold, sparsity_stats
    od.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    extra, prep = {}, None
    if cfg['prep'] == 'white':
        mu, Wz, wi = whitener(torch, Xtr, seed)
        torch.save({'mu': mu.cpu(), 'Wz': Wz.cpu()}, od / 'prep.pt')
        extra['white'] = wi
        prep = lambda x: (x.float() - mu) @ Wz  # noqa: E731
    pf = prep or (lambda x: x.float())
    din = Xtr.shape[1]
    g = np.random.default_rng(seed)
    ii = torch.from_numpy(np.sort(g.choice(len(Xtr), min(250_000, len(Xtr)), replace=False))).to(dev)
    init = pf(Xtr[ii])
    norm = TokenNorm(din).to(dev).fit_blocks(init, [N.D] * (din // N.D), seed=seed)
    sae = MatryoshkaBatchTopKSAE(d_in=din, n_latents=cfg['W'], prefixes=cfg['prefixes'], k=cfg['k'], seed=seed).to(dev)
    with torch.no_grad():
        sae.b_dec.copy_(geometric_median(norm(init[:100_000])))
    del init
    full = lambda x: norm(pf(x))  # noqa: E731
    cdf = None
    if cfg['sampler'] == 'cluster':
        cdf, extra['kmeans'] = kmeans_weights(torch, Xtr, full, seed)
    opt = torch.optim.Adam(sae.parameters(), lr=5e-4, betas=(0.9, 0.999))
    gen = torch.Generator(device=dev).manual_seed(seed)
    Nn, bs, steps = len(Xtr), 4096, cfg['steps']
    perm, p, hist, t0 = torch.randperm(Nn, generator=gen, device=dev), 0, [], time.time()
    for step in range(steps):
        if cdf is not None:
            b = torch.searchsorted(cdf, torch.rand(bs, generator=gen, device=dev, dtype=torch.float64)).clamp_max(Nn - 1)
        else:
            if p + bs > Nn:
                perm, p = torch.randperm(Nn, generator=gen, device=dev), 0
            b = perm[p:p + bs]
            p += bs
        logs, _ = _train_step(sae, opt, full(Xtr[b]), step, steps, 5e-4, 500, 1.0)
        if step % 5000 == 0 or step == steps - 1:
            logs.update(step=step, elapsed_s=round(time.time() - t0, 1))
            hist.append(logs)
    sae.eval()
    torch.backends.cuda.matmul.allow_tf32 = False
    tt = time.time() - t0
    th, _ = calibrate_threshold(sae, Xca, norm=full)
    set_threshold(sae, th)
    stt = sparsity_stats(sae, Xev, th, norm=full)
    save_checkpoint(od / 'sae_theta.pt', sae, norm, extra={'config': 'nnnt_round3', 'cfg': cfg, 'seed': seed})
    return {'train_time_s': tt, 'steps_per_s': steps / tt, 'n_train_tokens': Nn, 'theta_star': th,
            'eval_fve': stt['fve'], 'eval_l0': stt['l0'], 'dead_at_end_train':
            int((sae.tokens_since_fired >= sae.dead_tokens).sum()), 'history': hist, **extra}


def load_arm(torch, a, seed, dev):
    """-> sae, full-transform callable (raw input rows -> normalised SAE input)."""
    from src.eci.sae import load_sae
    cfg = arm_cfg(a)
    if 'reuse' in cfg:
        sae, norm, _ = load_sae(BEE2 / 'sae' / f'{cfg["reuse"]}_s{seed}' / 'sae_theta.pt', dev)
        return sae, (lambda x: norm(x.float()))
    od = BEE3 / 'sae' / f'{a}_s{seed}'
    sae, norm, _ = load_sae(od / 'sae_theta.pt', dev)
    if cfg['prep'] == 'white':
        pr = torch.load(od / 'prep.pt')
        mu, Wz = pr['mu'].to(dev), pr['Wz'].to(dev)
        return sae, (lambda x: norm((x.float() - mu) @ Wz))
    return sae, (lambda x: norm(x.float()))


def cmd_train(args):
    import torch
    dev = torch.device('cuda')
    loc = N.local_dir()
    R4.atomic_copy(BEE2 / 'pool', loc / 'pool')
    P = Store(loc / 'pool', BEE3 / 'geom_pool.npz', dev, torch)
    role = P.F.role.values[P.tf]
    edge = P.dist == 1
    res_f = OUT3 / (args.out or 'train.json')  # one file per concurrent job
    res = json.loads(res_f.read_text()) if res_f.exists() else {}
    seeds = [int(s) for s in args.seeds.split(',')]
    arms = [a for a in args.arms.split(',') if 'reuse' not in ARMS[a]]
    for key in sorted({(arm_cfg(a)['prep'] == 'ctx', arm_cfg(a)['sel']) for a in arms}):
        todo = [(a, s) for a in arms for s in seeds if (arm_cfg(a)['prep'] == 'ctx', arm_cfg(a)['sel']) == key
                and (args.force or not (BEE3 / 'sae' / f'{a}_s{s}' / 'sae_theta.pt').exists())]
        if not todo:
            continue
        t0 = time.time()
        sel = edge if key[1] == 'edge' else np.ones(len(edge), bool)
        prep = 'ctx' if key[0] else 'id'
        X = {r: P.inputs(prep, sel & (role == r)) for r in ('train', 'calib', 'eval')}
        X['calib'], X['eval'] = X['calib'].cpu(), X['eval'].cpu()  # calibration batches are moved to the GPU
        log(f'stream ctx={key[0]} sel={key[1]}: ' + ', '.join(f'{r} {len(x):,}' for r, x in X.items()) +
            f' tokens x {X["train"].shape[1]} built in {time.time() - t0:.0f}s; GPU '
            f'{torch.cuda.memory_allocated() / 2**30:.1f} GiB')
        for a, s in todo:
            name = f'{a}_s{s}'
            ti = train_arm(torch, arm_cfg(a), s, X['train'], X['calib'], X['eval'], BEE3 / 'sae' / name, dev)
            res[name] = {'cfg': {k: (list(v) if isinstance(v, tuple) else v) for k, v in arm_cfg(a).items()}, **ti}
            log(f'{name}: {ti["train_time_s"]:.0f}s ({ti["steps_per_s"]:.1f} steps/s), theta* {ti["theta_star"]:.4f} '
                f'eval FVE {ti["eval_fve"]:.4f} L0 {ti["eval_l0"]:.2f} dead {ti["dead_at_end_train"]} ' +
                json.dumps({k: ti[k] for k in ('white', 'kmeans') if k in ti}))
            OUT3.mkdir(parents=True, exist_ok=True)
            res_f.write_text(json.dumps(res, indent=1, default=float))
        del X
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------------------------- scoring
def read_codes(torch, D, sae, full, cfg, dev, fsel=None):
    """Frame codes (nf, W) fp16: max over kept (or edge) tokens of the frame."""
    W = sae.n_latents
    X = torch.zeros((D.nf, W), dtype=torch.float16, device=dev)
    edge_t = torch.from_numpy(D.dist == 1).to(dev) if cfg['sel'] == 'edge' else None
    fp = max(4, min(48, int(2**27 // W // 160)))
    with torch.no_grad():
        for f0, f1, a, b, z in D.chunks(fp):
            fl = torch.from_numpy(D.tf[a:b] - f0).to(dev)
            if cfg['prep'] == 'ctx':
                ps = torch.from_numpy(D.pos[a:b]).to(dev)
                zf = z.float()
                z = torch.cat([zf, ctx_of(torch, zf, ps, fl, f1 - f0)], 1)
            if edge_t is not None:
                m = edge_t[a:b]
                z, fl = z[m], fl[m]
            c = sae.encode(full(z), mode='threshold')
            X[f0:f1] = torch.zeros((f1 - f0, W), device=dev).index_reduce_(0, fl, c, 'amax', include_self=True).half()
    return X


def spec_metrics(X, out, Y, half_of, cf):
    """Specificity of the R~-selected neurons on the held-out halves (see module docstring)."""
    res = {}
    for li, lb in enumerate(('nose_nose', 'nose_tail')):
        recs = []
        for sn in [s for s in half_of if s.startswith('rnd')]:
            h = half_of[sn]
            for d_, te_h in ((0, 1), (1, 0)):
                j = out['splits'][sn][lb]['neurons'][d_]
                te = np.flatnonzero(h == te_h)
                v = X[:, j].float().cpu().numpy().astype(np.float64)[te]
                y = Y[te, li]
                c = cf[te]
                r = {}
                if y[c].sum() > 0 and (~y[c]).sum() > 0:
                    m = R2.metrics(v[c], y[c])
                    r['cR'], r['cA'] = m['R'], m['A']
                k = int(y.sum())
                o = np.argsort(-v, kind='stable')[:k]
                any_lab = Y[te][:, :2].any(1)
                r['top_label'] = float(y[o].mean())
                r['top_generic'] = float((c[o] & ~any_lab[o]).mean())
                r['top_other_label'] = float((any_lab[o] & ~y[o]).mean())
                r['top_contact'] = float(c[o].mean())
                recs.append(r)
        res[lb] = {q: float(np.nanmean([r.get(q, np.nan) for r in recs])) for q in
                   ('cR', 'cA', 'top_label', 'top_generic', 'top_other_label', 'top_contact')}
    res['primary_c'] = float(np.mean([res['nose_nose']['cR'], res['nose_tail']['cR']]))
    return res


def cmd_score(args):
    import torch
    dev = torch.device('cuda')
    loc = N.local_dir()
    R4.atomic_copy(BEE2 / 'diag', loc / 'diag')
    D = Store(loc / 'diag', BEE3 / 'geom_diag.npz', dev, torch)
    D.T = R2.tokens_to(torch, D.tok, dev)
    cf, _, _ = contact_frames(D.F, D.g)
    Y, unit = R2.labels_of(D.F.row.values)
    sc = json.loads((N.OUT / 'score.json').read_text())
    units, SPL = sc['units'], {k_: np.array(v_) for k_, v_ in sc['splits'].items()}
    uidx = pd.Series(range(len(units)), index=units).loc[unit].values
    half_of = {s_: h[uidx] for s_, h in SPL.items()}
    Yg = torch.from_numpy(Y).to(dev)
    res_f = OUT3 / (args.out or 'score.json')
    res = json.loads(res_f.read_text()) if res_f.exists() else {}
    for a in args.arms.split(','):
        for s in [int(x) for x in args.seeds.split(',')]:
            name = f'{a}_s{s}'
            if name in res and not args.force:
                continue
            try:
                sae, full = load_arm(torch, a, s, dev)
            except FileNotFoundError:
                log(f'{name}: no checkpoint, skipped')
                continue
            t0 = time.time()
            X = read_codes(torch, D, sae, full, arm_cfg(a), dev)
            te = time.time() - t0
            out, _ = N.score_set(torch, X, Yg, half_of, dev)
            out['spec'] = spec_metrics(X, out, Y, half_of, cf)
            out['width'] = int(sae.n_latents)
            out['encode_s'] = te
            out['frame_fire_rate_mean'] = float((X > 0).float().mean())
            res[name] = out
            m, sp = out['mean'], out['spec']
            log(f'{name}: nn R~ {m["nose_nose"]["R"]:.4f} AUROC {m["nose_nose"]["A"]:.4f} | nt R~ '
                f'{m["nose_tail"]["R"]:.4f} AUROC {m["nose_tail"]["A"]:.4f} | primary R~ {out["primary"]:.4f} | '
                f'contact-frame R~ nn {sp["nose_nose"]["cR"]:.4f} nt {sp["nose_tail"]["cR"]:.4f} | top generic nn '
                f'{sp["nose_nose"]["top_generic"]:.3f} nt {sp["nose_tail"]["top_generic"]:.3f} '
                f'(encode {te:.0f}s, {len(D.tf) / te / 1e6:.2f}M tok/s, GPU peak '
                f'{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB)')
            del X, sae
            torch.cuda.empty_cache()
            res_f.write_text(json.dumps(res, indent=1, default=float))


# ---------------------------------------------------------------------------------------------- sheets
def cmd_sheets(args):
    import torch
    dev = torch.device('cuda')
    loc = N.local_dir()
    R4.atomic_copy(BEE2 / 'diag', loc / 'diag')
    D = Store(loc / 'diag', BEE3 / 'geom_diag.npz', dev, torch)
    D.T = R2.tokens_to(torch, D.tok, dev)
    cf, A1, _ = contact_frames(D.F, D.g)
    Pl = N.load_plan()
    V = {v['obs']: v for v in Pl['videos']}
    res = json.loads((OUT3 / 'score.json').read_text())
    sc = json.loads((N.OUT / 'score.json').read_text())
    half = dict(zip(sc['units'], sc['splits']['rnd0']))
    hold = np.array([half[o.rsplit('_', 2)[0]] == 1 for o in D.F.obs.values])
    ann = pd.read_csv(REPO / C['ann'], usecols=['Y_nn', 'Y_np', 'Y_nt'])
    out_f = OUT3 / 'locality.json'
    out = json.loads(out_f.read_text()) if out_f.exists() else {}
    (OUT3 / 'sheets').mkdir(parents=True, exist_ok=True)
    fd_all = np.load(loc / 'diag' / 'tok_fdark.npy').astype(np.float32)
    for name in args.names.split(','):
        a, s = name.rsplit('_s', 1)
        cfg = arm_cfg(a)
        sae, full = load_arm(torch, a, int(s), dev)
        for lb in ('nose_nose', 'nose_tail'):
            j = res[name]['splits']['rnd0'][lb]['neurons'][0]
            act = torch.zeros(len(D.tf), dtype=torch.float32, device=dev)
            with torch.no_grad():
                for f0, f1, a_, b_, z in D.chunks(16 if cfg['W'] > 8192 else 48):
                    if cfg['prep'] == 'ctx':
                        fl = torch.from_numpy(D.tf[a_:b_] - f0).to(dev)
                        ps = torch.from_numpy(D.pos[a_:b_]).to(dev)
                        zf = z.float()
                        z = torch.cat([zf, ctx_of(torch, zf, ps, fl, f1 - f0)], 1)
                    act[a_:b_] = sae.encode(full(z), mode='threshold')[:, j]
            act = act.cpu().numpy()
            ro = act * (D.dist == 1) if cfg['sel'] == 'edge' else act  # the readout's tokens
            fmax = np.maximum.reduceat(ro, D.starts[:-1].clip(max=len(ro) - 1))
            fmax[np.diff(D.starts) == 0] = 0
            cand = pd.DataFrame({'f': np.arange(D.nf), 'v': fmax, 'obs': D.F.obs.values})[hold]
            cand = cand.sort_values('v', ascending=False).drop_duplicates('obs').head(24)
            tiles, recs = [], []
            for f, val in zip(cand.f.values, cand.v.values):
                o, k, row = D.F.obs.values[f], int(D.F.k.values[f]), int(D.F.row.values[f])
                v = V[o]
                fr = R4.decode(v['path'], v['offset'] + C['step'] * k, 1, C['step'], N.S, v['hflip'], C['fps'],
                               threads=4)[0]
                x = np.rot90(fr, v['rot'], axes=(0, 1)).transpose(2, 0, 1).copy()
                a0, b0 = D.starts[f], D.starts[f + 1]
                ps = D.pos[a0:b0]
                h = np.zeros(G * G, np.float32)
                h[ps] = act[a0:b0]
                pk = a0 + int(np.argmax(ro[a0:b0]))
                y = ann.iloc[row]
                lab = ''.join(c for c, col in (('nn', 'Y_nn'), ('np', 'Y_np'), ('nt', 'Y_nt')) if y[col] > 0) or '-'
                hi = h >= 0.5 * h.max() if h.max() > 0 else np.zeros_like(h, bool)
                fdk = np.zeros(G * G, np.float32)
                fdk[ps] = fd_all[a0:b0]
                r = {'obs': o, 'k': k, 'stored': float(val), 'label': lab, 'contact': bool(cf[f]),
                     'peak_dist': int(D.dist[pk]), 'peak_n_nb': int(D.nnb[pk]),
                     'peak_comp_over_A1': float(D.g['comp'][pk] / A1[o]), 'n_half_max': int(hi.sum()),
                     'mass_animal_px': float((h * fdk).sum() / h.sum()) if h.sum() > 0 else np.nan}
                recs.append(r)
                tiles.append((x, h.reshape(G, G), r))
            R = pd.DataFrame(recs)
            want = {'nose_nose': ('nn', 'np'), 'nose_tail': ('nt',)}[lb]
            pos_ = R.label.apply(lambda s_: any(w in s_ for w in want))
            summ = {'share_label_pos': float(pos_.mean()),
                    'share_generic_contact': float((R.contact & (R.label == '-')).mean()),
                    'share_other_label': float((~pos_ & (R.label != '-')).mean()),
                    'share_contact': float(R.contact.mean()),
                    'share_peak_boundary': float((R.peak_dist == 1).mean()),
                    'share_peak_interior': float((R.peak_dist >= 2).mean()),
                    'share_peak_tip': float((R.peak_n_nb <= 4).mean()),
                    'share_peak_in_merged_blob': float((R.peak_comp_over_A1 >= 1.5).mean()),
                    'mean_n_half_max': float(R.n_half_max.mean()), 'mass_animal_px': float(R.mass_animal_px.mean()),
                    'label_counts': R.label.value_counts().to_dict()}
            fn = OUT3 / 'sheets' / f'{name}_{lb}_n{j}.jpg'
            N.contact_sheet(tiles, fn, f'round 3 {name} neuron {j} ({lb}-selected on pools half 0 of rnd0; top held-out '
                                       f'frames, 1 per video; readout {cfg["sel"]})', G)
            out[f'{name}|{lb}'] = {'neuron': j, 'summary': summ, 'frames': recs, 'sheet': str(fn)}
            log(f'{name} {lb} n{j}: ' + json.dumps(summ))
            out_f.write_text(json.dumps(out, indent=1, default=float))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=('geom', 'train', 'score', 'sheets'))
    p.add_argument('--arms', default='R2,R3,R4,R5,R7')
    p.add_argument('--seeds', default='0,1,2,3,4,5')
    p.add_argument('--names', default='')
    p.add_argument('--out', default='')
    p.add_argument('--force', action='store_true')
    args = p.parse_args()
    {'geom': cmd_geom, 'train': cmd_train, 'score': cmd_score, 'sheets': cmd_sheets}[args.step](args)


if __name__ == '__main__':
    main()
