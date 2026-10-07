"""
Tracker-free multi-scale SAEs for ECI (src/eci/multiscale.py): does a sparse autoencoder on window tokens at the
right spatial scale grow neurons that are clean on annotated SINGLE / PAIRED behaviours, and which scale holds them?

Data: the foreground token stores and the video split of scripts/eci/spatial_sae_pilot.py (split seed 0):
    mice  fg448 store; train = 100 videos WITHOUT behaviour annotation; eval = every store frame (1 fps) of the 144
          annotated videos carrying Y_nn, Y_np, Y_nt
    ants  antsfg store (v2 + v3); train = 128 random videos; eval = the store frames of the other 128
No label is used in training. Reading, split, labels and the per-latent alignment scoring (align_columns: AUROC,
AP, precision in the top 1% of frames) are imported from spatial_sae_pilot.py.

Configs (src/eci/multiscale.py SCALES): one SAE per scale S1 / S_part / S_animal / S_pair / S_frame, and one joint
SAE on tokens of all five scales mixed (one-hot scale dims appended, 5 extra dims). Window token = [mean | max |
fg fraction, centroid row, centroid col] (1539 dims). TokenNorm.fit_blocks with blocks [768, 768, 3] (+ [5] joint):
each block gets average norm sqrt(block size), so every dim weighs about the same in the loss.
SAE = the deployed recipe: Matryoshka BatchTopK (src/eci/sae.py), 1024 latents, k = 16, prefixes 128/256/512/1024,
Adam lr 5e-4, 500 warmup steps, cosine to 10%, batches of 4096, AuxK, b_dec init = geometric median.
Training budget: every config gets the SAME number of steps (--steps, default 6000 = 24.6M token presentations) on
a pool of at most --cap unique window tokens drawn uniformly from all windows of the train frames (pool seed fixed,
so the 3 SAE seeds see the same data). Coarse scales have fewer unique windows than the cap (logged in train.json).
Joint pool: cap // 5 windows per scale (drawn with replacement where a scale has fewer).

Per-frame read-out: for each neuron, the max over that frame's windows at its scale (0 if no window). For the joint
SAE the neuron's scale = the scale with the highest per-window firing rate on the eval windows ('joint_own'); also
'joint_allmax' (max over all windows of all scales) and one read-out per scale ('joint@S_part' ...).

Steps (STEP of scripts/eci/multiscale_sae.sh):
    selftest  window builder vs a brute-force numpy reference on real store frames
    train     stage train frames to $LOCAL_DIR, build pools, train configs x seeds -> OUT/sae/<cfg>_s<seed>/
    evaluate  stage eval frames, encode, align, size control, Y2F/B2F discrimination, collective proxies, contact
              sheets -> OUT/{eval,align,codes_best,sheets}/, OUT/summary.json
"""
import argparse
import json
import math
import os
import shutil
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts/eci'))
import spatial_sae_pilot as ssp  # noqa: E402
from src.eci.domain import get_domain  # noqa: E402
from src.eci.multiscale import (D_TOKEN, D_WINDOW, EXTRA, GRID, SCALE_NAMES, scale_table, window_counts,  # noqa: E402
                                window_tokens)
from src.eci.sae import MatryoshkaBatchTopKSAE, TokenNorm, _train_step, geometric_median, load_sae, save_checkpoint  # noqa: E402

N_TRAIN = {'mice': 100, 'ants': 128}
CONFIGS = SCALE_NAMES + ('joint',)
N_SCALES = len(SCALE_NAMES)
FRAME_PX = 512
log = lambda s: print(s, flush=True)  # noqa: E731


# ---------------------------------------------------------------------------------------------- data
def local_dir():
    d = os.environ.get('LOCAL_DIR')
    if not d:
        raise RuntimeError('LOCAL_DIR is not set (the job staging dir /localhome/$USER/$SLURM_JOB_ID)')
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    return d


def stage(idx, frames, dst, chunk=8000):
    """Copy the tokens of `frames` (sorted store frame indices) from the NFS store to dst (local disk), in contiguous
    reads -> (tok memmap (N, 768) fp16, pos (N,) int16, lens (F,) int64)."""
    dst.mkdir(parents=True, exist_ok=True)
    lens = idx.nfg[frames].astype(np.int64)
    n = int(lens.sum())
    need = n * D_TOKEN * 2
    free = shutil.disk_usage(dst).free
    if free < need + 20e9:
        raise RuntimeError(f'{dst}: {free / 1e9:.0f} GB free, staging needs {need / 1e9:.0f} GB + 20 GB margin')
    t0 = time.time()
    tok = np.lib.format.open_memmap(dst / 'tok.npy', 'w+', np.float16, (n, D_TOKEN))
    pos = np.empty(n, np.int16)
    o = 0
    for c0 in range(0, len(frames), chunk):
        t, p, ln = idx.load(frames[c0:c0 + chunk])
        assert (ln == lens[c0:c0 + chunk]).all()
        tok[o:o + len(t)] = t
        pos[o:o + len(t)] = p
        o += len(t)
    assert o == n
    tok.flush()
    del tok
    log(f'  staged {len(frames):,} frames, {n:,} tokens ({need / 1e9:.1f} GB) to {dst} in {time.time() - t0:.0f}s')
    return np.load(dst / 'tok.npy', mmap_mode='r'), pos, lens


def iter_chunks(tok, pos, lens, chunk, dev):
    st = np.r_[0, np.cumsum(lens)]
    for f0 in range(0, len(lens), chunk):
        f1 = min(len(lens), f0 + chunk)
        lo, hi = st[f0], st[f1]
        yield (f0, f1, torch.from_numpy(np.ascontiguousarray(tok[lo:hi])).to(dev),
               torch.from_numpy(pos[lo:hi].astype(np.int64)).to(dev), torch.from_numpy(lens[f0:f1]).to(dev))


def eval_labels(domain, idx, ev):
    """Labels of the eval frames (store order) -> DataFrame (row, obs, labels...), label names, valid masks."""
    _, ev_all, lab, beh, _, _ = idx.split(domain, N_TRAIN[domain], 0)
    lab = lab.set_index('row').loc[idx.rows[ev]].reset_index()
    valid = {b: np.ones(len(lab), bool) for b in beh}
    ann = pd.read_csv(get_domain(domain).ann_path,
                      usecols=['frame_path', 'frame_idx'] + (['experiment', 'Y_YOL', 'Y_BOL', 'Y_FOL']
                                                             if domain == 'ants' else []))
    a = ann.iloc[lab['row'].values]
    lab['frame_path'] = a['frame_path'].values
    lab['frame_idx'] = a['frame_idx'].values
    if domain == 'ants':
        lab['experiment'] = a['experiment'].values
        for c, name in (('Y_YOL', 'onlid_yellow'), ('Y_BOL', 'onlid_blue'), ('Y_FOL', 'onlid_focal')):
            v = a[c].values.astype(float)
            valid[name] = ~np.isnan(v)
            lab[name] = np.nan_to_num(v) > 0
            beh = beh + [name]
    return lab, beh, valid


# ---------------------------------------------------------------------------------------------- selftest
def cmd_selftest(args):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    idx = ssp.StoreIndex(args.domain)
    fr = np.flatnonzero(idx.nfg > 0)[:40]
    tok, pos, lens = idx.load(fr)
    T = torch.from_numpy(tok).to(dev)
    P = torch.from_numpy(pos.astype(np.int64)).to(dev)
    L = torch.from_numpy(lens.astype(np.int64)).to(dev)
    st = np.r_[0, np.cumsum(lens)]
    worst = 0.0
    for name, (side, stride, mf) in scale_table(args.domain).items():
        feats, frame, win = window_tokens(T, P, L, side, stride, mf)
        feats, frame, win = feats.cpu().numpy(), frame.cpu().numpy(), win.cpu().numpy()
        cnt = window_counts(P, L, side, stride, mf).cpu().numpy()
        assert (np.bincount(frame, minlength=len(fr)) == cnt).all(), name
        if side == 1:  # fast path == generic path
            f2, fr2, w2 = window_tokens(T, P, L, side, stride, mf, fast=False)
            o1 = np.lexsort((win[:, 1], win[:, 0], frame))
            o2 = np.lexsort((w2[:, 1].cpu().numpy(), w2[:, 0].cpu().numpy(), fr2.cpu().numpy()))
            d = np.abs(feats[o1] - f2.cpu().numpy()[o2]).max()
            assert d < 1e-4, (name, d)
        from src.eci.multiscale import min_count, window_starts
        starts = window_starts(side, stride)
        n_ref = 0
        for f in range(len(fr)):
            x = tok[st[f]:st[f + 1]].astype(np.float32)
            p = pos[st[f]:st[f + 1]].astype(int)
            r, c = p // GRID, p % GRID
            for r0 in starts:
                for c0 in starts:
                    m = (r >= r0) & (r < r0 + side) & (c >= c0) & (c < c0 + side)
                    if m.sum() < min_count(side, mf):
                        continue
                    n_ref += 1
                    ref = np.r_[x[m].mean(0), x[m].max(0), m.sum() / side ** 2, ((r[m] + .5) / GRID).mean(),
                                ((c[m] + .5) / GRID).mean()]
                    j = np.flatnonzero((frame == f) & (win[:, 0] == r0) & (win[:, 1] == c0))
                    assert len(j) == 1, (name, f, r0, c0, len(j))
                    worst = max(worst, float(np.abs(feats[j[0]] - ref).max()))
        assert n_ref == len(feats), (name, n_ref, len(feats))
        log(f'{name}: side {side} stride {stride} min_frac {mf}: {len(feats)} windows on {len(fr)} frames '
            f'({len(feats) / len(fr):.1f}/frame) match the brute-force reference')
    log(f'selftest OK, max |feature - reference| {worst:.2e}')
    assert worst < 1e-3


# ---------------------------------------------------------------------------------------------- train
def train_one(X, blocks, seed, steps, out, extra, dev, bs=4096, lr=5e-4, warmup=500):
    """X: (N, d) fp16 tensor (GPU or CPU) of raw window tokens -> trains, saves out/sae.pt + train.json."""
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    N, d = X.shape
    g = np.random.default_rng(seed)
    init = X[torch.from_numpy(np.sort(g.choice(N, min(500_000, N), replace=False))).to(X.device)].to(dev)
    norm = TokenNorm(d).to(dev).fit_blocks(init, blocks, seed=seed)
    sae = MatryoshkaBatchTopKSAE(d_in=d, n_latents=1024, prefixes=(128, 256, 512, 1024), k=16, seed=seed).to(dev)
    with torch.no_grad():
        sae.b_dec.copy_(geometric_median(norm(init[:100_000])))
    del init
    opt = torch.optim.Adam(sae.parameters(), lr=lr, betas=(0.9, 0.999))
    gen = torch.Generator().manual_seed(seed)
    perm, p, epochs = torch.randperm(N, generator=gen), 0, 0
    history, t0 = [], time.time()
    for step in range(steps):
        if p + bs > N:
            perm, p, epochs = torch.randperm(N, generator=gen), 0, epochs + 1
        b = perm[p:p + bs] if N >= bs else torch.randint(N, (bs,), generator=gen)
        p += bs
        x = norm(X[b.to(X.device)].to(dev, non_blocking=True))
        logs, gn = _train_step(sae, opt, x, step, steps, lr, warmup, 1.0)
        if step % 1000 == 0 or step == steps - 1:
            logs.update(step=step, elapsed_s=round(time.time() - t0, 1))
            history.append(logs)
            log(f"    step {step}/{steps} fve {logs['fve']:.4f} l0 {logs['l0']:.1f} dead {logs['n_dead']} "
                f"{logs['elapsed_s']}s")
    sae.eval()
    save_checkpoint(out / 'sae.pt', sae, norm, extra={**extra, 'seed': seed})
    (out / 'train.json').write_text(json.dumps({**extra, 'seed': seed, 'n_unique_tokens': N, 'd_in': d,
                                                'steps': steps, 'token_presentations': steps * bs,
                                                'epochs_over_pool': steps * bs / N, 'train_time_s': time.time() - t0,
                                                'gpu': torch.cuda.get_device_name(0), 'history': history}, indent=1))
    log(f'    trained in {time.time() - t0:.0f}s, final fve {history[-1]["fve"]:.4f}')


def onehot(n, s, dev=None, dtype=torch.float32):
    o = torch.zeros(n, N_SCALES, dtype=dtype, device=dev)
    o[:, s] = 1
    return o


def cmd_train(args):
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = True
    out = Path(args.out_dir)
    loc = local_dir()
    idx = ssp.StoreIndex(args.domain)
    tr, _, _, _, train_v, eval_v = idx.split(args.domain, N_TRAIN[args.domain], 0)
    tr = np.sort(tr)
    if args.max_train_frames:
        tr = np.sort(np.random.default_rng(0).choice(tr, args.max_train_frames, replace=False))
    log(f'{args.domain}: {len(train_v)} train videos, {len(tr):,} train frames; {len(eval_v)} eval videos')
    tok, pos, lens = stage(idx, tr, loc / 'train')
    scales = scale_table(args.domain)
    # pass 1: window counts per frame -> uniform selection of at most cap windows per scale
    counts = {s: np.zeros(len(tr), np.int64) for s in scales}
    for f0, f1, _, P, L in iter_chunks(np.zeros((len(pos), 0), np.float16), pos, lens, 4096, dev):
        for s, (side, stride, mf) in scales.items():
            counts[s][f0:f1] = window_counts(P, L, side, stride, mf).cpu().numpy()
    rng = np.random.default_rng(args.pool_seed)
    sel, pools, meta = {}, {}, {}
    for s in scales:
        tot = int(counts[s].sum())
        sel[s] = np.sort(rng.choice(tot, min(args.cap, tot), replace=False))
        pools[s] = np.empty((len(sel[s]), D_WINDOW), np.float16)
        meta[s] = {'windows_total': tot, 'windows_per_frame': tot / len(tr), 'pool': len(sel[s])}
        log(f'  {s}: {tot:,} windows ({tot / len(tr):.1f}/frame) -> pool {len(sel[s]):,}')
    # pass 2: window tokens, keep the selected ones
    goff = {s: 0 for s in scales}
    fill = {s: 0 for s in scales}
    t0 = time.time()
    for f0, f1, T, P, L in iter_chunks(tok, pos, lens, args.chunk_frames, dev):
        for s, (side, stride, mf) in scales.items():
            feats, _, _ = window_tokens(T, P, L, side, stride, mf)
            n = feats.shape[0]
            assert n == counts[s][f0:f1].sum(), (s, n)
            a, b = np.searchsorted(sel[s], [goff[s], goff[s] + n])
            if b > a:
                li = torch.from_numpy(sel[s][a:b] - goff[s]).to(dev)
                pools[s][fill[s]:fill[s] + b - a] = feats[li].half().cpu().numpy()
                fill[s] += b - a
            goff[s] += n
    for s in scales:
        assert fill[s] == len(pools[s]), (s, fill[s], len(pools[s]))
    log(f'  pools built in {time.time() - t0:.0f}s')
    del tok
    shutil.rmtree(loc / 'train')
    # joint pool: cap // 5 per scale, one-hot scale dims appended
    share = args.cap // N_SCALES
    joint = np.empty((share * N_SCALES, D_WINDOW + N_SCALES), np.float16)
    jmeta = {}
    for si, s in enumerate(SCALE_NAMES):
        n = len(pools[s])
        ii = rng.choice(n, share, replace=n < share)
        joint[si * share:(si + 1) * share, :D_WINDOW] = pools[s][np.sort(ii)]
        joint[si * share:(si + 1) * share, D_WINDOW:] = 0
        joint[si * share:(si + 1) * share, D_WINDOW + si] = 1
        jmeta[s] = {'drawn': share, 'unique_available': n, 'with_replacement': bool(n < share)}
    common = {'domain': args.domain, 'train_videos': train_v, 'n_train_frames': len(tr), 'cap': args.cap,
              'pool_seed': args.pool_seed, 'scales': {s: list(v) for s, v in scales.items()}, 'window_meta': meta,
              'joint_meta': jmeta}
    (out).mkdir(parents=True, exist_ok=True)
    (out / 'pools.json').write_text(json.dumps(common, indent=1))
    for cfg in args.configs:
        X = pools[cfg] if cfg != 'joint' else joint
        blocks = [D_TOKEN, D_TOKEN, EXTRA] + ([N_SCALES] if cfg == 'joint' else [])
        Xt = torch.from_numpy(X)
        free = torch.cuda.mem_get_info()[0]
        if Xt.numel() * 2 < free - 8e9:
            Xt = Xt.to(dev)
        log(f'{cfg}: pool {X.shape[0]:,} x {X.shape[1]} on {Xt.device}')
        for seed in args.seeds:
            log(f'  seed {seed}')
            train_one(Xt, blocks, seed, args.steps, out / 'sae' / f'{cfg}_s{seed}',
                      {'config': cfg, 'blocks': blocks, **{k: common[k] for k in ('domain', 'cap', 'pool_seed')},
                       'scales': common['scales']}, dev)
        del Xt
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------------------------- evaluate
def spearman_cols(X, y):
    """Spearman rho of every column of X (n, m) with y (n,), ties averaged."""
    from scipy.stats import rankdata
    ry = rankdata(y)
    ry = (ry - ry.mean()) / ry.std()
    out = np.zeros(X.shape[1])
    for j0 in range(0, X.shape[1], 128):
        R = rankdata(X[:, j0:j0 + 128], axis=0)
        R = R - R.mean(0)
        sd = R.std(0)
        out[j0:j0 + 128] = np.where(sd > 0, (R * ry[:, None]).mean(0) / np.where(sd > 0, sd, 1), 0)
    return out


CTX = {}


def analyse(path):
    """One code set (memmap (F, 1024)) -> per-neuron metrics for every label (worker process; context in CTX)."""
    import warnings
    warnings.simplefilter('ignore', RuntimeWarning)
    C = CTX
    X = np.asarray(np.load(path, mmap_mode='r'), dtype=np.float32)
    res = {}
    for labs, mask in C['groups']:
        Y = C['Y'][mask][:, [C['beh'].index(b) for b in labs]]
        Xm = X if mask.all() else X[mask]
        r = ssp.align_columns(Xm, Y)
        dec = C['decile'][mask]
        sc = np.full((10, X.shape[1], len(labs)), np.nan)
        for d in range(10):
            sub = dec == d
            P = Y[sub].sum(0)
            if sub.sum() == 0:
                continue
            rd = ssp.align_columns(Xm[sub], Y[sub])['auc']
            ok = (P >= 10) & (sub.sum() - P >= 10)
            sc[d][:, ok] = rd[:, ok]
        scm = np.nanmean(sc, 0)
        for li, b in enumerate(labs):
            res[b] = {'auc': r['auc'][:, li], 'ap': r['ap'][:, li], 'prec_top': r['prec_top'][:, li],
                      'rate': r['rate'], 'base': float(r['base'][li]), 'n': int(mask.sum()),
                      'sc_auc': scm[:, li], 'sc_deciles_used': int(np.isfinite(sc[:, 0, li]).sum())}
    if 'disc_mask' in C:  # ants: Y2F vs B2F among frames with exactly one of the two
        m = C['disc_mask']
        y = C['Y'][m][:, C['beh'].index('groom_yellow')][:, None]
        a = ssp.align_columns(X[m], y)['auc'][:, 0]
        res['y_vs_b'] = {'auc_yellow': a, 'disc': np.maximum(a, 1 - a), 'n': int(m.sum()), 'base': float(y.mean())}
        cf = []
        for ha, hb in ((0, 1), (1, 0)):
            ma, mb = m & (C['half'] == ha), m & (C['half'] == hb)
            aa = ssp.align_columns(X[ma], C['Y'][ma][:, [C['beh'].index('groom_yellow')]])['auc'][:, 0]
            j = int(np.argmax(np.maximum(aa, 1 - aa)))
            ab = ssp.align_columns(X[mb][:, [j]], C['Y'][mb][:, [C['beh'].index('groom_yellow')]])['auc'][0, 0]
            cf.append({'neuron': j, 'select_auc': float(max(aa[j], 1 - aa[j])),
                       'test_disc': float(ab if aa[j] >= 0.5 else 1 - ab)})
        res['y_vs_b']['crossfit'] = cf
    for name, (v, ok) in C['proxies'].items():
        res[f'proxy_{name}'] = {'rho': spearman_cols(X[ok], v[ok]), 'n': int(ok.sum())}
    return path, res


def ants_dispersion(lab):
    """Mean pairwise centroid distance (tracking px) of the blue / yellow / focal ant per eval frame; NaN if any
    centroid is missing; and the tracker's blob count."""
    disp = np.full(len(lab), np.nan)
    nb = np.full(len(lab), -1)
    for (exp, obs), g in lab.groupby(['experiment', 'obs']):
        p = REPO / 'dataset/ants' / exp / 'tracking' / f'{obs}.csv'
        if not p.exists():
            continue
        t = pd.read_csv(p).set_index('frame_idx')
        t = t.reindex(g['frame_idx'].values)
        xy = [t[[f'{c}_x', f'{c}_y']].values for c in ('blue', 'yellow', 'focal')]
        d = (np.linalg.norm(xy[0] - xy[1], axis=1) + np.linalg.norm(xy[0] - xy[2], axis=1)
             + np.linalg.norm(xy[1] - xy[2], axis=1)) / 3
        disp[g.index.values] = d
        nb[g.index.values] = t['n_blobs'].fillna(-1).values
    return disp, nb


@torch.no_grad()
def cmd_evaluate(args):
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False
    out = Path(args.out_dir)
    loc = local_dir()
    lout = loc / 'out'
    for d in ('eval', 'align', 'codes_best', 'sheets'):
        (lout / d).mkdir(parents=True, exist_ok=True)
    idx = ssp.StoreIndex(args.domain)
    _, ev, _, _, _, eval_v = idx.split(args.domain, N_TRAIN[args.domain], 0)
    ev = np.sort(ev)
    if args.max_eval_frames:
        ev = ev[:args.max_eval_frames]
    lab, beh, valid = eval_labels(args.domain, idx, ev)
    lab.drop(columns=['frame_path']).to_parquet(lout / 'labels.parquet')
    log(f'{args.domain}: {len(ev):,} eval frames, {lab.obs.nunique()} videos; labels '
        + json.dumps({b: [int(valid[b].sum()), round(float(lab[b][valid[b]].mean()), 4)] for b in beh}))
    tok, pos, lens = stage(idx, ev, loc / 'eval')
    scales = scale_table(args.domain)
    models = {}
    for cfg in CONFIGS:
        for seed in args.seeds:
            pth = out / 'sae' / f'{cfg}_s{seed}' / 'sae.pt'
            sae, norm, _ = load_sae(pth, dev)
            models[(cfg, seed)] = (sae, norm)
    F_, m = len(ev), 1024
    sets = {}  # code-set name -> (cfg, seed, scale)
    for cfg in SCALE_NAMES:
        for seed in args.seeds:
            sets[f'{cfg}_s{seed}'] = (cfg, seed, cfg)
    for seed in args.seeds:
        for s in SCALE_NAMES:
            sets[f'joint@{s}_s{seed}'] = ('joint', seed, s)
    codes = {k: np.lib.format.open_memmap(loc / f'codes_{k}.npy', 'w+', np.float16, (F_, m)) for k in sets}
    st = {k: dict(sse=0.0, s1=torch.zeros(models[(c, sd)][0].d_in, dtype=torch.float64, device=dev), s2=0.0, n=0,
                  fire=torch.zeros(m, dtype=torch.float64, device=dev), l0=0.0) for k, (c, sd, s) in sets.items()}
    nwin = {s: 0 for s in scales}
    t0 = time.time()
    for f0, f1, T, P, L in iter_chunks(tok, pos, lens, args.chunk_frames, dev):
        nf = f1 - f0
        for si, (s, (side, stride, mf)) in enumerate(scales.items()):
            feats, frame, _ = window_tokens(T, P, L, side, stride, mf)
            nwin[s] += len(feats)
            for k, (cfg, seed, sc) in sets.items():
                if sc != s:
                    continue
                sae, norm = models[(cfg, seed)]
                mx = torch.zeros(nf, m, device=dev)
                a = st[k]
                for b0 in range(0, len(feats), 65536):
                    fb = feats[b0:b0 + 65536]
                    if cfg == 'joint':
                        fb = torch.cat([fb, onehot(len(fb), si, dev)], 1)
                    x = norm(fb)
                    z = sae.encode(x, mode='threshold')
                    xh = sae.decode(z)
                    a['s1'] += x.double().sum(0)
                    a['s2'] += float(x.double().pow(2).sum())
                    a['sse'] += float((x - xh).double().pow(2).sum())
                    a['n'] += len(x)
                    a['fire'] += (z > 0).sum(0).double()
                    a['l0'] += float((z > 0).sum())
                    mx.index_reduce_(0, frame[b0:b0 + 65536], z, 'amax', include_self=True)
                codes[k][f0:f1] = mx.cpu().numpy().astype(np.float16)
        if (f0 // args.chunk_frames) % 50 == 0:
            log(f'  encoded {f1:,}/{F_:,} frames [{time.time() - t0:.0f}s]')
    log(f'  encoding done in {time.time() - t0:.0f}s; eval windows ' + json.dumps(nwin))
    fire_rate = {}
    for k, (cfg, seed, s) in sets.items():
        codes[k].flush()
        a = st[k]
        tss = a['s2'] - float(a['s1'].pow(2).sum()) / a['n']
        fr = (a['fire'] / a['n']).cpu().numpy()
        fire_rate[k] = fr
        e = {'config': cfg, 'seed': seed, 'scale': s, 'n_windows': a['n'], 'fve': 1 - a['sse'] / tss,
             'l0_per_window': a['l0'] / a['n'], 'dead_frac': float((fr == 0).mean()), 'window_fire_rate': fr.tolist()}
        (lout / 'eval' / f'{k}.json').write_text(json.dumps(e))
        log(f'  {k}: FVE {e["fve"]:.4f} L0/window {e["l0_per_window"]:.2f} dead {100 * e["dead_frac"]:.1f}%')
    # joint read-outs: own scale (highest per-window firing rate) and max over all scales
    own = {}
    for seed in args.seeds:
        R = np.stack([fire_rate[f'joint@{s}_s{seed}'] for s in SCALE_NAMES])  # (5, m)
        own[seed] = R.argmax(0)
        share = R * np.array([nwin[s] for s in SCALE_NAMES])[:, None]
        share = share / np.maximum(share.sum(0, keepdims=True), 1e-12)
        np.savez(lout / 'eval' / f'joint_scales_s{seed}.npz', rate=R, share=share, own=own[seed],
                 scales=np.array(SCALE_NAMES))
        ko = np.lib.format.open_memmap(loc / f'codes_joint_own_s{seed}.npy', 'w+', np.float16, (F_, m))
        ka = np.lib.format.open_memmap(loc / f'codes_joint_allmax_s{seed}.npy', 'w+', np.float16, (F_, m))
        for r0 in range(0, F_, 20000):
            Z = np.stack([np.asarray(codes[f'joint@{s}_s{seed}'][r0:r0 + 20000]) for s in SCALE_NAMES])
            ko[r0:r0 + 20000] = np.take_along_axis(Z, own[seed][None, None, :].repeat(Z.shape[1], 1), 0)[0]
            ka[r0:r0 + 20000] = Z.max(0)
        ko.flush(); ka.flush()
        log(f'  joint s{seed}: own scale of the 1024 neurons ' + json.dumps(
            {s: int((own[seed] == i).sum()) for i, s in enumerate(SCALE_NAMES)}))
    all_sets = list(sets) + [f'joint_{r}_s{sd}' for sd in args.seeds for r in ('own', 'allmax')]
    # analysis context
    Y = lab[beh].values.astype(bool)
    groups = {}
    for b in beh:
        groups.setdefault(valid[b].tobytes(), (valid[b], []))[1].append(b)
    nfg = lens
    rank = np.argsort(np.argsort(nfg + np.random.default_rng(0).random(len(nfg)) * 1e-3))
    CTX.update(Y=Y, beh=beh, groups=[(labs, msk) for msk, labs in groups.values()],
               decile=np.minimum(rank * 10 // len(nfg), 9))
    # collective proxies (label-free): foreground spread from the mask; ants tracking dispersion
    sr = np.bincount(np.repeat(np.arange(F_), lens), weights=(pos // GRID).astype(float), minlength=F_)
    sc_ = np.bincount(np.repeat(np.arange(F_), lens), weights=(pos % GRID).astype(float), minlength=F_)
    sr2 = np.bincount(np.repeat(np.arange(F_), lens), weights=((pos // GRID).astype(float)) ** 2, minlength=F_)
    sc2 = np.bincount(np.repeat(np.arange(F_), lens), weights=((pos % GRID).astype(float)) ** 2, minlength=F_)
    n_ = np.maximum(lens, 1)
    spread = np.sqrt(np.maximum(sr2 / n_ - (sr / n_) ** 2 + sc2 / n_ - (sc_ / n_) ** 2, 0))
    proxies = {'fg_spread': (spread, lens >= 2), 'n_fg': (lens.astype(float), np.ones(F_, bool))}
    if args.domain == 'ants':
        disp, nb = ants_dispersion(lab)
        proxies['track_disp'] = (disp, np.isfinite(disp))
        proxies['track_disp_3blobs'] = (disp, np.isfinite(disp) & (nb == 3))
        y, b = lab['groom_yellow'].values, lab['groom_blue'].values
        CTX['disc_mask'] = y ^ b
        vids = sorted(lab.obs.unique())
        h = {v: i % 2 for i, v in enumerate(vids)}
        CTX['half'] = lab.obs.map(h).values
        ok = np.isfinite(disp)
        log(f'  tracking dispersion: {ok.sum():,}/{F_:,} frames with 3 centroids ({(ok & (nb == 3)).sum():,} with 3 '
            f'blobs); Spearman(dispersion, n_fg) {spearman_cols(lens[ok, None].astype(float), disp[ok])[0]:.3f}, '
            f'(dispersion, fg_spread) {spearman_cols(spread[ok, None], disp[ok])[0]:.3f}')
    CTX['proxies'] = proxies
    t0 = time.time()
    paths = [str(loc / f'codes_{k}.npy') for k in all_sets]
    with get_context('fork').Pool(args.workers) as pool:
        results = dict(pool.imap_unordered(analyse, paths))
    log(f'  analysis of {len(paths)} code sets in {time.time() - t0:.0f}s')
    A = {}
    for k in all_sets:
        r = results[str(loc / f'codes_{k}.npy')]
        A[k] = r
        flat = {}
        for lb, d in r.items():
            for q, v in d.items():
                if isinstance(v, np.ndarray):
                    flat[f'{lb}|{q}'] = v
        np.savez(lout / 'align' / f'{k}.npz', **flat)
    summary = summarise(args, A, all_sets, beh, own, lab)
    (lout / 'summary.json').write_text(json.dumps(summary, indent=1, default=float))
    # best-neuron codes (all frames) for later re-analysis
    for k in all_sets:
        js = sorted({int(np.argmax(A[k][b]['auc'])) for b in beh} | {int(np.argmax(A[k][b]['prec_top'])) for b in beh})
        Z = np.load(loc / f'codes_{k}.npy', mmap_mode='r')
        np.savez(lout / 'codes_best' / f'{k}.npz', neurons=np.array(js), codes=np.asarray(Z[:, js]))
    np.save(lout / 'rows.npy', idx.rows[ev])
    np.save(lout / 'n_fg.npy', lens)
    contact_sheets(args, A, lab, valid, tok, pos, lens, models, own, loc, lout, dev)
    print_table(summary)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copytree(lout, out, dirs_exist_ok=True)
    log(f'results copied to {out}')


def best_stats(r, b):
    d = r[b]
    j = int(np.argmax(d['auc']))
    sc = d['sc_auc']
    return {'neuron': j, 'auroc': float(d['auc'][j]), 'ap': float(d['ap'][j]), 'prec_top1': float(d['prec_top'][j]),
            'fire_rate': float(d['rate'][j]), 'base': d['base'], 'size_ctrl_auroc': float(sc[j]),
            'max_ap': float(d['ap'].max()), 'max_prec_top1': float(np.nanmax(d['prec_top'])),
            'max_prec_top1_neuron': int(np.nanargmax(d['prec_top'])),
            'max_size_ctrl_auroc': float(np.nanmax(sc)) if np.isfinite(sc).any() else float('nan'),
            'n_frames': d['n'], 'sc_deciles_used': d['sc_deciles_used']}


def summarise(args, A, all_sets, beh, own, lab):
    def kind(k):
        return k.rsplit('_s', 1)[0]
    kinds = list(dict.fromkeys(kind(k) for k in all_sets))
    per = {}
    for k in all_sets:
        seed = int(k.rsplit('_s', 1)[1])
        r = A[k]
        e = {b: best_stats(r, b) for b in beh}
        if kind(k) == 'joint_own':
            sh = np.load(Path(local_dir()) / 'out' / 'eval' / f'joint_scales_s{seed}.npz')
            for b in beh:
                j = e[b]['neuron']
                e[b]['own_scale'] = SCALE_NAMES[own[seed][j]]
                e[b]['window_fire_share_by_scale'] = dict(zip(SCALE_NAMES, sh['share'][:, j].round(4).tolist()))
                e[b]['window_fire_rate_by_scale'] = dict(zip(SCALE_NAMES, sh['rate'][:, j].round(5).tolist()))
        if 'y_vs_b' in r:
            d = r['y_vs_b']
            j = int(np.argmax(d['disc']))
            jy = e['groom_yellow']['max_prec_top1_neuron']
            jb = e['groom_blue']['max_prec_top1_neuron']
            e['y_vs_b'] = {'neuron': j, 'disc_auroc': float(d['disc'][j]), 'auc_yellow': float(d['auc_yellow'][j]),
                           'n_frames': d['n'], 'frac_yellow': d['base'],
                           'disc_at_best_y2f_neuron': float(d['disc'][e['groom_yellow']['neuron']]),
                           'disc_at_best_b2f_neuron': float(d['disc'][e['groom_blue']['neuron']]),
                           'disc_at_top1_y2f_neuron': float(d['disc'][jy]),
                           'disc_at_top1_b2f_neuron': float(d['disc'][jb]),
                           'crossfit_test_disc': float(np.mean([c['test_disc'] for c in d['crossfit']])),
                           'crossfit': d['crossfit']}
        for q in [q for q in r if q.startswith('proxy_')]:
            rho = r[q]['rho']
            j = int(np.argmax(np.abs(rho)))
            e[q] = {'neuron': j, 'rho': float(rho[j]), 'abs_rho': float(abs(rho[j])), 'n_frames': r[q]['n'],
                    'rho_at_best_groom_or_contact_neuron': float(rho[e[beh[0]]['neuron']])}
        per[k] = e
    agg = {}
    for kd in kinds:
        ks = [k for k in all_sets if kind(k) == kd]
        agg[kd] = {}
        for b in list(per[ks[0]]):
            vals = {q: [per[k][b][q] for k in ks] for q in per[ks[0]][b]
                    if isinstance(per[ks[0]][b][q], (int, float)) and q not in ('neuron', 'max_prec_top1_neuron')}
            agg[kd][b] = {q: {'mean': float(np.mean(v)), 'sd': float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                              'per_seed': v} for q, v in vals.items()}
            if kd == 'joint_own' and b in beh:
                agg[kd][b]['own_scale_per_seed'] = [per[k][b]['own_scale'] for k in ks]
    best_scale = {}
    for b in beh:
        m = {s: agg[s][b]['auroc']['mean'] for s in SCALE_NAMES}
        p = {s: agg[s][b]['max_prec_top1']['mean'] for s in SCALE_NAMES}
        best_scale[b] = {'by_auroc': max(m, key=m.get), 'by_max_prec_top1': max(p, key=p.get), 'auroc': m,
                         'max_prec_top1': p}
    return {'domain': args.domain, 'seeds': args.seeds, 'n_eval_frames': int(len(lab)),
            'n_eval_videos': int(lab.obs.nunique()), 'labels': beh, 'per_set': per, 'mean_sd': agg,
            'best_scale': best_scale}


def contact_sheets(args, A, lab, valid, tok, pos, lens, models, own, loc, lout, dev, n_show=16, per_video=2):
    """Seed 0: top-16 frames (<= 2 per video) of the best-AUROC neuron per label and config, the frame's
    argmax window outlined (green = label positive, red = negative). Only frames where the label is annotated
    (onLid: v3); y_vs_b: only frames with exactly one of Y2F / B2F (yellow box = Y2F, blue box = B2F)."""
    from PIL import Image, ImageDraw
    seed = args.seeds[0]
    st = np.r_[0, np.cumsum(lens)]
    scales = scale_table(args.domain)
    labs = ['nose_nose', 'nose_tail'] if args.domain == 'mice' else \
        ['groom_any', 'groom_yellow', 'groom_blue', 'onlid_yellow', 'onlid_blue']
    jobs = []
    for b in labs:
        for cfg in list(SCALE_NAMES) + ['joint_own']:
            k = f'{cfg}_s{seed}'
            jobs.append((b, cfg, k, int(np.argmax(A[k][b]['auc']))))
    if args.domain == 'ants':
        for cfg in list(SCALE_NAMES) + ['joint_own']:
            k = f'{cfg}_s{seed}'
            jobs.append(('y_vs_b', cfg, k, int(np.argmax(A[k]['y_vs_b']['disc']))))
    index = []
    for b, cfg, k, j in jobs:
        Z = np.load(loc / f'codes_{k}.npy', mmap_mode='r')
        x = np.asarray(Z[:, j], dtype=np.float32)
        lb = 'groom_yellow' if b == 'y_vs_b' else b
        elig = (lab['groom_yellow'].values ^ lab['groom_blue'].values) if b == 'y_vs_b' else valid[b]
        x = np.where(elig, x, 0)
        order = np.argsort(-x, kind='stable')
        pick, seen = [], {}
        for i in order:
            if x[i] <= 0 or len(pick) == n_show:
                break
            v = lab.obs.iat[i]
            if seen.get(v, 0) >= per_video:
                continue
            seen[v] = seen.get(v, 0) + 1
            pick.append(int(i))
        scale = cfg if cfg != 'joint_own' else SCALE_NAMES[own[seed][j]]
        si = SCALE_NAMES.index(scale)
        side, stride, mf = scales[scale]
        sae, norm = models[('joint' if cfg == 'joint_own' else cfg, seed)]
        th = 256
        sheet = Image.new('RGB', (4 * th, 4 * (th + 14)), 'white')
        dr = ImageDraw.Draw(sheet)
        n_pos = 0
        for q, i in enumerate(pick):
            T = torch.from_numpy(np.ascontiguousarray(tok[st[i]:st[i + 1]])).to(dev)
            P = torch.from_numpy(pos[st[i]:st[i + 1]].astype(np.int64)).to(dev)
            feats, _, win = window_tokens(T, P, torch.tensor([len(T)], device=dev), side, stride, mf)
            if cfg == 'joint_own':
                feats = torch.cat([feats, onehot(len(feats), si, dev)], 1)
            z = sae.encode(norm(feats), mode='threshold')[:, j]
            w = win[int(z.argmax())].tolist()
            im = Image.open(REPO / 'dataset' / lab.frame_path.iat[i]).convert('RGB')
            d2 = ImageDraw.Draw(im)
            c = 512 // GRID
            yv = bool(lab[lb].iat[i])
            n_pos += yv
            col = (0, 220, 0) if yv else (230, 0, 0)
            if b == 'y_vs_b':
                col = (240, 200, 0) if yv else (0, 120, 255)
            d2.rectangle([w[1] * c, w[0] * c, (w[1] + w[2]) * c - 1, (w[0] + w[2]) * c - 1], outline=col, width=4)
            im = im.resize((th, th))
            r_, c_ = divmod(q, 4)
            sheet.paste(im, (c_ * th, r_ * (th + 14)))
            dr.text((c_ * th + 2, r_ * (th + 14) + th), f'{lab.obs.iat[i]} f{lab.frame_idx.iat[i]} '
                    f'{"POS" if yv else "neg"} a={float(x[i]):.2f}', fill='black')
        fn = f'{b}__{cfg}_n{j}.jpg'
        sheet.save(lout / 'sheets' / fn, quality=85)
        index.append({'label': b, 'config': cfg, 'scale': scale, 'neuron': j, 'n_shown': len(pick),
                      'n_label_positive': int(n_pos), 'file': fn})
    (lout / 'sheets' / 'index.json').write_text(json.dumps(index, indent=1))
    log('  contact sheets: ' + json.dumps([(e['label'], e['config'], e['n_label_positive'], e['n_shown'])
                                           for e in index]))


def print_table(S):
    """Config x label table of summary.json (mean +- sd over seeds)."""
    ms = S['mean_sd']
    f = lambda d, q: f"{d[q]['mean']:.3f}+-{d[q]['sd']:.3f}"  # noqa: E731
    log(f"\n{S['domain']}: {S['n_eval_frames']:,} eval frames, {S['n_eval_videos']} videos, seeds {S['seeds']}")
    log('label | config | AUROC | AP@best | top1%@best | max top1% | max AP | size-ctrl AUROC@best | max size-ctrl '
        '| fire rate | base')
    for b in S['labels']:
        for kd in ms:
            if b not in ms[kd]:
                continue
            d = ms[kd][b]
            extra = f" own={','.join(d['own_scale_per_seed'])}" if 'own_scale_per_seed' in d else ''
            log(f"{b} | {kd} | {f(d, 'auroc')} | {f(d, 'ap')} | {f(d, 'prec_top1')} | {f(d, 'max_prec_top1')} | "
                f"{f(d, 'max_ap')} | {f(d, 'size_ctrl_auroc')} | {f(d, 'max_size_ctrl_auroc')} | "
                f"{d['fire_rate']['mean']:.3f} | {d['base']['mean']:.4f}{extra}")
    for kd in ms:
        if 'y_vs_b' in ms[kd]:
            d = ms[kd]['y_vs_b']
            log(f"y_vs_b | {kd} | disc AUROC (in-sample best) {f(d, 'disc_auroc')} | cross-fitted "
                f"{f(d, 'crossfit_test_disc')} | at best Y2F neuron {f(d, 'disc_at_best_y2f_neuron')} | at best B2F "
                f"neuron {f(d, 'disc_at_best_b2f_neuron')}")
    for kd in ms:
        for q in [q for q in ms[kd] if q.startswith('proxy_')]:
            log(f"{q} | {kd} | best neuron |rho| {f(ms[kd][q], 'abs_rho')} | rho at best {S['labels'][0]} neuron "
                f"{f(ms[kd][q], 'rho_at_best_groom_or_contact_neuron')}")
    log('best scale per label (per-scale SAEs, mean AUROC): ' + json.dumps(
        {b: (v['by_auroc'], v['by_max_prec_top1']) for b, v in S['best_scale'].items()}))


def cmd_table(args):
    print_table(json.loads((Path(args.out_dir) / 'summary.json').read_text()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['selftest', 'train', 'evaluate', 'table'])
    p.add_argument('--domain', default='mice', choices=['mice', 'ants'])
    p.add_argument('--out-dir')
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--configs', nargs='+', default=list(CONFIGS))
    p.add_argument('--cap', type=int, default=4_000_000)
    p.add_argument('--steps', type=int, default=6000)
    p.add_argument('--pool-seed', type=int, default=0)
    p.add_argument('--chunk-frames', type=int, default=128)
    p.add_argument('--max-train-frames', type=int, default=0, help='smoke tests only')
    p.add_argument('--max-eval-frames', type=int, default=0, help='smoke tests only')
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    {'selftest': cmd_selftest, 'train': cmd_train, 'evaluate': cmd_evaluate, 'table': cmd_table}[args.cmd](args)


if __name__ == '__main__':
    main()
