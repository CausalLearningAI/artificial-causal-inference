"""
Spatial-SAE pilot (src/eci/spatial_sae.py) against tokenwise SAEs on the ECI foreground token stores, judged by
per-frame alignment of single latents with human behaviour annotations (the alignment audit's harness:
auc_ap_columns of <scratchpad>/eci_alignment/common.py, copied here as align_columns so the repo does not depend on the
scratchpad).

Data: a foreground token store of scripts/eci/fg_extract_train.py (1 fps = every 5th frame, all foreground patches of
each frame with their 32 x 32 grid position). Video split, seeded (--split-seed):
    mice  train = --n-train-videos videos WITHOUT any behaviour annotation; eval = every store frame of the annotated
          videos that carries all three labels (Y_nn, Y_np, Y_nt) -> the SAEs never see an evaluation video
    ants  every video is annotated: train = --n-train-videos random videos, eval = the store frames of the others
No label is used in training.

Models (one config each, no hyperparameter search for any of them; same tokens, same TokenNorm fit, same b_dec
init = geometric median, same number of epochs):
    btk         src/eci/sae.py MatryoshkaBatchTopKSAE with ONE prefix (= plain BatchTopK), the repo's recipe
                (AuxK, lr 5e-4, 500 warmup, cosine to 10%, batches of 4096 random tokens)
    matryoshka  the same with prefixes 128/256/512/1024 (exactly the deployed fg448 / antsfg architecture)
    spatial     SpatialBatchTopKSAE, context attention, k_S:k_R = 12:4, mu = 1e-3, reanimation 1e-3, the paper's recipe
                (lr 1e-3, 100 warmup, cosine to 1e-5, batches of 32 whole frames ~ 4.7k tokens for mice)
    spatial_static  the paper's static-estimator control (same as spatial, smooth code from the patch itself)
All: 1024 latents, k = 16.

Steps:
    train   --model M --out-dir D                -> D/sae.pt, D/train.json
    encode  --ckpt NAME=PATH ... --out-dir D     -> D/NAME/{codes_max,codes_mean}.npy (float16, eval-frame order),
            D/rows.npy (annotations.csv row of each eval frame), D/NAME/eval.json (FVE, L0, dead latents on all eval
            tokens; threshold inference). A checkpoint of src/eci/sae.py (deployed) is encoded tokenwise.
    align   --codes-dir D [--deployed TAG]       -> D/align_top.csv, D/align_best.json (best single latent per
            behaviour, max and mean pooling); --deployed also scores the deployed SAE's stored codes_max / codes_mean
            on the same eval rows (read from the domain's codes dir)

Usage:
    python scripts/eci/spatial_sae_pilot.py train --domain mice --model spatial --out-dir <D>/spatial
    python scripts/eci/spatial_sae_pilot.py encode --domain mice --ckpt spatial=<D>/spatial/sae.pt \
        deployed=dataset/mice/v1/eci/sae/matryoshka_btk_1024_k16_fg448_s0/sae.pt --out-dir <D>/codes
    python scripts/eci/spatial_sae_pilot.py align --domain mice --codes-dir <D>/codes --deployed fg448
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.eci.domain import get_domain  # noqa: E402
from src.eci.foreground import FgTokenStore  # noqa: E402
from src.eci.sae import (MatryoshkaBatchTopKSAE, TokenNorm, _train_step, geometric_median, load_sae,  # noqa: E402
                         save_checkpoint)
from src.eci.spatial_sae import SpatialBatchTopKSAE, load_spatial, lr_paper, neighbor_index, save_spatial  # noqa: E402

STORES = {'mice': 'dataset/mice/v1/eci/train_tokens/dinov2_base_l-1_fg448_fps1',
          'ants': 'dataset/ants/eci/train_tokens/dinov2_base_l-1_antsfg_fps1'}
GRID = 32
log = lambda s: print(s, flush=True)  # noqa: E731


# ---------------------------------------------------------------------------------------------- data
def labelled_frames(domain):
    """annotations.csv rows with behaviour labels -> DataFrame(row, obs, behaviours...), behaviour names.
    Same definitions as the alignment audit (eci_alignment/common.py load_frames)."""
    dom = get_domain(domain)
    if domain == 'mice':
        a = pd.read_csv(dom.ann_path, usecols=['observation_id', 'Y_nn', 'Y_np', 'Y_nt'])
        ok = a[['Y_nn', 'Y_np', 'Y_nt']].notna().all(1).values
        sub = a[ok]
        nn, np_, nt = (sub[c].values > 0 for c in ('Y_nn', 'Y_np', 'Y_nt'))
        f = pd.DataFrame({'row': np.flatnonzero(ok), 'obs': sub['observation_id'].values,
                          'nose_nose': nn | np_, 'nose_tail': nt, 'nn_mutual': nn, 'np_directional': np_,
                          'any_contact': nn | np_ | nt})
        beh = ['nose_nose', 'nose_tail', 'nn_mutual', 'np_directional', 'any_contact']
    else:
        a = pd.read_csv(dom.ann_path, usecols=['observation_id', 'Y_Y2F', 'Y_B2F'])
        y, b = a['Y_Y2F'].fillna(0).values > 0, a['Y_B2F'].fillna(0).values > 0
        f = pd.DataFrame({'row': np.arange(len(a)), 'obs': a['observation_id'].values,
                          'groom_any': y | b, 'groom_yellow': y, 'groom_blue': b})
        beh = ['groom_any', 'groom_yellow', 'groom_blue']
    return f, beh


class StoreIndex:
    """Per-frame layout of a foreground token store: shard, first token, n tokens, annotations.csv row, video."""

    def __init__(self, domain):
        self.store = FgTokenStore(REPO / STORES[domain])
        sh, start, nfg, rows = [], [], [], []
        for s, d in enumerate(self.store.dirs):
            z = np.load(d / 'frames.npz')
            n = z['n_fg'].astype(np.int64)
            if n.sum() != self.store.sizes[s]:
                raise RuntimeError(f'{d}: n_fg sums to {n.sum()}, shard has {self.store.sizes[s]} tokens')
            sh.append(np.full(len(n), s)); start.append(np.r_[0, np.cumsum(n)[:-1]]); nfg.append(n); rows.append(z['rows'])
        self.shard, self.start, self.nfg = np.concatenate(sh), np.concatenate(start), np.concatenate(nfg)
        self.rows = np.concatenate(rows).astype(np.int64)
        obs = pd.read_csv(get_domain(domain).ann_path, usecols=['observation_id'])['observation_id'].values
        self.obs = obs[self.rows]

    def split(self, domain, n_train, seed):
        """-> (train frame idx, eval frame idx, eval labels DataFrame aligned to eval idx, behaviours)."""
        lab, beh = labelled_frames(domain)
        videos = np.array(sorted(set(self.obs)))
        annotated = set(lab['obs'])
        rng = np.random.default_rng(seed)
        if domain == 'mice':
            pool = np.array([v for v in videos if v not in annotated])
            train_v = set(rng.choice(pool, min(n_train, len(pool)), replace=False))
            eval_v = annotated
        else:
            train_v = set(rng.choice(videos, n_train, replace=False))
            eval_v = set(videos) - train_v
        tr = np.flatnonzero(np.isin(self.obs, list(train_v)) & (self.nfg > 0))
        lab = lab.set_index('row')
        ev = np.flatnonzero(np.isin(self.obs, list(eval_v)) & np.isin(self.rows, lab.index.values))
        return tr, ev, lab.loc[self.rows[ev]].reset_index(), beh, sorted(train_v), sorted(eval_v)

    def load(self, frames, with_pos=True):
        """Tokens of the given frames (sorted), contiguous per frame -> (tokens (N, d) fp16 np, pos (N,), lens (F,))."""
        frames = np.sort(frames)
        lens = self.nfg[frames]
        out = np.empty((int(lens.sum()), self.store.dim), dtype=np.float16)
        pos = np.empty(int(lens.sum()), dtype=np.int16)
        o = 0
        for s in np.unique(self.shard[frames]):
            fs = frames[self.shard[frames] == s]
            tok, ps = self.store.tokens(s), self.store.pos(s)
            st, n = self.start[fs], self.nfg[fs]
            # read contiguous runs of frames in one slice
            brk = np.flatnonzero(st[1:] != st[:-1] + n[:-1]) + 1
            for a, b in zip(np.r_[0, brk], np.r_[brk, len(fs)]):
                lo, hi = st[a], st[b - 1] + n[b - 1]
                out[o:o + hi - lo] = tok[lo:hi]
                pos[o:o + hi - lo] = ps[lo:hi]
                o += hi - lo
        assert o == len(out)
        return out, pos, lens


# ---------------------------------------------------------------------------------------------- train
def cmd_train(args):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = torch.device('cuda')
    idx = StoreIndex(args.domain)
    tr, ev, _, _, train_v, eval_v = idx.split(args.domain, args.n_train_videos, args.split_seed)
    t0 = time.time()
    tok, pos, lens = idx.load(tr)
    log(f'{args.domain}: {len(train_v)} train videos, {len(tr):,} frames, {len(tok):,} tokens loaded in '
        f'{time.time() - t0:.0f}s; {len(eval_v)} eval videos')
    tok_t = torch.from_numpy(tok)
    pos_t = torch.from_numpy(pos.astype(np.int64))
    starts = torch.from_numpy(np.r_[0, np.cumsum(lens)[:-1]])
    lens_t = torch.from_numpy(lens)
    g = np.random.default_rng(args.seed)
    init = tok_t[torch.from_numpy(np.sort(g.choice(len(tok), min(500_000, len(tok)), replace=False)))].to(dev)
    norm = TokenNorm(tok.shape[1]).to(dev).fit(init, seed=args.seed)

    if args.model in ('btk', 'matryoshka'):
        prefixes = (1024,) if args.model == 'btk' else (128, 256, 512, 1024)
        sae = MatryoshkaBatchTopKSAE(d_in=tok.shape[1], n_latents=1024, prefixes=prefixes, k=16, seed=args.seed).to(dev)
    else:
        sae = SpatialBatchTopKSAE(d_in=tok.shape[1], n_latents=1024, k=16, k_s=12,
                                  estimator='context' if args.model == 'spatial' else 'static', grid=(GRID, GRID),
                                  mu_s=1e-3, mu_r=1e-3, reanim_coef=1e-3, seed=args.seed).to(dev)
    with torch.no_grad():
        sae.b_dec.copy_(geometric_median(norm(init[:100_000])))
    del init
    n_params = sum(p.numel() for p in sae.parameters())
    gen = torch.Generator().manual_seed(args.seed)
    history, step, tt = [], 0, time.time()
    if args.model in ('btk', 'matryoshka'):
        bs = 4096
        lr, warmup = 5e-4, 500
        opt = torch.optim.Adam(sae.parameters(), lr=lr, betas=(0.9, 0.999))
        spe = len(tok) // bs
        total = args.epochs * spe
        log(f'{args.model}: {n_params:,} params, {total:,} steps of {bs} tokens')
        for ep in range(args.epochs):
            perm = torch.randperm(len(tok), generator=gen)
            for b in range(spe):
                x = norm(tok_t[perm[b * bs:(b + 1) * bs]].to(dev, non_blocking=True))
                logs, gn = _train_step(sae, opt, x, step, total, lr, warmup, 1.0)
                if step % 1000 == 0 or step == total - 1:
                    logs.update(step=step, epoch=ep, elapsed_s=round(time.time() - tt, 1))
                    history.append(logs)
                    log(f"  step {step}/{total} fve {logs['fve']:.4f} l0 {logs['l0']:.1f} dead {logs['n_dead']} "
                        f"aux {logs['aux']:.3f} {logs['elapsed_s']}s")
                step += 1
    else:
        fpb = args.frames_per_batch
        opt = torch.optim.Adam(sae.parameters(), lr=1e-3, betas=(0.9, 0.999))
        spe = len(tr) // fpb
        total = args.epochs * spe
        log(f'{args.model}: {n_params:,} params (c_S {sae.c_s}, c_R {sae.c_r}, kappa {sae.kappa:.3f}), '
            f'{total:,} steps of {fpb} frames (~{fpb * lens.mean():.0f} tokens)')
        n_skip = 0
        for ep in range(args.epochs):
            perm = torch.randperm(len(tr), generator=gen)
            for b in range(spe):
                fb = perm[b * fpb:(b + 1) * fpb]
                ln = lens_t[fb]
                rep = torch.repeat_interleave(torch.arange(fpb), ln)
                off = torch.arange(int(ln.sum())) - torch.repeat_interleave(torch.cumsum(ln, 0) - ln, ln)
                ti = starts[fb][rep] + off
                x = norm(tok_t[ti].to(dev, non_blocking=True))
                frame = rep.to(dev)
                nbr = neighbor_index(frame, pos_t[ti].to(dev), GRID, GRID, sae.offsets)
                for pg in opt.param_groups:
                    pg['lr'] = lr_paper(step, total)
                loss, logs = sae.forward_train(x, nbr, frame, fpb)
                if not torch.isfinite(loss):
                    n_skip += 1  # paper: steps with a non-finite loss are skipped
                    step += 1
                    continue
                opt.zero_grad(set_to_none=True)
                loss.backward()
                sae.remove_parallel_grad()
                gn = torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
                opt.step()
                sae.normalize_decoder()
                if step % 1000 == 0 or step == total - 1:
                    n_dead = int((sae.tokens_since_fired >= 2_000_000).sum())
                    logs.update(step=step, epoch=ep, grad_norm=float(gn), n_dead_2M=n_dead,
                                elapsed_s=round(time.time() - tt, 1))
                    history.append(logs)
                    log(f"  step {step}/{total} " + ' '.join(f'{a} {v:.4g}' if isinstance(v, float) else f'{a} {v}'
                                                            for a, v in logs.items()))
                step += 1
        if n_skip:
            log(f'{n_skip} non-finite steps skipped')
    train_s = time.time() - tt
    sae.eval()
    extra = {'model': args.model, 'domain': args.domain, 'train_videos': train_v, 'args': vars(args)}
    if args.model in ('btk', 'matryoshka'):
        save_checkpoint(out / 'sae.pt', sae, norm, extra=extra)
    else:
        save_spatial(out / 'sae.pt', sae, norm, extra=extra)
    (out / 'train.json').write_text(json.dumps({'model': args.model, 'n_params': n_params, 'n_train_frames': len(tr),
                                                'n_train_tokens': len(tok), 'n_train_videos': len(train_v),
                                                'steps': total, 'train_time_s': round(train_s, 1),
                                                'gpu': torch.cuda.get_device_name(0), 'args': vars(args),
                                                'history': history}, indent=1))
    log(f'trained in {train_s:.0f}s -> {out}')


# ---------------------------------------------------------------------------------------------- encode
def _load_any(path, dev):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    if ck.get('kind') == 'spatial_btk':
        sae, norm, _ = load_spatial(path, dev)
        return 'spatial', sae, norm
    sae, norm, _ = load_sae(path, dev)
    return 'tokenwise', sae, norm


@torch.no_grad()
def cmd_encode(args):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False  # exact fp32 inference
    idx = StoreIndex(args.domain)
    _, ev, lab, beh, _, eval_v = idx.split(args.domain, args.n_train_videos, args.split_seed)
    ev = np.sort(ev)
    if args.max_eval_frames:  # smoke tests only
        ev = ev[:args.max_eval_frames]
    lab = lab.set_index('row').loc[idx.rows[ev]].reset_index()
    np.save(out / 'rows.npy', idx.rows[ev])
    lab.to_parquet(out / 'labels.parquet')
    models = {}
    for spec in args.ckpt:
        name, path = spec.split('=', 1)
        models[name] = _load_any(path, dev)
    F_, m = len(ev), 1024
    acc = {}
    for name, (kind, sae, norm) in models.items():
        (out / name).mkdir(exist_ok=True)
        acc[name] = dict(mx=np.lib.format.open_memmap(out / name / 'codes_max.npy', 'w+', np.float16, (F_, m)),
                         mn=np.lib.format.open_memmap(out / name / 'codes_mean.npy', 'w+', np.float16, (F_, m)),
                         sse=0.0, fire=torch.zeros(m, dtype=torch.long, device=dev), l0=0.0, l0s=0.0,
                         s1=torch.zeros(idx.store.dim, dtype=torch.float64, device=dev), s2=0.0)
    n_tok = 0
    t0 = time.time()
    # chunks of ~args.chunk_frames eval frames, in store order (contiguous reads)
    for c0 in range(0, F_, args.chunk_frames):
        fr = ev[c0:c0 + args.chunk_frames]
        tok, pos, lens = idx.load(fr)
        x_raw = torch.from_numpy(tok).to(dev)
        pos_t = torch.from_numpy(pos.astype(np.int64)).to(dev)
        frame_all = torch.repeat_interleave(torch.arange(len(fr), device=dev), torch.from_numpy(lens).to(dev))
        starts = np.r_[0, np.cumsum(lens)]
        for name, (kind, sae, norm) in models.items():
            a = acc[name]
            sub = 64  # frames per GPU batch (spatial attention memory)
            for b0 in range(0, len(fr), sub):
                lo, hi = starts[b0], starts[min(b0 + sub, len(fr))]
                x = norm(x_raw[lo:hi])
                frame = frame_all[lo:hi] - b0
                if kind == 'spatial':
                    nbr = neighbor_index(frame, pos_t[lo:hi], GRID, GRID, sae.offsets)
                    S, R = sae.encode(x, nbr)
                    z = torch.cat([S, R], 1)
                    xh = sae.decode(S, R)
                    a['l0s'] += float((S > 0).sum())
                else:
                    z = sae.encode(x, mode='threshold')
                    xh = sae.decode(z)
                nf = int(frame.max()) + 1
                # total SS about the eval mean, in each model's own normalised space (FVE is invariant to it)
                a['s1'] += x.double().sum(0)
                a['s2'] += float(x.double().pow(2).sum())
                a['sse'] += float((x - xh).double().pow(2).sum())
                a['fire'] += (z > 0).sum(0)
                a['l0'] += float((z > 0).sum())
                mx = torch.zeros(nf, m, device=dev).index_reduce_(0, frame, z, 'amax', include_self=True)
                sm = torch.zeros(nf, m, device=dev).index_add_(0, frame, z)
                cnt = torch.bincount(frame, minlength=nf).clamp_min(1)[:, None].float()
                a['mx'][c0 + b0:c0 + b0 + nf] = mx.cpu().numpy().astype(np.float16)
                a['mn'][c0 + b0:c0 + b0 + nf] = (sm / cnt).cpu().numpy().astype(np.float16)
        n_tok += len(tok)
        log(f'  {min(c0 + args.chunk_frames, F_):,}/{F_:,} eval frames, {n_tok:,} tokens [{time.time() - t0:.0f}s]')
    for name, (kind, sae, norm) in models.items():
        a = acc[name]
        a['mx'].flush(); a['mn'].flush()
        total_ss = a['s2'] - float(a['s1'].pow(2).sum()) / n_tok
        fire = a['fire'].cpu().numpy()
        res = {'kind': kind, 'n_eval_frames': F_, 'n_eval_tokens': n_tok, 'n_eval_videos': len(eval_v),
               'fve': 1 - a['sse'] / total_ss, 'l0_per_token': a['l0'] / n_tok,
               'dead_frac': float((fire == 0).mean()), 'n_dead': int((fire == 0).sum()),
               'fire_rate': (fire / n_tok).tolist()}
        if kind == 'spatial':
            res.update(l0_smooth=a['l0s'] / n_tok, l0_innov=(a['l0'] - a['l0s']) / n_tok,
                       dead_frac_smooth=float((fire[:sae.c_s] == 0).mean()),
                       dead_frac_innov=float((fire[sae.c_s:] == 0).mean()), c_s=sae.c_s, c_r=sae.c_r)
        (out / name / 'eval.json').write_text(json.dumps(res))
        log(f'{name}: FVE {res["fve"]:.4f}  L0/token {res["l0_per_token"]:.2f}  dead {100 * res["dead_frac"]:.1f}% '
            + (f'(smooth L0 {res["l0_smooth"]:.2f}, innov L0 {res["l0_innov"]:.2f}, dead S {res["dead_frac_smooth"]:.3f}'
               f' R {res["dead_frac_innov"]:.3f})' if kind == 'spatial' else ''))


# ---------------------------------------------------------------------------------------------- align
def align_columns(X, Y, top_frac=0.01):
    """Copy of the alignment audit's auc_ap_columns (eci_alignment/common.py): per column of X (n, m) >= 0 and label
    column of Y (n, L) bool -> AUROC (ties averaged), AP (sklearn step definition), precision in the top 1%, rate."""
    n, m = X.shape
    L = Y.shape[1]
    P = Y.sum(0).astype(np.float64)
    N = n - P
    k_top = max(1, int(round(top_frac * n)))
    auc = np.zeros((m, L)); ap = np.zeros((m, L)); ptop = np.zeros((m, L)); ntop = np.zeros(m, int)
    rate = (X > 0).mean(0)
    Yf = Y.astype(np.int64)
    for j in range(m):
        x = X[:, j]
        if (x < 0).any():
            raise ValueError('activations must be >= 0 (zeros are one tie group)')
        nz = np.flatnonzero(x > 0)
        o = nz[np.argsort(-x[nz], kind='stable')]
        xs = x[o]
        ys = Yf[o]
        last = np.r_[np.flatnonzero(np.diff(xs) != 0), len(o) - 1] if len(o) else np.zeros(0, int)
        ctp = np.cumsum(ys, 0)
        tps = ctp[last].astype(np.float64) if len(o) else np.zeros((0, L))
        fps = (last + 1)[:, None] - tps
        if len(o) < n:
            tps = np.vstack([tps, P[None]])
            fps = np.vstack([fps, N[None]])
        tpr = np.vstack([np.zeros(L), tps / P])
        fpr = np.vstack([np.zeros(L), fps / N])
        auc[j] = np.trapezoid(tpr, fpr, axis=0) if hasattr(np, 'trapezoid') else np.trapz(tpr, fpr, axis=0)
        prec = tps / (tps + fps)
        rec = tps / P
        ap[j] = (np.diff(np.vstack([np.zeros(L), rec]), axis=0) * prec).sum(0)
        k = min(k_top, len(o))
        ntop[j] = k
        ptop[j] = ctp[k - 1] / k if k > 0 else np.nan
    return {'auc': auc, 'ap': ap, 'prec_top': ptop, 'n_top': ntop, 'rate': rate, 'base': P / n}


def cmd_align(args):
    d = Path(args.codes_dir)
    rows = np.load(d / 'rows.npy')
    lab = pd.read_parquet(d / 'labels.parquet')
    _, beh = labelled_frames(args.domain)
    assert (lab['row'].values == rows).all()
    Y = lab[beh].values.astype(bool)
    log(f'{args.domain}: {len(rows):,} eval frames, {lab.obs.nunique()} videos; base rates '
        + json.dumps(dict(zip(beh, Y.mean(0).round(4).tolist()))))
    sets = {p.name: p for p in sorted(d.iterdir()) if (p / 'codes_max.npy').exists()}
    if args.deployed:
        cd = get_domain(args.domain).codes_root / f'matryoshka_btk_1024_k16_{args.deployed}_s0'
        sets[f'stored_{args.deployed}'] = cd
    table, best = [], {}
    for name, p in sets.items():
        for pool in args.pools:
            Z = np.load(p / f'codes_{pool}.npy', mmap_mode='r')
            X = np.asarray(Z[rows] if name.startswith('stored_') else Z, dtype=np.float32)
            t0 = time.time()
            res = align_columns(X, Y)
            np.savez(d / f'align_{name}_{pool}.npz', beh=np.array(beh), **res)
            for li, b in enumerate(beh):
                o = np.argsort(-res['auc'][:, li])[:10]
                for r, j in enumerate(o):
                    table.append({'sae': name, 'pool': pool, 'behaviour': b, 'rank': r + 1, 'neuron': int(j),
                                  'auroc': res['auc'][j, li], 'ap': res['ap'][j, li], 'base_rate': res['base'][li],
                                  'prec_top1pct': res['prec_top'][j, li], 'fire_rate': res['rate'][j]})
                ja = int(np.argmax(res['ap'][:, li]))
                best[f'{name}|{pool}|{b}'] = {'auroc_neuron': int(o[0]), 'auroc': float(res['auc'][o[0], li]),
                                              'ap_at_auroc_neuron': float(res['ap'][o[0], li]),
                                              'prec_top1pct_at_auroc_neuron': float(res['prec_top'][o[0], li]),
                                              'max_ap_neuron': ja, 'max_ap': float(res['ap'][ja, li]),
                                              'max_prec_top1pct': float(np.nanmax(res['prec_top'][:, li])),
                                              'base_rate': float(res['base'][li])}
            log(f'{name} {pool}: best (neuron, AUROC, max AP, max top-1% precision) '
                + json.dumps({b: (best[f"{name}|{pool}|{b}"]["auroc_neuron"], round(best[f"{name}|{pool}|{b}"]["auroc"], 3),
                                  round(best[f"{name}|{pool}|{b}"]["max_ap"], 3),
                                  round(best[f"{name}|{pool}|{b}"]["max_prec_top1pct"], 3)) for b in beh})
                + f' [{time.time() - t0:.0f}s]')
            if args.check_equal and name.startswith('stored_') and 'deployed' in sets:
                Xd = np.asarray(np.load(sets['deployed'] / f'codes_{pool}.npy', mmap_mode='r'), dtype=np.float32)
                same_on = ((Xd > 0) == (X > 0)).mean()
                r = np.corrcoef(Xd[:, :64].ravel(), X[:, :64].ravel())[0, 1]
                log(f'  check: deployed re-encoded from the store vs stored {args.deployed} codes_{pool}: '
                    f'firing pattern agrees on {100 * same_on:.2f}% of (frame, latent), corr (first 64 latents) {r:.4f}, '
                    f'max |diff| {np.abs(Xd - X).max():.3f}')
    pd.DataFrame(table).to_csv(d / 'align_top.csv', index=False)
    (d / 'align_best.json').write_text(json.dumps(best, indent=1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['train', 'encode', 'align'])
    p.add_argument('--domain', default='mice', choices=list(STORES))
    p.add_argument('--model', choices=['btk', 'matryoshka', 'spatial', 'spatial_static'])
    p.add_argument('--out-dir')
    p.add_argument('--n-train-videos', type=int, default=100)
    p.add_argument('--split-seed', type=int, default=0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--frames-per-batch', type=int, default=32)
    p.add_argument('--ckpt', nargs='*', default=[])
    p.add_argument('--chunk-frames', type=int, default=12_000)
    p.add_argument('--max-eval-frames', type=int, default=0, help='encode: first N eval frames only (smoke test)')
    p.add_argument('--codes-dir')
    p.add_argument('--deployed', default=None)
    p.add_argument('--pools', nargs='+', default=['max', 'mean'])
    p.add_argument('--check-equal', action='store_true', help='align: compare the re-encoded deployed SAE codes '
                   '(--ckpt deployed=...) with the stored ones')
    args = p.parse_args()
    {'train': cmd_train, 'encode': cmd_encode, 'align': cmd_align}[args.cmd](args)


if __name__ == '__main__':
    main()
