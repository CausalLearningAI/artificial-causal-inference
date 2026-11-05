"""
ECI v2.6: static-input SAEs where time enters only through the loss (pre-registration scripts/eci/v2/v26_prereg.md).
Tokens / edge rule / stream = v2.5 (v25_train: store v2.1_m5, frames 2 <= t < T - 2, windows of 100 frames, chunks of 256,
rng 20261105 + pass, per-seed permutation, all arms of a seed see the same batches). Input of every arm: x_t (1024).
Arms (seeds 2601-2603):
  N0wc    4096 latents, prefixes 128/256/512/2048/4096, plain Matryoshka loss (= v2.5 N0w recipe)
  PRED    as N0wc + a second decoder D1 (W_dec1, b_dec1) on the SAME code z_t reconstructing x_{t+2}; loss = L(x_t) + LAM L1(x_{t+2})
          (Matryoshka prefix sum on both; W_dec1 init = W_dec, b_dec1 init = b_dec; x_{t+2} normalised by the arm's TokenNorm)
  SMOOTH  as N0wc + MU * mean_tokens || z_t - z_{t+2} ||_1, z_{t+2} = batch-top-k code of x_{t+2} (same patch), gradient through both
  RW      as N0wc with per-token loss weights w(x) on the reconstruction terms (AuxK unweighted): k-means (k = 256) cluster of x,
          w = f_c^{-1/2} normalised to token mean 1, clipped at 10 (step kmeans; OUT26/<dom>_rw.npz)
  N0x     8192 latents, prefixes 128/256/512/2048/4096/8192, plain loss
Inference / readout for every arm: threshold-mode codes of x_t from encoder + D0 (load_sae-compatible checkpoint; PRED's D1 is
saved separately in dec1.pt and never used for readout).
Steps: kmeans --dom D (CPU) | featcheck --dom D (CPU) | train --dom D (GPU) | rows --dom D (copy v2.5 rows)
"""
import argparse
import json
import math
import os
import shutil
import socket
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
for p in ('', 'scripts/eci/v2'):
    sys.path.insert(0, str(REPO / p))
import v25_train as TR  # noqa: E402

BEE26 = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_6')
OUT26 = REPO / 'results/vision/eci_v2_6'
L, BS = TR.L, TR.BS
SEEDS = (2601, 2602, 2603)
ARMS = ('N0wc', 'PRED', 'SMOOTH', 'RW', 'N0x')
LAM = 1.0
SMOOTH_FRAC = 0.10
STEPS = {'mice': 50_000, 'ants': 25_561}  # mice ~0.27 pass (budget), ants 1 pass = floor(104,700,084 / 4096)
K_RW, CLIP_RW = 256, 10.0
log = TR.log


def shape_of(arm):
    if arm == 'N0x':
        return 8192, (128, 256, 512, 2048, 4096, 8192)
    return 4096, (128, 256, 512, 2048, 4096)


def mode_of(arm):
    return {'PRED': 'pred', 'SMOOTH': 'smooth', 'RW': 'rw'}.get(arm, 'plain')


def make_sae_class():
    import torch
    import torch.nn as nn
    from src.eci.sae import MatryoshkaBatchTopKSAE

    class SAE26(MatryoshkaBatchTopKSAE):
        def __init__(self, mode='plain', lam=LAM, mu=0.0, **kw):
            super().__init__(**kw)
            self.mode, self.lam, self.mu = mode, lam, mu
            if mode == 'pred':
                self.W_dec1 = nn.Parameter(self.W_dec.data.clone())
                self.b_dec1 = nn.Parameter(torch.zeros(self.d_in))

        def mrecon(self, z, W, b, x, w=None):
            recon, losses, s = b.expand_as(x), [], 0
            for m in self.prefixes:
                recon = recon + z[:, s:m] @ W[s:m]
                e = (recon - x).pow(2).sum(1)
                losses.append((e * w).mean() if w is not None else e.mean())
                s = m
            return losses, recon

        def coherence(self, x, x2):
            z, _ = self.batch_topk(self.pre_acts(x))
            z2, _ = self.batch_topk(self.pre_acts(x2))
            return (z - z2).abs().sum(1).mean()

        def forward_train26(self, x, x2=None, w=None):
            pre = self.pre_acts(x)
            z, kept = self.batch_topk(pre)
            losses, x_hat = self.mrecon(z, self.W_dec, self.b_dec, x, w if self.mode == 'rw' else None)
            loss = torch.stack(losses).sum()
            rec = float(loss)
            extra = torch.zeros((), device=x.device)
            if self.mode == 'pred':
                l1, _ = self.mrecon(z, self.W_dec1, self.b_dec1, x2)
                extra = torch.stack(l1).sum()
                loss = loss + self.lam * extra
            elif self.mode == 'smooth':
                z2, _ = self.batch_topk(self.pre_acts(x2))
                extra = (z - z2).abs().sum(1).mean()
                loss = loss + self.mu * extra
            with torch.no_grad():
                fired = (z > 0).any(0)
                self.tokens_since_fired += x.shape[0]
                self.tokens_since_fired[fired] = 0
                pos = kept[kept > 0]
                if pos.numel():
                    mn = pos.min()
                    if self.threshold < 0:
                        self.threshold.copy_(mn)
                    else:
                        self.threshold.mul_(1 - self.threshold_lr).add_(self.threshold_lr * mn)
            dead = self.tokens_since_fired >= self.dead_tokens
            n_dead = int(dead.sum())
            aux = torch.zeros((), device=x.device)
            if n_dead > 0:
                k_aux = min(self.k_aux, n_dead)
                resid = (x - x_hat).detach()
                pre_dead = torch.relu(pre[:, dead])
                v, i = pre_dead.topk(k_aux, dim=1)
                z_aux = torch.zeros_like(pre_dead).scatter_(1, i, v)
                aux = ((z_aux @ self.W_dec[dead]) - resid).pow(2).sum(1).mean()
                if torch.isfinite(aux):
                    loss = loss + self.aux_coef * aux
            var = (x - x.mean(0)).pow(2).sum(1).mean()
            e_last = (x_hat - x).pow(2).sum(1).mean()
            logs = {'loss': float(loss), 'recon': rec, 'extra': float(extra), 'aux': float(aux), 'n_dead': n_dead,
                    'fve': float(1 - e_last.detach() / var), 'l0': float((z > 0).sum(1).float().mean()), 'threshold': float(self.threshold)}
            return loss, logs

        @torch.no_grad()
        def post_step(self):
            self.normalize_decoder()
            if self.mode == 'pred':
                self.W_dec1.data /= self.W_dec1.data.norm(dim=1, keepdim=True).clamp_min(1e-8)

        @torch.no_grad()
        def remove_parallel_grad(self):
            super().remove_parallel_grad()
            if self.mode == 'pred' and self.W_dec1.grad is not None:
                W = self.W_dec1.data
                self.W_dec1.grad -= (self.W_dec1.grad * W).sum(1, keepdim=True) * W

    return SAE26


def train_step(sae, opt, x, x2, w, step, total, lr, warmup):
    import torch
    from src.eci.sae import _lr_at
    for pg in opt.param_groups:
        pg['lr'] = _lr_at(step, total, lr, warmup)
    loss, logs = sae.forward_train26(x, x2, w)
    if not math.isfinite(logs['loss']):
        raise RuntimeError(f'non-finite loss at step {step}: {logs}')
    opt.zero_grad(set_to_none=True)
    loss.backward()
    sae.remove_parallel_grad()
    torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
    opt.step()
    sae.post_step()
    return logs


# ---------------------------------------------------------------------------------------------- loader (x_t, x_{t+2})
def xp_np(tok, I):
    return tok[I['c']], tok[I[L]]


def xp_torch(tok, I, dev):
    import torch
    t = torch.from_numpy(tok).to(dev)
    return t[torch.from_numpy(I['c']).to(dev)], t[torch.from_numpy(I[L]).to(dev)]


class Store(TR.Store):
    def load_xp(self, w, dev=None, keep=False):
        ser, obs, i0, i1, nf = w
        W = self.video(ser, obs).window(i0, i1)
        I = TR.centre_index(W, nf)
        x, xp = xp_np(W['tok'], I) if dev is None else xp_torch(W['tok'], I, dev)
        return x, xp, int(W['tok'].nbytes), ((W, I) if keep else None)


def sample_x(st, ws, n, seed):
    xs, tot = [], 0
    for i in np.random.default_rng(seed).permutation(len(ws)):
        x, _, _, _ = st.load_xp(ws[i])
        xs.append(x)
        tot += len(x)
        if tot >= n:
            break
    return np.concatenate(xs)[:n]


# ---------------------------------------------------------------------------------------------- kmeans (RW)
def cmd_kmeans(a):
    import torch
    dom = a.dom
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 16)))
    ws = TR.windows(dom)
    st = Store(dom)
    X = torch.from_numpy(sample_x(st, ws, 2_000_000, 6).astype(np.float32))  # = the v2.5 alpha sample windows (rng 6)
    g = torch.Generator().manual_seed(26)
    fit = X[torch.randperm(len(X), generator=g)[:500_000]]
    C = fit[torch.randperm(len(fit), generator=g)[:K_RW]].clone()
    for it in range(25):
        lab = assign(fit, C)
        cnt = torch.bincount(lab, minlength=K_RW).float()
        S = torch.zeros_like(C).index_add_(0, lab, fit)
        empty = cnt == 0
        C = torch.where(empty[:, None], C, S / cnt.clamp_min(1)[:, None])
    lab = assign(X, C)
    f = torch.bincount(lab, minlength=K_RW).double() / len(X)
    w = torch.where(f > 0, f.clamp_min(1e-12) ** -0.5, torch.zeros_like(f))
    w = w / (f * w).sum()
    wc = w.clamp(max=CLIP_RW)
    tok_w = wc[lab]
    res = {'k': K_RW, 'sample_tokens': len(X), 'fit_tokens': len(fit), 'iterations': 25, 'clip': CLIP_RW,
           'freq_min': float(f[f > 0].min()), 'freq_max': float(f.max()), 'empty_clusters': int((f == 0).sum()),
           'weight_min': float(wc[f > 0].min()), 'weight_max': float(wc.max()), 'clusters_clipped': int((w > CLIP_RW).sum()),
           'token_frac_clipped': float(f[w > CLIP_RW].sum()), 'token_mean_weight_after_clip': float(tok_w.mean()),
           'effective_sample_size_frac': float(tok_w.sum() ** 2 / (tok_w ** 2).sum() / len(tok_w))}
    OUT26.mkdir(parents=True, exist_ok=True)
    np.savez(OUT26 / f'{dom}_rw.npz', centroids=C.numpy(), weight=wc.float().numpy(), freq=f.numpy())
    (OUT26 / f'{dom}_rw.json').write_text(json.dumps(res, indent=1))
    log(f'kmeans {dom}: {res}')


def assign(X, C):
    import torch
    cn = (C ** 2).sum(1)
    out = torch.empty(len(X), dtype=torch.long)
    for b in range(0, len(X), 100_000):
        out[b:b + 100_000] = (cn[None] - 2 * X[b:b + 100_000] @ C.T).argmin(1)
    return out


# ---------------------------------------------------------------------------------------------- featcheck
def cmd_featcheck(a):
    dom = a.dom
    vids = TR.videos(dom)
    rng = np.random.default_rng(5)
    win = [(int(vi), int(i0)) for vi in rng.choice(len(vids), 40) for i0 in [rng.integers(L, vids[vi][2] - L - 3)]]
    win += [(int(vi), L if j % 2 == 0 else vids[vi][2] - L - 3) for j, vi in enumerate(rng.choice(len(vids), 10, replace=False))]
    st = Store(dom)
    res = {'windows': 0, 'tokens': 0, 'exact': True}
    for vi, i0 in win:
        ser, obs, nf = vids[vi]
        x, xp, _, _ = st.load_xp((ser, obs, i0, i0 + 3, nf))
        W0 = TR.T.Video(dom, ser, obs).window(i0, i0 + 3)
        rows = {(int(f), int(p)): r for r, (f, p) in enumerate(zip(W0['fi'], W0['pos']))}
        rx, rp = [], []
        for r in range(len(W0['fi'])):
            f, p = int(W0['fi'][r]), int(W0['pos'][r])
            if (W0['flag'][r] & 1) and i0 <= f < i0 + 3:
                rx.append(W0['tok'][r])
                rp.append(W0['tok'][rows[(f + L, p)]])
        res['exact'] &= bool(np.array_equal(np.stack(rx).view(np.uint16), x.view(np.uint16)) and np.array_equal(np.stack(rp).view(np.uint16), xp.view(np.uint16)))
        res['windows'] += 1
        res['tokens'] += len(rx)
    OUT26.mkdir(parents=True, exist_ok=True)
    (OUT26 / f'{dom}_featcheck.json').write_text(json.dumps(res, indent=1))
    log(f'featcheck {dom}: {res}')
    if not res['exact']:
        raise SystemExit('FEATURE IDENTITY FAILED')


def cmd_rows(a):
    od = BEE26 / a.dom / 'codes'
    od.mkdir(parents=True, exist_ok=True)
    for f in ('rows.parquet', 'rows_all.parquet'):
        shutil.copy(TR.BEE25 / a.dom / 'codes' / f, od / f)
    OUT26.mkdir(parents=True, exist_ok=True)
    shutil.copy(TR.OUT25 / f'{a.dom}_count.json', OUT26 / f'{a.dom}_count.json')
    log(f'rows {a.dom} copied from v2.5')


# ---------------------------------------------------------------------------------------------- train
def cmd_train(a):
    import torch
    from concurrent.futures import ThreadPoolExecutor
    from src.eci.sae import TokenNorm, geometric_median
    from src.eci.sae_calibrate import calibrate_threshold, set_threshold, sparsity_stats
    import v20 as V
    SAE26 = make_sae_class()
    dom = a.dom
    dev = torch.device('cuda')
    seeds = [int(s) for s in a.seeds.split(',')]
    arms = a.arms.split(',')
    steps = int(a.max_steps) if a.max_steps else STEPS[dom]
    total = STEPS[dom]
    ws = TR.windows(dom)
    o = np.random.default_rng(TR.WIN_RNG).permutation(len(ws))
    chunks = [[ws[i] for i in o[c0:c0 + TR.CW]] for c0 in range(0, len(o), TR.CW)]  # pass 0 of the v2.5 stream
    if dom == 'ants':
        assert total == json.loads((TR.OUT25 / 'ants_count.json').read_text())['M_used'] // BS  # 1 pass
    spec = V.R.spec()['sae']
    lr, warmup = spec['lr'], 500
    nthr = int(os.environ.get('SLURM_CPUS_PER_TASK', 16))
    st = Store(dom)
    ex = ThreadPoolExecutor(max(4, nthr - 2))
    rw = np.load(OUT26 / f'{dom}_rw.npz')
    Cg = torch.from_numpy(rw['centroids']).to(dev)
    Wg = torch.from_numpy(rw['weight']).to(dev)
    cn = (Cg ** 2).sum(1)
    log(f'{dom}: steps {steps:,} (schedule {total:,}), {len(chunks)} chunks/pass, arms {arms}, seeds {seeds}')
    fcheck = {'windows': 0, 'tokens': 0}

    def gather(fs):
        xs, ps, nb, wt = [], [], 0, 0.0
        for f in fs:
            t0_ = time.time()
            x, xp, nb_, keep = f.result()
            wt += time.time() - t0_
            nb += nb_
            if len(x):
                xs.append(x)
                ps.append(xp)
            if keep is not None and len(x):
                rx, rp = xp_np(keep[0]['tok'], keep[1])
                if not (np.array_equal(x.cpu().numpy().view(np.uint16), rx.view(np.uint16)) and np.array_equal(xp.cpu().numpy().view(np.uint16), rp.view(np.uint16))):
                    raise SystemExit('GPU FEATURE IDENTITY FAILED')
                fcheck['windows'] += 1
                fcheck['tokens'] += len(rx)
        return torch.cat(xs), torch.cat(ps), nb, wt
    pending = [ex.submit(st.load_xp, w, dev, i < 16) for i, w in enumerate(chunks[0])]
    Xn, Pn, nb, wait = gather(pending)
    log(f'GPU == numpy (x_t, x_t+2), bit-exact: {fcheck}')
    pending = [ex.submit(st.load_xp, w, dev) for w in chunks[1]]
    torch.backends.cuda.matmul.allow_tf32 = True

    def new_sae(arm, s, mu=0.0):
        torch.manual_seed(s)
        g = np.random.default_rng(s)
        ii = torch.from_numpy(np.sort(g.choice(len(Xn), min(250_000, len(Xn)), replace=False))).to(dev)
        init = Xn[ii].float()
        norm = TokenNorm(1024).to(dev).fit_blocks(init, [1024], seed=s)
        nl, pf = shape_of(arm)
        sae = SAE26(mode=mode_of(arm), lam=LAM, mu=mu, d_in=1024, n_latents=nl, prefixes=pf, k=spec['k'], seed=s).to(dev)
        with torch.no_grad():
            sae.b_dec.copy_(geometric_median(norm(init[:100_000])))
            if sae.mode == 'pred':
                sae.b_dec1.copy_(sae.b_dec)
        return sae, norm
    # ---- MU (label-free, before the SMOOTH arms exist): warm-up a throwaway plain 4096 SAE (seed 2600) for 2,000 steps on
    # the first chunk, measure recon (Matryoshka sum) and mean ||z_t - z_t+2||_1 over steps 1,500-1,999, MU = 0.1 R / C
    mu, mu_rec = 0.0, {}
    if 'SMOOTH' in arms:
        sae0, n0 = new_sae('N0wc', 2600)
        opt0 = torch.optim.Adam(sae0.parameters(), lr=lr, betas=(0.9, 0.999))
        gen0 = torch.Generator(device=dev).manual_seed(2600)
        Rs, Cs, sidx = [], [], 0
        perm = torch.randperm(len(Xn), generator=gen0, device=dev)
        for k in range(2000):
            if (sidx + 1) * BS > len(Xn):
                perm, sidx = torch.randperm(len(Xn), generator=gen0, device=dev), 0
            ib = perm[sidx * BS:(sidx + 1) * BS]
            sidx += 1
            xb, pb = n0(Xn[ib]), n0(Pn[ib])
            lg = train_step(sae0, opt0, xb, pb, None, k, 2000, lr, warmup)
            if k >= 1500:
                with torch.no_grad():
                    Cs.append(float(sae0.coherence(xb, pb)))
                Rs.append(lg['recon'])
        mu = SMOOTH_FRAC * float(np.mean(Rs)) / float(np.mean(Cs))
        mu_rec = {'R_mean': float(np.mean(Rs)), 'C_mean': float(np.mean(Cs)), 'mu': mu, 'warmup_steps': 2000, 'measured_steps': '1500-1999',
                  'seed': 2600, 'frac': SMOOTH_FRAC}
        log(f'MU (label-free warm-up): {mu_rec}')
        (OUT26 / 'train').mkdir(parents=True, exist_ok=True)
        (OUT26 / 'train' / f'{dom}_mu.json').write_text(json.dumps(mu_rec, indent=1))
        del sae0, opt0
    M = {}
    for arm in arms:
        for s in seeds:
            sae, norm = new_sae(arm, s, mu if arm == 'SMOOTH' else 0.0)
            M[(arm, s)] = {'sae': sae, 'norm': norm, 'opt': torch.optim.Adam(sae.parameters(), lr=lr, betas=(0.9, 0.999)), 'hist': []}
    gens = {s: torch.Generator(device=dev).manual_seed(s) for s in seeds}
    step, ci, cx, cp, used = 0, 0, None, None, 0
    t0 = time.time()
    tlog, slog, util, wsum = time.time(), 0, [], []
    while step < steps:
        Xc = Xn if cx is None else torch.cat([cx, Xn])
        Pc = Pn if cp is None else torch.cat([cp, Pn])
        del Xn, Pn
        n = (len(Xc) // BS) * BS
        last = ci == len(chunks) - 1
        cx, cp = (None, None) if last else (Xc[n:].clone(), Pc[n:].clone())
        Xg, Pg = Xc[:n], Pc[:n]
        del Xc, Pc
        perms = {s: torch.randperm(n, generator=gens[s], device=dev) for s in seeds}
        for b in range(n // BS):
            if step >= steps:
                break
            for s in seeds:
                ib = perms[s][b * BS:(b + 1) * BS]
                xb, pb = Xg[ib], Pg[ib]
                wb = None
                if 'RW' in arms:
                    xf = xb.float()
                    wb = Wg[(cn[None] - 2 * xf @ Cg.T).argmin(1)]
                    if step % 2000 == 0:
                        wsum.append(float(wb.mean()))
                for arm in arms:
                    m = M[(arm, s)]
                    nm = m['norm']
                    lg = train_step(m['sae'], m['opt'], nm(xb), nm(pb) if arm in ('PRED', 'SMOOTH') else None,
                                    wb if arm == 'RW' else None, step, total, lr, warmup)
                    if step % 5000 == 0 or step == steps - 1:
                        lg.update(step=step, elapsed_s=round(time.time() - t0, 1))
                        m['hist'].append(lg)
            used += BS
            step += 1
            if step % 2000 == 0:
                u, mem = TR.V23.gpu_util()
                util.append(u)
                el = time.time() - tlog
                k0 = next(iter(M))
                log(f'step {step:,}/{steps:,} chunk {ci + 1}: {(step - slog) / el:.2f} steps/s ({len(M)} SAEs), GPU util {u}% mem {mem} MiB, '
                    f'fve {k0} {M[k0]["hist"][-1]["fve"] if M[k0]["hist"] else float("nan"):.4f}, ETA {(steps - step) / ((step - slog) / el) / 60:.0f} min')
                tlog, slog = time.time(), step
        del Xg, Pg, perms
        if step >= steps or last:
            break
        ci += 1
        Xn, Pn, nb, wait = gather(pending)
        pending = [ex.submit(st.load_xp, w, dev) for w in chunks[ci + 1]] if ci + 1 < len(chunks) else []
    tt = time.time() - t0
    for f in pending:
        f.cancel()
    ex.shutdown(wait=False, cancel_futures=True)
    rec0 = {'node': socket.gethostname(), 'steps_run': step, 'steps_schedule': total, 'tokens_used_per_sae': used,
            'passes': used / json.loads((TR.OUT25 / f'{dom}_count.json').read_text())['M_used'], 'train_time_s': tt,
            'steps_per_s': step / tt, 'n_saes_in_job': len(M), 'gpu_util_samples': util, 'gpu_util_median': float(np.median(util)) if util else None,
            'gpu_identity': fcheck, 'mu': mu_rec, 'lam': LAM, 'rw_batch_mean_weight_samples': wsum}
    log(f'train loop done: {json.dumps(rec0)}')
    if a.max_steps:
        return
    xc = torch.from_numpy(sample_x(st, ws, TR.NCAL, 7))
    xe = torch.from_numpy(sample_x(st, ws, TR.NCAL, 8))
    from src.eci.sae import MatryoshkaBatchTopKSAE
    for (arm, s), m in M.items():
        sae, norm = m['sae'], m['norm']
        sae.eval()
        torch.backends.cuda.matmul.allow_tf32 = False
        th, _ = calibrate_threshold(sae, xc, norm=norm)
        set_threshold(sae, th)
        sst = sparsity_stats(sae, xe, th, norm=norm)
        fired = torch.zeros(sae.n_latents, dtype=torch.long, device=dev)
        with torch.no_grad():
            for b0 in range(0, len(xe), 65536):
                fired += (sae.encode(norm(xe[b0:b0 + 65536].to(dev).float()), mode='threshold') > 0).sum(0)
        key = f'{arm}_s{s}'
        nl, pf = shape_of(arm)
        plain = MatryoshkaBatchTopKSAE(d_in=1024, n_latents=nl, prefixes=pf, k=spec['k'], seed=s)
        sd = {k: v for k, v in sae.state_dict().items() if k not in ('W_dec1', 'b_dec1')}
        plain.load_state_dict(sd)
        cfg = {**spec, 'W': nl, 'prefixes': list(pf), 'steps': total, 'arm': arm, 'mode': mode_of(arm), 'lam': LAM if arm == 'PRED' else None,
               'mu': mu if arm == 'SMOOTH' else None, 'input': 'x_t', 'lag_frames': L, 'edge_rule': f'{L} <= t < T - {L}'}
        rec = {'cfg': cfg, 'seed': s, 'key': key, **rec0, 'theta_star': float(th), 'eval_fve': float(sst['fve']), 'eval_l0': float(sst['l0']),
               'dead_at_end_train': int((sae.tokens_since_fired >= sae.dead_tokens).sum()),
               'dead_frac_at_end_train': float((sae.tokens_since_fired >= sae.dead_tokens).float().mean()),
               'never_fire_on_eval_frac': float((fired == 0).float().mean()), 'history': m['hist']}
        out = BEE26 / dom / 'sae' / key
        out.mkdir(parents=True, exist_ok=True)
        from src.eci.sae import save_checkpoint
        save_checkpoint(out / 'sae_theta.pt', plain, norm, extra={'config': 'eci_v2_6', 'cfg': cfg, 'seed': s, 'theta_star': float(th)})
        if arm == 'PRED':
            torch.save({'W_dec1': sae.W_dec1.detach().cpu(), 'b_dec1': sae.b_dec1.detach().cpu()}, out / 'dec1.pt')
        (out / 'train.json').write_text(json.dumps(rec, indent=1, default=float))
        (OUT26 / 'train').mkdir(parents=True, exist_ok=True)
        (OUT26 / 'train' / f'{dom}_{key}.json').write_text(json.dumps(rec, indent=1, default=float))
        log(f'{dom} {key}: theta* {th:.4f} eval FVE {sst["fve"]:.5f} L0 {sst["l0"]:.2f} dead(train) {rec["dead_at_end_train"]} '
            f'never-fire(eval) {rec["never_fire_on_eval_frac"]:.4f}')
    log('ALL CHECKPOINTS SAVED')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=('kmeans', 'featcheck', 'train', 'rows'))
    p.add_argument('--dom', required=True, choices=('mice', 'ants'))
    p.add_argument('--arms', default=','.join(ARMS))
    p.add_argument('--seeds', default=','.join(map(str, SEEDS)))
    p.add_argument('--max_steps', type=int, default=0)
    a = p.parse_args()
    {'kmeans': cmd_kmeans, 'featcheck': cmd_featcheck, 'train': cmd_train, 'rows': cmd_rows}[a.step](a)
