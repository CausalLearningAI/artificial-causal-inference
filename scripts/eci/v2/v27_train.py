"""
ECI v2.7: confirm PRED on fresh seeds + three label-free ways to make behaviour neurons emerge earlier in the Matryoshka
hierarchy (pre-registration scripts/eci/v2/v27_prereg.md). Data / edge rule / stream / pairing = v2.6 (v26_train.Store:
x_t and x_{t+2}, frames 2 <= t < T - 2, windows of 100, chunks of 256, rng 20261105 pass 0, per-seed permutation; every
arm of a seed sees the same batches). Input x_t (1024); 4096 latents, prefixes 128/256/512/2048/4096, k 16; seeds 2701-2703.
Arms (all = PRED: encoder on x_t, D0 reconstructs x_t, D1 on the same code reconstructs a target; loss = L0 + LAM L1, both
Matryoshka prefix sums; per-token weights w multiply BOTH terms; AuxK unweighted):
  PREDc  target x_{t+2}, LAM 1, w = 1                                  (= v2.6 PRED)
  PREDd  target x_{t+2} - x_t (normalised space: norm(x_{t+2}) - norm(x_t)), LAM_D label-free (step train: warm-up), b_dec1 = 0
  PREDk  = PREDc, w = WC on contact-proxy tokens, 1 elsewhere (WC: proxy tokens carry 20% of the weight; cap 30)
  PREDh  = PREDc, w = min(10, e(x) / mean e), e = squared residual of x_t under a fixed top-64 PCA (label-free sample)
Contact proxy (label-free, from the stored mask v5 of frame t on the native G x G grid, 8-connectivity):
  (A) neck: mask token removed by a 3x3 erosion whose 5x5 neighbourhood contains >= 2 distinct components of the eroded mask
  (B) near-other: mask token whose 5x5 neighbourhood contains a mask token of a different (un-eroded) mask component
  proxy = A or B.
Steps: prep --dom D (CPU: proxy fraction, WC, PCA, weights, proxy identity check) | proxydiag (CPU, mice: proxy enrichment on
labelled frames, diagnostic only) | train --dom D (GPU)
"""
import argparse
import json
import math
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
for p in ('', 'scripts/eci/v2'):
    sys.path.insert(0, str(REPO / p))
import v25_train as TR  # noqa: E402
import v26_train as T6  # noqa: E402

BEE27 = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_7')
OUT27 = REPO / 'results/vision/eci_v2_7'
L, BS = TR.L, TR.BS
SEEDS = (2701, 2702, 2703)
ARMS = ('PREDc', 'PREDd', 'PREDk', 'PREDh')
STEPS = {'mice': 120_000, 'ants': 25_561}  # mice ~0.66 pass (budget), ants 1 pass
PREFIXES = (128, 256, 512, 2048, 4096)
K_PCA, CLIP_H, SHARE_K, CAP_K = 64, 10.0, 0.20, 30.0
log = TR.log
_G = {}


def grid_of(dom, ser, obs):
    if (ser, obs) not in _G:
        _G[(ser, obs)] = int(np.load(TR.STORE / dom / TR.VER / 'videos' / ser / obs / 'index.npz')['G'])
    return _G[(ser, obs)]


def proxy_grid(pos, G):
    """pos = mask positions of one frame -> bool proxy per position (same order)."""
    from scipy import ndimage as nd
    m = np.zeros(G * G, bool)
    m[pos] = True
    m = m.reshape(G, G)
    s8 = np.ones((3, 3), bool)
    big = 1 << 30
    lab, _ = nd.label(m, s8)
    mx = nd.maximum_filter(lab, 5)
    mn = nd.minimum_filter(np.where(lab > 0, lab, big), 5)
    B = m & (mx != mn)
    er = nd.binary_erosion(m, s8)
    le, _ = nd.label(er, s8)
    mxe = nd.maximum_filter(le, 5)
    mne = nd.minimum_filter(np.where(le > 0, le, big), 5)
    A = m & ~er & (mxe > 0) & (mne < big) & (mxe != mne)
    return (A | B).ravel()[pos]


def proxy_of_window(W, I, G):
    """contact proxy of every centre token (I['c'] rows) -> bool array."""
    c = I['c']
    fi, pos = W['fi'][c], W['pos'][c]
    out = np.zeros(len(c), bool)
    for f in np.unique(fi):
        q = np.flatnonzero(fi == f)
        out[q] = proxy_grid(pos[q].astype(np.int64), G)
    return out


class Store(T6.Store):
    def load_xpc(self, w, dev=None, keep=False):
        ser, obs, i0, i1, nf = w
        W = self.video(ser, obs).window(i0, i1)
        I = TR.centre_index(W, nf)
        x, xp = T6.xp_np(W['tok'], I) if dev is None else T6.xp_torch(W['tok'], I, dev)
        cp = proxy_of_window(W, I, grid_of(self.dom, ser, obs))
        if dev is not None:
            import torch
            cp = torch.from_numpy(cp).to(dev)
        return x, xp, cp, int(W['tok'].nbytes), ((W, I) if keep else None)


# ---------------------------------------------------------------------------------------------- prep
def cmd_prep(a):
    import torch
    dom = a.dom
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 16)))
    ws = TR.windows(dom)
    st = Store(dom)
    xs, cs, n = [], [], 0
    for i in np.random.default_rng(6).permutation(len(ws)):  # = the v2.5 alpha / v2.6 k-means sample windows
        x, _, c, _, _ = st.load_xpc(ws[i])
        xs.append(x)
        cs.append(c)
        n += len(x)
        if n >= 2_000_000:
            break
    X = torch.from_numpy(np.concatenate(xs).astype(np.float32))
    Cp = np.concatenate(cs)
    p = float(Cp.mean())
    wc_raw = SHARE_K * (1 - p) / ((1 - SHARE_K) * p)
    wc = min(CAP_K, wc_raw)
    share = wc * p / (wc * p + (1 - p))
    g = torch.Generator().manual_seed(27)
    fit = X[torch.randperm(len(X), generator=g)[:500_000]]
    mu = fit.mean(0)
    cov = ((fit - mu).T @ (fit - mu)) / len(fit)
    ev, V = torch.linalg.eigh(cov.double())
    V = V[:, -K_PCA:].float()
    err = residual(X, mu, V)
    me = float(err.mean())
    wh = (err / me).clamp(max=CLIP_H)
    q = np.percentile(wh.numpy(), [1, 10, 50, 90, 99])
    res = {'sample_tokens': len(X), 'contact': {'proxy_token_frac': p, 'w_raw': wc_raw, 'w': wc, 'capped': wc < wc_raw,
                                                'proxy_weight_share': share},
           'hard': {'k_pca': K_PCA, 'pca_fit_tokens': len(fit), 'pca_var_explained': float(ev[-K_PCA:].sum() / ev.sum()), 'mean_err': me,
                    'clip': CLIP_H, 'frac_clipped': float((err / me > CLIP_H).float().mean()), 'mean_after_clip': float(wh.mean()),
                    'quantiles_1_10_50_90_99': q.tolist(), 'effective_sample_size_frac': float(wh.sum() ** 2 / (wh ** 2).sum() / len(wh))}}
    # proxy identity: loader (vectorised filters) == an independent per-token loop on 20 windows (incl. 4 at the edge boundary)
    rng = np.random.default_rng(5)
    vids = TR.videos(dom)
    bad, tot = 0, 0
    for j, vi in enumerate(rng.choice(len(vids), 20, replace=False)):
        ser, obs, nf = vids[vi]
        i0 = L if j < 2 else (nf - L - 3 if j < 4 else int(rng.integers(L, nf - L - 3)))
        W = st.video(ser, obs).window(i0, i0 + 3)
        I = TR.centre_index(W, nf)
        got = proxy_of_window(W, I, grid_of(dom, ser, obs))
        ref = proxy_slow(W, I, grid_of(dom, ser, obs))
        bad += int((got != ref).sum())
        tot += len(got)
    res['proxy_identity'] = {'windows': 20, 'tokens': tot, 'mismatches': bad}
    OUT27.mkdir(parents=True, exist_ok=True)
    np.savez(OUT27 / f'{dom}_prep.npz', pca_mean=mu.numpy(), pca_V=V.numpy(), mean_err=np.float64(me))
    (OUT27 / f'{dom}_prep.json').write_text(json.dumps(res, indent=1))
    log(f'prep {dom}: {json.dumps(res)}')
    if bad:
        raise SystemExit('PROXY IDENTITY FAILED')


def residual(X, mu, V):
    xc = X - mu
    return (xc ** 2).sum(1) - ((xc @ V) ** 2).sum(1)


def proxy_slow(W, I, G):
    """independent reference: python loops over grid cells (no scipy filters except the component labelling)."""
    from scipy import ndimage as nd
    c = I['c']
    fi, pos = W['fi'][c], W['pos'][c]
    out = np.zeros(len(c), bool)
    s8 = np.ones((3, 3), bool)
    for f in np.unique(fi):
        q = np.flatnonzero(fi == f)
        m = np.zeros((G, G), bool)
        for p_ in pos[q]:
            m[int(p_) // G, int(p_) % G] = True
        lab, _ = nd.label(m, s8)
        le, _ = nd.label(nd.binary_erosion(m, s8), s8)
        for k, p_ in zip(q, pos[q]):
            r, cc = int(p_) // G, int(p_) % G
            win = lab[max(0, r - 2):r + 3, max(0, cc - 2):cc + 3]
            B = len(set(win[win > 0].tolist())) >= 2
            wine = le[max(0, r - 2):r + 3, max(0, cc - 2):cc + 3]
            A = le[r, cc] == 0 and len(set(wine[wine > 0].tolist())) >= 2
            out[k] = A or B
    return out


def cmd_proxydiag(a):
    """DIAGNOSTIC ONLY (never used for tuning): per mice ext frame (edge-excluded), the number of contact-proxy tokens; rate of
    frames with >= 1 proxy token and mean proxy tokens on frames labelled nose-nose / nose-tail vs other frames."""
    import pandas as pd
    from concurrent.futures import ThreadPoolExecutor
    R = pd.read_parquet(TR.BEE25 / 'mice' / 'codes' / 'rows.parquet')
    Fx = pd.read_parquet('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_3/ext/frames.parquet', columns=['Y_nn', 'Y_np', 'Y_nt'])
    Y = Fx.iloc[R.xrow.values].reset_index(drop=True)
    nn = ((Y.Y_nn.fillna(0) > 0) | (Y.Y_np.fillna(0) > 0)).values
    nt = (Y.Y_nt.fillna(0) > 0).values

    def one(g):
        (ser, obs), d = g
        z = np.load(TR.STORE / 'mice' / TR.VER / 'videos' / ser / obs / 'index.npz')
        ptr, idx, fl, G = z['ptr'], z['idx'], z['flag'], int(z['G'])
        out = np.zeros(len(d), np.int32)
        for i, f in enumerate(d.fi.values):
            r0, r1 = int(ptr[f]), int(ptr[f + 1])
            pos = idx[r0:r1][(fl[r0:r1] & 1) > 0].astype(np.int64)
            out[i] = int(proxy_grid(pos, G).sum()) if len(pos) else 0
        return d.index.values, out
    cnt = np.zeros(len(R), np.int32)
    with ThreadPoolExecutor(int(os.environ.get('SLURM_CPUS_PER_TASK', 16))) as ex:
        for ix, o in ex.map(one, R.groupby(['series', 'obs'], sort=False)):
            cnt[ix] = o
    res = {'frames': len(R)}
    for nm, y in (('nose-nose', nn), ('nose-tail', nt)):
        res[nm] = {'labelled_frames': int(y.sum()), 'P(proxy>0 | labelled)': float((cnt[y] > 0).mean()), 'P(proxy>0 | not labelled)': float((cnt[~y] > 0).mean()),
                   'mean proxy tokens | labelled': float(cnt[y].mean()), 'mean proxy tokens | not labelled': float(cnt[~y].mean())}
    res['note'] = 'diagnostic only; computed after the weights were fixed; never used for tuning'
    (OUT27 / 'mice_proxydiag.json').write_text(json.dumps(res, indent=1))
    log(f'proxydiag: {res}')


# ---------------------------------------------------------------------------------------------- SAE
def make_sae_class():
    import torch
    import torch.nn as nn
    base = T6.make_sae_class()

    class SAE27(base):
        def __init__(self, target='next', lam=1.0, **kw):
            super().__init__(mode='pred', lam=lam, **kw)
            self.target = target
            if target == 'delta':
                with torch.no_grad():
                    self.b_dec1.zero_()

        def forward_train27(self, x, x2, w=None):
            pre = self.pre_acts(x)
            z, kept = self.batch_topk(pre)
            l0, x_hat = self.mrecon(z, self.W_dec, self.b_dec, x, w)
            tgt = x2 - x if self.target == 'delta' else x2
            l1, _ = self.mrecon(z, self.W_dec1, self.b_dec1, tgt, w)
            L0, L1 = torch.stack(l0).sum(), torch.stack(l1).sum()
            loss = L0 + self.lam * L1
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
            return loss, {'loss': float(loss), 'L0': float(L0), 'L1': float(L1), 'aux': float(aux), 'n_dead': n_dead,
                          'fve': float(1 - e_last.detach() / var), 'l0': float((z > 0).sum(1).float().mean())}
    return SAE27


def train_step(sae, opt, x, x2, w, step, total, lr, warmup):
    import torch
    from src.eci.sae import _lr_at
    for pg in opt.param_groups:
        pg['lr'] = _lr_at(step, total, lr, warmup)
    loss, logs = sae.forward_train27(x, x2, w)
    if not math.isfinite(logs['loss']):
        raise RuntimeError(f'non-finite loss at step {step}: {logs}')
    opt.zero_grad(set_to_none=True)
    loss.backward()
    sae.remove_parallel_grad()
    torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
    opt.step()
    sae.post_step()
    return logs


def cmd_train(a):
    import torch
    from concurrent.futures import ThreadPoolExecutor
    from src.eci.sae import TokenNorm, geometric_median, save_checkpoint, MatryoshkaBatchTopKSAE
    from src.eci.sae_calibrate import calibrate_threshold, set_threshold, sparsity_stats
    import v20 as V
    SAE27 = make_sae_class()
    dom = a.dom
    dev = torch.device('cuda')
    seeds = [int(s) for s in a.seeds.split(',')]
    arms = a.arms.split(',')
    steps = int(a.max_steps) if a.max_steps else STEPS[dom]
    total = STEPS[dom]
    ws = TR.windows(dom)
    o = np.random.default_rng(TR.WIN_RNG).permutation(len(ws))
    chunks = [[ws[i] for i in o[c0:c0 + TR.CW]] for c0 in range(0, len(o), TR.CW)]
    spec = V.R.spec()['sae']
    lr, warmup = spec['lr'], 500
    nthr = int(os.environ.get('SLURM_CPUS_PER_TASK', 16))
    st = Store(dom)
    ex = ThreadPoolExecutor(max(4, nthr - 2))
    pj = json.loads((OUT27 / f'{dom}_prep.json').read_text())
    pz = np.load(OUT27 / f'{dom}_prep.npz')
    WC = pj['contact']['w']
    pmu, pV, pme = (torch.from_numpy(pz['pca_mean']).to(dev), torch.from_numpy(pz['pca_V']).to(dev), float(pz['mean_err']))
    log(f'{dom}: steps {steps:,} (schedule {total:,}), {len(chunks)} chunks/pass, arms {arms}, seeds {seeds}, WC {WC:.3f}')
    fcheck = {'windows': 0, 'tokens': 0}

    def gather(fs):
        xs, ps, cs = [], [], []
        for f in fs:
            x, xp, cp, _, keep = f.result()
            if len(x):
                xs.append(x)
                ps.append(xp)
                cs.append(cp)
            if keep is not None and len(x):
                rx, rp = T6.xp_np(keep[0]['tok'], keep[1])
                if not (np.array_equal(x.cpu().numpy().view(np.uint16), rx.view(np.uint16)) and np.array_equal(xp.cpu().numpy().view(np.uint16), rp.view(np.uint16))):
                    raise SystemExit('GPU FEATURE IDENTITY FAILED')
                fcheck['windows'] += 1
                fcheck['tokens'] += len(rx)
        return torch.cat(xs), torch.cat(ps), torch.cat(cs)
    pending = [ex.submit(st.load_xpc, w, dev, i < 16) for i, w in enumerate(chunks[0])]
    Xn, Pn, Cn = gather(pending)
    log(f'GPU == numpy (x_t, x_t+2), bit-exact: {fcheck}; contact-proxy fraction in chunk 0: {float(Cn.float().mean()):.4f}')
    pending = [ex.submit(st.load_xpc, w, dev) for w in chunks[1]]
    torch.backends.cuda.matmul.allow_tf32 = True

    def new_sae(arm, s, lam):
        torch.manual_seed(s)
        g = np.random.default_rng(s)
        ii = torch.from_numpy(np.sort(g.choice(len(Xn), min(250_000, len(Xn)), replace=False))).to(dev)
        init = Xn[ii].float()
        norm = TokenNorm(1024).to(dev).fit_blocks(init, [1024], seed=s)
        sae = SAE27(target='delta' if arm == 'PREDd' else 'next', lam=lam, d_in=1024, n_latents=4096, prefixes=PREFIXES, k=spec['k'], seed=s).to(dev)
        with torch.no_grad():
            sae.b_dec.copy_(geometric_median(norm(init[:100_000])))
            if arm != 'PREDd':
                sae.b_dec1.copy_(sae.b_dec)
        return sae, norm

    def weights(arm, xb, cb):
        if arm == 'PREDk':
            return torch.where(cb, torch.tensor(WC, device=dev), torch.tensor(1.0, device=dev))
        if arm == 'PREDh':
            return (residual(xb.float(), pmu, pV) / pme).clamp(max=CLIP_H)
        return None
    # ---- LAM_D (label-free): throwaway PREDd SAE (seed 2700, LAM 1), 2,000 steps on chunk 0; LAM_D = mean L0 / mean L1 over steps 1,500-1,999
    lam_rec, lam_d = {}, 1.0
    if 'PREDd' in arms:
        s0, n0 = new_sae('PREDd', 2700, 1.0)
        o0 = torch.optim.Adam(s0.parameters(), lr=lr, betas=(0.9, 0.999))
        g0 = torch.Generator(device=dev).manual_seed(2700)
        perm, si, L0s, L1s = torch.randperm(len(Xn), generator=g0, device=dev), 0, [], []
        for k in range(2000):
            if (si + 1) * BS > len(Xn):
                perm, si = torch.randperm(len(Xn), generator=g0, device=dev), 0
            ib = perm[si * BS:(si + 1) * BS]
            si += 1
            lg = train_step(s0, o0, n0(Xn[ib]), n0(Pn[ib]), None, k, 2000, lr, warmup)
            if k >= 1500:
                L0s.append(lg['L0'])
                L1s.append(lg['L1'])
        lam_d = float(np.mean(L0s) / np.mean(L1s))
        lam_rec = {'L0_mean': float(np.mean(L0s)), 'L1_mean': float(np.mean(L1s)), 'lam_delta': lam_d, 'warmup_steps': 2000, 'measured': '1500-1999', 'seed': 2700}
        log(f'LAM_D (label-free warm-up): {lam_rec}')
        (OUT27 / 'train').mkdir(parents=True, exist_ok=True)
        (OUT27 / 'train' / f'{dom}_lam_delta.json').write_text(json.dumps(lam_rec, indent=1))
        del s0, o0
    M = {}
    for arm in arms:
        for s in seeds:
            sae, norm = new_sae(arm, s, lam_d if arm == 'PREDd' else 1.0)
            M[(arm, s)] = {'sae': sae, 'norm': norm, 'opt': torch.optim.Adam(sae.parameters(), lr=lr, betas=(0.9, 0.999)), 'hist': []}
    gens = {s: torch.Generator(device=dev).manual_seed(s) for s in seeds}
    step, ci, cx, cp_, cc_, used = 0, 0, None, None, None, 0
    t0 = time.time()
    tlog, slog, util, wst = time.time(), 0, [], {'PREDk': [], 'PREDh': []}
    while step < steps:
        Xc = Xn if cx is None else torch.cat([cx, Xn])
        Pc = Pn if cp_ is None else torch.cat([cp_, Pn])
        Cc = Cn if cc_ is None else torch.cat([cc_, Cn])
        del Xn, Pn, Cn
        n = (len(Xc) // BS) * BS
        last = ci == len(chunks) - 1
        cx, cp_, cc_ = (None, None, None) if last else (Xc[n:].clone(), Pc[n:].clone(), Cc[n:].clone())
        Xg, Pg, Cg = Xc[:n], Pc[:n], Cc[:n]
        del Xc, Pc, Cc
        perms = {s: torch.randperm(n, generator=gens[s], device=dev) for s in seeds}
        for b in range(n // BS):
            if step >= steps:
                break
            for s in seeds:
                ib = perms[s][b * BS:(b + 1) * BS]
                xb, pb, cb = Xg[ib], Pg[ib], Cg[ib]
                for arm in arms:
                    m = M[(arm, s)]
                    nm = m['norm']
                    w = weights(arm, xb, cb)
                    if w is not None and step % 2000 == 0 and s == seeds[0]:
                        wst[arm].append([float(w.mean()), float((w * (cb if arm == 'PREDk' else torch.ones_like(cb))).sum() / w.sum())])
                    lg = train_step(m['sae'], m['opt'], nm(xb), nm(pb), w, step, total, lr, warmup)
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
        del Xg, Pg, Cg, perms
        if step >= steps or last:
            break
        ci += 1
        Xn, Pn, Cn = gather(pending)
        pending = [ex.submit(st.load_xpc, w, dev) for w in chunks[ci + 1]] if ci + 1 < len(chunks) else []
    tt = time.time() - t0
    for f in pending:
        f.cancel()
    ex.shutdown(wait=False, cancel_futures=True)
    NM = json.loads((TR.OUT25 / f'{dom}_count.json').read_text())['M_used']
    rec0 = {'node': socket.gethostname(), 'steps_run': step, 'steps_schedule': total, 'tokens_used_per_sae': used, 'passes': used / NM,
            'train_time_s': tt, 'steps_per_s': step / tt, 'n_saes_in_job': len(M), 'gpu_util_median': float(np.median(util)) if util else None,
            'gpu_identity': fcheck, 'lam_delta': lam_rec, 'contact_w': WC, 'batch_weight_stats_[mean, proxy share]': wst}
    log(f'train loop done: {json.dumps(rec0)}')
    if a.max_steps:
        return
    xc = torch.from_numpy(T6.sample_x(st, ws, TR.NCAL, 7))
    xe = torch.from_numpy(T6.sample_x(st, ws, TR.NCAL, 8))
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
        plain = MatryoshkaBatchTopKSAE(d_in=1024, n_latents=4096, prefixes=PREFIXES, k=spec['k'], seed=s)
        plain.load_state_dict({k: v for k, v in sae.state_dict().items() if k not in ('W_dec1', 'b_dec1')})
        cfg = {**spec, 'W': 4096, 'prefixes': list(PREFIXES), 'steps': total, 'arm': arm, 'target': sae.target, 'lam': sae.lam,
               'weights': {'PREDk': f'contact proxy x {WC:.4f}', 'PREDh': f'PCA-{K_PCA} residual / mean, clip {CLIP_H}'}.get(arm, 'none'),
               'input': 'x_t', 'lag_frames': L, 'edge_rule': f'{L} <= t < T - {L}'}
        rec = {'cfg': cfg, 'seed': s, 'key': key, **rec0, 'theta_star': float(th), 'eval_fve': float(sst['fve']), 'eval_l0': float(sst['l0']),
               'dead_at_end_train': int((sae.tokens_since_fired >= sae.dead_tokens).sum()),
               'never_fire_on_eval_frac': float((fired == 0).float().mean()), 'history': m['hist']}
        out = BEE27 / dom / 'sae' / key
        out.mkdir(parents=True, exist_ok=True)
        save_checkpoint(out / 'sae_theta.pt', plain, norm, extra={'config': 'eci_v2_7', 'cfg': cfg, 'seed': s, 'theta_star': float(th)})
        torch.save({'W_dec1': sae.W_dec1.detach().cpu(), 'b_dec1': sae.b_dec1.detach().cpu(), 'target': sae.target, 'lam': sae.lam}, out / 'dec1.pt')
        (out / 'train.json').write_text(json.dumps(rec, indent=1, default=float))
        (OUT27 / 'train').mkdir(parents=True, exist_ok=True)
        (OUT27 / 'train' / f'{dom}_{key}.json').write_text(json.dumps(rec, indent=1, default=float))
        log(f'{dom} {key}: theta* {th:.4f} eval FVE {sst["fve"]:.5f} L0 {sst["l0"]:.2f} dead(train) {rec["dead_at_end_train"]} '
            f'never-fire(eval) {rec["never_fire_on_eval_frac"]:.4f}')
    log('ALL CHECKPOINTS SAVED')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=('prep', 'proxydiag', 'train'))
    p.add_argument('--dom', default='mice', choices=('mice', 'ants'))
    p.add_argument('--arms', default=','.join(ARMS))
    p.add_argument('--seeds', default=','.join(map(str, SEEDS)))
    p.add_argument('--max_steps', type=int, default=0)
    a = p.parse_args()
    {'prep': cmd_prep, 'proxydiag': cmd_proxydiag, 'train': cmd_train}[a.step](a)
