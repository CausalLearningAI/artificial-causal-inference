"""
Synthetic sanity test of the Spatial-SAE (src/eci/spatial_sae.py): on patch data with spatially extended concepts,
does it recover the true concepts better than a tokenwise SAE at matched width and sparsity?

Data (paper arXiv 2610.04112, appendix 'Synthetic Experiments', default cell): c = 256 true unit-norm concept
directions in d = 32, images of 16 x 16 patches, 4-neighbour grid Laplacian L.
    smooth stream     m_S = 9 active concepts per image; g = relu((I + tau L)^-1 psi), psi ~ N(0, I_p), tau = 16;
                      keep g above its q = 0.5 quantile; S[:, a] = rho_S g / max g
    innovation stream m_R = 4 active concepts; a centre per concept; triangular envelope (1 - dist / r)_+, r = 4,
                      restricted to the grid bipartition side of the centre; R[:, a] = rho_R * envelope
    H = (S + R) D + N(0, sigma^2), sigma = 0.2;  rho_S = rho_R = 1 (not given in the paper)
Learned width w = nu * c = 128 (nu = 0.5), sparsity target k = round(true mean L0 per patch).
--mask-frac f > 0: each image keeps only a spatially coherent fraction f of its patches (a thresholded smooth random
field), mimicking the foreground-only patches of the ECI mice/ants SAEs (missing neighbours).

Models (all BatchTopK, Adam lr 1e-3, 100 warmup steps, cosine to 1e-5, grad clip 1, 5000 steps of 16 images,
decoder rows renormalised every step; fresh images every step):
    B        tokenwise BatchTopK SAE (PlainBatchTopKSAE, reanimation 5e-3)
    SS       Spatial-SAE, context attention, k_S:k_R = 3:1 (the paper's 12:4 BatchTopK split), mu_S = 1e-3,
             mu_R = mu_S / 3 (paper's synthetic setting)
    SS-mu0   same, mu = 0 (decomposition + neighbour estimator only, no spectral prior)
    SS-stat  static smooth estimator (paper's control), mu as SS
Metrics on 2000 held-out images (as SynthSAEBench / the paper):
    MCC      mean |cosine| of the Hungarian matching between learned decoder rows and true directions
    F1       mean support F1 (z_j > 0 vs true code_a > 0, over held-out patches) of the matched pairs
    R2       1 - SSE / SS about the mean
Usage: python scripts/eci/spatial_sae_synth.py --seeds 0 1 2 [--mask-frac 0.5] --out <json>
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.eci.spatial_sae import PlainBatchTopKSAE, SpatialBatchTopKSAE, lr_paper, neighbor_index  # noqa: E402

G = 16
P = G * G


def grid_L(g=G):
    A = np.zeros((g * g, g * g))
    for r in range(g):
        for c in range(g):
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dy, c + dx
                if 0 <= rr < g and 0 <= cc < g:
                    A[r * g + c, rr * g + cc] = 1
    return np.diag(A.sum(1)) - A


class DGP:
    def __init__(self, d=32, c=256, tau=16.0, q=0.5, r=4.0, m_s=9, m_r=4, sigma=0.2, rho_s=1.0, rho_r=1.0,
                 mask_frac=0.0, seed=0, device='cuda'):
        self.d, self.c, self.q, self.r, self.m_s, self.m_r, self.sigma = d, c, q, r, m_s, m_r, sigma
        self.rho_s, self.rho_r, self.mask_frac, self.device = rho_s, rho_r, mask_frac, device
        g = torch.Generator().manual_seed(10_000 + seed)
        D = torch.randn(c, d, generator=g)
        self.D = (D / D.norm(dim=1, keepdim=True)).to(device)
        L = torch.tensor(grid_L(), dtype=torch.float32)
        self.T = torch.linalg.inv(torch.eye(P) + tau * L).to(device)
        self.Tm = torch.linalg.inv(torch.eye(P) + 8.0 * L).to(device)  # for the coherent masks
        yy, xx = torch.meshgrid(torch.arange(G), torch.arange(G), indexing='ij')
        self.yx = torch.stack([yy.flatten(), xx.flatten()], 1).float().to(device)  # (P, 2)

    def sample(self, n, gen):
        dev, c = self.device, self.c
        # smooth stream
        Z_s = torch.zeros(n, P, c, device=dev)
        act_s = torch.argsort(torch.rand(n, c, generator=gen, device=dev), 1)[:, :self.m_s]       # (n, m_s)
        psi = torch.randn(n, self.m_s, P, generator=gen, device=dev)
        gfield = torch.relu(psi @ self.T.t())                                                      # (n, m_s, P)
        thr = torch.quantile(gfield, self.q, dim=2, keepdim=True)
        u = gfield * (gfield > thr)
        u = self.rho_s * u / u.amax(2, keepdim=True).clamp_min(1e-8)
        Z_s.scatter_add_(2, act_s[:, None, :].expand(n, P, self.m_s), u.transpose(1, 2))
        # innovation stream
        Z_r = torch.zeros(n, P, c, device=dev)
        act_r = torch.argsort(torch.rand(n, c, generator=gen, device=dev), 1)[:, :self.m_r]
        ctr = torch.randint(0, P, (n, self.m_r), generator=gen, device=dev)
        cyx = self.yx[ctr]                                                                          # (n, m_r, 2)
        dist = (self.yx[None, None] - cyx[:, :, None]).norm(dim=-1)                                 # (n, m_r, P)
        par = ((self.yx[None, None].sum(-1) - cyx[:, :, None].sum(-1)) % 2 == 0).float()
        e = self.rho_r * par * torch.relu(1 - dist / self.r)
        Z_r.scatter_add_(2, act_r[:, None, :].expand(n, P, self.m_r), e.transpose(1, 2))
        Z = Z_s + Z_r
        H = Z @ self.D + self.sigma * torch.randn(n, P, self.d, generator=gen, device=dev)
        if self.mask_frac > 0:
            mf = torch.randn(n, P, generator=gen, device=dev) @ self.Tm.t()
            keep = mf >= torch.quantile(mf, 1 - self.mask_frac, dim=1, keepdim=True)
        else:
            keep = torch.ones(n, P, dtype=torch.bool, device=dev)
        fi, pi = torch.nonzero(keep, as_tuple=True)
        return H[fi, pi], Z[fi, pi], fi, pi


def make_model(name, w, k, seed, mask):
    if name == 'B':
        return PlainBatchTopKSAE(32, w, k, seed=seed)
    k_s = max(1, int(round(k * 12 / 16)))
    k_s = min(k_s, k - 1)
    kw = dict(d_in=32, n_latents=w, k=k, k_s=k_s, n_heads=4, max_dist=2, grid=(G, G), mu_s=1e-3, mu_r=1e-3 / 3,
              reanim_coef=1e-3, seed=seed)
    if name == 'SS-mu0':
        kw.update(mu_s=0.0, mu_r=0.0)
    if name == 'SS-stat':
        kw.update(estimator='static')
    m = SpatialBatchTopKSAE(**kw)
    return m


def train(model, dgp, steps, n_img, seed, device, log):
    gen = torch.Generator(device=device).manual_seed(seed)
    model.to(device)
    with torch.no_grad():
        H, _, _, _ = dgp.sample(64, gen)
        model.b_dec.copy_(H.mean(0))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
    for step in range(steps):
        for pg in opt.param_groups:
            pg['lr'] = lr_paper(step, steps)
        H, _, fi, pi = dgp.sample(n_img, gen)
        nbr = neighbor_index(fi, pi, G, G, model.offsets) if hasattr(model, 'offsets') else None
        loss, logs = model.forward_train(H, nbr, fi, n_img)
        if not torch.isfinite(loss):
            continue  # paper: steps with a non-finite loss are skipped
        opt.zero_grad(set_to_none=True)
        loss.backward()
        model.remove_parallel_grad()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        model.normalize_decoder()
        if step % 1000 == 0 or step == steps - 1:
            log(f'    step {step} ' + ' '.join(f'{a} {b:.4g}' for a, b in logs.items()))
    return model


@torch.no_grad()
def evaluate(model, dgp, n_eval, device):
    gen = torch.Generator(device=device).manual_seed(123_456)
    model.eval()
    Zs, Ts, sse, xs, xss, nt = [], [], 0.0, 0.0, 0.0, 0
    for b in range(0, n_eval, 100):
        H, Zt, fi, pi = dgp.sample(100, gen)
        if isinstance(model, SpatialBatchTopKSAE):
            nbr = neighbor_index(fi, pi, G, G, model.offsets)
            S, R = model.encode(H, nbr)
            Z = torch.cat([S, R], 1)
            Hh = model.decode(S, R)
        else:
            Z = model.encode(H)
            Hh = Z @ model.W_dec + model.b_dec
        sse += float((H - Hh).pow(2).sum())
        xs = xs + H.double().sum(0)
        xss += float(H.double().pow(2).sum())
        nt += H.shape[0]
        Zs.append((Z > 0).cpu())
        Ts.append((Zt > 0).cpu())
    total_ss = xss - float((xs ** 2).sum()) / nt
    A = torch.cat(Zs).float()      # (N, w) learned support
    T = torch.cat(Ts).float()      # (N, c) true support
    Wd = model.W_dec.detach()
    cos = (Wd / Wd.norm(dim=1, keepdim=True)) @ dgp.D.t()             # (w, c)
    ri, ci = linear_sum_assignment(-cos.abs().cpu().numpy())
    mcc = float(cos.abs().cpu().numpy()[ri, ci].mean())
    tp = (A[:, ri] * T[:, ci]).sum(0)
    f1 = 2 * tp / (A[:, ri].sum(0) + T[:, ci].sum(0)).clamp_min(1)
    model.train()
    return {'mcc': mcc, 'f1': float(f1.mean()), 'r2': 1 - sse / total_ss, 'l0': float(A.sum(1).mean()),
            'dead_frac': float((A.sum(0) == 0).float().mean()), 'true_l0': float(T.sum(1).mean())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--models', nargs='+', default=['B', 'SS', 'SS-mu0', 'SS-stat'])
    p.add_argument('--steps', type=int, default=5000)
    p.add_argument('--n-img', type=int, default=16)
    p.add_argument('--n-eval', type=int, default=2000)
    p.add_argument('--nu', type=float, default=0.5)
    p.add_argument('--mask-frac', type=float, default=0.0)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log = lambda s: print(s, flush=True)  # noqa: E731
    res = []
    for seed in args.seeds:
        dgp = DGP(mask_frac=args.mask_frac, seed=seed, device=device)
        gen = torch.Generator(device=device).manual_seed(999 + seed)
        _, Zt, _, _ = dgp.sample(200, gen)
        k = max(2, int(round(float((Zt > 0).sum(1).float().mean()))))
        w = int(args.nu * dgp.c)
        log(f'seed {seed}: true mean L0 {float((Zt > 0).sum(1).float().mean()):.2f} -> k = {k}, w = {w}')
        for name in args.models:
            t0 = time.time()
            torch.manual_seed(seed)
            m = train(make_model(name, w, k, seed, args.mask_frac), dgp, args.steps, args.n_img, seed, device, log)
            ev = evaluate(m, dgp, args.n_eval, device)
            ev.update(model=name, seed=seed, k=k, w=w, mask_frac=args.mask_frac, train_s=round(time.time() - t0, 1))
            log(f'  {name:8s} ' + json.dumps({a: (round(b, 4) if isinstance(b, float) else b) for a, b in ev.items()}))
            res.append(ev)
    summ = {}
    for name in args.models:
        rs = [r for r in res if r['model'] == name]
        summ[name] = {m: (float(np.mean([r[m] for r in rs])), float(np.std([r[m] for r in rs])))
                      for m in ('mcc', 'f1', 'r2', 'l0', 'dead_frac')}
        log(f'SUMMARY {name:8s} ' + ' '.join(f'{m} {v[0]:.4f}+-{v[1]:.4f}' for m, v in summ[name].items()))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({'args': vars(args), 'runs': res, 'summary': summ}, indent=1))


if __name__ == '__main__':
    main()
