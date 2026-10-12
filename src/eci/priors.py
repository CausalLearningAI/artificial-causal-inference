"""
Label-free behavioural PRIORS as extra loss terms of the ECI Matryoshka BatchTopK SAE (C1 recipe).

INV (invariance): "a behaviour is the same wherever and in whatever orientation it happens".
    Every training frame gets ONE augmented copy by an exact symmetry of the 512 px frame (an element of the dihedral
    group D4 other than the identity: k quarter-turns counter-clockwise, then optionally a left-right flip). The 32 x 32
    patch grid (16 px patches of the 512 frame = 14 px patches of the 448 encoder input) maps exactly, so the token of
    patch p in the original frame is paired with the token at d4_pos_map(g)[p] in the augmented frame (frame t and
    frame t-5 augmented the same way, both encoded by the same FgEncoder at 448).
    term:  L_inv = mean over tokens of ||a(x) - a(x')||^2 / (||a(x)||^2 + eps),
           a(.) = relu(pre-activations) = the DENSE pre-TopK encoder activations (all latents). TopK selection is not
           differentiable, and the dense activations also carry the ranking that TopK reads; gradients flow through
           both a(x) and a(x'). eps = 1 (normalised input units; ||a(x)||^2 is O(10^3), so eps only guards a = 0).
    The reconstruction term stays on x only (x' enters only this term), so every arm sees the same token
    presentations in the reconstruction loss.
PRED (prediction): "a behaviour is what determines what the animal does next".
    A linear head P (W_P (m, 768), b_P (768,), zero init) maps the sparse BatchTopK code z_t (the training code, all m
    latents) to the patch's future change d+ = token_{t+5} - token_t (static block, same patch position, 1 s ahead),
    in the normalised static-block units (TokenNorm scale of the static block; no mean subtraction, b_P absorbs it).
    term:  L_pred = mean over tokens with a valid t+5 frame of ||d+ - P z_t||^2 / Var(d+),
           Var(d+) = mean squared distance of d+ to its mean over the valid training tokens (fixed before training).
    t+5 beyond the video's last frame -> the token is masked out of this term.
Total loss = Matryoshka loss (+ AuxK, unchanged) + lambda L_inv + mu L_pred. P has its own Adam (same lr / schedule);
the SAE optimiser, sampler, seeds and gradient clip are those of motion_t3.train_one, so lambda = mu = 0 reproduces it.
"""
import json
import math
import time

import numpy as np
import torch

from src.eci.sae import MatryoshkaBatchTopKSAE, TokenNorm, _lr_at, geometric_median, save_checkpoint

D = 768
GRID = 32
INV_EPS = 1.0


# ---------------------------------------------------------------------------------------------- D4 symmetries
def d4_apply(a, g):
    """a: (H, W[, C]) array, g in 0..7: rotate k = g % 4 quarter-turns counter-clockwise, then flip left-right if g >= 4."""
    b = np.rot90(a, int(g) % 4, axes=(0, 1))
    if g >= 4:
        b = b[:, ::-1]
    return np.ascontiguousarray(b)


def d4_pos_map(g, grid=GRID):
    """(grid*grid,) int64: the flat patch position, in the transformed frame, of every original patch position."""
    J = d4_apply(np.arange(grid * grid).reshape(grid, grid), g)
    inv = np.empty(grid * grid, np.int64)
    inv[J.ravel()] = np.arange(grid * grid)
    return inv


def check_d4(px=16, grid=GRID):
    """Self-test: every transformed pixel block equals the transformed original block at the mapped position."""
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (grid * px, grid * px, 3), dtype=np.uint8)
    for g in range(8):
        t = d4_apply(img, g)
        m = d4_pos_map(g, grid)
        for p in rng.choice(grid * grid, 64, replace=False):
            r, c = divmod(int(p), grid)
            r2, c2 = divmod(int(m[p]), grid)
            blk = d4_apply(img[r * px:(r + 1) * px, c * px:(c + 1) * px], g)
            if not np.array_equal(t[r2 * px:(r2 + 1) * px, c2 * px:(c2 + 1) * px], blk):
                raise AssertionError(f'D4 map wrong for g={g}, p={p}')
    return True


# ---------------------------------------------------------------------------------------------- loss terms
def inv_term(pre, pre_a, eps=INV_EPS):
    a, b = torch.relu(pre), torch.relu(pre_a)
    return ((a - b).pow(2).sum(1) / (a.pow(2).sum(1) + eps)).mean()


def pred_term(z, dn, ok, W, b, var):
    if not bool(ok.any()):
        return torch.zeros((), device=z.device)
    r = dn[ok] - (z[ok] @ W + b)
    return r.pow(2).sum(1).mean() / var


def recon_terms(sae, x, z):
    """Matryoshka per-prefix reconstruction SSE per token (list), as forward_train computes it."""
    recon = sae.b_dec.expand_as(x)
    out, start = [], 0
    for m in sae.prefixes:
        recon = recon + z[:, start:m] @ sae.W_dec[start:m]
        out.append((recon - x).pow(2).sum(1).mean())
        start = m
    return out


class PriorData:
    """GPU-resident training tensors. X (N, 1536) fp16 = [token_t, token_t - token_{t-5}] (the C1 input);
    Xa (N, 1536) fp16 = the same for the augmented frame pair (or None); Dr (N, 768) fp16 = token_{t+5} - token_t raw
    (or None); ok (N,) bool = t+5 inside the video."""

    def __init__(self, X, Xa=None, Dr=None, ok=None):
        self.X, self.Xa, self.Dr, self.ok = X, Xa, Dr, ok

    def dn(self, b, norm):
        return self.Dr[b].float() * norm.scale[:D]


def d_var(data, norm, idx):
    """Var(d+) in normalised units over the valid tokens of idx (mean squared distance to the mean)."""
    b = idx[data.ok[idx]]
    dn = data.dn(b, norm)
    return float((dn - dn.mean(0)).pow(2).sum(1).mean())


def build_model(data, seed, dev, width=4096, k=16):
    """The exact init of motion_t3.train_one: seeds, init subsample, per-block TokenNorm, b_dec = geometric median."""
    torch.manual_seed(seed)
    g = np.random.default_rng(seed)
    N, d = data.X.shape
    ii = torch.from_numpy(np.sort(g.choice(N, min(500_000, N), replace=False))).to(dev)
    init = data.X[ii].float()
    norm = TokenNorm(d).to(dev).fit_blocks(init, [D, D], seed=seed)
    sae = MatryoshkaBatchTopKSAE(d_in=d, n_latents=width, prefixes=(width // 8, width // 4, width // 2, width), k=k,
                                 seed=seed).to(dev)
    with torch.no_grad():
        sae.b_dec.copy_(geometric_median(norm(init[:100_000])))
    del init
    return sae, norm, ii


def train_prior(data, seed, steps, lam, mu, dev, log, bs=4096, lr=5e-4, warmup=500, total=None, out=None,
                extra=None, width=4096, k=16):
    """motion_t3.train_one + lambda L_inv + mu L_pred. total = schedule length (default steps; the warm-up run of the
    lambda / mu rule uses the first `steps` steps of a `total`-step schedule). Returns (sae, norm, P, var, history)."""
    sae, norm, ii = build_model(data, seed, dev, width, k)
    opt = torch.optim.Adam(sae.parameters(), lr=lr, betas=(0.9, 0.999))
    total = total or steps
    var = d_var(data, norm, ii) if data.Dr is not None else None
    W = torch.zeros(width, D, device=dev, requires_grad=True)
    bP = torch.zeros(D, device=dev, requires_grad=True)
    optP = torch.optim.Adam([W, bP], lr=lr, betas=(0.9, 0.999)) if mu > 0 else None
    N = data.X.shape[0]
    gen = torch.Generator(device=dev).manual_seed(seed)
    perm, p = torch.randperm(N, generator=gen, device=dev), 0
    history, t0 = [], time.time()
    for step in range(steps):
        if p + bs > N:
            perm, p = torch.randperm(N, generator=gen, device=dev), 0
        b = perm[p:p + bs]
        p += bs
        cur = _lr_at(step, total, lr, warmup)
        for pg in opt.param_groups:
            pg['lr'] = cur
        x = norm(data.X[b])
        loss, logs = sae.forward_train(x)
        if not math.isfinite(logs['loss']):
            raise RuntimeError(f'non-finite loss at step {step}: {logs}')
        if lam > 0 or mu > 0:
            pre = sae.pre_acts(x)
            if lam > 0:
                li = inv_term(pre, sae.pre_acts(norm(data.Xa[b])))
                loss = loss + lam * li
                logs['inv_raw'] = float(li)
            if mu > 0:
                z, _ = sae.batch_topk(pre)
                lp = pred_term(z, data.dn(b, norm), data.ok[b], W, bP, var)
                loss = loss + mu * lp
                logs['pred_raw'] = float(lp)
                for pg in optP.param_groups:
                    pg['lr'] = cur
                optP.zero_grad(set_to_none=True)
            logs['loss_total'] = float(loss)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        sae.remove_parallel_grad()
        gn = torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        opt.step()
        if optP is not None:
            torch.nn.utils.clip_grad_norm_([W, bP], 1.0)
            optP.step()
        sae.normalize_decoder()
        if step % 1000 == 0 or step == steps - 1:
            logs.update(step=step, elapsed_s=round(time.time() - t0, 1), grad_norm=float(gn))
            history.append(logs)
            log(f"    step {step}/{steps} fve {logs['fve']:.4f} l0 {logs['l0']:.1f} dead {logs['n_dead']} "
                + (f"inv {logs['inv_raw']:.4f} " if 'inv_raw' in logs else '')
                + (f"pred {logs['pred_raw']:.4f} " if 'pred_raw' in logs else '') + f"{logs['elapsed_s']}s")
    sae.eval()
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        dead = int((sae.tokens_since_fired >= sae.dead_tokens).sum())
        save_checkpoint(out / 'sae.pt', sae, norm, extra={**(extra or {}), 'seed': seed, 'lam': lam, 'mu': mu,
                                                           'var_dplus': var})
        if mu > 0:
            torch.save({'W': W.detach().cpu(), 'b': bP.detach().cpu(), 'var_dplus': var}, out / 'pred_head.pt')
        (out / 'train.json').write_text(json.dumps({**(extra or {}), 'seed': seed, 'lam': lam, 'mu': mu,
                                                    'var_dplus': var, 'n_pool': N, 'steps': steps,
                                                    'token_presentations': steps * bs,
                                                    'train_time_s': time.time() - t0, 'dead_at_end_train': dead,
                                                    'dead_frac_train': dead / width,
                                                    'norm_scale_blocks': [float(norm.scale[0]), float(norm.scale[D])],
                                                    'gpu': torch.cuda.get_device_name(0), 'history': history},
                                                   indent=1))
    return sae, norm, (W.detach(), bP.detach()), var, history


# ---------------------------------------------------------------------------------------------- measurement
@torch.no_grad()
def ridge_head(Z, Y, alpha_rel=1e-3):
    """Least-squares linear head with intercept, ridge alpha = alpha_rel x mean diagonal of the centred Gram.
    Z (n, m), Y (n, q) float32 GPU -> (W (m, q), b (q,))."""
    mz, my = Z.mean(0), Y.mean(0)
    Zc = Z - mz
    G = Zc.T @ Zc
    G.diagonal().add_(alpha_rel * float(G.diagonal().mean()) + 1e-6)
    W = torch.linalg.solve(G, Zc.T @ (Y - my))
    return W, my - mz @ W


@torch.no_grad()
def measure_terms(sae, norm, data, dev, n_fit=60, n_eval=25, bs=4096, seed=12345):
    """The three loss values on held batches of the training pool (fixed generator, independent of the training one):
    L_rec = Matryoshka sum over prefixes of the per-token SSE (BatchTopK code, as in training), L_inv raw, L_pred raw
    with the least-squares head fitted on n_fit (60) other batches (so it is defined for every model, PRED or not), and the
    training-code FVE."""
    gen = torch.Generator(device=dev).manual_seed(seed)
    N = data.X.shape[0]
    perm = torch.randperm(N, generator=gen, device=dev)
    var = d_var(data, norm, perm[:500_000]) if data.Dr is not None else None
    out = {'n_eval_tokens': n_eval * bs, 'n_fit_tokens': n_fit * bs, 'var_dplus': var}
    if data.Dr is not None:
        Zs, Ys = [], []
        for i in range(n_fit):
            b = perm[(n_eval + i) * bs:(n_eval + i + 1) * bs]
            b = b[data.ok[b]]
            z, _ = sae.batch_topk(sae.pre_acts(norm(data.X[b])))
            Zs.append(z)
            Ys.append(data.dn(b, norm))
        W, bb = ridge_head(torch.cat(Zs), torch.cat(Ys))
        del Zs, Ys
    rec, inv, pred, fve_num, fve_den = [], [], [], 0.0, 0.0
    for i in range(n_eval):
        b = perm[i * bs:(i + 1) * bs]
        x = norm(data.X[b])
        pre = sae.pre_acts(x)
        z, _ = sae.batch_topk(pre)
        r = recon_terms(sae, x, z)
        rec.append(float(torch.stack(r).sum()))
        fve_num += float(r[-1])
        fve_den += float((x - x.mean(0)).pow(2).sum(1).mean())
        if data.Xa is not None:
            inv.append(float(inv_term(pre, sae.pre_acts(norm(data.Xa[b])))))
        if data.Dr is not None:
            pred.append(float(pred_term(z, data.dn(b, norm), data.ok[b], W, bb, var)))
    out.update(L_rec=float(np.mean(rec)), fve_train_code=1 - fve_num / fve_den,
               L_inv_raw=float(np.mean(inv)) if inv else None, L_pred_raw_ridge=float(np.mean(pred)) if pred else None)
    if data.Xa is not None:  # how different the encoder input itself is under the symmetry
        b = perm[:bs]
        x, xa = norm(data.X[b]), norm(data.Xa[b])
        # normalised (centred) input: ||x - x'||^2 / ||x||^2 per block; raw-token cosine of the static block
        out['input_rel_change_static'] = float(((x[:, :D] - xa[:, :D]).pow(2).sum(1) / x[:, :D].pow(2).sum(1)).mean())
        out['input_rel_change_motion'] = float(((x[:, D:] - xa[:, D:]).pow(2).sum(1) / x[:, D:].pow(2).sum(1)).mean())
        out['input_cos_static_raw'] = float(torch.nn.functional.cosine_similarity(
            data.X[b][:, :D].float(), data.Xa[b][:, :D].float()).mean())
        out['frac_preact_pos'] = float((sae.pre_acts(x) > 0).float().mean())
        out['frac_preact_pos_aug'] = float((sae.pre_acts(xa) > 0).float().mean())
    return out


# ---------------------------------------------------------------------------------------------- composition helpers
def entropy_rows(H):
    """Row-wise entropy (nats) of count matrices H (n, c) (torch)."""
    s = H.sum(1, keepdim=True).clamp_min(1e-12)
    p = H / s
    return -(torch.where(p > 0, p * p.clamp_min(1e-30).log(), torch.zeros_like(p))).sum(1)


def href(cells, K, n_draw=200, seed=0):
    """Mean entropy of K patches drawn uniformly (without replacement) from all kept eval patches' 8 x 8 cells."""
    rng = np.random.default_rng(seed)
    h = []
    for _ in range(n_draw):
        c = np.bincount(cells[rng.choice(len(cells), K, replace=False)], minlength=64).astype(np.float64)
        p = c[c > 0] / c.sum()
        h.append(-(p * np.log(p)).sum())
    return float(np.mean(h))


@torch.no_grad()
def neuron_r2(Z, Y, half):
    """Cross-fitted single-neuron R^2 of Y (n, q) from each latent alone (Y ~ c + z_j w_j, fit on one half, R^2 on the
    other, mean of both directions). Z (n, m) float32 GPU, half (n,) bool GPU -> (m,) numpy."""
    out = []
    for a in (half, ~half):
        A, B = a, ~a
        za, ya = Z[A], Y[A]
        mz, my = za.mean(0), ya.mean(0)
        vz = (za - mz).pow(2).sum(0)  # (m,)
        w = ((za - mz).T @ (ya - my)) / vz.clamp_min(1e-12)[:, None]  # (m, q)
        zb, yb = Z[B] - mz, Y[B] - my
        sst = float(yb.pow(2).sum())
        cross = ((zb.T @ yb) * w).sum(1)  # sum_i z_ij (y_i . w_j)
        sse = sst - 2 * cross + w.pow(2).sum(1) * zb.pow(2).sum(0)
        r2 = 1 - sse / sst
        r2[vz <= 1e-12] = 0.0
        out.append(r2)
    return ((out[0] + out[1]) / 2).cpu().numpy()


@torch.no_grad()
def code_r2(Z, Y, half, alpha_rel=1e-3):
    """Cross-fitted R^2 of the ridge head on the whole code (mean of both directions)."""
    r = []
    for a in (half, ~half):
        W, b = ridge_head(Z[a], Y[a], alpha_rel)
        yb = Y[~a]
        r.append(1 - float((yb - (Z[~a] @ W + b)).pow(2).sum()) / float((yb - yb.mean(0)).pow(2).sum()))
    return float(np.mean(r))
