"""
Spatial-SAE (Mencattini, Nikolaou, Crisostomi, Fel, Montagna, Rodola, Locatello, "The Independence Prior of SAEs
Fragments Visual Concepts", arXiv 2610.04112), ported from the paper (no official code is linked; the authors extend
the `overcomplete` library). BatchTopK sparsifier, context-attention smooth-field estimator, as the paper's final
BatchTopK model (budget split k_S:k_R = 12:4 at k = 16, mu = 1e-3).

Model (Markov-Field LRH: patch code z_i = s_i + r_i, smooth field S + innovation field R, dictionary split
D = [D_S; D_R] with c_S / c = k_S / k):
    smooth      s_i = BatchTopK_S(relu(q_i + Att(LN(q_i), LN(KV_i), LN(KV_i))))
                q_i  = mean over the PRESENT neighbours j of (x_j - b_dec) E_S + b_S     (never sees patch i)
                KV_i = [(x_j - b_dec) E_S + b_S  or  a learned 'missing' vector] + relative-position embedding,
                       one row per neighbour offset (12 offsets, Manhattan distance 1..2)
                Att  = 4-head attention, inner width c_S (c_S / 4 per head)
    innovation  r_i = BatchTopK_R(relu((x_i - b_dec - sg(s_i D_S)) E_R + b_R))
    decoder     x_hat_i = s_i D_S + r_i D_R + b_dec, decoder rows unit norm
    estimator='static' (paper's control): s_i = BatchTopK_S(relu((x_i - b_dec) E_S + b_S)).

Loss per frame f with p_f patches (paper eq. app-vision-objective, width-normalised), averaged over the frames of
the batch:
    rec   ||X_f - X_hat_f||_F^2 / (p_f d)
    spec  mu_S Tr(S_f^T L_f S_f) / c_S  +  mu_R Tr(R_f^T (kappa I - L_f) R_f) / c_R
          Tr(S^T L S) = sum over graph edges ||s_i - s_j||^2; kappa = lambda_max of the Laplacian of the FULL grid
          graph (for a frame whose patches are a subset of the grid -- e.g. foreground patches only -- L_f is the
          Laplacian of the induced subgraph, and L_f <= kappa I still holds, so the innovation term stays >= 0)
    reanimation  - coef * sum of the pre-activations of latents that did not fire in the batch / (n_tokens * width),
          per stream (paper: 1e-3 per spatial stream)

Patch graph: unweighted, patches at Manhattan distance 1..max_dist (paper: 2 -> 12 neighbours). Patches missing from
a frame (outside the grid or outside the foreground) are not graph nodes; in the attention they are the learned
'missing' vector (the paper uses a learned vector for positions outside the grid).

Inference: per-stream global thresholds (EMA of the smallest BatchTopK-kept activation, as src/eci/sae.py), applied
in order (S first, then R on the residual of the thresholded S). The returned code is [S, R] (n, c_S + c_R).

Functions / classes:
    grid_offsets, grid_laplacian_lmax     patch-graph helpers
    neighbor_index                        (n,) frame id + grid position -> (n, K) neighbour token index (-1 missing)
    SpatialBatchTopKSAE                   the model (encode / forward_train)
    PlainBatchTopKSAE                     single-stream BatchTopK SAE with the same loss normalisation (synthetic test)
    save_spatial / load_spatial           checkpoint helpers
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------------------------- patch graph
def grid_offsets(max_dist=2):
    """Relative (drow, dcol) offsets with 1 <= |drow| + |dcol| <= max_dist, fixed order."""
    return [(dy, dx) for dy in range(-max_dist, max_dist + 1) for dx in range(-max_dist, max_dist + 1)
            if 0 < abs(dy) + abs(dx) <= max_dist]


def grid_laplacian_lmax(h, w, max_dist=2):
    """Largest eigenvalue of the combinatorial Laplacian of the full h x w grid graph."""
    offs = grid_offsets(max_dist)
    n = h * w
    A = np.zeros((n, n))
    for r in range(h):
        for c in range(w):
            for dy, dx in offs:
                rr, cc = r + dy, c + dx
                if 0 <= rr < h and 0 <= cc < w:
                    A[r * w + c, rr * w + cc] = 1
    L = np.diag(A.sum(1)) - A
    return float(np.linalg.eigvalsh(L)[-1])


def neighbor_index(frame, pos, grid_w, grid_h, offsets):
    """frame (n,) long, ids 0..n_frames-1 (any order); pos (n,) long, row-major grid position; a (frame, pos) pair
    appears at most once. -> (n, K) long: the row of the token at pos + offset in the same frame, -1 if absent."""
    device = frame.device
    n_frames = int(frame.max()) + 1 if frame.numel() else 0
    P = grid_w * grid_h
    table = torch.full((n_frames * P,), -1, dtype=torch.long, device=device)
    table[frame * P + pos] = torch.arange(frame.numel(), device=device)
    r, c = pos // grid_w, pos % grid_w
    out = []
    for dy, dx in offsets:
        rr, cc = r + dy, c + dx
        ok = (rr >= 0) & (rr < grid_h) & (cc >= 0) & (cc < grid_w)
        idx = torch.where(ok, frame * P + rr.clamp(0, grid_h - 1) * grid_w + cc.clamp(0, grid_w - 1),
                          torch.zeros_like(pos))
        out.append(torch.where(ok, table[idx], torch.full_like(pos, -1)))
    return torch.stack(out, 1)


def _batch_topk(pre, k):
    """Keep the n*k largest post-ReLU activations of pre (n, m) across the batch -> (z, kept values)."""
    acts = torch.relu(pre)
    flat = acts.flatten()
    n_keep = min(acts.shape[0] * k, flat.numel())
    v, i = flat.topk(n_keep, sorted=False)
    return torch.zeros_like(flat).scatter_(0, i, v).view_as(acts), v


def _update_threshold(buf, kept, lr):
    pos = kept[kept > 0]
    if pos.numel():
        mn = pos.min()
        if buf < 0:
            buf.copy_(mn)
        else:
            buf.mul_(1 - lr).add_(lr * mn)


def _per_frame_mean(per_token, frame, n_frames):
    """Mean over frames of the per-frame SUM of per_token (n,)."""
    s = torch.zeros(n_frames, device=per_token.device, dtype=per_token.dtype).index_add_(0, frame, per_token)
    return s.mean()


def _edge_sq(Z, nbr):
    """(n,) per token: 0.5 * sum over present neighbours ||z_i - z_j||^2 (each undirected edge counted once overall)."""
    present = nbr >= 0
    Zj = Z[nbr.clamp_min(0)]                                     # (n, K, m)
    d = (Z[:, None, :] - Zj).pow(2).sum(-1) * present            # (n, K)
    return 0.5 * d.sum(1)


# ---------------------------------------------------------------------------------------------- models
class SpatialBatchTopKSAE(nn.Module):
    def __init__(self, d_in=768, n_latents=1024, k=16, k_s=12, estimator='context', n_heads=4, max_dist=2,
                 grid=(32, 32), mu_s=1e-3, mu_r=1e-3, reanim_coef=1e-3, threshold_lr=0.01, seed=0):
        super().__init__()
        if not 0 < k_s < k:
            raise ValueError(f'need 0 < k_s < k, got k_s={k_s}, k={k}')
        if estimator not in ('context', 'static'):
            raise ValueError(estimator)
        # atoms in proportion to budget, largest-remainder rounding (2 streams -> plain rounding)
        c_s = int(round(n_latents * k_s / k))
        self.d_in, self.n_latents, self.k, self.k_s, self.k_r = d_in, n_latents, k, k_s, k - k_s
        self.c_s, self.c_r = c_s, n_latents - c_s
        self.estimator, self.n_heads, self.max_dist, self.grid = estimator, n_heads, max_dist, tuple(grid)
        self.mu_s, self.mu_r, self.reanim_coef, self.threshold_lr, self.seed = mu_s, mu_r, reanim_coef, threshold_lr, seed
        self.offsets = grid_offsets(max_dist)
        self.kappa = grid_laplacian_lmax(grid[0], grid[1], max_dist)

        g = torch.Generator().manual_seed(seed)
        W_dec = torch.randn(n_latents, d_in, generator=g)
        W_dec = W_dec / W_dec.norm(dim=1, keepdim=True)
        self.W_dec = nn.Parameter(W_dec)                              # rows [:c_s] = D_S, [c_s:] = D_R
        self.E_s = nn.Parameter(W_dec[:c_s].t().clone())
        self.E_r = nn.Parameter(W_dec[c_s:].t().clone())
        self.b_s = nn.Parameter(torch.zeros(c_s))
        self.b_r = nn.Parameter(torch.zeros(self.c_r))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        if estimator == 'context':
            K = len(self.offsets)
            if c_s % n_heads:
                raise ValueError(f'c_s={c_s} not divisible by n_heads={n_heads}')
            self.pos_emb = nn.Parameter(0.02 * torch.randn(K, c_s, generator=g))
            self.missing = nn.Parameter(0.02 * torch.randn(c_s, generator=g))
            self.ln_q = nn.LayerNorm(c_s)
            self.ln_kv = nn.LayerNorm(c_s)
            self.att = nn.MultiheadAttention(c_s, n_heads, batch_first=True)
        self.register_buffer('threshold_s', torch.tensor(-1.0))
        self.register_buffer('threshold_r', torch.tensor(-1.0))
        self.register_buffer('tokens_since_fired', torch.zeros(n_latents, dtype=torch.long))

    def hparams(self):
        return dict(d_in=self.d_in, n_latents=self.n_latents, k=self.k, k_s=self.k_s, estimator=self.estimator,
                    n_heads=self.n_heads, max_dist=self.max_dist, grid=list(self.grid), mu_s=self.mu_s,
                    mu_r=self.mu_r, reanim_coef=self.reanim_coef, threshold_lr=self.threshold_lr, seed=self.seed)

    # ------------------------------------------------------------------ encoder
    def smooth_pre(self, x, nbr):
        """Pre-activation of the smooth stream (n, c_s). x normalised (n, d); nbr (n, K) from neighbor_index."""
        P = (x - self.b_dec) @ self.E_s + self.b_s                    # projection of every token, (n, c_s)
        if self.estimator == 'static':
            return P
        present = (nbr >= 0)
        Pj = P[nbr.clamp_min(0)]                                      # (n, K, c_s)
        cnt = present.sum(1, keepdim=True)
        q = (Pj * present[..., None]).sum(1) / cnt.clamp_min(1)       # mean of PRESENT neighbours (0 if none)
        kv = torch.where(present[..., None], Pj, self.missing.expand_as(Pj)) + self.pos_emb
        kvn = self.ln_kv(kv)
        a, _ = self.att(self.ln_q(q)[:, None, :], kvn, kvn, need_weights=False)
        return q + a[:, 0]

    def innov_pre(self, x, S):
        resid = x - self.b_dec - (S @ self.W_dec[:self.c_s]).detach()
        return resid @ self.E_r + self.b_r

    @torch.no_grad()
    def encode(self, x, nbr):
        """Threshold inference -> (S (n, c_s), R (n, c_r)), both >= 0."""
        if self.threshold_s < 0 or self.threshold_r < 0:
            raise RuntimeError('inference thresholds not estimated (untrained SAE?)')
        s = torch.relu(self.smooth_pre(x, nbr))
        S = s * (s > self.threshold_s)
        r = torch.relu(self.innov_pre(x, S))
        return S, r * (r > self.threshold_r)

    def decode(self, S, R):
        return S @ self.W_dec[:self.c_s] + R @ self.W_dec[self.c_s:] + self.b_dec

    # ------------------------------------------------------------------ training
    def forward_train(self, x, nbr, frame, n_frames):
        """x (n, d) normalised tokens of n_frames whole frames; nbr (n, K); frame (n,) ids 0..n_frames-1."""
        n = x.shape[0]
        s_pre = self.smooth_pre(x, nbr)
        S, kept_s = _batch_topk(s_pre, self.k_s)
        r_pre = self.innov_pre(x, S)
        R, kept_r = _batch_topk(r_pre, self.k_r)
        x_hat = self.decode(S, R)

        p_f = torch.zeros(n_frames, device=x.device).index_add_(0, frame, torch.ones(n, device=x.device))
        err = (x - x_hat).pow(2).sum(1)                                # (n,)
        rec = (torch.zeros(n_frames, device=x.device).index_add_(0, frame, err) / (p_f * self.d_in)).mean()
        if self.mu_s > 0 or self.mu_r > 0:
            tr_s = _per_frame_mean(_edge_sq(S, nbr), frame, n_frames)
            tr_r = _per_frame_mean(self.kappa * R.pow(2).sum(1) - _edge_sq(R, nbr), frame, n_frames)
            spec = self.mu_s * tr_s / self.c_s + self.mu_r * tr_r / self.c_r
        else:
            tr_s = tr_r = spec = torch.zeros((), device=x.device)
        loss = rec + spec

        # reanimation: latents silent in this batch get their pre-activations pushed up (paper / overcomplete)
        reanim = torch.zeros((), device=x.device)
        if self.reanim_coef > 0:
            for pre, Z, w in ((s_pre, S, self.c_s), (r_pre, R, self.c_r)):
                dead_b = ~(Z > 0).any(0)
                if dead_b.any():
                    reanim = reanim - pre[:, dead_b].sum() / (n * w)
            loss = loss + self.reanim_coef * reanim

        with torch.no_grad():
            fired = torch.cat([(S > 0).any(0), (R > 0).any(0)])
            self.tokens_since_fired += n
            self.tokens_since_fired[fired] = 0
            _update_threshold(self.threshold_s, kept_s, self.threshold_lr)
            _update_threshold(self.threshold_r, kept_r, self.threshold_lr)
            var = (x - x.mean(0)).pow(2).sum(1).mean()
        logs = {'loss': float(loss), 'rec': float(rec), 'spec': float(spec), 'tr_s': float(tr_s), 'tr_r': float(tr_r),
                'reanim': float(reanim), 'fve': float(1 - err.mean() / var),
                'l0_s': float((S > 0).sum(1).float().mean()), 'l0_r': float((R > 0).sum(1).float().mean()),
                'thr_s': float(self.threshold_s), 'thr_r': float(self.threshold_r)}
        return loss, logs

    @torch.no_grad()
    def normalize_decoder(self):
        self.W_dec.data /= self.W_dec.data.norm(dim=1, keepdim=True).clamp_min(1e-8)

    @torch.no_grad()
    def remove_parallel_grad(self):
        if self.W_dec.grad is not None:
            W = self.W_dec.data
            self.W_dec.grad -= (self.W_dec.grad * W).sum(1, keepdim=True) * W


class PlainBatchTopKSAE(nn.Module):
    """Single-stream tokenwise BatchTopK SAE with the Spatial-SAE's loss normalisation and reanimation (the paper's
    baseline 'B'); the same encode / forward_train interface (nbr / frame ignored). Used by the synthetic test; the
    real-data baseline is src/eci/sae.py MatryoshkaBatchTopKSAE."""

    def __init__(self, d_in, n_latents, k, reanim_coef=5e-3, threshold_lr=0.01, seed=0):
        super().__init__()
        self.d_in, self.n_latents, self.k, self.reanim_coef, self.threshold_lr = d_in, n_latents, k, reanim_coef, threshold_lr
        g = torch.Generator().manual_seed(seed)
        W = torch.randn(n_latents, d_in, generator=g)
        W = W / W.norm(dim=1, keepdim=True)
        self.W_dec = nn.Parameter(W)
        self.W_enc = nn.Parameter(W.t().clone())
        self.b_enc = nn.Parameter(torch.zeros(n_latents))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.register_buffer('threshold', torch.tensor(-1.0))

    def pre(self, x):
        return (x - self.b_dec) @ self.W_enc + self.b_enc

    @torch.no_grad()
    def encode(self, x, nbr=None):
        z = torch.relu(self.pre(x))
        return z * (z > self.threshold)

    def forward_train(self, x, nbr, frame, n_frames):
        n = x.shape[0]
        pre = self.pre(x)
        Z, kept = _batch_topk(pre, self.k)
        x_hat = Z @ self.W_dec + self.b_dec
        p_f = torch.zeros(n_frames, device=x.device).index_add_(0, frame, torch.ones(n, device=x.device))
        err = (x - x_hat).pow(2).sum(1)
        rec = (torch.zeros(n_frames, device=x.device).index_add_(0, frame, err) / (p_f * self.d_in)).mean()
        loss = rec
        dead_b = ~(Z > 0).any(0)
        if self.reanim_coef > 0 and dead_b.any():
            loss = loss - self.reanim_coef * pre[:, dead_b].sum() / (n * self.n_latents)
        with torch.no_grad():
            _update_threshold(self.threshold, kept, self.threshold_lr)
            var = (x - x.mean(0)).pow(2).sum(1).mean()
        return loss, {'loss': float(loss), 'rec': float(rec), 'fve': float(1 - err.mean() / var),
                      'l0': float((Z > 0).sum(1).float().mean())}

    normalize_decoder = SpatialBatchTopKSAE.normalize_decoder
    remove_parallel_grad = SpatialBatchTopKSAE.remove_parallel_grad


def lr_paper(step, total, base_lr=1e-3, warmup=100, final_lr=1e-5):
    """Paper schedule: linear warmup from 0 over `warmup` steps, cosine decay to final_lr."""
    if step < warmup:
        return base_lr * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return final_lr + (base_lr - final_lr) * 0.5 * (1 + math.cos(math.pi * t))


def save_spatial(path, sae, norm, extra=None):
    torch.save({'state_dict': sae.state_dict(), 'norm': norm.state_dict(), 'hparams': sae.hparams(),
                'kind': 'spatial_btk', **(extra or {})}, path)


def load_spatial(path, device='cpu'):
    from src.eci.sae import TokenNorm
    ck = torch.load(path, map_location='cpu', weights_only=False)
    sae = SpatialBatchTopKSAE(**ck['hparams'])
    sae.load_state_dict(ck['state_dict'])
    norm = TokenNorm(ck['hparams']['d_in'])
    norm.load_state_dict(ck['norm'])
    return sae.to(device).eval(), norm.to(device), ck
