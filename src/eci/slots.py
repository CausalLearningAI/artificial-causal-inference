"""
Object-centric slots over frozen DINOv2 foreground patch tokens (DINOSAUR-style, Seitzer et al. 2023), for the
per-frame instance-split gate test (scripts/eci/split_test_slots.py). Unsupervised: the only training signal is
reconstructing the frame's own foreground tokens.

Input per frame: the foreground tokens of a token store (768-d DINOv2-base at 448, 32 x 32 grid) with their grid
positions; frames are padded to a common length with a validity mask.
    encoder    LayerNorm -> Linear(768, d) + Fourier 2D position embedding of the patch centre -> 2-layer MLP (residual)
    slots      K slots of dim d. 'random' init: learned mean + learned per-dim sigma times Gaussian noise (a fixed
               noise seed at inference). 'seeded' init: the first n_seeded slots start from an MLP of the Fourier
               embedding of seed points (batched k-means of the frame's foreground patch positions, k = n_seeded,
               farthest-point initialisation, label-free), the remaining slots random.
    attention  Slot Attention (Locatello et al. 2020): iters rounds, softmax over slots, weighted mean over the valid
               tokens, GRU + residual MLP update
    decoder    MLP broadcast decoder: for every slot and every token position, MLP(slot + position embedding) ->
               (768 features, 1 alpha logit); alpha = softmax over slots; reconstruction = sum_k alpha_k * feature_k
    loss       mean squared error to the (per-dimension standardised) input tokens, valid tokens only
Masks: per token, argmax_k alpha (the decoder's alpha masks, as in DINOSAUR).

Classes / functions: fourier_xy, SlotAttention, SlotModel, kmeans_seeds.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

GRID = 32


def fourier_xy(pos, n_freq=8, grid=GRID):
    """pos (B, N) int grid index (row * grid + col) or (B, N, 2) float xy in [-1, 1] -> (B, N, 4 n_freq + 2)."""
    if pos.dim() == 2:
        r, c = (pos // grid).float(), (pos % grid).float()
        xy = torch.stack([(c + 0.5) / grid * 2 - 1, (r + 0.5) / grid * 2 - 1], -1)
    else:
        xy = pos
    f = (2.0 ** torch.arange(n_freq, device=xy.device, dtype=xy.dtype)) * math.pi
    a = xy[..., None] * f  # (B, N, 2, n_freq)
    return torch.cat([xy, torch.sin(a).flatten(-2), torch.cos(a).flatten(-2)], -1)


class SlotAttention(nn.Module):
    def __init__(self, d, iters=3, hidden=512, eps=1e-8):
        super().__init__()
        self.iters, self.eps, self.scale = iters, eps, d ** -0.5
        self.norm_in, self.norm_slot, self.norm_mlp = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.q, self.k, self.v = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)
        self.gru = nn.GRUCell(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, d))

    def forward(self, x, valid, slots):
        """x (B, N, d), valid (B, N) bool, slots (B, K, d) -> slots, attn (B, K, N)."""
        B, K, d = slots.shape
        x = self.norm_in(x)
        k, v = self.k(x), self.v(x)
        for _ in range(self.iters):
            prev = slots
            q = self.q(self.norm_slot(slots))
            logits = torch.einsum('bkd,bnd->bkn', q, k) * self.scale
            attn = logits.softmax(1) * valid[:, None, :].float()  # softmax over slots
            w = attn / (attn.sum(-1, keepdim=True) + self.eps)
            upd = torch.einsum('bkn,bnd->bkd', w, v)
            slots = self.gru(upd.reshape(-1, d), prev.reshape(-1, d)).reshape(B, K, d)
            slots = slots + self.mlp(self.norm_mlp(slots))
        return slots, attn


def kmeans_seeds(pos, valid, k, iters=10, grid=GRID):
    """Batched k-means of the valid tokens' grid positions. pos (B, N) int, valid (B, N) -> (B, k, 2) xy in [-1, 1]
    (farthest-point initialisation from the token farthest from the frame's foreground centroid; frames with fewer
    than k tokens repeat points)."""
    r, c = (pos // grid).float(), (pos % grid).float()
    xy = torch.stack([(c + 0.5) / grid * 2 - 1, (r + 0.5) / grid * 2 - 1], -1)  # (B, N, 2)
    vf = valid.float()
    big = 1e9
    cen = (xy * vf[..., None]).sum(1) / vf.sum(1, keepdim=True).clamp_min(1)
    d0 = ((xy - cen[:, None]) ** 2).sum(-1).masked_fill(~valid, -1)
    seeds = [xy[torch.arange(len(xy)), d0.argmax(1)]]
    mind = ((xy - seeds[0][:, None]) ** 2).sum(-1)
    for _ in range(1, k):
        nxt = xy[torch.arange(len(xy)), mind.masked_fill(~valid, -1).argmax(1)]
        seeds.append(nxt)
        mind = torch.minimum(mind, ((xy - nxt[:, None]) ** 2).sum(-1))
    C = torch.stack(seeds, 1)  # (B, k, 2)
    for _ in range(iters):
        dist = ((xy[:, :, None] - C[:, None]) ** 2).sum(-1).masked_fill(~valid[..., None], big)  # (B, N, k)
        a = F.one_hot(dist.argmin(-1), k).float() * vf[..., None]
        cnt = a.sum(1)  # (B, k)
        newc = torch.einsum('bnk,bnd->bkd', a, xy) / cnt.clamp_min(1)[..., None]
        C = torch.where(cnt[..., None] > 0, newc, C)
    # order seeds deterministically (by y then x) so slot index has no meaning beyond position
    order = (C[..., 1] * 4 + C[..., 0]).argsort(1)
    return torch.gather(C, 1, order[..., None].expand(-1, -1, 2))


class SlotModel(nn.Module):
    def __init__(self, d_in=768, d=256, K=5, iters=3, n_seeded=0, dec_hidden=1024, n_freq=8):
        super().__init__()
        self.K, self.n_seeded, self.n_freq, self.d = K, n_seeded, n_freq, d
        pe = 4 * n_freq + 2
        self.in_norm = nn.LayerNorm(d_in)
        self.in_proj = nn.Linear(d_in, d)
        self.enc_pos = nn.Linear(pe, d)
        self.enc_mlp = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        self.mu = nn.Parameter(torch.randn(1, 1, d) * d ** -0.5)
        self.log_sigma = nn.Parameter(torch.full((1, 1, d), -1.0))
        self.seed_mlp = nn.Sequential(nn.Linear(pe, d), nn.ReLU(), nn.Linear(d, d))
        self.sa = SlotAttention(d, iters)
        self.dec_pos = nn.Linear(pe, d)
        self.dec = nn.Sequential(nn.Linear(d, dec_hidden), nn.ReLU(), nn.Linear(dec_hidden, dec_hidden), nn.ReLU(),
                                 nn.Linear(dec_hidden, dec_hidden), nn.ReLU(), nn.Linear(dec_hidden, d_in + 1))

    def init_slots(self, B, pos, valid, gen=None):
        dev = self.mu.device
        noise = torch.randn(B, self.K, self.d, device=dev, generator=gen)
        slots = self.mu + self.log_sigma.exp() * noise
        if self.n_seeded:
            seeds = kmeans_seeds(pos, valid, self.n_seeded)
            s = self.seed_mlp(fourier_xy(seeds, self.n_freq))
            slots = torch.cat([s + 0.1 * self.log_sigma.exp() * noise[:, :self.n_seeded],
                               slots[:, self.n_seeded:]], 1)
        return slots

    def forward(self, x, pos, valid, gen=None):
        """x (B, N, d_in) standardised tokens, pos (B, N) int grid index, valid (B, N) bool ->
        (recon (B, N, d_in), alpha (B, K, N), slots (B, K, d))."""
        pe = fourier_xy(pos, self.n_freq)
        h = self.in_proj(self.in_norm(x)) + self.enc_pos(pe)
        h = h + self.enc_mlp(h)
        slots, _ = self.sa(h, valid, self.init_slots(len(x), pos, valid, gen))
        out = self.dec(slots[:, :, None, :] + self.dec_pos(pe)[:, None])  # (B, K, N, d_in + 1)
        feat, logit = out[..., :-1], out[..., -1]
        alpha = logit.float().softmax(1)
        recon = (alpha[..., None] * feat.float()).sum(1)
        return recon, alpha, slots

    @staticmethod
    def loss(recon, x, valid):
        v = valid.float()
        return (((recon - x.float()) ** 2).mean(-1) * v).sum() / v.sum().clamp_min(1)
