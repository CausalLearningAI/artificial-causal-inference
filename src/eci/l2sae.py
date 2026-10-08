"""
Helpers for the second-level ("relational") SAE experiment (scripts/eci/l2sae.py): tracker-free behaviour neurons
from WHICH first-level patch neurons fire next to each other.

Level 1 = a patch-token SAE (scripts/eci/sae_levers.py checkpoints). Its sparse code of every foreground patch is kept
as the top-K (value, latent) pairs per token (SparseCodes; K = 48 by default, the truncation is measured and logged:
L0 per token is ~16 on average with threshold inference).

Level-2 token of foreground patch p (frame f, grid row r, col c on the 32 x 32 patch grid):
    [ code(p) | NEAR(p) | FAR(p) ]                                             (3 m dims, m = level-1 width)
    NEAR(p) = element-wise max of the level-1 codes over the foreground patches q of the same frame with Chebyshev
              distance 1 <= max(|dr|, |dc|) <= 2 from p
    FAR(p)  = the same over Chebyshev distance 3 .. r_far (mice 6, ants 5)
    p itself is excluded from NEAR / FAR (that is what makes the token relational); background patches count as 0
    (codes are >= 0, so they never win the max). Built per token by gathering the neighbours' sparse codes through a
    per-frame position -> token map and a scatter-amax (level2_features), never as dense grids.
Level-3 token of a frame = the mean of the level-1 codes over its foreground patches (the level-1 'mean' read-out).

Functions / classes:
    ring_offsets(r0, r1)          (n, 2) (dr, dc) with Chebyshev distance in [r0, r1]
    SparseCodes                   top-K sparse level-1 codes of a token set + its frame layout (token map)
    level2_features(sc, b, rings) (B, 3m) fp32 level-2 tokens of tokens b (global token indices)
    frame_reads(z, fr, lens)      per-frame read-outs of per-token codes z: max, mean of the top-8 patch activations
                                  (zero-padded when the frame has < 8 foreground patches), fraction of the frame's
                                  foreground patches where the latent fires, mean
    fit_block_norm(feat, idx, blocks, ...)  TokenNorm with one scale per block (as TokenNorm.fit_blocks), streamed
    level2_reference(...)         brute-force numpy level-2 tokens (selftest)
"""
import math

import numpy as np
import torch

from src.eci.sae import TokenNorm

GRID = 32


def ring_offsets(r0, r1):
    d = torch.arange(-r1, r1 + 1)
    dr, dc = torch.meshgrid(d, d, indexing='ij')
    cheb = torch.maximum(dr.abs(), dc.abs())
    keep = (cheb >= r0) & (cheb <= r1)
    return torch.stack([dr[keep], dc[keep]], 1)


class SparseCodes:
    """Top-K sparse codes of N tokens (frames contiguous, `lens` order) on `dev`:
    idx (N, K) int16 latent ids, val (N, K) fp16 values (0 = empty slot), frame (N,) int32, pos (N,) int16,
    tokmap (F, GRID*GRID) int32 = global token index at (frame, position) or -1."""

    def __init__(self, pos, lens, m, K, dev):
        n, F = int(lens.sum()), len(lens)
        if m > 32767:
            raise ValueError('latent ids are stored as int16')
        self.m, self.K, self.dev, self.n, self.F = m, K, dev, n, F
        self.idx = torch.zeros(n, K, dtype=torch.int16, device=dev)
        self.val = torch.zeros(n, K, dtype=torch.float16, device=dev)
        lens_t = torch.as_tensor(np.asarray(lens), dtype=torch.long, device=dev)
        self.frame = torch.repeat_interleave(torch.arange(F, device=dev, dtype=torch.int32), lens_t)
        self.pos = torch.as_tensor(np.asarray(pos).astype(np.int16), device=dev)
        self.tokmap = torch.full((F, GRID * GRID), -1, dtype=torch.int32, device=dev)
        self.tokmap[self.frame.long(), self.pos.long()] = torch.arange(n, device=dev, dtype=torch.int32)
        self.stats = {'n_tokens': n, 'K': K, 'tokens_over_K': 0, 'mass_total': 0.0, 'mass_dropped': 0.0}

    @torch.no_grad()
    def put(self, lo, z):
        """Store the dense codes z (n, m) (>= 0) of tokens lo .. lo + n."""
        v, i = z.topk(self.K, dim=1)
        self.idx[lo:lo + len(z)] = i.to(torch.int16)
        self.val[lo:lo + len(z)] = v.half()
        self.stats['tokens_over_K'] += int(((z > 0).sum(1) > self.K).sum())
        tot = float(z.sum())
        self.stats['mass_total'] += tot
        self.stats['mass_dropped'] += tot - float(v.sum())

    def dense(self, b):
        """Dense level-1 codes (B, m) fp32 of tokens b."""
        out = torch.zeros(len(b), self.m, device=self.dev)
        return out.scatter_(1, self.idx[b].long(), self.val[b].float())


@torch.no_grad()
def ring_max(sc, f, r, c, off):
    """Element-wise max of the codes of the foreground neighbours at offsets off (n_off, 2) of tokens at (f, r, c)."""
    B = len(f)
    rr = r[:, None] + off[None, :, 0]
    cc = c[:, None] + off[None, :, 1]
    ok = (rr >= 0) & (rr < GRID) & (cc >= 0) & (cc < GRID)
    q = rr.clamp(0, GRID - 1) * GRID + cc.clamp(0, GRID - 1)
    nb = sc.tokmap[f[:, None], q]
    valid = ok & (nb >= 0)
    nbl = nb.clamp_min(0).long()
    ii = sc.idx[nbl].long().view(B, -1)
    vv = (sc.val[nbl].float() * valid[..., None]).view(B, -1)
    out = torch.zeros(B, sc.m, device=sc.dev)
    out.scatter_reduce_(1, ii, vv, 'amax', include_self=True)
    return out, valid.sum(1)


@torch.no_grad()
def level2_features(sc, b, rings, with_counts=False):
    """Level-2 tokens [code | NEAR | FAR] (B, 3m) fp32 of global token indices b (LongTensor on sc.dev).
    rings: list of (n_off, 2) LongTensors on sc.dev (NEAR, FAR)."""
    f = sc.frame[b].long()
    p = sc.pos[b].long()
    r, c = p // GRID, p % GRID
    parts, counts = [sc.dense(b)], []
    for off in rings:
        o, k = ring_max(sc, f, r, c, off)
        parts.append(o)
        counts.append(k)
    x = torch.cat(parts, 1)
    return (x, counts) if with_counts else x


@torch.no_grad()
def frame_reads(z, fr, lens_c, top=8, want=('max', 'top8', 'frac', 'mean')):
    """z (n, m) per-token codes (>= 0) of whole frames, fr (n,) long frame index within the chunk (contiguous, sorted),
    lens_c (Fc,) long tokens per frame. -> dict of (Fc, m) fp32 read-outs."""
    Fc, m = len(lens_c), z.shape[1]
    dev = z.device
    out = {}
    den = lens_c.clamp_min(1).float()[:, None]
    if 'max' in want:
        out['max'] = torch.zeros(Fc, m, device=dev).index_reduce_(0, fr, z, 'amax', include_self=True)
    if 'frac' in want:
        out['frac'] = torch.zeros(Fc, m, device=dev).index_add_(0, fr, (z > 0).float()) / den
    if 'mean' in want:
        out['mean'] = torch.zeros(Fc, m, device=dev).index_add_(0, fr, z.float()) / den
    if 'top8' in want:
        st = torch.cumsum(lens_c, 0) - lens_c
        rank = torch.arange(len(z), device=dev) - st[fr]
        L = int(lens_c.max()) if Fc else 0
        if L == 0:
            out['top8'] = torch.zeros(Fc, m, device=dev)
        else:
            P = torch.zeros(Fc, m, L, dtype=torch.float16, device=dev)
            P[fr, :, rank] = z.half()
            out['top8'] = P.topk(min(top, L), dim=2).values.float().sum(2) / top
            del P
    return out


@torch.no_grad()
def fit_block_norm(feat_fn, idx, blocks, chunk=8192):
    """TokenNorm (per-block scale, as TokenNorm.fit_blocks) from the features feat_fn(b) of the token indices idx,
    computed in chunks (two passes: mean, then the average per-block norm about the mean)."""
    d = sum(blocks)
    s1, n = None, 0
    for a in range(0, len(idx), chunk):
        x = feat_fn(idx[a:a + chunk]).double()
        s1 = x.sum(0) if s1 is None else s1 + x.sum(0)
        n += len(x)
    mean = (s1 / n).float()
    nsum = torch.zeros(len(blocks), dtype=torch.float64, device=mean.device)
    for a in range(0, len(idx), chunk):
        x = feat_fn(idx[a:a + chunk]) - mean
        o = 0
        for k, bsz in enumerate(blocks):
            nsum[k] += x[:, o:o + bsz].norm(dim=1).double().sum()
            o += bsz
    scale = torch.empty(d, device=mean.device)
    o = 0
    for k, bsz in enumerate(blocks):
        scale[o:o + bsz] = math.sqrt(bsz) / float(nsum[k] / n)
        o += bsz
    norm = TokenNorm(d).to(mean.device)
    norm.mean.copy_(mean)
    norm.scale = scale
    return norm


def level2_reference(codes, pos, lens, r_near=(1, 2), r_far=(3, 6)):
    """Brute force: codes (N, m) dense numpy, pos (N,), lens (F,) -> (N, 3m) level-2 tokens."""
    st = np.r_[0, np.cumsum(lens)]
    N, m = codes.shape
    out = np.zeros((N, 3 * m), np.float32)
    for f in range(len(lens)):
        P = pos[st[f]:st[f + 1]].astype(int)
        R, C = P // GRID, P % GRID
        for a in range(len(P)):
            i = st[f] + a
            out[i, :m] = codes[i]
            ch = np.maximum(np.abs(R - R[a]), np.abs(C - C[a]))
            for k, (lo, hi) in enumerate((r_near, r_far)):
                sel = (ch >= lo) & (ch <= hi)
                if sel.any():
                    out[i, (k + 1) * m:(k + 2) * m] = codes[st[f] + np.flatnonzero(sel)].max(0)
    return out
