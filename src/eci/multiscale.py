"""
Tracker-free multi-scale window tokens for the ECI foreground token stores (scripts/eci/multiscale_sae.py).

A frame of a foreground store (scripts/eci/fg_extract_train.py) is a set of DINOv2 patch tokens on a 32 x 32 grid,
background patches dropped. Instead of tracking animals, every frame is covered by square windows at several spatial
scales matched to the animal size, and each window becomes one token:

    window token = [ mean of the foreground tokens in the window (768)
                   | element-wise max of the same tokens (768)
                   | foreground fraction of the window, centroid row, centroid col of its foreground patches (3) ]
    (centroid in grid units / 32, i.e. in [0, 1]; dims 1536-1538 = EXTRA)

Scales (side in patches, stride, minimum foreground fraction). Measured 2026-10-07 on 1500 random frames of one shard
per store (frames whose foreground mask splits into exactly one 8-connected component per animal):
    mice  ~40 foreground patches per mouse (162 per frame / 4), component bbox 8 x 6 patches
    ants  ~14-21 foreground patches per ant (43 per frame / 3), component bbox 6 x 5 patches
    scale       mice           ants          meaning
    S1          1x1  stride 1  1x1 stride 1  one patch (the token-wise baseline)
    S_part      4x4  stride 2  3x3 stride 2  ~half an animal
    S_animal    7x7  stride 3  5x5 stride 3  ~one animal
    S_pair      11x11 stride 5 8x8 stride 4  ~two animals side by side
    S_frame     the whole grid (one window = all foreground patches, no CLS token)
Window starts: 0, stride, 2*stride, ... plus a last window flush with the grid edge. A window is kept when it holds
at least max(1, ceil(min_frac * side^2)) foreground patches (min_frac 0.2 for part / animal / pair; any foreground for
S1 and S_frame).

Functions:
    scale_table(domain)                  -> {name: (side, stride, min_frac)}
    window_starts(side, stride)          -> list of window start indices along one grid axis
    window_tokens(tok, pos, lens, side, stride, min_frac)
                                         -> feats (W, 1539) fp32, frame (W,), win (W, 3) = (row0, col0, side)
"""
import math

import torch
import torch.nn.functional as F

GRID = 32
D_TOKEN = 768
EXTRA = 3
D_WINDOW = 2 * D_TOKEN + EXTRA
SCALE_NAMES = ('S1', 'S_part', 'S_animal', 'S_pair', 'S_frame')
SCALES = {
    'mice': {'S1': (1, 1, 0.0), 'S_part': (4, 2, 0.2), 'S_animal': (7, 3, 0.2), 'S_pair': (11, 5, 0.2),
             'S_frame': (GRID, GRID, 0.0)},
    'ants': {'S1': (1, 1, 0.0), 'S_part': (3, 2, 0.2), 'S_animal': (5, 3, 0.2), 'S_pair': (8, 4, 0.2),
             'S_frame': (GRID, GRID, 0.0)},
}


def scale_table(domain):
    return SCALES[domain]


def window_starts(side, stride, grid=GRID):
    s = list(range(0, grid - side + 1, stride))
    if s[-1] != grid - side:
        s.append(grid - side)
    return s


def min_count(side, min_frac):
    return max(1, math.ceil(min_frac * side * side - 1e-9))


@torch.no_grad()
def window_tokens(tok, pos, lens, side, stride, min_frac, grid=GRID, fast=True):
    """tok (N, d) tensor on the compute device (frames contiguous, in `lens` order), pos (N,) long grid position
    (row-major), lens (F,) long tokens per frame. -> feats (W, 2d+3) fp32, frame (W,) long, win (W, 3) long
    (row0, col0, side), sorted by frame then window row then window col (fast=False: the generic pooling path also
    for side 1, for the self-test)."""
    dev = tok.device
    n_fr, d = len(lens), tok.shape[1]
    frame = torch.repeat_interleave(torch.arange(n_fr, device=dev), lens)
    rr = (pos // grid).float()
    cc = (pos % grid).float()
    if side == 1 and fast:  # every foreground patch is a window (fast path; identical to the generic path)
        x = tok.float()
        one = torch.ones(len(tok), 1, device=dev)
        feats = torch.cat([x, x, one, ((rr + 0.5) / grid)[:, None], ((cc + 0.5) / grid)[:, None]], 1)
        win = torch.stack([pos // grid, pos % grid, torch.ones_like(pos)], 1)
        return feats, frame, win
    X = torch.zeros(n_fr, grid * grid, d, device=dev, dtype=torch.float32)
    X[frame, pos] = tok.float()
    M = torch.zeros(n_fr, grid * grid, device=dev, dtype=torch.float32)
    M[frame, pos] = 1.0
    X = X.view(n_fr, grid, grid, d).permute(0, 3, 1, 2)  # (F, d, g, g)
    M = M.view(n_fr, 1, grid, grid)
    st = torch.tensor(window_starts(side, stride, grid), device=dev)
    k2 = float(side * side)

    def pool_sum(t):
        return (F.avg_pool2d(t, side, stride=1) * k2)[:, :, st][:, :, :, st]

    cnt = pool_sum(M)[:, 0].round()  # (F, ns, ns)
    keep = cnt >= min_count(side, min_frac)
    f_i, a_i, b_i = keep.nonzero(as_tuple=True)
    S = pool_sum(X)  # (F, d, ns, ns)
    mean = S[f_i, :, a_i, b_i] / cnt[f_i, a_i, b_i][:, None]
    del S
    Xm = X.masked_fill(M == 0, float('-inf'))
    del X
    mx = F.max_pool2d(Xm, side, stride=1)[:, :, st][:, :, :, st][f_i, :, a_i, b_i]
    del Xm
    yy = torch.arange(grid, device=dev, dtype=torch.float32)
    R = M * ((yy + 0.5) / grid)[None, None, :, None]
    C = M * ((yy + 0.5) / grid)[None, None, None, :]
    c = cnt[f_i, a_i, b_i]
    cy = pool_sum(R)[:, 0][f_i, a_i, b_i] / c
    cx = pool_sum(C)[:, 0][f_i, a_i, b_i] / c
    feats = torch.cat([mean, mx, (c / k2)[:, None], cy[:, None], cx[:, None]], 1)
    win = torch.stack([st[a_i], st[b_i], torch.full_like(a_i, side)], 1)
    return feats, f_i, win


@torch.no_grad()
def window_counts(pos, lens, side, stride, min_frac, grid=GRID):
    """Kept windows per frame (F,) long, from positions only (cheap)."""
    dev = pos.device
    n_fr = len(lens)
    frame = torch.repeat_interleave(torch.arange(n_fr, device=dev), lens)
    if side == 1:
        return lens.clone()
    M = torch.zeros(n_fr, grid * grid, device=dev, dtype=torch.float32)
    M[frame, pos] = 1.0
    st = torch.tensor(window_starts(side, stride, grid), device=dev)
    cnt = (F.avg_pool2d(M.view(n_fr, 1, grid, grid), side, stride=1) * side * side)[:, 0][:, st][:, :, st].round()
    return (cnt >= min_count(side, min_frac)).flatten(1).sum(1).long()
