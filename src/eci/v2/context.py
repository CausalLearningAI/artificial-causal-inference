"""
Spatial context tokens (ECI v2).

A token t of a frame is replaced by x_t = [z_t, c_t]: its own encoder token z_t (1024-d) and the mean c_t of the OTHER
kept tokens of the same frame inside the w x w window centred on t (t excluded; c_t = 0 when no other kept token lies
in the window). Only kept (masked-in) tokens enter the mean, so the background never leaks into a context. w = 5 is the
round-3 'R5' context (scripts/eci/nnnt/r3.py ctx_of, identical numbers on the 64 grid); w = 3 is the 3 x 3 variant;
w = 0 means no context (x_t = z_t).

Also here: the mask geometry used by the edge readout (round-3 'R7' reference arm): chessboard distance of each kept
token to the nearest non-kept cell (1 = on the mask boundary).
"""
import numpy as np
import torch
import torch.nn.functional as F


def context_of(z, pos, fl, B, G, w):
    """z (n, d) float tokens of a chunk of B frames (kept tokens only), pos (n,) long grid positions (row-major on the
    G x G grid), fl (n,) long frame index of each token in 0..B-1, window w (odd) -> c (n, d) float: mean of the other
    kept tokens of the same frame in the w x w window around each token (0 when there is none)."""
    d = z.shape[1]
    r = w // 2
    Gr = torch.zeros((B, G * G, d), device=z.device, dtype=torch.float32)
    Gr[fl, pos] = z.float()
    Cn = torch.zeros((B, G * G), device=z.device, dtype=torch.float32)
    Cn[fl, pos] = 1
    S = F.avg_pool2d(Gr.view(B, G, G, d).permute(0, 3, 1, 2), w, 1, r, divisor_override=1)
    Sc = F.avg_pool2d(Cn.view(B, 1, G, G), w, 1, r, divisor_override=1)
    s = S.permute(0, 2, 3, 1).reshape(B, G * G, d)[fl, pos] - z.float()
    c = Sc.view(B, G * G)[fl, pos] - 1
    return s / c.clamp_min(1)[:, None]


def neighbour_count(pos, fl, B, G, w):
    """Number of OTHER kept tokens in each token's w x w window (the n_t of context_of)."""
    r = w // 2
    Cn = torch.zeros((B, G * G), device=pos.device, dtype=torch.float32)
    Cn[fl, pos] = 1
    Sc = F.avg_pool2d(Cn.view(B, 1, G, G), w, 1, r, divisor_override=1)
    return Sc.view(B, G * G)[fl, pos] - 1


def sae_input(z, pos, fl, B, G, w):
    """SAE input rows (n, d or 2d) float32: z (w = 0) or [z, context_of(..., w)]."""
    if not w:
        return z.float()
    return torch.cat([z.float(), context_of(z, pos, fl, B, G, w)], 1)


def edge_distance(tf, pos, nf, G):
    """Per kept token (numpy): chessboard distance to the nearest non-kept cell of its frame (1 = mask boundary)."""
    from scipy import ndimage
    starts = np.r_[0, np.cumsum(np.bincount(tf, minlength=nf))]
    dist = np.zeros(len(tf), np.int8)
    for f in range(nf):
        a, b = starts[f], starts[f + 1]
        if a == b:
            continue
        p = pos[a:b].astype(np.int64)
        g = np.zeros(G * G, bool)
        g[p] = True
        gp = np.pad(g.reshape(G, G), 1)
        dist[a:b] = np.minimum(ndimage.distance_transform_cdt(gp, metric='chessboard')[1:-1, 1:-1].ravel()[p], 127)
    return dist
