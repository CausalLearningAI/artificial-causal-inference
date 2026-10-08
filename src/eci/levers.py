"""
Helpers for the SAE-lever experiment (scripts/eci/sae_levers.py): what an unsupervised SAE prioritises on the ECI
foreground token stores, and how single latents are scored against behaviour labels.

    align_gpu(X, Y, rows)         per-column AUROC / AP / top-1% precision / firing rate of non-negative fp16 codes
                                  X (n, m) on the GPU, against bool labels Y (n, L), exact with ties averaged. Same
                                  definitions as spatial_sae_pilot.align_columns (AUROC = Mann-Whitney with ties
                                  averaged, AP = sklearn step definition over distinct thresholds, the zero group being
                                  the last threshold), computed from per-column histograms over the fp16 bit patterns
                                  (non-negative fp16 values sort like their int16 bit patterns). Top-1% precision:
                                  the k = round(0.01 n) highest frames (k = number of non-zero frames if smaller);
                                  frames tied at the cut-off are counted pro rata (the expected precision under a random
                                  tie-break; align_columns breaks such ties by frame order instead).
    frame_components(pos, lens)   8-connected foreground components on the 32 x 32 patch grid of every frame ->
                                  (n components, largest component area) per frame, plus every component's area
    CellMeans                     per-(video, grid position) and per-video mean tokens, accumulated on the GPU, and the
                                  centring x - mean (cells with < min_count tokens fall back to the per-video mean)
"""
import numpy as np
import torch
from scipy import ndimage

GRID = 32
NB = 0x7C01  # non-negative finite fp16 bit patterns: 0 (= 0.0) ... 0x7C00 (= inf)


@torch.no_grad()
def align_gpu(X, Y, rows=None, top_frac=0.01, block=512, cols=None):
    """X: (n, m) fp16 tensor (GPU), all values finite and >= 0. Y: (n, L) bool tensor (same device). rows: optional
    LongTensor of rows to score on (a subset of frames). cols: optional LongTensor of columns. Returns dict of numpy
    arrays auc / ap / prec_top (m', L), rate (m',), base (L,), n."""
    dev = X.device
    if rows is not None:
        Y = Y[rows]
    n, L = Y.shape
    cols = torch.arange(X.shape[1], device=dev) if cols is None else cols
    m = len(cols)
    P = Y.sum(0).double()
    N = n - P
    k_top = max(1, int(round(top_frac * n)))
    auc = torch.full((m, L), float('nan'), dtype=torch.float64, device=dev)
    ap, ptop = auc.clone(), auc.clone()
    rate = torch.zeros(m, dtype=torch.float64, device=dev)
    for j0 in range(0, m, block):
        c = cols[j0:j0 + block]
        b = len(c)
        xb = X.index_select(1, c)
        if rows is not None:
            xb = xb.index_select(0, rows)
        if (xb < 0).any() or not torch.isfinite(xb).all():
            raise ValueError('codes must be finite and >= 0')
        bits = (xb.contiguous().view(torch.int16).long() & 0x7FFF) + NB * torch.arange(b, device=dev)[None]  # -0 -> 0
        del xb
        tot = torch.bincount(bits.flatten(), minlength=b * NB).view(b, NB).double()
        rate[j0:j0 + b] = 1 - tot[:, 0] / n
        tot_d = tot.flip(1)
        ccnt = tot_d.cumsum(1)
        nnz = n - tot[:, 0]
        k = torch.minimum(torch.full_like(nnz, k_top), nnz)
        ib = torch.searchsorted(ccnt.contiguous(), k[:, None]).clamp_max(NB - 1)  # first bin with cum count >= k
        for li in range(L):
            if P[li] == 0 or N[li] == 0:
                continue
            pos = torch.bincount(bits[Y[:, li]].flatten(), minlength=b * NB).view(b, NB).double().flip(1)
            neg = tot_d - pos
            ctp, cfp = pos.cumsum(1), neg.cumsum(1)
            auc[j0:j0 + b, li] = (pos * (N[li] - cfp) + 0.5 * pos * neg).sum(1) / (P[li] * N[li])
            prec = torch.where(ctp + cfp > 0, ctp / (ctp + cfp).clamp_min(1), torch.zeros_like(ctp))
            ap[j0:j0 + b, li] = (pos / P[li] * prec).sum(1)
            cb = ccnt.gather(1, ib)[:, 0]
            tb = tot_d.gather(1, ib)[:, 0]
            pb = pos.gather(1, ib)[:, 0]
            tp = ctp.gather(1, ib)[:, 0] - pb + pb * (k - (cb - tb)) / tb.clamp_min(1)
            pt = tp / k.clamp_min(1)
            ptop[j0:j0 + b, li] = torch.where(k > 0, pt, torch.full_like(pt, float('nan')))
        del bits
    return {'auc': auc.cpu().numpy(), 'ap': ap.cpu().numpy(), 'prec_top': ptop.cpu().numpy(),
            'rate': rate.cpu().numpy(), 'base': (P / max(n, 1)).cpu().numpy(), 'n': n}


def frame_components(pos, lens, grid=GRID):
    """pos (N,) int grid positions (row * grid + col), lens (F,) foreground tokens per frame (contiguous) ->
    n_comp (F,), largest (F,), comp_area (all components), comp_frame (frame of each component)."""
    st = np.r_[0, np.cumsum(lens)]
    struct = np.ones((3, 3), bool)
    n_comp = np.zeros(len(lens), np.int32)
    largest = np.zeros(len(lens), np.int32)
    areas, frames = [], []
    m = np.zeros((grid, grid), bool)
    for f in range(len(lens)):
        p = pos[st[f]:st[f + 1]].astype(np.int64)
        if len(p) == 0:
            continue
        m[:] = False
        m.flat[p] = True
        lab, k = ndimage.label(m, structure=struct)
        a = np.bincount(lab.ravel(), minlength=k + 1)[1:]
        n_comp[f], largest[f] = k, a.max()
        areas.append(a)
        frames.append(np.full(k, f, np.int32))
    return n_comp, largest, np.concatenate(areas), np.concatenate(frames)


class CellMeans:
    """Mean token per (video, grid position) and per video, over every token added. Centring modes:
    'cell'  x - mean(video, position), falling back to the per-video mean where the cell has < min_count tokens
    'video' x - mean(video)"""

    def __init__(self, n_videos, dim, dev, grid=GRID):
        self.V, self.C = n_videos, grid * grid
        self.sum = torch.zeros(n_videos * self.C, dim, dtype=torch.float64, device=dev)
        self.cnt = torch.zeros(n_videos * self.C, dtype=torch.float64, device=dev)

    @torch.no_grad()
    def add(self, tok, vid, pos):
        """tok (n, d) tensor, vid (n,) long, pos (n,) long, all on the device."""
        cell = vid * self.C + pos
        self.sum.index_add_(0, cell, tok.double())
        self.cnt.index_add_(0, cell, torch.ones_like(cell, dtype=torch.float64))

    @torch.no_grad()
    def finalize(self, min_count=20):
        s = self.sum.view(self.V, self.C, -1)
        c = self.cnt.view(self.V, self.C)
        vmean = s.sum(1) / c.sum(1, keepdim=True).clamp_min(1)  # (V, d)
        cmean = s / c[..., None].clamp_min(1)
        ok = c >= min_count
        cell = torch.where(ok[..., None], cmean, vmean[:, None, :])
        self.mean = {'cell': cell.reshape(self.V * self.C, -1).float(),
                     'video': vmean.float()}
        tokens_in_ok = float(c[ok].sum() / c.sum().clamp_min(1))
        self.stats = {'n_videos': self.V, 'cells_with_tokens': int((c > 0).sum()), 'cells_ge_min': int(ok.sum()),
                      'min_count': min_count, 'frac_tokens_in_cells_ge_min': tokens_in_ok}
        del self.sum, self.cnt
        return self

    def center(self, x, vid, pos, mode):
        if mode == 'cell':
            return x.float() - self.mean['cell'][vid * self.C + pos]
        if mode == 'video':
            return x.float() - self.mean['video'][vid]
        raise ValueError(mode)
