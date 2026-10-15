"""
Animal mask rule "U": ONE label-free rule for every domain (no per-domain switches, no tracker points).
Pre-registration, parameters and the test: scripts/benchmark/eci_mask_u.sh (header).

Inputs per video (label-free): n sample frames spread evenly over the video (grey, any H x W, a `valid` pixel mask for
padded canvases) with their mask-encoder tokens (DINOv2-base at 448 = a 32 x 32 grid; cell = side / 32 px), and the
target frames (grey + tokens). Everything below is per video; every parameter is a fraction / quantile (U_RULE).

    blur      grey frames are box-blurred with a square of side k = 2 * floor(cell / 4) + 1 px (about half a patch), so
              floor texture (bedding chips) averages out while animals (several patches long) keep their contrast
    M         per-pixel temporal MEDIAN of the blurred sample frames
    floor     Otsu split of M over the valid pixels; the floor class = the Otsu side holding most of the CENTRAL box
              (middle 50% of the valid rows and columns); opened with a cell x cell square; C = its largest connected
              component; F = convex hull of C (arenas / dishes are convex: the hull also closes notches cut by animals
              resting at the edge). f0 = median of M over C (the floor's typical value); sig_s = 1.4826 x MAD of M over C
              (the floor's spatial texture spread)
    thr_pix   still_k x 1.4826 x max(MAD of (blurred sample frame - M) over F and all sample frames, 1 grey level)
    B         empty-floor reference image, per TIME BLOCK (the sample frames split into n_blocks consecutive groups; a
              frame uses the block nearest in time): B_b = the per-pixel median M_b of the block, except on "still"
              pixels, set to f0: pixels of F with |M_b - f0| > thr_still = max(thr_pix, still_k x sig_s), in connected
              components of area <= still_area_max x |C| that are COMPACT (area <= still_compact x r_max^2, r_max =
              largest inscribed radius) - an animal that never leaves during the block (dark mice on a bright floor, a
              still beetle); thin structures (dish rims, wall shadows) and large ones keep B = M_b
    pixel cue fraction of a patch's pixels inside F with |blurred grey - B| > thr_pix (EITHER sign: dark and pale animals)
    token ref per-position masked median of the sample tokens over the frames whose patch is not within 1 patch of a
              pixel-cue core patch (pixel fraction > pix_frac); a position with fewer than tok_min_frac x n such frames
              (an animal never leaves it) is filled from its clean 8-neighbours ring by ring (foreground.fill_leaks); if
              no position is clean, the plain median
    token cue dist = 1 - cos(token, token ref); thr_tok = the tok_q quantile of dist over the sample patches farther than
              tok_far patches from any pixel-cue core patch; feat = drop_isolated(dist > thr_tok)
    mask      dilate_1(feat OR pixel fraction > pix_frac) onto ALL 8-neighbours, restricted to cells holding a valid pixel

Functions:
    U_RULE              the parameters
    blur_kernel         k from the cell size
    UMasker             per-video state: fit(sample grey, sample tokens, valid) then mask(grey, tokens) -> (B, 1024) bool;
                        parts() for sheets; info for logs
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

from src.eci.foreground import GRID, dilate, drop_isolated, fill_leaks

U_RULE = {'grid': GRID, 'blur': 'box, side 2 * floor(cell / 4) + 1 px', 'floor': 'Otsu class holding most of the central '
          'box, largest component after a cell x cell opening, convex hull', 'n_blocks': 4, 'still_area_max': 0.10, 'still_compact': 25.0, 'still_k': 6.0,
          'pix_frac': 0.15, 'tok_min_frac': 0.25, 'tok_q': 0.995, 'tok_far': 2, 'drop_isolated': True, 'dilate': 1,
          'mask_encoder': 'DINOv2-base 448 (32 x 32), last layer'}


def blur_kernel(cell):
    return 2 * int(math.floor(cell / 4)) + 1


def box_blur(g, k):
    """g (n, H, W) float -> (n, H, W) box mean over k x k (border: mean over the in-image part)."""
    if k <= 1:
        return g
    return F.avg_pool2d(g[:, None], k, stride=1, padding=k // 2, count_include_pad=False)[:, 0]


def cell_pool(x, grid=GRID):
    """(n, H, W) float -> (n, grid*grid) mean over each cell (H, W divisible by grid)."""
    n, H, W = x.shape
    assert H % grid == 0 and W % grid == 0, (H, W)
    return F.avg_pool2d(x[:, None], (H // grid, W // grid)).flatten(1)


def fit_floor(M, valid, k, still_area_max, open_px=None):
    """M (H, W) float32 numpy median image, valid (H, W) bool -> dict floor masks / values (numpy)."""
    from scipy import ndimage as ndi
    from skimage.filters import threshold_otsu
    from skimage.morphology import convex_hull_image
    v = M[valid]
    t_o = float(threshold_otsu(v))
    rr, cc = np.nonzero(valid)
    r0, r1 = np.quantile(rr, [0.25, 0.75]).astype(int)
    c0, c1 = np.quantile(cc, [0.25, 0.75]).astype(int)
    centre = M[r0:r1 + 1, c0:c1 + 1][valid[r0:r1 + 1, c0:c1 + 1]]
    bright = bool((centre > t_o).mean() >= 0.5)
    mode = float(np.median(centre))
    cls = (M > t_o) if bright else (M <= t_o)
    cls &= valid
    o = open_px or k
    opened = ndi.binary_opening(cls, np.ones((o, o), bool)) if o > 1 else cls
    lab, n = ndi.label(opened)
    if n == 0:
        return {'ok': False, 'F': valid.copy(), 'C': cls, 'cls': cls, 'f0': float(np.median(v)), 'sig_s': 0.0,
                't_otsu': t_o, 'mode': mode, 'bright': bright}
    sizes = ndi.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    C = lab == (1 + int(np.argmax(sizes)))
    Fh = convex_hull_image(C) & valid
    f0 = float(np.median(M[C]))
    sig_s = float(1.4826 * np.median(np.abs(M[C] - f0)))
    return {'ok': True, 'F': Fh, 'C': C, 'cls': cls, 'f0': f0, 'sig_s': sig_s, 't_otsu': t_o, 'mode': mode,
            'bright': bright, 'floor_px': int(C.sum()), 'hull_px': int(Fh.sum())}


def still_pixels(M, Fh, f0, thr, area_max, compact):
    """pixels inside the floor hull Fh whose median M differs from the floor value f0 by > thr, in connected components
    of area <= area_max that are COMPACT (area <= compact x r_max^2, r_max = the largest inscribed radius from the distance
    transform): animals that never leave during the median's frames; thin structures (dish rims, wall shadows) and large
    ones (fixed objects) keep B = M."""
    from scipy import ndimage as ndi
    cand = Fh & (np.abs(M - f0) > thr)
    hl, nh = ndi.label(cand)
    if not nh:
        return np.zeros_like(cand)
    idx = np.arange(1, nh + 1)
    hs = ndi.sum(np.ones_like(hl), hl, index=idx)
    r = np.asarray(ndi.maximum(ndi.distance_transform_edt(cand), hl, index=idx))
    ok = (hs <= area_max) & (hs <= compact * np.maximum(r, 0.5) ** 2)
    return np.isin(hl, idx[ok])


class UMasker:
    """Rule U for one video at a time. grey: (n, H, W) uint8 / float torch on dev; tokens (n, 1024, d) on dev.
    t_s / t: video frame indices of the sample / target frames (time blocks of the pixel reference); None = one block."""

    def __init__(self, dev, rule=U_RULE):
        self.dev, self.rule = dev, rule

    @torch.no_grad()
    def fit(self, grey_s, tok_s, valid=None, t_s=None):
        R = self.rule
        n, H, W = grey_s.shape
        self.cell = min(H, W) / R['grid']
        self.k = blur_kernel(self.cell)
        valid = torch.ones((H, W), dtype=torch.bool, device=self.dev) if valid is None else valid.to(self.dev)
        self.valid = valid
        gb = box_blur(grey_s.float().to(self.dev), self.k)
        M = gb.median(0).values
        fl = fit_floor(M.cpu().numpy(), valid.cpu().numpy(), self.k, R['still_area_max'], int(round(self.cell)))
        self.fl = fl
        Ft = torch.from_numpy(fl['F']).to(self.dev)
        self.F = Ft
        # pixel threshold: robust temporal noise of the blurred frames around the per-pixel median, over the floor hull
        r = (gb - M[None])[:, Ft].flatten()
        if r.numel() > 4_000_000:  # quantiles of a random subsample (torch quantile size limit)
            r = r[torch.randperm(r.numel(), device=self.dev, generator=torch.Generator(self.dev).manual_seed(0))[:4_000_000]]
        mad = float((r - r.median()).abs().median())
        self.thr_pix = float(R['still_k'] * 1.4826 * max(mad, 1.0))
        thr_still = max(self.thr_pix, R['still_k'] * fl['sig_s'])
        # time blocks: n_blocks consecutive groups of the sample frames; block median, still pixels -> f0
        t_s = np.arange(n) if t_s is None else np.asarray(t_s)
        order = np.argsort(t_s, kind='stable')
        nb = max(1, min(R['n_blocks'], n // 4))
        groups = np.array_split(order, nb)
        self.t_cut = np.array([(t_s[groups[b][-1]] + t_s[groups[b + 1][0]]) / 2 for b in range(nb - 1)])
        Bs, still_px = [], 0
        area_max = R['still_area_max'] * fl.get('floor_px', 0)
        for g in groups:
            Mb = gb[torch.from_numpy(g).to(self.dev)].median(0).values
            if fl['ok']:
                st = still_pixels(Mb.cpu().numpy(), fl['F'], fl['f0'], thr_still, area_max, R['still_compact'])
                Mb[torch.from_numpy(st).to(self.dev)] = fl['f0']
                still_px += int(st.sum())
            Bs.append(Mb)
        self.B = torch.stack(Bs)
        blk = np.empty(n, np.int64)
        for b, g in enumerate(groups):
            blk[g] = b
        self.valid_cells = cell_pool(valid.float()[None])[0] > 0
        pf = self.pixel_frac(gb, blocks=blk, blurred=True)
        core = pf > R['pix_frac']
        excl = dilate(core, 1)
        t = tok_s.float().to(self.dev)
        plain = t.median(0).values
        n_ok = (~excl).sum(0)
        masked = t.masked_fill(excl[..., None], float('nan')).nanmedian(0).values
        leak = n_ok < math.ceil(R['tok_min_frac'] * n)
        masked[leak] = plain[leak]
        if leak.any() and not leak.all():
            masked = fill_leaks(masked[None], leak[None])[0]
        self.tok_ref = masked
        d = 1 - F.cosine_similarity(t, masked[None], dim=-1)
        d[:, ~self.valid_cells] = 0
        far = ~dilate(core, R['tok_far'])
        far &= self.valid_cells[None]
        v = d[far]
        if v.numel() > 4_000_000:
            v = v[torch.randperm(v.numel(), device=self.dev)[:4_000_000]]
        self.thr_tok = float(torch.quantile(v, R['tok_q'])) if v.numel() else float('inf')
        self.info = {'thr_pix': self.thr_pix, 'pix_mad': mad, 'thr_tok': self.thr_tok, 'k_blur': self.k,
                     'cell_px': self.cell, 'n_blocks': nb, 'tok_leak_positions': int(leak.sum()), 'f0': fl['f0'],
                     'floor_ok': fl['ok'], 'floor_bright': fl.get('bright'), 'floor_px': fl.get('floor_px', 0), 'sig_s': fl['sig_s'], 'thr_still': thr_still,
                     'hull_px': fl.get('hull_px', 0), 'still_px_blocks': still_px, 't_otsu': fl['t_otsu']}
        return self

    def block_of(self, t):
        if t is None or len(self.t_cut) == 0:
            return None
        return np.searchsorted(self.t_cut, np.asarray(t), side='right')

    @torch.no_grad()
    def pixel_frac(self, grey, t=None, blocks=None, blurred=False):
        g = grey.float().to(self.dev)
        gb = g if blurred else box_blur(g, self.k)
        blk = self.block_of(t) if blocks is None else blocks
        B = self.B[0][None] if blk is None else self.B[torch.from_numpy(np.asarray(blk)).to(self.dev)]
        cue = ((gb - B).abs() > self.thr_pix) & self.F[None]
        return cell_pool(cue.float(), self.rule['grid'])

    @torch.no_grad()
    def parts(self, grey, tok, t=None):
        """-> (token cue, pixel core, final mask), each (b, 1024) bool."""
        R = self.rule
        pf = self.pixel_frac(grey, t)
        d = 1 - F.cosine_similarity(tok.float().to(self.dev), self.tok_ref[None], dim=-1)
        feat = d > self.thr_tok
        if R['drop_isolated']:
            feat = drop_isolated(feat)
        pc = pf > R['pix_frac']
        m = dilate(feat | pc, R['dilate']) & self.valid_cells[None]
        return feat & self.valid_cells[None], pc, m

    def mask(self, grey, tok, t=None):
        return self.parts(grey, tok, t)[2]
