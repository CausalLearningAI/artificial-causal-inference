"""
Thin-part animal pixel cue (ECI v2 CANDIDATE, tested in scripts/eci/v2_mask2/; NOT part of the frozen v2 until that test
adopts it; the frozen tight mask src/eci/v2/mask.py is unchanged and stays the default).

Why: the frozen cue (dark pixels: grey < dark_abs AND > dark_rel below the per-pixel high-quantile background) never
sees mouse tails (lighter than bodies), misses pale / albino mice and loses ~40% of ant leg / antenna pixels. The cue
here keeps everything the frozen cue keeps (its pixels are the CORE) and adds pixels that (1) differ from the video's
EMPTY-ARENA background and (2) are connected to an animal body through such pixels (hysteresis / geodesic growth), so
thin attached parts (tails, antennae, legs) are kept and isolated noise is not.

Per video (label-free): bg = per-pixel, per-channel temporal MEDIAN of N_BG frames evenly spread over the video (or the
background window), at the encoder input side S, in canonical coordinates (median_background).
Per frame (hyst_pixels), all sizes in pixels at the encoder input side S, h = the domain's body half-width (calib):
    d      = max over RGB channels of |blur(frame) - blur(bg)| (Gaussian blur sigma = SIGMA px)
    noise  = 1.4826 * MAD of d over the frame's pixels farther than 2 h from any core pixel (robust per-frame level)
    cand   = d > max(T_LO, K_NOISE * noise)                                   (low threshold)
    strong = opening of (d > T_HI) by a (2 r + 1) square, r = round(h / 2)   (body-wide, large change: pale animals)
    core   = frozen cue pixels | strong
    grow   = pixels reachable from core through cand pixels by an 8-connected path of length <= REACH = 6 h
             (geodesic reconstruction capped at REACH steps: about 3 body widths, i.e. a mouse tail / ant antenna
             length; farther candidate pixels are not attached parts)
    pixels = core | grow
Token rules (thin_fraction / keep):
    'r25'   token kept iff >= 25% of its 16 x 16 patch are animal pixels (the frozen rule).
    'r25t'  r25 OR >= MIN_THIN_PX (16 = a 1-px line across the patch) animal pixels that are GROWN (not core): a thin
            attached part crossing the patch; isolated core specks below 25% stay out as in the frozen rule.
"""
import numpy as np
import torch
import torch.nn.functional as F

SIGMA, T_LO, T_HI, K_NOISE, REACH_H, MIN_THIN_PX = 1.0, 20.0, 40.0, 4.0, 6.0, 16


def grey_f(x):
    """(B, 3, H, W) uint8 -> (B, H, W) float grey (PIL 'L' weights, as src/eci/v2/mask.grey_of)."""
    t = x.float()
    return (t[:, 0] * 19595 + t[:, 1] * 38470 + t[:, 2] * 7471 + 32768) / 65536


def gauss_blur(x, sigma=SIGMA):
    """x (B, C, H, W) float -> separable Gaussian blur, reflect padding."""
    if sigma <= 0:
        return x
    r = max(1, int(np.ceil(3 * sigma)))
    t = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-t * t / (2 * sigma * sigma))
    k = k / k.sum()
    C = x.shape[1]
    x = F.pad(x, (r, r, 0, 0), mode='reflect')
    x = F.conv2d(x, k.view(1, 1, 1, -1).repeat(C, 1, 1, 1), groups=C)
    x = F.pad(x, (0, 0, r, r), mode='reflect')
    return F.conv2d(x, k.view(1, 1, -1, 1).repeat(C, 1, 1, 1), groups=C)


@torch.no_grad()
def median_background(frames, chunk=128):
    """frames (N, 3, H, W) uint8 tensor (any device) -> (3, H, W) float32 per-pixel per-channel median (lower median)."""
    N, C, H, W = frames.shape
    out = torch.empty((C, H, W), dtype=torch.float32, device=frames.device)
    for y0 in range(0, H, chunk):
        out[:, y0:y0 + chunk] = frames[:, :, y0:y0 + chunk].median(0).values.float()
    return out


@torch.no_grad()
def background_mad(frames, bg, sigma=SIGMA, chunk=128):
    """frames (N, 3, H, W) uint8, bg (3, H, W) float median -> (H, W) float per-pixel temporal MAD of
    max_c |blur(frame) - blur(bg)| (robust per-pixel change level of the empty arena: bedding chips that move, flicker)."""
    bb = gauss_blur(bg[None].float(), sigma)
    ds = []
    for i in range(0, len(frames), 8):
        ds.append((gauss_blur(frames[i:i + 8].float(), sigma) - bb).abs().amax(1))
    d = torch.cat(ds)
    return d.median(0).values


@torch.no_grad()
def grey_quantiles(frames, qs=(0.02, 0.98), chunk=32):
    """(N, 3, H, W) uint8 -> tuple of (H, W) float per-pixel grey quantiles over the N frames."""
    g = grey_f(frames)
    out = []
    for q in qs:
        out.append(torch.cat([torch.quantile(g[:, y0:y0 + chunk].flatten(1), q, dim=0).view(-1, g.shape[2])
                              for y0 in range(0, g.shape[1], chunk)]))
    return tuple(out)


def _maxpool(m, r):
    return F.max_pool2d(m, 2 * r + 1, 1, r)


def _open(m, r):
    """binary opening of m (B, 1, H, W) float {0, 1} by a (2r+1) square."""
    if r <= 0:
        return m
    return _maxpool(-_maxpool(-m, r), r)


@torch.no_grad()
def hyst_pixels(x, bg, core0, h, t_lo=T_LO, t_hi=T_HI, k_noise=K_NOISE, reach_h=REACH_H, sigma=SIGMA,
                strong=True, roi=None, mad=None, k_mad=0.0, q_lo=None, q_hi=None, ghost=0.0, bad=None):
    """x (B, 3, S, S) uint8, bg (3, S, S) or (B, 3, S, S) float (median background), core0 (B, S, S) bool (frozen cue
    pixels), h body half-width px -> dict pix (B, S, S) bool, grown (pixels not in core), core, d (B, S, S) float,
    thr (B,) float. roi (S, S) bool or None: pixels outside never count."""
    B = x.shape[0]
    xf = gauss_blur(x.float(), sigma)
    bf = gauss_blur((bg if bg.dim() == 4 else bg[None]).float(), sigma)
    d = (xf - bf).abs().amax(1)  # (B, S, S)
    c0 = core0.float()[:, None]
    far = _maxpool(c0, max(1, int(round(2 * h))))[:, 0] < 0.5
    thr = torch.empty(B, device=x.device)
    for i in range(B):
        v = d[i][far[i]][::7] if far[i].any() else d[i].flatten()[::7]
        med = v.median()
        mad = (v - med).abs().median()
        thr[i] = torch.clamp(k_noise * 1.4826 * mad, min=t_lo)
    tl = thr[:, None, None]
    if mad is not None and k_mad > 0:
        tl = torch.maximum(tl, k_mad * mad[None])
    cand = d > tl
    core = core0.clone()
    if strong:
        st = _open((d > t_hi).float()[:, None], max(1, int(round(h / 2))))[:, 0] > 0.5
        core |= st
    if ghost > 0 and (q_lo is not None or q_hi is not None):
        # the median background is unreliable where an animal rested for > half of the background frames: there the
        # median is much darker than the high grey quantile (dark animal) or much brighter than the low one (pale)
        gb = grey_f(bg[None] if bg.dim() == 3 else bg)[:1]
        bad = torch.zeros_like(gb, dtype=torch.bool)
        if q_hi is not None:
            bad |= (q_hi[None].float() - gb) > ghost
        if q_lo is not None:
            bad |= (gb - q_lo[None].float()) > ghost
        cand &= ~bad
        core = core0 | (core & ~bad)
    if bad is not None:  # ghost pixels: the frozen dark cue fires on the median background itself (resting animal)
        cand &= ~bad
        core = core0 | (core & ~bad)
    if roi is not None:
        cand &= roi
        core &= roi
    g = core.float()[:, None]
    cf = (cand | core).float()[:, None]
    R = int(round(reach_h * h))
    for it in range(R):
        g2 = _maxpool(g, 1) * cf
        if it % 16 == 15 and torch.equal(g2, g):
            break
        g = g2
    pix = g[:, 0] > 0.5
    return {'pix': pix, 'grown': pix & ~core, 'core': core, 'd': d, 'thr': thr}


@torch.no_grad()
def background_pack(frames, dark_abs=60, dark_rel=40):
    """background frames (N, 3, S, S) uint8 of one video / window -> dict bg (median), mad, q02, q98, ghost (frozen dark
    rule (grey < dark_abs AND q98 - grey > dark_rel) true ON THE MEDIAN BACKGROUND: an animal rested there in more than
    half of the background frames, so the median is not the empty arena), all at side S."""
    bg = median_background(frames)
    q02, q98 = grey_quantiles(frames)
    gb = grey_f(bg[None])[0]
    return {'bg': bg, 'mad': background_mad(frames, bg, sigma=SIGMA), 'q02': q02, 'q98': q98,
            'ghost': (gb < dark_abs) & ((q98 - gb) > dark_rel)}


@torch.no_grad()
def cue(x, pack, core0, h):
    """The candidate cue with its pre-registered parameters (module constants)."""
    return hyst_pixels(x, pack['bg'], core0, h, t_lo=T_LO, t_hi=T_HI, k_noise=K_NOISE, reach_h=REACH_H, sigma=SIGMA,
                       strong=True, mad=pack['mad'], k_mad=K_MAD, bad=pack['ghost'])


def token_fractions(pix, grown, patch=16):
    """-> (frac (B, G*G) animal-pixel fraction, n_grown (B, G*G) grown pixel count) per patch."""
    fr = F.avg_pool2d(pix.float()[:, None], patch).flatten(1)
    ng = F.avg_pool2d(grown.float()[:, None], patch).flatten(1) * patch * patch
    return fr, ng


def keep(frac, n_grown, rule):
    if rule == 'r25':
        return frac >= 0.25
    if rule == 'r25t':
        return (frac >= 0.25) | (n_grown >= MIN_THIN_PX)
    raise ValueError(rule)
