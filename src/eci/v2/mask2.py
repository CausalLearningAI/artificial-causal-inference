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
    'r25t'  r25 OR >= MIN_THIN_PX (16 = a 1-px line across the patch) APPENDAGE pixels (appendage(): animal pixels
            outside the opened body, within REACH of a body): a thin attached part crossing the patch (ant legs and
            antennae are often frozen-cue pixels already, so the rule counts appendage pixels, not only grown ones);
            isolated specks far from a body stay out.
Parameters (development 2026-10-22 on dev data only, explore.py configs C / Cg / Cgh / Eg / G1 / G2; G2 chosen by
eye: tails kept, no bedding speckle on mice v1, white CalMS21 intruder kept): SIGMA 1.5, T_LO 35, K_MAD 6 (cand also
needs d > 6 x the per-pixel temporal median |d| of the background frames, background_mad), T_HI 40, ghost guard =
the frozen dark rule true on the median background (resting animal) -> no candidate / strong pixels there.
AMENDMENT 2026-10-22 ~14:50 (after the GPU selftest panel of mice diag showed a ring-shaped leak around a resting
spot, before any held-out benchmark score existed; the running validation jobs were cancelled unread): the ghost
mask is dilated by (2 round(h) + 1) square before use.
"""
import numpy as np
import torch
import torch.nn.functional as F

SIGMA, T_LO, T_HI, K_NOISE, K_MAD, REACH_H, MIN_THIN_PX = 1.5, 35.0, 40.0, 4.0, 6.0, 6.0, 16


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


def _maxpool_ref(m, r):
    return F.max_pool2d(m, 2 * r + 1, 1, r)


def _maxpool(m, r):
    """(2r+1) square max filter = a (2r+1, 1) then a (1, 2r+1) max filter (max is separable over a rectangle; the
    implicit -inf padding is the same), bit-identical to _maxpool_ref and O(r) instead of O(r^2) per pixel (v2.0
    rollout 2026-10-25: the 2h / REACH filters of 129 / 385 px dominated the cue's cost at 1024 px)."""
    if r <= 0:
        return m
    return F.max_pool2d(F.max_pool2d(m, (2 * r + 1, 1), 1, (r, 0)), (1, 2 * r + 1), 1, (0, r))


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
        tl = torch.maximum(tl, k_mad * (mad if mad.dim() == 3 else mad[None]))  # (S, S) or per-frame (B, S, S)
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
    if bad is not None:  # ghost pixels: the frozen dark cue fires on the median background itself (resting animal),
        # dilated by a disk of radius round(h) so the blurred rim of the resting animal in the median is covered too
        bad = _maxpool(bad.float()[None, None], max(1, int(round(h))))[0, 0] > 0.5 if bad.dim() == 2 else \
            _maxpool(bad.float()[:, None], max(1, int(round(h))))[:, 0] > 0.5
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


def _disk(r, dev):
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return torch.from_numpy((x * x + y * y <= r * r).astype(np.float32))[None, None].to(dev)


def _dil(m, K):
    return F.conv2d(m.float()[:, None], K, padding=K.shape[-1] // 2)[:, 0] > 0.5


def _ero(m, K):
    return F.conv2d(m.float()[:, None], K, padding=K.shape[-1] // 2)[:, 0] > float(K.sum()) - 0.5


def _disk_or(m, r, pad):
    """(B, H, W) bool -> OR of m over a disk of radius r (x^2 + y^2 <= r^2) around each pixel, outside the image = pad.
    Rows of the disk are horizontal runs: OR over dy of the (1, 2 w(dy) + 1) max filter shifted by dy. Exact (binary
    maxima), O(r) passes instead of the O(r^2)-per-pixel disk convolution (v2.0 rollout 2026-10-25: the 33 px disk
    convolutions dominated the cue at 1024 px)."""
    B, H, W = m.shape
    x = F.pad(m.float()[:, None], (r, r, r, r), value=float(pad))
    out = torch.zeros((B, 1, H, W), device=m.device)
    hm = {}
    for dy in range(-r, r + 1):
        w = int(np.floor(np.sqrt(r * r - dy * dy) + 1e-9))
        if w not in hm:
            hm[w] = F.max_pool2d(x, (1, 2 * w + 1), 1, (0, w))
        out = torch.maximum(out, hm[w][:, :, r + dy:r + dy + H, r:r + W])
    return out[:, 0] > 0.5


def _dil_r(m, r):
    return _disk_or(m, r, 0)


def _ero_r(m, r):
    return ~_disk_or(~m, r, 1)


@torch.no_grad()
def appendage_ref(pix, h):
    """The original disk-convolution implementation of appendage() (reference for the identity test)."""
    Ko = _disk(max(1, int(round(h / 2))), pix.device)
    Kt = _disk(max(1, int(round(h / 4))), pix.device)
    body = _dil(_ero(pix, Ko), Ko)
    near = _maxpool(body.float()[:, None], int(round(REACH_H * h)))[:, 0] > 0.5
    return body, pix & ~_dil(body, Kt) & near


@torch.no_grad()
def appendage(pix, h):
    """pix (B, S, S) bool -> (body, app): body = opening of pix by a disk of radius round(h/2) (parts at least about
    a body width wide), app = pix outside body dilated by a disk of radius max(1, round(h/4)) AND within REACH_H * h of
    the body (thin parts attached near a body: tails, antennae, legs; isolated specks far from any body excluded)."""
    ro, rt = max(1, int(round(h / 2))), max(1, int(round(h / 4)))
    body = _dil_r(_ero_r(pix, ro), ro)
    near = _maxpool(body.float()[:, None], int(round(REACH_H * h)))[:, 0] > 0.5
    return body, pix & ~_dil_r(body, rt) & near


def token_fractions(pix, app, patch=16):
    """-> (frac (B, G*G) animal-pixel fraction, n_app (B, G*G) appendage pixel count) per patch."""
    fr = F.avg_pool2d(pix.float()[:, None], patch).flatten(1)
    na = F.avg_pool2d(app.float()[:, None], patch).flatten(1) * patch * patch
    return fr, na


def keep(frac, n_app, rule):
    if rule == 'r25':
        return frac >= 0.25
    if rule == 'r25t':
        return (frac >= 0.25) | (n_app >= MIN_THIN_PX)
    raise ValueError(rule)
