"""
T2: one-rule tight animal mask for the ECI pipeline (every domain), replacing the three-part foreground rule of
src/eci/foreground.py (token cue + dark cue + 1-patch dilation).

RULE (one rule, no dilation, no domain-specific cue):
    keep patch p of frame f  iff  frac(f, p) >= X,
    frac(f, p) = fraction of the 16 x 16 pixels of p whose grey level differs from the video's per-pixel background
                 image by more than delta grey levels in absolute value (darker OR lighter).

Per-pixel background image (pixel_background_blocks): the per-pixel MEDIAN grey level of the video's n_bg = 200
background sample frames (the same frames, rows and time blocks as the token background bg_seg of
scripts/eci/fg_background.py), computed separately in each of the n_seg = 4 time blocks (seg_bounds), over the frames
where the pixel's patch is NOT near the existing dark cue (dark fraction > 0.15, dilated by 1 patch, exactly the
exclusion leak_mask uses), so animals do not leak in. Fallbacks per pixel: fewer than MIN_SEG_FRAMES = 10 clean frames
in the block -> the video-level masked median (>= MIN_BG_FRAMES = 20 clean frames) -> the existing per-pixel 0.98
quantile pix_bg. A frame uses the block nearest in time (foreground.segment_of).
Why a median and not pix_bg: pix_bg is a HIGH quantile (the brightest bedding, built for a darker-only cue); an
absolute-difference rule needs the centre of the empty-arena distribution, or empty floor reads as 'darker'.

Noise level (label-free): sigma = 1.4826 x median |grey - background| over the clean sample pixels of a video (a
robust standard deviation of the empty arena around its own background), floored at 1 grey level when choosing
the parameters (quantisation step).

Functions:
    patch_pixels / pixel_excl   patch <-> pixel helpers; per-pixel exclusion from the per-patch dark fraction
    pixel_background_blocks     (n_seg, 512, 512) uint8 background + source code per pixel + clean counts
    robust_sigma                video noise level
    diff_fraction               (B, 1024) fraction of pixels with |grey - bg| > delta, for one or several deltas
    tight_mask                  (B, 1024) bool kept patches
"""
import numpy as np

GRID, PX, FRAME_PX = 32, 16, 512
MIN_SEG_FRAMES, MIN_BG_FRAMES = 10, 20   # as src/eci/foreground.py
DARK_FRAC = 0.15                         # the existing dark-cue core (exclusion only)
# fixed by scripts/eci/tight_mask.py choose (label-free, train videos only; results/vision/eci_t2t4/rule.json):
# sigma mice 2.97, ants 0 (-> floor 1) grey levels; delta >= 6 x 2.97 = 18; the first delta with an X <= 0.25 holding the
# far (empty-floor) keep rate <= 0.5% in both domains is 30 (mice far rate 0.49%, ants 0.001% at X = 0.25; dark-cue core
# patches kept 99.5% / 99.98%)
TIGHT_RULE = {'delta': 30, 'frac': 0.25}


def patch_to_pixels(m):
    """(..., 1024) -> (..., 512, 512) by repeating every patch value over its 16 x 16 block."""
    m = np.asarray(m)
    sh = m.shape[:-1]
    m = m.reshape(sh + (GRID, 1, GRID, 1))
    return np.broadcast_to(m, sh + (GRID, PX, GRID, PX)).reshape(sh + (FRAME_PX, FRAME_PX))


def dilate_np(mask, r=1):
    """(B, 1024) bool -> dilated by a (2r+1)^2 square on the 32 x 32 grid (numpy, as foreground.dilate)."""
    from scipy import ndimage as ndi
    m = np.asarray(mask, bool).reshape(-1, GRID, GRID)
    st = np.zeros((1, 2 * r + 1, 2 * r + 1), bool)
    st[0] = True
    return ndi.binary_dilation(m, structure=st).reshape(len(m), -1)


def pixel_excl(dark):
    """(N, 1024) dark fraction of the background sample frames -> (N, 1024) bool excluded patches (dark core dilated
    by one patch, as foreground.leak_mask)."""
    return dilate_np(np.asarray(dark, np.float32) > DARK_FRAC, 1)


def pixel_background_blocks(grey, dark, seg_bounds, pix_bg):
    """grey (N, 512, 512) uint8 sample frames, dark (N, 1024), seg_bounds (n_seg + 1,), pix_bg (512, 512) uint8 ->
    bg (n_seg, 512, 512) uint8, src (n_seg, 512, 512) uint8 (0 block median, 1 video median, 2 pix_bg),
    n_clean (n_seg, 1024) int16 clean frames per patch and block, excl (N, 1024) bool."""
    excl = pixel_excl(dark)
    g = grey.astype(np.float32)
    gm = np.where(patch_to_pixels(excl), np.nan, g)
    n_video = (~excl).sum(0)
    with np.errstate(all='ignore'):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            video = np.nanmedian(gm, axis=0)
    video_ok = patch_to_pixels(n_video >= MIN_BG_FRAMES)
    n_seg = len(seg_bounds) - 1
    bg = np.zeros((n_seg, FRAME_PX, FRAME_PX), np.uint8)
    src = np.zeros((n_seg, FRAME_PX, FRAME_PX), np.uint8)
    n_clean = np.zeros((n_seg, GRID * GRID), np.int16)
    for s, (a, b) in enumerate(zip(seg_bounds[:-1], seg_bounds[1:])):
        n_s = (~excl[a:b]).sum(0)
        n_clean[s] = n_s
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            blk = np.nanmedian(gm[a:b], axis=0)
        ok = patch_to_pixels(n_s >= MIN_SEG_FRAMES)
        out = np.where(ok, blk, np.where(video_ok, video, pix_bg.astype(np.float32)))
        src[s] = np.where(ok, 0, np.where(video_ok, 1, 2))
        bg[s] = np.clip(np.round(out), 0, 255).astype(np.uint8)
    return bg, src, n_clean, excl


def robust_sigma(grey, bg, excl, seg_bounds, max_px=2_000_000, seed=0):
    """1.4826 x median |grey - block background| over clean sample pixels (a random subset of max_px)."""
    dev = []
    for s, (a, b) in enumerate(zip(seg_bounds[:-1], seg_bounds[1:])):
        d = np.abs(grey[a:b].astype(np.int16) - bg[s][None].astype(np.int16))
        dev.append(d[~patch_to_pixels(excl[a:b])])
    dev = np.concatenate(dev)
    if len(dev) > max_px:
        dev = np.random.default_rng(seed).choice(dev, max_px, replace=False)
    return float(1.4826 * np.median(dev)), dev


def diff_fraction(grey, bg, deltas):
    """grey (B, 512, 512) uint8, bg (B, 512, 512) uint8 (each frame's own block background), deltas: int or list ->
    (B, 1024) float32 (int) or (len(deltas), B, 1024) float32: fraction of each patch's pixels with |grey - bg| > d."""
    d = np.abs(grey.astype(np.int16) - bg.astype(np.int16))
    one = np.isscalar(deltas)
    out = np.stack([(d > t).reshape(-1, GRID, PX, GRID, PX).mean((2, 4), dtype=np.float32).reshape(len(d), -1)
                    for t in np.atleast_1d(deltas)])
    return out[0] if one else out


def tight_mask(grey, bg, rule=None):
    """(B, 1024) bool kept patches of the tight rule."""
    rule = rule or TIGHT_RULE
    if rule['delta'] is None or rule['frac'] is None:
        raise RuntimeError('TIGHT_RULE parameters not fixed yet (scripts/eci/tight_mask.py choose)')
    return diff_fraction(grey, bg, rule['delta']) >= rule['frac']
