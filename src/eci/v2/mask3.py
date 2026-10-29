"""
Animal mask v3 CANDIDATE (= N2 of src/eci/v2/mask2.py + a ghost-free background). NOT part of the frozen v2.0: mask2.py
and the v2.0 encode (scripts/eci/v2/v20.py N2Series) are unchanged, so v2.0 stays reproducible.

One algorithm for every domain; the only per-domain parameter is the existing body half-width h (calibration) plus
the frozen core cue's own config thresholds (dark_abs / dark_rel, unchanged).

Why: an animal that rests for more than half of the background frames is IN the per-video median background
('ghost'). N2 then (i) guards the ghost region (no candidate / strong pixels there: mask2 'bad') and (ii) the
per-pixel background change level (MAD) there is inflated by the wrong median, so the K_MAD threshold reaches ~185
grey levels. Together they remove the tails / bodies of animals that later pass through the old huddle spot
(diagnosis: scripts/eci/v2/mask3_diag.py, results/vision/eci_v2_0B/mask_approval/diag/).

Rule (fix_pack, once per video / background window, on the stored mask2 pack in its own coordinates):
    ghost  = frozen dark rule TRUE ON THE MEDIAN background: grey(bg) < dark_abs AND q98 - grey(bg) > dark_rel
    region = ghost dilated by a disk of radius round(GHOST_DIL_H * h)  (about one body width)
    bg     : inside region, replaced (FILL, see below)
    mad    : inside region, inpainted from the surrounding ring (cv2 Telea, radius INPAINT_R)
    Empty region (no ghost) -> the pack is returned unchanged (bit-identical N2 inputs).
Per frame (keep3): mask2.hyst_pixels with the fixed pack, the same parameters as N2 (SIGMA, T_LO, T_HI, K_NOISE, K_MAD,
REACH_H, strong core), NO ghost guard (bad = None; the fixed background is the empty arena there), then
mask2.appendage and the r25t token rule (>= 25% animal pixels OR >= MIN_THIN_PX appendage pixels). Isolated
background stays out through the unchanged connected-to-a-body growth (REACH_H * h).
"""
import numpy as np

from src.eci.v2 import mask2 as M2

GHOST_DIL_H, INPAINT_R = 2.0, 5
FILL = 'q98'  # 'q98': the per-pixel 0.98 grey quantile (the arena when the dark animal was away >= 2% of the time);
# 'telea': cv2 Telea inpainting of the median from the ring around the region


def grey_np(bg):
    """(3, H, W) -> (H, W) float64 grey (PIL 'L' weights, as mask2.grey_f)."""
    t = np.asarray(bg, np.float64)
    return (t[0] * 19595 + t[1] * 38470 + t[2] * 7471 + 32768) / 65536


def disk(r):
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y <= r * r).astype(np.uint8)


def fix_pack(bg, q98, mad, h, dark_abs=60, dark_rel=40, fill=None):
    """bg (3, H, W) uint8 / float median, q98 (H, W), mad (H, W) -> dict bg (3, H, W) float32, mad (H, W) float32,
    ghost (H, W) bool, region (H, W) bool. numpy in / out."""
    import cv2
    fill = fill or FILL
    bg = np.asarray(bg, np.float32)
    q98 = np.asarray(q98, np.float32)
    mad = np.asarray(mad, np.float32)
    g = grey_np(bg)
    ghost = (g < dark_abs) & ((q98 - g) > dark_rel)
    if not ghost.any():
        return {'bg': bg, 'mad': mad, 'ghost': ghost, 'region': ghost.copy()}
    reg = cv2.dilate(ghost.astype(np.uint8), disk(max(1, int(round(GHOST_DIL_H * h))))) > 0
    m8 = reg.astype(np.uint8)
    bg2 = bg.copy()
    if fill == 'q98':
        bg2[:, reg] = q98[reg][None]
    elif fill == 'q98gap':  # only where the median is clearly darker than the upper envelope (an animal was there)
        reg = reg & ((q98 - g) > dark_rel / 2)
        m8 = reg.astype(np.uint8)
        bg2[:, reg] = q98[reg][None]
    elif fill == 'telea':
        hwc = np.ascontiguousarray(bg.transpose(1, 2, 0).round().clip(0, 255).astype(np.uint8))
        bg2 = cv2.inpaint(hwc, m8, INPAINT_R, cv2.INPAINT_TELEA).transpose(2, 0, 1).astype(np.float32)
    else:
        raise ValueError(fill)
    mad2 = cv2.inpaint(np.ascontiguousarray(mad), m8, INPAINT_R, cv2.INPAINT_TELEA)
    return {'bg': bg2, 'mad': mad2.astype(np.float32), 'ghost': ghost, 'region': reg}


def appendage_r(pix, h, reach_h):
    """mask2.appendage with the reach as a parameter (identical to it at reach_h = mask2.REACH_H)."""
    import torch.nn.functional as F
    if reach_h == M2.REACH_H:
        return M2.appendage(pix, h)
    ro, rt = max(1, int(round(h / 2))), max(1, int(round(h / 4)))
    body = M2._dil_r(M2._ero_r(pix, ro), ro)
    near = M2._maxpool(body.float()[:, None], int(round(reach_h * h)))[:, 0] > 0.5
    return body, pix & ~M2._dil_r(body, rt) & near


def keep3(x, bg, mad, core0, h, roi=None, k_mad=None, reach_h=None, t_lo=None):
    """x (B, 3, S, S) uint8, bg (3, S, S) or (B, 3, S, S) float tensor (fixed), mad (S, S) or (B, S, S) tensor, core0 (B, S, S)
    frozen-cue pixels -> (keep (B, G*G) bool, pix (B, S, S) bool, app (B, S, S) bool)."""
    import torch.nn.functional as F
    k_mad = M2.K_MAD if k_mad is None else k_mad
    reach_h = M2.REACH_H if reach_h is None else reach_h
    t_lo = M2.T_LO if t_lo is None else t_lo
    r = M2.hyst_pixels(x, bg, core0, h, t_lo=t_lo, t_hi=M2.T_HI, k_noise=M2.K_NOISE, reach_h=reach_h,
                       sigma=M2.SIGMA, strong=True, mad=mad, k_mad=k_mad, bad=None, roi=roi)
    _, app = appendage_r(r['pix'], h, reach_h)
    frn = F.avg_pool2d(r['pix'].float()[:, None], 16).flatten(1)
    ng = F.avg_pool2d(app.float()[:, None], 16).flatten(1) * 256
    return (frn >= 0.25) | (ng >= M2.MIN_THIN_PX), r['pix'], app


def keep3b(x, bg_fix, bg_med, core0, h, roi=None):
    """POST-HOC AMENDMENT 1 (2026-10-29, after the pre-registered mask3 metrics showed wall / halo blocks next to old
    huddle spots on mice): a pixel is a change candidate only if it differs from BOTH the fixed background and the
    stored median: d = min(d_fixed, d_median) (identical to N2's d outside the fixed region, where both are equal). The
    rest is mask2.hyst_pixels' arithmetic with no ghost guard, written out because hyst_pixels takes one background:
    per-frame noise threshold thr = max(T_LO, K_NOISE * 1.4826 * MAD of d far (> 2h) from core pixels) and the
    term mask2 actually applies, max(thr, K_MAD * that same scalar MAD) (mask2's per-pixel MAD argument is overwritten
    by the per-frame scalar inside its loop, so the per-pixel background MAD never takes part; at batch 1 this
    function equals hyst_pixels exactly when bg_fix == bg_med); strong core = opening of d > T_HI by a square of side
    2 round(h/2) + 1; growth through candidates <= REACH_H * h steps; then mask2.appendage and the r25t token rule."""
    import torch
    import torch.nn.functional as F
    xf = M2.gauss_blur(x.float(), M2.SIGMA)
    b1 = M2.gauss_blur((bg_fix if bg_fix.dim() == 4 else bg_fix[None]).float(), M2.SIGMA)
    b2 = M2.gauss_blur((bg_med if bg_med.dim() == 4 else bg_med[None]).float(), M2.SIGMA)
    d = torch.minimum((xf - b1).abs().amax(1), (xf - b2).abs().amax(1))
    B = x.shape[0]
    far = M2._maxpool(core0.float()[:, None], max(1, int(round(2 * h))))[:, 0] < 0.5
    tl = torch.empty(B, device=x.device)
    for i in range(B):
        v = d[i][far[i]][::7] if far[i].any() else d[i].flatten()[::7]
        med = v.median()
        s = (v - med).abs().median()
        tl[i] = torch.maximum(torch.clamp(M2.K_NOISE * 1.4826 * s, min=M2.T_LO), M2.K_MAD * s)
    cand = d > tl[:, None, None]
    core = core0 | (M2._open((d > M2.T_HI).float()[:, None], max(1, int(round(h / 2))))[:, 0] > 0.5)
    if roi is not None:
        cand &= roi
        core &= roi
    g = core.float()[:, None]
    cf = (cand | core).float()[:, None]
    for it in range(int(round(M2.REACH_H * h))):
        g2 = M2._maxpool(g, 1) * cf
        if it % 16 == 15 and torch.equal(g2, g):
            break
        g = g2
    pix = g[:, 0] > 0.5
    _, app = M2.appendage(pix, h)
    frn = F.avg_pool2d(pix.float()[:, None], 16).flatten(1)
    ng = F.avg_pool2d(app.float()[:, None], 16).flatten(1) * 256
    return (frn >= 0.25) | (ng >= M2.MIN_THIN_PX), pix, app
