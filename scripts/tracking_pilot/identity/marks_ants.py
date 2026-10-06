"""Ant colour-mark scorer: classes focal (no mark), blue, yellow (+ 'unreadable').

Per isolated ant: square crop of CROP px around the track centre; foreground = pixels that differ
from the video background by > FG_THR (max over BGR channels), the component nearest the crop
centre, dilated by 2 px (the dot sits on the gaster, at the edge of the dark body). Within it:
    blue_px   largest 8-connected blob with H in [95, 135], S >= 60, V >= 60
    yellow_px largest blob with (H <= 15 or H >= 172), S >= 90, V >= 90   (orange dot; the ant's
              brown cuticle is also red-hued but dark, hence the V floor)
    fg_area   foreground area (px)
    core_area leg-free body area: foreground opened with a radius-4 disk, component nearest the
              centre. Two touching ants that the detector reports as ONE box have core_area
              >= 1.39 x the single-ant reference on the labelled crops (singles <= 1.21), so a read
              with core_area > CORE_MAX x reference is 'too big' (two animals) and unreadable.
HSV ranges are the v3 tracking config ones (configs/tracking/ants/v3.yaml), tightened on V for
yellow because the source video is not background-subtracted here. Copied, not imported: the
existing tracker (src/tracking/) is not modified.
Classifier: multinomial logistic regression on [log1p(blue_px), log1p(yellow_px), fg_area/1000]
trained on hand-labelled crops (labels assigned with temporal context, so marked ants whose dot is
hidden ARE in the training set: P(focal | no colour) is calibrated against hidden dots).
'unreadable' = not isolated (decided upstream), or fg_area outside [0.5, 1.6] x median, or max
probability < READ_P.
"""
import cv2
import numpy as np

CROP = 110
FG_THR = 40
READ_P = 0.6
CORE_MAX = 1.3
IDS = ['focal', 'blue', 'yellow']


def _largest(mask):
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return int(st[1:, 4].max()) if n > 1 else 0


def crop(img, cx, cy, side=CROP):
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    H, W = img.shape[:2]
    pad = side
    big = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    return big[y0 + pad:y0 + pad + side, x0 + pad:x0 + pad + side]


def features(frame_bgr, bg_bgr, cx, cy):
    c = crop(frame_bgr, cx, cy)
    b = crop(bg_bgr, cx, cy)
    fg = (cv2.absdiff(c, b).max(axis=2) > FG_THR).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(fg)
    if n <= 1:
        return np.array([0, 0, 0, 0], np.float32), c
    d = [np.hypot(cen[i, 0] - CROP / 2, cen[i, 1] - CROP / 2) if st[i, 4] > 50 else 1e9
         for i in range(1, n)]
    k = 1 + int(np.argmin(d))
    body = cv2.dilate((lab == k).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    blue = body & (H >= 95) & (H <= 135) & (S >= 60) & (V >= 60)
    yel = body & ((H <= 15) | (H >= 172)) & (S >= 90) & (V >= 90)
    core = cv2.morphologyEx(fg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    nc, _, stc, cenc = cv2.connectedComponentsWithStats(core)
    core_area = 0
    if nc > 1:
        kc = 1 + int(np.argmin([np.hypot(cenc[i, 0] - CROP / 2, cenc[i, 1] - CROP / 2) for i in range(1, nc)]))
        core_area = stc[kc, 4]
    return np.array([_largest(blue), _largest(yel), st[k, 4], core_area], np.float32), c


def design(F):
    F = np.asarray(F, np.float32)[:, :3]
    return np.c_[np.log1p(F[:, 0]), np.log1p(F[:, 1]), F[:, 2] / 1000.0]
