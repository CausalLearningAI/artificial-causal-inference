"""Mouse shave-mark scorer: classes m1_none, m2_head, m3_middle, m4_back (+ 'unreadable').

Crop: square of CROP_SRC source pixels around the track centre, rotated so the body's principal
axis is horizontal (the head may point left or right: 180 deg ambiguous), resampled to 224 px.
The axis is measured from the dark-fur mask inside the crop (tracker heading is not trusted).

Why orientation does not need to be solved: head vs back shave differ only in WHERE along the
body the pale patch sits relative to the head. Both feature sets below are made invariant to the
180 deg flip (and the left/right mirror) by construction, so the classifier must learn the
relation 'patch next to the ears/snout' vs 'patch next to the rump', which is visible inside the
crop whichever way the mouse faces; it can never be confused by a wrong head/tail guess.
  hand  : body mask + 'pale' pixels (grey well above the body's fur level) inside it; profiles
          along the axis (8 bins) of the pale fraction on the MIDLINE band and on the LATERAL
          bands (ears are lateral, shaves are dorsal-midline), plus body width profile. Each
          profile is folded into an orientation-free form by ordering it from the end with more
          lateral-pale mass (the ears) -> 'ear end first'; the fold rule is reported, not learned.
  dino  : DINOv2-base CLS + mean patch token, averaged over the 4 views {crop, 180 deg, mirror,
          mirror+180 deg} (exactly invariant to the head/tail flip and the mirror).
Classifier: multinomial logistic regression (sklearn), standardised features.
Readability: a read is 'unreadable' when the row is not isolated (decided upstream) OR the crop
body mask is implausible (area outside [0.5, 1.6] x the median single-mouse area, or touching
the crop border on > 15 % of the border) OR the classifier's max probability < READ_P.
"""
import cv2
import numpy as np
import torch

CROP_SRC = 560   # source px square (mouse ~ 400 x 150 px at 2064 px)
OUT = 224
READ_P = 0.6
IDS = ['m1_none', 'm2_head', 'm3_middle', 'm4_back']


def disk(r):
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y <= r * r).astype(np.uint8)


def crop_rotated(gray, cx, cy, angle, side=CROP_SRC, out=OUT):
    """Rotate about (cx, cy) by `angle` (rad) so that direction becomes +x, crop side x side,
    resample to out x out. Out-of-frame = edge replicate."""
    s = out / side
    a = np.degrees(angle)
    M = cv2.getRotationMatrix2D((cx, cy), a, s)
    M[0, 2] += out / 2 - cx
    M[1, 2] += out / 2 - cy
    return cv2.warpAffine(gray, M, (out, out), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def body_mask(crop, bg_crop):
    """Dark-fur body mask on a crop (same rule as standin.mice_mask, radii scaled to 224/560)."""
    gi, bi = crop.astype(np.int16), bg_crop.astype(np.int16)
    core = ((gi < 70) & (bi - gi > 50)).astype(np.uint8)
    core = cv2.morphologyEx(core, cv2.MORPH_OPEN, disk(1))
    near = cv2.dilate(core, disk(10))
    cand = ((gi < 160) & (bi - gi > 50) & (near > 0)).astype(np.uint8)
    m = cv2.morphologyEx(core | cand, cv2.MORPH_CLOSE, disk(5))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, disk(4))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return np.zeros_like(m, bool)
    # keep the component closest to the crop centre (the tracked animal)
    c = OUT / 2
    d = [np.hypot(cent[i, 0] - c, cent[i, 1] - c) if stats[i, 4] > 200 else 1e9 for i in range(1, n)]
    k = 1 + int(np.argmin(d))
    from scipy import ndimage as ndi
    return ndi.binary_fill_holes(lab == k)


def principal_angle(mask):
    yy, xx = np.nonzero(mask)
    if len(xx) < 10:
        return 0.0
    cov = np.cov(np.stack([xx - xx.mean(), yy - yy.mean()]))
    w, v = np.linalg.eigh(cov)
    return float(np.arctan2(v[1, 1], v[0, 1]))


def aligned_crop(gray, bg_full_scaled, bg_scale, cx, cy):
    """-> (crop 224 grey aligned to the body axis, mask, bg crop, angle). Two passes: first an
    unrotated crop to measure the axis, then the rotated crop."""
    bgu = cv2.resize(bg_full_scaled, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR) \
        if bg_full_scaled.shape != gray.shape else bg_full_scaled
    c0 = crop_rotated(gray, cx, cy, 0.0)
    b0 = crop_rotated(bgu, cx, cy, 0.0)
    m0 = body_mask(c0, b0)
    if m0.sum() < 200:
        return c0, m0, b0, 0.0
    yy, xx = np.nonzero(m0)
    # recentre on the body, then rotate
    sc = CROP_SRC / OUT
    cx2 = cx + (xx.mean() - OUT / 2) * sc
    cy2 = cy + (yy.mean() - OUT / 2) * sc
    ang = principal_angle(m0)
    c1 = crop_rotated(gray, cx2, cy2, ang)
    b1 = crop_rotated(bgu, cx2, cy2, ang)
    return c1, body_mask(c1, b1), b1, ang


def mask_ok(mask, med_area):
    a = mask.sum()
    if med_area and not (0.5 * med_area <= a <= 1.6 * med_area):
        return False
    border = np.r_[mask[0], mask[-1], mask[:, 0], mask[:, -1]]
    return border.mean() < 0.15


# ----------------------------------------------------------------------------- hand features
NB = 8


def hand_features(crop, mask):
    """Orientation-free profiles (see module doc). Returns float vector."""
    if mask.sum() < 200:
        return np.zeros(4 * NB + 3, np.float32)
    yy, xx = np.nonzero(mask)
    fur = np.median(crop[mask])
    pale = mask & (crop.astype(np.int16) > fur + 35)
    x0, x1 = np.percentile(xx, 2), np.percentile(xx, 98)
    yc = np.median(yy)
    half_w = np.percentile(np.abs(yy - yc), 90) + 1
    xb = np.clip(((np.arange(OUT) - x0) / max(x1 - x0, 1) * NB).astype(int), -1, NB)
    Y = np.arange(OUT)[:, None] - yc
    mid = (np.abs(Y) < 0.45 * half_w)
    lat = ~mid
    prof = {}
    for name, band in [('mid', mid), ('lat', lat)]:
        num = np.zeros(NB)
        den = np.zeros(NB)
        for b in range(NB):
            cols = (xb == b)[None, :]
            sel = mask & band & cols
            den[b] = sel.sum()
            num[b] = (pale & sel).sum()
        prof[name] = num / np.maximum(den, 1)
    width = np.array([mask[:, xb == b].sum() / max((xb == b).sum(), 1) for b in range(NB)])
    width = width / max(width.max(), 1)
    # fold: put the end with more lateral pale (ears) first
    lat_p = prof['lat']
    flip = lat_p[:NB // 2].sum() < lat_p[NB // 2:].sum()
    f = (lambda v: v[::-1]) if flip else (lambda v: v)
    tot_pale = pale.sum() / mask.sum()
    return np.r_[f(prof['mid']), f(prof['lat']), f(width), f(prof['mid']) - f(prof['mid'])[::-1],
                 tot_pale, (x1 - x0) / OUT, 2 * half_w / OUT].astype(np.float32)


# ----------------------------------------------------------------------------- DINOv2 features
_DINO = {}


def dino_model(device='cuda'):
    if 'm' not in _DINO:
        from transformers import AutoModel
        _DINO['m'] = AutoModel.from_pretrained('facebook/dinov2-base').to(device).eval()
        _DINO['dev'] = device
    return _DINO['m']


@torch.no_grad()
def dino_features(crops, device='cuda', bs=64):
    """crops: (n, 224, 224) uint8 grey. -> (n, 1536) float32, mean over the 4 flip views of
    [CLS, mean patch token]."""
    m = dino_model(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    out = []
    for i in range(0, len(crops), bs):
        x = torch.from_numpy(np.ascontiguousarray(crops[i:i + bs])).to(device).float() / 255
        x = x[:, None].expand(-1, 3, -1, -1)
        x = (x - mean) / std
        acc = 0
        for v in [x, x.flip(-1).flip(-2), x.flip(-2), x.flip(-1)]:
            h = m(pixel_values=v).last_hidden_state
            acc = acc + torch.cat([h[:, 0], h[:, 1:].mean(1)], 1)
        out.append((acc / 4).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 1536), np.float32)
