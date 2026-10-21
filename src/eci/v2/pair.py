"""
Pair-centred crops (ECI v2 CANDIDATE P; not part of the frozen v2 pipeline). Label-free and tracking-free: everything
here is computed from ONE frame's pixel mask, with no identity across frames and no annotation.

    pixel mask     the domain's tight-mask pixel cue (src/eci/v2/mask.py) at the NATIVE frame (canonical orientation).
    analysis grid  the pixel mask area-pooled by an integer factor f (mice 8, ants 2: ~32-38 grid px per body length),
                   a grid cell is animal iff >= 50% of its pixels are; then a 3 x 3 binary opening (drops tails, legs,
                   antennae and pixel noise thinner than ~2 cells, i.e. ~6% of a body length).
    blobs          8-connected components; fragments smaller than 0.3 x (total animal area / N) are dropped (N = the
                   number of animals in the arena by design: mice 4, ants 3; a prior of the experiment, not tracking).
    animals/blob   A1 = single-animal area = median over the frames of the decode run (10-60 frames of one video) of
                   (total animal area / N); a blob holds n = max(1, round(area / A1)) animals, so a blob of >= 1.5 A1
                   is a MERGED blob of >= 2 animals.
    split          a merged blob is split into its n animals by k-means on the (y, x) coordinates of its cells
                   (deterministic farthest-point initialisation from the cell farthest from the blob centroid, 20 Lloyd
                   iterations). WHY k-means and not the principal axis / farthest extremities: it handles any n (huddles
                   of 3-4 mice) with one rule, and for two animals in line (nose to nose, nose to tail) its boundary
                   is the perpendicular bisector of the two body centroids, i.e. it falls at the meeting point; the
                   principal axis alone gives no contact POINT, and the two farthest extremities are the two tails
                   (or tail and nose), whose midpoint again is the meeting point only for equal-sized animals.
    contacts       (a) inside a merged blob: every pair of k-means parts that share >= 2 boundary cells (8-adjacency);
                       contact point = mean of those boundary cells; (b) between blobs: every pair of blobs whose
                       closest cells are <= 0.15 body lengths apart (the tight mask leaves a gap of a few px between
                       animals that touch, and a sniff can leave a small gap); contact point = midpoint of the closest
                       cell pair, the parts = the k-means parts holding those cells. One contact per pair of parts.
                   PAIR AXIS = the direction from part i's centroid to part j's centroid (i < j in label order). The
                   axis sign is arbitrary (no identity, no head detection); the 180-degree ambiguity is SYMMETRISED:
                   every contact gives two crops, at angle theta and theta + 180 deg, both added to the frame's token
                   set, so the frame max is invariant to the arbitrary choice.
    crop           a square of side 1.5 body lengths (native px) centred on the contact point, rotated so that the pair
                   axis is horizontal (variant 'rot'), or NOT rotated (variant 'can': the canonical frame axes, angles
                   0 and 180 deg; the control that isolates the egocentric alignment). Sampled bilinearly from the
                   native frame at native spacing, then (only when the native crop is larger than the encoder side)
                   area-antialiased down to the encoder side S_c = 16 x round(...) (never upsampled beyond native
                   spacing by more than the multiple-of-16 rounding). Outside the frame: edge replication.
    crop mask      the same tight rule on the crop: token kept iff >= 25% of its 16 x 16 patch are animal pixels by the
                   pixel cue applied to the crop and the identically sampled background plane.

Coordinates: native px of the CANONICAL (rotated) frame, x = column, y = row, pixel centres at integer + 0.5 in
continuous coordinates. A crop token (row r, col c of the G_c grid) maps to the frame point
    centre + R(theta) @ ((c + 0.5) * s - side / 2, (r + 0.5) * s - side / 2),  s = side / G_c,
with R(theta) the rotation by theta (the crop's +x axis = the pair axis).
"""
import numpy as np
import torch
import torch.nn.functional as F

OPEN_ITERS = 1
FRAG = 0.3
GAP_BODY = 0.15
SIDE_BODY = 1.5


# ---------------------------------------------------------------------------------------------- analysis mask (GPU)
@torch.no_grad()
def analysis_mask(px, f):
    """px (B, H, W) bool native pixel mask -> (B, H // f, W // f) bool (>= 50% animal pixels per f x f cell)."""
    B, H, W = px.shape
    Hc, Wc = H // f, W // f
    m = F.avg_pool2d(px[:, :Hc * f, :Wc * f].float()[:, None], f)[:, 0]
    return m >= 0.5


# ---------------------------------------------------------------------------------------------- blobs and contacts (CPU)
def blobs(m):
    """m (h, w) bool analysis mask -> (lab (h, w) int32 after opening, comp sizes (n + 1,), total area)."""
    from scipy import ndimage
    st = np.ones((3, 3), bool)
    mo = ndimage.binary_opening(m, structure=st, iterations=OPEN_ITERS)
    lab, n = ndimage.label(mo, structure=st)
    sz = np.bincount(lab.ravel(), minlength=n + 1)
    sz[0] = 0
    return lab, sz, int(sz.sum())


def kmeans_cells(yx, n, iters=20):
    """yx (m, 2) float cell coordinates, n parts -> labels (m,) in 0..n-1 (deterministic farthest-point init)."""
    if n <= 1 or len(yx) <= n:
        return np.zeros(len(yx), np.int64)
    c0 = yx.mean(0)
    cent = [yx[np.argmax(((yx - c0) ** 2).sum(1))]]
    d = ((yx - cent[0]) ** 2).sum(1)
    for _ in range(1, n):
        cent.append(yx[np.argmax(d)])
        d = np.minimum(d, ((yx - cent[-1]) ** 2).sum(1))
    C = np.stack(cent).astype(np.float64)
    lab = np.zeros(len(yx), np.int64)
    for _ in range(iters):
        lab = np.argmin(((yx[:, None, :] - C[None]) ** 2).sum(2), 1)
        newC = np.stack([yx[lab == i].mean(0) if (lab == i).any() else C[i] for i in range(n)])
        if np.allclose(newC, C):
            break
        C = newC
    return lab


def contacts_of(lab, sz, A1, n_animals, body_cells):
    """One frame. lab / sz from blobs(), A1 single-animal area (cells), body length in analysis cells ->
    (contacts [dict(y, x, theta, kind, parts, n_border | gap)], frame info dict). Coordinates in analysis cells
    (continuous, cell centres at integer + 0.5)."""
    from scipy.spatial import cKDTree
    total = int(sz.sum())
    keep = [c for c in range(1, len(sz)) if sz[c] >= FRAG * total / max(n_animals, 1) and sz[c] > 0]
    info = {'n_blobs': len(keep), 'area': total, 'A1': float(A1), 'n_parts': 0, 'n_merged': 0, 'n_contacts': 0,
            'n_merge_contacts': 0, 'n_gap_contacts': 0}
    if not keep:
        return [], info
    h, w = lab.shape
    parts = []  # (blob, yx cells (m, 2) int, centroid)
    part_map = -np.ones((h, w), np.int64)
    for c in keep:
        yy, xx = np.nonzero(lab == c)
        yx = np.stack([yy, xx], 1)
        n = max(1, int(np.floor(sz[c] / A1 + 0.5))) if A1 > 0 else 1
        n = min(n, n_animals)
        if n >= 2:
            info['n_merged'] += 1
        pl = kmeans_cells(yx.astype(np.float64), n)
        for i in range(n):
            sel = yx[pl == i]
            if not len(sel):
                continue
            part_map[sel[:, 0], sel[:, 1]] = len(parts)
            parts.append((c, sel, sel.mean(0) + 0.5))
    info['n_parts'] = len(parts)
    out = {}
    # (a) inside merged blobs: parts sharing boundary cells (8-adjacency)
    P = np.pad(part_map, 1, constant_values=-1)
    core = P[1:-1, 1:-1]
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            nb = P[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
            hit = (core >= 0) & (nb >= 0) & (core != nb)
            if not hit.any():
                continue
            ys, xs = np.nonzero(hit)
            a, b = core[ys, xs], nb[ys, xs]
            for y_, x_, a_, b_ in zip(ys, xs, a, b):
                key = (min(a_, b_), max(a_, b_))
                out.setdefault(key, {'cells': set(), 'kind': 'merge'})['cells'].add((int(y_), int(x_)))
    cont = []
    for (i, j), r in out.items():
        if len(r['cells']) < 2:
            continue
        cc = np.array(sorted(r['cells']), np.float64) + 0.5
        cy, cx = cc.mean(0)
        d = parts[j][2] - parts[i][2]
        cont.append({'y': float(cy), 'x': float(cx), 'theta': float(np.arctan2(d[0], d[1])), 'kind': 'merge',
                     'parts': (int(i), int(j)), 'n_border': int(len(cc)), 'gap': 0.0})
    # (b) between blobs: closest cells <= GAP_BODY body lengths
    gap = GAP_BODY * body_cells
    trees = {}
    for c in keep:
        yy, xx = np.nonzero(lab == c)
        trees[c] = (cKDTree(np.stack([yy, xx], 1)), np.stack([yy, xx], 1))
    for ia, ca in enumerate(keep):
        for cb in keep[ia + 1:]:
            ta, pa = trees[ca]
            tb, pb = trees[cb]
            dd, ii = tb.query(pa, k=1, distance_upper_bound=gap + 1e-6)
            if not np.isfinite(dd).any():
                continue
            m = int(np.argmin(dd))
            if dd[m] > gap:
                continue
            ya, xa = pa[m]
            yb, xb = pb[ii[m]]
            i_, j_ = int(part_map[ya, xa]), int(part_map[yb, xb])
            if i_ < 0 or j_ < 0 or i_ == j_:
                continue
            i, j = min(i_, j_), max(i_, j_)
            d = parts[j][2] - parts[i][2]
            cont.append({'y': float((ya + yb) / 2 + 0.5), 'x': float((xa + xb) / 2 + 0.5),
                         'theta': float(np.arctan2(d[0], d[1])), 'kind': 'gap', 'parts': (i, j), 'n_border': 0,
                         'gap': float(dd[m])})
    # one contact per pair of parts (merge contacts win over gap contacts of the same pair)
    seen, final = set(), []
    for c in sorted(cont, key=lambda c: c['kind'] != 'merge'):
        if c['parts'] in seen:
            continue
        seen.add(c['parts'])
        final.append(c)
    info['n_contacts'] = len(final)
    info['n_merge_contacts'] = sum(c['kind'] == 'merge' for c in final)
    info['n_gap_contacts'] = sum(c['kind'] == 'gap' for c in final)
    return final, info


def run_A1(areas, n_animals):
    """A1 of a decode run: median over its frames of total animal area / N (frames with no animal ignored)."""
    a = np.asarray([x for x in areas if x > 0], np.float64)
    return float(np.median(a) / n_animals) if len(a) else 0.0


# ---------------------------------------------------------------------------------------------- crops (GPU)
def crop_side(body_native, S_cap=None):
    """Crop side in native px (1.5 body lengths) and the encoder side S_c (multiple of 16, <= native side)."""
    side = SIDE_BODY * body_native
    S_c = int(16 * np.floor(side / 16))
    if S_cap:
        S_c = min(S_c, S_cap)
    return float(side), S_c


@torch.no_grad()
def sample_crops(img, planes, centres, thetas, side, S_c):
    """img (3, H, W) float tensor (canonical native frame), planes (k, H, W) float tensor (e.g. the pixel background)
    or None; centres (n, 2) native (x, y) continuous px; thetas (n,) radians (crop +x axis direction in frame coords,
    y down) -> crops (n, 3, S_c, S_c) uint8, planes (n, k, S_c, S_c) float or None."""
    dev = img.device
    n = len(centres)
    H, W = img.shape[1:]
    ns = max(S_c, int(round(side)))
    u = (torch.arange(ns, device=dev, dtype=torch.float32) + 0.5) * (side / ns) - side / 2
    vv, uu = torch.meshgrid(u, u, indexing='ij')  # vv = crop row offset (down), uu = column offset (right)
    c = torch.as_tensor(np.asarray(centres, np.float32), device=dev)
    t = torch.as_tensor(np.asarray(thetas, np.float32), device=dev)
    ct, st = torch.cos(t)[:, None, None], torch.sin(t)[:, None, None]
    X = c[:, 0, None, None] + ct * uu[None] - st * vv[None]
    Y = c[:, 1, None, None] + st * uu[None] + ct * vv[None]
    # pixel CENTRES: continuous coordinate p (px index + 0.5) -> normalised 2 p / W - 1 (align_corners=False)
    grid = torch.stack([2 * X / W - 1, 2 * Y / H - 1], -1)
    src = img[None] if planes is None else torch.cat([img, planes], 0)[None]
    Cn = src.shape[1]
    out = []
    for a in range(0, n, 32):
        g = grid[a:a + 32]
        m = len(g)
        # one source image, the crops' grids stacked vertically (no per-crop copy of the frame)
        o = F.grid_sample(src, g.reshape(1, m * ns, ns, 2), mode='bilinear', padding_mode='border',
                          align_corners=False)
        o = o.view(Cn, m, ns, ns).permute(1, 0, 2, 3)
        if ns != S_c:
            o = F.interpolate(o, size=(S_c, S_c), mode='bilinear', antialias=True, align_corners=False)
        out.append(o)
    o = torch.cat(out) if out else torch.zeros((0, Cn, S_c, S_c), device=dev)
    crops = o[:, :3].round().clamp(0, 255).to(torch.uint8)
    return crops, (o[:, 3:] if planes is not None else None)


def crop_token_to_frame(pos, G_c, centre, theta, side):
    """Crop token grid positions (m,) -> (m, 2) frame (x, y) native px of the token centres."""
    r, c = np.divmod(np.asarray(pos, np.int64), G_c)
    s = side / G_c
    u = (c + 0.5) * s - side / 2
    v = (r + 0.5) * s - side / 2
    ct, st = np.cos(theta), np.sin(theta)
    return np.stack([centre[0] + ct * u - st * v, centre[1] + st * u + ct * v], 1)
