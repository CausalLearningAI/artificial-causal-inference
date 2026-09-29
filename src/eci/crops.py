"""
Mouse-centric crops for the ECI pipeline on mice v1 (prototype), WITHOUT tracking or pose.
No behaviour annotation is used anywhere here.

Per frame (512 x 512 grey, 4 black mice on light bedding):
    dark mask   grey < dark_abs AND pix_bg - grey > dark_rel (the dark cue of
                src/eci/foreground.py, pix_bg = the video's per-pixel 0.98 quantile background)
    clean       binary opening with a disk of radius open_r (removes speckle and cuts thin
                tails, so two mice linked only by a tail stay separate), fill holes
    blobs       8-connected components with area >= min_area pixels
    per blob    area, centroid, principal-axis angle, major-axis length (4 sqrt(lambda_1))
Blob classes (area thresholds estimated label-free, see calibrate_areas):
    single      single_lo <= area <= single_hi: looks like one mouse
    merged      area > single_hi: two or more mice in contact (contact / huddle crop)
    fragment    min_area <= area < single_lo: a partly occluded / partly visible mouse;
                no crop of its own, but used for pair distances
Crops (one fixed-size square per unit, resampled to 224 x 224 for DINOv2):
    single  (type 0) side SINGLE_PX, centred on the blob centroid, rotated so that the blob's
            principal axis is horizontal (head / tail sign is NOT resolved: 180 deg ambiguity)
    pair    (type 1) for every pair of blobs whose nearest-pixel distance < pair_d: side PAIR_PX,
            centred on the midpoint of the two closest pixels, rotated so that the line joining
            the two blob centroids is horizontal (sign ambiguity as above)
    merged  (type 2) side PAIR_PX, centred on the merged blob's centroid, rotated to its
            principal axis
Frame geometry saved for baselines: n_blobs, min nearest-pixel distance between blobs (0 if a
merged blob exists).

Functions:
    DEFAULTS                    thresholds (areas / distance filled in from calibration)
    dark_mask                   grey + pix_bg -> bool dark mask
    find_blobs                  mask -> list of blob dicts
    nearest_pair                closest pixels of two blobs (via their boundary pixels)
    frame_units                 one frame -> blobs + crop specs + frame geometry
    crop_grid                   crop specs -> affine grids for torch grid_sample
    CropFrameDataset            frame path -> (rgb uint8 tensor, crop specs, frame geometry)
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

FRAME_PX = 512
SINGLE_PX = 128
PAIR_PX = 160
OUT_PX = 224

DEFAULTS = {'dark_abs': 60, 'dark_rel': 40, 'open_r': 2, 'min_area': 150,
            'single_lo': None, 'single_hi': None, 'pair_d': None}


def _disk(r):
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def dark_mask(grey, pix_bg, p=DEFAULTS):
    g = grey.astype(np.int16)
    return (g < p['dark_abs']) & ((pix_bg.astype(np.int16) - g) > p['dark_rel'])


def find_blobs(mask, p=DEFAULTS):
    """-> list of dicts: area, cy, cx, angle (rad, principal axis, image coords y down), major,
    minor (4 sqrt(eigenvalue)), pix (N, 2) int (y, x), edge (M, 2) boundary pixels."""
    m = ndi.binary_opening(mask, structure=_disk(p['open_r'])) if p['open_r'] > 0 else mask
    m = ndi.binary_fill_holes(m)
    lab, n = ndi.label(m, structure=np.ones((3, 3), bool))
    if n == 0:
        return []
    areas = np.bincount(lab.ravel(), minlength=n + 1)
    er = lab * (~ndi.binary_erosion(m, structure=np.ones((3, 3), bool)))
    out = []
    objs = ndi.find_objects(lab)
    for i in range(1, n + 1):
        if areas[i] < p['min_area']:
            continue
        sl = objs[i - 1]
        yy, xx = np.nonzero(lab[sl] == i)
        yy, xx = yy + sl[0].start, xx + sl[1].start
        ey, ex = np.nonzero(er[sl] == i)
        cy, cx = yy.mean(), xx.mean()
        cov = np.cov(np.stack([xx - cx, yy - cy])) if len(xx) > 2 else np.eye(2)
        w, v = np.linalg.eigh(cov)
        ang = float(np.arctan2(v[1, 1], v[0, 1]))  # direction of the largest eigenvector (x, y)
        out.append({'area': int(areas[i]), 'cy': float(cy), 'cx': float(cx), 'angle': ang,
                    'major': float(4 * np.sqrt(max(w[1], 0))), 'minor': float(4 * np.sqrt(max(w[0], 0))),
                    'edge': np.stack([ey + sl[0].start, ex + sl[1].start], 1)})
    return out


def nearest_pair(a, b):
    """-> (distance, point on a (y, x), point on b) between the boundary pixels of blobs a, b."""
    t = cKDTree(b['edge'])
    d, j = t.query(a['edge'], k=1)
    i = int(np.argmin(d))
    return float(d[i]), a['edge'][i].astype(float), b['edge'][int(j[i])].astype(float)


def blob_class(area, p):
    if area > p['single_hi']:
        return 'merged'
    if area >= p['single_lo']:
        return 'single'
    return 'fragment'


def frame_units(grey, pix_bg, p):
    """One frame -> (blobs, crops, geom).
    crops: float32 (n, 5) rows [type, cy, cx, angle, side]; plus int (n, 2) blob indices
    (second = -1 for single / merged). geom: dict n_blobs, n_single, n_merged, n_fragment,
    min_dist (nearest-pixel distance between any two blobs; 0 if a merged blob exists; 999 if < 2 blobs
    and none merged)."""
    blobs = find_blobs(dark_mask(grey, pix_bg, p), p)
    cls = [blob_class(b['area'], p) for b in blobs]
    crops, idx = [], []
    for i, (b, c) in enumerate(zip(blobs, cls)):
        if c == 'single':
            crops.append([0, b['cy'], b['cx'], b['angle'], SINGLE_PX]); idx.append([i, -1])
        elif c == 'merged':
            crops.append([2, b['cy'], b['cx'], b['angle'], PAIR_PX]); idx.append([i, -1])
    dmin = 999.0
    for i in range(len(blobs)):
        for j in range(i + 1, len(blobs)):
            # cheap reject: centroid distance minus both half-lengths
            cd = np.hypot(blobs[i]['cy'] - blobs[j]['cy'], blobs[i]['cx'] - blobs[j]['cx'])
            if cd - blobs[i]['major'] / 2 - blobs[j]['major'] / 2 > max(p['pair_d'], dmin) + 20:
                continue
            d, pa, pb = nearest_pair(blobs[i], blobs[j])
            dmin = min(dmin, d)
            if d < p['pair_d']:
                mid = (pa + pb) / 2
                ang = float(np.arctan2(blobs[j]['cy'] - blobs[i]['cy'], blobs[j]['cx'] - blobs[i]['cx']))
                crops.append([1, mid[0], mid[1], ang, PAIR_PX]); idx.append([i, j])
    n_m = sum(c == 'merged' for c in cls)
    geom = {'n_blobs': len(blobs), 'n_single': sum(c == 'single' for c in cls), 'n_merged': n_m,
            'n_fragment': sum(c == 'fragment' for c in cls), 'min_dist': 0.0 if n_m else dmin}
    return blobs, np.asarray(crops, np.float32).reshape(-1, 5), np.asarray(idx, np.int32).reshape(-1, 2), geom


def crop_grid(specs, out_px=OUT_PX, frame_px=FRAME_PX):
    """specs (n, 5) [type, cy, cx, angle, side] -> (n, out, out, 2) grid for F.grid_sample
    (align_corners=False) on a frame_px frame: output x axis = the crop's rotation direction."""
    s = torch.as_tensor(specs, dtype=torch.float32)
    cy, cx, a, side = s[:, 1], s[:, 2], s[:, 3], s[:, 4]
    c, sn = torch.cos(a), torch.sin(a)
    h = side / frame_px  # half side in normalized units (normalized frame spans [-1, 1] = frame_px)
    theta = torch.zeros(len(s), 2, 3)
    theta[:, 0, 0], theta[:, 0, 1] = c * h, -sn * h
    theta[:, 1, 0], theta[:, 1, 1] = sn * h, c * h
    theta[:, 0, 2] = (cx + 0.5) / frame_px * 2 - 1
    theta[:, 1, 2] = (cy + 0.5) / frame_px * 2 - 1
    return F.affine_grid(theta, (len(s), 3, out_px, out_px), align_corners=False)


class CropFrameDataset(torch.utils.data.Dataset):
    """rows -> (rgb (3, 512, 512) uint8, crops (n, 5), idx (n, 2), geom array (5,), row).
    pix_bg per video from bg_dir/<observation_id>.npz (fg448 backgrounds)."""

    def __init__(self, rows, frame_paths, obs_ids, bg_dir, dataset_dir, params):
        self.rows, self.fp, self.obs, self.bg_dir, self.ds, self.p = rows, frame_paths, obs_ids, Path(bg_dir), \
            Path(dataset_dir), params
        self._bg = {}

    def pix_bg(self, obs):
        if obs not in self._bg:
            if len(self._bg) > 4:
                self._bg.pop(next(iter(self._bg)))
            self._bg[obs] = np.load(self.bg_dir / f'{obs}.npz')['pix_bg']
        return self._bg[obs]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = int(self.rows[i])
        with Image.open(self.ds / self.fp[r]) as im:
            rgb = np.asarray(im.convert('RGB'))
        grey = np.asarray(Image.fromarray(rgb).convert('L'))
        _, crops, idx, g = frame_units(grey, self.pix_bg(self.obs[r]), self.p)
        geom = np.array([g['n_blobs'], g['n_single'], g['n_merged'], g['n_fragment'], g['min_dist']], np.float32)
        return torch.from_numpy(rgb.copy()).permute(2, 0, 1), torch.from_numpy(crops), torch.from_numpy(idx), \
            torch.from_numpy(geom), r


def collate(batch):
    rgb = torch.stack([b[0] for b in batch])
    crops = [b[1] for b in batch]
    fi = torch.cat([torch.full((len(c),), k, dtype=torch.long) for k, c in enumerate(crops)])
    return rgb, torch.cat(crops), torch.cat([b[2] for b in batch]), fi, torch.stack([b[3] for b in batch]), \
        np.array([b[4] for b in batch])


IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@torch.no_grad()
def sample_crops(rgb, crops, fi, out_px=OUT_PX):
    """rgb (B, 3, 512, 512) uint8 (device), crops (n, 5), fi (n,) frame index -> (n, 3, out, out) float in [0, 1]
    (bilinear; outside the frame = white-ish border replicated)."""
    if len(crops) == 0:
        return torch.zeros(0, 3, out_px, out_px, device=rgb.device)
    grid = crop_grid(crops, out_px).to(rgb.device)
    src = rgb[fi.to(rgb.device)].float() / 255
    return F.grid_sample(src, grid, mode='bilinear', padding_mode='border', align_corners=False)
