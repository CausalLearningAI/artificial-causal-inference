"""
Foreground ("mouse patches") selection for the ECI pipeline on mice v1.

The first SAEs were trained on every DINOv2 patch of a 224 center crop, and most of
their concepts describe the background (bedding, rim, lighting). Here the model sees
the WHOLE 512x512 frame resized to 448x448 (no crop, 32x32 = 1024 patches, each patch
= a 16x16 pixel block of the original frame), and only patches that differ from the
video's own background are kept. No behaviour annotation is used anywhere here.

Per video (observation), from n_bg frames spread evenly over the video:
    pix_bg      per-pixel HIGH quantile (0.98, fg_background.py --pix-q) of the grey level: the bright bedding
                seen when no (black) mouse sits there, robust to mice resting in one place
                for up to ~90% of the video.
    dark(f, p)  fraction of the 16x16 pixels of patch p in frame f that are dark
                (grey < dark_abs) AND much darker than pix_bg (drop > dark_rel):
                mice are black on light bedding; static dark objects (water spout,
                shadows) are dark in pix_bg too and are not counted.
    bg_median   per-position MEDIAN DINOv2 token over the n_bg frames.
    bg_masked   per-position median over the frames where that patch is NOT dark-
                foreground (dark < dark_frac), falling back to bg_median when fewer than
                min_bg_frames such frames exist. Mice huddling in one corner for most of
                the video do not leak into this background.
    bg_seg      the same masked median computed separately in n_seg=4 time blocks of the
                sample (bedding gets rearranged over a 20-40 min video); a frame uses the
                block nearest in time (segment_of).
Per frame and patch:
    dist        cosine distance 1 - cos(token, bg_seg[block, position])
    foreground  FG_RULE: core = drop_isolated(dist > thr_video) OR (dark > 0.15), then a
                1-patch (3x3) dilation so noses / tails at mouse edges are kept.
                thr_video = 0.995 quantile of dist over the video's sample patches that are
                > 2 patches away from any dark-cue patch (surely not a black mouse).
                The dark cue is what keeps mice sleeping still for most of a video (stage 4
                huddles), whose tokens leak into the median background. The experimenter's
                hand / the white card at the start and end of videos is foreground too.
See scripts/eci/fg_background.py (per-video backgrounds) and FG_RULE for the rule used.

Functions / classes:
    PATCH_PX, GRID              geometry (512 px frame, 448 input, 32 x 32 patches)
    FrameDatasetFG              frame -> (DINOv2 pixel_values, grey uint8 frame)
    load_encoder_fg             DINOv2 on the whole frame at 448 (extract.load_encoder, no crop)
    encode_batch                pixel_values -> (B, 1024, 768) patch tokens, fp16-rounded
    pixel_background            per-pixel high quantile of grey frames
    dark_fraction               per-patch fraction of dark, darker-than-background pixels
    patch_background            per-position median (optionally masked) token
    cosine_distance             (B, P, d) tokens vs (P, d) background -> (B, P)
    video_threshold             robust per-video threshold on dist
    dilate                      3x3 (or larger) dilation of a (B, P) mask on the 32x32 grid
    foreground_mask             the full rule
    load_background             one video's saved background file
    obs_rows                    observation -> contiguous annotations.csv rows
    segment_of                  time block of a frame for bg_seg
    FgBackgrounds               all videos' backgrounds, mask(tokens, grey, rows) for any batch
    FgTokenStore                reader for the foreground training-token shards
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.eci.extract import load_encoder

FRAME_PX = 512
RESOLUTION = 448
GRID = 32
PATCH_PX = FRAME_PX // GRID  # 16 original pixels per patch side

# the rule used for training / encoding (chosen by visual validation, see the report)
FG_RULE = {
    'background': 'bg_seg',  # 'bg_median' | 'bg_masked' | 'bg_seg' (time-local masked median)
    'thr_mode': 'far_q', 'thr_q': 0.995, 'far_r': 2, 'thr_min': 0.0, 'mad_k': 6.0,
    'dark_abs': 60, 'dark_rel': 40, 'dark_frac': 0.15,
    'use_dark': True, 'drop_isolated': True, 'dilate': 1,
}


class FrameDatasetFG(torch.utils.data.Dataset):
    """frame path -> (pixel_values (3, 448, 448) float32, grey (512, 512) uint8[, row])."""

    def __init__(self, paths, processor, rows=None):
        self.paths, self.processor, self.rows = paths, processor, rows

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            image = im.convert('RGB')
        if image.size != (FRAME_PX, FRAME_PX):
            raise ValueError(f'{self.paths[i]}: size {image.size}, expected {FRAME_PX}x{FRAME_PX}')
        grey = torch.from_numpy(np.asarray(image.convert('L'), dtype=np.uint8).copy())
        pix = self.processor(images=image, return_tensors='pt')['pixel_values'][0]
        return (pix, grey) if self.rows is None else (pix, grey, int(self.rows[i]))


def load_encoder_fg(name='dinov2_base', device='cuda'):
    return load_encoder(name, RESOLUTION, device, center_crop=False)


@torch.no_grad()
def encode_batch(model, pix, device):
    """(B, 3, 448, 448) -> patch tokens (B, 1024, d) float16 (last_hidden_state, fp32 forward,
    rounded to fp16 exactly as stored training tokens)."""
    n_prefix = 1 + (getattr(model.config, 'num_register_tokens', 0) or 0)
    with torch.inference_mode():
        hs = model(pixel_values=pix.to(device, non_blocking=True)).last_hidden_state.float()
    if hs.shape[1] != n_prefix + GRID * GRID or not torch.isfinite(hs).all():
        raise RuntimeError(f'bad DINOv2 output: shape {tuple(hs.shape)}')
    return hs[:, n_prefix:].half()


@torch.no_grad()
def pixel_background(grey, q=0.9):
    """grey (N, H, W) uint8 tensor -> (H, W) uint8 per-pixel q-quantile."""
    k = max(1, min(grey.shape[0], int(round(q * grey.shape[0]))))
    return grey.float().kthvalue(k, dim=0).values.round().to(torch.uint8)


@torch.no_grad()
def dark_fraction(grey, pix_bg, dark_abs=60, dark_rel=40):
    """grey (B, 512, 512) uint8, pix_bg (512, 512) uint8 -> (B, 1024) float fraction of
    pixels per patch that are dark and darker than the pixel background by > dark_rel."""
    g = grey.float()
    dark = (g < dark_abs) & ((pix_bg.float()[None] - g) > dark_rel)
    return F.avg_pool2d(dark.float()[:, None], PATCH_PX).flatten(1)


@torch.no_grad()
def patch_background(tokens, exclude=None, min_frames=20):
    """tokens (N, P, d); exclude (N, P) bool (True = frame not used at that position).
    -> (P, d) float32 per-position median (NaN-masked median where exclude is given;
    positions with < min_frames usable frames fall back to the plain median)."""
    t = tokens.float()
    plain = t.median(0).values
    if exclude is None:
        return plain, None
    n_ok = (~exclude).sum(0)
    t = t.masked_fill(exclude[..., None], float('nan'))
    masked = t.nanmedian(0).values
    fallback = n_ok < min_frames
    masked[fallback] = plain[fallback]
    return masked, n_ok


@torch.no_grad()
def cosine_distance(tokens, bg):
    """tokens (B, P, d), bg (P, d) -> (B, P) 1 - cos."""
    return 1 - F.cosine_similarity(tokens.float(), bg.float()[None].to(tokens.device), dim=-1)


def video_threshold(dist, dark=None, rule=None):
    """Robust per-video threshold on the (N, P) distance sample of one video.
    thr_mode 'mad': median + mad_k * 1.4826 * MAD over all patches.
    thr_mode 'far_q': the thr_q quantile of dist over patches farther than far_r patches
        from any dark-cue patch (i.e. surely not a black mouse), floored at thr_min."""
    rule = rule or FG_RULE
    d = np.asarray(dist, dtype=np.float32)
    if rule.get('thr_mode', 'mad') == 'mad':
        v = d.ravel().astype(np.float64)
        med = np.median(v)
        return float(max(rule.get('thr_min', 0.0), med + rule['mad_k'] * 1.4826 * np.median(np.abs(v - med))))
    near = dilate(torch.from_numpy(np.asarray(dark, dtype=np.float32) > rule['dark_frac']), rule['far_r']).numpy()
    far = d[~near]
    return float(max(rule.get('thr_min', 0.0), np.quantile(far, rule['thr_q'])))


@torch.no_grad()
def dilate(mask, r=1):
    """(B, 1024) bool -> dilated by a (2r+1)x(2r+1) square on the 32x32 grid."""
    if r <= 0:
        return mask
    m = mask.view(-1, 1, GRID, GRID).float()
    return (F.max_pool2d(m, 2 * r + 1, stride=1, padding=r) > 0).view(mask.shape[0], -1)


@torch.no_grad()
def drop_isolated(mask):
    """Remove patches of a (B, 1024) mask that have no 8-neighbour in the mask."""
    m = mask.view(-1, 1, GRID, GRID).float()
    nb = F.conv2d(m, torch.ones(1, 1, 3, 3, device=m.device), padding=1) - m
    return mask & (nb.view(mask.shape[0], -1) > 0)


@torch.no_grad()
def foreground_parts(dist, dark, thr, rule=FG_RULE):
    """-> (feature core, dark core, final mask), each (B, 1024) bool."""
    feat = dist > thr
    if rule.get('drop_isolated', False):
        feat = drop_isolated(feat)
    dk = dark > rule['dark_frac'] if rule.get('use_dark', True) else torch.zeros_like(feat)
    return feat, dk, dilate(feat | dk, rule.get('dilate', 1))


@torch.no_grad()
def foreground_mask(dist, dark, thr, rule=FG_RULE):
    """dist, dark (B, 1024) tensors, thr the video threshold -> (B, 1024) bool."""
    return foreground_parts(dist, dark, thr, rule)[2]


def segment_of(rows, sample_rows, seg_bounds):
    """Time block (index into bg_seg) of annotations.csv rows: the block whose sample
    frames are nearest in time (boundaries halfway between consecutive blocks)."""
    sample_rows, seg_bounds = np.asarray(sample_rows), np.asarray(seg_bounds)
    cuts = [(sample_rows[b - 1] + sample_rows[b]) / 2 for b in seg_bounds[1:-1]]
    return np.searchsorted(np.asarray(cuts), np.asarray(rows), side='right')


def load_background(bg_dir, observation_id, rule=FG_RULE, device='cpu'):
    """-> dict: bg (n_seg or 1, 1024, d) float32 tensor, pix_bg (512, 512) uint8 tensor,
    thr float, rows / seg_bounds (to place frames in time blocks, see segment_of)."""
    z = np.load(Path(bg_dir) / f'{observation_id}.npz')
    key = rule['background']
    bg = z[key].astype(np.float32)
    bg = bg if bg.ndim == 3 else bg[None]
    thr = video_threshold(z[f'dist_{key}'], z['dark'], rule)
    seg_bounds = z['seg_bounds'] if key == 'bg_seg' else np.array([0, len(z['rows'])])
    return {'bg': torch.from_numpy(bg).to(device), 'pix_bg': torch.from_numpy(z['pix_bg']).to(device),
            'thr': thr, 'rows': z['rows'], 'seg_bounds': seg_bounds}


def obs_rows(ann_path):
    """observation_id -> (lo, hi) contiguous row range of annotations.csv."""
    import pandas as pd
    obs = pd.read_csv(ann_path, usecols=['observation_id'])['observation_id']
    change = np.r_[0, np.nonzero(obs.values[1:] != obs.values[:-1])[0] + 1, len(obs)]
    ids = obs.values[change[:-1]]
    if len(set(ids)) != len(ids):
        raise RuntimeError('observations are not contiguous in annotations.csv')
    return {o: (int(a), int(b)) for o, a, b in zip(ids, change[:-1], change[1:])}


class FgBackgrounds:
    """All per-video backgrounds, looked up by annotations.csv row.

    obs_ranges: {observation_id: (lo, hi)} contiguous rows (fg_background.obs_rows).
    mask(tokens, grey, rows): foreground mask (B, 1024) for a batch of frames that may
    span several videos (rows sorted or not)."""

    def __init__(self, bg_dir, obs_ranges, rule=FG_RULE, device='cuda', cache=4):
        self.bg_dir, self.rule, self.device, self.cache_n = Path(bg_dir), rule, torch.device(device), cache
        items = sorted(obs_ranges.items(), key=lambda kv: kv[1][0])
        self.ids = [k for k, _ in items]
        self.starts = np.array([v[0] for _, v in items], dtype=np.int64)
        self.ends = np.array([v[1] for _, v in items], dtype=np.int64)
        self._cache = {}

    def get(self, obs):
        if obs not in self._cache:
            if len(self._cache) >= self.cache_n:
                self._cache.pop(next(iter(self._cache)))
            self._cache[obs] = load_background(self.bg_dir, obs, self.rule, self.device)
        return self._cache[obs]

    def obs_index(self, rows):
        k = np.searchsorted(self.starts, np.asarray(rows), side='right') - 1
        assert (np.asarray(rows) < self.ends[k]).all()
        return k

    @torch.no_grad()
    def mask(self, tokens, grey, rows):
        """tokens (B, 1024, d) on device, grey (B, 512, 512) uint8 on device, rows (B,) numpy.
        -> (mask (B, 1024) bool, dist (B, 1024) float)."""
        rows = np.asarray(rows)
        k = self.obs_index(rows)
        out = torch.zeros(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        dist = torch.zeros(tokens.shape[:2], dtype=torch.float32, device=tokens.device)
        for kk in np.unique(k):
            sel = np.nonzero(k == kk)[0]
            b = self.get(self.ids[kk])
            seg = segment_of(rows[sel], b['rows'], b['seg_bounds'])
            sel_t = torch.from_numpy(sel).to(tokens.device)
            bg = b['bg'][torch.from_numpy(seg).to(b['bg'].device)]  # (n, 1024, d)
            d = 1 - F.cosine_similarity(tokens[sel_t].float(), bg, dim=-1)
            dk = dark_fraction(grey[sel_t], b['pix_bg'], self.rule['dark_abs'], self.rule['dark_rel'])
            out[sel_t] = foreground_mask(d, dk, b['thr'], self.rule)
            dist[sel_t] = d
        return out, dist


class FgTokenStore:
    """Foreground token shards written by scripts/eci/fg_extract_train.py.

    tokens(s) -> np.memmap (n, d) float16 of shard s; row / pos -> per-token annotations.csv
    row and patch position; frames -> per-frame (rows, n_fg) of all shards concatenated."""

    def __init__(self, root):
        self.root = Path(root)
        self.dirs = sorted(d for d in (self.root / 'shards').iterdir() if d.is_dir() and not d.name.endswith('.tmp'))
        missing = [str(d) for d in self.dirs if not (d / 'DONE').exists()]
        if missing:
            raise RuntimeError(f'unfinished shards: {missing}')
        import json
        self.info = [json.loads((d / 'shard.json').read_text()) for d in self.dirs]
        self.dim = self.info[0]['dim']
        self.sizes = np.array([i['n_tokens'] for i in self.info], dtype=np.int64)

    def tokens(self, s):
        return np.memmap(self.dirs[s] / 'tokens.f16', dtype=np.float16, mode='r', shape=(int(self.sizes[s]), self.dim))

    def row(self, s):
        return np.fromfile(self.dirs[s] / 'row.i32', dtype=np.int32)

    def pos(self, s):
        return np.fromfile(self.dirs[s] / 'pos.i16', dtype=np.int16)

    def frames(self):
        z = [np.load(d / 'frames.npz') for d in self.dirs]
        return np.concatenate([a['rows'] for a in z]), np.concatenate([a['n_fg'] for a in z])
