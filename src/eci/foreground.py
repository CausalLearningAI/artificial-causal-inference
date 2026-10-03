"""
Foreground ("mouse patches") selection for the ECI pipeline on mice v1 (and ants, RULES).

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
                FG_RULE_V3 (mask v3): the dilation only adds patches with >= 1 dark pixel
                (dark > 0), so bedding / static objects next to a mouse are not admitted.
                thr_video = 0.995 quantile of dist over the video's sample patches that are
                > 2 patches away from any dark-cue patch (surely not a black mouse).
                The dark cue is what keeps mice sleeping still for most of a video (stage 4
                huddles), whose tokens leak into the median background. The experimenter's
                hand / the white card at the start and end of videos is foreground too.
See scripts/eci/fg_background.py (per-video backgrounds) and FG_RULE for the rule used.

Rules (RULES, chosen by name with --rule; the SAE checkpoint records it as 'fg_rule'):
    fg448   FG_RULE (mice foreground SAE)
    v3      FG_RULE_V3 (mice, dilation only onto dark patches)
    ants    ants-only foreground: starts as FG_RULE_V3 (ants are dark on light ground, and small, so the
            dilation must not admit bedding); its parameters are tuned on the ants backgrounds with
            scripts/eci/fg_validate.py --domain ants before any token extraction
    all     every patch is foreground (the whole frame at 448, 1024 patches); needs no background
    frogs   FG_RULE_FROGS: dark frog pixels (frog-free pixel background, dish ROI, near the frame's SLEAP nodes), no
            DINOv2 cue; backgrounds from scripts/eci/frogs_background.py

Background-subtracted SAE input (train_sae_fg.py --bg-sub, checkpoint 'bg_sub'): token - the video's background
token at the same patch position and time block (the rule's background, bg_seg for fg448 / v3 / ants: the masked
per-position median DINOv2 token of the empty arena from fg_background.py). The mask is unchanged. Where an animal
stayed still for > 90% of the sample (leak_mask: the position fell back to the plain median, so its background token
is the animal), the background token is replaced by the mean of the clean neighbouring positions (fill_leaks).

Encoders (FgEncoder): the foreground mask is ALWAYS computed from DINOv2-base tokens at 448 (the rules and the
per-video backgrounds above are DINOv2 quantities). The SAE tokens come from the chosen encoder: 'dinov2_base'
(default, the same tokens as the mask) or 'dinov3_base' (DINOv3 ViT-B/16, whole 512 px frame, no resize, patch 16
-> the same 32 x 32 grid; each patch is exactly the 16 x 16 pixel block that dark_fraction pools; CLS and the 4
register tokens are dropped) or 'dinov3_small' (DINOv3 ViT-S/16, 384-dim tokens, otherwise the same as 'dinov3_base').
Both encoders therefore see identical foreground patches.

Odor alignment (--align odor, tag 'fg448al'; mice only): every frame of a video is rotated by a multiple of 90 degrees
(lossless, before DINOv2 and before the grey frame of the dark cue) so that the video's odor corner
(dataset/mice/v1/eci/odor_corner.csv, scripts/eci/mice_odor_corner.py) lands at the top right of the image:
TR 0, BR 90 degrees counter-clockwise, BL 180, TL 270. All 432 mice videos are rotations of one cage layout (the water
spout and the wall vent land on the left wall, the bag at the top right; checked on the per-video backgrounds, no
mirror needed). Backgrounds, masks, training tokens and codes of an aligned SAE are all computed on aligned frames
(fg448al/background, whose npz files record align='odor'; patch positions are aligned-frame positions). 'none' (the
default) is the unrotated pipeline, unchanged. The SAE checkpoint and the token store record 'align' (absent = none).

Functions / classes:
    PATCH_PX, GRID              geometry (512 px frame, 448 input, 32 x 32 patches)
    ALIGNS, align_rot90         frame alignment names; per annotations.csv row: number of 90-degree CCW turns
    FrameDatasetFG              frame -> (DINOv2 pixel_values, grey uint8 frame[, row][, 2nd encoder pixel_values])
    load_encoder_fg             DINOv2 on the whole frame at 448 (extract.load_encoder, no crop); DINOv3 at 512
    FgEncoder                   mask encoder (DINOv2 448) + SAE-token encoder (DINOv2 itself or DINOv3 512)
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
    near_nodes, frog_pixels     rule 'frogs': pixels near the frame's SLEAP nodes; dark frog pixels
    FgBackgrounds               all videos' backgrounds, mask(tokens, grey, rows) for any batch;
                                background(rows) = the background tokens of a batch (background-subtracted input)
    leak_mask, fill_leaks       background positions that are the animal itself; filled from clean neighbours
    input_background            the background of the background-subtracted input (leaks filled)
    subtract_background_tokens  token - background token (float32, rounded to fp16)
    subtract_background_np      the same for a numpy token store (rows + patch positions), CPU
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

# mask v3 ("mice only"): as FG_RULE, but the 1-patch dilation may only add patches that contain at
# least one dark mouse pixel (dark fraction > 0). Diagnosis of fg448 latent 46: 87% of its in-mask
# activation sat on dilation-only patches (75% on dilation patches with no dark pixel at all), i.e.
# the white odor object in the arena corner entered the mask whenever a mouse sat next to it. A
# temporal-change gate on the feature cue was measured as well and rejected: it removed no further
# object activation and dropped still tails (see the v3 report).
FG_RULE_V3 = {**FG_RULE, 'dilate_requires_dark': True}
# ants: see the module docstring (a starting point, tuned before use)
FG_RULE_ANTS = {**FG_RULE_V3}
# whole frame: no background, no threshold, every patch kept (foreground_parts / FgBackgrounds.mask)
FG_RULE_ALL = {'all': True}
# frogs: one dark froglet per backlit dish (src/eci/domain.py FrogsDomain). No DINOv2 background and no feature cue: a
# pixel is frog when it is darker than the video's frog-free pixel background by > dark_rel, inside the dish ROI and
# within near_px of a SLEAP node of that frame (anywhere in the ROI when no node was predicted). The dark dish rim,
# labels and static debris are dark in the background too, so they never count. Patches with a frog-pixel fraction >
# dark_frac form the core (limbs are thin: 0.05, not 0.15), then the 1-patch dilation onto patches with any frog pixel
# (as v3). Backgrounds: scripts/eci/frogs_background.py (pix_bg, roi, per-row SLEAP nodes at 512 px).
FG_RULE_FROGS = {'frogs': True, 'background': 'frog_free', 'dark_rel': 30, 'dark_frac': 0.05, 'near_px': 16,
                 'use_dark': True, 'drop_isolated': False, 'dilate': 1, 'dilate_requires_dark': True}
RULES = {'fg448': FG_RULE, 'v3': FG_RULE_V3, 'ants': FG_RULE_ANTS, 'all': FG_RULE_ALL, 'frogs': FG_RULE_FROGS}

MASK_ENCODER = 'dinov2_base'
# input resolution per encoder: 32 x 32 patches for both (448 / 14 = 512 / 16 = 32)
ENCODER_RESOLUTION = {'dinov2_base': RESOLUTION, 'dinov3_base': FRAME_PX, 'dinov3_small': FRAME_PX}
ENCODERS = tuple(ENCODER_RESOLUTION)

# frame alignment (module docstring): name -> per-video rotation. 'odor': np.rot90 / PIL ROTATE_90 counter-clockwise
# turns that bring the video's odor corner to the top right.
ALIGNS = ('none', 'odor')
ODOR_ROT90 = {'TR': 0, 'BR': 1, 'BL': 2, 'TL': 3}
_PIL_ROT = {1: Image.Transpose.ROTATE_90, 2: Image.Transpose.ROTATE_180, 3: Image.Transpose.ROTATE_270}


def align_tag(align):
    """Directory tag of the foreground pipeline: 'fg448' (none) or 'fg448al' (odor)."""
    return {'none': 'fg448', 'odor': 'fg448al'}[align]


def align_rot90(align, ann_path, corner_csv=None):
    """-> None ('none') or (n_rows,) int8: 90-degree counter-clockwise turns of every annotations.csv row's frame.
    'odor': from the odor-corner table (default <annotations dir>/eci/odor_corner.csv); every video must be in it."""
    if align not in ALIGNS:
        raise ValueError(f'unknown align {align!r}; known: {ALIGNS}')
    if align == 'none':
        return None
    import pandas as pd
    corner_csv = Path(corner_csv) if corner_csv else Path(ann_path).parent / 'eci' / 'odor_corner.csv'
    if not corner_csv.exists():
        raise FileNotFoundError(f'align odor needs the odor-corner table {corner_csv} (scripts/eci/mice_odor_corner.py)')
    corner = pd.read_csv(corner_csv).set_index('observation_id')['odor_corner']
    ranges = obs_rows(ann_path)
    missing = sorted(set(ranges) - set(corner.index))
    if missing:
        raise RuntimeError(f'{len(missing)} videos without an odor corner, e.g. {missing[:3]}')
    out = np.full(max(hi for _, hi in ranges.values()), -1, np.int8)
    for o, (lo, hi) in ranges.items():
        out[lo:hi] = ODOR_ROT90[corner[o]]
    assert (out >= 0).all()
    return out


def rotate_image(image, k):
    """PIL image turned k x 90 degrees counter-clockwise (lossless; k = 0 returns it unchanged)."""
    k = int(k) % 4
    return image if k == 0 else image.transpose(_PIL_ROT[k])


class FrameDatasetFG(torch.utils.data.Dataset):
    """frame path -> (pixel_values (3, 448, 448) float32, grey (512, 512) uint8[, row][, pixel_values_2]).
    processor2 (optional): a second encoder's processor; its pixel_values are appended last.
    rot (optional): per path, 90-degree counter-clockwise turns applied to the frame first (align_rot90)."""

    def __init__(self, paths, processor, rows=None, processor2=None, rot=None):
        self.paths, self.processor, self.rows, self.processor2 = paths, processor, rows, processor2
        self.rot = None if rot is None else np.asarray(rot)
        if self.rot is not None and len(self.rot) != len(paths):
            raise ValueError(f'rot has {len(self.rot)} entries for {len(paths)} paths')

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            image = im.convert('RGB')
        if image.size != (FRAME_PX, FRAME_PX):
            raise ValueError(f'{self.paths[i]}: size {image.size}, expected {FRAME_PX}x{FRAME_PX}')
        if self.rot is not None:
            image = rotate_image(image, self.rot[i])
        grey = torch.from_numpy(np.asarray(image.convert('L'), dtype=np.uint8).copy())
        pix = self.processor(images=image, return_tensors='pt')['pixel_values'][0]
        out = (pix, grey) if self.rows is None else (pix, grey, int(self.rows[i]))
        if self.processor2 is not None:
            out = out + (self.processor2(images=image, return_tensors='pt')['pixel_values'][0],)
        return out


def load_encoder_fg(name='dinov2_base', device='cuda'):
    return load_encoder(name, ENCODER_RESOLUTION[name], device, center_crop=False)


class FgEncoder:
    """Mask encoder (DINOv2-base at 448: the foreground rule's tokens) and SAE-token encoder.
    encoder == 'dinov2_base': one model, the SAE tokens are the mask tokens (model2 is None).
    Otherwise model2 / processor2 encode the same frames (FrameDatasetFG appends their pixel_values)."""

    def __init__(self, encoder='dinov2_base', device='cuda'):
        if encoder not in ENCODER_RESOLUTION:
            raise ValueError(f'unknown encoder {encoder!r}; known: {ENCODERS}')
        self.encoder, self.device = encoder, torch.device(device)
        _, self.processor, self.model = load_encoder_fg(MASK_ENCODER, self.device)
        self.processor2 = self.model2 = None
        if encoder != MASK_ENCODER:
            _, self.processor2, self.model2 = load_encoder_fg(encoder, self.device)

    def dataset(self, paths, rows=None, rot=None):
        return FrameDatasetFG(paths, self.processor, rows, self.processor2, rot)

    def sae_tokens(self, mask_tok, pix2=None):
        """SAE input tokens (B, 1024, d): the mask tokens, or the second encoder's tokens of pix2."""
        if self.model2 is None:
            return mask_tok
        return encode_batch(self.model2, pix2, self.device)


@torch.no_grad()
def encode_batch(model, pix, device):
    """(B, 3, 448, 448) -> patch tokens (B, 1024, d) float16 (last_hidden_state, fp32 forward,
    rounded to fp16 exactly as stored training tokens)."""
    n_prefix = 1 + (getattr(model.config, 'num_register_tokens', 0) or 0)
    with torch.inference_mode():
        hs = model(pixel_values=pix.to(device, non_blocking=True)).last_hidden_state.float()
    if hs.shape[1] != n_prefix + GRID * GRID or not torch.isfinite(hs).all():
        raise RuntimeError(f'bad encoder output: shape {tuple(hs.shape)}')
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
    if rule.get('all', False):
        every = torch.ones_like(dist, dtype=torch.bool)
        return every, torch.zeros_like(every), every
    feat = dist > thr
    if rule.get('drop_isolated', False):
        feat = drop_isolated(feat)
    dk = dark > rule['dark_frac'] if rule.get('use_dark', True) else torch.zeros_like(feat)
    core = feat | dk
    grown = dilate(core, rule.get('dilate', 1))
    if rule.get('dilate_requires_dark', False):
        grown = core | (grown & (dark > 0))
    return feat, dk, grown


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
    thr float, rows / seg_bounds (to place frames in time blocks, see segment_of), src (bg_dir, observation_id,
    rule) for input_background, align (the frame alignment the file was computed on, absent = 'none').
    Rule 'frogs': pix_bg, roi (512, 512) bool, nodes (n_rows, K, 2) float32 SLEAP nodes of every row of the video
    (512 px, NaN = not predicted), lo (the video's first annotations.csv row), align, src."""
    z = np.load(Path(bg_dir) / f'{observation_id}.npz')
    if rule.get('frogs', False):
        return {'pix_bg': torch.from_numpy(z['pix_bg']).to(device), 'roi': torch.from_numpy(z['roi']).to(device),
                'nodes': torch.from_numpy(z['nodes'].astype(np.float32)).to(device), 'lo': int(z['lo']),
                'rows': z['rows'], 'src': (bg_dir, observation_id, rule),
                'align': str(z['align']) if 'align' in z.files else 'none'}
    key = rule['background']
    bg = z[key].astype(np.float32)
    bg = bg if bg.ndim == 3 else bg[None]
    thr = video_threshold(z[f'dist_{key}'], z['dark'], rule)
    seg_bounds = z['seg_bounds'] if key == 'bg_seg' else np.array([0, len(z['rows'])])
    return {'bg': torch.from_numpy(bg).to(device), 'pix_bg': torch.from_numpy(z['pix_bg']).to(device),
            'thr': thr, 'rows': z['rows'], 'seg_bounds': seg_bounds, 'src': (bg_dir, observation_id, rule),
            'align': str(z['align']) if 'align' in z.files else 'none'}


@torch.no_grad()
def near_nodes(nodes, r, size=FRAME_PX):
    """nodes (B, K, 2) float x, y (NaN = not predicted) -> (B, size, size) bool: pixel centre within r of a node of
    its frame; a frame without any node is True everywhere."""
    c = torch.arange(size, device=nodes.device, dtype=torch.float32) + 0.5
    out = torch.zeros((nodes.shape[0], size, size), dtype=torch.bool, device=nodes.device)
    for k in range(nodes.shape[1]):
        x, y = nodes[:, k, 0], nodes[:, k, 1]
        ok = torch.isfinite(x) & torch.isfinite(y)
        dx2 = (c[None] - torch.where(ok, x, 0)[:, None]) ** 2  # (B, size)
        dy2 = (c[None] - torch.where(ok, y, 0)[:, None]) ** 2
        out |= ((dy2[:, :, None] + dx2[:, None, :]) < r * r) & ok[:, None, None]
    none = ~(torch.isfinite(nodes[..., 0]).any(1))
    out[none] = True
    return out


@torch.no_grad()
def frog_pixels(grey, pix_bg, roi, nodes, rule=FG_RULE_FROGS):
    """grey (B, 512, 512) uint8, pix_bg (512, 512) uint8, roi (512, 512) bool, nodes (B, K, 2) -> (B, 512, 512) bool
    frog pixels of rule 'frogs' (FG_RULE_FROGS)."""
    dark = (pix_bg.float()[None] - grey.float()) > rule['dark_rel']
    return dark & roi[None] & near_nodes(nodes, rule['near_px'])


# fg_background.py defaults: a time block needs MIN_SEG_FRAMES frames free of the dark cue at a position (else the
# video-level masked median is used), the video-level masked median MIN_BG_FRAMES (else the plain median)
MIN_SEG_FRAMES, MIN_BG_FRAMES = 10, 20


def leak_mask(dark, n_ok, seg_bounds=None, rule=FG_RULE):
    """(n_seg, 1024) bool: background positions that ended up as the PLAIN median token (the dark animal cue covered
    the position in > 90% of the 200 sample frames of the video, and in the time block too). There the background
    token is the animal itself (checked visually: huddles of sleeping mice, ants that sat still).
    seg_bounds None: one block (bg_masked)."""
    video = np.asarray(n_ok) < MIN_BG_FRAMES
    if seg_bounds is None:
        return video[None]
    excl = dilate(torch.from_numpy(np.asarray(dark, dtype=np.float32) > rule['dark_frac']), 1).numpy()
    n_seg = np.stack([(~excl[a:b]).sum(0) for a, b in zip(seg_bounds[:-1], seg_bounds[1:])])
    return (n_seg < MIN_SEG_FRAMES) & video[None]


@torch.no_grad()
def fill_leaks(bg, leak):
    """bg (n_seg, 1024, d) float32, leak (n_seg, 1024) bool -> bg with every leaked position replaced by the mean of
    its 8-neighbour positions that are clean (or already filled), filled ring by ring from the outside in."""
    bg, todo = bg.clone(), leak.clone()
    for s in range(bg.shape[0]):
        if todo[s].all():  # nothing clean to fill from (never seen): leave as is
            continue
        while todo[s].any():
            ok = (~todo[s]).float().view(1, 1, GRID, GRID)
            k = torch.ones(1, 1, 3, 3, device=bg.device)
            cnt = F.conv2d(ok, k, padding=1).view(-1)
            g = (bg[s] * ok.view(-1, 1)).T.reshape(-1, 1, GRID, GRID)
            sm = F.conv2d(g, k, padding=1).reshape(bg.shape[2], -1).T
            new = todo[s] & (cnt > 0)
            bg[s][new] = sm[new] / cnt[new, None]
            todo[s] &= ~new
    return bg


def input_background(b):
    """The background used for the background-subtracted SAE input: the rule background with leaked positions
    (leak_mask) filled from clean neighbours (fill_leaks), cached in the load_background dict b (b['leak'] too)."""
    if 'bg_input' not in b:
        bg_dir, obs, rule = b['src']
        if rule['background'] == 'bg_median':
            raise ValueError('background subtraction needs a masked background (bg_masked / bg_seg)')
        z = np.load(Path(bg_dir) / f'{obs}.npz')
        leak = leak_mask(z['dark'], z['n_ok'], b['seg_bounds'] if rule['background'] == 'bg_seg' else None, rule)
        b['leak'] = torch.from_numpy(leak).to(b['bg'].device)
        b['bg_input'] = fill_leaks(b['bg'], b['leak'])
    return b['bg_input']


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
    span several videos (rows sorted or not).
    align: the frame alignment the caller's frames have; every background file read must record the same
    (npz 'align', absent = 'none'), else it raises (aligned frames against unaligned backgrounds or vice versa)."""

    def __init__(self, bg_dir, obs_ranges, rule=FG_RULE, device='cuda', cache=4, align='none'):
        self.bg_dir, self.rule, self.device, self.cache_n = Path(bg_dir), rule, torch.device(device), cache
        self.align = align
        items = sorted(obs_ranges.items(), key=lambda kv: kv[1][0])
        self.ids = [k for k, _ in items]
        self.starts = np.array([v[0] for _, v in items], dtype=np.int64)
        self.ends = np.array([v[1] for _, v in items], dtype=np.int64)
        self._cache = {}

    def get(self, obs):
        if obs not in self._cache:
            if len(self._cache) >= self.cache_n:
                self._cache.pop(next(iter(self._cache)))
            b = load_background(self.bg_dir, obs, self.rule, self.device)
            if b['align'] != self.align:
                raise RuntimeError(f'{self.bg_dir}/{obs}.npz: background align {b["align"]!r}, frames {self.align!r}')
            self._cache[obs] = b
        return self._cache[obs]

    def obs_index(self, rows):
        k = np.searchsorted(self.starts, np.asarray(rows), side='right') - 1
        assert (np.asarray(rows) < self.ends[k]).all()
        return k

    @torch.no_grad()
    def mask(self, tokens, grey, rows):
        """tokens (B, 1024, d) on device, grey (B, 512, 512) uint8 on device, rows (B,) numpy.
        -> (mask (B, 1024) bool, dist (B, 1024) float)."""
        if self.rule.get('all', False):  # whole frame: no background is read (dist = 0)
            return (torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device),
                    torch.zeros(tokens.shape[:2], dtype=torch.float32, device=tokens.device))
        rows = np.asarray(rows)
        k = self.obs_index(rows)
        out = torch.zeros(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        dist = torch.zeros(tokens.shape[:2], dtype=torch.float32, device=tokens.device)
        if self.rule.get('frogs', False):  # pixel cue only (FG_RULE_FROGS); dist stays 0
            for kk in np.unique(k):
                sel = np.nonzero(k == kk)[0]
                b = self.get(self.ids[kk])
                sel_t = torch.from_numpy(sel).to(tokens.device)
                nodes = b['nodes'][torch.from_numpy(rows[sel] - b['lo']).to(b['nodes'].device)].to(tokens.device)
                px = frog_pixels(grey[sel_t], b['pix_bg'].to(tokens.device), b['roi'].to(tokens.device), nodes, self.rule)
                dk = F.avg_pool2d(px.float()[:, None], PATCH_PX).flatten(1)
                out[sel_t] = foreground_mask(dist[sel_t], dk, float('inf'), self.rule)
            return out, dist
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

    @torch.no_grad()
    def background(self, rows):
        """(B, 1024, d) float32 background tokens of a batch of rows for the background-subtracted input: the rule's
        background (bg_seg: the video's empty-arena median token at each position, in the frame's time block), leaked
        positions filled from clean neighbours (input_background)."""
        if self.rule.get('frogs', False):
            raise ValueError("rule 'frogs' has no background tokens (pixel cue only)")
        rows = np.asarray(rows)
        k = self.obs_index(rows)
        out = None
        for kk in np.unique(k):
            sel = np.nonzero(k == kk)[0]
            b = self.get(self.ids[kk])
            seg = segment_of(rows[sel], b['rows'], b['seg_bounds'])
            bg = input_background(b)[torch.from_numpy(seg).to(b['bg'].device)]
            if out is None:
                out = torch.empty((len(rows),) + tuple(bg.shape[1:]), dtype=torch.float32, device=bg.device)
            out[torch.from_numpy(sel).to(bg.device)] = bg
        return out


def subtract_background_tokens(tokens, bg):
    """Background-subtracted SAE input: fp16 token - float32 background token, in float32, rounded to fp16
    (the same arithmetic for the training store and for encoding). tokens / bg: torch tensors or numpy."""
    if isinstance(tokens, np.ndarray):
        return (tokens.astype(np.float32) - bg).astype(np.float16)
    return (tokens.float() - bg).half()


def subtract_background_np(tokens, rows, pos, bg_dir, obs_ranges, rule, chunk=1_000_000):
    """numpy (n, d) fp16 tokens with their annotations.csv rows and patch positions -> (n, d) fp16
    token - background token of the same video, time block and patch position (FgBackgrounds.background)."""
    rows, pos = np.asarray(rows), np.asarray(pos).astype(np.int64)
    bgs = FgBackgrounds(bg_dir, obs_ranges, rule, 'cpu', cache=1)
    out = np.empty(tokens.shape, np.float16)
    k = bgs.obs_index(rows)
    for kk in np.unique(k):
        sel = np.nonzero(k == kk)[0]
        b = load_background(bg_dir, bgs.ids[kk], rule)
        seg = segment_of(rows[sel], b['rows'], b['seg_bounds'])
        bg = input_background(b).numpy()
        for a in range(0, len(sel), chunk):
            s = sel[a:a + chunk]
            out[s] = subtract_background_tokens(np.asarray(tokens[s]), bg[seg[a:a + chunk], pos[s]])
    return out


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

    def prev(self, s):
        """Tokens of the same patches Delta frames earlier (stores written with --motion-delta)."""
        return np.memmap(self.dirs[s] / 'prev.f16', dtype=np.float16, mode='r', shape=(int(self.sizes[s]), self.dim))

    def frames(self):
        z = [np.load(d / 'frames.npz') for d in self.dirs]
        return np.concatenate([a['rows'] for a in z]), np.concatenate([a['n_fg'] for a in z])
