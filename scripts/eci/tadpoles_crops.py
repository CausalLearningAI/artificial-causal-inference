"""
Body crops of ONE tadpole video (frogs stage 44-48, src/eci/domain.py TadpolesDomain; SLURM array task = row of the
valid rows of data/frogs/v1_tadpole/experiment.csv).

A stage 44-48 tadpole is ~38 px long (median over videos of the median eyes -> tail-tip skeleton length, range 22-60,
scripts/eci/frogs_sleap.py --stage 44-48 quality.csv) in a 740-800 px frame whose dish is ~640 px wide (all videos:
same px / mm). The whole frame at 448 would give the tadpole ~2 x 1 DINOv2 patches, so each 5 fps frame is replaced by
a FIXED-SIZE square crop centred on the tracked body, NOT rotated (orientation kept as filmed: no body-axis estimate
to get wrong when nodes are missing, and heading stays visible to the encoder), resized to 512 px:

    SIDE = 96 source px = 2.5 median body lengths. Measured: the farthest SLEAP node from the node centroid is
           <= 35.2 px in 99.9% of the frames with all 11 nodes (2.47M frames, 173 videos) and <= 40.9 px in the
           largest-tadpole video (Scrambled_467_5, 99.9th pct), so half the side (48) leaves >= 7 px for the body
           outline beyond the midline nodes.
    centre = scripts/eci/frogs_sleap.py 'centre' (mean over source frames 12 i + 3 .. 12 i + 7 of the centroid of the
           predicted nodes of frames with >= 6 of 11 nodes; +-33 ms, so no lag on fast swims). Frames without it:
           gaps of <= GAP_INTERP frames (1 s) linearly interpolated (status 1); longer gaps take the nearest tracked
           frame (status 2: the tadpole is usually invisible then, hidden in the dark dish rim where it was last
           seen); a video with no tracked frame at all fails.
    The frame is edge-padded by SIDE / 2, so a crop at the frame border keeps its size and centre.

Frames: ffmpeg decodes the raw 60 fps video as grey and keeps exactly source frames 5, 17, 29, ... (select
mod(n, 12) = 5, the frames of the SLEAP 5 fps tables; checked: the count equals ceil((source_frames - 5) / 12)).
Output:
    dataset/frogs/v1_tadpole/frames/crop/<id>/frame_%06d.jpg   512 x 512 grey JPEG (quality 90), DONE when complete
    dataset/frogs/v1_tadpole/crops/<id>.npz   x0, y0 (n5,) int32 top-left of the crop in source px (may be < 0),
                                              side, out_px, status (n5,) uint8 (0 tracked, 1 interpolated, 2 held),
                                              centre (n5, 2) float32 (the raw SLEAP centre, NaN where missing),
                                              and the tadpole-free pixel background of the FULL source frame, from
                                              the same decoding pass (as scripts/eci/frogs_background.py, at source
                                              px): bg_idx (N_BG,) 5 fps frames used, pix_bg (h, w) uint8 = per-pixel
                                              BG_Q quantile of the grey values of the N_BG frames, leaving out pixels
                                              within R_EX source px of a SLEAP node of that frame (frames without
                                              any node left out entirely), pixels with < MIN_FREE samples filled
                                              from neighbours; n_ok (h, w) int16; n_filled; roi (h, w) bool (pixel
                                              centres inside the ImageJ oval of experiment.csv); dark_px (n5,) int32
                                              tadpole pixels in the crop (source px, DARK_REL), present = the
                                              fraction of frames with dark_px >= MIN_DARK_PX
--quality-only: decode and write the crops npz (track, background, dark_px / present) but no JPEG frames (cheap, no
disk): the presence numbers for the exclusion rule (frogs_prepare.py --stage 44-48 --step exclude) before the crops.
Finished outputs are skipped. Usage: python scripts/eci/tadpoles_crops.py --task 3 [--obs WT_100_1] [--quality-only]
"""
import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
SIDE, OUT_PX, GAP_INTERP, QUALITY = 96, 512, 5, 90
N_BG, BG_Q, R_EX, MIN_FREE = 200, 0.9, 10.0, 5  # background: frames, quantile, node exclusion radius (source px)
# per-frame presence check: tadpole pixels = darker than the background by > DARK_REL inside the dish (the mask rule's
# dark_rel, src/eci/foreground.py FG_RULE_TADPOLES, without the near-node restriction); a frame shows the tadpole
# when it has >= MIN_DARK_PX of them (a tadpole covers ~150-300 source px)
DARK_REL, MIN_DARK_PX = 15, 20
STEP, OFFSET = 12, 5
CROPS = REPO / 'dataset/frogs/v1_tadpole/crops'
FRAMES = REPO / 'dataset/frogs/v1_tadpole/frames/crop'


def crop_track(centre, gap_interp=GAP_INTERP):
    """centre (n, 2) float, NaN = untracked -> (xy (n, 2) float, status (n,) uint8: 0 tracked, 1 interpolated (gap of
    <= gap_interp frames between two tracked frames), 2 nearest tracked frame (longer gap, or before the first /
    after the last tracked frame))."""
    ok = np.isfinite(centre).all(1)
    if not ok.any():
        raise ValueError('no tracked frame')
    idx = np.arange(len(centre))
    good = idx[ok]
    a = good[np.clip(np.searchsorted(good, idx, side='right') - 1, 0, len(good) - 1)]  # last tracked <= i (or first)
    b = good[np.clip(np.searchsorted(good, idx, side='left'), 0, len(good) - 1)]       # first tracked >= i (or last)
    xy = centre.astype(np.float64).copy()
    status = np.zeros(len(centre), np.uint8)
    short = ~ok & (a < idx) & (b > idx) & (b - a - 1 <= gap_interp)
    w = (idx - a) / np.maximum(b - a, 1)
    xy[short] = (1 - w[short, None]) * centre[a[short]] + w[short, None] * centre[b[short]]
    status[short] = 1
    rest = ~ok & ~short
    near = np.where(np.abs(idx - a) <= np.abs(b - idx), a, b)  # a == b outside the tracked span
    xy[rest] = centre[near[rest]]
    status[rest] = 2
    return xy, status


def roi_mask(r):
    """(h, w) bool: source pixel centres inside the dish oval (ImageJ roi bounding box of experiment.csv)."""
    h, w = int(r['height']), int(r['width'])
    t, l, b, rr = (float(r[k]) for k in ('roi_top', 'roi_left', 'roi_bottom', 'roi_right'))
    cy, cx, ay, ax = (t + b) / 2, (l + rr) / 2, (b - t) / 2, (rr - l) / 2
    yy, xx = np.arange(h) + 0.5, np.arange(w) + 0.5
    return ((yy[:, None] - cy) / ay) ** 2 + ((xx[None] - cx) / ax) ** 2 <= 1


def background(grey, nodes, r_ex=R_EX, q=BG_Q, min_free=MIN_FREE):
    """grey (N, h, w) uint8 sample frames, nodes (N, K, 2) source px (NaN = not predicted) -> (pix_bg (h, w) uint8,
    n_ok (h, w) int16, n_filled): per-pixel q quantile of the samples not within r_ex of a node of their frame."""
    import torch
    sys.path.insert(0, str(REPO / 'scripts/eci'))
    from frogs_background import fill_pixels
    N, h, w = grey.shape
    yy, xx = np.arange(h)[:, None] + 0.5, np.arange(w)[None] + 0.5
    g = grey.astype(np.float32)
    for i in range(N):
        ok = np.isfinite(nodes[i, :, 0])
        if not ok.any():
            g[i] = np.nan
            continue
        near = np.zeros((h, w), bool)
        for x, y in nodes[i, ok]:
            x0, x1 = int(max(0, x - r_ex - 1)), int(min(w, x + r_ex + 2))
            y0, y1 = int(max(0, y - r_ex - 1)), int(min(h, y + r_ex + 2))
            near[y0:y1, x0:x1] |= (yy[y0:y1] - y) ** 2 + (xx[:, x0:x1] - x) ** 2 < r_ex * r_ex
        g[i][near] = np.nan
    n_ok = np.isfinite(g).sum(0).astype(np.int16)
    gt = torch.from_numpy(g.reshape(N, -1))
    bg = torch.cat([torch.nanquantile(gt[:, c:c + 60000], q, dim=0) for c in range(0, gt.shape[1], 60000)])
    bg = np.nan_to_num(bg.view(h, w).numpy())
    todo = n_ok < min_free
    bg = fill_pixels(torch.from_numpy(bg), torch.from_numpy(todo)).numpy()
    return np.clip(np.round(bg), 0, 255).astype(np.uint8), n_ok, int(todo.sum())


def decode(src, w, h, n_expected, max_frames=None):
    """Yield the grey (h, w) uint8 frames 5, 17, 29, ... of the raw video (testing: only the first max_frames)."""
    lim = [] if max_frames is None else ['-frames:v', str(max_frames)]
    cmd = ['ffmpeg', '-v', 'error', '-threads', '4', '-i', str(src), '-vf', f'select=eq(mod(n\\,{STEP})\\,{OFFSET})',
           '-vsync', '0'] + lim + ['-f', 'rawvideo', '-pix_fmt', 'gray', '-']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=w * h * 8)
    k = 0
    while True:
        b = p.stdout.read(w * h)
        if len(b) < w * h:
            break
        yield np.frombuffer(b, np.uint8).reshape(h, w)
        k += 1
    if p.wait() != 0:
        raise SystemExit(f'{src}: ffmpeg failed')
    if k != (n_expected if max_frames is None else min(max_frames, n_expected)):
        raise SystemExit(f'{src}: decoded {k} frames, expected {n_expected}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', type=int, default=None)
    p.add_argument('--obs', default=None, help='observation id instead of --task')
    p.add_argument('--max-frames', type=int, default=None, help='testing: stop after this many 5 fps frames')
    p.add_argument('--quality-only', action='store_true', help='crops npz only, no JPEG frames')
    args = p.parse_args()
    rows = [r for r in csv.DictReader(open(REPO / 'data/frogs/v1_tadpole/experiment.csv')) if r['valid'] == '1']
    r = [x for x in rows if x['observation_id'] == args.obs][0] if args.obs else rows[args.task]
    o = r['observation_id']
    out = FRAMES / o
    done = out / 'DONE'
    if done.exists() or (args.quality_only and (CROPS / f'{o}.npz').exists()):
        print(f'[SKIP] {o}: {done if done.exists() else CROPS / f"{o}.npz"}')
        return
    t0 = time.time()
    w, h, n_src = int(r['width']), int(r['height']), int(r['source_frames'])
    n5 = (n_src - OFFSET + STEP - 1) // STEP
    z = np.load(REPO / 'dataset/frogs/v1_tadpole/sleap' / f'{o}.npz')
    centre = z['centre']
    if len(centre) != n5:
        raise SystemExit(f'{o}: {len(centre)} SLEAP centres, expected {n5}')
    xy, status = crop_track(centre)
    half = SIDE // 2
    cx, cy = np.round(xy[:, 0]).astype(np.int32), np.round(xy[:, 1]).astype(np.int32)
    x0, y0 = cx - half, cy - half
    n_dec = n5 if args.max_frames is None else min(n5, args.max_frames)
    bg_idx = np.unique(np.round(np.linspace(0, n_dec - 1, N_BG)).astype(np.int64))
    samples = np.empty((len(bg_idx), h, w), np.uint8)
    raw = np.empty((n_dec, SIDE, SIDE), np.uint8)
    if not args.quality_only:
        out.mkdir(parents=True, exist_ok=True)
    k, j = 0, 0
    frames = decode(r['source_file'], w, h, n5, args.max_frames)
    for i, fr in enumerate(frames):
        if j < len(bg_idx) and bg_idx[j] == i:
            samples[j] = fr
            j += 1
        pad = np.pad(fr, half, mode='edge')
        c = pad[y0[i] + half:y0[i] + half + SIDE, x0[i] + half:x0[i] + half + SIDE]
        raw[i] = c
        k += 1
        if args.quality_only:
            continue
        Image.fromarray(c).resize((OUT_PX, OUT_PX), Image.BICUBIC).save(out / f'frame_{i:06d}.jpg', quality=QUALITY)
    pix_bg, n_ok, n_filled = background(samples, z['nodes'][bg_idx])
    roi = roi_mask(r)
    pbg, proi = np.pad(pix_bg, half, mode='edge').astype(np.int16), np.pad(roi, half, constant_values=False)
    dark_px = np.empty(k, np.int32)
    for i in range(k):
        sl = (slice(y0[i] + half, y0[i] + half + SIDE), slice(x0[i] + half, x0[i] + half + SIDE))
        dark_px[i] = (((pbg[sl] - raw[i]) > DARK_REL) & proi[sl]).sum()
    CROPS.mkdir(parents=True, exist_ok=True)
    tmp = CROPS / f'{o}.tmp.npz'
    np.savez(tmp, x0=x0, y0=y0, side=SIDE, out_px=OUT_PX, status=status, centre=centre, bg_idx=bg_idx,
             pix_bg=pix_bg, n_ok=n_ok, n_filled=n_filled, roi=roi, dark_px=dark_px,
             present=float(np.mean(dark_px >= MIN_DARK_PX)))
    tmp.rename(CROPS / f'{o}.npz')
    if args.max_frames is None and not args.quality_only:
        done.touch()
    print(f'{o}: {k} crops, status tracked {np.mean(status == 0):.4f} / interpolated {np.mean(status == 1):.4f} / '
          f'held {np.mean(status == 2):.4f}; tadpole present (>= {MIN_DARK_PX} dark px) in {np.mean(dark_px >= MIN_DARK_PX):.4f} '
          f'of the crops (tracked {np.mean(dark_px[status[:k] == 0] >= MIN_DARK_PX):.4f}, held '
          f'{np.mean(dark_px[status[:k] == 2] >= MIN_DARK_PX) if (status[:k] == 2).any() else float("nan"):.4f}); '
          f'background filled {n_filled} px, min samples in the dish {int(n_ok[roi].min())}; {time.time() - t0:.0f}s',
          flush=True)


if __name__ == '__main__':
    main()
