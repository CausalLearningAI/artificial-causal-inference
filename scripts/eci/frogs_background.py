"""
Per-video backgrounds of the frog foreground mask (src/eci/foreground.py rule 'frogs', FG_RULE_FROGS).

For each frogs video: n_bg frames evenly spread over the video (5 fps frames of dataset/frogs/eci/annotations.csv,
grey as src/eci/foreground.py FrameDatasetFG reads it) -> every pixel within r_ex px of a SLEAP node of its frame is
left out (the frog itself, limbs included; a frame without any predicted node is left out entirely) -> per-pixel
pix_q quantile of the remaining grey values = the frog-free pixel background. Pixels with fewer than min_free
frog-free samples (a frog that never left a spot) are filled ring by ring with the mean of their filled 8-neighbours.
No DINOv2 pass, no behaviour annotation.

Output: dataset/frogs/eci/fg448/background/{observation_id}.npz
    rows (n_bg,) int64          annotations.csv rows of the sample frames
    pix_bg (512, 512) uint8     frog-free pixel background
    n_ok (512, 512) int16       frog-free samples per pixel (before filling)
    n_filled int                pixels filled from neighbours
    roi (512, 512) bool         the dish (ImageJ oval of data/frogs/v1/experiment.csv, scaled 800 -> 512 px)
    nodes (n_rows, K, 2) float16  SLEAP nodes (x, y at 512 px, NaN = not predicted) of EVERY row of the video
                                (dataset/frogs/v1/sleap/<id>.npz, scripts/eci/frogs_sleap.py), indexed by row - lo;
                                mirrored (x and Left_ / Right_ names, mirror_nodes) for hflip videos, like the frames
    lo int                      the video's first annotations.csv row
Existing files are skipped (resumable).

Usage: python scripts/eci/frogs_background.py [--obs WT_157_1,...]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.eci.domain import get_domain  # noqa: E402
from src.eci.foreground import FRAME_PX, align_tag, near_nodes, obs_rows  # noqa: E402

SRC_PX = 800  # source frame side; frames are 512 x 512


def roi_mask(r):
    """(512, 512) bool: pixel centres inside the dish oval (experiment.csv roi_* in 800 px, of the raw video; mirrored
    when hflip = 1)."""
    s = FRAME_PX / SRC_PX
    c = np.arange(FRAME_PX) + 0.5
    cx = (r.roi_left + r.roi_right) / 2
    cx = SRC_PX - cx if r.hflip == 1 else cx
    cy, cx = (r.roi_top + r.roi_bottom) / 2 * s, cx * s
    ay, ax = (r.roi_bottom - r.roi_top) / 2 * s, (r.roi_right - r.roi_left) / 2 * s
    return ((c[:, None] - cy) / ay) ** 2 + ((c[None, :] - cx) / ax) ** 2 <= 1


def mirror_nodes(nodes, names):
    """SLEAP nodes (n, K, 2) of a video mirrored left-right at standardization (experiment.csv hflip = 1), in 800 px
    source coordinates: x -> SRC_PX - 1 - x (ffmpeg hflip maps column c to W - 1 - c) and every Left_* / Right_* node
    swapped with its partner (the frog's left side is now on its right)."""
    names = [str(n) for n in names]
    out = nodes.copy()
    out[..., 0] = SRC_PX - 1 - out[..., 0]
    swap = [names.index(n.replace('Left_', 'Right_') if n.startswith('Left_') else n.replace('Right_', 'Left_'))
            for n in names]
    return out[:, swap]


def fill_pixels(bg, todo):
    """bg (H, W) float tensor, todo (H, W) bool -> bg with todo pixels filled ring by ring from filled 8-neighbours."""
    bg, todo = bg.clone(), todo.clone()
    k = torch.ones(1, 1, 3, 3)
    while todo.any():
        ok = (~todo).float()[None, None]
        cnt = F.conv2d(ok, k, padding=1)[0, 0]
        sm = F.conv2d((bg * ~todo)[None, None], k, padding=1)[0, 0]
        new = todo & (cnt > 0)
        if not new.any():
            raise RuntimeError('fill_pixels: nothing to fill from')
        bg[new] = sm[new] / cnt[new]
        todo &= ~new
    return bg


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset-dir', default=str(REPO / 'dataset'))
    p.add_argument('--out-dir', default=None, help='default <dataset dir>/frogs/eci/fg448/background')
    p.add_argument('--obs', default=None, help='comma list of observation ids (default all)')
    p.add_argument('--n-bg', type=int, default=200)
    p.add_argument('--pix-q', type=float, default=0.9, help='quantile of the frog-free grey values')
    p.add_argument('--r-ex', type=float, default=22, help='pixels within this distance (512 px) of a node are frog')
    p.add_argument('--min-free', type=int, default=5, help='fewer frog-free samples -> filled from neighbours')
    args = p.parse_args()

    dom = get_domain('frogs')
    ds = Path(args.dataset_dir)
    out = Path(args.out_dir) if args.out_dir else ds / dom.eci_rel / align_tag('none') / 'background'
    out.mkdir(parents=True, exist_ok=True)
    ann = ds / dom.ann_rel
    ranges = obs_rows(ann)
    frame_idx = pd.read_csv(ann, usecols=['frame_idx'])['frame_idx'].values
    paths = pd.read_csv(ann, usecols=['frame_path'])['frame_path'].values
    exp = pd.read_csv(REPO / 'data/frogs/v1/experiment.csv').set_index('observation_id')
    ids = args.obs.split(',') if args.obs else list(ranges)
    s = FRAME_PX / SRC_PX
    t0 = time.time()
    for o in ids:
        f = out / f'{o}.npz'
        if f.exists():
            continue
        lo, hi = ranges[o]
        z = np.load(ds / 'frogs/v1/sleap' / f'{o}.npz')
        sl = z['nodes']
        if exp.loc[o, 'hflip'] == 1:
            sl = mirror_nodes(sl, z['node_names'])
        sl = sl * s
        fi = frame_idx[lo:hi]
        if abs(len(sl) - len(fi)) > 1 or fi[-1] > len(sl):
            raise SystemExit(f'{o}: {len(fi)} frames vs {len(sl)} SLEAP frames')
        nodes = np.full((hi - lo,) + sl.shape[1:], np.nan, np.float32)
        ok = fi < len(sl)
        nodes[ok] = sl[fi[ok]]
        rows = np.unique(np.round(np.linspace(lo, hi - 1, args.n_bg)).astype(np.int64))
        assert len(rows) == args.n_bg, (o, len(rows))
        grey = np.stack([np.asarray(Image.open(ds / paths[r]).convert('RGB').convert('L')) for r in rows])
        if grey.shape[1:] != (FRAME_PX, FRAME_PX):
            raise SystemExit(f'{o}: frames {grey.shape[1:]}')
        frog = near_nodes(torch.from_numpy(nodes[rows - lo]), args.r_ex)
        frog[~torch.from_numpy(np.isfinite(nodes[rows - lo, :, 0]).any(1))] = True  # no node: the frame is unused
        g = torch.from_numpy(grey).float().masked_fill(frog, float('nan'))
        n_ok = (~frog).sum(0)
        bg = torch.nanquantile(g.flatten(1), args.pix_q, dim=0).view(FRAME_PX, FRAME_PX)
        todo = n_ok < args.min_free
        bg = fill_pixels(torch.nan_to_num(bg), todo)
        roi = roi_mask(exp.loc[o])
        tmp = out / f'{o}.tmp.npz'
        np.savez(tmp, rows=rows, pix_bg=bg.round().clamp(0, 255).to(torch.uint8).numpy(), n_ok=n_ok.short().numpy(),
                 n_filled=int(todo.sum()), roi=roi, nodes=nodes.astype(np.float16), lo=lo)
        tmp.rename(f)
        print(f'  {o}: rows {hi - lo}, nodes missing in {np.isnan(nodes[..., 0]).all(1).mean():.4f} of frames, '
              f'min frog-free samples in the dish {int(n_ok[torch.from_numpy(roi)].min())}, filled {int(todo.sum())} px, '
              f'background median in the dish {float(bg[torch.from_numpy(roi)].median()):.0f}  '
              f'{time.time() - t0:.0f}s', flush=True)
    print(f'Done {len(ids)} observations -> {out}')


if __name__ == '__main__':
    main()
