"""
Per-video backgrounds for the foreground ("mouse patches") ECI pipeline (src/eci/foreground.py).

For each observation: n_bg frames evenly spread over the video -> DINOv2-base on the whole
frame at 448 (no crop) -> per-position median token (bg_median), pixel background (per-pixel
0.9 quantile of grey), dark-pixel cue, masked median token (bg_masked, frames where the patch
is dark-foreground excluded), and the cosine distances of the n_bg frames to both backgrounds
(used for the per-video threshold and for validation). No behaviour annotation is used.

Output: dataset/mice/v1/eci/fg448/background/{observation_id}.npz with
    rows (n_bg,) int64        rows of annotations.csv
    bg_median, bg_masked      (1024, 768) float16
    n_ok (1024,) int16        frames used per position by bg_masked
    pix_bg (512, 512) uint8   per-pixel --pix-q quantile of grey
    bg_seg (n_seg, 1024, 768) float16   masked median per time block (seg_bounds: sample indices)
    dist_bg_median, dist_bg_masked, dist_bg_seg, dark   (n_bg, 1024) float16
Existing files are skipped (resumable).

Usage:
    python scripts/eci/fg_background.py --task 0 --n-tasks 8
    python scripts/eci/fg_background.py --obs wt_ash1l_m_1_S_H,...
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.eci.foreground import (FG_RULE, FrameDatasetFG, cosine_distance, dark_fraction, dilate,  # noqa: E402
                                encode_batch, load_encoder_fg, obs_rows, patch_background,
                                pixel_background)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset-dir', default=str(REPO / 'dataset'))
    p.add_argument('--out-dir', default=str(REPO / 'dataset/mice/v1/eci/fg448/background'))
    p.add_argument('--n-bg', type=int, default=200)
    p.add_argument('--task', type=int, default=0)
    p.add_argument('--n-tasks', type=int, default=1)
    p.add_argument('--obs', default=None, help='comma list of observation ids (overrides --task)')
    p.add_argument('--n-seg', type=int, default=4, help='time blocks for bg_seg')
    p.add_argument('--min-seg-frames', type=int, default=10)
    p.add_argument('--pix-q', type=float, default=0.98, help='quantile for the pixel background')
    p.add_argument('--batch-size', type=int, default=50)
    p.add_argument('--num-workers', type=int, default=16)
    args = p.parse_args()

    ds, out = Path(args.dataset_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ann = ds / 'mice/v1/annotations.csv'
    ranges = obs_rows(ann)
    ids = sorted(ranges)
    ids = args.obs.split(',') if args.obs else ids[args.task::args.n_tasks]
    ids = [o for o in ids if not (out / f'{o}.npz').exists()]
    print(f'{len(ids)} observations to do', flush=True)
    if not ids:
        return
    paths_all = pd.read_csv(ann, usecols=['frame_path'])['frame_path'].values

    rows, owner = [], []
    for o in ids:
        lo, hi = ranges[o]
        r = np.unique(np.round(np.linspace(lo, hi - 1, args.n_bg)).astype(np.int64))
        assert len(r) == args.n_bg, (o, len(r))
        rows.append(r)
        owner += [o] * len(r)
    rows = np.concatenate(rows)

    device = torch.device('cuda')
    _, processor, model = load_encoder_fg(device=device)
    loader = torch.utils.data.DataLoader(
        FrameDatasetFG([str(ds / pth) for pth in paths_all[rows]], processor), batch_size=args.batch_size,
        num_workers=args.num_workers, shuffle=False, pin_memory=True, prefetch_factor=4)

    t0, cur, buf_t, buf_g = time.time(), 0, [], []
    for pix, grey in loader:
        buf_t.append(encode_batch(model, pix, device))
        buf_g.append(grey.to(device))
        cur += pix.shape[0]
        n_buf = sum(b.shape[0] for b in buf_t)
        if n_buf < args.n_bg:
            continue
        assert n_buf == args.n_bg
        o = owner[cur - 1]
        assert owner[cur - args.n_bg] == o
        tok, gr = torch.cat(buf_t), torch.cat(buf_g)
        buf_t, buf_g = [], []
        pix_bg = pixel_background(gr, args.pix_q)
        dark = dark_fraction(gr, pix_bg, FG_RULE['dark_abs'], FG_RULE['dark_rel'])
        excl = dilate(dark > FG_RULE['dark_frac'], 1)
        bg_med, _ = patch_background(tok)
        bg_msk, n_ok = patch_background(tok, excl)
        d_med, d_msk = cosine_distance(tok, bg_med), cosine_distance(tok, bg_msk)
        # time-local backgrounds: n_seg equal blocks of the sample, masked median per block
        seg = np.array_split(np.arange(args.n_bg), args.n_seg)
        bg_seg, d_seg = [], torch.empty_like(d_med)
        for s_ in seg:
            b, n_s = patch_background(tok[s_], excl[s_], min_frames=args.min_seg_frames)
            few = n_s < args.min_seg_frames
            b[few] = bg_msk[few]
            bg_seg.append(b)
            d_seg[s_] = cosine_distance(tok[s_], b)
        bg_seg = torch.stack(bg_seg)
        r = rows[cur - args.n_bg:cur]
        tmp = out / f'{o}.tmp.npz'
        np.savez(tmp, rows=r, bg_median=bg_med.half().cpu().numpy(), bg_masked=bg_msk.half().cpu().numpy(),
                 n_ok=n_ok.short().cpu().numpy(), pix_bg=pix_bg.cpu().numpy(),
                 dist_bg_median=d_med.half().cpu().numpy(), dist_bg_masked=d_msk.half().cpu().numpy(),
                 dark=dark.half().cpu().numpy(), bg_seg=bg_seg.half().cpu().numpy(),
                 dist_bg_seg=d_seg.half().cpu().numpy(), seg_bounds=np.array([s_[0] for s_ in seg] + [args.n_bg]))
        tmp.rename(out / f'{o}.npz')
        print(f'  {o}: dist med {float(d_med.median()):.3f}/{float(d_msk.median()):.3f}  '
              f'dark>{FG_RULE["dark_frac"]} {float((dark > FG_RULE["dark_frac"]).float().mean()):.3f}  '
              f'min n_ok {int(n_ok.min())}  {cur}/{len(rows)} frames {cur / (time.time() - t0):.1f} f/s', flush=True)
    assert cur == len(rows)
    print(f'Done {len(ids)} observations in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
