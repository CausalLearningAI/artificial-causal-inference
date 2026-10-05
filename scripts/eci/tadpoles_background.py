"""
Per-video backgrounds of the tadpole crop mask (src/eci/foreground.py rule 'tadpoles', FG_RULE_TADPOLES), assembled
from scripts/eci/tadpoles_crops.py outputs once the tables exist (frogs_prepare.py --stage 44-48 --step tables).

Output: dataset/frogs/eci_tadpole/fg448/background/{observation_id}.npz
    pix_bg (h + 2 pad, w + 2 pad) uint8   tadpole-free background of the full source frame, edge-padded by pad
    roi (h + 2 pad, w + 2 pad) bool       dish ROI, padded with False
    x0, y0 (n_rows,) int32                crop box top-left (source px) of every row of the video, side, pad (= side / 2)
    nodes (n_rows, 11, 2) float16         SLEAP nodes in crop coordinates (512 px; NaN = not predicted)
    rows (n_bg,) int64                    annotations.csv rows of the background sample frames
    lo int                                the video's first annotations.csv row
    status (n_rows,) uint8                crop track status (0 tracked, 1 interpolated, 2 held)
Existing files are skipped. Usage: python scripts/eci/tadpoles_background.py [--obs WT_100_1,...]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.eci.domain import get_domain  # noqa: E402
from src.eci.foreground import FRAME_PX, align_tag, obs_rows  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--obs', default=None)
    args = p.parse_args()
    D = get_domain('tadpoles')
    out = D.eci_dir / align_tag('none') / 'background'
    out.mkdir(parents=True, exist_ok=True)
    ranges = obs_rows(D.ann_path)
    fi_all = pd.read_csv(D.ann_path, usecols=['frame_idx'])['frame_idx'].values
    ids = args.obs.split(',') if args.obs else list(ranges)
    for o in ids:
        f = out / f'{o}.npz'
        if f.exists():
            continue
        lo, hi = ranges[o]
        c = np.load(REPO / 'dataset/frogs/v1_tadpole/crops' / f'{o}.npz')
        z = np.load(REPO / 'dataset/frogs/v1_tadpole/sleap' / f'{o}.npz')
        n = hi - lo
        n_all = len(c['x0'])  # < n_all only in a test with frogs_prepare.py --max-frames (TADPOLE_TAG set)
        if (n != n_all and not D.TAG) or n > n_all or len(z['nodes']) != n_all \
                or not np.array_equal(fi_all[lo:hi], np.arange(n)):
            raise SystemExit(f'{o}: {n} rows, {len(c["x0"])} crop boxes, {len(z["nodes"])} SLEAP frames')
        side = int(c['side'])
        pad = side // 2
        x0, y0 = c['x0'][:n].astype(np.int32), c['y0'][:n].astype(np.int32)
        nodes = (z['nodes'][:n] - np.stack([x0, y0], 1)[:, None].astype(np.float32)) * (FRAME_PX / side)
        tmp = out / f'{o}.tmp.npz'
        np.savez(tmp, pix_bg=np.pad(c['pix_bg'], pad, mode='edge'), roi=np.pad(c['roi'], pad, constant_values=False),
                 x0=x0, y0=y0, side=side, pad=pad, nodes=nodes.astype(np.float16), rows=lo + c['bg_idx'][c['bg_idx'] < n], lo=lo,
                 status=c['status'][:n])
        tmp.rename(f)
        print(f'  {o}: rows {n}, crop status tracked {np.mean(c["status"] == 0):.4f}', flush=True)
    print(f'Done {len(ids)} observations -> {out}')


if __name__ == '__main__':
    main()
