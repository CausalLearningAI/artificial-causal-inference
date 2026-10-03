"""
SLEAP poses of the frogs v1 videos at the 5 fps frames (one npz per video), for the frog foreground mask
(src/eci/foreground.py rule 'frogs', scripts/eci/frogs_background.py).

Input: the re-predicted SLEAP analysis files (data/frogs/v1/experiment.csv 'sleap_h5'; one model for every video),
tracks (1 track, xy, 22 nodes, every 60 fps frame) in 800 px source coordinates.
Output: dataset/frogs/v1/sleap/<observation_id>.npz
    nodes        (n5, 22, 2) float32  x, y of every node at source frames 5, 17, 29, ... (the 5 fps frames of
                                      dataset/frogs/v1/frames; NaN = node not predicted)
    node_names   (22,) str
    source_frames  int                60 fps frames of the video (checked equal to experiment.csv)

Run with a Python that has h5py (the crl environment has none; /usr/bin/python3 does). numpy + h5py only.
Usage: /usr/bin/python3 scripts/eci/frogs_sleap.py [--overwrite]
"""

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# 60 fps source -> 5 fps frames: frame i of the standardized video (ffmpeg fps=5) is source frame 12 i + 5 (measured:
# the extracted frames match source frame 12 i + 5 best among offsets -12..12 whenever the frog moves)
STEP, OFFSET = 12, 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()
    out = ROOT / 'dataset/frogs/v1/sleap'
    out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(ROOT / 'data/frogs/v1/experiment.csv')))
    for r in rows:
        f = out / f'{r["observation_id"]}.npz'
        if f.exists() and not args.overwrite:
            continue
        with h5py.File(r['sleap_h5'], 'r') as h:
            tr = h['tracks'][:]
            names = np.array([n.decode() for n in h['node_names'][:]])
        n_src = int(r['source_frames'])
        if tr.shape[:3] != (1, 2, len(names)) or tr.shape[3] != n_src:
            raise SystemExit(f'{r["observation_id"]}: tracks {tr.shape}, expected (1, 2, {len(names)}, {n_src})')
        nodes = np.transpose(tr[0][:, :, OFFSET::STEP], (2, 1, 0)).astype(np.float32)  # (n5, nodes, xy)
        tmp = out / f'{r["observation_id"]}.tmp.npz'
        np.savez(tmp, nodes=nodes, node_names=names, source_frames=n_src)
        tmp.rename(f)
        miss = np.isnan(nodes[..., 0])
        print(f'{r["observation_id"]}: {len(nodes)} frames, all nodes missing in {miss.all(1).mean():.4f}, '
              f'mean nodes visible {(~miss).sum(1).mean():.1f}/{len(names)}', flush=True)
    print(f'done: {len(rows)} videos in {out}')


if __name__ == '__main__':
    main()
