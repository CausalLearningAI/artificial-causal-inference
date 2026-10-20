"""
Build an ISTAnt manifest in the ORIGINAL repo's format (labels.parquet + images/img_%07d.jpg) from our ants v2
source videos, so the original s2/s3/s4 scripts can run unchanged. Frame picking = src/old/data.py (cv2
sequential decode, every 15th source frame 0..17985 -> 1200 frames per video = 10 min at 2 fps, BGR -> RGB,
PIL JPEG at default quality), the 44 valid videos in CSV order. Columns as in the original labels.parquet
(seen in notebooks/istant): path, video_id (row of experiment.csv, so b3 = 11 is absent), treatment
(CSV treatment 2 -> 1), grooming_yellow2focal, grooming_blue2focal, exp_batch_0..4, exp_pos_1..9, grooming_2focal.
"""
import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'scripts/eci/v0'))
from ants_v2_v0 import load_labels, N_FRAMES, FPS_FACTOR  # noqa: E402


def dump(args):
    vi, src, img_dir = args
    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(str(src))
    n, kept = 0, 0
    while kept < N_FRAMES:
        ok, fr = cap.read()
        if not ok:
            break
        if n % FPS_FACTOR == 0:
            Image.fromarray(np.ascontiguousarray(fr[:, :, ::-1])).save(Path(img_dir) / f'img_{vi * N_FRAMES + kept:07d}.jpg', format='JPEG')
            kept += 1
        n += 1
    cap.release()
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video-dir', required=True)
    ap.add_argument('--manifest-dir', required=True)
    ap.add_argument('--workers', type=int, default=16)
    a = ap.parse_args()
    md = Path(a.manifest_dir)
    (md / 'images').mkdir(parents=True, exist_ok=True)
    exp = pd.read_csv(REPO / 'data/ants/v2/experiment.csv')
    valid = exp[exp.valid == 1]
    with Pool(a.workers) as p:
        kept = p.map(dump, [(i, Path(a.video_dir) / f, md / 'images') for i, f in enumerate(valid.observation_file)])
    assert all(k == N_FRAMES for k in kept), kept
    rows = []
    for i, (csv_row, v) in enumerate(valid.iterrows()):
        lab = load_labels(REPO / 'data/ants/v2/annotations' / v.annotation_file)
        d = pd.DataFrame(dict(path=[str(md / 'images' / f'img_{i * N_FRAMES + k:07d}.jpg') for k in range(N_FRAMES)]))
        d['video_id'] = csv_row
        d['treatment'] = int(v.treatment == 2)
        d['grooming_yellow2focal'] = lab[:, 0].astype(int)
        d['grooming_blue2focal'] = lab[:, 1].astype(int)
        for b, name in enumerate('abcde'):
            d[f'exp_batch_{b}'] = int(v.batch == name)
        for q in range(1, 10):
            d[f'exp_pos_{q}'] = int(v.position == q)
        d['grooming_2focal'] = (lab.sum(1) > 0).astype(int)
        rows.append(d)
    df = pd.concat(rows, ignore_index=True)
    df.to_parquet(md / 'labels.parquet')
    print(f'manifest: {len(df)} rows, {df.video_id.nunique()} videos, grooming_2focal prevalence {df.grooming_2focal.mean():.4f}')


if __name__ == '__main__':
    main()
