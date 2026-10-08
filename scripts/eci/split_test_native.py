"""
Instance-split GATE test, mice B-native input (CPU): decode the eval frames (and their B2 windows) from the native
source videos (data/mice/source/*.mp4, 2064 px HEVC, 30 fps) and store them at 1024 px, the resolution SAM 2 runs at
(the 512 px frames are upsampled x2 by SAM, the native ones downsampled x2.016, so these carry twice the linear detail).

Frame mapping: 512 px frame k (5 fps) = source frame start_frame + 6 k + offset (src/data/standardize.py: ffmpeg
output seek + fps=5); offset checked first by decoding source frames 6k-3 .. 6k+3 of 3 frames x 8 videos, scaling to 512
and taking the offset with the lowest mean absolute grey difference to the dataset JPEG (must be the same for all).
Decoding: ffmpeg -ss (accurate seek) -i src, select every 6th frame, scale 1024, JPEG q2, one call per eval frame
(frames s .. t for frames with a B2 start, else t only), parallel over --workers processes, to the job's local disk.

Only the main contact / non-contact sets (not the pilot-only frames).
Output results/vision/eci_split_test/mice/native.tar (members <observation_id>/<frame_idx:06d>.jpg), native.json.
Usage: python scripts/eci/split_test_native.py --workers 16
"""
import argparse
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts/eci'))
from split_test_common import OUT  # noqa: E402

FPS, STEP = 30.0, 6
log = lambda s: print(s, flush=True)  # noqa: E731


def decode(src, first, n, step, size, tmp):
    """source frames first, first + step, ... (n of them) -> list of JPEG bytes at size x size."""
    with tempfile.TemporaryDirectory(dir=tmp) as d:
        vf = (f"select='not(mod(n\\,{step}))'," if step > 1 else '') + f'scale={size}:{size}'
        cmd = ['ffmpeg', '-v', 'error', '-ss', f'{first / FPS:.4f}', '-i', str(src), '-vf', vf, '-vsync', '0',
               '-frames:v', str(n), '-q:v', '2', f'{d}/%05d.jpg']
        subprocess.run(cmd, check=True)
        out = [Path(d, f'{i + 1:05d}.jpg').read_bytes() for i in range(n) if Path(d, f'{i + 1:05d}.jpg').exists()]
    return out


def job(a):
    obs, src, start, k0, k1, off, tmp = a
    b = decode(src, start + STEP * k0 + off, k1 - k0 + 1, STEP, 1024, tmp)
    return obs, k0, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=16)
    args = ap.parse_args()
    t0 = time.time()
    tmp = os.environ.get('STAGE', '/tmp')
    E = pd.read_parquet(OUT / 'mice/eval.parquet')
    E = E[E.set.isin(['contact', 'noncontact'])]  # the main sets only (pilot-only frames skipped: NFS reads)
    exp = pd.read_csv(REPO / 'data/mice/v1/experiment.csv').set_index('observation_id')
    src = {o: REPO / 'data/mice/source' / exp.loc[o, 'observation_file'] for o in E.obs.unique()}
    start = {o: int(exp.loc[o, 'start_frame']) for o in E.obs.unique()}
    # offset check
    rng = np.random.default_rng(0)
    vids = rng.choice(sorted(E.obs.unique()), 8, replace=False)
    res = []
    for o in vids:
        for _, r in E[E.obs == o].head(3).iterrows():
            ref = np.asarray(Image.open(REPO / 'dataset' / r.frame_path).convert('L')).astype(np.float32)
            b = decode(src[o], start[o] + STEP * int(r.frame_idx) - 3, 7, 1, 512, tmp)
            mad = [float(np.abs(np.asarray(Image.open(io.BytesIO(x)).convert('L')).astype(np.float32) - ref).mean())
                   for x in b]
            res.append(mad)
            log(f'  offset check {o} k={r.frame_idx}: MAD for offsets -3..+3 = {np.round(mad, 2).tolist()}')
    best = [int(np.argmin(m)) - 3 for m in res if len(m) == 7]
    off = int(pd.Series(best).mode().iloc[0])
    log(f'offset per check: {best} -> using {off} [{time.time() - t0:.0f}s]')
    jobs = []
    for _, r in E.iterrows():
        k1 = int(r.frame_idx)
        k0 = k1 - int(r.b2_d) if r.b2_d >= 0 else k1
        jobs.append((r.obs, str(src[r.obs]), start[r.obs], k0, k1, off, tmp))
    tpath = Path(tmp) / 'native.tar'
    n_fr, short = 0, 0
    with tarfile.open(tpath, 'w') as tf, Pool(args.workers) as pool:
        for i, (o, k0, frames) in enumerate(pool.imap(job, jobs, chunksize=2)):
            want = jobs[i][4] - k0 + 1
            short += len(frames) < want
            for j, b in enumerate(frames):
                ti = tarfile.TarInfo(f'{o}/{k0 + j:06d}.jpg')
                ti.size = len(b)
                tf.addfile(ti, io.BytesIO(b))
                n_fr += 1
            if i % 200 == 0:
                log(f'  {i}/{len(jobs)} clips, {n_fr} frames [{time.time() - t0:.0f}s]')
    os.system(f'cp {tpath} {OUT}/mice/native.tar')
    (OUT / 'mice/native.json').write_text(json.dumps({'offset': off, 'offset_checks': best, 'mad': res,
                                                      'n_frames': n_fr, 'n_clips_short': short}, indent=1))
    log(f'done: {n_fr} frames, {short} clips short, {os.path.getsize(tpath) / 1e9:.2f} GB [{time.time() - t0:.0f}s]')


if __name__ == '__main__':
    main()
