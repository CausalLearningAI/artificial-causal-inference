"""
Frame-mapping check for the mice native decoding of scripts/eci/res_pool.py (CPU, a few videos).

Question: which SEQUENTIALLY decoded source frame (index n from the start of the stream, as res_pool decode) is the
dataset's 5 fps frame k? And which sequential frame does the seek decoding of scripts/eci/native_res.py (-ss s0/30 before
-i, then select n = x - s0) return for the nominal source frame x = 6 k + 2?
Method (as native_res_ants.exact_offsets): reproduce the standardisation (src/data/standardize.py: fps=5 + scale + pad
to 512 on the decoded stream) LOSSLESSLY for 100 frames k in [k0, k0 + 100) and run the same scale chain on every
source frame of that window; output frame k must equal one source frame exactly (mean abs difference 0) -> its
sequential index. Then the native_res seek decode of x = 6 k + 2 (scaled to 512) is matched to the same window.
Usage: python scripts/eci/res_pool_offset.py --videos 6 --k0 0 4400   -> results/vision/eci_res_pool/mice/offset_check.json
"""
import argparse
import json
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts/eci'))
OUT = REPO / 'results/vision/eci_res_pool'
SRC = REPO / 'data/mice/source'
CHAIN = 'scale=512:512:force_original_aspect_ratio=decrease,pad=512:512:(ow-iw)/2:(oh-ih)/2'
PX = 512 * 512 * 3


def raw(cmd):
    return np.frombuffer(subprocess.run(cmd, check=True, capture_output=True).stdout, np.uint8).reshape(-1, PX)


def check(a):
    obs, src, k0s, ffmpeg = a
    r = {}
    for k0 in k0s:
        A = raw([ffmpeg, '-v', 'error', '-threads', '1', '-i', str(src), '-ss', '0', '-vf',
                 f"fps=5,select='between(n\\,{k0}\\,{k0 + 99})',{CHAIN}", '-vsync', '0', '-frames:v', '100',
                 '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-']).astype(np.int16)
        s0, s1 = max(0, 6 * k0 - 6), 6 * (k0 + 99) + 8
        B = raw([ffmpeg, '-v', 'error', '-threads', '1', '-i', str(src), '-vf', f"select='between(n\\,{s0}\\,{s1})',{CHAIN}",
                 '-vsync', '0', '-frames:v', str(s1 - s0 + 1), '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-']).astype(np.int16)
        offs, best, second = [], [], []
        for i in range(len(A)):
            c = 6 * (k0 + i) - s0
            js = list(range(max(0, c - 6), min(len(B), c + 9)))
            m = np.array([np.abs(A[i] - B[j]).mean() for j in js])
            o = np.argsort(m)
            offs.append(js[o[0]] + s0 - 6 * (k0 + i))
            best.append(float(m[o[0]]))
            second.append(float(m[o[1]]))
        # native_res seek decode of x = 6 k + 2 for 10 of these k, matched to the sequential window
        seek = []
        for i in range(0, len(A), 10):
            x = 6 * (k0 + i) + 2
            S = raw([ffmpeg, '-v', 'error', '-threads', '1', '-ss', f'{x / 30.0:.4f}', '-i', str(src), '-vf',
                     f"select='eq(n\\,0)',{CHAIN}", '-vsync', '0', '-frames:v', '1', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                     '-']).astype(np.int16)[0]
            c = x - s0
            js = list(range(max(0, c - 6), min(len(B), c + 7)))
            m = np.array([np.abs(S - B[j]).mean() for j in js])
            o = np.argsort(m)
            seek.append({'k': k0 + i, 'seq_index_of_seek_frame_minus_x': int(js[o[0]] + s0 - x), 'mad': float(m[o[0]]),
                         'second': float(m[o[1]])})
        u, n = np.unique(offs, return_counts=True)
        r[str(k0)] = {'n': len(A), 'seq_offsets': {int(a): int(b) for a, b in zip(u, n)}, 'max_best_mad': max(best),
                      'min_second_mad': min(second), 'seek': seek}
    return obs, r


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--videos', type=int, default=6)
    p.add_argument('--k0', type=int, nargs='+', default=[0, 4400])
    p.add_argument('--ffmpeg', default='ffmpeg')
    a = p.parse_args()
    exp = pd.read_csv(REPO / 'data/mice/v1/experiment.csv').set_index('observation_id')
    P = json.loads((OUT / 'plan.json').read_text())['mice']['videos']
    ev = sorted(o for o, v in P.items() if v['set'] == 'eval')
    pick = ev[::max(1, len(ev) // a.videos)][:a.videos]
    jobs = []
    for o in pick:
        n_k = int(exp.loc[o, 'end_frame'] - exp.loc[o, 'start_frame']) // 6
        jobs.append((o, SRC / exp.loc[o, 'observation_file'], [k for k in a.k0 if k + 100 <= n_k], a.ffmpeg))
    with Pool(len(jobs)) as pool:
        res = dict(pool.map(check, jobs))
    for o, r in res.items():
        print(o, {k: (v['seq_offsets'], round(v['max_best_mad'], 3), round(v['min_second_mad'], 2),
                      sorted({s['seq_index_of_seek_frame_minus_x'] for s in v['seek']}),
                      round(max(s['mad'] for s in v['seek']), 3)) for k, v in r.items()}, flush=True)
    (OUT / 'mice').mkdir(parents=True, exist_ok=True)
    (OUT / 'mice' / f'offset_check_{Path(a.ffmpeg).name}.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
