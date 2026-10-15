"""
Shared constants and helpers for the CRIM13 single-neuron alignment test (rules: scripts/benchmark/crim13/eci_align.sh).

Data (READ-ONLY, BeeGFS): dataset/benchmark/crim13 -> /mnt/beegfs/locatgrp/shared/rcadei/benchmark/crim13
    {train,test}/<name>_t.seq          Norpix .seq, JPEG frames, 8-bit grey, 640 x 480 or 656 x 490, 25 fps (top view)
    {train,test}/<name>_t.txt          Caltech Behavior Annotator labels, stream S1, 1-based inclusive: label frame L =
                                       0-based frame L - 1 (offset +0, verified on 236 / 236 videos by the inventory)
    {train,test}/<name>_t-track.mat    Y{1,1,t}: 0-2 rows (x, y) native px; row 1 = resident, row 2 = intruder
    {train,test}/<name>_t-seek.mat     byte offset of every frame record
Outputs: results/vision/benchmark_crim13/align/ (gitignored). Token stores: /mnt/beegfs/locatgrp/shared/rcadei/eci_tok/crim13
Canvas: a W x H frame is PADDED (no crop) to a W x W square, (W - H) // 2 rows above, the rest below, in the ImageNet mean
colour (the grey frame is replicated to 3 channels); a 656-wide canvas is then resized (bicubic, antialias) to 640, so
every encoder sees the same 640 x 640 canvas (one 40 x 40 cell = 16 x 16 canvas px).
"""
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
ROOT = Path('/mnt/beegfs/locatgrp/shared/rcadei/benchmark/crim13')
OUT = Path(os.environ.get('CRIM_OUT') or REPO / 'results/vision/benchmark_crim13/align')
INV = REPO / 'results/vision/benchmark_crim13/inventory/inventory.json'
PLAN = OUT / 'plan'
TOK = Path(os.environ.get('CRIM_TOK') or '/mnt/beegfs/locatgrp/shared/rcadei/eci_tok/crim13')
CANVAS = 640
GRID = 40
CELL = CANVAS // GRID           # 16 canvas px
PADV = (124, 116, 104)          # ImageNet mean colour
FPS = 25
STEP = 5                        # 5 fps sampling grid: v % 5 == 0
LAG = 50                        # 2 s at 25 fps
VOCAB = ['approach', 'attack', 'copulation', 'chase', 'circle', 'drink', 'eat', 'clean', 'human', 'sniff', 'up',
         'walk_away', 'other']
SCORED = ['approach', 'attack', 'copulation', 'chase', 'circle', 'drink', 'eat', 'clean', 'sniff', 'up', 'walk_away']
PAIR = ['sniff', 'approach', 'attack', 'chase', 'circle', 'walk_away', 'copulation']
RESIDENTS = ['A22', 'A24', 'A25', 'A29']
log = lambda s: print(time.strftime('%H:%M:%S'), s, flush=True)  # noqa: E731


def stage_dir():
    d = Path(os.environ.get('LOCAL_DIR') or '')
    if not str(d).startswith('/localhome'):
        raise RuntimeError(f'LOCAL_DIR must be under /localhome, got {d!r}')
    d.mkdir(parents=True, exist_ok=True)
    return d


def meta_of(name):
    """file name <date>_<A##>_Block<k>_<intruder> -> resident code, intruder strain (BC / C57 / ?)"""
    p = name.split('_')
    res = p[1]
    intr = re.sub(r'\d+$', '', p[3]) if len(p) > 3 else ''
    strain = 'BC' if 'BC' in intr else ('C57' if '57' in intr else '?')
    return res, intr, strain


def read_labels(path, n):
    """-> (n,) int8 index into VOCAB (S1 stream; label frame L -> 0-based frame L - 1); -1 = uncovered."""
    lab = np.full(n, -1, np.int8)
    txt = Path(path).read_text().splitlines()
    k = next(i for i, ln in enumerate(txt) if ln.startswith('S1:'))
    for ln in txt[k + 2:]:
        q = ln.split()
        if len(q) != 3 or not q[0].isdigit():
            if ln.startswith('S2:'):
                break
            continue
        a, b, t = int(q[0]), int(q[1]), q[2]
        lab[a - 1:b] = VOCAB.index(t)
    return lab


def load_track(p):
    """Y{1,1,t} -> (T, 2, 2) positions (row 0 resident, row 1 intruder), (T,) number of mice (inventory.py)."""
    import scipy.io as sio
    Y = sio.loadmat(p)['Y']
    pos = np.full((Y.shape[2], 2, 2), np.nan)
    nm = np.zeros(Y.shape[2], np.int8)
    for k in range(Y.shape[2]):
        a = np.asarray(Y[0, 0, k])
        if a.size:
            a = a.reshape(-1, 2)
            nm[k] = min(len(a), 2)
            pos[k, :nm[k]] = a[:2]
    return pos, nm


def load_seek(p):
    import scipy.io as sio
    m = sio.loadmat(p)
    k = [x for x in m if not x.startswith('__')]
    return np.asarray(m[k[0]]).ravel().astype(np.int64)


def geom(W, H):
    """native frame -> canvas: (pad_top, scale) with canvas_xy = (x, (y + pad_top)) * scale."""
    return (W - H) // 2, CANVAS / W


def video_cells(W, H):
    """(GRID*GRID,) bool: 40-grid cells of the 640 canvas holding at least one video pixel."""
    pt, s = geom(W, H)
    y0, y1 = pt * s, (pt + H) * s
    r = np.arange(GRID)
    ok = (r * CELL + CELL > y0) & (r * CELL < y1)
    return np.repeat(ok, GRID)


def track_cell(xy, W, H):
    """(..., 2) native xy -> (...,) cell index on the 40 grid (-1 where NaN / outside)."""
    pt, s = geom(W, H)
    x, y = xy[..., 0] * s, (xy[..., 1] + pt) * s
    ok = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x < CANVAS) & (y >= 0) & (y < CANVAS)
    cx = np.clip(np.nan_to_num(x) // CELL, 0, GRID - 1).astype(int)
    cy = np.clip(np.nan_to_num(y) // CELL, 0, GRID - 1).astype(int)
    return np.where(ok, cy * GRID + cx, -1)


def read_frames(seq, offs, want, gap=1 << 19, threads=4):
    """Decode the frames `want` (0-based indices) of a .seq given its seek offsets: records are read in offset order,
    neighbouring records (gap < 512 KB) coalesced into ONE sequential read each; JPEG decode (cv2, grey) in threads.
    -> dict index -> (H, W) uint8, bytes read."""
    import cv2
    want = sorted(set(int(x) for x in want))
    runs, cur = [], [want[0]]
    for i in want[1:]:
        if offs[i] - offs[cur[-1]] < gap:
            cur.append(i)
        else:
            runs.append(cur)
            cur = [i]
    runs.append(cur)
    blobs, nbytes = {}, 0
    with open(seq, 'rb', buffering=0) as f:
        for r in runs:
            a = int(offs[r[0]])
            f.seek(a)
            # the last record's size is read from its own 4-byte header inside the run
            f2 = f.read(int(offs[r[-1]]) - a + 4)
            n_last = int(np.frombuffer(f2[-4:], '<u4')[0])
            buf = f2 + f.read(n_last - 4 + 16)
            nbytes += len(buf)
            for i in r:
                o = int(offs[i]) - a
                n = int(np.frombuffer(buf[o:o + 4], '<u4')[0])
                blobs[i] = buf[o + 4:o + n]

    def dec(i):
        return i, cv2.imdecode(np.frombuffer(blobs[i], np.uint8), cv2.IMREAD_GRAYSCALE)
    with ThreadPoolExecutor(threads) as ex:
        out = dict(ex.map(dec, want))
    return out, nbytes


def canvas_gpu(frames, W, H, dev):
    """list of (H, W) uint8 grey -> (n, 3, 640, 640) float 0-255 GPU canvas (pad, then resize 656 -> 640)."""
    import torch
    import torch.nn.functional as F
    pt, _ = geom(W, H)
    x = torch.from_numpy(np.ascontiguousarray(np.stack(frames))).to(dev)
    c = torch.empty((len(frames), 3, W, W), dtype=torch.float32, device=dev)
    c[:] = torch.tensor(PADV, dtype=torch.float32, device=dev)[None, :, None, None]
    c[:, :, pt:pt + H] = x[:, None].float()
    if W != CANVAS:
        c = F.interpolate(c, size=(CANVAS, CANVAS), mode='bicubic', align_corners=False,
                          antialias=True).round().clamp(0, 255)
    return c
