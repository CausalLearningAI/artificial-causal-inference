"""
Shared constants and helpers for the MABe22 mouse-triplet ECI evaluation (scripts/benchmark/mabe22/eci_*.py).

Data (READ-ONLY, never written): dataset/benchmark/mabe22/{videos,raw}. Outputs: results/vision/benchmark_mabe22/eci/.
Frame convention: a clip has 1,800 labelled frames (label index i = 0..1799) and 1,801 video frames; the video frame of
label i is i + OFFSET, OFFSET measured by eci_prep.py (keypoints vs dark pixels) and stored in plan/offset.json.
Sampling grid: label indices i = 30, 36, ..., 1794 (5 fps); the lag frame (1 s) is label index i - 30.
"""
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / 'dataset/benchmark/mabe22'
RAW = DATA / 'raw'
VID = DATA / 'videos'
OUT = Path(os.environ.get('MABE_ECI_OUT') or REPO / 'results/vision/benchmark_mabe22/eci')
PLAN = OUT / 'plan'
TOK = Path(os.environ.get('MABE_TOK') or OUT / 'tok')
FFMPEG = os.environ.get('PR_FFMPEG') or str(Path.home() / '.conda/envs/ffmpeg611/bin/ffmpeg')
EXCLUDE = {'QO9ONQTPOYAI26MYHPHY', 'R6R4SNHGFM0ZOOH9A1TN'}  # CRC mismatch / not in the zip
LAG = 30  # 1 s at 30 fps
STEP = 6  # 5 fps
GRID_I = np.arange(LAG, 1800, STEP)  # label indices of the 5 fps grid with a lag frame (295 per clip)
BEHAVIOURS = ['oral_oral_contact', 'oral_genital_contact', 'oral_ear_contact', 'approaches', 'huddles',
              'contact', 'close', 'watching', 'chases']
PRIMARY = ['oral_oral_contact', 'oral_genital_contact', 'oral_ear_contact', 'approaches', 'huddles']
SECONDARY = ['contact', 'close', 'watching']
SHORT = {'oral_oral_contact': 'oral_oral', 'oral_genital_contact': 'oral_genital', 'oral_ear_contact': 'oral_ear',
         'approaches': 'approaches', 'huddles': 'huddles', 'contact': 'contact', 'close': 'close',
         'watching': 'watching', 'chases': 'chases'}
FRAME = 224
log = lambda s: print(time.strftime('%H:%M:%S'), s, flush=True)  # noqa: E731


def stage_dir():
    d = Path(os.environ.get('LOCAL_DIR') or '')
    if not str(d).startswith('/localhome'):
        tmp = os.environ.get('TMPDIR') or '/nonexistent'
        if not str(d).startswith(tmp):
            raise RuntimeError(f'LOCAL_DIR must be under /localhome or $TMPDIR, got {d!r}')
    d.mkdir(parents=True, exist_ok=True)
    return d


def video_path(set_, clip):
    return VID / set_ / f'{clip}.avi'


def decode_clip(set_, clip, stage, fmt='rgb24'):
    """copy the clip to local disk, decode every frame -> (T, 224, 224, 3) uint8 (rgb24) or (T, 224, 224) (gray)."""
    src = video_path(set_, clip)
    loc = Path(stage) / f'_{clip}_{os.getpid()}.avi'
    shutil.copyfile(src, loc)
    try:
        r = subprocess.run([FFMPEG, '-nostdin', '-v', 'error', '-threads', '1', '-i', str(loc), '-f', 'rawvideo',
                            '-pix_fmt', fmt, '-'], check=True, capture_output=True)
    finally:
        loc.unlink()
    c = 3 if fmt == 'rgb24' else 1
    a = np.frombuffer(r.stdout, np.uint8)
    n = a.size // (FRAME * FRAME * c)
    assert n * FRAME * FRAME * c == a.size, (clip, a.size)
    a = a.reshape(n, FRAME, FRAME, c) if c == 3 else a.reshape(n, FRAME, FRAME)
    return a


def offset():
    return int(json.loads((PLAN / 'offset.json').read_text())['offset'])
