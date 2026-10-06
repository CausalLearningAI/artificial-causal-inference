"""Shared registry, video I/O and track I/O for the identity layer of the tracking pilot.

Coordinates everywhere are SOURCE pixels (mice 2064 x 2064, ants 824 x 824), frames are SOURCE
frame indices at 30 fps (`frame_src`). Human annotations (BORIS for mice, the ant behaviour csvs)
are in the same source-frame units (verified against src/data/standardize.py, which trims with
start_frame / source_fps, and src/dataset/get_annotations.py, which states 'annotations are in
source-video frame indices').

Track contract (from the tracker worker), one row per (frame, track):
    frame_src, t_sec, track_id, cx, cy, w, h, heading, detected (bool), conf, variant
"""
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / 'results' / 'tracking_pilot'
FPS = 30.0

IDENTITIES = {
    'mice': ['m1_none', 'm2_head', 'm3_middle', 'm4_back'],
    'ants': ['focal', 'blue', 'yellow'],
}
N_ANIMALS = {'mice': 4, 'ants': 3}

VIDEOS = {
    'mice': {
        'rd25': '2024-12-02_15-34-27_BHVScreen_rd25_SocialOdor2_Habit1.mp4',
        'rd32': '2025-02-07_12-18-47_BHVScreen_rd32_SocialOdor_Test.mp4',
        'rd18': '2024-10-14_16-00-16_BHVScreen_rd18_SocialOdor_Post.mp4',
    },
    'ants': {'3_1_1': '3_1_1.mkv', '3_6_6': '3_6_6.mkv', '3_21_2': '3_21_2.mkv'},
}

TRACK_COLS = ['frame_src', 't_sec', 'track_id', 'cx', 'cy', 'w', 'h', 'heading', 'detected', 'conf',
              'variant']


def video_path(domain, vid):
    if domain == 'mice':
        return ROOT / 'data/mice/source' / VIDEOS[domain][vid]
    return ROOT / 'data/ants/v3/observations/source' / VIDEOS[domain][vid]


def out_dir(domain, vid, sub='identity'):
    d = RESULTS / domain / vid / sub
    d.mkdir(parents=True, exist_ok=True)
    return d


def annotated_window(domain, vid):
    """(start_frame, end_frame) in source frames, from the experiment.csv of the dataset."""
    if domain == 'mice':
        exp = pd.read_csv(ROOT / 'data/mice/v1/experiment.csv')
        r = exp[exp.observation_file == VIDEOS[domain][vid]].iloc[0]
    else:
        exp = pd.read_csv(ROOT / 'data/ants/v3/experiment.csv')
        r = exp[exp.observation_id == vid].iloc[0]
    return int(r.start_frame), int(r.end_frame)


def video_info(path):
    out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                          'stream=width,height,r_frame_rate', '-of', 'csv=p=0', str(path)],
                         capture_output=True, text=True, check=True).stdout.strip().split(',')
    num, den = out[2].split('/')
    return int(out[0]), int(out[1]), float(num) / float(den)


def first_frame_index(path):
    """Timestamp index (round(pts * fps)) of the first decoded frame. The ant mkvs start at
    pts = 1/30 s, so their first decoded frame is frame 1, which is also how the ant annotation
    files count ('Beginning-frame 1' = 0:00:00.033). The mice mp4s start at 0."""
    W, H, fps = video_info(path)
    out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                          'frame=best_effort_timestamp_time', '-of', 'csv=p=0', '-read_intervals', '%+#8',
                          str(path)], capture_output=True, text=True, check=True).stdout.split()
    return int(round(min(float(x.strip(',')) for x in out if x.strip(',')) * fps))


def iter_frames(path, stride=1, scale=1.0, gray=False, start=0, end=None):
    """Sequential decode through an ffmpeg pipe. Yields (frame_src, image) for every `stride`-th
    frame in [start, end). frame_src is the TIMESTAMP index round(pts * fps), the same convention
    as read_frame and the human annotations. image is uint8 HxW (gray) or HxWx3 BGR, resized by
    `scale`."""
    W, H, fps = video_info(path)
    f0 = first_frame_index(path)
    if start <= f0:
        start = 0
    w, h = int(round(W * scale)), int(round(H * scale))
    vf = [f"select='not(mod(n\\,{stride}))'"] if stride > 1 else []
    if scale != 1.0:
        vf.append(f'scale={w}:{h}:flags=area')
    cmd = ['ffmpeg', '-v', 'error']
    if start:
        cmd += ['-ss', f'{(start - 0.5) / fps:.6f}']
    cmd += ['-i', str(path)]
    if vf:
        cmd += ['-vf', ','.join(vf)]
    cmd += ['-vsync', '0', '-f', 'rawvideo', '-pix_fmt', 'gray' if gray else 'bgr24', '-']
    nb = w * h * (1 if gray else 3)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=nb * 2)
    k = 0
    try:
        while True:
            buf = p.stdout.read(nb)
            if len(buf) < nb:
                break
            f = (start if start else f0) + k * stride
            if end is not None and f >= end:
                break
            img = np.frombuffer(buf, np.uint8).reshape((h, w) if gray else (h, w, 3))
            yield f, img
            k += 1
    finally:
        p.stdout.close()
        p.kill()
        p.wait()


def read_frame(path, frame_src, gray=False):
    """Random access to one source frame (accurate seek)."""
    W, H, fps = video_info(path)
    cmd = ['ffmpeg', '-v', 'error', '-ss', f'{max(frame_src - 0.5, 0) / fps:.6f}', '-i', str(path), '-frames:v', '1',
           '-f', 'rawvideo', '-pix_fmt', 'gray' if gray else 'bgr24', '-']
    buf = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(buf[:W * H * (1 if gray else 3)], np.uint8).reshape((H, W) if gray else (H, W, 3))


def load_tracks(path):
    """Load and validate a tracks.parquet in the tracker contract. Fails loudly on missing columns."""
    df = pd.read_parquet(path)
    missing = [c for c in TRACK_COLS if c not in df.columns]
    if missing:
        raise ValueError(f'{path}: missing contract columns {missing}; has {list(df.columns)}')
    df = df.copy()
    df['frame_src'] = df['frame_src'].astype(int)
    df['track_id'] = df['track_id'].astype(int)
    df['detected'] = df['detected'].astype(bool)
    for c in ['cx', 'cy', 'w', 'h', 'heading', 'conf', 't_sec']:
        df[c] = df[c].astype(float)
    if df.duplicated(['variant', 'frame_src', 'track_id']).any():
        raise ValueError(f'{path}: duplicated (variant, frame_src, track_id) rows')
    return df.sort_values(['variant', 'frame_src', 'track_id']).reset_index(drop=True)


def frame_stride(tracks):
    """Most common gap between consecutive frames that carry rows."""
    fr = np.unique(tracks['frame_src'].values)
    if len(fr) < 2:
        return 1
    d = np.diff(fr)
    return int(np.bincount(d).argmax())
