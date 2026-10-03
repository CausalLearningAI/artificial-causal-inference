"""
Tables of the frogs ECI domain (src/eci/domain.py FrogsDomain): Xenopus juvenile froglets (Sweeney group, ISTA), one
frog per backlit dish, top-down grayscale video, 800 x 800 px, 60 fps. Hour 1 of every frog of three groups:

    WT     Juv/WT                         13 frogs   control
    FoxP1  Juv/FoxP1 gRNA1 half           14 frogs   half-embryo FoxP1 crispant (mutant_side = the injected side)
    En1    Juv/En1 gRNA4 half Genotyped    8 frogs   half-embryo En1 crispant

Raw file name: {session}_{date}_{time}_{group}_[{6-digit animal id}_]stJuv_[{animal id}_]{A}_{B}.mp4, A = hour (1 or 2) of one
continuous 2-h recording, B = dish slot 1-6. Frog = (session, B); observation_id = <group>_<session>_<B>. Only A = 1 is
used (one video per frog). The raw data is read-only (never written); poses come from the re-predicted SLEAP folder
(one model for every video), checked to exist for every video.

--step source (before scripts/00_data, src/data/standardize.py experiment=frogs/v1):
    data/frogs/v1/experiment.csv           one row per frog: observation_id, observation_file (= <id>.mp4), group, T
                                           (0 WT, 1 FoxP1, 2 En1), session, date, time, slot, hour, animal_id,
                                           mutant_side (blank for WT), roi_top/left/bottom/right (ImageJ oval of the
                                           dish, 800 px, raw = unflipped), source_fps, source_frames, source_file,
                                           sleap_h5, roi_file, valid, start_frame, end_frame, hflip (1 = mirrored
                                           left-right at standardization, FLIP_SEED below), flip_seed
    data/frogs/v1/observations/source/     <id>.mp4 symlinks to the raw videos (standardize.py reads them)
--step tables (after src/dataset/get_frames.py experiment=frogs/v1, 5 fps 512 px frames):
    dataset/frogs/eci/annotations.csv      one row per frame, videos in experiment.csv order: observation_id,
                                           frame_idx, fps, frame_path, group, T, source_frame (= 12 frame_idx + 5, the
                                           60 fps frame it was taken from, measured). No behaviour labels. Row order = the row
                                           order of every frogs codes memmap, written once and never reordered.
    dataset/frogs/eci/experiment.csv       one row per video (the design columns of experiment.csv)

Checks (fail loud): every frog of the three groups has exactly one hour-1 file and the group counts are 13 / 14 / 8;
ids unique; mutant_side is left / right (stripped) for the half groups and the original and re-predicted json agree;
the ROI is an ImageJ oval; every video has a re-predicted SLEAP file; 60 fps, 800 x 800; every video's frame folder
holds frame_000000 .. frame_{n-1} with n = 5 fps x the source duration (+-1). Existing outputs are kept unless
--overwrite.

mutant_side stays the ORIGINAL side; after the flip every treated frog's mutant side is on the right of the frog.

Usage: python scripts/eci/frogs_prepare.py --step source
       python scripts/eci/frogs_prepare.py --step tables
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.eci.contrasts import observation_blocks  # noqa: E402
from src.eci.domain import get_domain  # noqa: E402

RAW = Path('/nfs/scistore23/sweengrp/ftoma/Behavior_MNV1/MNV1_revision_dataset/Juv')
REPRED = Path('/nfs/scistore23/sweengrp/ftoma/Behavior_MNV1/MNV1_revision_dataset_REPRED/Juv')
# group -> (raw folder, re-predicted folder, T, expected frogs)
GROUPS = {'WT': ('WT', 'WT REPRED', 0, 13),
          'FoxP1': ('FoxP1 gRNA1 half', 'FoxP1 gRNA1 half REPRED', 1, 14),
          'En1': ('En1 gRNA4 half Genotyped', 'En1 gRNA4 half Gen REPRED', 2, 8)}
# the animal id sits before or after 'stJuv' (both occur)
NAME = re.compile(r'^(\d+)_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})_(.+?)_(?:(\d{6})_)?stJuv_(?:(\d{6})_)?(\d)_(\d)\.mp4$')
# Mirror flip (user design): every treated video whose mutant side is 'left' is flipped left-right at the source
# (src/data/standardize.py, experiment.csv 'hflip'), so the mutant side is always the frog's RIGHT. Flipping also
# mirrors the embossed dish label text, so a seeded random ceil(n_WT x treated flip rate) WT videos are flipped too.
FLIP_SEED = 0
SOURCE_FPS, STEP, OFFSET = 60, 12, 5  # 5 fps frame i = source frame 12 i + 5 (scripts/eci/frogs_sleap.py)
DESIGN_COLS = ['observation_id', 'group', 'T', 'session', 'date', 'time', 'slot', 'hour', 'animal_id', 'mutant_side',
               'hflip']
DATA = ROOT / 'data/frogs/v1'


def read_roi(path):
    """ImageJ .roi -> (type, top, left, bottom, right); the frogs dishes are ovals (type 2)."""
    b = Path(path).read_bytes()
    if b[:4] != b'Iout':
        raise SystemExit(f'{path}: not an ImageJ roi')
    return (b[6],) + tuple(int.from_bytes(b[o:o + 2], 'big', signed=True) for o in (8, 10, 12, 14))


def probe(path):
    """-> (r_frame_rate, width, height, nb_frames) of the video stream."""
    o = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                        'stream=r_frame_rate,width,height,nb_frames', '-of', 'json', str(path)],
                       capture_output=True, text=True, check=True)
    s = json.loads(o.stdout)['streams'][0]
    return s['r_frame_rate'], int(s['width']), int(s['height']), int(s['nb_frames'])


def mutant_side(path):
    return json.loads(Path(path).read_text())['mutant_side'].strip()


def step_source(overwrite):
    csv = DATA / 'experiment.csv'
    if csv.exists() and not overwrite:
        print(f'[SKIP] {csv} exists (--overwrite to rebuild)')
        return
    rows = []
    for g, (folder, rfolder, T, n_expected) in GROUPS.items():
        files = sorted(p.name for p in (RAW / folder).glob('*.mp4'))
        parsed = []
        for f in files:
            m = NAME.match(f)
            if not m:
                raise SystemExit(f'{folder}/{f}: unparsed file name')
            parsed.append((f,) + m.groups())
        frogs = {}
        for f, ses, date, time, tag, aid1, aid2, A, B in parsed:
            if aid1 and aid2:
                raise SystemExit(f'{folder}/{f}: two animal ids')
            aid = aid1 or aid2
            frogs.setdefault((int(ses), int(B)), []).append((int(A), f, date, time, aid))
        n_h1 = {k: sum(a == 1 for a, *_ in v) for k, v in frogs.items()}
        bad = {k: n for k, n in n_h1.items() if n != 1}
        if bad or len(frogs) != n_expected:
            raise SystemExit(f'{g}: {len(frogs)} frogs (expected {n_expected}); hour-1 files per frog != 1: {bad}')
        for (ses, B), v in sorted(frogs.items()):
            (A, f, date, time, aid), = [x for x in v if x[0] == 1]
            src = RAW / folder / f
            h5 = REPRED / rfolder / f'{f}.predictions.analysis.h5'
            roi = src.with_suffix('.roi')
            for p in (h5, roi):
                if not p.exists():
                    raise SystemExit(f'missing {p}')
            side = ''
            if g != 'WT':
                side = mutant_side(src.with_suffix('.json'))
                rj = REPRED / rfolder / f'{src.stem}.json'
                if rj.exists() and mutant_side(rj) != side:
                    raise SystemExit(f'{f}: mutant_side {side!r} (original) vs {mutant_side(rj)!r} (re-predicted)')
                if side not in ('left', 'right'):
                    raise SystemExit(f'{f}: mutant_side {side!r}')
            rtype, top, left, bottom, right = read_roi(roi)
            if rtype != 2:
                raise SystemExit(f'{roi}: roi type {rtype}, expected an oval (2)')
            fps, w, h, nb = probe(src)
            if fps != f'{SOURCE_FPS}/1' or (w, h) != (800, 800):
                raise SystemExit(f'{f}: {fps} fps, {w} x {h}')
            oid = f'{g}_{ses}_{B}'
            rows.append({'observation_id': oid, 'observation_file': f'{oid}.mp4', 'group': g, 'T': T, 'session': ses,
                         'date': date, 'time': time.replace('-', ':'), 'slot': B, 'hour': A,
                         'animal_id': aid or '', 'mutant_side': side, 'roi_top': top, 'roi_left': left,
                         'roi_bottom': bottom, 'roi_right': right, 'source_fps': SOURCE_FPS, 'source_frames': nb,
                         'source_file': str(src), 'sleap_h5': str(h5), 'roi_file': str(roi), 'valid': 1,
                         'start_frame': 0, 'end_frame': nb - 1})
        print(f'{g}: {n_expected} frogs, sessions {sorted({s for s, _ in frogs})}', flush=True)
    e = pd.DataFrame(rows)
    if e['observation_id'].duplicated().any():
        raise SystemExit('duplicate observation ids')
    treated = e['group'] != 'WT'
    e['hflip'] = (treated & (e['mutant_side'] == 'left')).astype(int)
    rate = e.loc[treated, 'hflip'].mean()
    wt = sorted(e.loc[~treated, 'observation_id'])
    n_wt = int(np.ceil(len(wt) * rate))
    e.loc[e['observation_id'].isin(np.random.default_rng(FLIP_SEED).choice(wt, n_wt, replace=False)), 'hflip'] = 1
    e['flip_seed'] = FLIP_SEED
    print(f'hflip: treated rate {rate:.3f} -> {n_wt} WT videos (seed {FLIP_SEED}); per group '
          f'{e.groupby("group")["hflip"].agg(["sum", "size"]).to_dict("index")}', flush=True)
    src_dir = DATA / 'observations/source'
    src_dir.mkdir(parents=True, exist_ok=True)
    for r in e.itertuples():
        link = src_dir / r.observation_file
        if link.is_symlink() or link.exists():
            if Path(link.resolve()) != Path(r.source_file).resolve():
                raise SystemExit(f'{link} points to {link.resolve()}, expected {r.source_file}')
            continue
        link.symlink_to(r.source_file)
    e.to_csv(csv, index=False)
    print(f'wrote {csv}: {len(e)} videos; {len(e)} symlinks in {src_dir}')
    print(e.groupby(['group', 'session']).size().to_string())


def step_tables(overwrite):
    D = get_domain('frogs')
    if D.ann_path.exists() and D.experiment_csv.exists() and not overwrite:
        print(f'[SKIP] {D.ann_path} and {D.experiment_csv} exist (--overwrite to rebuild)')
        return
    e = pd.read_csv(DATA / 'experiment.csv', dtype={'animal_id': str, 'mutant_side': str})
    frames_root = ROOT / 'dataset/frogs/v1/frames/full'
    blocks = []
    for r in e.itertuples():
        d = frames_root / r.observation_id
        n = len(list(d.glob('frame_*.jpg'))) if d.exists() else 0
        expect = r.source_frames / STEP
        if abs(n - expect) > 1:
            raise SystemExit(f'{r.observation_id}: {n} frames in {d}, expected {expect:.1f} (5 fps of {r.source_frames})')
        missing = [i for i in (0, n - 1) if not (d / f'frame_{i:06d}.jpg').exists()]
        if missing:
            raise SystemExit(f'{r.observation_id}: frames not numbered 0..{n - 1}')
        fi = np.arange(n)
        blocks.append(pd.DataFrame({'observation_id': r.observation_id, 'frame_idx': fi, 'fps': 5.0,
                                    'frame_path': [f'frogs/v1/frames/full/{r.observation_id}/frame_{i:06d}.jpg' for i in fi],
                                    'group': r.group, 'T': r.T, 'source_frame': fi * STEP + OFFSET}))
    ann = pd.concat(blocks, ignore_index=True)
    D.eci_dir.mkdir(parents=True, exist_ok=True)
    ann.to_csv(D.ann_path, index=False)
    b = observation_blocks(D.ann_path)  # re-read: contiguous blocks, increasing frame_idx
    if not np.array_equal(b['observation_id'].values, e['observation_id'].values):
        raise SystemExit('annotations.csv block order differs from experiment.csv')
    e[DESIGN_COLS].to_csv(D.experiment_csv, index=False)
    print(f'wrote {D.ann_path}: {len(ann)} rows, {len(b)} videos; {D.experiment_csv}')
    print(ann.groupby('group')['observation_id'].agg(['nunique', 'size']).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--step', required=True, choices=('source', 'tables'))
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()
    {'source': step_source, 'tables': step_tables}[args.step](args.overwrite)


if __name__ == '__main__':
    main()
