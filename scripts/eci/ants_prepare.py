"""
Per-frame and per-video tables of the ants ECI domain (src/eci/domain.py AntsDomain): the experiments of one layout
pooled, so one SAE and one codes memmap cover them all. The layout is the domain's ANTS_ECI (environment):

    eci     (default, the 10-min layout of the published atlas) v2 and v3, every treatment, 3000 frames (10 min)
    eci30   v2 (every treatment, 3000 frames: no longer footage), v3 t = 2/6/8, v4 t = 2/6 and v4_infested t = 2/6,
            9000 frames (30 min) each

    dataset/ants/<layout>/annotations.csv   one row per frame, experiment by experiment (LAYOUTS order), videos in
                                            experiment.csv order: observation_id, frame_idx (0 .. n - 1), fps (5),
                                            frame_path (ants/<v>/frames/full/<obs>/frame_<idx:06d>.jpg), experiment, T
                                            (raw treatment value), Y_* (behaviour labels of dataset/ants/<v>/
                                            annotations.csv joined on (observation_id, frame_idx), union over the
                                            experiments; blank where a frame has no label). Row order = the row order
                                            of every codes memmap of the layout, so it is written once, never reordered.
    dataset/ants/<layout>/experiment.csv    one row per kept video (data/ants/<v>/experiment.csv, valid == 1, treatment
                                            kept): observation_id, experiment, T, batch, position, annotator,
                                            recording_date, nestbox (blank for v2), infested (1 for v4_infested, else 0).

ECI uses no behaviour label (labels only feed the label-alignment diagnostics), so the frame list comes from
experiment.csv and the frame-naming rule of the PPCI stage-1 frames, not from annotations.csv. An experiment whose
labels are required (LAYOUTS) must have every kept video in its annotations.csv; v4_infested (53 of 119 videos
annotated, first 10 min only) is not required to. Checks (fail loud): observation ids unique over all experiments;
every annotated video is valid; on every annotated row of the window the frame_path and T agree with the generated
row; the first and last frame JPG of every video exist (--no-frame-check skips it, for a dry run before the frames
are extracted); each video one contiguous block in increasing frame_idx. Existing outputs are kept unless --overwrite.

Usage: python scripts/eci/ants_prepare.py [--overwrite]
       ANTS_ECI=eci30 python scripts/eci/ants_prepare.py
       ANTS_ECI=eci30 python scripts/eci/ants_prepare.py --out-dir <test dir> --no-frame-check [--experiments v2,v3]
       (dry run)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.eci.contrasts import observation_blocks  # noqa: E402
from src.eci.domain import get_domain  # noqa: E402

FPS = 5.0
# layout -> {experiment: (treatment values kept (None = all), frames per video at 5 fps = the analysis window,
#                         labels required)}
LAYOUTS = {'eci': {'v2': (None, 3000, True), 'v3': (None, 3000, True)},
           'eci30': {'v2': (None, 3000, True), 'v3': ((2, 6, 8), 9000, True), 'v4': ((2, 6), 9000, True),
                     'v4_infested': ((2, 6), 9000, False)}}
INFESTED = ('v4_infested',)  # experiments of infested colonies (design column infested = 1)
DESIGN_COLS = ['observation_id', 'experiment', 'T', 'batch', 'position', 'annotator', 'recording_date', 'nestbox',
               'infested']


def load_experiment(name, path, keep=None):
    """Valid videos of one experiment (treatment in keep, None = all) -> DESIGN_COLS (nestbox missing in v2 -> NA)."""
    e = pd.read_csv(path, dtype={'observation_id': str, 'batch': str, 'position': str})
    e = e[e['valid'] == 1].rename(columns={'treatment': 'T'}).assign(experiment=name, infested=int(name in INFESTED))
    if keep is not None:
        e = e[e['T'].isin(keep)]
    if 'nestbox' not in e:
        e['nestbox'] = pd.NA
    return e[DESIGN_COLS].reset_index(drop=True)


def frame_rows(name, e, n):
    """Frames 0 .. n - 1 of every video of e: observation_id, frame_idx, fps, frame_path, experiment, T."""
    oid = np.repeat(e['observation_id'].values, n)
    fi = np.tile(np.arange(n), len(e))
    return pd.DataFrame({'observation_id': oid, 'frame_idx': fi, 'fps': FPS,
                         'frame_path': [f'ants/{name}/frames/full/{o}/frame_{i:06d}.jpg' for o, i in zip(oid, fi)],
                         'experiment': name, 'T': np.repeat(e['T'].values, n)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--out-dir', default=None, help='default: the domain eci dir (dataset/ants/<ANTS_ECI>)')
    ap.add_argument('--no-frame-check', action='store_true', help='skip the first / last frame JPG check')
    ap.add_argument('--experiments', default=None, help='comma list, a subset of the layout (dry runs only)')
    args = ap.parse_args()
    D = get_domain('ants')
    if D.ECI not in LAYOUTS:
        raise SystemExit(f'ANTS_ECI={D.ECI}: no layout (known: {sorted(LAYOUTS)})')
    out = Path(args.out_dir) if args.out_dir else D.eci_dir
    ann_path, exp_path = out / 'annotations.csv', out / 'experiment.csv'
    if ann_path.exists() and exp_path.exists() and not args.overwrite:
        print(f'[SKIP] {ann_path} and {exp_path} exist (--overwrite to rebuild)')
        return
    out.mkdir(parents=True, exist_ok=True)

    frames, videos, missing = [], [], []
    layout = LAYOUTS[D.ECI]
    if args.experiments:
        if not args.out_dir:
            raise SystemExit('--experiments is for dry runs: give --out-dir')
        layout = {k: v for k, v in layout.items() if k in args.experiments.split(',')}
    for name, (keep, n, need_labels) in layout.items():
        e_all = load_experiment(name, D.raw_experiments[name])
        e = load_experiment(name, D.raw_experiments[name], keep)
        f = frame_rows(name, e, n)
        src = D.sources[name]
        ycols = []
        if src.exists():
            a = pd.read_csv(src, dtype={'observation_id': str})
            ids_a = set(a['observation_id'])
            if ids_a - set(e_all['observation_id']):
                raise SystemExit(f'{name}: annotated but not valid {sorted(ids_a - set(e_all["observation_id"]))}')
            if need_labels and set(e['observation_id']) - ids_a:
                raise SystemExit(f'{name}: kept videos not in {src}: {sorted(set(e["observation_id"]) - ids_a)}')
            ycols = [c for c in a.columns if c.startswith('Y_')]
            a = a[a['observation_id'].isin(set(e['observation_id'])) & (a['frame_idx'] < n)]
            j = f[['observation_id', 'frame_idx', 'frame_path', 'T']].merge(
                a[['observation_id', 'frame_idx', 'frame_path', 'T'] + ycols], on=['observation_id', 'frame_idx'],
                how='left', suffixes=('', '_a'), validate='1:1')
            lab = j['frame_path_a'].notna().values
            if len(a) != lab.sum():
                raise SystemExit(f'{name}: {len(a) - lab.sum()} annotated rows of the window are not frames 0..{n - 1}')
            if not (j.loc[lab, 'frame_path'] == j.loc[lab, 'frame_path_a']).all():
                raise SystemExit(f'{name}: frame_path of {src} differs from ants/{name}/frames/full/<obs>/frame_<idx>.jpg')
            if not (j.loc[lab, 'T'] == j.loc[lab, 'T_a']).all():
                raise SystemExit(f'{name}: per-frame T of {src} differs from experiment.csv treatment')
            f = f.join(j[ycols].astype('Int64'))
            n_lab = int(a['observation_id'].nunique())
        elif need_labels:
            raise SystemExit(f'{name}: {src} missing (labels required)')
        else:
            n_lab = 0
        if not args.no_frame_check:
            for o in e['observation_id']:
                for i in (0, n - 1):
                    p = ROOT / 'dataset' / f'ants/{name}/frames/full/{o}/frame_{i:06d}.jpg'
                    if not p.exists():
                        missing.append(str(p.relative_to(ROOT)))
        frames.append(f)
        videos.append(e)
        print(f'{name}: {len(e)} videos (treatments {"all" if keep is None else keep}), {n} frames each = {len(f)} rows, '
              f'{n_lab} videos with label rows, T counts {e["T"].value_counts().sort_index().to_dict()}, '
              f'labels {ycols}', flush=True)
    if missing:
        raise SystemExit(f'{len(missing)} frame JPGs missing, e.g. {missing[:5]} (frames not extracted yet?)')
    ann = pd.concat(frames, ignore_index=True)
    ycols = sorted(c for c in ann.columns if c.startswith('Y_'))
    ann = ann[['observation_id', 'frame_idx', 'fps', 'frame_path', 'experiment', 'T'] + ycols]
    design = pd.concat(videos, ignore_index=True)
    if design['observation_id'].duplicated().any():
        raise SystemExit(f'observation ids shared by two experiments: '
                         f'{design[design.observation_id.duplicated()].observation_id.tolist()}')

    ann.to_csv(ann_path, index=False)
    blocks = observation_blocks(ann_path)  # re-read: contiguous blocks, increasing frame_idx
    if not np.array_equal(blocks['observation_id'].values, design['observation_id'].values):
        raise SystemExit('video blocks of annotations.csv are not in experiment.csv order')
    design.to_csv(exp_path, index=False)
    print(f'wrote {ann_path}: {len(ann)} rows, {len(blocks)} videos; {exp_path}: {len(design)} videos')
    print(design.groupby(['experiment', 'T', 'recording_date']).size().to_string())


if __name__ == '__main__':
    main()
