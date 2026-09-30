"""
Per-frame and per-video tables of the ants ECI domain (src/eci/domain.py AntsDomain): the v2 and v3
experiments pooled, so one SAE and one codes memmap cover both.

    dataset/ants/eci/annotations.csv   one row per frame, v2 rows then v3 rows (each in its own
                                       annotations.csv order): observation_id, frame_idx, fps,
                                       frame_path, experiment (v2 / v3), T (raw treatment value),
                                       Y_* (behaviour labels, union of both; blank where an experiment
                                       has no such label). Row order = the row order of every ants codes
                                       memmap, so it is written once and never reordered.
    dataset/ants/eci/experiment.csv    one row per valid video (data/ants/{v2,v3}/experiment.csv,
                                       valid == 1): observation_id, experiment, T, batch, position,
                                       annotator, recording_date, nestbox (v3 only, blank for v2).

Checks (fail loud): observation ids unique over both experiments; every annotated video is valid and
every valid video is annotated; T of every frame equals the video's treatment; frame_idx increasing
within a video; each video one contiguous block. Existing outputs are kept unless --overwrite.

Usage: python scripts/eci/ants_prepare.py [--overwrite]
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

EXPERIMENTS = ('v2', 'v3')
DESIGN_COLS = ['observation_id', 'experiment', 'T', 'batch', 'position', 'annotator', 'recording_date', 'nestbox']


def load_experiment(name, path):
    """Valid videos of one experiment -> DESIGN_COLS (nestbox missing in v2 -> NA)."""
    e = pd.read_csv(path, dtype={'observation_id': str, 'batch': str, 'position': str})
    e = e[e['valid'] == 1].rename(columns={'treatment': 'T'}).assign(experiment=name)
    if 'nestbox' not in e:
        e['nestbox'] = pd.NA
    return e[DESIGN_COLS].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()
    D = get_domain('ants')
    if D.ann_path.exists() and D.experiment_csv.exists() and not args.overwrite:
        print(f'[SKIP] {D.ann_path} and {D.experiment_csv} exist (--overwrite to rebuild)')
        return
    D.eci_dir.mkdir(parents=True, exist_ok=True)

    frames, videos = [], []
    for name in EXPERIMENTS:
        e = load_experiment(name, D.raw_experiments[name])
        a = pd.read_csv(D.sources[name], dtype={'observation_id': str})
        ids_a, ids_e = set(a['observation_id']), set(e['observation_id'])
        if ids_a != ids_e:
            raise SystemExit(f'{name}: annotated but not valid {sorted(ids_a - ids_e)}, '
                             f'valid but not annotated {sorted(ids_e - ids_a)}')
        t = a['observation_id'].map(e.set_index('observation_id')['T'])
        if not (a['T'].values == t.values).all():
            raise SystemExit(f'{name}: per-frame T differs from experiment.csv treatment')
        ycols = [c for c in a.columns if c.startswith('Y_')]
        frames.append(a[['observation_id', 'frame_idx', 'fps', 'frame_path']].assign(experiment=name, T=a['T'])
                      .join(a[ycols].astype('Int64')))
        videos.append(e)
        print(f'{name}: {len(e)} videos, {len(a)} frames, T counts {e["T"].value_counts().sort_index().to_dict()}, '
              f'labels {ycols}', flush=True)
    ann = pd.concat(frames, ignore_index=True)
    ycols = sorted(c for c in ann.columns if c.startswith('Y_'))
    ann = ann[['observation_id', 'frame_idx', 'fps', 'frame_path', 'experiment', 'T'] + ycols]
    design = pd.concat(videos, ignore_index=True)
    if design['observation_id'].duplicated().any():
        raise SystemExit(f'observation ids shared by v2 and v3: {design[design.observation_id.duplicated()].observation_id.tolist()}')

    ann.to_csv(D.ann_path, index=False)
    blocks = observation_blocks(D.ann_path)  # re-read: contiguous blocks, increasing frame_idx
    if not np.array_equal(blocks['observation_id'].values, design['observation_id'].values):
        design = design.set_index('observation_id').loc[blocks['observation_id']].reset_index()
    design.to_csv(D.experiment_csv, index=False)
    print(f'wrote {D.ann_path}: {len(ann)} rows, {len(blocks)} videos; {D.experiment_csv}: {len(design)} videos')
    print(design.groupby(['experiment', 'T']).size().to_string())


if __name__ == '__main__':
    main()
