"""
Parse raw BORIS annotation CSVs into per-frame per-pair behavior labels.

Output: dataset/mice/v1/pair_labels.parquet
Schema: observation_id, frame_idx (5fps), agent1 (0-3), agent2 (0-3), label (1=nt, 2=nn)
Only positive (non-none) rows are stored.
"""
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE_FPS = 30
TARGET_FPS = 5
FPS_RATIO = SOURCE_FPS // TARGET_FPS  # 6

# nn and np both collapse to label=2 (nn class); nt → label=1
LABEL_MAP = {'nn': 2, 'np': 2, 'np?': 2, 'nt': 1}


def build_pair_labels(data_dir='./data', dataset_dir='./dataset', overwrite=False):
    data_dir = Path(data_dir)
    dataset_dir = Path(dataset_dir)
    out = dataset_dir / 'mice' / 'v1' / 'pair_labels.parquet'

    if out.exists() and not overwrite:
        existing = pd.read_parquet(out)
        print(f'[SKIP] {out} exists ({len(existing):,} rows)')
        return out

    exp = pd.read_csv(data_dir / 'mice' / 'v1' / 'experiment.csv')
    ann_dir = data_dir / 'mice' / 'v1' / 'annotations'

    annotated = exp[
        exp['annotation_file'].notna()
        & (exp['annotation_file'].astype(str).str.strip() != '')
    ]

    records = []
    for _, row in annotated.iterrows():
        obs_id = row['observation_id']
        ann_path = ann_dir / str(row['annotation_file']).strip()
        start_offset = int(row['start_frame'])
        end_frame = int(row['end_frame'])
        n_frames_5fps = round((end_frame - start_offset) / FPS_RATIO)

        if not ann_path.exists():
            print(f'[WARN] missing annotation file: {ann_path.name}')
            continue

        try:
            boris = pd.read_csv(ann_path, skipfooter=1, engine='python')
        except Exception as e:
            print(f'[WARN] failed to parse {ann_path.name}: {e}')
            continue

        for _, b in boris.iterrows():
            behavior = str(b['behavior_type']).strip()
            if behavior not in LABEL_MAP:
                continue
            label = LABEL_MAP[behavior]
            a1 = int(b['agent1(active)']) - 1  # convert 1-4 → 0-3
            a2 = int(b['agent2']) - 1
            if not (0 <= a1 <= 3 and 0 <= a2 <= 3):
                continue  # skip annotation errors (agent index out of quadruplet range)
            src_start = int(b['Image index start'])
            src_end = int(b['Image index stop'])

            # 5fps frame i corresponds to source frame start_offset + i*FPS_RATIO.
            # Include frame i when: src_start <= start_offset + i*FPS_RATIO < src_end
            i_start = max(0, math.ceil((src_start - start_offset) / FPS_RATIO))
            i_end = min(n_frames_5fps, math.ceil((src_end - start_offset) / FPS_RATIO))
            if i_start >= i_end:
                continue

            # nn is symmetric: label both (a1→a2) and (a2→a1)
            pairs = [(a1, a2), (a2, a1)] if behavior == 'nn' else [(a1, a2)]

            for fi in range(i_start, i_end):
                for p1, p2 in pairs:
                    records.append((obs_id, fi, p1, p2, label))

    df = (
        pd.DataFrame(records, columns=['observation_id', 'frame_idx', 'agent1', 'agent2', 'label'])
        .astype({'frame_idx': 'int32', 'agent1': 'int8', 'agent2': 'int8', 'label': 'int8'})
        .drop_duplicates(['observation_id', 'frame_idx', 'agent1', 'agent2'])
        .sort_values(['observation_id', 'frame_idx', 'agent1', 'agent2'])
        .reset_index(drop=True)
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f'Saved {len(df):,} positive pair-label rows → {out}')
    print(f'  label counts: {df["label"].value_counts().to_dict()}')
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='./data')
    p.add_argument('--dataset-dir', default='./dataset')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    build_pair_labels(args.data_dir, args.dataset_dir, args.overwrite)
