"""
Ground-truth observation_id -> pool mapping for mice v1.

The naive approach of parsing pool from observation_id as line_sex_seed
(previously used across run_train.py/diagnose.py/grid_search.py) is NOT
unique: 6 of those derived keys each silently merge 2 distinct physical
mouse quadruplets that happen to share line+sex+seed. Use the literal
`pool` column in experiment.csv instead — 22 real annotated pools, not 16.
"""
from pathlib import Path

import pandas as pd


def load_obs_to_pool_map(data_dir='./data', version='v1') -> dict:
    exp = pd.read_csv(Path(data_dir) / 'mice' / version / 'experiment.csv')
    return dict(zip(exp['observation_id'], exp['pool']))
