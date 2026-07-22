"""
Ground-truth observation_id -> pool mapping for mice v1.

The naive approach of parsing pool from observation_id as line_sex_seed
(previously used across run_train.py/diagnose.py/grid_search.py) is NOT
unique: 6 of those derived keys each silently merge 2 distinct physical
mouse quadruplets that happen to share line+sex+seed. Use the literal
`pool` column in experiment.csv instead — 22 real annotated pools, not 16.
"""
import hashlib
from pathlib import Path

import pandas as pd


def load_obs_to_pool_map(data_dir='./data', version='v1') -> dict:
    exp = pd.read_csv(Path(data_dir) / 'mice' / version / 'experiment.csv')
    return dict(zip(exp['observation_id'], exp['pool']))


def get_val_pools(pools, val_frac: float = 0.2, seed: int = 42) -> set:
    """Stable train/val pool split, robust to the pool set growing over time.

    Every mice_behavior training/search script previously did
    `rng.shuffle(sorted(pools)); val = shuffled[:n_val]` with a fixed seed — but
    Fisher-Yates shuffle output depends on the FULL LIST LENGTH, not just each
    element's identity. Adding one new annotated pool (e.g. rd64) changes len(pools)
    and silently reshuffles which OTHER pools land in val — confirmed in practice:
    adding rd64 swapped rd14/rd35_3 out of val for rd32/rd64, which happened to
    roughly halve nt-behavior prevalence in val and made every subsequent search's
    numbers look like a regression when it was actually just a harder, incomparable
    validation set.

    Fix: hash each pool independently (seed+pool_id, not the pool LIST), so an
    existing pool's assignment never changes when new pools are added — inserting
    a new pool can only add it to train or val, or shift the val/train boundary by
    at most one pool near the cutoff, never reshuffle arbitrary existing pools.
    """
    def _pool_score(pool_id):
        h = hashlib.sha256(f'{seed}-{pool_id}'.encode()).hexdigest()
        return int(h, 16)

    ordered = sorted(pools, key=_pool_score)
    n_val = max(1, round(len(ordered) * val_frac))
    return set(ordered[:n_val])


def get_kfold_assignment(pools, k: int = 5, seed: int = 42) -> dict:
    """Assigns every pool to a fold index 0..k-1, independently per pool (same
    insertion-stability rationale as get_val_pools) — a new pool only ever adds one
    new assignment, never disturbs which fold an existing pool belongs to."""
    def _fold_score(pool_id):
        h = hashlib.sha256(f'{seed}-kfold-{pool_id}'.encode()).hexdigest()
        return int(h, 16)

    return {p: _fold_score(p) % k for p in pools}
