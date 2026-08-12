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


# Standing validation set for mice v1, fixed 2026-08-12. Use this, not the hash split.
#
# Composition: 1 wt + 3 het -- proportional to the annotated population (6 wt : 18 het), and
# it keeps 5 of the 6 scarce wt pools in training. Chosen by DESIGN constraints only (1 wt,
# 3 het, all three lines ash1l/kdm6b/kmt5b, 2 female + 2 male), then taken as the first of
# the 918 qualifying 4-pool combinations in sorted order. Deliberately NOT selected on
# behaviour rates: picking pools by their outcome would inflate the very rate-correlation
# metric this set exists to measure.
#
# Why 4 pools suffices even though pool-mean rates are nearly flat here (nn spans just
# 1.59-1.86% across the four): the metric is computed over the 24 OBSERVATIONS, and a
# variance decomposition over all 144 annotated observations shows 57% (nt) / 73% (nn) of
# rate variance is WITHIN pool, driven by the odor x phase conditions. At observation level
# this set spans nt 0.05-3.32% (sd 0.83) and nn 0.48-4.50% (sd 1.05), retaining 66% / 56%
# of the spread of all 144 observations.
#
# Caveat to state whenever this is used: with only 4 pools, between-pool generalisation is
# estimated from 4 clusters, so a cluster bootstrap over pools gives wide intervals. The set
# is adequate for ranking configs against each other, not for an absolute claim.
#
# Leaves 20 pools / 120 observations for training (5 wt + 15 het).
#
# NOTE: results measured on this split are NOT comparable to anything produced before
# 2026-08-12, which used get_val_pools()'s hash split (rd11_2/rd32/rd34/rd35_2/rd41_3 --
# only ONE wt pool, and no design balance). Three of the four pools here were in TRAIN
# under that split, so old checkpoints are contaminated w.r.t. this val set and must be
# retrained, not merely re-scored.
VAL_POOLS_V1 = frozenset({'rd11_2', 'rd13', 'rd14', 'rd18'})


def get_fixed_val_pools(pools=None) -> set:
    """The standing v1 validation set (see VAL_POOLS_V1). Prefer this over get_val_pools()."""
    if pools is not None:
        missing = VAL_POOLS_V1 - set(pools)
        if missing:
            raise ValueError(f'fixed val pools missing from the available pool set: {sorted(missing)}')
    return set(VAL_POOLS_V1)


def get_val_pools(pools, val_frac: float = 0.2, seed: int = 42) -> set:
    """DEPRECATED for new work -- use get_fixed_val_pools(). Retained only to reproduce
    results from before 2026-08-12. Its split has just one wt pool of six and no balance
    across line/sex, so it cannot support genotype-aware validation.

    Stable train/val pool split, robust to the pool set growing over time.

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
