"""Default head hyperparameters for the mice frame/pair classifiers.

These values used to be read at runtime out of a RESULT directory --
`results/vision/mice/frame/patchgrid4x4_dinov2/config.json` -- by six scripts including the
live trainer. That made a deletable, gitignored artifact directory load-bearing for training,
and it silently coupled every new run to one 2026-07 grid search. They are inlined here so the
code owns its defaults and `results/` stays disposable.

PROVENANCE AND HEALTH WARNING. This is the winning trial of a grid search run against:
  * a coarse 4x4 (16-token) patch grid, with encoder tokens cached ahead of time
  * the retired 4-pool validation split rd14 / rd19 / rd29 / rd35_3
  * neg_ratio 15, i.e. 15 sampled negatives per positive
It scored 0.1971 best monitor macro AP in that setting. Current runs use a 32x32 (1024-token)
grid at 448px, the VAL_POOLS_V1 split, neg_ratio 1, online encoding and (since 2026-08-13)
a partly unfrozen encoder. Every one of the conditions this was tuned under has since changed,
so treat these as an arbitrary-but-reproducible starting point, NOT as tuned values.

`neg_ratio`, `context_k` and `stride` are carried for completeness but every current entry
point passes its own on the command line, so in practice only n_heads / hidden_dim / dropout /
weight_decay / lr are still inherited -- and --lr / --weight-decay / --dropout override those.

The 32-trial successive-halving search in search_online_aug.py is the natural replacement;
promote its winner here once it has been re-run at 448px (it was searched at 504px).
"""

# Winning trial, patchgrid4x4_dinov2 grid search (2026-07). See warning above.
DEFAULT_HEAD_CFG = {
    'n_heads': 8,
    'context_k': 2,
    'stride': 1,
    'hidden_dim': 384,
    'neg_ratio': 15,
    'dropout': 0.4,
    'weight_decay': 0.0001,
    'lr': 0.0003,
}

# Best monitor macro AP that trial reached, on the retired 4x4-era split. Retained only so
# retrain_patchgrid_fulldata.py can still quote its historical baseline; not comparable to
# anything scored on VAL_POOLS_V1.
LEGACY_4X4_BASELINE_AP = 0.19708005315311883


def get_head_cfg() -> dict:
    """A fresh copy, so a caller mutating its config can't poison the module-level default."""
    return dict(DEFAULT_HEAD_CFG)
