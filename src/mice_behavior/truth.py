"""The human ground truth for mice v1, read the one way every downstream estimate must read it.

WHY THIS MODULE EXISTS
======================
`dataset/mice/v1/annotations.csv` carries THREE behaviour columns, and one of them is not a
third behaviour:

    Y_nt   nose-to-tail
    Y_nn   nose-to-nose, MUTUAL       (BORIS `nn`,  reciprocated)
    Y_np   nose-to-nose, DIRECTIONAL  (BORIS `np`,  one-sided; the passive animal stays 0)

`np` is the one-sided form of the same nose-to-nose contact, not a separate behaviour and not
anything anogenital -- see `build_pair_labels.py`'s docstring for the cross-tabulation that
settles it. The unit of a label is a DIRECTED pair (i -> j), so under that definition the two
codes are one class at different reciprocity, and `build_pair_labels.LABEL_MAP` maps both to
label 2.

The model is trained, validated, thresholded and scored on that union. Everything that reads
the truth back to compare against the model therefore has to read the SAME union, and for a
long time ten separate call sites did not: they took the raw `Y_nn` column, which is
mutual-only. Measured on the 144 annotated observations that is not a rounding difference --
mutual-only fires on 0.754% of frames against the union's 2.114%, and at frame level the
model's own nn target disagrees with mutual-only on 11,754 frames while disagreeing with the
union on 1 (a single boundary frame, and `Y_nt` disagrees with the model's nt target on the
same one, so it is a shared off-by-one in the two pipelines, not a label question).

Read the truth through `read_truth()` rather than through `pd.read_csv` so that definition
lives in one place.

WHY THIS ALWAYS DROPS THE UNLABELLED ROWS
=========================================
288 of v1's 432 observations are unannotated and carry NaN, not 0, in all three columns. The
union is an OR, and `NaN > 0.5` is False, so computing it on a frame that still holds the
unlabelled rows would silently turn "no annotation" into "no behaviour" -- the exact
zero-fill mistake the callers were written to avoid. Dropping first makes that unreachable.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

# The raw columns the nn truth is assembled from. Both are nose-to-nose; see the module
# docstring. Nothing outside this module should name Y_np.
NN_RAW = ('Y_nn', 'Y_np')


def annotations_csv(version: str = 'v1', dataset_dir=None) -> Path:
    return Path(dataset_dir or (ROOT / 'dataset')) / 'mice' / version / 'annotations.csv'


def read_truth(extra_cols=(), version: str = 'v1', dataset_dir=None, path=None) -> pd.DataFrame:
    """Per-frame human labels on the annotated observations, with `Y_nn` as the DIRECTED-PAIR
    UNION of the mutual and directional nose-to-nose codes -- the target the model is scored on.

    Returns observation_id, frame_idx, Y_nt, Y_nn (+ `extra_cols`), unlabelled rows dropped,
    `Y_nn` still float 0.0/1.0 so the callers' `v == 1` bout-onset tests are unchanged.
    """
    extra = [c for c in extra_cols if c not in ('observation_id', 'frame_idx', 'Y_nt', *NN_RAW)]
    a = pd.read_csv(path or annotations_csv(version, dataset_dir),
                    usecols=['observation_id', 'frame_idx', 'Y_nt', *NN_RAW, *extra],
                    low_memory=False).dropna(subset=['Y_nt'])
    a['Y_nn'] = ((a['Y_nn'] > 0.5) | (a['Y_np'] > 0.5)).astype(float)
    return a.drop(columns=['Y_np'])
