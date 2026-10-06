"""Synthetic unit tests for the identity layer.

Test 1 (swap through a merge): animals A and B walk towards each other, their boxes overlap for
~1 s, and the tracker SWAPS track ids inside the merge (track 0 follows A before and B after).
Required: each track is cut into separate tracklets either side of the merge, and the identities
after the merge are recovered from marks (track 0 after the merge = B).

Test 2 (linking error without a merge): A and B walk in parallel 80 px apart (boxes never touch,
the jump is below the speed limit, so no cut), and the track ids are swapped mid-way. Required:
metric B flags the tracklet as contradictory.

Test 3 (elimination): 3 animals; animal C's marks are never readable; A and B are confirmed, so C
must be 'inferred' while all three are present, and the inferred label must be right.

    python test_synthetic.py
"""
import numpy as np
import pandas as pd

from assign import DEFAULTS, run_assignment, per_frame
from metrics import consistency

RNG = np.random.default_rng(0)


def tracks_df(rows):
    df = pd.DataFrame(rows, columns=['frame_src', 'track_id', 'cx', 'cy', 'true'])
    df['t_sec'] = df.frame_src / 30.0
    df['w'] = df['h'] = 60.0
    df['heading'] = 0.0
    df['detected'] = True
    df['conf'] = 1.0
    df['variant'] = 'synthetic'
    return df


def reads_for(df, ids, p_true=0.85, unreadable=()):
    """One read per row; readable when isolated (decided by geometry later -> here: by distance)."""
    rows = []
    for r in df.itertuples():
        others = df[(df.frame_src == r.frame_src) & (df.track_id != r.track_id)]
        iso = (np.hypot(others.cx - r.cx, others.cy - r.cy) > 70).all()
        p = np.full(len(ids), (1 - p_true) / (len(ids) - 1))
        p[ids.index(r.true)] = p_true
        p = RNG.dirichlet(p * 30)  # noisy probabilities around the truth
        rows.append({'frame_src': r.frame_src, 'track_id': r.track_id,
                     'readable': bool(iso and r.true not in unreadable),
                     **{f'p_{i}': p[k] for k, i in enumerate(ids)}})
    return pd.DataFrame(rows)


def test_swap_through_merge():
    ids, rows = ['A', 'B'], []
    for f in range(0, 600, 3):
        xa, xb = 100 + f * 0.7, 520 - f * 0.7  # meet around f = 300
        ta, tb = (0, 1) if f < 300 else (1, 0)  # tracker swaps ids inside the merge
        rows += [(f, ta, xa, 200, 'A'), (f, tb, xb, 205, 'B')]
    tr0 = tracks_df(rows)
    reads = reads_for(tr0, ids)
    tr, tab = run_assignment(tr0, reads, ids, 2, DEFAULTS)
    pf = per_frame(tr, tab, ids)
    # contact frames
    cont = tr[tr.in_contact].frame_src
    c0, c1 = cont.min(), cont.max()
    t0 = tr[tr.track_id == 0]
    before = t0[t0.frame_src < c0].tracklet.unique()
    after = t0[t0.frame_src > c1].tracklet.unique()
    assert len(before) == 1 and len(after) == 1 and before[0] != after[0], (before, after)
    assert tab.at[after[0], 'identity'] == 'B', tab.loc[after[0]]
    assert tab.at[before[0], 'identity'] == 'A'
    # per-frame identity A after the merge sits on track 1
    a_after = pf[(pf.identity == 'A') & (pf.frame_src > c1)]
    assert (a_after.track_id == 1).all() and (a_after.state == 'confirmed').all()
    in_merge = pf[(pf.frame_src >= c0) & (pf.frame_src <= c1)]
    print(f'[test 1] contact frames {c0}-{c1}; track 0 tracklets before/after = '
          f'{before[0]}/{after[0]}; identities {tab.at[before[0], "identity"]} -> '
          f'{tab.at[after[0], "identity"]}; states inside merge: '
          f'{in_merge.state.value_counts().to_dict()}  PASS')
    return tr, tab


def test_linking_error_caught():
    ids, rows = ['A', 'B'], []
    for f in range(0, 900, 3):
        x = 100 + f * 0.5
        ta, tb = (0, 1) if f < 450 else (1, 0)  # id swap with NO merge
        rows += [(f, ta, x, 200, 'A'), (f, tb, x, 280, 'B')]
    tr0 = tracks_df(rows)
    reads = reads_for(tr0, ids)
    tr, tab = run_assignment(tr0, reads, ids, 2, DEFAULTS)
    assert tab.shape[0] == 2, f'expected no cut, got {len(tab)} tracklets'
    m = consistency(tr, tab, reads, ids, tr0, 2, DEFAULTS)
    assert m['n_contradictory_tracklets'] == 2, m
    # control: same scene without the swap -> no contradiction
    rows2 = [(f, 0 if t == 'A' else 1, x, y, t) for f, _, x, y, t in rows]
    tr0b = tracks_df(rows2)
    reads_b = reads_for(tr0b, ids)
    trb, tabb = run_assignment(tr0b, reads_b, ids, 2, DEFAULTS)
    mb = consistency(trb, tabb, reads_b, ids, tr0b, 2, DEFAULTS)
    assert mb['n_contradictory_tracklets'] == 0, mb
    print(f'[test 2] injected swap: {m["n_contradictory_tracklets"]}/2 tracklets flagged '
          f'contradictory (purity {m["mean_vote_purity"]:.2f}); no-swap control: '
          f'{mb["n_contradictory_tracklets"]} flagged (purity {mb["mean_vote_purity"]:.2f}), '
          f'held-out agreement {mb["heldout_even_odd"]["assigned_any"]["agreement"]:.3f}  PASS')


def test_elimination():
    ids, rows = ['A', 'B', 'C'], []
    for f in range(0, 600, 3):
        rows += [(f, 0, 100 + f * 0.2, 100, 'A'), (f, 1, 100, 400 + f * 0.1, 'B'),
                 (f, 2, 500, 500, 'C')]
    tr0 = tracks_df(rows)
    reads = reads_for(tr0, ids, unreadable=('C',))
    tr, tab = run_assignment(tr0, reads, ids, 3, DEFAULTS)
    k = tr[tr.track_id == 2].tracklet.iloc[0]
    assert tab.at[k, 'identity'] == 'C' and tab.at[k, 'state'] == 'inferred', tab
    print(f'[test 3] unreadable animal inferred by elimination as {tab.at[k, "identity"]} '
          f'(conf {tab.at[k, "conf"]:.3f})  PASS')


if __name__ == '__main__':
    test_swap_through_merge()
    test_linking_error_caught()
    test_elimination()
    print('all synthetic tests passed')
