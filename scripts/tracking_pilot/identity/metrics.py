"""Evaluation metrics for identity-tracked multi-animal output. Nothing here is tuned on the
external ground truth (section C): C is computed after all thresholds are fixed.

A. completeness      frames with exactly N detections, frames with N positions present (rows of
                     any kind, i.e. after the tracker's own filling), frame-identity slot states.
B. consistency       per tracklet with >= 2 readable reads: vote purity; contradictory tracklets
                     (a confident switch inside a tracklet = linking error); held-out check
                     (assign with even-second reads only, score agreement on odd-second reads).
C. external          ants: during annotated Y2F / B2F bouts, how often the assigned yellow / blue is
                     within contact distance of the assigned focal (wrong colour as baseline).
                     mice: per BORIS event, is the annotated pair the closest of the 6 pairs, for
                     every one of the 24 numbering maps annotator index -> shave identity.
"""
from itertools import permutations

import numpy as np
import pandas as pd

from assign import run_assignment, per_frame

CONF_P = 0.8  # a read is 'confident' if its max probability >= this


# ----------------------------------------------------------------------------- A
def completeness(tr, pf, N):
    by_f = tr.groupby('frame_src')
    n_det = by_f.detected.sum()
    n_any = by_f.size()
    st = pf.state.value_counts(normalize=True).reindex(['confirmed', 'inferred', 'unknown']).fillna(0)
    per_id = pf.groupby('identity').state.value_counts(normalize=True).unstack().fillna(0)
    all_known = pf.assign(k=pf.state != 'unknown').groupby('frame_src').k.all()
    return {
        'n_frames': int(len(n_det)),
        'pct_frames_exactly_N_detected': float((n_det == N).mean() * 100),
        'pct_frames_fewer_than_N_detected': float((n_det < N).mean() * 100),
        'pct_frames_more_than_N_detected': float((n_det > N).mean() * 100),
        'pct_frames_N_positions_present': float((n_any >= N).mean() * 100),
        'pct_slots': {k: float(v * 100) for k, v in st.items()},
        'pct_slots_per_identity': {i: {k: float(per_id.loc[i].get(k, 0) * 100) for k in
                                       ['confirmed', 'inferred', 'unknown']} for i in per_id.index},
        'pct_frames_all_identities_known': float(all_known.mean() * 100),
    }


# ----------------------------------------------------------------------------- B
def _reads_on_tracklets(tr, reads):
    r = reads[reads.readable].merge(tr[['frame_src', 'track_id', 'tracklet']],
                                    on=['frame_src', 'track_id'])
    return r[r.tracklet >= 0]


def contradiction(labels, conf, min_side=3, purity=0.8):
    """labels: time-ordered argmax of confident reads in one tracklet. A tracklet is contradictory
    iff a single change point splits it into two blocks, each >= min_side reads, each with
    purity >= `purity`, whose majority labels differ (a block switch, which is what a linking
    error looks like; scattered scorer noise does not form two pure blocks)."""
    lab = np.asarray(labels)[np.asarray(conf)]
    n = len(lab)
    if n < 2 * min_side:
        return False, None
    for c in range(min_side, n - min_side + 1):
        a, b = lab[:c], lab[c:]
        va, ca = np.unique(a, return_counts=True)
        vb, cb = np.unique(b, return_counts=True)
        ma, mb = va[ca.argmax()], vb[cb.argmax()]
        if ma != mb and ca.max() / len(a) >= purity and cb.max() / len(b) >= purity:
            return True, c
    return False, None


def consistency(tr, tab, reads, ids, tracks, N, P):
    r = _reads_on_tracklets(tr, reads)
    pc = r[[f'p_{i}' for i in ids]].values
    r = r.assign(lab=np.array(ids)[pc.argmax(1)], pmax=pc.max(1))
    pur, n_contra, n_eval, contra_ids = [], 0, 0, []
    for k, g in r.sort_values('frame_src').groupby('tracklet'):
        if len(g) < 2:
            continue
        n_eval += 1
        vc = g.lab.value_counts()
        pur.append(vc.iloc[0] / len(g))
        c, _ = contradiction(g.lab.values, g.pmax.values >= CONF_P)
        if c:
            n_contra += 1
            contra_ids.append(int(k))
    # held-out: assign from even-second reads, test on odd-second reads
    even = (np.floor(reads.frame_src / 30.0) % 2 == 0)
    tr2, tab2 = run_assignment(tracks, reads, ids, N, P, keep=even)
    odd = _reads_on_tracklets(tr2, reads[~even])
    odd = odd.merge(tab2[['identity', 'state']], left_on='tracklet', right_index=True)
    po = odd[[f'p_{i}' for i in ids]].values
    odd = odd.assign(lab=np.array(ids)[po.argmax(1)], pmax=po.max(1))
    odd_c = odd[odd.pmax >= CONF_P]
    held = {}
    for s in ['confirmed', 'inferred']:
        g = odd_c[odd_c.state == s]
        held[s] = {'n_odd_confident_reads': int(len(g)),
                   'agreement': float((g.lab == g.identity).mean()) if len(g) else None}
    g = odd_c[odd_c.state != 'unknown']
    held['assigned_any'] = {'n_odd_confident_reads': int(len(g)),
                            'agreement': float((g.lab == g.identity).mean()) if len(g) else None}
    held['n_odd_confident_reads_on_unknown_tracklets'] = int((odd_c.state == 'unknown').sum())
    return {
        'n_tracklets': int(len(tab)),
        'n_tracklets_ge2_reads': int(n_eval),
        'mean_vote_purity': float(np.mean(pur)) if pur else None,
        'median_vote_purity': float(np.median(pur)) if pur else None,
        'n_contradictory_tracklets': int(n_contra),
        'contradictory_tracklets': contra_ids[:50],
        'n_conflicts_overlapping_strong_votes': int(tab.conflict.sum()),
        'tracklet_cut_reasons': {str(k): int(v) for k, v in tab.start_reason.value_counts().items()},
        'median_tracklet_s': float(((tab.f_end - tab.f_start) / 30.0).median()),
        'heldout_even_odd': held,
    }


# ----------------------------------------------------------------------------- C helpers
def held(pf, hold_s):
    """Evaluation-only position hold. Interactions are contacts, and the identity layer refuses to
    label animals in contact, so at hold 0 the external checks have (almost) no coverage. With
    hold_s > 0 every identity's assigned (confirmed / inferred) position is carried forward AND
    backward for at most hold_s seconds (nearest known frame). This is never written to the
    identity output; it only asks 'where was the animal called X just before / after?'."""
    if hold_s <= 0:
        return pf
    out = []
    for i, g in pf.sort_values('frame_src').groupby('identity'):
        g = g.copy()
        known = g.state != 'unknown'
        fr = g.frame_src.values
        kf = fr[known.values]
        if len(kf) == 0:
            out.append(g)
            continue
        kx, ky = g.cx.values[known.values], g.cy.values[known.values]
        j = np.clip(np.searchsorted(kf, fr), 0, len(kf) - 1)
        jp = np.clip(j - 1, 0, len(kf) - 1)
        use = np.where(np.abs(kf[jp] - fr) < np.abs(kf[j] - fr), jp, j)
        dt = np.abs(kf[use] - fr)
        ok = dt <= hold_s * 30
        g['cx'] = np.where(ok, kx[use], np.nan)
        g['cy'] = np.where(ok, ky[use], np.nan)
        g['state'] = np.where(known.values, g.state.values, np.where(ok, 'held', 'unknown'))
        out.append(g)
    return pd.concat(out, ignore_index=True)


# ----------------------------------------------------------------------------- C ants
def ants_external(pf, ann, contact_px, hold_s=0.0):
    """ann: per source frame booleans Y2F, B2F (indexed by frame_src). Rates are over annotated
    frames where both identities of the pair have a position (coverage reported)."""
    pf = held(pf, hold_s)
    wide = pf.pivot(index='frame_src', columns='identity', values=['cx', 'cy'])
    st = pf.pivot(index='frame_src', columns='identity', values='state')
    a = ann.reindex(wide.index).fillna(False)

    def dist(i, j):
        return np.hypot(wide['cx'][i] - wide['cx'][j], wide['cy'][i] - wide['cy'][j])

    out = {'contact_px': contact_px, 'hold_s': hold_s}
    for beh, right, wrong in [('Y2F', 'yellow', 'blue'), ('B2F', 'blue', 'yellow')]:
        sel = a[beh] & ~a['Y2F' if beh == 'B2F' else 'B2F']  # bout frames of this behaviour only
        res = {'n_bout_frames': int(sel.sum())}
        for name, col in [('assigned', right), ('wrong_colour_baseline', wrong)]:
            d = dist(col, 'focal')[sel]
            ok = d.notna()
            res[name] = {'coverage_pct': float(ok.mean() * 100) if len(ok) else None,
                         'pct_within_contact': float((d[ok] < contact_px).mean() * 100) if ok.any() else None,
                         'median_dist_px': float(d[ok].median()) if ok.any() else None}
            both_conf = ((st.loc[sel.index[sel], col] == 'confirmed') &
                         (st.loc[sel.index[sel], 'focal'] == 'confirmed'))
            dc = d[both_conf.reindex(d.index).fillna(False).astype(bool)]
            res[name]['confirmed_only_pct_within_contact'] = float((dc < contact_px).mean() * 100) \
                if len(dc) else None
            res[name]['confirmed_only_n'] = int(len(dc))
        out[beh] = res
    return out


# ----------------------------------------------------------------------------- C mice
def mice_external(pf, events, ids, pad_s=0.0, hold_s=0.0):
    """events: DataFrame start, stop (source frames), a1, a2 (annotator indices 1-4), code.
    For each event, frames in [start - pad, stop + pad] where ALL 4 identities have a position;
    pair distance = median centroid distance over those frames. Event hit iff the annotated
    (unordered) pair is the closest of the 6 pairs. Reported for every numbering map."""
    pf = held(pf, hold_s)
    wide = pf.pivot(index='frame_src', columns='identity', values=['cx', 'cy'])
    full = wide['cx'].notna().all(axis=1)
    fr = wide.index.values
    pairs = [(a, b) for a in range(4) for b in range(a + 1, 4)]
    X, Y = wide['cx'][ids].values, wide['cy'][ids].values
    pad = int(round(pad_s * 30))
    closest, n_cov = [], 0
    for _, e in events.iterrows():
        m = (fr >= e.start - pad) & (fr <= e.stop + pad) & full.values
        if not m.any():
            closest.append(None)
            continue
        n_cov += 1
        D = [np.median(np.hypot(X[m, a] - X[m, b], Y[m, a] - Y[m, b])) for a, b in pairs]
        closest.append(frozenset(pairs[int(np.argmin(D))]))
    ev = events.assign(closest=closest)
    ev = ev[ev.closest.notna()]
    per_map = {}
    for perm in permutations(range(4)):
        # perm[a-1] = identity index (into ids) of annotator animal a
        hits = [frozenset((perm[int(r.a1) - 1], perm[int(r.a2) - 1])) == r.closest
                for r in ev.itertuples()]
        per_map[''.join(str(p + 1) for p in perm)] = float(np.mean(hits)) if hits else None
    order = sorted(per_map.items(), key=lambda kv: -(kv[1] or 0))
    return {'n_events': int(len(events)), 'n_events_covered': int(n_cov),
            'coverage_pct': float(n_cov / max(len(events), 1) * 100),
            'pad_s': pad_s, 'hold_s': hold_s, 'chance': 1 / 6,
            'hit_rate_identity_map_1234': per_map['1234'],
            'hit_rate_by_map_top5': order[:5],
            'hit_rate_by_map_all': per_map}
