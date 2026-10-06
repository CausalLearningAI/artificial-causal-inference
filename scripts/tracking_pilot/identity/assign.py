"""Identity assignment over tracklets. Never trusts linking through contact.

1. Tracklets. Every track is cut at every ambiguity:
     - a row that is not `detected` (filled / predicted) belongs to no tracklet;
     - a time gap longer than `max_gap_s`, or a speed above `vmax` (implausible jump);
     - entering or leaving CONTACT (its box overlaps another detected box, expanded by
       `contact_margin` x the median single-animal box size), or a change in who it touches;
     - entering or leaving OVERSIZE (box area > `oversize` x the median single-animal box area:
       probably more than one animal in the box);
     - a COUNT CHANGE nearby: when a track ends or starts, every track within `near` body lengths
       of that position is cut at that frame (two animals that merge into one detection and split
       again must not carry identities across). `count_rule='global'` instead cuts every track
       whenever the number of detections changes or is below N (more conservative).
2. Votes. Each tracklet sums log-probabilities of its READABLE mark reads (reads spaced >=
   `read_dt` s apart, probabilities clipped at `p_floor` so one bad read cannot dominate):
       L[k, i] = sum_r log max(p_r(i), p_floor)
   margin[k, i] = L[k, i] - max_{j != i} L[k, j]  (> 0 for at most one identity).
3. Confirmed identities. A tracklet is a candidate for identity i iff margin[k, i] >= tau. Because a
   tracklet has at most one candidate, the constraint 'each identity used at most once at any
   frame' decouples by identity: for each identity, pick the set of non-overlapping candidate
   tracklets with maximum total margin (weighted interval scheduling, exact DP). This is the exact
   optimum of the ILP  max sum_k,i margin x_ki  s.t. sum_i x_ki <= 1, sum_{k active at t} x_ki <= 1.
   Candidates that lose (overlap a stronger tracklet with the same identity) are reported as
   'conflicts' and stay unknown.
   The UNMARKED identity (ants focal, mice m1_none; P['absence']) is never confirmed this way:
   'no mark seen' is also what a marked animal with a hidden dot / patch looks like, and that
   failure persists over consecutive reads (held-out ant labels: 10/140 crops of marked ants read
   as unmarked, 2/35 tracklets would be confirmed wrongly). Its reads are collapsed to one vote
   per tracklet and it is assigned by elimination only (step 4), where its evidence can veto.
4. Inferred identities (elimination). On every time segment where exactly N tracklets are active,
   all plausible single animals, and exactly one of them is unassigned, the unassigned one must
   carry the one unused identity. A tracklet is inferred as identity c iff every such segment of
   its life gives c, c is free over its whole life, and its own mark evidence does not contradict
   c (margin[k, c] > -tau). Repeated until nothing changes.
Output per (frame, identity): position, state in {confirmed, inferred, unknown}, confidence.
"""
import numpy as np
import pandas as pd

DEFAULTS = {
    'max_gap_s': 0.5, 'vmax': None, 'contact_margin': 0.0, 'oversize': 1.7, 'near': 1.5,
    'count_rule': 'local', 'read_dt': 0.5, 'p_floor': 0.02, 'tau': 3.0, 'absence': None,
}
ABSENCE = {'ants': 'focal', 'mice': 'm1_none'}  # the identity defined by NOT carrying a mark
# vmax in source px / s; None -> 4 median body diagonals per second x 3
STATES = ['confirmed', 'inferred', 'unknown']


def _median_size(tr, N=None):
    """Single-animal reference size. A plain median over all detections is wrong when animals
    spend most of the time merged (ants 3_6_6: 3 separate boxes in only 3 % of frames), so the
    reference is the median over frames where exactly N boxes are detected (all animals
    separate), if there are >= 50 such frames; otherwise the 25th percentile of all boxes."""
    d = tr[tr.detected]
    nd = d.groupby('frame_src').frame_src.transform('size')
    ref = d[nd == N] if N is not None else d.iloc[:0]
    if ref.frame_src.nunique() >= 50:
        return float(np.median(ref.w * ref.h)), float(np.median(np.hypot(ref.w, ref.h)))
    return float(np.percentile(d.w * d.h, 25)), float(np.percentile(np.hypot(d.w, d.h), 25))


def frame_geometry(tr, P=DEFAULTS, N=None):
    """Per detected row: in_contact (box overlaps another detected box), partners (frozenset of
    track ids it touches), oversize, n_det (detections in that frame). Returns a copy of tr."""
    tr = tr.copy()
    med_area, med_diag = _median_size(tr, N)
    m = P['contact_margin'] * np.sqrt(med_area)
    tr['oversize'] = tr.detected & (tr.w * tr.h > P['oversize'] * med_area)
    tr['n_det'] = tr.groupby('frame_src').detected.transform('sum').astype(int)
    partners = {}
    for f, g in tr[tr.detected].groupby('frame_src', sort=False):
        if len(g) < 2:
            continue
        x0, x1 = (g.cx - g.w / 2 - m).values, (g.cx + g.w / 2 + m).values
        y0, y1 = (g.cy - g.h / 2 - m).values, (g.cy + g.h / 2 + m).values
        ov = (x0[:, None] < x1[None]) & (x0[None] < x1[:, None]) & (y0[:, None] < y1[None]) & \
             (y0[None] < y1[:, None])
        np.fill_diagonal(ov, False)
        ids = g.track_id.values
        for a, idx in enumerate(g.index):
            if ov[a].any():
                partners[idx] = frozenset(ids[ov[a]].tolist())
    tr['partners'] = pd.Series(partners, dtype=object).reindex(tr.index)
    tr['partners'] = tr['partners'].apply(lambda v: v if isinstance(v, frozenset) else frozenset())
    tr['in_contact'] = tr.partners.apply(len) > 0
    tr.attrs['med_area'], tr.attrs['med_diag'] = med_area, med_diag
    return tr


def isolated(tr):
    """Rows where a mark can be read: detected, touching nobody, not oversize, not tiny."""
    return tr.detected & ~tr.in_contact & ~tr.oversize & (tr.w * tr.h > 0.35 * tr.attrs['med_area'])


def make_tracklets(tr, N, P=DEFAULTS):
    """tr: one variant, with frame_geometry columns. Adds 'tracklet' (-1 = none) and 'cut_reason'
    (why the tracklet containing this row started). Returns (tr, tracklet table)."""
    tr = tr.sort_values(['track_id', 'frame_src']).copy()
    med_diag = tr.attrs['med_diag']
    vmax = P['vmax'] or 12.0 * med_diag
    max_gap = P['max_gap_s'] * 30.0
    det = tr[tr.detected]

    # count-change events: (frame, x, y) where a track starts or ends (excluding video edges)
    f_first, f_last = det.frame_src.min(), det.frame_src.max()
    ev_cut = {}  # (track_id, frame) -> reason : cut BEFORE this frame's row
    if P['count_rule'] == 'local':
        runs = []
        for tid, g in det.groupby('track_id'):
            fr = g.frame_src.values
            br = np.nonzero(np.diff(fr) > max_gap)[0]
            starts = np.r_[0, br + 1]
            ends = np.r_[br, len(fr) - 1]
            for s, e in zip(starts, ends):
                runs.append((tid, fr[s], g.cx.values[s], g.cy.values[s], 'birth'))
                runs.append((tid, fr[e], g.cx.values[e], g.cy.values[e], 'death'))
        byf = {f: g for f, g in det.groupby('frame_src')}
        frames = np.array(sorted(byf))
        for tid, f, x, y, kind in runs:
            if (kind == 'birth' and f == f_first) or (kind == 'death' and f == f_last):
                continue
            # neighbours at the event frame (birth) / the next frame (death) are cut there
            if kind == 'birth':
                fe = f
            else:
                k = np.searchsorted(frames, f, side='right')
                if k >= len(frames):
                    continue
                fe = frames[k]
            g = byf[fe]
            d = np.hypot(g.cx.values - x, g.cy.values - y)
            for t2 in g.track_id.values[(d < P['near'] * med_diag) & (g.track_id.values != tid)]:
                ev_cut[(t2, fe)] = f'count_{kind}'
    else:
        nd = det.groupby('frame_src').size()
        chg = nd.index[np.r_[False, np.diff(nd.values) != 0] | (nd.values < N)]
        chg = set(chg.tolist())
        for idx, r in det.iterrows():
            if r.frame_src in chg:
                ev_cut[(r.track_id, r.frame_src)] = 'count_global'

    tl = np.full(len(tr), -1, int)
    reason = np.array([''] * len(tr), dtype=object)
    next_id = 0
    rows = tr.reset_index(drop=True)
    tr = rows
    for tid, g in tr.groupby('track_id', sort=False):
        idx = g.index.values
        fr, x, y = g.frame_src.values, g.cx.values, g.cy.values
        dt_, cont, ovs = g.detected.values, g.in_contact.values, g.oversize.values
        prt = g.partners.values
        cur, prev = -1, None
        for a in range(len(idx)):
            if not dt_[a]:
                cur, prev = -1, None
                continue
            why = None
            if prev is None:
                why = 'start'
            else:
                dtf = fr[a] - fr[prev]
                if dtf > max_gap:
                    why = 'gap'
                elif np.hypot(x[a] - x[prev], y[a] - y[prev]) / (dtf / 30.0) > vmax:
                    why = 'jump'
                elif cont[a] != cont[prev] or (cont[a] and prt[a] != prt[prev]):
                    why = 'contact'
                elif ovs[a] != ovs[prev]:
                    why = 'oversize'
                elif (tid, fr[a]) in ev_cut:
                    why = ev_cut[(tid, fr[a])]
            if why is not None:
                cur = next_id
                next_id += 1
                reason[idx[a]] = why
            tl[idx[a]] = cur
            prev = a
    tr['tracklet'] = tl
    tr['cut_reason'] = reason
    d = tr[tr.tracklet >= 0]
    tab = d.groupby('tracklet').agg(track_id=('track_id', 'first'), f_start=('frame_src', 'min'),
                                     f_end=('frame_src', 'max'), n_rows=('frame_src', 'size'),
                                     contact=('in_contact', 'any'), oversize=('oversize', 'any'))
    tab['start_reason'] = d[d.cut_reason != ''].set_index('tracklet').cut_reason.reindex(tab.index)
    return tr, tab


def tracklet_votes(tr, reads, ids, tab, P=DEFAULTS, keep=None):
    """reads: frame_src, track_id, readable, p_<id>... ; keep: optional bool mask over reads
    (e.g. even seconds only). Returns tab with L_<id>, n_reads, margin_<id>."""
    r = reads[reads.readable].copy()
    if keep is not None:
        r = r[keep[reads.readable].values]
    r = r.merge(tr[['frame_src', 'track_id', 'tracklet']], on=['frame_src', 'track_id'])
    r = r[r.tracklet >= 0].sort_values(['tracklet', 'frame_src'])
    # thin to >= read_dt spacing within a tracklet
    step = P['read_dt'] * 30.0
    keep_rows, last = [], {}
    for i, (k, f) in enumerate(zip(r.tracklet.values, r.frame_src.values)):
        if k not in last or f - last[k] >= step - 1e-6:
            keep_rows.append(i)
            last[k] = f
    r = r.iloc[keep_rows]
    pcols = [f'p_{i}' for i in ids]
    lp = np.log(np.clip(r[pcols].values, P['p_floor'], 1.0))
    L = pd.DataFrame(lp, columns=[f'L_{i}' for i in ids], index=r.index)
    L['tracklet'] = r.tracklet.values
    if P.get('absence') in ids:
        # 'no mark seen' reads are not independent (a hidden dot / patch stays hidden while the
        # posture lasts): all absence-voting reads of a tracklet count as ONE read (their mean)
        ab = lp.argmax(1) == ids.index(P['absence'])
        Lc = [c for c in L.columns if c.startswith('L_')]
        agg = L[~ab].groupby('tracklet')[Lc].sum().add(L[ab].groupby('tracklet')[Lc].mean(), fill_value=0)
        agg['n_absence_reads'] = L[ab].groupby('tracklet').size()
        agg['n_absence_reads'] = agg['n_absence_reads'].fillna(0).astype(int)
    else:
        agg = L.groupby('tracklet').sum()
    agg['n_reads'] = L.groupby('tracklet').size()
    if 'too_big' in reads.columns:  # share of ALL reads (readable or not) whose mask was 2-animal sized
        rb = reads.merge(tr[['frame_src', 'track_id', 'tracklet']], on=['frame_src', 'track_id'])
        rb = rb[rb.tracklet >= 0]
        agg = agg.join(rb.groupby('tracklet').too_big.mean().rename('frac_big'), how='outer')
    out = tab.drop(columns=[c for c in tab.columns if c.startswith(('L_', 'margin_')) or
                            c in ('n_reads', 'n_absence_reads', 'frac_big')], errors='ignore').join(agg)
    out['n_reads'] = out['n_reads'].fillna(0).astype(int)
    out['frac_big'] = out['frac_big'].fillna(0.0) if 'frac_big' in out else 0.0
    Lm = out[[f'L_{i}' for i in ids]].fillna(0.0).values
    for a, i in enumerate(ids):
        other = np.delete(Lm, a, axis=1).max(axis=1)
        out[f'margin_{i}'] = np.where(out.n_reads > 0, Lm[:, a] - other, 0.0)
    return out


def _schedule(iv):
    """Weighted interval scheduling. iv: list of (start, end, weight, key), inclusive ends.
    Returns the chosen keys."""
    if not iv:
        return []
    iv = sorted(iv, key=lambda t: t[1])
    ends = np.array([t[1] for t in iv])
    n = len(iv)
    best = np.zeros(n + 1)
    choose = np.zeros(n, bool)
    pidx = np.searchsorted(ends, [t[0] for t in iv], side='left')  # intervals ending before start
    for j in range(n):
        take = iv[j][2] + best[pidx[j]]
        if take > best[j]:
            best[j + 1], choose[j] = take, True
        else:
            best[j + 1] = best[j]
    out, j = [], n
    while j > 0:
        if choose[j - 1]:
            out.append(iv[j - 1][3])
            j = pidx[j - 1]
        else:
            j -= 1
    return out


def assign(tab, ids, N, P=DEFAULTS, single_ok=None):
    """tab: tracklet table with margin_<id>. Returns tab with identity, state, conf, conflict."""
    tab = tab.copy()
    M = tab[[f'margin_{i}' for i in ids]].values
    tab['identity'] = None
    tab['state'] = 'unknown'
    tab['conf'] = 0.0
    tab['conflict'] = False
    Lm = tab[[f'L_{i}' for i in ids]].fillna(0.0).values
    post = np.exp(Lm - Lm.max(1, keepdims=True))
    post /= post.sum(1, keepdims=True)
    for a, i in enumerate(ids):
        if i == P.get('absence'):
            continue  # the unmarked identity is never confirmed from 'no mark seen'; elimination only
        cand = tab.index[M[:, a] >= P['tau']]
        iv = [(tab.at[k, 'f_start'], tab.at[k, 'f_end'], M[tab.index.get_loc(k), a], k) for k in cand]
        chosen = set(_schedule(iv))
        for k in cand:
            if k in chosen:
                tab.at[k, 'identity'], tab.at[k, 'state'] = i, 'confirmed'
                tab.at[k, 'conf'] = float(post[tab.index.get_loc(k), a])
            else:
                tab.at[k, 'conflict'] = True
    if single_ok is None:  # elimination only onto tracklets that look like ONE animal
        single_ok = ~tab.oversize & (tab.frac_big < 0.5)
    _eliminate(tab, ids, N, P, single_ok)
    return tab


def _segments(tab):
    b = np.unique(np.r_[tab.f_start.values, tab.f_end.values + 1])
    return list(zip(b[:-1], b[1:] - 1))


def _eliminate(tab, ids, N, P, single_ok):
    changed = True
    segs = _segments(tab)
    st, en = tab.f_start.values, tab.f_end.values
    keys = tab.index.values
    # sweep: segment boundaries are tracklet starts / ends + 1, so activity is constant per segment
    seg_starts = np.array([s for s, _ in segs])
    seg_active = [[] for _ in segs]
    a0 = np.searchsorted(seg_starts, st, side='left')
    a1 = np.searchsorted(seg_starts, en, side='right')
    for k, i0, i1 in zip(keys, a0, a1):
        for si in range(i0, i1):
            seg_active[si].append(k)
    seg_of = {k: [] for k in keys}
    for si, act in enumerate(seg_active):
        for k in act:
            seg_of[k].append(si)
    while changed:
        changed = False
        for k in keys:
            if tab.at[k, 'state'] != 'unknown' or not single_ok[k]:
                continue
            answers, ok, confs = set(), True, []
            used_any = set()
            for si in seg_of[k]:
                act = seg_active[si]
                others = [o for o in act if o != k]
                used = {tab.at[o, 'identity'] for o in others if tab.at[o, 'identity'] is not None}
                used_any |= used
                if len(act) == N and all(single_ok[o] for o in act) and len(used) == N - 1:
                    answers.add(next(i for i in ids if i not in used))
                    confs.append(min(tab.at[o, 'conf'] for o in others))
            if len(answers) != 1:
                continue
            c = answers.pop()
            if c in used_any or tab.at[k, f'margin_{c}'] <= -P['tau']:
                continue
            tab.at[k, 'identity'], tab.at[k, 'state'] = c, 'inferred'
            tab.at[k, 'conf'] = float(min(confs))
            changed = True


def per_frame(tr, tab, ids):
    """-> DataFrame one row per (frame_src, identity): cx, cy, w, h, track_id, tracklet, state,
    conf. Frames = every frame that has any track row."""
    frames = np.unique(tr.frame_src.values)
    d = tr[tr.tracklet >= 0].drop(columns=['conf']).merge(tab[['identity', 'state', 'conf']], left_on='tracklet',
                                   right_index=True)
    d = d[d.identity.notna()]
    base = pd.MultiIndex.from_product([frames, ids], names=['frame_src', 'identity']).to_frame(index=False)
    cols = ['frame_src', 'identity', 'cx', 'cy', 'w', 'h', 'track_id', 'tracklet', 'state', 'conf']
    d = d[cols].drop_duplicates(['frame_src', 'identity'])
    out = base.merge(d, on=['frame_src', 'identity'], how='left')
    out['state'] = out.state.fillna('unknown')
    out['conf'] = out.conf.fillna(0.0)
    out['track_id'] = out.track_id.fillna(-1).astype(int)
    out['tracklet'] = out.tracklet.fillna(-1).astype(int)
    return out


def run_assignment(tracks, reads, ids, N, P=DEFAULTS, keep=None):
    tr = frame_geometry(tracks, P, N)
    tr, tab = make_tracklets(tr, N, P)
    tab = tracklet_votes(tr, reads, ids, tab, P, keep=keep)
    tab = assign(tab, ids, N, P)
    return tr, tab
