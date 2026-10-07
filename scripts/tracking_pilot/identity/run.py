"""Identity layer + evaluation on one video's tracks (any tracker in the contract).

    python scripts/tracking_pilot/identity/run.py --domain mice --video rd25 \
        --tracks results/tracking_pilot/mice/rd25/amadeus/tracks.parquet [--tag amadeus]

Steps: (1) mark-reading pass over isolated rows (skipped if feats exist, --reread to force),
(2) trained scorer -> reads, (3) per variant: tracklets, votes, assignment, (4) metrics A-C ->
metrics.json, (5) audit renders (D) -> audit/.
Outputs: results/tracking_pilot/{domain}/{video}/identity/{tag}/
    identity_tracks_<variant>.parquet   one row per (frame_src, identity)
    tracklets_<variant>.parquet         tracklet table with votes, identity, state
    metrics.json                        {variant: {A, B, C}}
    audit/                              PNGs + index.csv
Thresholds are fixed in assign.DEFAULTS / the scorer; nothing is fit on section C.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from assign import ABSENCE, DEFAULTS, per_frame, run_assignment
from audit import make_audit
from common import (IDENTITIES, N_ANIMALS, ROOT, VIDEOS, annotated_window, load_tracks, out_dir)
from metrics import ants_external, completeness, consistency, mice_external
from score import apply_scorer, read_pass

BRIDGE = 2  # frames; AMADEUS spike-replaced boxes come as isolated 1-frame runs


def ants_annotations(vid, frames):
    p = ROOT / f'data/ants/v3/annotations/{vid}.csv'
    lines = [l.strip() for l in open(p) if l.strip()]
    hdr = next(i for i, l in enumerate(lines) if l.startswith('Petri-dish-index'))
    cols = [c.strip() for c in lines[hdr].split(',')]
    recs = []
    for l in lines[hdr + 1:]:
        if l.startswith('---'):
            break
        recs.append([c.strip() for c in l.split(',')])
    a = pd.DataFrame(recs, columns=cols)
    ann = pd.DataFrame(False, index=pd.Index(frames, name='frame_src'), columns=['Y2F', 'B2F'])
    m = {'groom-orange': ['Y2F'], 'groom-blue': ['B2F'], 'groom-orangeandblue': ['Y2F', 'B2F']}
    for r in a.itertuples():
        b = r.Behavior
        if b not in m:
            continue  # onLid-* behaviours are not identity-pair events
        s, e = int(r[cols.index('Beginning-frame') + 1]), int(r[cols.index('End-frame') + 1])
        for c in m[b]:
            ann.loc[(ann.index >= s) & (ann.index <= e), c] = True
    return ann


def mice_events(vid):
    p = ROOT / 'data/mice/v1/annotations' / VIDEOS['mice'][vid].replace('.mp4', '.csv')
    b = pd.read_csv(p)
    ev = pd.DataFrame({'start': b['Image index start'], 'stop': b['Image index stop'],
                       'a1': b['agent1(active)'], 'a2': b['agent2'], 'code': b['behavior_type']})
    ev = ev.dropna(subset=['start', 'stop', 'a1', 'a2'])
    ev = ev[ev.a1.astype(str).str.fullmatch(r'[1-4](\.0)?') & ev.a2.astype(str).str.fullmatch(r'[1-4](\.0)?')]
    ev[['a1', 'a2']] = ev[['a1', 'a2']].astype(float).astype(int)
    return ev[ev.a1 != ev.a2].reset_index(drop=True)


def ant_contact_px(tracks):
    """A-priori contact distance: 1.2 x median box diagonal of the variant with real boxes."""
    real = tracks[tracks.variant != 'standin_anttracker']
    d = real[real.detected]
    return float(1.2 * np.median(np.hypot(d.w, d.h)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--domain', required=True, choices=['mice', 'ants'])
    ap.add_argument('--video', required=True)
    ap.add_argument('--tracks', required=True)
    ap.add_argument('--tag', default=None, help='output subfolder (default: parent folder of tracks)')
    ap.add_argument('--reread', action='store_true')
    ap.add_argument('--no_audit', action='store_true')
    ap.add_argument('--dump_every', type=int, default=0, help='save every k-th read crop as JPEG (0 = none)')
    a = ap.parse_args()
    dom, vid = a.domain, a.video
    tag = a.tag or Path(a.tracks).parent.name
    ids, N = IDENTITIES[dom], N_ANIMALS[dom]
    od = out_dir(dom, vid) / tag
    P = dict(DEFAULTS, absence=ABSENCE[dom])
    P0 = P
    tracks = load_tracks(a.tracks)
    s, e = annotated_window(dom, vid)
    tracks = tracks[(tracks.frame_src >= s) & (tracks.frame_src <= e)]
    if a.reread or not (od / 'feats.parquet').exists():
        read_pass(dom, vid, tracks, tag, a.dump_every)
    reads_all = apply_scorer(dom, vid, tag)
    out = {}
    # strict cutter (every not-detected row cuts) and, when the tracker has not-detected rows, the
    # bridge setting (runs of <= BRIDGE contact-free not-detected rows do not cut), reported side by side
    settings = [('', P)]
    if (~tracks.detected).any():
        settings.append((f'+bridge{BRIDGE}', dict(P, bridge_nondet=BRIDGE)))
    for (variant, tr0), (suffix, P) in [(vt, st) for vt in tracks.groupby('variant') for st in settings]:
        tr0 = tr0.reset_index(drop=True)
        reads = reads_all[reads_all.variant == variant].drop(columns='variant')
        tr, tab = run_assignment(tr0, reads, ids, N, P)
        variant = variant + suffix
        pf = per_frame(tr, tab, ids)
        pf.to_parquet(od / f'identity_tracks_{variant}.parquet')
        tab.to_parquet(od / f'tracklets_{variant}.parquet')
        res = {'tracks': str(a.tracks), 'n_reads': int(len(reads)),
               'n_readable_reads': int(reads.readable.sum()),
               'params': P,
               'A_completeness': completeness(tr, pf, N),
               'B_consistency': consistency(tr, tab, reads, ids, tr0, N, P)}
        if dom == 'ants':
            ann = ants_annotations(vid, np.unique(tr.frame_src))
            cpx = ant_contact_px(tracks)
            res['C_external'] = {f'hold_{h}s': ants_external(pf, ann, cpx, h) for h in (0.0, 1.0, 2.0, 5.0, 10.0)}
        else:
            ev = mice_events(vid)
            res['C_external'] = {f'pad_{p}s_hold_{h}s': mice_external(pf, ev, ids, p, h)
                                 for p, h in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.0, 2.0)]}
        out[variant] = res
        if not a.no_audit:
            make_audit(dom, vid, variant, tr, pf, ids, N, od / 'audit')
        A, B = res['A_completeness'], res['B_consistency']
        print(f'[{dom}/{vid}/{variant}] slots {A["pct_slots"]} | tracklets {B["n_tracklets"]} '
              f'contradictory {B["n_contradictory_tracklets"]} purity {B["mean_vote_purity"]} '
              f'held-out {B["heldout_even_odd"]["assigned_any"]}')
    with open(od / 'metrics.json', 'w') as fh:
        json.dump(out, fh, indent=1, default=lambda o: o if not isinstance(o, (np.generic,)) else o.item())
    print('wrote', od / 'metrics.json')


if __name__ == '__main__':
    main()
