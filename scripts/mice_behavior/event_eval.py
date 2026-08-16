#!/usr/bin/env python3
"""Score the frame classifier as an EVENT detector rather than a frame classifier.

Why the unit matters
--------------------
Three facts about this dataset make frame-exact scoring the wrong default:

1. The annotation is natively EVENTS. The lab produced 7,843 BORIS intervals with start/stop
   times at 30 fps. Per-frame labels are a derived quantisation to 5 fps, and 22.3% of all
   bouts (31.9% of nn) are SHORTER than one 5 fps frame. For that fifth of the data, *which*
   frame carries the label is decided by sub-frame phase -- noise the pipeline introduced, not
   annotator error and not something a better model can fix.

2. The causal signal lives in event COUNTS, not in occupancy. Measured on the 24 annotated
   pools: bouts-per-minute is significant in 6/6 nn phase contrasts and 3/4 nt ones, against
   6/12 for time-in-behaviour, while mean bout DURATION is null in every cell. The phase
   transition changes how often a behaviour is initiated, not how long it lasts. So the
   quantity the downstream estimand needs the model to get right is the count.

3. Effective sample size is events, not frames. Train holds ~1,186 nt and ~3,696 nn bouts,
   with median length 3 and 2 frames. Frame metrics quote an n of 19,320 positives and thereby
   report a precision the data does not have; a +-1 frame boundary disagreement corrupts
   33-50% of a median bout at frame level while leaving the event untouched.

What this computes
------------------
Predicted bouts = connected runs of p >= threshold, after optionally closing gaps of <=
`--merge-gap` frames and dropping runs shorter than `--min-len`. A true bout counts as
DETECTED if any predicted bout overlaps it at all (any-overlap matching, the standard for
short-event ethology); a predicted bout counts as a HIT on the same rule. Sweeping the
threshold gives an event-level PR curve, which is the honest analogue of frame AP.

The last block is the one that matters for the causal work: predicted bouts-per-minute per
observation against true bouts-per-minute, and the correlation of their WITHIN-POOL PHASE
DIFFERENCES -- the exact quantity PPI's variance reduction depends on.

Usage:
    python scripts/mice_behavior/event_eval.py --tag res448_k2_frozen_d4photo_sslinit
    python scripts/mice_behavior/event_eval.py --all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
FRAME_DIR = ROOT / 'results' / 'vision' / 'mice' / 'frame'
FPS = 5.0


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True runs as inclusive [start, end] index pairs."""
    if not mask.any():
        return []
    d = np.diff(mask.astype(np.int8), prepend=0, append=0)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1))


def postprocess(mask: np.ndarray, merge_gap: int, min_len: int) -> np.ndarray:
    if merge_gap > 0:
        for a, b in runs(~mask):
            if b - a + 1 <= merge_gap and a > 0 and b < len(mask) - 1:
                mask[a:b + 1] = True
    if min_len > 1:
        for a, b in runs(mask):
            if b - a + 1 < min_len:
                mask[a:b + 1] = False
    return mask


def overlap_counts(true_m, pred_m):
    """(#true bouts detected, #true bouts, #pred bouts hitting truth, #pred bouts)."""
    tb, pb = runs(true_m), runs(pred_m)
    det = sum(any(not (pe < ts or ps > te) for ps, pe in pb) for ts, te in tb)
    hit = sum(any(not (te < ps or ts > pe) for ts, te in tb) for ps, pe in pb)
    return det, len(tb), hit, len(pb)


def evaluate(tag, merge_gap, min_len, thresholds):
    d = np.load(FRAME_DIR / tag / 'val_probs.npz', allow_pickle=True)
    exp = pd.read_csv(ROOT / 'data' / 'mice' / 'v1' / 'experiment.csv')[
        ['observation_id', 'pool', 'phase', 'odor']]
    df = pd.DataFrame({'obs': d['obs'], 'gi': d['gi'],
                       'p_nt': d['probs'][:, 0], 'p_nn': d['probs'][:, 1],
                       'y_nt': d['labels'][:, 0], 'y_nn': d['labels'][:, 1]}).sort_values(['obs', 'gi'])
    out = {}
    for j, lab in enumerate(('nt', 'nn')):
        curve, per_obs = [], []
        for th in thresholds:
            D = T = H = P = 0
            for oid, g in df.groupby('obs', sort=False):
                tm = g['y_' + lab].to_numpy() > 0.5
                pm = postprocess(g['p_' + lab].to_numpy() >= th, merge_gap, min_len)
                a, b, c, e = overlap_counts(tm, pm)
                D += a; T += b; H += c; P += e
            rec = D / T if T else np.nan
            pre = H / P if P else np.nan
            f1 = 2 * pre * rec / (pre + rec) if pre and rec and (pre + rec) > 0 else 0.0
            curve.append((th, pre, rec, f1, P))
        curve = pd.DataFrame(curve, columns=['thr', 'precision', 'recall', 'f1', 'n_pred'])
        best = curve.loc[curve.f1.idxmax()]
        # count agreement at the best-F1 operating point -- the causal-relevant quantity
        for oid, g in df.groupby('obs', sort=False):
            tm = g['y_' + lab].to_numpy() > 0.5
            pm = postprocess(g['p_' + lab].to_numpy() >= best.thr, merge_gap, min_len)
            mins = len(g) / FPS / 60
            per_obs.append((oid, len(runs(tm)) / mins, len(runs(pm)) / mins))
        po = pd.DataFrame(per_obs, columns=['observation_id', 'true_bpm', 'pred_bpm']).merge(exp, on='observation_id')
        r_lvl = np.corrcoef(po.true_bpm, po.pred_bpm)[0, 1]
        wt = po.pivot_table(index=['pool', 'odor'], columns='phase', values='true_bpm')
        wp = po.pivot_table(index=['pool', 'odor'], columns='phase', values='pred_bpm')
        dt = np.concatenate([(wt['O'] - wt['H']).values, (wt['P'] - wt['O']).values])
        dp = np.concatenate([(wp['O'] - wp['H']).values, (wp['P'] - wp['O']).values])
        out[lab] = dict(curve=curve, best=best, po=po, r_level=r_lvl,
                        r_delta=np.corrcoef(dt, dp)[0, 1],
                        true_bpm=po.true_bpm.mean(), pred_bpm=po.pred_bpm.mean())
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag', action='append')
    p.add_argument('--all', action='store_true')
    p.add_argument('--merge-gap', type=int, default=1,
                   help='close prediction gaps of <= this many frames before forming bouts. '
                        'Bouts are contiguous and errors are partly jitter, so a 1-frame hole '
                        'inside a real detection should not split it into two events.')
    p.add_argument('--min-len', type=int, default=1,
                   help='drop predicted bouts shorter than this. 1 = keep all, which is right '
                        'here: 22%% of TRUE bouts are a single frame, so filtering by length '
                        'discards real events faster than it removes noise.')
    args = p.parse_args()
    tags = ([d.name for d in sorted(FRAME_DIR.iterdir()) if (d / 'val_probs.npz').exists()]
            if args.all else (args.tag or []))
    if not tags:
        raise SystemExit('pass --tag or --all')
    ths = np.round(np.arange(0.05, 1.0, 0.05), 2)
    print(f'EVENT-LEVEL EVALUATION  (any-overlap matching, merge_gap={args.merge_gap}, '
          f'min_len={args.min_len}, 24 val observations)\n')
    print(f"{'run':42s} {'lab':4s} {'P':>6} {'R':>6} {'F1':>6} {'thr':>5} "
          f"{'true/min':>9} {'pred/min':>9} {'r_lvl':>6} {'r_delta':>8}")
    for t in tags:
        try:
            res = evaluate(t, args.merge_gap, args.min_len, ths)
        except Exception as e:
            print(f'{t:42s}  FAILED ({e.__class__.__name__}: {e})'); continue
        for lab in ('nt', 'nn'):
            r, b = res[lab], res[lab]['best']
            print(f'{t:42s} {lab:4s} {b.precision:6.3f} {b.recall:6.3f} {b.f1:6.3f} {b.thr:5.2f} '
                  f'{r["true_bpm"]:9.3f} {r["pred_bpm"]:9.3f} {r["r_level"]:6.3f} {r["r_delta"]:8.3f}')
    print('\nP/R/F1 are per EVENT (a bout counts as found if any predicted bout overlaps it).')
    print('true/min, pred/min = bouts per minute, the outcome the causal contrast is computed on.')
    print('r_delta = correlation of WITHIN-POOL phase differences in bouts/min -- the quantity')
    print('          PPI variance reduction actually depends on.')


if __name__ == '__main__':
    main()
