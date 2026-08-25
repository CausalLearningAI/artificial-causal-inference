#!/usr/bin/env python3
"""Compute EVERY phase-transition estimate the report shows, and dump one JSON for the figure.

The interactive figure in the status report is a view over this file. Nothing is computed in the
browser and nothing is transcribed by hand: the grid below is the complete set of numbers, and
the selectors just choose which slice to draw.

THE GRID
========
    experiment  v1 (72 pools, 24 annotated) | v2 (36 pools, 0 annotated)
    unit        events (bouts per minute) | time (occupancy, pp) | decay (mean onset, min)
    behaviour   nn (nose-to-nose)           | nt (nose-to-tail)
    model       which predictor supplies f
    stratum     all | v1: line x genotype (6) | v2: line (3)
    exposure    fear | social                  -- separate treatments, never pooled
    transition  H->O | O->P                    -- consecutive only
    method      CI (human labels, v1) | PPI++ (human+model, v1) | PPCI (model only, both)

WHERE EACH INPUT COMES FROM
===========================
truth, v1 labelled
    dataset/mice/v1/annotations.csv, per frame, 24 pools. Occupancy is the mean; bouts/min
    counts contiguous runs.

f on v1 labelled pools -- OUT OF FOLD
    xfit_f1/f2/f3 val_probs.npz. The three folds tile all 24 annotated pools, 8 at a time, so
    every labelled observation is scored by a model that never saw its pool. This is the
    condition PPI's rectifier needs and the standing 4-pool split cannot supply.

f on v1 unlabelled (48 pools) and on all of v2 (36 pools)
    pred_dense_{v1,v2}.csv from EACH fold, AVERAGED (cross-prediction). Averaging is not a
    refinement here, it is what keeps the estimator unbiased: the labelled side uses f_{k(j)}
    and the unlabelled side must have the same expectation, which (1/K) sum_k f_k does and a
    single fold's model does not. The previous version of this analysis used xfit_f2 alone on
    the unlabelled pools against out-of-fold predictions on the labelled ones, which biases the
    estimate by lam * (E[f_f2] - E[f_oof]); `--report-mismatch` prints that gap.

    These dumps are DENSE (stride 1). The earlier ones were stride 16 -- one frame every 3.2 s
    against bouts whose median length is 2-3 frames -- which is why every previous PPI number
    was on occupancy even though the report argues the outcome should be event counts. Bout
    counts are not recoverable at stride 16; they are not defined.

THE THRESHOLD, AND WHY IT IS CHOSEN THE WAY IT IS
=================================================
Occupancy needs no threshold; bout counts do. There are no labels on the pools we deploy to, so
the threshold cannot be chosen there -- and `event_eval.py` picks its operating point by
maximising F1 on the very split it then scores, which is an oracle and is not available here.

This script uses LEAVE-ONE-FOLD-OUT selection: the threshold applied to fold k's held-out pools
is the max-F1 threshold measured on the OTHER folds' out-of-fold predictions, so it never sees a
label from the pools it is applied to. Unlabelled and v2 pools use the mean of the three. The
one residual imperfection, stated rather than hidden: tau_k is fitted on models f_j (j != k) and
applied to f_k, so it assumes the folds are similarly scaled. Measured max-F1 thresholds are
0.85-0.95 across folds, so they are.

Usage
-----
    python scripts/mice_behavior/build_estimates.py
    python scripts/mice_behavior/build_estimates.py --report-mismatch
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from src.mice_behavior.phase_ate import (                                  # noqa: E402
    TRANSITIONS, classical, pool_deltas, ppci, ppi)
from event_eval import postprocess, runs                                   # noqa: E402

FRAME = ROOT / 'results' / 'vision' / 'mice' / 'frame'
OUT = FRAME / '_figures'
FPS = 5.0
FOLDS = ('xfit_f1', 'xfit_f2', 'xfit_f3')

# EVERY PREDICTOR THE GRID IS BUILT FOR, so the effects figure can be read against a change of
# model rather than resting on one. `model` on each cell is the key below.
#
#   folds   the runs whose predictions are averaged. Three folds of 8 tile all 24 annotated pools,
#           so each labelled pool is scored by a model that never saw it.
#   human   whether CI and PPI++ are admissible. They need OUT-OF-FOLD predictions on the labelled
#           pools: an in-sample f is shrunk toward the labels, the rectifier under-corrects and the
#           interval undercovers. Only a complete cross-fitted set earns this.
#
# A predictor whose runs have not all landed is SKIPPED, not partially built -- two folds of three
# would leave eight annotated pools with no out-of-fold prediction and quietly invalidate PPI++.
PREDICTORS = {
    'xfit_dense': {'folds': FOLDS, 'human': True,
                   'nice': 'cross-fitted, 3 folds (SSL encoder) -- deployed'},
    'xfit_bit6_dense': {'folds': ('xfit_bit6_f1', 'xfit_bit6_f2', 'xfit_bit6_f3'), 'human': True,
                        'nice': 'cross-fitted, 3 folds (BitFit-6) -- the accuracy leader'},
    'xfit_derm_dense': {'folds': ('xfit_derm_f1', 'xfit_derm_f2', 'xfit_derm_f3'), 'human': True,
                        'nice': 'cross-fitted, 3 folds (DERM, phase environments)'},
}

# WHY A SINGLE, NON-CROSS-FITTED MODEL IS NOT IN THIS FIGURE, even though PPCI needs no labels and
# could in principle use one. `predict_dense.py` dumps only the UNANNOTATED pools, and a single
# model has no out-of-fold predictions on the annotated 24 -- it trained on 20 of them. So its
# PPCI would average over 48 pools while the deployed one averages over 72: a different
# population, silently, on the same axis. The model-vs-model comparison that IS honest holds the
# pool set fixed, and section 05.3 does it that way (build_ppci_robustness.py) on the 52 pools
# both predictors are out of sample on.


def predictor_ready(spec) -> bool:
    return all((FRAME / t / 'pred_dense_v1.csv').exists()
               and (FRAME / t / 'val_probs.npz').exists() for t in spec['folds'])
# The coarse 0.05 grid, plus fine steps above 0.95. DERM reweights environments, so its scores
# pile up against 1 and its max-F1 operating point sits at 0.95-0.995 -- past the coarse grid's
# last candidate, where the bout count moves 25-35% per 0.01 of threshold. The ERM predictors'
# optima are interior (0.85-0.90), so the extra candidates cannot move them; this is the same
# resolution fix build_derm.py's rate_match() made, applied to the max-F1 search.
THRESHOLDS = np.round(np.concatenate([np.arange(0.05, 1.0, 0.05),
                                      np.arange(0.955, 1.0, 0.005)]), 3)
LABELS = ('nt', 'nn')
BEHAV_NICE = {'nt': 'nose-to-tail', 'nn': 'nose-to-nose'}
ODOURS = (('F', 'fear'), ('S', 'social'))
UNITS = {'events': 'bouts per minute', 'time': 'occupancy (pp)',
         'decay': 'decay (mean onset, min)'}
ALL_UNITS = ('events', 'time', 'decay')

# DECAY = the mean START TIME of a phase's bouts, in minutes, inside a common 15-minute window.
# Flat process -> 7.5; front-loaded -> lower; back-loaded -> higher. O and P run exactly 15
# minutes so their whole recording is the window; only H (30 min) is truncated by it, which is
# what makes the three phases comparable at all.
#
# THIS REPLACED A FRONT-LOADING FRACTION (bouts in minutes 0-5 / bouts in minutes 0-15, flat ->
# 0.33) on 2026-08-24, for three reasons:
#   * that null was an artefact of the nesting -- a 0-5 / 0-10 split would have nulled at 0.5 --
#     so the number did not interpret itself, whereas 7.5 minutes is half the window and says so;
#   * it collapsed every bout to which side of minute 5 it fell on, discarding the rest of the
#     information in the onset times;
#   * it was a ratio of two correlated counts, so its sampling distribution was awkward, where a
#     mean has an ordinary standard error.
# A Delta on this unit reads directly: "the exposure pushes bouts X minutes later into the phase".
# Both are undefined when a recording has no bout in the window, which is unavoidable.
#
# THE MEDIAN ONSET WAS TESTED AND IS WORSE, on every axis this report decides units by:
#
#                    within-cell CV nt / nn   mean rDelta   worst-cell rDelta   CI width
#     mean onset           0.511 / 0.375          0.502           0.293           3.096
#     median onset         0.590 / 0.542          0.483           0.174           3.463
#
# Not because the median is coarse -- it takes 130 of 144 distinct values here, since onsets are
# frame indices -- but because the tail it discards is SIGNAL, not noise. A flatter habituation
# curve shows up precisely as more late onsets, so trimming them costs sensitivity, which is what
# the worst-cell rDelta drop from 0.293 to 0.174 is. The median wins only on resolved contrasts
# (6/8 against 5/8), which this report explicitly refuses to choose on. Median robustness would be
# worth having if late onsets were measurement error; here they are the phenomenon.
WIN15 = int(15 * 60 * FPS)

# THE OBSERVATION WINDOW, and it applies to every unit.
# ====================================================
# Section 02b's decision: match the first 15 minutes of EVERY phase. H is 9,000 frames (30 min at
# 5 fps), O and P are 4,500 (15 min), so capping at WIN15 truncates H alone and leaves O and P
# bit-for-bit identical -- which is why O->P is unaffected by the choice and only H->O moves.
#
# The reason is a confound, not tidiness: every phase is a separate recording the experimenter
# starts by opening the cage, and the onset spike that follows is LARGEST in P, where the odour is
# removed. A response peaking when the odour is taken away is handling, not odour, so matching
# onset position puts it on both sides of every contrast, where it cancels.
#
# The grid used to be cut on the full recording while 02b said otherwise. It no longer is. `decay`
# was always windowed (mean_onset caps at WIN15); events and time now are too, which is what makes
# the three units describe the same stretch of tape.
WINDOW = WIN15


def in_window(frame_idx) -> np.ndarray:
    """Boolean mask for the matched window. Frame indices are 0-based within an observation."""
    return np.asarray(frame_idx) < WINDOW


def mean_onset(starts) -> float:
    """Mean bout-onset minute inside the first 15 minutes. NaN, not 0, when there are none."""
    s = np.asarray(starts, dtype=float)
    s = s[s < WIN15]
    return float(s.mean() / FPS / 60.0) if len(s) else np.nan


# --------------------------------------------------------------------------- labelled v1 truth
def labelled_truth() -> pd.DataFrame:
    """Per-observation TRUE occupancy and bouts/min on the 24 annotated pools."""
    a = pd.read_csv(ROOT / 'dataset' / 'mice' / 'v1' / 'annotations.csv',
                    usecols=['observation_id', 'frame_idx', 'Y_nt', 'Y_nn'],
                    low_memory=False).dropna(subset=['Y_nt'])
    a = a.sort_values(['observation_id', 'frame_idx'])
    rows = []
    for oid, g in a.groupby('observation_id', sort=False):
        g = g[in_window(g['frame_idx'].to_numpy())]          # 02b's matched window
        n = len(g); rec = {'observation_id': oid}
        fi = g['frame_idx'].to_numpy()
        for lab in LABELS:
            v = g['Y_' + lab].to_numpy()
            rec[f't_time_{lab}'] = v.mean() * 100
            starts = (v == 1) & (np.r_[0, v[:-1]] == 0)
            rec[f't_events_{lab}'] = int(starts.sum()) / (n / FPS / 60)
            rec[f't_decay_{lab}'] = mean_onset(fi[starts])
        rows.append(rec)
    return pd.DataFrame(rows)


# ------------------------------------------------------------- out-of-fold f on labelled pools
def fold_frames(tag):
    d = np.load(FRAME / tag / 'val_probs.npz', allow_pickle=True)
    return pd.DataFrame({'obs': d['obs'], 'gi': d['gi'],
                         'p_nt': d['probs'][:, 0], 'p_nn': d['probs'][:, 1],
                         'y_nt': d['labels'][:, 0], 'y_nn': d['labels'][:, 1]}
                        ).sort_values(['obs', 'gi'])


def best_f1_threshold_events(frames: pd.DataFrame, lab: str) -> float:
    """Event-level max-F1 threshold (any-overlap matching), measured on `frames`."""
    best, best_th = -1.0, 0.5
    for th in THRESHOLDS:
        D = T = H = P = 0
        for _, g in frames.groupby('obs', sort=False):
            tm = g['y_' + lab].to_numpy() > 0.5
            pm = postprocess(g['p_' + lab].to_numpy() >= th, 1, 1)
            tb, pb = runs(tm), runs(pm)
            D += sum(any(not (pe < ts or ps > te) for ps, pe in pb) for ts, te in tb)
            T += len(tb)
            H += sum(any(not (te < ps or ts > pe) for ts, te in tb) for ps, pe in pb)
            P += len(pb)
        rec, pre = (D / T if T else 0), (H / P if P else 0)
        f1 = 2 * pre * rec / (pre + rec) if (pre + rec) > 0 else 0.0
        if f1 > best:
            best, best_th = f1, float(th)
    return best_th


def out_of_fold_predictions(folds=FOLDS):
    """Per-observation out-of-fold f on all 24 labelled pools, plus the per-fold thresholds.

    Thresholds are LEAVE-ONE-FOLD-OUT: tau applied to fold k comes from folds != k.
    """
    frames = {t: fold_frames(t) for t in folds}
    taus = {}
    for k in folds:
        others = pd.concat([frames[j] for j in folds if j != k]) if len(folds) > 1 else frames[k]
        taus[k] = {lab: best_f1_threshold_events(others, lab) for lab in LABELS}
        print(f'  {k}: threshold from the other two folds -> '
              + ', '.join(f'{lab} {taus[k][lab]:.2f}' for lab in LABELS), flush=True)
    rows = []
    for k in folds:
        for oid, g in frames[k].groupby('obs', sort=False):
            g = g.iloc[:WINDOW]                              # frames are stride 1, sorted by gi
            n = len(g); mins = n / FPS / 60
            rec = {'observation_id': oid, 'fold': k}
            for lab in LABELS:
                p = g['p_' + lab].to_numpy()
                rec[f'f_time_{lab}'] = p.mean() * 100
                bouts = runs(postprocess(p >= taus[k][lab], 1, 1))
                rec[f'f_events_{lab}'] = len(bouts) / mins
                # gi is sorted within an observation and every frame is present, so position ==
                # frame_idx and a run's start index is its frame number
                rec[f'f_decay_{lab}'] = mean_onset([b[0] for b in bouts])
            rows.append(rec)
    mean_tau = {lab: float(np.mean([taus[k][lab] for k in folds])) for lab in LABELS}
    return pd.DataFrame(rows), taus, mean_tau


# ------------------------------------------------------------------- f on the unlabelled pools
def dense_predictions(version: str, mean_tau: dict, folds=FOLDS):
    """Cross-prediction: average the K fold models' per-observation predictions.

    Occupancy and bout counts come from the per-fold CSV, which already holds them. The decay
    the decay unit does not -- it needs bout START times, so it comes from the companion .npz,
    which carries the per-frame probability at stride 1 keyed by observation_id with array index
    equal to frame_idx. Averaging happens on the per-observation summaries, matching what the
    labelled side does, rather than on the frame probabilities.
    """
    parts = []
    for t in folds:
        p = FRAME / t / f'pred_dense_{version}.csv'
        if not p.exists():
            print(f'  [missing] {p.relative_to(ROOT)}', flush=True)
            continue
        d = pd.read_csv(p)
        rec = d[['observation_id', 'pool', 'phase', 'odor', 'line', 'sex', 'genotype']].copy()
        npz = FRAME / t / f'pred_dense_{version}.npz'
        if npz.exists():
            # ALL THREE UNITS come from the .npz, not from the CSV's `po_*` and `p_*_t*` columns.
            # Those are aggregates over the WHOLE recording, and 02b's window has to be applied
            # before aggregating, which cannot be undone afterwards. The npz carries the per-frame
            # probability at stride 1 keyed by observation_id, so the cap is a slice.
            z = np.load(npz, allow_pickle=True)
            for lab in LABELS:
                j = LABELS.index(lab)
                time_, ev = [], []
                for o in rec.observation_id:
                    if o not in z.files:
                        time_.append(np.nan); ev.append(np.nan); continue
                    q = z[o][:WINDOW, j].astype(np.float32)
                    time_.append(float(q.mean()) * 100)
                    ev.append(len(runs(postprocess(q >= mean_tau[lab], 1, 1)))
                              / (len(q) / FPS / 60))
                rec[f'f_time_{lab}'] = time_
                rec[f'f_events_{lab}'] = ev
            for lab in LABELS:
                j = LABELS.index(lab)
                rec[f'f_decay_{lab}'] = [
                    mean_onset([b[0] for b in runs(postprocess(
                        z[o][:WINDOW, j].astype(np.float32) >= mean_tau[lab], 1, 1))])
                    if o in z.files else np.nan for o in rec.observation_id]
        else:
            # Without the npz the window cannot be applied, and an unwindowed number would be
            # silently inconsistent with every other estimate. Refuse rather than mix.
            print(f'  [no npz] {npz.name}: SKIPPING this fold -- the window needs per-frame '
                  f'probabilities, and the CSV only has whole-recording aggregates', flush=True)
            continue
        rec['fold'] = t
        parts.append(rec)
    if not parts:
        return None
    allf = pd.concat(parts)
    keys = ['observation_id', 'pool', 'phase', 'odor', 'line', 'sex', 'genotype']
    vals = [f'f_{u}_{l}' for u in ALL_UNITS for l in LABELS]
    return allf.groupby(keys, as_index=False)[vals].mean()


# ------------------------------------------------------------------------------------ the grid
def strata_of(exp_df, version):
    """('all', mask) plus one entry per stratum: line x genotype on v1, line on v2."""
    out = [('all', 'all pools', np.ones(len(exp_df), bool))]
    if version == 'v1':
        for line in sorted(exp_df.line.unique()):
            for g in ('wt', 'het'):
                out.append((f'{line}_{g}', f'{line} · {g}',
                            (exp_df.line == line).to_numpy() & (exp_df.genotype == g).to_numpy()))
    else:
        for line in sorted(exp_df.line.unique()):
            out.append((f'{line}', f'{line} · mixed', (exp_df.line == line).to_numpy()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--report-mismatch', action='store_true',
                    help='print E[f] on labelled vs unlabelled pools per fold -- the quantity '
                         'the old single-fold unlabelled dump got wrong')
    ap.add_argument('--out', default=str(OUT / 'estimates.json'))
    a = ap.parse_args()

    truth = labelled_truth()
    exp1 = pd.read_csv(ROOT / 'data' / 'mice' / 'v1' / 'experiment.csv')
    META1 = ['observation_id', 'pool', 'phase', 'odor', 'line', 'sex', 'genotype', 'annotator']

    ready = {k: v for k, v in PREDICTORS.items() if predictor_ready(v)}
    skipped = [k for k in PREDICTORS if k not in ready]
    if skipped:
        print('predictors not yet complete, skipped: ' + ', '.join(skipped))
    if not ready:
        raise SystemExit('no predictor has all of its dense passes on disk')

    per_pred = {}
    for pkey, spec in ready.items():
        print(f"\n=== {pkey}: {spec['nice']}")
        folds = spec['folds']
        oof, taus, mean_tau = out_of_fold_predictions(folds)
        if spec['human']:
            lab = (truth.merge(oof, on='observation_id')
                   .merge(exp1[META1], on='observation_id'))
            print(f'  labelled: {lab.pool.nunique()} pools, {len(lab)} observations')
        else:
            # Truth is WITHHELD on purpose. This predictor trained on 20 of the 24 annotated
            # pools, so any r or rectifier computed against it would be in sample. PPCI needs no
            # label anywhere, so it is the only estimator this frame can support.
            lab = None
            print('  no out-of-fold predictions: CI and PPI++ withheld, PPCI only')
        print(f'  threshold: ' + ', '.join(f'{k} {v:.2f}' for k, v in mean_tau.items()))
        u1 = dense_predictions('v1', mean_tau, folds)
        u2 = dense_predictions('v2', mean_tau, folds)
        for nm, d in (('v1 unlabelled', u1), ('v2', u2)):
            print(f'  {nm}: ' + (f'{d.pool.nunique()} pools, {len(d)} observations'
                                 if d is not None else 'NOT AVAILABLE'))
        per_pred[pkey] = dict(spec=spec, lab=lab, u1=u1, u2=u2, taus=taus, mean_tau=mean_tau)

    prime = per_pred.get('xfit_dense') or next(iter(per_pred.values()))
    lab1, unl1, v2 = prime['lab'], prime['u1'], prime['u2']
    taus, mean_tau = prime['taus'], prime['mean_tau']

    if a.report_mismatch and unl1 is not None:
        print('\nE[f] mismatch check -- occupancy (pp), mean over observations:')
        for lab in LABELS:
            print(f'  {lab}: labelled out-of-fold {lab1[f"f_time_{lab}"].mean():7.3f}   '
                  f'unlabelled cross-predicted {unl1[f"f_time_{lab}"].mean():7.3f}')
            for t in FOLDS:
                p = FRAME / t / 'pred_dense_v1.csv'
                if p.exists():
                    print(f'      single fold {t}: {pd.read_csv(p)["po_" + lab].mean():7.3f}')

    cells, missing = [], []
    for pkey, P in per_pred.items():
      human = P['spec']['human']
      for version, labdf, unldf in (('v1', P['lab'], P['u1']), ('v2', None, P['u2'])):
        if unldf is None and version == 'v2':
            missing.append(f'{pkey} v2'); continue
        exp_df = pd.read_csv(ROOT / 'data' / 'mice' / version / 'experiment.csv')
        # one frame carrying truth (NaN where unlabelled) and f for every pool of the cohort
        if version == 'v1':
            if unldf is None:
                missing.append(f'{pkey} v1 unlabelled'); frame = labdf.copy()
            elif labdf is None:
                # no truth anywhere for this predictor: every pool is an unlabelled pool
                frame = unldf.copy()
            else:
                frame = pd.concat([labdf, unldf], ignore_index=True)
        else:
            frame = unldf.copy()
            for lab in LABELS:
                for u in ALL_UNITS:
                    frame[f't_{u}_{lab}'] = np.nan
        for sid, snice, _ in strata_of(exp_df.drop_duplicates('pool'), version):
            if sid == 'all':
                sub = frame
            elif version == 'v1':
                line, g = sid.rsplit('_', 1)
                sub = frame[(frame.line == line) & (frame.genotype == g)]
            else:
                sub = frame[frame.line == sid]
            for unit in ALL_UNITS:
                for lab in LABELS:
                    tcol, fcol = f't_{unit}_{lab}', f'f_{unit}_{lab}'
                    for od, odn in ODOURS:
                        for tr in TRANSITIONS:
                            d = pool_deltas(sub, tcol if tcol in sub else None, fcol, od, tr)
                            base = dict(exp=version, unit=unit, behav=lab, stratum=sid,
                                        stratum_label=snice, odour=odn,
                                        trans=f'{tr[0]}->{tr[1]}', model=pkey,
                                        r=None if not np.isfinite(d.r()) else round(d.r(), 4))
                            ests = []
                            if version == 'v1' and human:
                                # CI  -- human annotations only, on the 24 annotated pools
                                e = classical(d); e.method = 'ci'; ests.append(e)
                                # PPI++ -- human annotations rectify the model on all 72
                                ests.append(ppi(d))
                            # PPCI -- model annotations only, UNCALIBRATED. Needs no label
                            # anywhere, which is why it is the only estimator that exists on v2.
                            # It reports on the MODEL's scale: sign and pattern, never magnitude.
                            e = ppci(d, raw=True); e.method = 'ppci'; ests.append(e)
                            for e in ests:
                                if e is None:
                                    continue
                                cells.append({**base, **e.as_dict()})
        n = sum(1 for c in cells if c['exp'] == version and c['model'] == pkey)
        print(f'  {pkey} {version}: built {n} estimates')

    payload = {
        'meta': {
            'behaviours': {k: BEHAV_NICE[k] for k in LABELS},
            'units': UNITS,
            'thresholds_per_fold': taus,
            'threshold_unlabelled': mean_tau,
            'folds': list(FOLDS),
            'predictors': {k: v['nice'] for k, v in ready.items()},
            'design': {
                'v1': {'pools': 72, 'per_line_genotype': 12, 'annotated_pools': 24,
                       'strata': 'line x genotype (6)'},
                'v2': {'pools': 36, 'per_line': 12, 'annotated_pools': 0,
                       'composition': '3 wt + 1 het per cage', 'strata': 'line (3)'}},
            'missing': missing,
        },
        'cells': cells,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(payload, open(a.out, 'w'), indent=1)
    print(f'\nwrote {a.out}  ({len(cells)} estimates)')
    if missing:
        print(f'INCOMPLETE -- still waiting on: {", ".join(missing)}')


if __name__ == '__main__':
    main()
