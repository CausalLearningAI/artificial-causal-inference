#!/usr/bin/env python3
"""Per-model report: what the model does on the target it was trained on, and what that buys PPI++.

Why this exists rather than another AP column
---------------------------------------------
Three separate things were being conflated in the run index:

1. **Frame-exact AP is scored at a resolution the labels do not have.** The annotations are
   7,843 BORIS intervals with start/stop times at 30 fps; frame labels are a derived
   quantization to 5 fps. 22.3% of all bouts (31.9% of nn) are SHORTER than one 5 fps frame,
   so for a large minority of positives, which frame carries the label is set by sub-frame
   phase. That is not annotator error, it is a resolution mismatch the pipeline introduced,
   and no amount of model improvement removes it. So this report adds a **bout-level** view
   (any-overlap matching), which is scored in the units the annotation actually has: did the
   model find the event at all, and how many of its detections correspond to a real event.

2. **`rate_report`'s advertised PPI gain is the N -> infinity limit and is too optimistic.**
   `ppi.projected_variance_factor` says the finite-sample factor is `1 - r^2/(1 + n/N)`, not
   `1 - r^2`. In v1 the het arm has n/N = 18/18 = 1, so its best possible variance factor is
   `1 - r^2/2` -- HALF the advertised reduction. This report quotes the correct one, per arm.

3. **r itself carries an annotator component that no model can predict.** The 4 val pools were
   labelled by 5 different people, and within a single line x genotype cell annotators differ
   by up to 3.7x in behaviour rate (SD 1.67% vs SG 4.40% on kmt5b het, n=18 each). Part of the
   between-observation variance r is asked to explain is therefore annotator identity, not
   biology or video content. That is a mechanism for the already-observed fact that r does not
   track AP run to run. So r is reported with a **cluster bootstrap CI over observations**,
   never as a bare point estimate.

Everything here is computed from `val_probs.npz` (written by training since 2026-08-14, or by
regen_confusion_figs.py for older runs) plus config.json. No GPU, no encoder, seconds to run.

Usage:
    python scripts/mice_behavior/model_report.py                  # every run with a config
    python scripts/mice_behavior/model_report.py --tag res448_k2_ft4_d4 --deep
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
sys.path.insert(0, str(ROOT / 'src'))

from mice_behavior.ppi import projected_variance_factor          # noqa: E402
from mice_behavior.viz import best_f1_threshold                  # noqa: E402

FRAME_DIR = ROOT / 'results' / 'vision' / 'mice' / 'frame'
LABELS = ('nt', 'nn')
# (labeled pools, unlabeled pools) per arm -- the real v1 design, mirrored from ppi_report.py.
DESIGN = {'het': (18, 18), 'wt': (6, 30)}


def bouts(mask: np.ndarray, gi: np.ndarray, obs: np.ndarray) -> list[tuple[int, int]]:
    """Maximal runs of consecutive True frames within one observation, as (start, end) indices.

    Consecutive means consecutive in GLOBAL frame index AND same observation: global indices
    run straight across the observation boundary, so without the second condition the last
    bout of one video would merge with the first of the next.
    """
    out, i, n = [], 0, len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and mask[j + 1] and gi[j + 1] == gi[j] + 1 and obs[j + 1] == obs[j]:
            j += 1
        out.append((i, j))
        i = j + 1
    return out


def bout_metrics(y: np.ndarray, pred: np.ndarray, gi, obs) -> dict:
    """Any-overlap bout matching: did the model find the event, and are its detections real?

    Any-overlap rather than temporal IoU because IoU is not meaningful here -- with 22% of
    bouts shorter than a single frame, the denominator of an IoU is frequently one frame, so
    the metric would be dominated by the same quantization it is supposed to be robust to.
    'Found the event at all' is the weakest claim that is still scientifically meaningful, and
    it is exactly the claim a per-observation RATE depends on.
    """
    gt, pr = bouts(y == 1, gi, obs), bouts(pred, gi, obs)
    if not gt:
        return {}
    gt_hit = sum(1 for a, b in gt if pred[a:b + 1].any())
    pr_hit = sum(1 for a, b in pr if (y[a:b + 1] == 1).any())
    rec = gt_hit / len(gt)
    prec = pr_hit / len(pr) if pr else 0.0
    return {'n_gt_bouts': len(gt), 'n_pred_bouts': len(pr), 'bout_recall': rec,
            'bout_precision': prec,
            'bout_f1': 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0,
            'frag': len(pr) / len(gt)}


def fp_decomposition(y, p, thr, yo, gi, obs, k=3) -> dict:
    """Split confident false positives by what the annotator recorded nearby.

    A false positive next to an annotated bout of the SAME behavior is a boundary or gap
    disagreement; next to the OTHER behavior it is a behaviour confusion; next to nothing it
    is either a genuine model error or an annotator omission, and those two cannot be
    separated without a second annotator (v1 has none -- 0 of 432 observations are labelled
    twice). Reported against the base rate over all negative frames, because a share means
    nothing without one: at 1.7% prevalence some adjacency happens by chance.
    """
    near = np.zeros(len(y), bool); nearo = np.zeros(len(y), bool)
    for d in range(-k, k + 1):
        idx = np.clip(np.searchsorted(gi, gi + d), 0, len(gi) - 1)
        hit = (gi[idx] == gi + d) & (obs[idx] == obs)
        near |= hit & (y[idx] == 1)
        nearo |= hit & (yo[idx] == 1)
    FP = np.where((y == 0) & (p >= thr))[0]
    neg = np.where(y == 0)[0]
    if not len(FP):
        return {}
    return {'n_fp': len(FP), 'fp_boundary': near[FP].mean(),
            'fp_other_beh': (nearo[FP] & ~near[FP]).mean(),
            'fp_unexplained': (~near[FP] & ~nearo[FP]).mean(),
            'fp_boundary_null': near[neg].mean()}


def obs_rates(y, p, obs):
    """Per-observation true and predicted rate -- the quantity PPI++ actually consumes."""
    df = pd.DataFrame({'obs': obs, 'y': y, 'p': p}).groupby('obs').mean()
    return df.y.to_numpy(), df.p.to_numpy()


def r_with_ci(yt, pt, reps=4000, seed=0):
    """Pearson r plus a bootstrap CI resampling OBSERVATIONS (the independent unit).

    n=24, so the SE is ~0.2 and the interval is wide. Printing r bare invites ranking models
    by a number whose CI spans most of its range; the CI is the point.
    """
    if len(yt) < 3:
        return float('nan'), (float('nan'), float('nan'))
    r = float(np.corrcoef(yt, pt)[0, 1])
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(reps):
        i = rng.integers(0, len(yt), len(yt))
        if yt[i].std() < 1e-12 or pt[i].std() < 1e-12:
            continue
        bs.append(np.corrcoef(yt[i], pt[i])[0, 1])
    return r, (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', action='append', default=None, help='run dir (repeatable)')
    ap.add_argument('--deep', action='store_true',
                    help='bout-level metrics and FP decomposition (needs val_probs.npz)')
    ap.add_argument('--context', type=int, default=3, help='+-frames for the FP decomposition')
    args = ap.parse_args()

    tags = args.tag or sorted(d.name for d in FRAME_DIR.iterdir()
                              if (d / 'config.json').exists())
    rows, deep = [], []
    for tag in tags:
        run = FRAME_DIR / tag
        cfg = json.load(open(run / 'config.json'))
        apr, rr = cfg.get('ap_report', {}), cfg.get('rate_report', {})
        if 'macro/tol0' not in apr:
            continue
        npz = run / 'val_probs.npz'
        d = np.load(npz, allow_pickle=True) if npz.exists() else None

        row = {'run': tag, 'mAP0': apr['macro/tol0']['ap'], 'mAP2': apr['macro/tol2']['ap']}
        for li, nm in enumerate(LABELS):
            row[f'AP_{nm}'] = apr[f'{nm}/tol0']['ap']
            if d is not None:
                y, p, gi, obs = (d['labels'][:, li].astype(int), d['probs'][:, li],
                                 d['gi'], d['obs'])
                yt, pt = obs_rates(y, p, obs)
                r, (lo, hi) = r_with_ci(yt, pt)
            else:
                r = rr.get(nm, {}).get('pearson_r', float('nan')); lo = hi = float('nan')
            row[f'r_{nm}'] = r; row[f'r_{nm}_lo'] = lo; row[f'r_{nm}_hi'] = hi
            # correct finite-N factor per arm, not rate_report's 1-r^2
            for arm, (n, N) in DESIGN.items():
                row[f'ppi_{nm}_{arm}'] = projected_variance_factor(r, n, N)
        row['has_npz'] = d is not None
        rows.append(row)

        if args.deep and d is not None:
            for li, nm in enumerate(LABELS):
                y = d['labels'][:, li].astype(int); p = d['probs'][:, li]
                yo = d['labels'][:, 1 - li].astype(int)
                gi, obs = d['gi'], d['obs']
                thr, prec, rec, f1 = best_f1_threshold(y, p)
                e = {'run': tag, 'beh': nm, 'thr': thr, 'frame_P': prec, 'frame_R': rec,
                     'frame_F1': f1}
                e |= bout_metrics(y, p >= thr, gi, obs)
                e |= fp_decomposition(y, p, thr, yo, gi, obs, k=args.context)
                deep.append(e)

    df = pd.DataFrame(rows).sort_values('mAP0', ascending=False)
    pd.set_option('display.width', 200, 'display.max_columns', 50)

    print('=' * 108)
    print('FRAME TARGET (what the model was trained on) and PROJECTED PPI++ GAIN')
    print('=' * 108)
    show = df[['run', 'mAP0', 'mAP2', 'AP_nt', 'AP_nn', 'r_nt', 'r_nn',
               'ppi_nt_het', 'ppi_nt_wt', 'ppi_nn_het', 'ppi_nn_wt', 'has_npz']]
    print(show.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    print('\nppi_* = Var(PPI++)/Var(classical) = 1 - r^2/(1+n/N) at the real v1 design')
    print('        (het 18 labeled/18 unlabeled, wt 6/30). LOWER is better; 1.000 = no gain.')
    print('        This is NOT rate_report\'s 1-r^2, which is the N->inf limit and too optimistic.')
    print('        r is an OBSERVATION-level correlation used as a stand-in for the POOL-level')
    print('        one PPI actually needs; a real number requires cross-fitted pool predictions')
    print('        (dump_pool_predictions.py --kfold), which no run has yet.')

    if not df.has_npz.any():
        print('\n[!] no val_probs.npz found -- r values are the stored point estimates, no CI.')
    else:
        print('\n' + '=' * 108)
        print('r WITH CLUSTER BOOTSTRAP CI OVER OBSERVATIONS (n=24) -- runs with val_probs.npz')
        print('=' * 108)
        for _, x in df[df.has_npz].iterrows():
            print(f'  {x["run"]:34s} '
                  f'nt r={x["r_nt"]:+.3f} [{x["r_nt_lo"]:+.3f},{x["r_nt_hi"]:+.3f}]   '
                  f'nn r={x["r_nn"]:+.3f} [{x["r_nn_lo"]:+.3f},{x["r_nn_hi"]:+.3f}]')

    if deep:
        dd = pd.DataFrame(deep)
        print('\n' + '=' * 108)
        print('BOUT LEVEL (the units the annotation actually has) and FALSE-POSITIVE ANATOMY')
        print('=' * 108)
        c = ['run', 'beh', 'thr', 'frame_F1', 'n_gt_bouts', 'bout_recall', 'bout_precision',
             'bout_f1', 'frag', 'n_fp', 'fp_boundary', 'fp_other_beh', 'fp_unexplained',
             'fp_boundary_null']
        print(dd[[x for x in c if x in dd]].to_string(index=False,
                                                      float_format=lambda v: f'{v:.3f}'))
        print('\nbout_recall     = annotated bouts with >=1 frame detected (any-overlap match)')
        print('bout_precision  = predicted segments overlapping a real bout')
        print('frag            = predicted segments per annotated bout (>1 = over-segmentation)')
        print('fp_boundary     = confident FPs within +-{} frames of a bout of the SAME behavior'
              .format(args.context))
        print('fp_boundary_null= same measure over ALL negative frames -- the chance rate')
        print('fp_unexplained  = nothing annotated nearby: genuine error OR annotator omission,')
        print('                  indistinguishable without a second annotator (v1 has none).')
    return df


if __name__ == '__main__':
    main()
