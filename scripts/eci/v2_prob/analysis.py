"""
ECI v2: a LABEL-FREE per-neuron PROBABILITY READOUT, and an AUDIT of the alignment metrics (R~ and the others).

==================================================================================================================
PRE-REGISTRATION (written 2026-10-25 ~01:30, BEFORE any score of this study existed: no readout, metric or label was
computed on the codes below; the probe-set encodes of scripts/eci/v2_prob/study.py were still queued). Known before
writing: the earlier binarisation test (src/eci/v2/prob.py: the per-neuron log-normal mixture's MAP 'on' covers 43-54%
of all frames for behaviour neurons; mice nose-nose R~ 0.202 continuous -> 0.090 prob / 0.08 MAP), the release
tables (results/vision/eci_v2_diag/*_table.md) and the label bout statistics (mice atlas window: nose-nose 2.1% of
frames, 3,778 bouts, median 2 frames = 0.4 s; nose-tail 1.0%, 1,058 bouts, median 3 frames; ants v2 grooming 22%,
201 bouts, median 78 frames = 16 s).

QUESTION 1. Can a simple, label-free per-neuron model turn a neuron's frame code into P(on) that (a) keeps the
neuron's alignment ranking, (b) discretises into bouts that match the behaviour, (c) is calibrated against the
behaviour, (d) gives per-video quantities that track the annotated ones?
QUESTION 2. Is R~ (normalised R-precision) the right alignment metric? Which metric should be primary?

DATA (labels are read ONLY for selection and scoring; every readout is fitted without labels)
  Tier A ('full'): the two deployed dictionaries whose contiguous 5 fps codes already exist for every labelled frame:
      v1     deployed matryoshka_btk_1024_k16_{fg448al | antsfg}_s0, its stored codes_max rows;
      v2dep  the frozen deployed v2 (mice s7, ants s6), its stored per-video codes_max (raw frame max).
      Frames: mice = all 144 annotated videos in the atlas window (habituation '_H': frames >= 4500; 648,000 frames);
      ants v2 = 44 videos x frames 0-2999 (132,000); ants v3 = 107 videos x frames 0-2999 (321,000).
  Tier B ('probe L', scripts/eci/v2_prob/study.py): 2-min (mice) / 2-min (ants v2) / 1-min (ants v3) contiguous
      windows of the same labelled videos, codes of 52 dictionaries per domain: v1 s0 (deployed) + v1 retrained s1-10,
      V2F_M0 s900-919, V2F_N2 s900-919, v2dep (bridge to Tier A).
  Cells (label, series): mice nose_nose = Y_nn | Y_np, mice nose_tail = Y_nt; ants v2 groom_any = Y2F | B2F;
      ants v3 groom_any; ants v3 on_lid_any = YOL | BOL (frames where both exist). 5 cells.
  Units / splits: mice pools (observation_id without its last two fields), ants videos. 'atlas' split = sorted units
      alternating; rnd0-9 = random halves (rng 1000 + s, src/eci/v2/score.splits_of). Tier A uses atlas + rnd0-9;
      Tier B atlas + rnd0-4 (cost); the metric audit (Question 2) uses atlas + rnd0-4 in both tiers. Every split is used in both directions (A -> B and B -> A).
  Continuous code: c = frame max (raw); s = the 0.6 s centred mean of c within the video / window (src/eci/v2/
      readout.smooth, h = 1), the current v2 readout. All dictionaries are read the same way (also v1).

QUESTION 1: READOUTS. Per cell, split and direction, the 'best-aligned neuron' j* = argmax over the full dictionary of
  R~(s) on half A (the current selection rule). Every readout of j* is FITTED ON HALF A WITHOUT LABELS and scored on
  half B:
    CONT   s itself (ranking and per-video mean only; it has no probability and no discrete state).
    HMM2   src/eci/v2/hmm.py, K = 2, on the RAW 5 fps frame code c (the smoothing is replaced by the state model);
           P(on) = forward-backward posterior; discrete = Viterbi path in the on state.
    HMM3   the same with K = 3 (silent / low / high); on = the state with the largest expected code.
    MIX    the old prob readout (src/eci/v2/prob.py fit_neuron, unchanged: zero-inflated 2-component log-normal mixture
           on s, monotone posterior), fitted on half A; discrete = MAP (P > 0.5). P kept in float64.
    Q95    fixed quantile baseline: discrete = s > the 95th percentile of s on half A (the runners' pooled q 0.95 rule,
           per neuron); P = the binary.
    FNORM  the deployed v2 firing rule (src/eci/v2/readout.calibrate_fnorm, k = 16, all neurons of the dictionary,
           half-A frames; Tier A every 5th frame for the global threshold), discrete = fires; P = the binary.
  Metrics on half B (labelled frames of the cell):
    ranking      R~ and AP / nAP = (AP - pi) / (1 - pi) of P (CONT: of s);
    calibration  Brier, Brier skill BSS = 1 - Brier / (pi (1 - pi)) (> 0 = better than the constant base rate),
                 ECE (10 equal-width bins of P, |mean P - mean y| weighted by bin size), mean P / pi; reliability
                 bins pooled over splits and directions are saved;
    discrete     frame precision / recall / F1; bout F1 = segmental F1 at IoU >= 0.25 (a predicted bout is a hit iff
                 its best-IoU true bout in the same video has IoU >= 0.25 and is not yet matched; recall = matched
                 true bouts / true bouts), also reported at IoU 0.1 and 0.5;
    per video    (half-B videos; ants v3 lid: valid frames) Pearson r of the expected fraction of time (mean P; CONT:
                 mean s) with the annotated fraction; Pearson r of the discrete bout count and the annotated bout
                 count; the NES / NESS summaries of the discrete state: fraction of time, bout rate per minute,
                 latency (first on frame; censored = window length) vs the annotated ones (Pearson for fraction and
                 bout rate, Spearman for latency). Tier A videos are 10-15 min (the meaningful per-video test);
                 Tier B windows are 1-2 min.
    label-free   share of half-B frames in the on state, HMM stationary on share and mean on-bout length.
  DIAGNOSIS of the MIX failure (descriptive, Tier A, the R~-selected neurons): share of frames with s > 0, MIX MAP on
    share vs the label base rate, R~ of MIX P in float64 vs fp16 (the stored '_prob' codes are fp16: saturation ties),
    share of the top-N (N = positives) frames with P >= 0.999.
  DECISION RULE (readout). Cells for the decision = Tier A {v1, v2dep} x 5 + Tier B {v1 (mean of s0-10), V2F_M0
    (mean of 20 seeds), V2F_N2 (mean of 20), v2dep} x 5 = 30 cells, equal weight; per cell the value is the mean over
    splits and directions. Candidates HMM2, HMM3, MIX, Q95, FNORM.
    (i) ranking kept: mean over cells of nAP(P) - nAP(CONT) >= -0.03;
    (ii) among (i)-passing candidates, RECOMMEND the one with the highest mean frame F1 of its discrete state; within
         0.005, the higher mean bout F1 @ 0.25.
    Reported but not deciding: whether P(on) is CALIBRATED to the behaviour, per cell: BSS > 0 AND ECE < 0.05. If the
    winner fails that on most mice cells, the report says P(on) is the probability of the neuron's own state, not of
    the behaviour.

QUESTION 2: METRIC AUDIT. Per dictionary (Tier B: all 52; Tier A: 2), cell and ELIGIBLE neuron (s > 0 on >= 0.2% of
  the cell's frames), on s:
    Rt      R~ (ties pro rata, = src/eci/v2/score.py);
    AP      average precision (sklearn definition, ties grouped); nAP = (AP - pi) / (1 - pi);
    AUROC   Mann-Whitney, ties averaged;
    F1max   the best frame F1 over thresholds (on iff s >= t);
    BF1     bout F1 @ 0.25 of the F1max threshold's binary;
    VR      Pearson r over videos of the mean s with the annotated fraction.
  Within one cell AP and nAP rank neurons identically; nAP is the one compared across cells / prevalences.
  Analyses:
    A1 neuron-level agreement: Spearman between every metric pair over the eligible neurons (all frames of the cell),
       median over dictionaries; also over the TOP SET (neurons in the top 50 of at least one metric).
    A2 best-neuron agreement: share of (dictionary, cell) where two metrics pick the same best neuron.
    A3 representation-level agreement: per dictionary, metric M's CROSS-FITTED value (best neuron by M on half A,
       its M on half B; mean over splits and directions); Spearman between metrics over the 52 dictionaries per cell;
       arm means (v1 s0, v1 s1-10, V2F_M0, V2F_N2, v2dep).
    A4 reliability: split-half Spearman of M over the eligible neurons (half A vs half B), median over dictionaries
       and splits; arm discriminability eta^2 = between-arm share of the variance of the cross-fitted M over the
       seeds of v1 (s0-10), V2F_M0, V2F_N2.
    A5 downstream validity (what causal analysis needs = per-video quantities and a discretised state): for metric M,
       the neuron selected by M on half A; on half B: DS = mean of (VR of its mean s, frame F1 and bout F1 @ 0.25 at
       the threshold that maximises frame F1 on half A, i.e. the threshold transferred). V(M) = mean DS over the
       Tier B dictionaries, cells, splits and directions.
    PREVALENCE (the known R~ issue): for each Tier A cell's R~-selected neuron (all frames), resample to base rates
       0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5 (drop negatives or positives at random, 20 draws) and recompute Rt, AP,
       nAP, AUROC, F1max.
  DECISION RULE (metric). Candidates Rt, nAP, AUROC, F1max, BF1, VR. PRIMARY = the highest V(M) (A5); if the top two
    are within 0.01, the one with the higher A4 split-half reliability. SECONDARY = the next two by V(M), plus AUROC as
    the prevalence-free descriptor if it is not among them.
BUDGET: <= 15 GPU-h in total (encodes ~8, codes ~1); this file is CPU only.
==================================================================================================================

Steps (scripts/eci/v2_prob/job.sh with SCRIPT=analysis.py, EXTRA = arguments):
    tierA_extract --dom D          (CPU)  Tier A codes -> BEE/<dom>/tierA/<dict>.npy (fp16, frame-table order)
    run --dom D --tier A|B --dict NAME   (CPU)  readouts + audit for one dictionary -> OUT/<dom>/<tier>/<dict>.{json,npz}
    report                         (CPU)  decisions, tables -> OUT/{decision.json, table.md}
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit, prange

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from src.eci.v2 import hmm as HMM  # noqa: E402
from src.eci.v2 import prob as PROB  # noqa: E402
from src.eci.v2.readout import calibrate_fnorm, smooth  # noqa: E402

OUT = REPO / 'results/vision/eci_v2_prob'
BEE = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_prob')
EC = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes')
FPS = 5.0
CELLS = {'mice': [('v1', 'nose_nose'), ('v1', 'nose_tail')],
         'ants': [('v2', 'groom_any'), ('v3', 'groom_any'), ('v3', 'on_lid_any')]}
LABCOLS = {'nose_nose': ['Y_nn', 'Y_np'], 'nose_tail': ['Y_nt'], 'groom_any': ['Y_Y2F', 'Y_B2F'],
           'on_lid_any': ['Y_YOL', 'Y_BOL']}
READOUTS = ('CONT', 'HMM2', 'HMM3', 'MIX', 'Q95', 'FNORM')
METRICS = ('Rt', 'nAP', 'AUROC', 'F1max', 'BF1', 'VR')
ELIG = 0.002
TIERA = {'mice': {'v1': 'dataset/mice/v1/eci/codes/matryoshka_btk_1024_k16_fg448al_s0/codes_max.npy',
                  'v2dep_s7': 'v2/mice/videos/v1'},
         'ants': {'v1': 'dataset/ants/eci30/codes/matryoshka_btk_1024_k16_antsfg_s0/codes_max.npy',
                  'v2dep_s6': 'v2/ants/videos'}}


def log(s):
    print(time.strftime('%H:%M:%S'), s, flush=True)


# ---------------------------------------------------------------------------------------------- frames
def frame_table(dom, tier):
    """Frames of a tier (rows = code rows), with series, obs, k, unit, row (annotations.csv row), per-cell y / valid."""
    if dom == 'mice':
        a = pd.read_csv(REPO / 'dataset/mice/v1/annotations.csv',
                        usecols=['observation_id', 'frame_idx', 'Y_nn', 'Y_np', 'Y_nt'])
        a['experiment'] = 'v1'
    else:
        a = pd.read_csv(REPO / 'dataset/ants/eci30/annotations.csv', dtype={'observation_id': str},
                        usecols=['observation_id', 'frame_idx', 'experiment', 'Y_Y2F', 'Y_B2F', 'Y_YOL', 'Y_BOL'])
    a['row'] = np.arange(len(a))
    if tier == 'A':
        if dom == 'mice':
            F = a[a.Y_nn.notna() & (~a.observation_id.str.endswith('_H') | (a.frame_idx >= 4500))]
        else:
            F = a[a.experiment.isin(['v2', 'v3']) & a.Y_Y2F.notna()]
        F = F.rename(columns={'observation_id': 'obs', 'frame_idx': 'k', 'experiment': 'series'})
    else:
        Pr = pd.read_parquet(OUT / dom / 'probe.parquet')
        idx = pd.Series(np.arange(len(a)), index=pd.MultiIndex.from_arrays([a.observation_id.values,
                                                                            a.frame_idx.values]))
        r = idx.reindex(pd.MultiIndex.from_arrays([Pr.obs.values.astype(str), Pr.k.values.astype(int)])).values
        assert not np.isnan(r).any()
        F = a.iloc[r.astype(int)].rename(columns={'observation_id': 'obs', 'frame_idx': 'k'})
        F['series'] = Pr.series.values
    F = F.reset_index(drop=True)
    F['unit'] = F.obs.str.rsplit('_', n=2).str[0] if dom == 'mice' else F.obs
    for ser, lb in CELLS[dom]:
        cols = LABCOLS[lb]
        F[f'y_{lb}'] = (F[cols].fillna(0).to_numpy() > 0).any(1)
        F[f'v_{ser}_{lb}'] = F[cols].notna().all(1).to_numpy() & (F.series == ser).to_numpy()
    return F


def seq_starts(F):
    """contiguous runs (same obs, consecutive k) -> starts (m + 1,)"""
    o = F.obs.values
    k = F.k.values
    br = np.r_[True, (o[1:] != o[:-1]) | (k[1:] != k[:-1] + 1)]
    return np.r_[np.flatnonzero(br), len(F)].astype(np.int64)


def smooth_seq(C, starts):
    out = np.empty(C.shape, np.float32)
    for a, b in zip(starts[:-1], starts[1:]):
        out[a:b] = smooth(C[a:b], 1)
    return out


def splits(units, names):
    u = sorted(set(units))
    out = {}
    for nm in names:
        if nm == 'atlas':
            h = np.array([i % 2 for i in range(len(u))])
        else:
            s = int(nm[3:])
            h = np.zeros(len(u), int)
            h[np.random.default_rng(1000 + s).permutation(len(u))[:len(u) // 2]] = 1
        out[nm] = dict(zip(u, h))
    return out


# ---------------------------------------------------------------------------------------------- metric kernels
@njit(cache=True)
def rank_metrics(s, y):
    """s float64 (n,), y bool (n,) -> (Rt, AP, AUROC, F1max, threshold of F1max, base rate)."""
    n = s.shape[0]
    P = 0
    for i in range(n):
        P += y[i]
    N = n - P
    if P == 0 or N == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, P / max(n, 1)
    o = np.argsort(-s, kind='mergesort')
    ctp, cfp = 0, 0
    ap, auc, rt_tp = 0.0, 0.0, -1.0
    f1b, thr = -1.0, np.inf
    i = 0
    while i < n:
        v = s[o[i]]
        j = i
        gp = 0
        while j < n and s[o[j]] == v:
            gp += y[o[j]]
            j += 1
        g = j - i
        gn = g - gp
        before = ctp + cfp
        if rt_tp < 0 and before + g >= P:
            rt_tp = ctp + gp * (P - before) / g
        auc += gp * (N - cfp - gn) + 0.5 * gp * gn
        ctp += gp
        cfp += gn
        if gp > 0:
            ap += gp / P * ctp / (ctp + cfp)
        f1 = 2.0 * ctp / (ctp + cfp + P)
        if f1 > f1b:
            f1b, thr = f1, v
        i = j
    pi = P / n
    return (rt_tp / P - pi) / (1 - pi), ap, auc / (P * N), f1b, thr, pi


@njit(cache=True)
def _segs(b, a0, a1):
    st = []
    en = []
    t = a0
    while t < a1:
        if b[t]:
            u = t
            while u < a1 and b[u]:
                u += 1
            st.append(t)
            en.append(u)
            t = u
        else:
            t += 1
    return st, en


@njit(cache=True)
def bout_f1(pred, y, starts, tau):
    """segmental F1 at IoU >= tau over the sequences; -> (F1, n_pred, n_true, tp)."""
    tp, npred, ntrue = 0, 0, 0
    for q in range(starts.shape[0] - 1):
        a0, a1 = starts[q], starts[q + 1]
        ps, pe = _segs(pred, a0, a1)
        ts, te = _segs(y, a0, a1)
        npred += len(ps)
        ntrue += len(ts)
        used = np.zeros(len(ts), np.bool_)
        lo = 0
        for i in range(len(ps)):
            while lo < len(ts) and te[lo] <= ps[i]:
                lo += 1
            best, arg = 0.0, -1
            m = lo
            while m < len(ts) and ts[m] < pe[i]:
                inter = min(pe[i], te[m]) - max(ps[i], ts[m])
                uni = max(pe[i], te[m]) - min(ps[i], ts[m])
                iou = inter / uni
                if iou > best:
                    best, arg = iou, m
                m += 1
            if arg >= 0 and best >= tau and not used[arg]:
                used[arg] = True
                tp += 1
    if npred + ntrue == 0:
        return np.nan, 0, 0, 0
    return 2.0 * tp / (npred + ntrue), npred, ntrue, tp


@njit(parallel=True, cache=True)
def metrics_cols(X, rows, y, starts, vv, nvid):
    """X (n, m) float32 Fortran-ordered codes, rows (r,) the frames used, y (r,) bool, starts sequences over rows,
    vv (r,) video code -> (m, 6) [Rt, nAP, AUROC, F1max, BF1, VR], (m,) F1max threshold."""
    m = X.shape[1]
    out = np.full((m, 6), np.nan)
    thr = np.full(m, np.nan)
    cnt = np.zeros(nvid)
    sy = np.zeros(nvid)
    for t in range(rows.shape[0]):
        cnt[vv[t]] += 1
        sy[vv[t]] += y[t]
    for q in prange(m):
        col = X[:, q]
        x = np.empty(rows.shape[0])
        for t in range(rows.shape[0]):
            x[t] = col[rows[t]]
        rt, ap, au, f1, th, pi = rank_metrics(x, y)
        if not np.isfinite(rt):
            continue
        bf = bout_f1(x >= th, y, starts, 0.25)[0]
        sx = np.zeros(nvid)
        for t in range(rows.shape[0]):
            sx[vv[t]] += x[t]
        # Pearson over videos with frames
        k = 0
        a = np.empty(nvid)
        b = np.empty(nvid)
        for v in range(nvid):
            if cnt[v] > 0:
                a[k] = sx[v] / cnt[v]
                b[k] = sy[v] / cnt[v]
                k += 1
        a, b = a[:k], b[:k]
        vr = np.nan
        if k >= 3 and a.std() > 0 and b.std() > 0:
            vr = ((a - a.mean()) * (b - b.mean())).mean() / (a.std() * b.std())
        out[q, 0], out[q, 1], out[q, 2], out[q, 3], out[q, 4], out[q, 5] = rt, (ap - pi) / (1 - pi), au, f1, bf, vr
        thr[q] = th
    return out, thr


def sub_starts(mask, starts):
    """restrict sequences to frames where mask is True (a masked-out frame breaks a sequence) -> (idx, starts)."""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return idx, np.zeros(1, np.int64)
    seq = np.searchsorted(starts, idx, side='right') - 1
    br = np.r_[True, (np.diff(idx) != 1) | (np.diff(seq) != 0)]
    return idx, np.r_[np.flatnonzero(br), len(idx)].astype(np.int64)


def video_stats(score, y, vid_of, nvid):
    """per video means of score and y -> Pearson r (nan if < 3 videos or constant)."""
    cnt = np.bincount(vid_of, minlength=nvid)
    ok = cnt > 0
    ms = np.bincount(vid_of, score, minlength=nvid)[ok] / cnt[ok]
    my = np.bincount(vid_of, y.astype(float), minlength=nvid)[ok] / cnt[ok]
    return pearson(ms, my)


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b):
    from scipy.stats import spearmanr
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or a[ok].std() == 0 or b[ok].std() == 0:
        return float('nan')
    return float(spearmanr(a[ok], b[ok]).correlation)


def per_video_summaries(b, y, starts, P=None):
    """b, y bool over frames with sequence starts (one sequence per video window) -> dict of arrays per sequence."""
    rows = []
    for a0, a1 in zip(starts[:-1], starts[1:]):
        L = a1 - a0
        if L == 0:
            continue
        r = {}
        for nm, x in (('pred', b[a0:a1]), ('true', y[a0:a1])):
            st = int(x[0]) + int((x[1:] & ~x[:-1]).sum())
            first = int(np.argmax(x)) if x.any() else L
            r[nm] = (float(x.mean()), st / (L / FPS / 60), first / FPS, st)
        r['meanP'] = float(P[a0:a1].mean()) if P is not None else np.nan
        rows.append(r)
    return rows


def summary_corr(rows):
    if len(rows) < 3:
        return {}
    pr = np.array([r['pred'] for r in rows])
    tr = np.array([r['true'] for r in rows])
    mp = np.array([r['meanP'] for r in rows])
    return {'r_meanP_fraction': pearson(mp, tr[:, 0]) if np.isfinite(mp).all() else float('nan'),
            'r_fraction': pearson(pr[:, 0], tr[:, 0]), 'r_bout_rate': pearson(pr[:, 1], tr[:, 1]),
            'rho_latency': spearman(pr[:, 2], tr[:, 2]), 'r_bout_count': pearson(pr[:, 3], tr[:, 3]),
            'mean_pred_fraction': float(pr[:, 0].mean()), 'mean_true_fraction': float(tr[:, 0].mean()),
            'mean_pred_bouts': float(pr[:, 3].mean()), 'mean_true_bouts': float(tr[:, 3].mean())}


def calib(P, y, nb=10):
    P = np.asarray(P, float)
    yv = y.astype(float)
    pi = yv.mean()
    br = float(((P - yv) ** 2).mean())
    b = np.minimum((P * nb).astype(int), nb - 1)
    cnt = np.bincount(b, minlength=nb)
    sp = np.bincount(b, P, minlength=nb)
    sy = np.bincount(b, yv, minlength=nb)
    ece = float(np.abs(sp - sy).sum() / len(P))
    return {'brier': br, 'bss': 1 - br / max(pi * (1 - pi), 1e-12), 'ece': ece, 'meanP_over_pi': float(P.mean() / pi),
            'bins': np.stack([cnt, sp, sy]).tolist()}


# ---------------------------------------------------------------------------------------------- one dictionary
def load_codes(dom, tier, dname, F):
    if tier == 'A':
        return np.load(BEE / dom / 'tierA' / f'{dname}.npy', mmap_mode='r')
    return np.load(BEE / dom / 'codes' / f'{dname}.npy', mmap_mode='r')


def cmd_tierA_extract(args):
    dom = args.dom
    F = frame_table(dom, 'A')
    od = BEE / dom / 'tierA'
    od.mkdir(parents=True, exist_ok=True)
    F[['obs', 'k', 'series', 'row']].to_parquet(od / 'frames.parquet')
    for dname, src in TIERA[dom].items():
        if dname == 'v1':
            cm = np.load(REPO / src, mmap_mode='r')
            rows = F.row.values
            o = np.argsort(rows)
            X = np.empty((len(rows), cm.shape[1]), np.float16)
            for c0 in range(0, len(rows), 50000):
                ii = o[c0:c0 + 50000]
                X[ii] = cm[rows[ii]]
        else:
            X = np.zeros((len(F), 2048), np.float16)
            for (ser, obs), g in F.groupby(['series', 'obs'], sort=False):
                p = EC / src / (f'{ser}/{obs}.npz' if dom == 'ants' else f'{obs}.npz')
                X[g.index.values] = np.load(p)['codes_max'][g.k.values]
        np.save(od / f'{dname}.npy', X)
        log(f'tierA {dom} {dname}: {X.shape}, active share {(X > 0).mean():.4f}')


def run_dict(dom, tier, dname, split_names, audit_splits, do_readouts=True, do_audit=True):
    F = frame_table(dom, tier)
    if tier == 'A':
        chk = pd.read_parquet(BEE / dom / 'tierA' / 'frames.parquet')
        assert (chk.row.values == F.row.values).all()
    C = np.asarray(load_codes(dom, tier, dname, F), np.float32)
    starts = seq_starts(F)
    S = smooth_seq(C, starts)
    W = C.shape[1]
    SP = splits(F.unit.values, split_names)
    res = {'dict': dname, 'tier': tier, 'dom': dom, 'n_frames': len(F), 'W': W, 'cells': {}}
    audit = {}
    fn_cache = {}
    vids = F.obs.values
    vcode = pd.factorize(vids)[0]
    nvid = vcode.max() + 1
    for ser, lb in CELLS[dom]:
        cell = f'{ser}_{lb}'
        y_all = F[f'y_{lb}'].values
        v_all = F[f'v_{ser}_{lb}'].values
        sermask = (F.series == ser).values
        t0 = time.time()
        per = {}
        # ---- audit (all eligible neurons), full frames and every split half
        idx_c, st_c = sub_starts(v_all, starts)
        yc = y_all[idx_c]
        Sc = S[idx_c]
        rate = (Sc > 0).mean(0)
        elig = np.flatnonzero(rate >= ELIG)
        halves = {nm: np.array([SP[nm][u] for u in F.unit.values[idx_c]]) for nm in split_names}

        Xc = np.asfortranarray(Sc[:, elig])
        vc = vcode[idx_c].astype(np.int64)

        def metrics_on(sel, st_sel, cols=None):
            return metrics_cols(Xc, sel.astype(np.int64), yc[sel], st_sel, vc[sel], nvid)

        if do_audit:
            full_m, full_t = metrics_on(np.arange(len(idx_c)), st_c)
            for nm in audit_splits:
                for h in (0, 1):
                    sel = np.flatnonzero(halves[nm] == h)
                    _, st_h = sub_starts(halves[nm] == h, st_c)
                    per[f'{nm}_{h}'] = metrics_on(sel, st_h)
            # A5: downstream of each metric's selection (threshold transferred)
            ds = {}
            for nm in audit_splits:
                for h in (0, 1):
                    mA, tA = per[f'{nm}_{h}']
                    selB = np.flatnonzero(halves[nm] == 1 - h)
                    _, stB = sub_starts(halves[nm] == 1 - h, st_c)
                    yB = yc[selB]
                    vB = vcode[idx_c][selB]
                    for mi, M in enumerate(METRICS):
                        col = mA[:, mi]
                        if not np.isfinite(col).any():
                            continue
                        q = int(np.nanargmax(col))
                        j = elig[q]
                        x = Sc[selB, j].astype(np.float64)
                        b = x >= tA[q]
                        tp = float((b & yB).sum())
                        f1 = 2 * tp / max(b.sum() + yB.sum(), 1)
                        bf = bout_f1(b, yB, stB, 0.25)[0]
                        vr = video_stats(x, yB, vB, nvid)
                        mB = per[f'{nm}_{1 - h}'][0][q]
                        ds.setdefault(M, []).append({'split': nm, 'dir': h, 'neuron': int(j), 'cf_value': float(mB[mi]),
                                                     'cf_all': mB.tolist(), 'f1_transfer': f1, 'bf1_transfer': bf,
                                                     'vr': vr, 'DS': float(np.nanmean([vr, f1, bf]))})
            audit[cell] = {'elig': elig, 'full': full_m, 'full_thr': full_t,
                           'per': {k_: v[0] for k_, v in per.items()}, 'ds': ds, 'rate': rate[elig]}
            log(f'{dname} {cell}: audit {len(elig)} eligible neurons, {time.time() - t0:.0f}s')
        # ---- readouts of the R~-selected neuron
        if not do_readouts:
            continue
        rec = {'base_rate': float(yc.mean()), 'n': int(len(yc)), 'splits': []}
        idx_s, st_s = sub_starts(sermask, starts)  # all frames of the series (HMM sequences)
        pos_in_s = np.full(len(F), -1)
        pos_in_s[idx_s] = np.arange(len(idx_s))
        unit_s = F.unit.values[idx_s]
        for nm in split_names:
            hs = np.array([SP[nm][u] for u in unit_s])
            for h in (0, 1):
                if f'{nm}_{h}' in per:
                    rtA = per[f'{nm}_{h}'][0][:, 0]
                else:
                    selA_c = np.flatnonzero(halves[nm] == h)
                    _, stA_c = sub_starts(halves[nm] == h, st_c)
                    rtA = metrics_on(selA_c, stA_c)[0][:, 0]
                j = int(elig[int(np.nanargmax(rtA))])
                mA_s = hs == h
                _, stA = sub_starts(mA_s, st_s)
                xa_raw = C[idx_s[mA_s], j].astype(np.float64)
                xa_s = S[idx_s[mA_s], j].astype(np.float64)
                mB_s = hs != h
                iB_s, stB = sub_starts(mB_s, st_s)
                xb_raw = C[idx_s[iB_s], j].astype(np.float64)
                xb_s = S[idx_s[iB_s], j].astype(np.float64)
                # half-B labelled positions inside the series' half-B frames
                gB = idx_s[iB_s]
                lab = v_all[gB]
                yB = y_all[gB]
                vB = vcode[gB]
                P, D = {}, {}
                P['CONT'] = xb_s
                info = {}
                for K in (2, 3):
                    p = HMM.fit(xa_raw, stA, K)
                    Pk, Vk = HMM.apply(xb_raw, stB, p)
                    P[f'HMM{K}'], D[f'HMM{K}'] = Pk, Vk
                    so, bl = HMM.stationary_on(p)
                    info[f'HMM{K}'] = {'stationary_on': so, 'mean_on_bout_frames': bl,
                                       'iters': None if p is None else p['iters'],
                                       'z': None if p is None else p['z'].tolist(),
                                       'mu': None if p is None else p['mu'].tolist(),
                                       'on': None if p is None else p['on']}
                    D[f'HMM{K}_p05'] = Pk > 0.5
                fm = PROB.fit_neuron(xa_s[xa_s > 0], np.random.default_rng(0))
                if fm is None:
                    Pm = (xb_s > 0).astype(float)
                else:
                    Pm = np.where(xb_s > 0, np.interp(np.log(np.where(xb_s > 0, xb_s, 1)), fm['grid'], fm['curve']), 0)
                P['MIX'], D['MIX'] = Pm, Pm > 0.5
                P['MIX_fp16'] = Pm.astype(np.float16).astype(np.float64)
                q95 = np.quantile(xa_s, 0.95)
                D['Q95'] = xb_s > q95
                P['Q95'] = D['Q95'].astype(float)
                # FNORM: all neurons, half-A frames (Tier A: every 5th)
                if (nm, h, ser) not in fn_cache:
                    fa = idx_s[mA_s][::5] if tier == 'A' else idx_s[mA_s]
                    fn_cache[(nm, h, ser)] = calibrate_fnorm([S[fa]], k=16)[:2]
                sig, th = fn_cache[(nm, h, ser)]
                D['FNORM'] = xb_s / sig[j] > th
                P['FNORM'] = D['FNORM'].astype(float)
                # score on labelled half-B frames
                il, stl = sub_starts(lab, stB)
                yl = yB[il]
                vl = vB[il]
                out = {'split': nm, 'dir': h, 'neuron': j, 'label_free': info, 'readouts': {}}
                for r_, Pr_ in P.items():
                    pl = Pr_[il]
                    rt, ap, au, f1m, th_, pi = rank_metrics(pl.astype(np.float64), yl)
                    m = {'Rt': rt, 'nAP': (ap - pi) / (1 - pi), 'AUROC': au, 'VR_meanP': video_stats(pl, yl, vl, nvid),
                         'base_rate': pi}
                    if r_ != 'CONT':
                        m.update(calib(np.clip(pl, 0, 1), yl))
                        m['top_saturated'] = float((np.sort(pl)[::-1][:max(int(yl.sum()), 1)] >= 0.999).mean())
                    out['readouts'][r_] = m
                for r_, Dr in D.items():
                    dl = Dr[il]
                    tp = float((dl & yl).sum())
                    prec = tp / max(dl.sum(), 1)
                    reca = tp / max(yl.sum(), 1)
                    m = out['readouts'].setdefault(r_, {})
                    m.update({'prec': prec, 'rec': reca, 'F1': 2 * tp / max(dl.sum() + yl.sum(), 1),
                              'on_share': float(dl.mean()),
                              'BF1': bout_f1(dl, yl, stl, 0.25)[0], 'BF1_10': bout_f1(dl, yl, stl, 0.1)[0],
                              'BF1_50': bout_f1(dl, yl, stl, 0.5)[0]})
                    m['video'] = summary_corr(per_video_summaries(dl, yl, stl, P.get(r_.replace('_p05', ''), None)[il]
                                                                  if r_.replace('_p05', '') in P else None))
                # current continuous summaries (mean s per video) for reference
                out['readouts']['CONT']['video'] = summary_corr(per_video_summaries(D['Q95'][il], yl, stl, xb_s[il]))
                out['s_active_share'] = float((xb_s[il] > 0).mean())
                rec['splits'].append(out)
        res['cells'][cell] = rec
        log(f'{dname} {cell}: readouts done, {time.time() - t0:.0f}s')
    return res, audit


def cmd_run(args):
    dom, tier = args.dom, args.tier
    names = ['atlas'] + [f'rnd{s}' for s in range(10 if tier == 'A' else 5)]
    anames = ['atlas'] + [f'rnd{s}' for s in range(5)]
    od = OUT / dom / f'tier{tier}'
    od.mkdir(parents=True, exist_ok=True)
    for dname in args.dict.split(','):
        if (od / f'{dname}.json').exists() and not args.force:
            log(f'skip {dname} (exists)')
            continue
        res, audit = run_dict(dom, tier, dname, names, anames, do_readouts=not args.audit_only)
        np.savez_compressed(od / f'{dname}_audit.npz', **{
            f'{c}__{k}': v for c, a in audit.items() for k, v in a.items() if k in ('elig', 'full', 'full_thr', 'rate')},
            **{f'{c}__per__{p}': v for c, a in audit.items() for p, v in a['per'].items()})
        res['ds'] = {c: a['ds'] for c, a in audit.items()}
        (od / f'{dname}.json').write_text(json.dumps(res, default=float))
        log(f'saved {od / dname}')


# ---------------------------------------------------------------------------------------------- prevalence
def cmd_prevalence(args):
    """Tier A: the R~-selected neuron (all frames) of every cell, resampled to fixed base rates."""
    dom = args.dom
    F = frame_table(dom, 'A')
    starts = seq_starts(F)
    out = {}
    rng = np.random.default_rng(0)
    for dname in TIERA[dom]:
        C = np.load(BEE / dom / 'tierA' / f'{dname}.npy', mmap_mode='r')
        z = np.load(OUT / dom / 'tierA' / f'{dname}_audit.npz')
        for ser, lb in CELLS[dom]:
            cell = f'{ser}_{lb}'
            el, full = z[f'{cell}__elig'], z[f'{cell}__full']
            j = int(el[int(np.nanargmax(full[:, 0]))])
            idx, _ = sub_starts(F[f'v_{ser}_{lb}'].values, starts)
            x = smooth_seq(np.asarray(C[:, j:j + 1], np.float32), starts)[idx, 0].astype(np.float64)
            y = F[f'y_{lb}'].values[idx]
            pos, neg = np.flatnonzero(y), np.flatnonzero(~y)
            rec = {'neuron': j, 'base_rate': float(y.mean())}
            for pi in (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5):
                vals = []
                for _ in range(20):
                    if pi <= y.mean():
                        npos = int(round(pi / (1 - pi) * len(neg)))
                        sel = np.r_[rng.choice(pos, npos, replace=False), neg]
                    else:
                        nneg = int(round((1 - pi) / pi * len(pos)))
                        sel = np.r_[pos, rng.choice(neg, nneg, replace=False)]
                    rt, ap, au, f1, th, p_ = rank_metrics(x[sel], y[sel])
                    vals.append((rt, (ap - p_) / (1 - p_), ap, au, f1))
                rec[str(pi)] = dict(zip(('Rt', 'nAP', 'AP', 'AUROC', 'F1max'), np.mean(vals, 0).tolist()))
            out[f'{dname}|{cell}'] = rec
            log(f'prevalence {dname} {cell}: ' + json.dumps({k: (v if not isinstance(v, dict) else
                                                                  {a: round(b, 3) for a, b in v.items()})
                                                              for k, v in rec.items()}))
    (OUT / dom / 'prevalence.json').write_text(json.dumps(out, indent=1))


# ---------------------------------------------------------------------------------------------- report
def arm_of(d):
    if d.startswith('v1_s0') or d == 'v1':
        return 'v1_dep'
    if d.startswith('v1_s'):
        return 'v1_retrain'
    return d.rsplit('_s', 1)[0]


def cmd_report(args):
    R = {}
    for dom in ('mice', 'ants'):
        for tier in ('A', 'B'):
            od = OUT / dom / f'tier{tier}'
            for f in sorted(od.glob('*.json')) if od.exists() else []:
                R[(dom, tier, f.stem)] = json.loads(f.read_text())
    L = ['# ECI v2 probability readout and metric audit (scripts/eci/v2_prob/analysis.py; pre-registered rules)', '']
    dec = {}
    # ---------------- Q1: readouts
    RO = ('CONT', 'HMM2', 'HMM2_p05', 'HMM3', 'MIX', 'Q95', 'FNORM')
    rows = []
    for (dom, tier, d), r in R.items():
        for cell, c in r['cells'].items():
            for sp in c['splits']:
                for ro, m in sp['readouts'].items():
                    v = m.get('video', {})
                    rows.append(dict(dom=dom, tier=tier, dict=d, arm=arm_of(d), cell=f'{dom} {cell}', readout=ro,
                                     split=sp['split'], dir=sp['dir'], neuron=sp['neuron'], base=m.get('base_rate'),
                                     Rt=m.get('Rt'), nAP=m.get('nAP'), AUROC=m.get('AUROC'), F1=m.get('F1'),
                                     prec=m.get('prec'), rec=m.get('rec'), on_share=m.get('on_share'),
                                     BF1=m.get('BF1'), BF1_10=m.get('BF1_10'), BF1_50=m.get('BF1_50'),
                                     brier=m.get('brier'), bss=m.get('bss'), ece=m.get('ece'),
                                     meanP_over_pi=m.get('meanP_over_pi'), top_sat=m.get('top_saturated'),
                                     VR=v.get('r_meanP_fraction', m.get('VR_meanP')),
                                     r_frac=v.get('r_fraction'), r_bouts=v.get('r_bout_count'),
                                     r_rate=v.get('r_bout_rate'), rho_lat=v.get('rho_latency'),
                                     pred_bouts=v.get('mean_pred_bouts'), true_bouts=v.get('mean_true_bouts'),
                                     s_active=sp.get('s_active_share'),
                                     hmm2_stat_on=sp['label_free']['HMM2']['stationary_on'],
                                     hmm2_bout=sp['label_free']['HMM2']['mean_on_bout_frames'],
                                     hmm3_stat_on=sp['label_free']['HMM3']['stationary_on'],
                                     hmm3_bout=sp['label_free']['HMM3']['mean_on_bout_frames']))
    D = pd.DataFrame(rows)
    if len(D):
        D.to_parquet(OUT / 'readouts_long.parquet')
        num = [c for c in D.columns if c not in ('dom', 'tier', 'dict', 'arm', 'cell', 'readout', 'split', 'dir')]
        D[num] = D[num].astype(float)
        # per (tier, arm, cell, readout): mean over splits, directions and seeds
        G = D.groupby(['tier', 'arm', 'cell', 'readout'])[num].mean().reset_index()
        G = G[~((G.tier == 'B') & G.arm.isin(['v1_dep']))]  # Tier B v1 = mean of s0-10 below
        Bv1 = D[(D.tier == 'B') & D.arm.isin(['v1_dep', 'v1_retrain'])].groupby(['cell', 'readout'])[num].mean() \
            .reset_index().assign(tier='B', arm='v1_all')
        G = pd.concat([G[~((G.tier == 'B') & (G.arm == 'v1_retrain'))], Bv1], ignore_index=True)
        G.to_csv(OUT / 'readouts_cells.csv', index=False)
        cont = G[G.readout == 'CONT'].set_index(['tier', 'arm', 'cell'])['nAP']
        G['dnAP'] = G.apply(lambda r: r.nAP - cont.get((r.tier, r.arm, r.cell), np.nan), axis=1)
        cand = ('HMM2', 'HMM3', 'MIX', 'Q95', 'FNORM')
        S_ = G[G.readout.isin(cand)].groupby('readout')[['dnAP', 'F1', 'BF1', 'VR', 'bss', 'ece']].mean()
        S_['n_cells'] = G[G.readout.isin(cand)].groupby('readout').size()
        S_['pass_i'] = S_.dnAP >= -0.03
        ok = S_[S_.pass_i].sort_values(['F1', 'BF1'], ascending=False)
        win = None
        if len(ok):
            win = ok.index[0]
            if len(ok) > 1 and ok.F1.iloc[0] - ok.F1.iloc[1] < 0.005 and ok.BF1.iloc[1] > ok.BF1.iloc[0]:
                win = ok.index[1]
        dec['readout'] = {'summary': S_.reset_index().to_dict('records'), 'recommended': win}
        L += ['## Q1. Readout decision (30 cells, mean over splits / directions / seeds)', '',
              '| readout | n cells | nAP(P) - nAP(cont) | frame F1 | bout F1 @0.25 | video r (mean P vs annotated fraction) | Brier skill | ECE | rule (i) |',
              '|---|---|---|---|---|---|---|---|---|']
        for ro, r in S_.iterrows():
            L.append(f'| {ro} | {int(r.n_cells)} | {r.dnAP:+.3f} | {r.F1:.3f} | {r.BF1:.3f} | {r.VR:.3f} | {r.bss:+.3f} | '
                     f'{r.ece:.3f} | {"pass" if r.pass_i else "FAIL"} |')
        L += ['', f'RECOMMENDED (rule ii): **{win}**', '']
        cols = ['Rt', 'nAP', 'F1', 'prec', 'rec', 'on_share', 'base', 'BF1', 'bss', 'ece', 'meanP_over_pi', 'VR',
                'r_bouts', 'r_rate', 'rho_lat']
        L += ['## Q1. Per cell', '', '| tier | arm | cell | readout | ' + ' | '.join(cols) + ' |',
              '|' + '---|' * (4 + len(cols))]
        for _, r in G.sort_values(['cell', 'tier', 'arm', 'readout']).iterrows():
            if r.readout not in RO:
                continue
            L.append(f'| {r.tier} | {r.arm} | {r.cell} | {r.readout} | ' +
                     ' | '.join('' if not np.isfinite(r[c]) else f'{r[c]:.3f}' for c in cols) + ' |')
        # diagnosis (Tier A MIX)
        L += ['', '## Q1. Diagnosis of the mixture (MIX) readout, Tier A', '',
              '| arm | cell | base rate | s > 0 share | MIX on share | Rt cont | Rt MIX (float64) | Rt MIX (fp16) | top-N with P >= 0.999 | HMM2 on share | HMM3 on share | HMM2 on-bout (frames) | HMM3 on-bout |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
        A = G[G.tier == 'A'].set_index(['arm', 'cell', 'readout'])
        for (arm, cell) in sorted({(a, c) for a, c, _ in A.index}):
            g = lambda ro, c: A.loc[(arm, cell, ro), c] if (arm, cell, ro) in A.index else np.nan  # noqa: E731
            L.append(f'| {arm} | {cell} | {g("MIX", "base"):.4f} | {g("MIX", "s_active"):.3f} | {g("MIX", "on_share"):.3f} | '
                     f'{g("CONT", "Rt"):.3f} | {g("MIX", "Rt"):.3f} | {g("MIX_fp16", "Rt"):.3f} | {g("MIX", "top_sat"):.3f} | '
                     f'{g("HMM2", "on_share"):.3f} | {g("HMM3", "on_share"):.3f} | {g("MIX", "hmm2_bout"):.1f} | {g("MIX", "hmm3_bout"):.1f} |')
        # calibration verdict per cell for the winner
        if win:
            W_ = G[G.readout == win]
            W_ = W_.assign(calibrated=(W_.bss > 0) & (W_.ece < 0.05))
            dec['readout']['calibrated_cells'] = W_[['tier', 'arm', 'cell', 'bss', 'ece', 'calibrated']].to_dict('records')
            L += ['', f'Calibration of the recommended readout ({win}) against the behaviour (BSS > 0 and ECE < 0.05): '
                  f'{int(W_.calibrated.sum())} of {len(W_)} cells', '']
    # ---------------- Q2: metric audit
    A_rows, ds_rows, rel_rows = [], [], []
    from scipy.stats import spearmanr
    for (dom, tier, d), r in R.items():
        f = OUT / dom / f'tier{tier}' / f'{d}_audit.npz'
        if not f.exists():
            continue
        z = np.load(f)
        for ser, lb in CELLS[dom]:
            cell = f'{ser}_{lb}'
            if f'{cell}__full' not in z:
                continue
            M = z[f'{cell}__full']
            ok = np.isfinite(M).all(1)
            M = M[ok]
            top = np.zeros(len(M), bool)
            for mi in range(6):
                top[np.argsort(-M[:, mi])[:50]] = True
            for setname, X in (('all', M), ('top', M[top])):
                rho = spearmanr(X).correlation if len(X) > 3 else np.full((6, 6), np.nan)
                A_rows.append(dict(dom=dom, tier=tier, dict=d, cell=f'{dom} {cell}', set=setname, n=len(X),
                                   rho=np.asarray(rho).tolist(), best=[int(np.argmax(M[:, mi])) for mi in range(6)]))
            for k_ in [k for k in z.files if k.startswith(f'{cell}__per__') and k.endswith('_0')]:
                a0, a1 = z[k_], z[k_[:-1] + '1']
                for mi, mname in enumerate(METRICS):
                    rel_rows.append(dict(dom=dom, tier=tier, dict=d, cell=f'{dom} {cell}', metric=mname,
                                         rel=spearman(a0[:, mi], a1[:, mi])))
            for mname, lst in r.get('ds', {}).get(cell, {}).items():
                for e in lst:
                    ds_rows.append(dict(dom=dom, tier=tier, dict=d, arm=arm_of(d), cell=f'{dom} {cell}', metric=mname,
                                        cf=e['cf_value'], DS=e['DS'], vr=e['vr'], f1=e['f1_transfer'],
                                        bf1=e['bf1_transfer'], split=e['split'], dir=e['dir'], neuron=e['neuron'],
                                        **{f'cf_{m2}': e['cf_all'][i] for i, m2 in enumerate(METRICS)}))
    if A_rows:
        AR = pd.DataFrame(A_rows)
        RL = pd.DataFrame(rel_rows)
        DSr = pd.DataFrame(ds_rows)
        DSr.to_parquet(OUT / 'audit_ds_long.parquet')
        L += ['', '## Q2. Metric audit', '']
        for setname in ('all', 'top'):
            for dom in ('mice', 'ants'):
                sub = AR[(AR.set == setname) & (AR.dom == dom) & (AR.tier == 'B')]
                if not len(sub):
                    continue
                med = np.nanmedian(np.stack([np.asarray(x, float) for x in sub.rho]), 0)
                L += [f'A1 neuron-level Spearman, {dom}, Tier B, {setname} eligible neurons '
                      f'(median over {len(sub)} dictionary x cell)', '', '| | ' + ' | '.join(METRICS) + ' |',
                      '|' + '---|' * 7]
                for i, m1 in enumerate(METRICS):
                    L.append(f'| {m1} | ' + ' | '.join(f'{med[i, j]:.2f}' for j in range(6)) + ' |')
                L.append('')
                if setname == 'all':
                    bests = np.stack(sub.best.values)
                    same = np.array([[np.mean(bests[:, i] == bests[:, j]) for j in range(6)] for i in range(6)])
                    L += [f'A2 same best neuron, {dom}, Tier B (share of dictionary x cell)', '',
                          '| | ' + ' | '.join(METRICS) + ' |', '|' + '---|' * 7]
                    for i, m1 in enumerate(METRICS):
                        L.append(f'| {m1} | ' + ' | '.join(f'{same[i, j]:.2f}' for j in range(6)) + ' |')
                    L.append('')
        # A3: representation-level, cross-fitted values per dictionary
        B = DSr[DSr.tier == 'B']
        if len(B):
            cf = B.groupby(['dom', 'cell', 'dict', 'arm', 'metric'])['cf'].mean().unstack('metric')
            L += ['A3 representation-level Spearman of cross-fitted metric values over the Tier B dictionaries '
                  '(mean over cells)', '', '| | ' + ' | '.join(METRICS) + ' |', '|' + '---|' * 7]
            mats = []
            for (dom, cell), g in cf.groupby(level=['dom', 'cell']):
                mats.append(spearmanr(g[list(METRICS)].values).correlation)
            mm = np.nanmean(np.stack(mats), 0)
            for i, m1 in enumerate(METRICS):
                L.append(f'| {m1} | ' + ' | '.join(f'{mm[i, j]:.2f}' for j in range(6)) + ' |')
            L += ['', 'A3 arm means of the cross-fitted value (Tier B)', '',
                  '| cell | arm | n dicts | ' + ' | '.join(METRICS) + ' |', '|' + '---|' * 9]
            am = cf.groupby(level=['dom', 'cell', 'arm']).agg(['mean', 'std', 'count'])
            for (dom, cell, arm), r in am.iterrows():
                L.append(f'| {cell} | {arm} | {int(r[(METRICS[0], "count")])} | ' +
                         ' | '.join(f'{r[(m, "mean")]:.3f} ± {r[(m, "std")]:.3f}' if np.isfinite(r[(m, 'std')])
                                    else f'{r[(m, "mean")]:.3f}' for m in METRICS) + ' |')
            # arm ranking agreement + eta2
            eta = {}
            for m in METRICS:
                vals = []
                for (dom, cell), g in cf.groupby(level=['dom', 'cell']):
                    g = g.reset_index()
                    g = g[g.arm.isin(['v1_dep', 'v1_retrain', 'V2F_M0', 'V2F_N2'])].copy()
                    g['grp'] = g.arm.replace({'v1_dep': 'v1', 'v1_retrain': 'v1'})
                    x = g[m].values
                    ssb = sum(len(gg) * (gg[m].mean() - x.mean()) ** 2 for _, gg in g.groupby('grp'))
                    sst = ((x - x.mean()) ** 2).sum()
                    vals.append(ssb / sst if sst > 0 else np.nan)
                eta[m] = float(np.nanmean(vals))
            # A4 reliability
            rel = RL[RL.tier == 'B'].groupby('metric').rel.median()
            V = B.groupby('metric')[['DS', 'vr', 'f1', 'bf1']].mean()
            V['reliability'] = rel
            V['eta2_arms'] = pd.Series(eta)
            V = V.reindex(list(METRICS))
            V.to_csv(OUT / 'metric_summary.csv')
            order = V.sort_values('DS', ascending=False)
            prim = order.index[0]
            if order.DS.iloc[0] - order.DS.iloc[1] < 0.01 and order.reliability.iloc[1] > order.reliability.iloc[0]:
                prim = order.index[1]
            rest = [m for m in order.index if m != prim][:2]
            if 'AUROC' not in rest and prim != 'AUROC':
                rest.append('AUROC')
            dec['metric'] = {'V': V.reset_index().to_dict('records'), 'primary': prim, 'secondary': rest}
            L += ['', 'A4 / A5 metric summary (Tier B): DS = mean of held-out video r, transferred-threshold frame F1 and '
                  'bout F1 of the neuron each metric selects', '',
                  '| metric | V(M) = mean DS | video r | frame F1 | bout F1 | split-half reliability | arm eta^2 |',
                  '|---|---|---|---|---|---|---|']
            for m, r in V.iterrows():
                L.append(f'| {m} | {r.DS:.3f} | {r.vr:.3f} | {r.f1:.3f} | {r.bf1:.3f} | {r.reliability:.3f} | '
                         f'{r.eta2_arms:.3f} |')
            L += ['', f'PRIMARY metric (rule): **{prim}**; secondary: {rest}', '']
            Vc = B.groupby(['cell', 'metric'])['DS'].mean().unstack('metric')[list(METRICS)]
            L += ['A5 per cell (mean DS)', '', '| cell | ' + ' | '.join(METRICS) + ' |', '|' + '---|' * 7]
            for c, r in Vc.iterrows():
                L.append(f'| {c} | ' + ' | '.join(f'{r[m]:.3f}' for m in METRICS) + ' |')
    for dom in ('mice', 'ants'):
        f = OUT / dom / 'prevalence.json'
        if f.exists():
            P_ = json.loads(f.read_text())
            L += ['', f'Prevalence test ({dom}, Tier A, R~-selected neuron, 20 resamples)', '',
                  '| dict / cell | true base | target base | Rt | nAP | AP | AUROC | F1max |', '|---|---|---|---|---|---|---|---|']
            for k, rec in P_.items():
                for pi in ('0.01', '0.02', '0.05', '0.1', '0.2', '0.3', '0.5'):
                    v = rec[pi]
                    L.append(f'| {k} | {rec["base_rate"]:.3f} | {pi} | {v["Rt"]:.3f} | {v["nAP"]:.3f} | {v["AP"]:.3f} | '
                             f'{v["AUROC"]:.3f} | {v["F1max"]:.3f} |')
    (OUT / 'decision.json').write_text(json.dumps(dec, indent=1, default=float))
    (OUT / 'table.md').write_text('\n'.join(L))
    print('\n'.join(L[:60]))


def cmd_selftest(args):
    """metric kernels vs sklearn / score.py definitions on random data."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    rng = np.random.default_rng(0)
    for it in range(5):
        n = 5000
        y = rng.random(n) < 0.05
        s = np.where(rng.random(n) < 0.5, 0, rng.gamma(1, 1, n) + 1.0 * y)
        s = np.round(s, 1)
        rt, ap, au, f1, th, pi = rank_metrics(s, y)
        assert abs(ap - average_precision_score(y, s)) < 1e-9, (ap, average_precision_score(y, s))
        assert abs(au - roc_auc_score(y, s)) < 1e-9
        # R~ pro rata by brute force over random tie-breaking
        P = y.sum()
        ex = np.mean([y[np.lexsort((rng.random(n), -s))][:P].mean() for _ in range(400)])
        assert abs((ex - pi) / (1 - pi) - rt) < 0.02, (rt, (ex - pi) / (1 - pi))
        f1s = [2 * (y & (s >= t)).sum() / ((s >= t).sum() + P) for t in np.unique(s)]
        assert abs(max(f1s) - f1) < 1e-9
    y = np.array([0, 1, 1, 1, 0, 0, 1, 1, 0, 1], bool)
    p = np.array([0, 1, 1, 0, 0, 1, 1, 1, 1, 0], bool)
    st = np.array([0, 5, 10])
    f, npred, ntrue, tp = bout_f1(p, y, st, 0.25)
    assert (npred, ntrue) == (2, 3) and tp == 2, (npred, ntrue, tp)
    # the parallel kernel = the single-neuron functions
    n, m = 3000, 7
    X = np.asfortranarray(np.where(rng.random((n, m)) < 0.6, 0, rng.gamma(1, 1, (n, m))).astype(np.float32))
    y = rng.random(n) < 0.1
    X[:, 0] += 2 * y
    rows = np.flatnonzero(rng.random(n) < 0.7).astype(np.int64)
    vv = (rows // 300).astype(np.int64)
    st = np.r_[0, np.flatnonzero(np.diff(vv)) + 1, len(rows)].astype(np.int64)
    out, thr = metrics_cols(X, rows, y[rows], st, vv, int(vv.max()) + 1)
    for q in range(m):
        x = X[rows, q].astype(np.float64)
        rt, ap, au, f1, th, pi = rank_metrics(x, y[rows])
        ref = (rt, (ap - pi) / (1 - pi), au, f1, bout_f1(x >= th, y[rows], st, 0.25)[0],
               video_stats(x, y[rows], vv, int(vv.max()) + 1))
        assert np.allclose(out[q], ref, atol=1e-9, equal_nan=True), (q, out[q], ref)
    log('selftest passed')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('step')
    p.add_argument('--dom', default='')
    p.add_argument('--tier', default='A')
    p.add_argument('--dict', default='')
    p.add_argument('--audit_only', action='store_true')
    p.add_argument('--force', action='store_true')
    a = p.parse_args()
    {'tierA_extract': cmd_tierA_extract, 'run': cmd_run, 'selftest': cmd_selftest, 'prevalence': cmd_prevalence,
     'report': cmd_report}[a.step](a)


if __name__ == '__main__':
    main()
