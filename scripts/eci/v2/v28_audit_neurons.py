"""
ECI v2.8 audit (analysis only, no training): per-DICTIONARY neuron statistics for the stored v2.5-v2.7 SAEs.
POST HOC: written 2026-11-24 after all v2.6 / v2.7 results were seen. Nothing here is pre-registered.

mice (edge-excluded ext frames, 863,424; selection = annotator SG pools, held-out = 12 non-SG pools, 72 videos):
  every neuron: AUROC (P(on) readout, = the selection statistic) and average precision (AP; P(on) ranked, raw code as
  tie-break) on SG and non-SG frames for nose-nose and nose-tail, discrimination AUROC (nose-tail vs nose-nose), eligibility.
  picks under rules auc / dist / dist2 (v25_eval.select, recomputed and checked against the stored v2.7 tags when present).
  candidates (top-20 per behaviour by SG AUROC and by SG AP, the picks, the held-out-best neuron): precision@top-k/video
  (k = 3, 10, 30; raw-code ranking, earliest frame first, valid frames, videos with >= 1 positive = the official
  precision@top3/video for k = 3) on SG and non-SG frames.
  picks: bootstrap (1,000) over held-out VIDEOS and over held-out POOLS of AUROC, AP, precision@top-k/video.
ants (v3 + v4, 1 fps, halves A / B): grooming-to-focal (any) pick per direction (best A AUROC -> scored on B, and B -> A),
  candidates as above, held-out AUROC vs grooming-to-focal (any) / single / double, precision@top3/video, video bootstrap.
usage: v28_audit_neurons.py mice --key K | ants --key K   (CPU job; outputs results/vision/eci_v2_8/audit/work/)
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit, prange

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v27_eval  # noqa: E402,F401  (patches v25_eval / v22_core to the v2.7 edge-excluded mice table)
EV = v27_eval.EV
C, E = EV.C, EV.E
import v23_ants as AN  # noqa: E402

REPO = C.REPO
OUT = REPO / 'results/vision/eci_v2_8/audit'
WORK = OUT / 'work'
BEE = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes')
NB = 1000
KS = (3, 10, 30)
NCAND = 20


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def mice_paths(key):
    arm, s = key.rsplit('_s', 1)
    if arm in ('N0wc', 'SMOOTH', 'RW', 'N0x'):
        d = BEE / 'v2_6' / 'mice' / 'cache'
    else:
        d = BEE / 'v2_7' / 'mice' / 'cache'
    return d / f'{key}_raw.npy', d / f'{key}_pon.npy'


def ants_paths(key):
    arm, s = key.rsplit('_s', 1)
    if arm in ('N0wc', 'SMOOTH', 'RW', 'N0x', 'PRED'):
        d = BEE / 'v2_6' / 'ants' / 'cache'
    elif arm in ('PREDc', 'PREDd', 'PREDk', 'PREDh'):
        d = BEE / 'v2_7' / 'ants' / 'cache'
    else:
        d = BEE / 'v2_3' / 'ants' / 'cache'
    return d / f'{key}_raw.npy', d / f'{key}_pon.npy'


# ---------------------------------------------------------------------------------------------- kernels
@njit(parallel=True, cache=True)
def _ap_multi(St, cat):
    """St (W, n) float64 scores, cat (n, Q) int8 (0 out, 1 negative, 2 positive) -> (Q, W) average precision (step-wise,
    frames sorted by descending score; ties in the sort order, not averaged)."""
    W, n = St.shape
    Q = cat.shape[1]
    out = np.full((Q, W), np.nan)
    for j in prange(W):
        o = np.argsort(-St[j])
        tp = np.zeros(Q)
        cnt = np.zeros(Q)
        acc = np.zeros(Q)
        for i in range(n):
            r = o[i]
            for q in range(Q):
                c = cat[r, q]
                if c > 0:
                    cnt[q] += 1
                    if c == 2:
                        tp[q] += 1
                        acc[q] += tp[q] / cnt[q]
        for q in range(Q):
            if tp[q] > 0:
                out[q, j] = acc[q] / tp[q]
    return out


def score_of(X, R, j):
    """the official ranking: P(on) (X) with the raw code as tie-break (v22_core.neuron_stats p100)."""
    x = np.asarray(X[:, j], np.float64)
    r = np.asarray(R[:, j], np.float64)
    return x + 1e-6 * r / max(r.max(), 1e-12)


def block_stats(X, R, cats, nblk=256):
    """AUROC and AP of every column for the query columns cats -> (A (Q, W), AP (Q, W))."""
    W = X.shape[1]
    A = np.full((cats.shape[1], W), np.nan)
    P = np.full((cats.shape[1], W), np.nan)
    rmax = np.zeros(W)
    for c0 in range(0, len(R), 50000):
        rmax = np.maximum(rmax, np.asarray(R[c0:c0 + 50000], np.float32).max(0))
    for c0 in range(0, W, nblk):
        c1 = min(W, c0 + nblk)
        Xt = np.ascontiguousarray(X[:, c0:c1].T, dtype=np.float32)
        A[:, c0:c1] = C._auc_multi(Xt, cats)
        St = Xt.astype(np.float64) + 1e-6 * np.ascontiguousarray(R[:, c0:c1].T, dtype=np.float64) / np.maximum(rmax[c0:c1, None], 1e-12)
        P[:, c0:c1] = _ap_multi(St, cats)
        del Xt, St
    return A, P


def topk_video(obs, y, valid, raw, k):
    """per video (contiguous rows): hits in the top-k valid frames by raw code (earliest first) and positives."""
    codes, vids = pd.factorize(obs)
    st = np.r_[np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]]), len(codes)]
    assert len(st) - 1 == len(vids)
    hits, pos = np.zeros(len(vids)), np.zeros(len(vids))
    for i, (a, b) in enumerate(zip(st[:-1], st[1:])):
        ii = np.flatnonzero(valid[a:b])
        yy, r = y[a:b][ii], raw[a:b][ii].astype(np.float64)
        o = np.lexsort((np.arange(len(ii)), -r))[:k]
        hits[i], pos[i] = yy[o].sum(), yy.sum()
    return np.asarray(vids), hits, pos


def topk_point(hits, pos, k):
    e = pos > 0
    return float(hits[e].sum() / (k * e.sum())) if e.any() else float('nan')


def video_weights(nv, rng, groups=None):
    """(NB, nv) bootstrap counts; groups = per-video group index (pool bootstrap: resample groups, a video gets its group's count)."""
    if groups is None:
        return np.stack([np.bincount(rng.integers(0, nv, nv), minlength=nv) for _ in range(NB)]).astype(float)
    ng = groups.max() + 1
    Wg = np.stack([np.bincount(rng.integers(0, ng, ng), minlength=ng) for _ in range(NB)]).astype(float)
    return Wg[:, groups]


def boot_metrics(score, raw, y, valid, obs, vcode, Wv, k_list):
    """bootstrap replicates (rows of Wv = per-video weights) of AUROC, AP and precision@top-k/video for one neuron."""
    nv = Wv.shape[1]
    A, Pp, Nn = E.uu_auc(score, y, vcode, nv)
    auc = np.einsum('ru,uv,rv->r', Wv, A, Wv) / ((Wv @ Pp) * (Wv @ Nn))
    o = np.argsort(-score, kind='stable')
    yo, vo = y[o], vcode[o]
    ap = np.empty(len(Wv))
    for r in range(len(Wv)):
        w = Wv[r][vo]
        tp = np.cumsum(w * yo)
        cn = np.cumsum(w)
        m = yo & (w > 0)
        ap[r] = (w[m] * tp[m] / cn[m]).sum() / max(w[m].sum(), 1e-12)
    out = {'auc': auc, 'ap': ap}
    for k in k_list:
        _, h, p = topk_video(obs, y, valid, raw, k)
        e = p > 0
        out[f'top{k}'] = (Wv[:, e] @ h[e]) / (k * Wv[:, e].sum(1))
    return out


# ---------------------------------------------------------------------------------------------- mice
def cmd_mice(a):
    t0 = time.time()
    key = a.key
    F, _ = C.mice_table()
    pr, pp = mice_paths(key)
    R = np.load(pr)
    X = R.astype(np.float32) if key.startswith('v1_') else np.load(pp)
    log(f'{key}: loaded {X.shape} ({time.time() - t0:.0f}s)')
    n = len(F)
    assert len(X) == n
    el = np.zeros(X.shape[1])
    for c0 in range(0, n, 50000):
        el += (X[c0:c0 + 50000] > 0).sum(0)
    el = el / n >= C.ELIG
    sg, non = F.sg.values, F.nonsg.values
    cls = F.cls.values
    Y = {b: F[f'y_{b}'].values for b in C.BEH}
    cols, names = [], []
    for nm, m in (('sg', sg), ('non', non)):
        for b in C.BEH:
            cols.append(np.where(m, np.where(Y[b], 2, 1), 0)); names.append(f'{nm}|{b}')
        ex = m & ((cls == 1) | (cls == 2))
        cols.append(np.where(ex, np.where(cls == 1, 2, 1), 0)); names.append(f'{nm}|disc')
    cats = np.ascontiguousarray(np.stack(cols, 1).astype(np.int8))
    A, P = block_stats(X, R, cats)
    S = {f'auc|{nm}': A[i] for i, nm in enumerate(names)}
    S.update({f'ap|{nm}': P[i] for i, nm in enumerate(names) if not nm.endswith('disc')})
    log(f'{key}: all-neuron AUROC / AP ({time.time() - t0:.0f}s)')
    st = {'fire': np.where(el, 1.0, 0.0), 'auc_nose_nose': S['auc|sg|nose_nose'], 'auc_nt': S['auc|sg|nt'], 'disc': S['auc|sg|disc']}
    picks = {r: EV.select(st, r) for r in ('auc', 'dist', 'dist2')}
    tp = REPO / 'results/vision/eci_v2_7/mice/tags' / f'{key}.pkl'
    check = None
    if tp.exists():
        T = pickle.load(open(tp, 'rb'))
        check = {r: T['sets'][f'sg2non|{r}'][0]['picks'] == picks[r] for r in picks}
        log(f'{key}: picks identical to stored v2.7 tag: {check}')
    cand = {}
    for b in C.BEH:
        sa = np.where(el, S[f'auc|sg|{b}'], -np.inf)
        sp = np.where(el, S[f'ap|sg|{b}'], -np.inf)
        ho = np.where(el, S[f'auc|non|{b}'], -np.inf)
        c = set(np.argsort(-sa)[:NCAND].tolist()) | set(np.argsort(-sp)[:NCAND].tolist()) | {int(np.argmax(ho))}
        c |= {p[b] for p in picks.values()} | {p[bb] for p in picks.values() for bb in C.BEH}
        cand[b] = sorted(int(j) for j in c)
    js = sorted(set(cand['nose_nose']) | set(cand['nt']))
    rows = {'sg': np.flatnonzero(sg), 'non': np.flatnonzero(non)}
    topk = {}
    for nm, rr in rows.items():
        obs = F.obs.values[rr]
        for b in C.BEH:
            y, v = Y[b][rr], F[f'v_{b}'].values[rr]
            for j in js:
                raw = np.asarray(R[rr, j], np.float32)
                for k in KS:
                    _, h, p = topk_video(obs, y, v, raw, k)
                    topk[(nm, b, j, k)] = topk_point(h, p, k)
    log(f'{key}: top-k of {len(js)} candidates ({time.time() - t0:.0f}s)')
    # bootstrap of the picks (held-out): videos and pools
    rr = rows['non']
    obs = F.obs.values[rr]
    vcode, vids = pd.factorize(obs)
    units = F.unit.values[rr]
    vunit = pd.Series(units).groupby(vcode).first().values
    gcode = pd.factorize(vunit)[0]
    rng = np.random.default_rng(20261124)
    Wvid = video_weights(len(vids), rng)
    Wpool = video_weights(len(vids), rng, gcode)
    boot = {}
    for r, pk in picks.items():
        for b in C.BEH:
            j = pk[b]
            if (j, b) in boot:
                continue
            sc = score_of(X, R, j)[rr]
            raw = np.asarray(R[rr, j], np.float32)
            y, v = Y[b][rr], F[f'v_{b}'].values[rr]
            boot[(j, b)] = {'video': boot_metrics(sc, raw, y, v, obs, vcode, Wvid, KS), 'pool': boot_metrics(sc, raw, y, v, obs, vcode, Wpool, KS)}
    log(f'{key}: bootstrap ({time.time() - t0:.0f}s)')
    res = {'key': key, 'W': X.shape[1], 'eligible': el, 'stats': S, 'picks': picks, 'picks_match_tag': check, 'cand': cand,
           'topk': topk, 'boot': boot, 'n_sg_rows': int(sg.sum()), 'n_non_rows': int(non.sum()), 'n_non_videos': len(vids),
           'n_non_pools': int(gcode.max() + 1)}
    WORK.mkdir(parents=True, exist_ok=True)
    pickle.dump(res, open(WORK / f'mice_{key}.pkl', 'wb'))
    log(f'{key}: done ({time.time() - t0:.0f}s)')


# ---------------------------------------------------------------------------------------------- ants
ABEH = ('groom_any', 'single', 'double')


def cmd_ants(a):
    t0 = time.time()
    key = a.key
    F = AN.table()
    pr, pp = ants_paths(key)
    X, R = np.load(pp), np.load(pr)
    assert len(X) == len(F)
    el = (X > 0).mean(0) >= C.ELIG
    halves = {h: F.half.values == h for h in ('A', 'B')}
    Y = {b: F[f'y_{b}'].values for b in ABEH}
    cols, names = [], []
    for h, m in halves.items():
        for b in ABEH:
            cols.append(np.where(m, np.where(Y[b], 2, 1), 0)); names.append(f'{h}|{b}')
    cats = np.ascontiguousarray(np.stack(cols, 1).astype(np.int8))
    A, P = block_stats(X, R, cats)
    S = {f'auc|{nm}': A[i] for i, nm in enumerate(names)}
    S.update({f'ap|{nm}': P[i] for i, nm in enumerate(names)})
    res = {'key': key, 'W': X.shape[1], 'eligible': el, 'stats': S, 'dirs': {}}
    rng = np.random.default_rng(20261125)
    for dn, (s_, e_) in AN.DIRS.items():
        sa = np.where(el, S[f'auc|{s_}|groom_any'], -np.inf)
        pick = int(np.argmax(sa))
        tp = REPO / 'results/vision/eci_v2_7/ants/tags' / f'{key}.pkl'
        tp6 = REPO / 'results/vision/eci_v2_6/ants/tags' / f'{key}.pkl'
        stored = None
        for t in (tp, tp6):
            if t.exists():
                stored = pickle.load(open(t, 'rb'))['sets'][dn][0]['picks']['groom_any']
                break
        cand = sorted(set(np.argsort(-sa)[:NCAND].tolist()) | {pick} | {int(np.argmax(np.where(el, S[f'auc|{e_}|groom_any'], -np.inf)))})
        rr = np.flatnonzero(halves[e_])
        rs = np.flatnonzero(halves[s_])
        obs = F.obs.values[rr]
        top = {}
        for j in cand:
            for nm, rows_ in (('sel', rs), ('held', rr)):
                for b in ABEH:
                    _, h, p = topk_video(F.obs.values[rows_], Y[b][rows_], np.ones(len(rows_), bool), np.asarray(R[rows_, j], np.float32), 3)
                    top[(nm, b, j)] = topk_point(h, p, 3)
        vcode, vids = pd.factorize(obs)
        Wv = video_weights(len(vids), rng)
        sc = score_of(X, R, pick)[rr]
        boot = {b: boot_metrics(sc, np.asarray(R[rr, pick], np.float32), Y[b][rr], np.ones(len(rr), bool), obs, vcode, Wv, (3,)) for b in ABEH}
        res['dirs'][dn] = {'pick': pick, 'stored_pick': stored, 'cand': cand, 'top3': top, 'boot': boot, 'held_videos': len(vids)}
        log(f'ants {key} {dn}: pick {pick} (stored {stored})')
    WORK.mkdir(parents=True, exist_ok=True)
    pickle.dump(res, open(WORK / f'ants_{key}.pkl', 'wb'))
    log(f'ants {key}: done ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('dom', choices=('mice', 'ants'))
    p.add_argument('--key', required=True)
    a = p.parse_args()
    (cmd_mice if a.dom == 'mice' else cmd_ants)(a)
