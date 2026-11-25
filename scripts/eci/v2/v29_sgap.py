"""
ECI v2.9 per-dictionary scoring (pre-registration scripts/eci/v2/v29_prereg.md). Reuses the v2.8 audit kernels
(v28_audit_neurons: block_stats, topk_video, boot_metrics, score_of, video_weights).
mice --key K: all-neuron SG / held-out AUROC + AP (P(on) readout, raw tie-break) and raw firing rates; rule sgap (strict
  candidates = top-20 SG AUROC U top-20 SG AP among eligible, firing cap < 10% of all frames); the official picks of rules
  auc / dist / dist2 from the v2.9 tag (recomputed and checked); per pick: held-out AUROC / AP / precision@top3,10/video
  (+ chance) with a 1,000-resample video bootstrap (rng 20261125), raw firing rate (all / SG / held-out frames), Matryoshka
  level, per-video time bias / MAE (sum P(on) x 0.2 s - true time), top-3-frame overlap of each pair.
ants --key K: grooming-to-focal pick per direction (best selection-half AUROC of any) and held-out any / single / double
  (v28_audit_neurons.cmd_ants with the v2.9 cache paths).
-> results/vision/eci_v2_9/<dom>/dicts/K.json
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v28_audit_neurons as A  # noqa: E402  (imports v27_eval: the edge-excluded mice table, 863,424 frames)

C, EV = A.C, A.EV
REPO = C.REPO
OUT = REPO / 'results/vision/eci_v2_9'
BEE = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes')
PREF = (128, 256, 512, 2048, 4096)
FIRE_CAP, NCAND, DISC = 0.10, 20, 0.4
KS = (3, 10)
DT = 0.2
log = A.log


def paths(dom, key):
    d = BEE / 'v2_9' / dom / 'cache'
    return d / f'{key}_raw.npy', d / f'{key}_pon.npy'


def level(key, j):
    return None if key.startswith('v1_') else 1 + int(np.searchsorted(PREF, j, side='right'))


def sgap(S, el, fire, cap=True):
    """rule sgap on SG statistics only -> (picks or None, candidate sets)."""
    cand = {}
    for b in C.BEH:
        sa = np.where(el, S[f'auc|sg|{b}'], -np.inf)
        sp = np.where(el, np.nan_to_num(S[f'ap|sg|{b}'], nan=-np.inf), -np.inf)
        cand[b] = sorted(set(np.argsort(-sa)[:NCAND].tolist()) | set(np.argsort(-sp)[:NCAND].tolist()))
    ok = (lambda j: fire[j] < FIRE_CAP) if cap else (lambda j: True)
    cn = [j for j in cand['nose_nose'] if S['auc|sg|disc'][j] <= DISC and ok(j)]
    if not cn:
        return None, cand
    nn = cn[int(np.nanargmax([S['ap|sg|nose_nose'][j] for j in cn]))]
    ct = [j for j in cand['nt'] if j != nn and S['auc|sg|disc'][j] >= 1 - DISC and ok(j)]
    if not ct:
        return None, cand
    nt = ct[int(np.nanargmax([S['ap|sg|nt'][j] for j in ct]))]
    return {'nose_nose': int(nn), 'nt': int(nt)}, cand


def cmd_mice(a):
    t0 = time.time()
    key = a.key
    F, _ = C.mice_table()
    n = len(F)
    assert n == 863_424, n
    pr, pp = paths('mice', key)
    R = np.load(pr)
    isv1 = key.startswith('v1_')
    X = R.astype(np.float32) if isv1 else np.load(pp)
    assert len(X) == n and len(R) == n
    log(f'{key}: loaded {X.shape} ({time.time() - t0:.0f}s)')
    sg, non = F.sg.values, F.nonsg.values
    fire = {nm: np.zeros(X.shape[1]) for nm in ('all', 'sg', 'non')}
    elc = np.zeros(X.shape[1])
    for c0 in range(0, n, 50000):
        rb = np.asarray(R[c0:c0 + 50000]) > 0
        fire['all'] += rb.sum(0)
        fire['sg'] += rb[sg[c0:c0 + 50000]].sum(0)
        fire['non'] += rb[non[c0:c0 + 50000]].sum(0)
        elc += (X[c0:c0 + 50000] > 0).sum(0)
    fire = {'all': fire['all'] / n, 'sg': fire['sg'] / sg.sum(), 'non': fire['non'] / non.sum()}
    el = elc / n >= C.ELIG
    cls = F.cls.values
    Y = {b: F[f'y_{b}'].values for b in C.BEH}
    cols, names = [], []
    for nm, m in (('sg', sg), ('non', non)):
        for b in C.BEH:
            cols.append(np.where(m, np.where(Y[b], 2, 1), 0)); names.append(f'{nm}|{b}')
        ex = m & ((cls == 1) | (cls == 2))
        cols.append(np.where(ex, np.where(cls == 1, 2, 1), 0)); names.append(f'{nm}|disc')
    cats = np.ascontiguousarray(np.stack(cols, 1).astype(np.int8))
    Aa, P = A.block_stats(X, R, cats)
    S = {f'auc|{nm}': Aa[i] for i, nm in enumerate(names)}
    S.update({f'ap|{nm}': P[i] for i, nm in enumerate(names) if not nm.endswith('disc')})
    log(f'{key}: all-neuron AUROC / AP ({time.time() - t0:.0f}s)')
    picks = {}
    picks['sgap'], cand = sgap(S, el, fire['all'], cap=True)
    picks['sgap_nocap'], _ = sgap(S, el, fire['all'], cap=False)
    st = {'fire': np.where(el, 1.0, 0.0), 'auc_nose_nose': S['auc|sg|nose_nose'], 'auc_nt': S['auc|sg|nt'], 'disc': S['auc|sg|disc']}
    tag_check = {}
    tp = OUT / 'mice' / 'tags' / f'{key}.pkl'
    T = pickle.load(open(tp, 'rb')) if tp.exists() else None
    for r in ('auc', 'dist', 'dist2'):
        mine = EV.select(st, r)
        if T is not None:
            stored = T['sets'][f'sg2non|{r}'][0]['picks']
            tag_check[r] = stored == mine
            picks[r] = stored
        else:
            picks[r] = mine
    log(f'{key}: picks {picks}; tag check {tag_check}')
    # held-out quantities of every picked neuron
    rr = np.flatnonzero(non)
    obs = F.obs.values[rr]
    vcode, vids = pd.factorize(obs)
    rng = np.random.default_rng(20261125)
    Wv = A.video_weights(len(vids), rng)
    one = np.ones((1, len(vids)))
    neur = {}
    for r, pk in picks.items():
        if pk is None:
            continue
        for b in C.BEH:
            j = pk[b]
            if (j, b) in neur:
                continue
            sc = A.score_of(X, R, j)
            raw = np.asarray(R[:, j], np.float32)
            d = {'neuron': int(j), 'level': level(key, j), 'fire_raw_all': float(fire['all'][j]), 'fire_raw_sg': float(fire['sg'][j]),
                 'fire_raw_heldout': float(fire['non'][j]), 'eligible': bool(el[j])}
            for nm, m in (('sg', sg), ('non', non)):
                d[f'{nm}|auc'] = float(S[f'auc|{nm}|{b}'][j])
                d[f'{nm}|ap'] = float(S[f'ap|{nm}|{b}'][j])
                d[f'{nm}|disc'] = float(S[f'auc|{nm}|disc'][j])
                ri = np.flatnonzero(m)
                for k in KS:
                    _, h, p = A.topk_video(F.obs.values[ri], Y[b][ri], F[f'v_{b}'].values[ri], raw[ri], k)
                    d[f'{nm}|top{k}'] = A.topk_point(h, p, k)
            y, v = Y[b][rr], F[f'v_{b}'].values[rr]
            pt = A.boot_metrics(sc[rr], raw[rr], y, v, obs, vcode, one, KS)
            bt = A.boot_metrics(sc[rr], raw[rr], y, v, obs, vcode, Wv, KS)
            d['ho'] = {m_: float(arr[0]) for m_, arr in pt.items()}
            d['ho_ci'] = {m_: np.nanpercentile(arr, [2.5, 97.5]).tolist() for m_, arr in bt.items()}
            # chance of precision@top-k/video = mean per-video prevalence (valid frames) over videos with >= 1 positive
            pv = pd.DataFrame({'c': vcode[v], 'y': y[v]}).groupby('c').y.agg(['sum', 'count'])
            pv = pv[pv['sum'] > 0]
            d['ho_chance_topk'] = float((pv['sum'] / pv['count']).mean())
            d['ho_eligible_videos'] = int(len(pv))
            if not isv1:  # per-video time: sum P(on) x DT vs true time (v22_eval.held 'video' S / T, all held-out frames)
                Sv = np.bincount(vcode, weights=X[rr, j].astype(np.float64)) * DT
                Tv = np.bincount(vcode, weights=y.astype(np.float64)) * DT
                e = Sv - Tv
                d['video_bias_s'] = [float(e.mean()), float(e.std(ddof=1) / np.sqrt(len(e)))]
                d['video_mae_s'] = [float(np.abs(e).mean()), float(np.abs(e).std(ddof=1) / np.sqrt(len(e)))]
                d['video_true_s'] = [float(Tv.mean()), float(Tv.std(ddof=1) / np.sqrt(len(Tv)))]
            neur[(int(j), b)] = d
    log(f'{key}: held-out metrics of {len(neur)} picks ({time.time() - t0:.0f}s)')
    pairs = {}
    st_ = np.r_[np.flatnonzero(np.r_[True, vcode[1:] != vcode[:-1]]), len(vcode)]
    for r, pk in picks.items():
        if pk is None:
            continue
        tops = {}
        for b in C.BEH:
            Rj = np.asarray(R[rr, pk[b]], np.float64)
            s = []
            for a0, b0 in zip(st_[:-1], st_[1:]):
                s += (a0 + np.lexsort((np.arange(b0 - a0), -Rj[a0:b0]))[:3]).tolist()
            tops[b] = set(s)
        pairs[r] = {'distinct': pk['nose_nose'] != pk['nt'], 'top3_overlap': len(tops['nose_nose'] & tops['nt']) / len(tops['nose_nose']),
                    'act_corr_heldout': float(np.corrcoef(X[rr, pk['nose_nose']], X[rr, pk['nt']])[0, 1]),
                    'sg_ap_sum': float(S['ap|sg|nose_nose'][pk['nose_nose']] + S['ap|sg|nt'][pk['nt']])}
    G = F[non]
    gt = {b: {'heldout_frames': int(len(G)), 'heldout_videos': int(G.obs.nunique()), 'positive_frames': int(G[f'y_{b}'].sum()),
              'prevalence': float(G[f'y_{b}'].mean()), 'videos_with_positive': int((G.groupby('obs')[f'y_{b}'].sum() > 0).sum())} for b in C.BEH}
    res = {'key': key, 'frames_scored': int(n), 'sg_frames': int(sg.sum()), 'heldout_frames': int(non.sum()), 'n_eligible': int(el.sum()),
           'picks': picks, 'tag_check': tag_check, 'cand': cand, 'neurons': {f'{j}|{b}': d for (j, b), d in neur.items()}, 'pairs': pairs,
           'ground_truth': gt, 'n_neurons_fire_lt_cap': int((fire['all'] < FIRE_CAP).sum()),
           'fire_raw_all_quantiles_eligible': np.percentile(fire['all'][el], [10, 50, 90]).tolist()}
    od = OUT / 'mice' / 'dicts'
    od.mkdir(parents=True, exist_ok=True)
    (od / f'{key}.json').write_text(json.dumps(res, indent=1, default=float))
    log(f'{key}: done ({time.time() - t0:.0f}s)')


def cmd_ants(a):
    A.ants_paths = lambda key: paths('ants', key) if not key.startswith(('V_', 'v1_')) else A.ants_paths0(key)
    A.WORK = OUT / 'ants' / 'work'
    A.cmd_ants(a)
    r = pickle.load(open(A.WORK / f'ants_{a.key}.pkl', 'rb'))
    S = r['stats']
    out = {'key': a.key, 'dirs': {}}
    tp = OUT / 'ants' / 'tags' / f'{a.key}.pkl'
    T = pickle.load(open(tp, 'rb')) if tp.exists() else None
    for dn, d in r['dirs'].items():
        s_, e_ = AN_DIRS[dn]
        j = d['pick']
        row = {'pick': j, 'level': level(a.key, j), 'tag_pick': T['sets'][dn][0]['picks']['groom_any'] if T else None,
               'sel_auc_any': float(S[f'auc|{s_}|groom_any'][j]), 'sel_top3_any': d['top3'][('sel', 'groom_any', j)], 'held_videos': d['held_videos']}
        for b in ('groom_any', 'single', 'double'):
            row[f'{b}|auc'] = float(S[f'auc|{e_}|{b}'][j])
            row[f'{b}|ap'] = float(S[f'ap|{e_}|{b}'][j])
            row[f'{b}|top3'] = d['top3'][('held', b, j)]
            for m in ('auc', 'top3', 'ap'):
                row[f'{b}|{m}|ci'] = np.nanpercentile(d['boot'][b][m], [2.5, 97.5]).tolist()
        out['dirs'][dn] = row
    od = OUT / 'ants' / 'dicts'
    od.mkdir(parents=True, exist_ok=True)
    (od / f'{a.key}.json').write_text(json.dumps(out, indent=1, default=float))
    log(f'ants {a.key}: {json.dumps(out)}')


A.ants_paths0 = A.ants_paths
AN_DIRS = A.AN.DIRS

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('dom', choices=('mice', 'ants'))
    p.add_argument('--key', required=True)
    a = p.parse_args()
    (cmd_mice if a.dom == 'mice' else cmd_ants)(a)
