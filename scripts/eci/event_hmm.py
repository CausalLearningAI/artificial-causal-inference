"""
Label-free EVENT CONFIDENCE of the C1 deployment SAE neurons (src/eci/event_hmm.py): a two-state zero-inflated HMM per
neuron on its 5 fps read-out codes_max, fitted by EM on UNLABELLED videos; P(ON | video) from forward-backward is the
event confidence. Ablation: the same emission model as a per-frame mixture (no temporal coupling). Labels are used for
validation only.

Fit / evaluation videos (fixed before looking at any result)
    mice   fit on the 288 videos without human annotation, evaluate on the 144 annotated ones
    ants   every video is annotated: 2-fold cross-fit over videos (sorted observation ids, index parity = the rollout's
           unit halves); fit on one half, posterior on the other, both halves pooled for the metrics
    frogs  no behaviour labels: fit on all 35 videos (in-sample), descriptive per-group summary only

Scores compared on the labelled evaluation frames (all are label-free)
    raw      codes_max
    rescale  min(1, codes_max / q99) (readout_norm.npz of the SAE; same ranking as raw except the ties at 1)
    smooth   centred 11-frame (2.2 s) moving average of the firing indicator codes_max > 0, within video (a plain
             temporal-smoothing control for the HMM)
    mixture  per-frame posterior P(ON | a_t) of the zero-inflated two-component mixture
    hmm      smoothed posterior P(ON | video) of the zero-inflated two-state HMM

Steps (scripts/eci/event_hmm.sh runs each in Slurm, CPU)
    run     --domain d   one sequential chunked pass over codes_max keeps the key-neuron columns (cached as
                         OUT/<d>/cols.npz), fits, validates -> OUT/<d>/event_hmm.json, fig_data.npz, posteriors.npz
    export  --domain d   from the saved fold parameters: HMM P(ON) of EVERY video per key neuron (held-out where a
                         fold did not fit on the video) -> OUT/<d>/n<neuron>_pon.npz, params_hmm.json
    all     --domain ants  every neuron (codes_max read into RAM in row chunks): 2-fold cross-fitted HMM + mixture,
                         per-neuron parameters and AUROC raw / mixture / hmm for the three grooming labels
                         -> OUT/ants/all_neurons.csv, all_neurons.json
    figure  OUT/{mice,ants}/fig_data.npz -> OUT/fig_event_confidence.{png,pdf}

Outputs: results/vision/eci_event_hmm/
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts/eci'))
import rollout_c1 as rc  # noqa: E402
from src.eci import event_hmm as eh  # noqa: E402
from src.eci.domain import get_domain  # noqa: E402
from src.eci.foreground import obs_rows  # noqa: E402
from src.eci.readout_norm import load_norm  # noqa: E402

OUT = REPO / 'results/vision/eci_event_hmm'
FPS = 5.0
CHUNK = 50_000
SMOOTH = 11
KEY = {'mice': [915, 3460, 180], 'ants': [186, 418], 'frogs': [500]}
DESC = {915: 'contact (nose-nose best AUROC)', 3460: 'nose-nose top-1% pick', 180: 'bag (no human label)',
        186: 'grooming (best AUROC)', 418: 'yellow grooming (top-1% pick)', 500: 'FoxP1 NES hit'}
TARGET = {915: 'nose_nose', 3460: 'nose_nose', 180: None, 186: 'groom_any', 418: 'groom_yellow', 500: None}
LABELS = {'mice': ['nose_nose', 'nose_tail'], 'ants': ['groom_any', 'groom_yellow', 'groom_blue']}
PRIMARY = {'mice': 'nose_nose', 'ants': 'groom_any'}
FIG_NEURON = {'mice': 915, 'ants': 186}
FIG_SECONDS = 120
FIG_RULE = ('among the evaluation videos, sorted by their primary-label frame rate (ties by observation id), the one at '
            'index (n - 1) // 2 (the lower median); in it, the first non-overlapping 2-min block from the video start '
            '(frames [600 k, 600 k + 600) at 5 fps) with >= 1 annotated-positive frame')
SCORES = ('raw', 'rescale', 'smooth', 'mixture', 'hmm')
log = lambda s: print(time.strftime('%H:%M:%S'), s, flush=True)  # noqa: E731


# ---------------------------------------------------------------------------------------------- data
def theta_of(d):
    import torch
    ck = torch.load(rc.sae_dir(d) / 'sae.pt', map_location='cpu', weights_only=False)
    return float(ck['state_dict']['threshold'])


def videos(d):
    """[(obs, lo, hi)] in row order."""
    return sorted(((o, lo, hi) for o, (lo, hi) in obs_rows(get_domain(d).ann_path).items()), key=lambda x: x[1])


def concat(vids):
    """[(obs, lo, hi)] -> rows (int64), starts (V + 1,)."""
    rows = np.concatenate([np.arange(lo, hi) for _, lo, hi in vids])
    starts = np.r_[0, np.cumsum([hi - lo for _, lo, hi in vids])]
    return rows, starts


def read_columns(d, cols):
    """One sequential pass over codes_max in row chunks (no random column access over NFS) -> (N, len(cols)) fp16."""
    X = np.load(rc.codes_dir(d) / 'codes_max.npy', mmap_mode='r')
    N = X.shape[0]
    out = np.empty((N, len(cols)), np.float16)
    t0 = time.time()
    for a in range(0, N, CHUNK):
        b = min(N, a + CHUNK)
        out[a:b] = np.asarray(X[a:b])[:, cols]
    log(f'{d}: read {len(cols)} columns of {N:,} frames in {time.time() - t0:.0f}s')
    return out


def labels_all(d):
    """Labelled rows -> DataFrame(row, obs, labels...) (as spatial_sae_pilot.labelled_frames)."""
    import spatial_sae_pilot as ssp
    lab, _ = ssp.labelled_frames(d)
    return lab


def split(d, lab):
    """-> list of (fit videos, eval videos) folds, plus a description."""
    V = videos(d)
    if d == 'mice':
        ann = set(lab.obs)
        fit_v = [v for v in V if v[0] not in ann]
        ev = [v for v in V if v[0] in ann]
        return [(fit_v, ev)], f'fit on the {len(fit_v)} unannotated videos, evaluate on the {len(ev)} annotated'
    if d == 'ants':
        ids = sorted(v[0] for v in V)
        half = {o: i % 2 for i, o in enumerate(ids)}
        h = [[v for v in V if half[v[0]] == k] for k in (0, 1)]
        return [(h[0], h[1]), (h[1], h[0])], (f'all {len(V)} videos are annotated: 2-fold cross-fit over videos (sorted '
                                              f'ids, index parity; {len(h[0])} / {len(h[1])}), fit on one half, '
                                              'posterior on the other')
    return [(V, V)], f'no labels: fit and posterior on all {len(V)} videos (in-sample)'


def smooth_fire(a, starts, w=SMOOTH):
    """Centred w-frame moving average of (a > 0) within each video (edges: average over the frames available)."""
    f = (np.asarray(a) > 0).astype(np.float64)
    out = np.empty(f.shape, np.float32)
    h = w // 2
    for s, e in zip(starts[:-1], starts[1:]):
        x = f[s:e]
        c = np.vstack([np.zeros((1,) + x.shape[1:]), np.cumsum(x, 0)])
        i = np.arange(e - s)
        lo, hi = np.maximum(i - h, 0), np.minimum(i + h + 1, e - s)
        out[s:e] = (c[hi] - c[lo]) / (hi - lo)[:, None]
    return out


# ---------------------------------------------------------------------------------------------- metrics
def auc_ap(y, s):
    from sklearn.metrics import average_precision_score, roc_auc_score
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def reliability(p, y, n_bins=10):
    b = np.minimum((p * n_bins).astype(int), n_bins - 1)
    out = []
    for k in range(n_bins):
        m = b == k
        out.append({'bin': [k / n_bins, (k + 1) / n_bins], 'n': int(m.sum()),
                    'mean_pred': float(p[m].mean()) if m.any() else None,
                    'label_rate': float(y[m].mean()) if m.any() else None})
    return out


def runs(x, rows):
    """Bouts of a bool sequence over (sorted) rows; a gap in rows ends a bout. -> n bouts, durations (frames)."""
    x = np.asarray(x, bool)
    if not x.any():
        return 0, np.zeros(0, int)
    brk = np.r_[True, np.diff(rows) != 1]
    start = x & (brk | ~np.r_[False, x[:-1]])
    lab = np.cumsum(start) * x
    dur = np.bincount(lab)[1:]
    return int(start.sum()), dur


def corr(a, b):
    from scipy.stats import pearsonr, spearmanr
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return {'pearson': None, 'spearman': None}
    return {'pearson': float(pearsonr(a, b)[0]), 'spearman': float(spearmanr(a, b)[0])}


# ---------------------------------------------------------------------------------------------- run
def cmd_run(args):
    d = args.domain
    (OUT / d).mkdir(parents=True, exist_ok=True)
    neurons = KEY[d]
    theta = theta_of(d)
    qn = load_norm(rc.sae_dir(d))['q'][neurons].astype(np.float32)
    cache = OUT / d / 'cols.npz'
    if cache.exists() and list(np.load(cache)['neurons']) == neurons:
        X = np.load(cache)['X']
        log(f'{d}: cached columns {cache}')
    else:
        X = read_columns(d, neurons)
        np.savez(cache, X=X, neurons=np.array(neurons))
    X = X.astype(np.float32)
    N, J = X.shape
    lab = labels_all(d) if d in LABELS else None
    folds, split_desc = split(d, lab if lab is not None else pd.DataFrame({'obs': []}))
    R = {'domain': d, 'sae': rc.sae_name(d), 'theta': theta, 'fps': FPS, 'neurons': neurons, 'split': split_desc,
         'model': 'two-state HMM; P(a=0|s) = z_s; u = log(a/theta) | a>0, s ~ Gamma(k_s, c_s); 2x2 transitions; '
                  'initial probabilities; EM (Baum-Welch); ON = state with the higher expected read-out E[a|s]',
         'folds': []}
    post = {m: np.full((N, J), np.nan, np.float32) for m in ('hmm', 'mixture')}
    is_eval = np.zeros(N, bool)
    for f, (fit_v, ev_v) in enumerate(folds):
        rf, sf = concat(fit_v)
        re_, se = concat(ev_v)
        Xf = X[rf]
        F = {'fold': f, 'n_fit_videos': len(fit_v), 'n_fit_frames': int(len(rf)), 'n_eval_videos': len(ev_v),
             'fit_firing_rate': [float(x) for x in (Xf > 0).mean(0)]}
        for m, temporal in (('hmm', True), ('mixture', False)):
            t0 = time.time()
            P = eh.fit(Xf, sf, theta, temporal=temporal)
            F[f'{m}_fit_s'] = round(time.time() - t0, 1)
            F[f'{m}_n_iter'] = int(P['n_iter'])
            F[f'{m}_converged'] = [bool(c) for c in P['converged']]
            F[f'{m}_loglik_per_frame'] = [float(x) for x in P['loglik']]
            F[f'{m}_loglik_monotone'] = bool(np.all(np.diff(P['history'], axis=0) > -1e-7))
            F[f'{m}_params'] = {str(j): s for j, s in zip(neurons, eh.summary(P, theta, FPS))}
            post[m][re_] = eh.posterior(X[re_], se, theta, P)
            np.savez(OUT / d / f'params_{m}_fold{f}.npz', **{k: P[k] for k in ('z', 'k', 'sc', 'A', 'pi')},
                     neurons=np.array(neurons), theta=theta)
            log(f'{d} fold {f} {m}: {F[f"{m}_n_iter"]} iters, {F[f"{m}_fit_s"]}s, converged {F[f"{m}_converged"]}, '
                f'monotone {F[f"{m}_loglik_monotone"]}: ' + json.dumps(F[f'{m}_params'], default=float)[:1500])
        is_eval[re_] = True
        R['folds'].append(F)
    # ---- all-video score arrays
    V = videos(d)
    rall, sall = concat(V)
    assert np.array_equal(rall, np.arange(N))
    S = {'raw': X, 'rescale': np.minimum(1.0, X / qn[None]), 'smooth': smooth_fire(X, sall), **post}
    np.savez(OUT / d / 'posteriors.npz', rows=np.flatnonzero(is_eval), neurons=np.array(neurons),
             **{m: post[m][is_eval].astype(np.float16) for m in post})
    # ---- ON occupancy on the evaluation videos
    occ = {}
    for i, j in enumerate(neurons):
        ph = post['hmm'][is_eval, i]
        occ[str(j)] = {'mean_p_on': float(ph.mean()), 'frac_decoded_on': float((ph > 0.5).mean()),
                       'stationary_p_on_fit': [F['hmm_params'][str(j)]['stationary_p_on'] for F in R['folds']],
                       'raw_firing_rate': float((X[is_eval, i] > 0).mean()),
                       'mixture_mean_p_on': float(post['mixture'][is_eval, i].mean())}
    R['eval_occupancy'] = occ
    log(f'{d} eval occupancy: {json.dumps(occ)}')
    if lab is None:
        cmd_frogs_groups(d, V, S, neurons, R)
        (OUT / d / 'event_hmm.json').write_text(json.dumps(R, indent=1, default=float))
        log(f'-> {OUT / d}')
        return
    # ---- validation on labelled evaluation frames
    lab = lab[is_eval[lab['row'].values]].sort_values('row').reset_index(drop=True)
    rows = lab['row'].values
    R['n_eval_labelled_frames'] = int(len(rows))
    R['n_eval_labelled_videos'] = int(lab.obs.nunique())
    val = {}
    for i, j in enumerate(neurons):
        val[str(j)] = {}
        for b in LABELS[d]:
            y = lab[b].values.astype(bool)
            r = {'base_rate': float(y.mean()), 'is_target_label': TARGET[j] == b}
            for m in SCORES:
                s = S[m][rows, i]
                r[f'auroc_{m}'], r[f'ap_{m}'] = auc_ap(y, s)
                if m != 'raw':
                    r[f'brier_{m}'] = float(np.mean((s - y) ** 2))
            r['brier_base_rate_reference'] = float(y.mean() * (1 - y.mean()))
            r['reliability_hmm'] = reliability(S['hmm'][rows, i], y)
            r['reliability_mixture'] = reliability(S['mixture'][rows, i], y)
            # bouts per evaluation video
            per = []
            for o, g in lab.groupby('obs', sort=True):
                rr = g['row'].values
                ya = g[b].values.astype(bool)
                na, da = runs(ya, rr)
                nr, dr = runs(X[rr, i] > 0, rr)
                nh, dh = runs(post['hmm'][rr, i] > 0.5, rr)
                nm, dm = runs(post['mixture'][rr, i] > 0.5, rr)
                per.append({'obs': o, 'n_frames': len(rr), 'annot': na, 'raw': nr, 'hmm': nh, 'mixture': nm,
                            'annot_rate': float(ya.mean()), 'raw_rate': float((X[rr, i] > 0).mean()),
                            'hmm_mean_p': float(post['hmm'][rr, i].mean()),
                            'mixture_mean_p': float(post['mixture'][rr, i].mean()),
                            'annot_med_dur_s': float(np.median(da) / FPS) if na else None,
                            'raw_med_dur_s': float(np.median(dr) / FPS) if nr else None,
                            'hmm_med_dur_s': float(np.median(dh) / FPS) if nh else None})
            P_ = pd.DataFrame(per)
            r['bouts'] = {
                'n_videos': len(P_),
                'total_annot': int(P_.annot.sum()), 'total_raw': int(P_.raw.sum()), 'total_hmm': int(P_.hmm.sum()),
                'total_mixture': int(P_.mixture.sum()),
                'corr_raw_vs_annot': corr(P_.raw, P_.annot), 'corr_hmm_vs_annot': corr(P_.hmm, P_.annot),
                'corr_mixture_vs_annot': corr(P_.mixture, P_.annot),
                'median_bout_s_annot': float(np.nanmedian(P_.annot_med_dur_s.astype(float))),
                'median_bout_s_raw': float(np.nanmedian(P_.raw_med_dur_s.astype(float))),
                'median_bout_s_hmm': float(np.nanmedian(P_.hmm_med_dur_s.astype(float))),
                'video_rate_corr_raw': corr(P_.raw_rate, P_.annot_rate),
                'video_rate_corr_hmm': corr(P_.hmm_mean_p, P_.annot_rate),
                'video_rate_corr_mixture': corr(P_.mixture_mean_p, P_.annot_rate)}
            P_.to_csv(OUT / d / f'bouts_{j}_{b}.csv', index=False)
            val[str(j)][b] = r
            log(f'{d} {j} {b}: ' + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()
                                               if not k.startswith('reliab') and k != 'bouts'}))
            log(f'   bouts: ' + json.dumps(r['bouts'], default=float))
    R['validation'] = val
    # ---- figure data (fixed rule)
    b0 = PRIMARY[d]
    rate_v = lab.groupby('obs')[b0].mean().reset_index().sort_values([b0, 'obs']).reset_index(drop=True)
    k = (len(rate_v) - 1) // 2
    obs = rate_v.loc[k, 'obs']
    lo, hi = obs_rows(get_domain(d).ann_path)[obs]
    lv = lab[lab.obs == obs].set_index('row')
    L = int(FIG_SECONDS * FPS)
    start = next((s for s in range(0, hi - lo - L + 1, L)
                  if lv[b0].reindex(np.arange(lo + s, lo + s + L), fill_value=False).values.any()), None)
    if start is None:
        raise SystemExit(f'{d} {obs}: no 2-min block with a positive frame')
    r = np.arange(lo + start, lo + start + L)
    fd = {'rows': r, 'start_frame': start, 'fps': FPS, 'obs': np.array(obs), 'rule': np.array(FIG_RULE),
          'video_rate': float(rate_v.loc[k, b0]), 'video_rank': k, 'n_videos': len(rate_v), 'theta': theta,
          'primary_label': np.array(b0), 'neurons': np.array(neurons)}
    for b in LABELS[d]:
        fd[f'y_{b}'] = lv[b].reindex(r, fill_value=False).values.astype(bool)
    for i, j in enumerate(neurons):
        for m in SCORES:
            fd[f'{m}_{j}'] = S[m][r, i]
        rel = val[str(j)][b0]['reliability_hmm']
        fd[f'rel_pred_{j}'] = np.array([x['mean_pred'] if x['n'] else np.nan for x in rel], float)
        fd[f'rel_rate_{j}'] = np.array([x['label_rate'] if x['n'] else np.nan for x in rel], float)
        fd[f'rel_n_{j}'] = np.array([x['n'] for x in rel])
        fd[f'brier_{j}'] = val[str(j)][b0]['brier_hmm']
        fd[f'brier_ref_{j}'] = val[str(j)][b0]['brier_base_rate_reference']
        fd[f'brier_rate_{j}'] = val[str(j)][b0]['base_rate']
    np.savez(OUT / d / 'fig_data.npz', **fd)
    R['figure'] = {'obs': obs, 'start_frame': int(start), 'video_rate': fd['video_rate'], 'rank': int(k),
                   'n_videos': int(len(rate_v)), 'rule': FIG_RULE, 'segment_primary_rate': float(fd[f'y_{b0}'].mean())}
    log(f'figure: {json.dumps(R["figure"])}')
    (OUT / d / 'event_hmm.json').write_text(json.dumps(R, indent=1, default=float))
    log(f'-> {OUT / d}')


def cmd_frogs_groups(d, V, S, neurons, R):
    exp = pd.read_csv(get_domain(d).experiment_csv).set_index('observation_id')
    per = []
    for o, lo, hi in V:
        for i, j in enumerate(neurons):
            ph = S['hmm'][lo:hi, i]
            nh, _ = runs(ph > 0.5, np.arange(lo, hi))
            per.append({'obs': o, 'group': exp.loc[o, 'group'], 'neuron': j, 'mean_p_on': float(ph.mean()),
                        'frac_decoded_on': float((ph > 0.5).mean()), 'raw_rate': float((S['raw'][lo:hi, i] > 0).mean()),
                        'hmm_bouts_per_min': nh / ((hi - lo) / FPS / 60)})
    P_ = pd.DataFrame(per)
    P_.to_csv(OUT / d / 'per_video.csv', index=False)
    R['per_group_descriptive'] = {
        str(j): {g: {'n_videos': int(len(x)), **{c: float(x[c].mean()) for c in
                                                 ('mean_p_on', 'frac_decoded_on', 'raw_rate', 'hmm_bouts_per_min')}}
                 for g, x in P_[P_.neuron == j].groupby('group')} for j in neurons}
    R['per_group_note'] = 'in-sample, descriptive (video means); not a test'
    log(f'{d} per group: {json.dumps(R["per_group_descriptive"])}')


# ---------------------------------------------------------------------------------------------- export
def cmd_export(args):
    """Per-frame HMM P(ON) of EVERY video for each key neuron, from the saved fold parameters (no refit), and the
    fitted parameters as JSON. Each video gets the posterior of a fold that did NOT fit on it where one exists
    (mice annotated videos, every ants video); else the in-sample posterior (mice unannotated = the fit videos,
    frogs): flagged per frame in 'heldout'.
    -> OUT/<d>/n<neuron>_pon.npz (rows, p_on float16, video_id, heldout), OUT/<d>/params_hmm.json"""
    d = args.domain
    neurons = KEY[d]
    z = np.load(OUT / d / 'cols.npz')
    assert list(z['neurons']) == neurons
    X = z['X'].astype(np.float32)
    N = len(X)
    theta = theta_of(d)
    lab = labels_all(d) if d in LABELS else pd.DataFrame({'obs': []})
    folds, split_desc = split(d, lab)
    P = [dict(np.load(OUT / d / f'params_hmm_fold{f}.npz')) for f in range(len(folds))]
    for p in P:
        assert list(p['neurons']) == neurons and float(p['theta']) == theta
    pon = np.full((N, len(neurons)), np.nan, np.float32)
    held = np.zeros(N, bool)
    for f, (fit_v, ev_v) in enumerate(folds):
        fit_ids = {v[0] for v in fit_v}
        out_v = [v for v in ev_v if v[0] not in fit_ids]
        if out_v:
            r, s = concat(out_v)
            pon[r] = eh.posterior(X[r], s, theta, P[f])
            held[r] = True
    rest = [v for v in videos(d) if np.isnan(pon[v[1], 0])]
    if rest:
        r, s = concat(rest)
        pon[r] = eh.posterior(X[r], s, theta, P[0])
    assert not np.isnan(pon).any()
    vid = np.empty(N, object)
    for o, lo, hi in videos(d):
        vid[lo:hi] = o
    vid = vid.astype(str)
    paths = []
    for i, j in enumerate(neurons):
        p = OUT / d / f'n{j}_pon.npz'
        np.savez(p, rows=np.arange(N, dtype=np.int64), p_on=pon[:, i].astype(np.float16), video_id=vid, heldout=held,
                 neuron=j, theta=theta, fps=FPS)
        paths.append(str(p.relative_to(REPO)))
    J = {'domain': d, 'sae': rc.sae_name(d), 'theta': theta, 'fps': FPS, 'split': split_desc,
         'model': 'state 0 = OFF, 1 = ON; P(a=0|s) = z[s]; u = log(a/theta) | a>0, s ~ Gamma(shape k[s], scale sc[s]); '
                  'A[from][to] per frame; pi = initial state probabilities; ON = higher expected read-out',
         'pon_files': paths, 'heldout_frames': int(held.sum()), 'in_sample_frames': int((~held).sum()), 'folds': []}
    for f, p in enumerate(P):
        J['folds'].append({'fold': f, 'fit_videos': [v[0] for v in folds[f][0]], 'neurons': {
            str(j): {'z': p['z'][i].tolist(), 'k': p['k'][i].tolist(), 'sc': p['sc'][i].tolist(),
                     'A': p['A'][i].tolist(), 'pi': p['pi'][i].tolist(), **eh.summary(
                         {k: p[k][i:i + 1] for k in ('z', 'k', 'sc', 'A', 'pi')}, theta, FPS)[0]}
            for i, j in enumerate(neurons)}})
    (OUT / d / 'params_hmm.json').write_text(json.dumps(J, indent=1, default=float))
    # consistency: the stored evaluation posteriors must be reproduced exactly
    E = np.load(OUT / d / 'posteriors.npz')
    dev = float(np.abs(pon[E['rows']].astype(np.float16).astype(np.float32) - E['hmm'].astype(np.float32)).max())
    log(f'{d}: wrote {paths} + params_hmm.json; held-out frames {held.sum():,} / {N:,}; max |P(ON) - stored eval '
        f'posterior| = {dev:.2e}')


# ---------------------------------------------------------------------------------------------- all neurons
def fast_auc(score, y):
    """AUROC with average ranks for ties (Mann-Whitney)."""
    from scipy.stats import rankdata
    r = rankdata(score)
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def cmd_all(args):
    d = args.domain
    ncpu = int(os.environ.get('SLURM_CPUS_PER_TASK', '8'))
    theta = theta_of(d)
    Xm = np.load(rc.codes_dir(d) / 'codes_max.npy', mmap_mode='r')
    N, m = Xm.shape
    t0 = time.time()
    Xall = np.empty((N, m), np.float16)
    for a in range(0, N, CHUNK):
        Xall[a:a + CHUNK] = Xm[a:a + CHUNK]
    log(f'{d}: read codes_max {Xall.shape} in {time.time() - t0:.0f}s')
    lab = labels_all(d).sort_values('row')
    assert np.array_equal(lab['row'].values, np.arange(N))
    Y = {b: lab[b].values.astype(bool) for b in LABELS[d]}
    folds, split_desc = split(d, lab)
    fr = [concat(f) for f, _ in folds]
    er = [concat(e) for _, e in folds]
    rows_out = []
    pool = ThreadPoolExecutor(ncpu)
    nb = args.block
    T0 = time.time()
    for c0 in range(0, m, nb):
        cols = np.arange(c0, min(m, c0 + nb))
        X = Xall[:, cols].astype(np.float32)
        post = {k: np.zeros((N, len(cols)), np.float32) for k in ('hmm', 'mixture')}
        summ = {}
        conv = {}
        for f in range(len(folds)):
            (rf, sf), (re_, se) = fr[f], er[f]
            for k, temporal in (('hmm', True), ('mixture', False)):
                P = eh.fit(X[rf], sf, theta, temporal=temporal, n_iter=args.n_iter)
                post[k][re_] = eh.posterior(X[re_], se, theta, P)
                summ[(k, f)] = eh.summary(P, theta, FPS)
                conv[(k, f)] = P['converged']
        jobs = []
        for i, j in enumerate(cols):
            for b in LABELS[d]:
                jobs += [(i, j, b, 'raw', X[:, i]), (i, j, b, 'mixture', post['mixture'][:, i]),
                         (i, j, b, 'hmm', post['hmm'][:, i])]
        aucs = list(pool.map(lambda t: fast_auc(t[4], Y[t[2]]), jobs))
        A = {(t[1], t[2], t[3]): a for t, a in zip(jobs, aucs)}
        for i, j in enumerate(cols):
            r = {'neuron': int(j), 'firing_rate': float((X[:, i] > 0).mean())}
            for f in range(len(folds)):
                s = summ[('hmm', f)][i]
                r.update({f'f{f}_{k}': s[k] for k in ('stationary_p_on', 'mean_on_bout_s', 'mean_off_bout_s',
                                                      'on_p_fire', 'off_p_fire')})
                r[f'f{f}_hmm_converged'] = bool(conv[('hmm', f)][i])
                r[f'f{f}_mixture_converged'] = bool(conv[('mixture', f)][i])
                r[f'f{f}_mixture_p_on'] = summ[('mixture', f)][i]['stationary_p_on']
            r['hmm_mean_p_on'] = float(post['hmm'][:, i].mean())
            for b in LABELS[d]:
                for k in ('raw', 'mixture', 'hmm'):
                    r[f'auroc_{k}_{b}'] = A[(j, b, k)]
            rows_out.append(r)
        log(f'{d} neurons {c0}-{cols[-1]} done ({time.time() - T0:.0f}s total)')
    T = pd.DataFrame(rows_out)
    T.to_csv(OUT / d / 'all_neurons.csv', index=False)
    S = {'n_neurons': int(len(T)), 'split': split_desc, 'n_iter_cap': args.n_iter,
         'hmm_converged_both_folds': int((T.f0_hmm_converged & T.f1_hmm_converged).sum()),
         'stationary_p_on': {'median': float(T.f0_stationary_p_on.median()),
                             'p10_p90': [float(x) for x in T.f0_stationary_p_on.quantile([.1, .9])]},
         'stationary_p_on_fold_agreement_spearman': float(T.f0_stationary_p_on.corr(T.f1_stationary_p_on, 'spearman')),
         'spearman_stationary_vs_firing_rate': float(T.f0_stationary_p_on.corr(T.firing_rate, 'spearman')),
         'mean_on_bout_s': {'median': float(T.f0_mean_on_bout_s.median()),
                            'p10_p90': [float(x) for x in T.f0_mean_on_bout_s.quantile([.1, .9])]}}
    for b in LABELS[d]:
        for thr in (0.6, 0.7):
            al = T[f'auroc_raw_{b}'] >= thr
            g = T.loc[al, f'auroc_hmm_{b}'] - T.loc[al, f'auroc_raw_{b}']
            gm = T.loc[al, f'auroc_mixture_{b}'] - T.loc[al, f'auroc_raw_{b}']
            S[f'{b}_raw_auroc_ge_{thr}'] = {
                'n': int(al.sum()), 'frac_hmm_gt_raw': float((g > 0).mean()) if al.any() else None,
                'median_gain_hmm': float(g.median()) if al.any() else None,
                'median_gain_mixture': float(gm.median()) if al.any() else None}
        S[f'{b}_best'] = {k: {'neuron': int(T.loc[T[f'auroc_{k}_{b}'].idxmax(), 'neuron']),
                              'auroc': float(T[f'auroc_{k}_{b}'].max())} for k in ('raw', 'mixture', 'hmm')}
    (OUT / d / 'all_neurons.json').write_text(json.dumps(S, indent=1, default=float))
    log(f'{d} all neurons: {json.dumps(S)}')


# ---------------------------------------------------------------------------------------------- figure
def cmd_figure(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'Nimbus Sans',
                                                                        'DejaVu Sans'],
                         'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 7, 'xtick.labelsize': 6.5,
                         'ytick.labelsize': 6.5, 'axes.linewidth': 0.5, 'xtick.major.width': 0.5,
                         'ytick.major.width': 0.5, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
                         'legend.fontsize': 6.3, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    BLUE, ORANGE, GREY, INK, MUTED = '#0072B2', '#E69F00', '#D9D9D9', '#222222', '#666666'  # Okabe-Ito
    NAMES = {'nose_nose': 'nose-to-nose contact (annotated)', 'groom_any': 'grooming (annotated)'}
    DOM = {'mice': 'a  Mice, neuron 915 (contact)', 'ants': 'b  Ants, neuron 186 (grooming)'}
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig = plt.figure(figsize=(7.2, 4.1))
    gs = fig.add_gridspec(4, 2, width_ratios=[4.3, 1], height_ratios=[0.8, 1, 0.8, 1], hspace=0.35, wspace=0.22,
                          left=0.07, right=0.985, top=0.93, bottom=0.155)
    for di, d in enumerate(('mice', 'ants')):
        f = np.load(OUT / d / 'fig_data.npz')
        j = FIG_NEURON[d]
        b0 = str(f['primary_label'])
        y = f[f'y_{b0}']
        t = np.arange(len(y)) / float(f['fps'])
        dy = np.diff(np.r_[0, y.astype(int), 0])
        spans = list(zip(np.flatnonzero(dy == 1), np.flatnonzero(dy == -1)))
        a0 = fig.add_subplot(gs[2 * di, 0])
        a1 = fig.add_subplot(gs[2 * di + 1, 0], sharex=a0)
        raw = f[f'raw_{j}']
        for ax in (a0, a1):
            for s, e in spans:
                ax.axvspan(t[s] - 0.1, t[e - 1] + 0.1, color=GREY, lw=0, zorder=0)
            ax.spines[['top', 'right']].set_visible(False)
            ax.tick_params(colors=INK)
            ax.set_xlim(0, t[-1] + 0.2)
        a0.plot(t, raw, color=INK, lw=0.5, zorder=2)
        a0.axhline(float(f['theta']), color=MUTED, lw=0.4, ls=(0, (2, 2)), zorder=1)
        a0.set_ylabel('read-out', labelpad=2)
        a0.set_ylim(0, max(raw.max() * 1.08, float(f['theta']) * 1.5))
        plt.setp(a0.get_xticklabels(), visible=False)
        start_s = int(f['start_frame']) / float(f['fps'])
        a0.set_title(DOM[d], loc='left', fontweight='bold', pad=3)
        a0.text(1.0, 1.04, f'video {str(f["obs"])}, min {start_s / 60:.0f}-{start_s / 60 + 2:.0f} '
                f'(lower-median video by annotated rate)', transform=a0.transAxes, ha='right', va='bottom',
                fontsize=6.3, color=MUTED)
        a1.plot(t, f[f'hmm_{j}'], color=BLUE, lw=0.8, zorder=3)
        a1.set_ylim(-0.03, 1.03)
        a1.set_yticks([0, 0.5, 1])
        a1.set_ylabel('P(ON)', labelpad=2)
        if di == 1:
            a1.set_xlabel('time in segment (s)')
        else:
            plt.setp(a1.get_xticklabels(), visible=False)
        h = [Line2D([], [], color=INK, lw=0.6, label='SAE read-out (codes_max)'),
             Line2D([], [], color=MUTED, lw=0.5, ls=(0, (2, 2)), label=r'firing threshold $\theta$'),
             Line2D([], [], color=BLUE, lw=0.9, label='HMM P(ON | video)'),
             Patch(color=GREY, lw=0, label='human annotation (a: nose-to-nose, b: grooming)')]
        if di == 1:
            fig.legend(handles=h, loc='lower left', bbox_to_anchor=(0.065, 0.0), ncol=4, frameon=False,
                       handlelength=1.5, columnspacing=1.3)
        # reliability
        ar = fig.add_subplot(gs[2 * di:2 * di + 2, 1])
        pr, rt, n = f[f'rel_pred_{j}'], f[f'rel_rate_{j}'], f[f'rel_n_{j}']
        ok = n >= 20
        ar.plot([0, 1], [0, 1], color=MUTED, lw=0.4, ls=(0, (2, 2)))
        ar.plot(pr[ok], rt[ok], color=BLUE, lw=0.7, marker='o', ms=2.6, mec='white', mew=0.3)
        ar.set_xlim(0, 1)
        ar.set_ylim(0, 1)
        ar.set_aspect('equal')
        ar.set_xticks([0, 0.5, 1])
        ar.set_yticks([0, 0.5, 1])
        ar.spines[['top', 'right']].set_visible(False)
        ar.set_xlabel('HMM P(ON), 10 bins', labelpad=1)
        ar.set_ylabel('annotated rate', labelpad=1)
        ar.text(0.04, 0.96, f'Brier {float(f[f"brier_{j}"]):.3f}\n(base-rate ref. {float(f[f"brier_ref_{j}"]):.3f})\n'
                f'label rate {100 * float(f[f"brier_rate_{j}"]):.1f}%',
                transform=ar.transAxes, va='top', fontsize=6, color=INK)
        ar.set_title('held-out reliability', fontsize=6.5, pad=3, color=INK)
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'fig_event_confidence.{ext}', dpi=300)
    log(f'-> {OUT}/fig_event_confidence.{{png,pdf}}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=('run', 'export', 'all', 'figure'))
    p.add_argument('--domain', default='mice', choices=rc.DOMAINS)
    p.add_argument('--block', type=int, default=512, help='all: neurons per block')
    p.add_argument('--n-iter', type=int, default=300, help='all: EM iteration cap')
    args = p.parse_args()
    {'run': cmd_run, 'export': cmd_export, 'all': cmd_all, 'figure': cmd_figure}[args.step](args)


if __name__ == '__main__':
    main()
