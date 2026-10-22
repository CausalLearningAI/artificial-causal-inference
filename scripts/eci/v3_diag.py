"""
Diagnosis: why the best single SAE neuron's grooming alignment (r4 align.json, groom_any) drops from ants v2 to ants v3.
CPU only, no re-encoding. Outputs -> results/vision/eci_v3_diag/.

Steps (scripts/eci/v3_diag.sh runs each as a defaultp CPU job, inputs staged on $LOCAL_DIR)
    labels  independent rebuild of the per-frame labels from the raw annotation CSVs (data/ants/v{2,3}/annotations) and
            comparison with the labels align.json used (the r4 layout annotations.csv), the tracking csv labels, the
            annotation window (frames past annotation_end_frame must be NaN), the file <-> observation match, base rates.
    score   the align.json frames and labels re-derived exactly as rollout_r4.cmd_align; for r4 d2 / d0, C1, deployed:
            cross-fitted best neuron by AUROC and by normalised R-precision R~ = (P@N_pos - pi) / (1 - pi) (ties pro rata),
            reproduction of align.json cf_auroc (fails loud), per video, per group (treatment / day / nestbox / annotator /
            batch / weak marking / role), within-video rank-normalised pooled AUROC, motion terciles (tracking centroids,
            2 s displacement), video-grouped logistic probes (codes, n_fg only) as ceilings.
    seeds   every SAE seed (3 per variant) applied to the stored prepass tokens (no encoder) of the labelled prepass frames:
            cross-fitted AUROC / R~ per seed, and chosen-seed prepass codes vs the full-pass codes_max on the same frames.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts/eci'))
OUT = REPO / 'results/vision/eci_v3_diag'
R4OUT = REPO / 'results/vision/eci_rollout_r4'
BEE = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/r4')
DOMS = ('ants_v2', 'ants_v3')
SER = {'ants_v2': 'v2', 'ants_v3': 'v3'}
log = lambda s: print(time.strftime('%H:%M:%S'), s, flush=True)  # noqa: E731


def local_dir():
    d = os.environ.get('LOCAL_DIR')
    if not d:
        raise SystemExit('LOCAL_DIR not set')
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    return d


def stage(src, loc):
    """copy src (file) into loc once, return the local path"""
    dst = loc / str(src).strip('/').replace('/', '__')
    if not dst.exists():
        t0 = time.time()
        subprocess.run(['cp', str(src), str(dst)], check=True)
        log(f'staged {src} ({dst.stat().st_size / 1e9:.2f} GB, {time.time() - t0:.0f}s)')
    return dst


# ---------------------------------------------------------------------------------------------- metrics
def rtilde_cols(X, y, block=256):
    """normalised R-precision of every column of X (n, m) (>= 0, any float) against bool y (n,): precision among the top
    N_pos frames (ties at the cut pro rata), (P@N_pos - pi) / (1 - pi). Uniform frame weights (every labelled frame of
    the align set counts once)."""
    import torch
    y = np.asarray(y, bool)
    n = len(y)
    P = int(y.sum())
    pi = P / n
    out = np.full(X.shape[1], np.nan)
    if P == 0 or P == n:
        return out
    yt = torch.from_numpy(y)
    for j0 in range(0, X.shape[1], block):
        xb = torch.as_tensor(np.asarray(X[:, j0:j0 + block], np.float32))
        v = torch.topk(xb, P, dim=0).values[-1]  # value at rank P
        gt = xb > v[None]
        eq = xb == v[None]
        g, e = gt.sum(0).double(), eq.sum(0).double()
        pg, pe = (gt & yt[:, None]).sum(0).double(), (eq & yt[:, None]).sum(0).double()
        tp = pg + pe * (P - g) / e
        out[j0:j0 + block] = ((tp / P - pi) / (1 - pi)).numpy()
    return out


def auc_cols(X, y):
    """AUROC of every column (ties = 1/2), via src.eci.levers.align_gpu on CPU; X fp16 (n, m) >= 0."""
    import torch
    from src.eci.levers import align_gpu
    y = np.asarray(y, bool)
    if y.all() or not y.any():
        return np.full(X.shape[1], np.nan), np.full(X.shape[1], np.nan)
    R = align_gpu(torch.from_numpy(np.ascontiguousarray(X, np.float16)), torch.from_numpy(y[:, None]))
    return R['auc'][:, 0], R['rate']


def auc1(x, y):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y, bool)
    return float(roc_auc_score(y, x)) if 0 < y.sum() < len(y) else float('nan')


def rt1(x, y):
    return float(rtilde_cols(np.asarray(x, np.float32)[:, None], y)[0])


# ---------------------------------------------------------------------------------------------- shared: align frames
def align_frames(dom, loc):
    """rows / labels / halves exactly as rollout_r4.cmd_align (ants branch)."""
    from fg_extract_train import fps1_rows
    P = json.loads((R4OUT / dom / 'plan.json').read_text())
    ann = stage(BEE / dom / 'layout/annotations.csv', loc)
    a = pd.read_csv(ann, usecols=['observation_id', 'frame_idx', 'Y_Y2F', 'Y_B2F', 'Y_YOL'])
    ok = a[['Y_Y2F', 'Y_B2F']].notna().all(1).values
    lab = {'groom_any': (a.Y_Y2F.values > 0) | (a.Y_B2F.values > 0), 'groom_yellow': a.Y_Y2F.values > 0,
           'groom_blue': a.Y_B2F.values > 0}
    ranges = {v['obs']: (v['lay_lo'], v['lay_lo'] + v['n']) for v in P['videos']}
    # layout <-> plan grid check: row lay_lo + k is video obs, frame k
    for v in P['videos']:
        sl = slice(v['lay_lo'], v['lay_lo'] + v['n'])
        assert (a.observation_id.values[sl] == v['obs']).all() and (a.frame_idx.values[sl] == np.arange(v['n'])).all(), v['obs']
    assert ranges and sum(v['n'] for v in P['videos']) == len(a), 'layout rows != plan frames'
    rows = np.sort(np.concatenate(list(fps1_rows(ranges, 5, 0).values())))
    rows = rows[ok[rows]]
    obs = a.observation_id.values[rows]
    units = sorted(set(obs))
    half = pd.Series(obs).map({u: i % 2 for i, u in enumerate(units)}).values
    return dict(P=P, a=a, rows=rows, obs=obs, half=half, k=a.frame_idx.values[rows],
                Y={b: lab[b][rows] for b in lab})


# ---------------------------------------------------------------------------------------------- labels
def parse_raw(path):
    lines = Path(path).read_text().splitlines()
    head = {ln.split(',', 1)[0].strip(): ln.split(',', 1)[1].strip() for ln in lines[:2] if ',' in ln}
    rows = []
    for ln in lines[4:]:
        if ln.startswith('-----') or not ln.strip():
            continue
        f = [x.strip() for x in ln.split(',')]
        rows.append((f[3], int(f[4]), int(f[5])))
    return head, rows


def cmd_labels(args):
    loc = local_dir()
    res = {}
    for dom in DOMS:
        v = SER[dom]
        P = json.loads((R4OUT / dom / 'plan.json').read_text())
        a = pd.read_csv(stage(BEE / dom / 'layout/annotations.csv', loc), usecols=['observation_id', 'frame_idx', 'Y_Y2F', 'Y_B2F'])
        E = pd.read_csv(REPO / f'data/ants/{v}/experiment.csv').set_index('observation_id')
        per, unknown = [], set()
        for vid in P['videos']:
            o, lo, n = vid['obs'], vid['lay_lo'], vid['n']
            af = E.loc[o, 'annotation_file']
            end = int(E.loc[o, 'annotation_end_frame'])
            head, ev = parse_raw(REPO / f'data/ants/{v}/annotations' / af)
            k = np.arange(n)
            rec = {'obs': o, 'annotation_file': af, 'csv_video_file': head.get('Video file'), 'n_events': len(ev),
                   'annotation_end_frame': end}
            rec['name_match'] = Path(af).stem == o and (Path(str(head.get('Video file'))).stem in (o, o.split('_', 1)[-1])
                                                        if v == 'v3' else True)
            Yl = a.Y_Y2F.values[lo:lo + n]
            Bl = a.Y_B2F.values[lo:lo + n]
            win = 6 * k < end
            rec['n_window'] = int(win.sum())
            rec['nan_outside_window'] = bool(np.isnan(Yl[~win]).all() and np.isnan(Bl[~win]).all())
            rec['nan_inside_window'] = int(np.isnan(Yl[win]).sum() + np.isnan(Bl[win]).sum())
            for off in (0, 1, 2, 3, -3):
                f = 6 * k + off
                y2 = np.zeros(n, bool)
                b2 = np.zeros(n, bool)
                for beh, s, e in ev:
                    m = (f >= s) & (f < e)
                    if beh in ('groom-orange', 'groom-orangeandblue'):
                        y2 |= m
                    if beh in ('groom-blue', 'groom-orangeandblue'):
                        b2 |= m
                    if beh not in ('groom-orange', 'groom-blue', 'groom-orangeandblue', 'onLid-focal', 'onLid-orange',
                                   'onLid-blue'):
                        unknown.add(beh)
                mm = int(((y2[win] != (Yl[win] > 0)) | (b2[win] != (Bl[win] > 0))).sum())
                rec[f'mismatch_off{off}'] = mm
                if off == 0:
                    rec['base_any'] = float((y2 | b2)[win].mean())
                    rec['base_y2f'] = float(y2[win].mean())
                    rec['base_b2f'] = float(b2[win].mean())
                    # behaviour beyond the window (annotators sometimes go past 10 min)
                    rec['events_past_window'] = int(sum(e > end for _, s, e in ev))
            tr = REPO / f'dataset/ants/{v}/tracking/{o}.csv'
            if tr.exists():
                T = pd.read_csv(tr, usecols=['frame_idx', 'Y2F', 'B2F'])
                T = T.set_index('frame_idx').reindex(k)
                rec['tracking_label_mismatch'] = int(((T.Y2F.values[win] > 0) != (Yl[win] > 0)).sum()
                                                     + ((T.B2F.values[win] > 0) != (Bl[win] > 0)).sum())
            per.append(rec)
        D = pd.DataFrame(per)
        D.to_csv(OUT / f'labels_{dom}.csv', index=False)
        files = E.loc[[x['obs'] for x in P['videos']], 'annotation_file']
        res[dom] = {'n_videos': len(D), 'annotation_files_unique': bool(files.is_unique),
                    'name_mismatch': D.obs[~D.name_match].tolist(),
                    'csv_video_file_examples': D.csv_video_file.head(5).tolist(),
                    'videos_with_label_mismatch_off0': int((D.mismatch_off0 > 0).sum()),
                    'total_mismatch_by_offset': {c: int(D[c].sum()) for c in D if c.startswith('mismatch_off')},
                    'all_nan_outside_window': bool(D.nan_outside_window.all()),
                    'nan_inside_window': int(D.nan_inside_window.sum()),
                    'n_window_frames_per_video': sorted(D.n_window.unique().tolist()),
                    'tracking_label_mismatch_total': int(D.tracking_label_mismatch.sum()) if 'tracking_label_mismatch' in D else None,
                    'unknown_behaviours': sorted(unknown),
                    'videos_zero_events': int((D.n_events == 0).sum()),
                    'videos_zero_groom': int((D.base_any == 0).sum()),
                    'base_any_mean_over_videos': float(D.base_any.mean()),
                    'base_any_quartiles': np.percentile(D.base_any, [0, 25, 50, 75, 100]).round(3).tolist(),
                    'events_past_window_total': int(D.events_past_window.sum())}
        log(f'{dom}: ' + json.dumps(res[dom]))
    (OUT / 'labels.json').write_text(json.dumps(res, indent=1))


# ---------------------------------------------------------------------------------------------- score
def load_models(dom, F, loc):
    import importlib
    R4 = importlib.import_module('rollout_r4')
    c = R4.DOMS[dom]
    best = json.loads((R4OUT / dom / 'sae/best.json').read_text())
    sets = {f'r4_{v_}': (BEE / dom / 'layout/codes' / best[v_]['name'], F['rows']) for v_ in ('d2', 'd0')}
    A1 = pd.read_csv(REPO / c['c1_ann'], usecols=['observation_id', 'frame_idx', 'Y_Y2F', 'Y_B2F'])
    key = pd.Series(np.arange(len(A1)), index=pd.MultiIndex.from_arrays([A1.observation_id, A1.frame_idx]))
    r1 = key.reindex(pd.MultiIndex.from_arrays([F['obs'], F['k']])).values
    assert not np.isnan(r1).any()
    r1 = r1.astype(np.int64)
    # same frames: the 10-min layout rows are the same (video, frame) and carry the same labels
    assert (A1.observation_id.values[r1] == F['obs']).all() and (A1.frame_idx.values[r1] == F['k']).all()
    y1 = (A1.Y_Y2F.values[r1] > 0) | (A1.Y_B2F.values[r1] > 0)
    same_lab = {'groom_any_mismatch_vs_10min_layout': int((y1 != F['Y']['groom_any']).sum()),
                'nan_in_10min_layout_labels': int(np.isnan(A1.Y_Y2F.values[r1]).sum())}
    sets['C1'] = (REPO / c['c1'], r1)
    sets['deployed'] = (REPO / c['deployed'], r1)
    M = {}
    for nm, (cd, rr) in sets.items():
        cm = np.load(stage(Path(cd) / 'codes_max.npy', loc), mmap_mode='r')
        o = np.argsort(rr)
        X = np.empty((len(rr), cm.shape[1]), np.float16)
        X[o] = cm[rr[o]]
        nfp = Path(cd) / 'n_fg.npy'
        nf = np.load(nfp, mmap_mode='r')[rr].astype(np.float64)
        n32 = np.load(Path(cd) / 'n_fg32.npy', mmap_mode='r')[rr].astype(np.float64) if (Path(cd) / 'n_fg32.npy').exists() else None
        M[nm] = dict(X=X, nf=nf, n32=n32, codes=str(cd))
        del cm
    return M, same_lab


def motion(dom, F):
    """2 s displacement (frames k vs k - 10, clipped) of the tracked blue / yellow / focal centroids, mean of three, px at
    the 512 tracking frame."""
    v = SER[dom]
    out = np.full(len(F['rows']), np.nan)
    for o in np.unique(F['obs']):
        j = np.flatnonzero(F['obs'] == o)
        T = pd.read_csv(REPO / f'dataset/ants/{v}/tracking/{o}.csv').set_index('frame_idx')
        k = F['k'][j]
        kp = np.maximum(k - 10, 0)
        d = 0
        for a_ in ('blue', 'yellow', 'focal'):
            p = T[[f'{a_}_x', f'{a_}_y']].reindex(k).values
            q = T[[f'{a_}_x', f'{a_}_y']].reindex(kp).values
            d = d + np.sqrt(((p - q) ** 2).sum(1))
        out[j] = d / 3
    return out


def cf_select(X, y, half):
    """per direction s -> t: AUROC- and R~-selected neuron on half s; returns dirs and the align.json-style cf numbers"""
    dirs = []
    A, Rt, rate = {}, {}, {}
    for h in (0, 1):
        m = half == h
        A[h], rate[h] = auc_cols(X[m], y[m])
        Rt[h] = rtilde_cols(X[m], y[m])
    for s, t in ((0, 1), (1, 0)):
        sa = np.where(np.isfinite(A[s]), A[s], -np.inf)
        sr = np.where(np.isfinite(Rt[s]), Rt[s], -np.inf)
        ja, jr = int(np.argmax(sa)), int(np.argmax(sr))
        dirs.append({'s': s, 't': t, 'j_auc': ja, 'j_rt': jr, 'test_auc': float(A[t][ja]), 'test_rt_at_auc': float(Rt[t][ja]),
                     'test_rt': float(Rt[t][jr]), 'test_auc_at_rt': float(A[t][jr]), 'test_rate_auc_pick': float(rate[t][ja]),
                     'test_rate_rt_pick': float(rate[t][jr])})
    cf = {q: float(np.mean([d[q] for d in dirs])) for q in ('test_auc', 'test_rt_at_auc', 'test_rt', 'test_auc_at_rt',
                                                             'test_rate_auc_pick', 'test_rate_rt_pick')}
    return dirs, cf, A, Rt


def group_metrics(X, y, half, dirs, mask):
    """metric of the cross-fitted neurons restricted to the frames in mask: per direction on mask & test half, weighted
    mean over the directions with both classes; AUROC of the AUROC pick, R~ of the R~ pick."""
    va, vr, w = [], [], []
    for d in dirs:
        m = mask & (half == d['t'])
        if m.sum() < 20 or y[m].all() or not y[m].any():
            continue
        va.append(auc1(X[m, d['j_auc']].astype(np.float32), y[m]))
        vr.append(rt1(X[m, d['j_rt']], y[m]))
        w.append(m.sum())
    if not w:
        return float('nan'), float('nan'), int(mask.sum())
    return float(np.average(va, weights=w)), float(np.average(vr, weights=w)), int(mask.sum())


def rank_within(x, obs):
    r = np.empty(len(x))
    for o in np.unique(obs):
        j = np.flatnonzero(obs == o)
        r[j] = (pd.Series(x[j]).rank(method='average').values - 0.5) / len(j)
    return r


def probe(Xf, y, obs, seed=0, C=1.0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    p = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(Xf, y, obs):
        sc = StandardScaler().fit(Xf[tr])
        clf = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(Xf[tr]), y[tr])
        p[te] = clf.predict_proba(sc.transform(Xf[te]))[:, 1]
    pv = [auc1(p[obs == o], y[obs == o]) for o in np.unique(obs)]
    return {'auroc': auc1(p, y), 'rtilde': rt1(p, y), 'mean_within_video_auroc': float(np.nanmean(pv))}


def cmd_score(args):
    loc = local_dir()
    al_all = {d: json.loads((R4OUT / d / 'align.json').read_text()) for d in DOMS}
    summary, pv_all, grp_all, mot_all = {}, [], [], []
    for dom in args.domains:
        t0 = time.time()
        v = SER[dom]
        F = align_frames(dom, loc)
        M, same_lab = load_models(dom, F, loc)
        E = pd.read_csv(REPO / f'data/ants/{v}/experiment.csv').set_index('observation_id')
        role = {x['obs']: x['role'] for x in F['P']['videos']}
        obs, half = F['obs'], F['half']
        nper = pd.Series(obs).value_counts()
        S = {'n_frames': int(len(obs)), 'n_videos': int(len(nper)), 'frames_per_video': sorted(set(nper.tolist())),
             'same_frames': same_lab, 'labels': {}}
        mv = motion(dom, F)
        S['motion_nan'] = int(np.isnan(mv).sum())
        mv = np.nan_to_num(mv, nan=0.0)
        tq = np.quantile(mv, [1 / 3, 2 / 3])
        mt = np.digitize(mv, tq)
        S['motion_tercile_edges_px'] = tq.tolist()
        for b in (('groom_any', 'groom_yellow', 'groom_blue') if args.all_labels else ('groom_any',)):
            y = F['Y'][b]
            S['labels'][b] = {'base_rate': float(y.mean()), 'models': {}}
            for nm, m in M.items():
                X = m['X']
                dirs, cf, A, Rt = cf_select(X, y, half)
                ref = al_all[dom]['models'][nm][b]['cf_auroc']
                if abs(cf['test_auc'] - ref) > 2e-4:
                    raise SystemExit(f'{dom} {nm} {b}: reproduced cf AUROC {cf["test_auc"]:.5f} != align.json {ref:.5f}')
                Aall, rall = auc_cols(X, y)
                rec = {'cf_auroc': cf['test_auc'], 'align_json_cf_auroc': ref, 'cf_rtilde': cf['test_rt'],
                       'cf_rtilde_at_auroc_pick': cf['test_rt_at_auc'], 'cf_auroc_at_rtilde_pick': cf['test_auc_at_rt'],
                       'rate_auroc_pick': cf['test_rate_auc_pick'], 'rate_rtilde_pick': cf['test_rate_rt_pick'],
                       'neurons_auroc_pick': [d['j_auc'] for d in dirs], 'neurons_rtilde_pick': [d['j_rt'] for d in dirs],
                       'silent_neuron_frac_on_these_frames': float((rall == 0).mean()), 'width': int(X.shape[1])}
                # within-video rank normalisation of the AUROC-picked neuron (is the pooled AUROC lost between videos?)
                ra, rr_, w = [], [], []
                for d in dirs:
                    mm = half == d['t']
                    xr = rank_within(X[mm, d['j_auc']].astype(np.float64), obs[mm])
                    ra.append(auc1(xr, y[mm]))
                    xr2 = rank_within(X[mm, d['j_rt']].astype(np.float64), obs[mm])
                    rr_.append(rt1(xr2, y[mm]))
                rec['cf_auroc_within_video_ranks'] = float(np.mean(ra))
                rec['cf_rtilde_within_video_ranks'] = float(np.mean(rr_))
                # per video
                pv = []
                for o in nper.index:
                    j = obs == o
                    t = int(half[j][0])
                    d = [d_ for d_ in dirs if d_['t'] == t][0]
                    pv.append({'domain': dom, 'model': nm, 'label': b, 'obs': o, 'half': t, 'n': int(j.sum()),
                               'base': float(y[j].mean()), 'auroc': auc1(X[j, d['j_auc']].astype(np.float32), y[j]),
                               'rtilde': rt1(X[j, d['j_rt']], y[j]), 'role': role[o],
                               'mean_act_auroc_pick': float(X[j, d['j_auc']].astype(np.float32).mean()),
                               'rate_auroc_pick': float((X[j, d['j_auc']] > 0).mean()),
                               'motion_mean_px': float(mv[j].mean()),
                               **{f'meta_{c_}': E.loc[o, c_] for c_ in E.columns
                                  if c_ in ('treatment', 'nestbox', 'batch', 'position', 'recording_date', 'recording_hour',
                                            'annotator', 'weak_marking')}})
                PV = pd.DataFrame(pv)
                pv_all.append(PV)
                rec['mean_per_video_auroc'] = float(PV.auroc.mean())
                rec['median_per_video_auroc'] = float(PV.auroc.median())
                rec['mean_per_video_rtilde'] = float(PV.rtilde.mean())
                rec['median_per_video_rtilde'] = float(PV.rtilde.median())
                rec['n_videos_defined'] = int(PV.auroc.notna().sum())
                # per group
                for c_ in [c for c in PV if c.startswith('meta_')] + ['role']:
                    gv = PV.set_index('obs')[c_]
                    gfr = pd.Series(obs).map(gv).values
                    for g in sorted(set(gv.astype(str))):
                        mask = gfr.astype(str) == g
                        ga, gr, gn = group_metrics(X, y, half, dirs, mask)
                        sub = PV[PV[c_].astype(str) == g]
                        grp_all.append({'domain': dom, 'model': nm, 'label': b, 'group': c_.replace('meta_', ''), 'value': g,
                                        'n_videos': len(sub), 'n_frames': gn, 'base': float(y[mask].mean()),
                                        'pooled_auroc': ga, 'pooled_rtilde': gr, 'mean_video_auroc': float(sub.auroc.mean()),
                                        'mean_video_rtilde': float(sub.rtilde.mean())})
                # motion terciles
                for q in range(3):
                    ga, gr, gn = group_metrics(X, y, half, dirs, mt == q)
                    mot_all.append({'domain': dom, 'model': nm, 'label': b, 'tercile': q, 'n_frames': gn,
                                    'base': float(y[mt == q].mean()), 'pooled_auroc': ga, 'pooled_rtilde': gr,
                                    'motion_px_mean': float(mv[mt == q].mean())})
                # within-video motion terciles (each video split at its own terciles)
                mtv = np.zeros(len(obs), int)
                for o in nper.index:
                    j = np.flatnonzero(obs == o)
                    mtv[j] = np.digitize(mv[j], np.quantile(mv[j], [1 / 3, 2 / 3]))
                for q in range(3):
                    ga, gr, gn = group_metrics(X, y, half, dirs, mtv == q)
                    mot_all.append({'domain': dom, 'model': nm, 'label': b, 'tercile': f'within_video_{q}', 'n_frames': gn,
                                    'base': float(y[mtv == q].mean()), 'pooled_auroc': ga, 'pooled_rtilde': gr,
                                    'motion_px_mean': float(mv[mtv == q].mean())})
                S['labels'][b]['models'][nm] = rec
                log(f'{dom} {nm} {b}: ' + json.dumps({k_: (round(x, 4) if isinstance(x, float) else x) for k_, x in rec.items()}))
        # probes (groom_any)
        y = F['Y']['groom_any']
        PR = {}
        if not args.no_probe:
            for nm, m in M.items():
                PR[nm] = probe(np.log1p(m['X'].astype(np.float32)), y, obs)
                log(f'{dom} probe {nm}: {PR[nm]}')
            nf = M['r4_d2']
            feats = np.stack([np.log1p(nf['nf']), np.log1p(nf['n32'])], 1)
            PR['n_fg_only_r4'] = probe(feats, y, obs)
            PR['n_fg_only_C1'] = probe(np.log1p(M['C1']['nf'])[:, None], y, obs)
            PR['motion_only'] = probe(np.log1p(mv)[:, None], y, obs)
            PR['n_fg_plus_motion'] = probe(np.c_[feats, np.log1p(mv)], y, obs)
            log(f'{dom} probes: {PR}')
        S['probes_groom_any_video_grouped_5fold'] = PR
        summary[dom] = S
        log(f'{dom} done in {time.time() - t0:.0f}s')
    tag = '_'.join(args.domains)
    (OUT / f'score_{tag}.json').write_text(json.dumps(summary, indent=1, default=float))
    pd.concat(pv_all).to_csv(OUT / f'per_video_{tag}.csv', index=False)
    pd.DataFrame(grp_all).to_csv(OUT / f'per_group_{tag}.csv', index=False)
    pd.DataFrame(mot_all).to_csv(OUT / f'motion_{tag}.csv', index=False)


# ---------------------------------------------------------------------------------------------- seeds
def cmd_seeds(args):
    import torch
    from src.eci.sae import load_sae
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    loc = local_dir()
    out = {}
    for dom in args.domains:
        pre = BEE / dom / 'prepass'
        lp = loc / f'{dom}_prepass'
        if not lp.exists():
            t0 = time.time()
            shutil.copytree(pre, lp)
            log(f'staged {pre} in {time.time() - t0:.0f}s')
        Fp = pd.read_parquet(lp / 'frames.parquet')
        tf = np.load(lp / 'tok_frame.npy')
        N = len(tf)
        tok = np.memmap(lp / 'tok.f16', np.float16, 'r', shape=(N, 1024))
        prv = np.memmap(lp / 'prev.f16', np.float16, 'r', shape=(N, 1024))
        a = pd.read_csv(stage(BEE / dom / 'layout/annotations.csv', loc), usecols=['Y_Y2F', 'Y_B2F'])
        Y2, B2 = a.Y_Y2F.values[Fp.lay_row.values], a.Y_B2F.values[Fp.lay_row.values]
        okf = ~(np.isnan(Y2) | np.isnan(B2))
        y = ((Y2 > 0) | (B2 > 0))[okf]
        fidx = np.flatnonzero(okf)
        obs = Fp.obs.values[fidx]
        units = sorted(set(obs))
        half = pd.Series(obs).map({u: i % 2 for i, u in enumerate(units)}).values
        remap = np.full(len(Fp), -1, np.int64)
        remap[fidx] = np.arange(len(fidx))
        sel = np.flatnonzero(remap[tf] >= 0)
        fr_of = torch.from_numpy(remap[tf[sel]])
        best = json.loads((R4OUT / dom / 'sae/best.json').read_text())
        tj = json.loads((R4OUT / dom / 'train.json').read_text())
        D = {'n_frames_labelled': int(len(fidx)), 'n_videos': len(units), 'base_rate': float(y.mean()), 'n_tokens': int(len(sel)),
             'variants': {}}
        for v_ in ('d2', 'd0'):
            for seed in (0, 1, 2):
                t0 = time.time()
                sae, norm, _ = load_sae(R4OUT / dom / 'sae' / f'{v_}_s{seed}' / 'sae_theta.pt', 'cpu')
                fv = torch.zeros((len(fidx), 512), dtype=torch.float32)
                with torch.no_grad():
                    for a_ in range(0, len(sel), 100_000):
                        s_ = sel[a_:a_ + 100_000]
                        t = torch.from_numpy(np.ascontiguousarray(tok[s_])).float()
                        x = torch.cat([t, (t - torch.from_numpy(np.ascontiguousarray(prv[s_])).float()).half().float()], 1) \
                            if v_ == 'd2' else t
                        z = sae.encode(norm(x), mode='threshold')
                        fv.index_reduce_(0, fr_of[a_:a_ + 100_000], z, 'amax', include_self=True)
                X = fv.numpy().astype(np.float16)
                dirs, cf, _, _ = cf_select(X, y, half)
                S_ = tj['variants'][v_]['seeds'][str(seed)]
                rec = {'cf_auroc': cf['test_auc'], 'cf_rtilde': cf['test_rt'], 'cf_rtilde_at_auroc_pick': cf['test_rt_at_auc'],
                       'eval_fve': S_['eval_fve'], 'eval_dead_frac': S_['eval_dead_frac'], 'eval_l0': S_['eval_l0_per_token'],
                       'chosen': best[v_]['seed'] == seed, 'silent_on_labelled_prepass': float((X.max(0) == 0).mean())}
                if best[v_]['seed'] == seed:  # prepass codes == full-pass codes on the same (video, frame)?
                    cm = np.load(BEE / dom / 'layout/codes' / best[v_]['name'] / 'codes_max.npy', mmap_mode='r')
                    lr = Fp.lay_row.values[fidx]
                    o = np.argsort(lr)
                    Xf = np.empty_like(X)
                    Xf[o] = cm[lr[o]]
                    dd = np.abs(Xf.astype(np.float32) - X.astype(np.float32))
                    rec['vs_fullpass'] = {'max_abs_diff': float(dd.max()), 'mean_abs_diff': float(dd.mean()),
                                          'frac_exact': float((dd == 0).mean()),
                                          'corr': float(np.corrcoef(Xf.astype(np.float32).ravel(), X.astype(np.float32).ravel())[0, 1])}
                D['variants'].setdefault(v_, {})[str(seed)] = rec
                log(f'{dom} {v_} s{seed} ({time.time() - t0:.0f}s): ' + json.dumps(rec))
        out[dom] = D
    (OUT / f'seeds_{"_".join(args.domains)}.json').write_text(json.dumps(out, indent=1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=('labels', 'score', 'seeds'))
    p.add_argument('--domains', nargs='+', default=list(DOMS))
    p.add_argument('--all-labels', action='store_true')
    p.add_argument('--no-probe', action='store_true')
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {'labels': cmd_labels, 'score': cmd_score, 'seeds': cmd_seeds}[a.step](a)


if __name__ == '__main__':
    main()
