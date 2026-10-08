"""
Video-level usefulness of mice contact neurons for NES (Neural Effect Search), mice v1. Analysis only, CPU.

Question: a neuron with weak per-frame precision may still track HOW OFTEN a behaviour happens in each video and how
that rate changes at the phase switches, which is all NES uses (per-video mean activation, paired within pool).

Units and windows
    144 annotated videos = 24 pools x 6 stages (C.STAGES: 1 S-H, 2 S-O, 3 S-P, 4 F-H, 5 F-O, 6 F-P). Every video is
    read inside its NES analysis window, MiceDomain.window_start (src/eci/domain.py, commit f0600af): habituation
    (stages 1, 4; 9000 frames) from frame n_frames - 4500 (= its last 15 min), odour and post (4500 frames) whole.
    The same window is applied to neurons, probe scores and labels (frame offset within the video's row block).
Ground truth (dataset/mice/v1/annotations.csv, 5 fps, every frame in the window)
    nose_nose = Y_nn OR Y_np, nose_tail = Y_nt. rate = fraction of window frames positive; bouts/min = number of
    maximal runs of consecutive positive 5 fps frames (no gap merging: one negative frame, 0.2 s, ends a bout; a run
    cut by the window start counts once) / window minutes. rate_1fps = the same rate on the 1 fps eval frames only.
Candidates
    sae_levers  results/vision/eci_sae_levers/mice/ (commits 29ef688, 1bc76c6): configs x seeds; per behaviour and
                selection rule (auc = best AUROC, ap = best AP, top1 = honest top-1% precision) the neuron chosen on
                cross-fit half A (eval/<key>.json labels.<b>.dirs) is read from codes_best/<key>.npz (frame-max codes
                on the 172,800 eval frames, 1 fps) and scored ONLY on the videos / pools of half B, and the reverse.
                Halves = half.npy (pool-level, alternating over sorted pool prefixes).
    deployed    fg448al SAE (dataset/mice/v1/eci/codes/matryoshka_btk_1024_k16_fg448al_s0) neurons 611 (nose-nose)
                and 414 (nose-tail), pooling codes_mean (the NES primary) and codes_max, on the 5 fps frames (what NES
                reads) and on the 1 fps eval frames (like-for-like with sae_levers). A fixed pick from an earlier
                in-sample audit: it is NOT cross-fitted; scored on each half and on all 24 pools. These codes cover
                all 432 videos, so the neuron-only effect test is also run on all 72 pools and on the 48 unannotated
                pools (never used for any label-based choice).
    probe       supervised ceiling: out-of-fold scores (5 folds grouped by pool) of results/vision/eci_repr_diag/
                mice/a/oof_scores.parquet; per frame sigmoid(logit) averaged over seeds; mil_ctx (attention probe)
                and mean_lin. All 24 pools are out-of-fold.
Measurements (per candidate x behaviour)
    1 video tracking  Pearson / Spearman over videos of per-video mean activation vs rate and vs bouts/min, per
                      held-out half (72 videos each) and their mean.
    2 change tracking per pool and switch (social H->O 1->2, O->P 2->3; fear H->O 4->5, O->P 5->6):
                      d = later - earlier stage, for the neuron mean and the annotated rate / bouts; Pearson of
                      d_neuron vs d_rate over the 12 held-out pools of a half, per switch and pooled (48 pairs), per
                      half and mean.
    3 effect recovery per switch, one-sample t-test of d over the 24 held-out pools (each half's pools read with the
                      neuron chosen on the other half; to combine two neurons, each half's neuron means are divided
                      by their sd over that half's 72 videos, label-free). dz = mean(d) / sd(d). Same for the annotated
                      rate and bouts/min.
    4 verdict         'useful for NES' iff held-out video r(rate) >= 0.7 AND pooled d-correlation >= 0.5 AND the sign
                      of mean d_neuron equals the sign of every annotated rate effect with p < 0.05.
Output: OUT/results.json, OUT/per_video.parquet, OUT/table.md (OUT = results/vision/eci_repr_diag/mice/video_level).

Usage (CPU job, scripts/eci/diag_video_level.sh): python scripts/eci/diag_video_level.py --stage <local dir>
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.eci import contrasts as C  # noqa: E402
from src.eci.domain import MiceDomain  # noqa: E402

LEV = REPO / 'results/vision/eci_sae_levers/mice'
DEP = REPO / 'dataset/mice/v1/eci/codes/matryoshka_btk_1024_k16_fg448al_s0'
OOF = REPO / 'results/vision/eci_repr_diag/mice/a/oof_scores.parquet'
OUT = REPO / 'results/vision/eci_repr_diag/mice/video_level'
BEH = ('nose_nose', 'nose_tail')
DEPLOYED = {'nose_nose': 611, 'nose_tail': 414}
SWITCHES = {'social_H>O': (1, 2), 'social_O>P': (2, 3), 'fear_H>O': (4, 5), 'fear_O>P': (5, 6)}
CONFIGS = ('base', 'w4096', 'w4096+cell')
SEEDS = (0, 1, 2)
SELS = ('auc', 'ap', 'top1')
FPS = 5.0
R_MIN, DR_MIN, P_SIG = 0.7, 0.5, 0.05


def log(m):
    print(time.strftime('%H:%M:%S'), m, flush=True)


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return {'pearson': float('nan'), 'spearman': float('nan'), 'n': int(len(x))}
    return {'pearson': float(stats.pearsonr(x, y)[0]), 'spearman': float(stats.spearmanr(x, y)[0]), 'n': int(len(x))}


def ttest(d):
    d = np.asarray(d, float)
    t, p = stats.ttest_1samp(d, 0.0)
    return {'mean': float(d.mean()), 'dz': float(d.mean() / d.std(ddof=1)), 'p': float(p), 't': float(t), 'n': len(d)}


def bouts(y):
    """Number of maximal runs of True in a 1-d bool array (a run cut by the array start counts once)."""
    y = y.astype(np.int8)
    return int(y[0] + (np.diff(y) == 1).sum()) if len(y) else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, help='local staging dir (copies of the big inputs)')
    ap.add_argument('--out', default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stg = Path(args.stage)
    stg.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    # ---------------------------------------------------------------- stage big inputs once
    ann_src = MiceDomain.ann_path
    for src in (ann_src, DEP / 'codes_mean.npy', DEP / 'codes_max.npy'):
        dst = stg / src.name
        if not dst.exists():
            shutil.copyfile(src, dst)
    log(f'staged inputs in {time.time() - t0:.0f}s')
    dom = MiceDomain()
    design = C.load_design(stg / ann_src.name, MiceDomain.experiment_csv)  # observation_blocks checks contiguity/order
    design['wstart'] = dom.window_start(design['n_frames'].values, design['stage'].values)
    log(f'design: {len(design)} videos, {design.pool.nunique()} pools; window starts '
        f'{design.groupby("stage").wstart.agg(["min", "max"]).to_dict()}')
    ann = pd.read_csv(stg / ann_src.name, usecols=['observation_id', 'frame_idx', 'Y_nn', 'Y_np', 'Y_nt'])
    n_rows = len(ann)
    # per-row video index and in-window mask
    vid = np.repeat(np.arange(len(design)), design['n_frames'].values)
    off = np.arange(n_rows) - np.repeat(design['row_start'].values, design['n_frames'].values)
    inwin = off >= design['wstart'].values[vid]
    assert (ann['observation_id'].values == design['observation_id'].values[vid]).all()
    assert (off == ann['frame_idx'].values).all(), 'frame_idx is not the row offset within the video'
    # annotated videos: all three label columns present on every frame
    lab_ok = ann[['Y_nn', 'Y_np', 'Y_nt']].notna().all(axis=1).values
    v_ann = np.bincount(vid, weights=lab_ok, minlength=len(design)) == design['n_frames'].values
    v_any = np.bincount(vid, weights=lab_ok, minlength=len(design)) > 0
    assert (v_ann == v_any).all(), 'a video is only partly annotated'
    design['annotated'] = v_ann
    log(f'annotated videos {v_ann.sum()}, pools {design[v_ann].pool.nunique()} '
        f'(every annotated pool has all 6 stages: {(design[v_ann].groupby("pool").size() == 6).all()})')
    Y = {'nose_nose': (ann['Y_nn'].fillna(0).values > 0) | (ann['Y_np'].fillna(0).values > 0),
         'nose_tail': ann['Y_nt'].fillna(0).values > 0}
    # ---------------------------------------------------------------- ground truth per video (5 fps, window)
    gt = design[['observation_id', 'pool', 'stage', 'genotype', 'annotated']].copy()
    wlen = (design['n_frames'] - design['wstart']).values
    gt['win_frames'] = wlen
    for b in BEH:
        r, bo = np.full(len(design), np.nan), np.full(len(design), np.nan)
        for i in np.flatnonzero(v_ann):
            s, e = design.row_start.iat[i] + design.wstart.iat[i], design.row_end.iat[i]
            y = Y[b][s:e]
            r[i] = y.mean()
            bo[i] = bouts(y) / (len(y) / FPS / 60.0)
        gt[f'{b}_rate'], gt[f'{b}_bpm'] = r, bo
    # ---------------------------------------------------------------- eval frames (1 fps) of sae_levers
    lab = pd.read_parquet(LEV / 'labels.parquet')
    half_f = np.load(LEV / 'half.npy')
    rows = lab['row'].values
    assert len(rows) == len(half_f) == 172800
    assert (ann['observation_id'].values[rows] == lab['obs'].values).all()
    assert (ann['frame_idx'].values[rows] == lab['frame_idx'].values).all()
    for b in BEH:
        assert (Y[b][rows] == lab[b].values).all(), f'{b}: eval labels differ from annotations.csv'
    ev_vid, ev_in = vid[rows], inwin[rows]
    assert set(np.unique(ev_vid)) == set(np.flatnonzero(v_ann)), 'eval frames do not cover exactly the annotated videos'
    # half per video / pool
    vh = pd.Series(half_f).groupby(ev_vid).agg(['min', 'max'])
    assert (vh['min'] == vh['max']).all()
    gt['half'] = -1
    gt.loc[vh.index, 'half'] = vh['min'].values
    ph = gt[gt.annotated].groupby('pool').half.agg(['min', 'max'])
    assert (ph['min'] == ph['max']).all(), 'a pool is split over the halves'
    cover = {}
    for i in np.flatnonzero(v_ann):
        m = ev_vid == i
        fi = lab['frame_idx'].values[m]
        cover[design.observation_id.iat[i]] = (int(m.sum()), int(ev_in[m].sum()), int(fi.min()), int(fi.max()))
    cv = np.array(list(cover.values()))
    log(f'eval frames per video {np.unique(cv[:, 0]).tolist()}, in window {np.unique(cv[:, 1]).tolist()}, '
        f'frame_idx range [{cv[:, 2].min()}..{cv[:, 2].max()}, {cv[:, 3].min()}..{cv[:, 3].max()}]')
    for b in BEH:
        gt[f'{b}_rate_1fps'] = np.nan
        y1 = lab[b].values
        r1 = pd.Series(y1[ev_in]).groupby(ev_vid[ev_in]).mean()
        gt.loc[r1.index, f'{b}_rate_1fps'] = r1.values

    def vmean_eval(x):
        """per-video mean over the in-window eval (1 fps) frames -> (n_videos,) with NaN for unannotated."""
        o = np.full(len(design), np.nan)
        s = pd.Series(np.asarray(x, np.float64)[ev_in]).groupby(ev_vid[ev_in]).mean()
        o[s.index] = s.values
        return o

    # ---------------------------------------------------------------- candidate per-video means
    # cand[name][b] = {'per_half': {test_half: (n_videos,) means}} (half-specific neuron) or {'all': means}
    cand, meta = {}, {}
    for cfg in CONFIGS:
        for seed in SEEDS:
            key = f'{cfg}_s{seed}'
            ev = json.loads((LEV / 'eval' / f'{key}.json').read_text())
            z = np.load(LEV / 'codes_best' / f'{key}.npz')
            neurons, codes = list(z['neurons']), z['codes']
            assert codes.shape[0] == 172800
            for sel in SELS:
                name = f'{key}|{sel}'
                cand[name], meta[name] = {}, {'family': 'sae_levers', 'config': cfg, 'seed': seed, 'selection': sel}
                for b in BEH:
                    ph_ = {}
                    picks = {}
                    for d in ev['labels'][b]['dirs']:
                        j = d[sel]['neuron']
                        ph_[d['test_half']] = vmean_eval(codes[:, neurons.index(j)])
                        picks[f"select{d['select_half']}_test{d['test_half']}"] = {
                            'neuron': j, 'frame_test_auroc': d[sel]['test_auc'], 'frame_test_ap': d[sel]['test_ap']}
                    cand[name][b] = {'per_half': ph_}
                    meta[name][b] = picks
    log(f'sae_levers candidates: {len(cand)}')
    # deployed fg448al
    for pool_ in ('mean', 'max'):
        X = np.load(stg / f'codes_{pool_}.npy', mmap_mode='r')
        assert X.shape[0] == n_rows
        cols = [DEPLOYED[b] for b in BEH]
        D = np.empty((n_rows, len(cols)), np.float32)
        for a in range(0, n_rows, 216_000):
            D[a:a + 216_000] = X[a:a + 216_000][:, cols]
        del X
        for fr in ('5fps', '1fps'):
            name = f'deployed_{pool_}_{fr}'
            cand[name], meta[name] = {}, {'family': 'deployed', 'pooling': f'codes_{pool_}', 'frames': fr,
                                          'neurons': DEPLOYED}
            for k, b in enumerate(BEH):
                if fr == '1fps':
                    m = vmean_eval(D[rows, k])
                else:
                    w = inwin
                    s = pd.Series(D[w, k].astype(np.float64)).groupby(vid[w]).mean()
                    m = np.full(len(design), np.nan)
                    m[s.index] = s.values  # all 432 videos
                cand[name][b] = {'all': m}
        log(f'deployed codes_{pool_} read ({time.time() - t0:.0f}s)')
    # supervised probe (out-of-fold)
    oof = pd.read_parquet(OOF, columns=['probe', 'behaviour', 'seed', 'row', 'score'])
    for pr in ('mil_ctx', 'mean_lin'):
        name = f'probe_{pr}'
        cand[name], meta[name] = {}, {'family': 'probe', 'probe': pr}
        for b in BEH:
            o = oof[(oof.probe == pr) & (oof.behaviour == b)]
            if not len(o):
                continue
            p = o.assign(prob=1 / (1 + np.exp(-o.score.astype(np.float64)))).groupby('row').prob.mean()
            assert len(p) == 172800 and set(p.index) == set(rows)
            meta[name][b] = {'seeds': sorted(o.seed.unique().tolist())}
            cand[name][b] = {'all': vmean_eval(p.reindex(rows).values)}
    del oof
    log(f'candidates: {len(cand)} ({time.time() - t0:.0f}s)')

    # ---------------------------------------------------------------- measurements
    A = gt[gt.annotated]
    pools = sorted(A.pool.unique())
    pool_half = A.groupby('pool').half.first()
    idx = {(p, s): i for i, p, s in zip(A.index, A.pool, A.stage)}

    def deltas(vals, pool_list, key_vals=None):
        """{switch: (n_pools,) later - earlier} for a per-video array."""
        return {sw: np.array([vals[idx[(p, b_)]] - vals[idx[(p, a_)]] for p in pool_list])
                for sw, (a_, b_) in SWITCHES.items()}

    truth = {b: {q: gt[f'{b}_{q}'].values for q in ('rate', 'bpm', 'rate_1fps')} for b in BEH}
    res = {'truth': {}, 'candidates': {}}
    # annotated effects (all 24 pools) and their sanity numbers
    for b in BEH:
        tr = {}
        for q in ('rate', 'bpm'):
            dd = deltas(truth[b][q], pools)
            tr[q] = {sw: ttest(v) for sw, v in dd.items()}
        tr['rate_5fps_vs_1fps_video_r'] = corr(truth[b]['rate'][A.index], truth[b]['rate_1fps'][A.index])
        tr['rate_vs_bpm_video_r'] = corr(truth[b]['rate'][A.index], truth[b]['bpm'][A.index])
        tr['mean_rate'] = float(np.nanmean(truth[b]['rate'][A.index]))
        res['truth'][b] = tr

    def neuron_for_half(c, h):
        return c['per_half'][h] if 'per_half' in c else c['all']

    for name, cb in cand.items():
        R = {'meta': meta[name]}
        for b, c in cb.items():
            out_b = {'halves': {}}
            dn_all, dr_all, db_all = {sw: [] for sw in SWITCHES}, {sw: [] for sw in SWITCHES}, \
                {sw: [] for sw in SWITCHES}
            for h in (0, 1):
                x = neuron_for_half(c, h)
                vids_h = A.index[A.half == h]
                pl = [p for p in pools if pool_half[p] == h]
                e = {'video_rate': corr(x[vids_h], truth[b]['rate'][vids_h]),
                     'video_bpm': corr(x[vids_h], truth[b]['bpm'][vids_h]),
                     'video_rate_1fps': corr(x[vids_h], truth[b]['rate_1fps'][vids_h])}
                dn, dr, dbp = deltas(x, pl), deltas(truth[b]['rate'], pl), deltas(truth[b]['bpm'], pl)
                e['delta_rate'] = {sw: corr(dn[sw], dr[sw])['pearson'] for sw in SWITCHES}
                e['delta_bpm'] = {sw: corr(dn[sw], dbp[sw])['pearson'] for sw in SWITCHES}
                cat = lambda d_: np.concatenate([d_[sw] for sw in SWITCHES])  # noqa: E731
                e['delta_rate_pooled'] = corr(cat(dn), cat(dr))['pearson']
                e['delta_bpm_pooled'] = corr(cat(dn), cat(dbp))['pearson']
                # within-switch centred pooled (removes the between-switch mean differences)
                cen = lambda d_: np.concatenate([d_[sw] - d_[sw].mean() for sw in SWITCHES])  # noqa: E731
                e['delta_rate_pooled_centred'] = corr(cen(dn), cen(dr))['pearson']
                sdh = np.std(x[vids_h], ddof=1)
                for sw in SWITCHES:
                    dn_all[sw].append(dn[sw] / sdh)
                    dr_all[sw].append(dr[sw])
                    db_all[sw].append(dbp[sw])
                out_b['halves'][h] = e
            H = out_b['halves']
            mh = lambda f: float(np.mean([f(H[h]) for h in (0, 1)]))  # noqa: E731
            out_b['video_r_rate'] = mh(lambda e: e['video_rate']['pearson'])
            out_b['video_rho_rate'] = mh(lambda e: e['video_rate']['spearman'])
            out_b['video_r_bpm'] = mh(lambda e: e['video_bpm']['pearson'])
            out_b['video_rho_bpm'] = mh(lambda e: e['video_bpm']['spearman'])
            out_b['delta_r_pooled'] = mh(lambda e: e['delta_rate_pooled'])
            out_b['delta_r_pooled_bpm'] = mh(lambda e: e['delta_bpm_pooled'])
            out_b['delta_r_pooled_centred'] = mh(lambda e: e['delta_rate_pooled_centred'])
            out_b['delta_r_switch'] = {sw: mh(lambda e: e['delta_rate'][sw]) for sw in SWITCHES}
            # effect recovery over all 24 held-out pools
            eff = {}
            for sw in SWITCHES:
                n_ = ttest(np.concatenate(dn_all[sw]))
                r_ = ttest(np.concatenate(dr_all[sw]))
                eff[sw] = {'neuron': n_, 'rate': r_, 'sign_agree': bool(np.sign(n_['mean']) == np.sign(r_['mean'])),
                           'neuron_sig_same_sign': bool(n_['p'] < P_SIG and np.sign(n_['mean']) == np.sign(r_['mean']))}
            out_b['effects'] = eff
            sig = [sw for sw in SWITCHES if res['truth'][b]['rate'][sw]['p'] < P_SIG]
            ok_sign = all(eff[sw]['sign_agree'] for sw in sig)
            out_b['verdict'] = {'video_r_ok': out_b['video_r_rate'] >= R_MIN, 'delta_r_ok': out_b['delta_r_pooled'] >= DR_MIN,
                                'signs_ok': ok_sign, 'significant_switches': sig,
                                'n_sig_reproduced_significantly': int(sum(eff[sw]['neuron_sig_same_sign'] for sw in sig))}
            out_b['verdict']['useful'] = bool(out_b['verdict']['video_r_ok'] and out_b['verdict']['delta_r_ok'] and ok_sign)
            # 72-pool / 48-unannotated-pool neuron-only test (deployed: codes for every video)
            if 'all' in c and np.isfinite(c['all']).sum() == len(design):
                x = c['all']
                ix = {(p, s): i for i, p, s in zip(design.index, design.pool, design.stage)}
                allp = sorted(design.pool.unique())
                unann = [p for p in allp if p not in set(pools)]
                for lbl, pl in (('pools72', allp), ('pools48_unannotated', unann)):
                    out_b[lbl] = {sw: ttest([x[ix[(p, b_)]] - x[ix[(p, a_)]] for p in pl])
                                  for sw, (a_, b_) in SWITCHES.items()}
            R[b] = out_b
        res['candidates'][name] = R
    res['notes'] = {
        'window': 'MiceDomain.window_start: habituation from frame n_frames-4500 (last 15 min), odour/post whole',
        'bout_rule': 'maximal run of consecutive positive 5 fps frames, no gap merging; per window minute',
        'combine_halves': "effect tests pool both halves' held-out pools; each half's neuron means divided by their "
                          "sd over that half's 72 videos (label-free) before the paired differences are pooled",
        'deployed': 'fixed neurons 611/414 from an in-sample audit, not cross-fitted; reported on each half',
        'probe': 'out-of-fold (5 pool-grouped folds), sigmoid(logit) averaged over seeds',
        'eval_coverage': {'eval_frames_per_video': np.unique(cv[:, 0]).tolist(),
                          'in_window_eval_frames_per_video': np.unique(cv[:, 1]).tolist()},
        'n_unannotated_codes_sae_levers': 'none: codes_best holds only the 172,800 eval frames of the 144 annotated videos',
        'wall_s': round(time.time() - t0, 1)}
    (out / 'results.json').write_text(json.dumps(res, indent=1, default=float))
    pv = gt.copy()
    for name, cb in cand.items():
        for b, c in cb.items():
            if 'all' in c:
                pv[f'{name}|{b}'] = c['all']
            else:
                for h, x in c['per_half'].items():
                    pv[f'{name}|{b}|test{h}'] = x
    pv.to_parquet(out / 'per_video.parquet')
    log(f'wrote {out}/results.json, per_video.parquet ({time.time() - t0:.0f}s)')
    shutil.rmtree(stg, ignore_errors=True)


def fmt_p(p):
    return f'{p:.0e}' if p < 1e-3 else f'{p:.3f}'


def summary(out):
    """OUT/results.json -> OUT/table.md (login-node safe)."""
    res = json.loads((out / 'results.json').read_text())
    C_ = res['candidates']
    sw = list(SWITCHES)
    L = []
    for b in BEH:
        tr = res['truth'][b]
        L.append(f'\n## {b} (mean annotated rate {tr["mean_rate"]:.4f}; video r rate 5fps vs 1fps '
                 f'{tr["rate_5fps_vs_1fps_video_r"]["pearson"]:.3f}, rate vs bouts/min {tr["rate_vs_bpm_video_r"]["pearson"]:.3f})\n')
        L.append('| candidate | video r rate (h0/h1) | rho rate | r bouts | rho bouts | pooled d-r (centred) | '
                 + ' | '.join(f'{s} dz (p)' for s in sw) + ' | verdict |')
        L.append('|' + '---|' * (7 + len(sw)))
        L.append('| ANNOTATED rate | | | | | | ' + ' | '.join(
            f'{tr["rate"][s]["dz"]:+.2f} ({fmt_p(tr["rate"][s]["p"])})' for s in sw) + ' | |')
        L.append('| ANNOTATED bouts/min | | | | | | ' + ' | '.join(
            f'{tr["bpm"][s]["dz"]:+.2f} ({fmt_p(tr["bpm"][s]["p"])})' for s in sw) + ' | |')
        names = [f'{c}_s{s}|auc' for c in CONFIGS for s in SEEDS] + ['deployed_mean_5fps', 'deployed_max_1fps',
                                                                    'deployed_mean_1fps', 'deployed_max_5fps',
                                                                    'probe_mil_ctx', 'probe_mean_lin']
        for n in names:
            if n not in C_ or b not in C_[n]:
                continue
            e = C_[n][b]
            H = e['halves']
            v = e['verdict']
            lab_ = n
            if '|' in n:
                pk = C_[n]['meta'][b]
                lab_ = n.split('|')[0] + ' n' + '/'.join(str(pk[k]['neuron']) for k in sorted(pk))
            cells = []
            for s in sw:
                f = e['effects'][s]
                mk = ('=' if f['sign_agree'] else 'X') + ('*' if f['neuron']['p'] < P_SIG else '')
                cells.append(f'{f["neuron"]["dz"]:+.2f} ({fmt_p(f["neuron"]["p"])}) {mk}')
            L.append(f'| {lab_} | {e["video_r_rate"]:.2f} ({H["0"]["video_rate"]["pearson"]:.2f}/'
                     f'{H["1"]["video_rate"]["pearson"]:.2f}) | {e["video_rho_rate"]:.2f} | {e["video_r_bpm"]:.2f} | '
                     f'{e["video_rho_bpm"]:.2f} | {e["delta_r_pooled"]:.2f} ({e["delta_r_pooled_centred"]:.2f}) | '
                     + ' | '.join(cells) + f' | {"USEFUL" if v["useful"] else "no"} '
                     f'(r{"+" if v["video_r_ok"] else "-"} d{"+" if v["delta_r_ok"] else "-"} '
                     f's{"+" if v["signs_ok"] else "-"}) |')
        L.append('\nPer-switch held-out d-correlation (mean of halves), AUROC picks / references:\n')
        L.append('| candidate | ' + ' | '.join(sw) + ' |')
        L.append('|' + '---|' * (1 + len(sw)))
        for n in names:
            if n in C_ and b in C_[n]:
                L.append(f'| {n.split("|")[0]} | ' + ' | '.join(f'{C_[n][b]["delta_r_switch"][s]:.2f}' for s in sw) + ' |')
        L.append('\nAll selection rules, seed mean (sd) of the held-out numbers; USEFUL count over seeds:\n')
        L.append('| config | rule | video r rate | r bouts | pooled d-r | useful seeds |')
        L.append('|---|---|---|---|---|---|')
        for c in CONFIGS:
            for sel in SELS:
                es = [C_[f'{c}_s{s}|{sel}'][b] for s in SEEDS]
                ms = lambda k: f'{np.mean([e[k] for e in es]):.2f} ({np.std([e[k] for e in es]):.2f})'  # noqa: E731
                L.append(f'| {c} | {sel} | {ms("video_r_rate")} | {ms("video_r_bpm")} | {ms("delta_r_pooled")} | '
                         f'{sum(e["verdict"]["useful"] for e in es)}/3 |')
        for n in ('deployed_mean_5fps', 'deployed_max_5fps'):
            if 'pools72' in C_[n][b]:
                L.append(f'\n{n} neuron-only effect tests, 72 pools / 48 unannotated pools: ' + '; '.join(
                    f'{s} {C_[n][b]["pools72"][s]["dz"]:+.2f} ({fmt_p(C_[n][b]["pools72"][s]["p"])}) / '
                    f'{C_[n][b]["pools48_unannotated"][s]["dz"]:+.2f} ({fmt_p(C_[n][b]["pools48_unannotated"][s]["p"])})'
                    for s in sw))
    L.insert(0, '# Video-level NES usefulness, mice v1 (scripts/eci/diag_video_level.py)\n\n'
             'Held out = neuron chosen on the other cross-fit half (12 pools / 72 videos per half). Effect columns: '
             'dz = mean/sd of the per-pool change over all 24 held-out pools, p = one-sample t-test; '
             '"=" sign agrees with the annotated rate change, "X" disagrees, "*" neuron p < 0.05. '
             f'Verdict: USEFUL iff video r >= {R_MIN}, pooled d-r >= {DR_MIN}, signs of all annotated p < 0.05 switches '
             'reproduced (r/d/s flags). Deployed 611/414 are NOT cross-fitted.')
    (out / 'table.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


def ols_intercept(y, x):
    """y = a + b x; returns a / sd(resid), p(a = 0), b, R^2."""
    X = np.c_[np.ones(len(x)), x]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ beta
    s2 = r @ r / (len(y) - 2)
    se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[0, 0])
    t = beta[0] / se
    return {'a_over_sd': float(beta[0] / np.sqrt(s2)), 'p_a': float(2 * stats.t.sf(abs(t), len(y) - 2)),
            'b': float(beta[1]), 'r2': float(1 - (r @ r) / ((y - y.mean()) @ (y - y.mean())))}


def extras(out):
    """Follow-up checks on OUT/per_video.parquet (login-node safe) -> OUT/extras.json, printed.
    (1) the neuron's switch change NOT explained by the annotated change: per switch, over the 24 held-out pools,
        d_neuron = a + b d_truth (truth = rate or bouts/min); a != 0 means the neuron moves at the switch beyond what
        the annotated behaviour change predicts. (2) sign agreement with the significant annotated BOUTS/MIN effects
        (the outcome of the mice phase-ATE work). (3) probes: the per-video mean with the per-fold mean removed
        (out-of-fold scores come from 5 different fold models; label-free). (4) video r over all 144 videos."""
    pv = pd.read_parquet(out / 'per_video.parquet')
    res = json.loads((out / 'results.json').read_text())
    A = pv[pv.annotated].copy()
    pools = sorted(A.pool.unique())
    folds = json.loads((REPO / 'results/vision/eci_repr_diag/mice/a/results.json').read_text())['folds']
    fold_of = {p: k for k, f in enumerate(folds) for p in f}
    A['fold'] = A.pool.map(fold_of)
    assert A.fold.notna().all()
    names = [f'{c}_s{s}|auc' for c in CONFIGS for s in SEEDS] + ['deployed_mean_5fps', 'deployed_max_1fps',
                                                                'probe_mil_ctx', 'probe_mean_lin']
    E = {}
    for n in names:
        E[n] = {}
        for b in BEH:
            x = np.full(len(A), np.nan)
            for h in (0, 1):
                col = f'{n}|{b}|test{h}' if f'{n}|{b}|test{h}' in A else f'{n}|{b}'
                m = (A.half == h).values
                x[m] = A[col].values[m] / np.std(A[col].values[m], ddof=1)
            A['_x'] = x
            P = A.pivot(index='pool', columns='stage', values=['_x', f'{b}_rate', f'{b}_bpm']).loc[pools]
            e = {'video_r_all144_rate': corr(x, A[f'{b}_rate'])['pearson'], 'switch': {}}
            for sw, (a_, b_) in SWITCHES.items():
                dx = (P['_x'][b_] - P['_x'][a_]).values
                f = {}
                for q in ('rate', 'bpm'):
                    dt = (P[f'{b}_{q}'][b_] - P[f'{b}_{q}'][a_]).values
                    f[f'resid_on_{q}'] = ols_intercept(dx, dt / dt.std(ddof=1))
                tb = res['truth'][b]['bpm'][sw]
                f['bpm_sig'] = tb['p'] < P_SIG
                f['sign_agree_bpm'] = bool(np.sign(dx.mean()) == np.sign(tb['mean']))
                e['switch'][sw] = f
            e['signs_ok_bpm'] = all(f['sign_agree_bpm'] for f in e['switch'].values() if f['bpm_sig'])
            if n.startswith('probe'):
                raw = A[f'{n}|{b}'].values
                fc = raw - A.groupby('fold')[f'{n}|{b}'].transform('mean').values
                e['fold_centred_video_r_rate'] = {h: corr(fc[(A.half == h).values], A[f'{b}_rate'].values[(A.half == h).values])
                                                  ['pearson'] for h in (0, 1)}
                e['fold_centred_video_r_rate']['all144'] = corr(fc, A[f'{b}_rate'])['pearson']
                e['logit_free_note'] = 'mean of sigmoid(logit) over frames, seeds averaged'
            E[n][b] = e
    (out / 'extras.json').write_text(json.dumps(E, indent=1, default=float))
    for b in BEH:
        print(f'\n## {b}: change beyond the annotated change (a/sd, p) per switch, regressed on rate | on bouts/min;'
              ' video r all 144; bouts sign check')
        for n in names:
            e = E[n][b]
            s = ' | '.join(f"{sw} {f['resid_on_rate']['a_over_sd']:+.2f} ({fmt_p(f['resid_on_rate']['p_a'])}) | "
                           f"{f['resid_on_bpm']['a_over_sd']:+.2f} ({fmt_p(f['resid_on_bpm']['p_a'])})"
                           for sw, f in e['switch'].items())
            extra = f" fold-centred r {e['fold_centred_video_r_rate']}" if 'fold_centred_video_r_rate' in e else ''
            print(f"{n.split('|')[0]}: r144 {e['video_r_all144_rate']:.2f} bouts-signs {'ok' if e['signs_ok_bpm'] else 'NO'}"
                  f" || {s}{extra}")


if __name__ == '__main__':
    if '--extras' in sys.argv:
        extras(OUT)
    elif '--summary' in sys.argv:
        summary(Path(sys.argv[sys.argv.index('--summary') + 1]) if len(sys.argv) > sys.argv.index('--summary') + 1
                else OUT)
    else:
        main()
