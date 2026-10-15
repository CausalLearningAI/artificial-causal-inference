"""
MABe22 mouse ECI evaluation, step 1 (CPU Slurm job): label checks, frame offset, pixel background, frame plans.

    1 labels      vocabulary / row order of mouse_submission_labels.npy, frame map, per-row stats; a SEMANTIC row check
                  from the submission keypoints (nose-nose, nose-tailbase, nose-ear and centroid distances of the
                  positive frames of every behaviour row), so a name / row shift would show.
    2 offset      which video frame (of 1,801) holds label frame i: mean grey level under the keypoints for video frame
                  i + d, d in -2..2, both (x, y) orders (mice are black on a bright floor -> the right d is darkest).
    3 pixel bg    per-clip per-pixel 0.98 quantile of 30 evenly spaced grey frames (the lab's pix_bg) for every clip,
                  the GLOBAL background = per-pixel median over the userTrain clips, and per-clip agreement stats.
    4 plans       train pool (userTrain, label-free): 13 train + 3 calibration frames per clip on the 5 fps grid (rng 0);
                  eval (submission, stratified, inverse-probability weights): see make_eval().
Outputs: OUT/plan/{labels_check.json, offset.json, pixbg_global.npy, pixbg_stats.json, train_frames.parquet,
         eval_frames.parquet, eval_plan.json}
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eci_common import (BEHAVIOURS, EXCLUDE, GRID_I, LAG, PLAN, RAW, VID, decode_clip, log, stage_dir)  # noqa: E402

KP = {'nose': 0, 'ear_l': 1, 'ear_r': 2, 'tail_base': 9}  # assumed MABe22 mouse order (checked semantically below)


def load_labels():
    sub = np.load(RAW / 'mouse_submission_labels.npy', allow_pickle=True).item()
    fm = np.load(RAW / 'mouse_frame_number_map.npy', allow_pickle=True).item()
    return sub, fm


def kp_distances(kp):
    """kp (T, 3, 12, 2) -> dict of (T,) min over ordered mouse pairs of nose-nose, nose-tailbase, nose-ear, centroid."""
    kp = kp.astype(np.float32)
    n, e1, e2, tb = (kp[:, :, KP[k]] for k in ('nose', 'ear_l', 'ear_r', 'tail_base'))
    cen = kp.mean(2)
    out = {'nose_nose': [], 'nose_tailbase': [], 'nose_ear': [], 'centroid': []}
    for a in range(3):
        for b in range(3):
            if a == b:
                continue
            d = lambda p, q: np.linalg.norm(p[:, a] - q[:, b], axis=1)  # noqa: E731
            if a < b:
                out['nose_nose'].append(d(n, n))
                out['centroid'].append(d(cen, cen))
            out['nose_tailbase'].append(d(n, tb))
            out['nose_ear'].append(np.minimum(d(n, e1), d(n, e2)))
    return {k: np.min(np.stack(v), 0) for k, v in out.items()}


def cmd_labels(sub, fm, sk):
    vocab = list(sub['vocabulary'])
    L = sub['label_array']
    info = {'vocabulary_in_file': vocab, 'task_type': list(sub.get('task_type', [])), 'label_array_shape': list(L.shape),
            'label_array_dtype': str(L.dtype), 'n_clips_fmap': len(fm)}
    std = np.load(RAW / 'mouse_frame_number_map.npy', allow_pickle=True).item()
    info['fmap_equals_labels_file_fmap'] = bool(std == sub['frame_number_map'])
    lens = {k: v[1] - v[0] for k, v in fm.items()}
    info['clip_len_values'] = sorted(set(lens.values()))
    starts = sorted(v[0] for v in fm.values())
    info['fmap_contiguous'] = bool(starts == list(range(0, 1800 * len(fm), 1800)))
    rows = []
    for r in range(L.shape[0]):
        v = L[r]
        fin = np.isfinite(v)
        u = np.unique(v[fin])
        rows.append({'row': r, 'name': vocab[r] if r < len(vocab) else None, 'n_unique': int(len(u)),
                     'binary': bool(set(u.tolist()) <= {0.0, 1.0}), 'pos_rate': float(np.mean(v[fin] == 1)),
                     'nan_frames': int((~fin).sum())})
    info['rows'] = rows
    # semantic check from keypoints: per row, mean distances over positive frames (all submission clips with keypoints)
    clips = sorted(fm)
    D = {k: np.full(L.shape[1], np.nan, np.float32) for k in ('nose_nose', 'nose_tailbase', 'nose_ear', 'centroid')}
    for c in clips:
        a, b = fm[c]
        d = kp_distances(sk[c]['keypoints'])
        for k in D:
            D[k][a:b] = d[k]
    sem = {'all_frames': {k: float(np.nanmean(v)) for k, v in D.items()}}
    for r in rows:
        if not r['binary'] or r['pos_rate'] in (0.0, 1.0):
            continue
        pos = L[r['row']] == 1
        sem[r['name']] = {k: float(np.nanmean(v[pos])) for k, v in D.items()}
        sem[r['name']]['n_pos'] = int(pos.sum())
    info['semantic_kp_distance_means_px'] = sem
    info['kp_order_assumed'] = KP
    return info


def _offset_one(job):
    set_, clip, kp, stage = job
    g = decode_clip(set_, clip, stage, 'gray').astype(np.float32)
    T = kp.shape[0]
    res = {}
    for order in (0, 1, 2, 3):  # 0 / 1 = (x, y) / (y, x) on the 224 scale; 2 / 3 = the same on a 512 scale
        sc = 1.0 if order < 2 else 224 / 512
        x = kp[..., order % 2].astype(np.float32) * sc
        y = kp[..., 1 - order % 2].astype(np.float32) * sc
        xi, yi = np.clip(np.round(x).astype(int), 0, 223), np.clip(np.round(y).astype(int), 0, 223)
        for d in range(-2, 3):
            i = np.arange(T)
            ok = (i + d >= 0) & (i + d < len(g))
            v = g[(i + d)[ok][:, None, None], yi[ok], xi[ok]]
            res[f'order{order}_d{d}'] = float(v.mean())
    # dark-blob centroid (pixels < 60 grey and > 40 below the clip's 0.98-quantile background) vs keypoint centroid
    bg = np.sort(g[np.linspace(0, len(g) - 1, 30).round().astype(int)], 0)[28]
    dk = (g < 60) & (bg[None] - g > 40)
    cnt = dk.sum((1, 2)).clip(1)
    yy, xx = np.mgrid[0:224, 0:224]
    bx, by = (dk * xx).sum((1, 2)) / cnt, (dk * yy).sum((1, 2)) / cnt
    for order in (0, 1):
        kx, ky = kp[..., order].reshape(T, -1).mean(1), kp[..., 1 - order].reshape(T, -1).mean(1)
        for d in range(-2, 3):
            i = np.arange(T)
            ok = (i + d >= 0) & (i + d < len(g)) & (cnt[np.clip(i + d, 0, len(g) - 1)] > 50)
            # correlation of the frame-to-frame MOTION (first differences), sensitive to a 1-frame shift
            dx = np.diff(bx[(i + d)[ok]]), np.diff(kx[ok])
            dy = np.diff(by[(i + d)[ok]]), np.diff(ky[ok])
            res[f'corr_motion_order{order}_d{d}'] = float(np.corrcoef(*dx)[0, 1] + np.corrcoef(*dy)[0, 1]) / 2
    res['n_video_frames'] = int(len(g))
    res['floor_mean'] = float(g.mean())
    return clip, res


def cmd_offset(sub, fm, sk, ut, stage, workers):
    rng = np.random.default_rng(0)
    clips = sorted(c for c in fm if c not in EXCLUDE)
    pick = rng.choice(clips, 24, replace=False)
    jobs = [('mouse_submission', c, sk[c]['keypoints'], str(stage)) for c in pick]
    ucl = sorted(ut['sequences'])
    jobs += [('mouse_userTrain', c, ut['sequences'][c]['keypoints'], str(stage)) for c in rng.choice(ucl, 8, replace=False)]
    with Pool(workers) as p:
        R = dict(p.map(_offset_one, jobs))
    keys = [k for k in next(iter(R.values())) if k.startswith('order')]
    ckeys = [k for k in next(iter(R.values())) if k.startswith('corr_motion')]
    sub_m = {k: float(np.mean([R[c][k] for c in pick])) for k in keys}
    ut_m = {k: float(np.mean([R[c][k] for c in R if c not in pick])) for k in keys}
    best = min(sub_m, key=sub_m.get)
    order, d = int(best[5]), int(best.split('_d')[1])
    # second check: per-clip correlation over frames of the keypoint centroid of all 3 mice with the dark-pixel
    # centroid of the frame, for each offset (the coordinator's suggested test)
    # per-clip agreement: fraction of submission clips whose own darkest d equals the global best
    agree = float(np.mean([min((k for k in keys if k.startswith(f'order{order}')), key=R[c].get) == best for c in pick]))
    corr_m = {k: float(np.nanmean([R[c][k] for c in pick])) for k in ckeys}
    corr_best = max(corr_m, key=corr_m.get)
    info = {'corr_motion_submission': corr_m, 'corr_best': corr_best,
            'corr_best_offset_agrees': int(corr_best.split('_d')[1]) == d,
            'offset': d, 'xy_order': order, 'rule': 'video frame = label index + offset', 'mean_grey_under_kp_submission':
            sub_m, 'mean_grey_under_kp_usertrain': ut_m, 'best_usertrain': min(ut_m, key=ut_m.get),
            'frac_submission_clips_agreeing': agree, 'n_video_frames': sorted({R[c]['n_video_frames'] for c in R}),
            'clips': list(map(str, pick))}
    return info


def _pixbg_one(job):
    set_, clip, stage = job
    g = decode_clip(set_, clip, stage, 'gray')
    idx = np.linspace(0, len(g) - 1, 30).round().astype(int)
    s = np.sort(g[idx], 0)
    k = int(round(0.98 * 30)) - 1  # = the kthvalue(round(q * n)) of foreground.pixel_background
    return clip, s[k]


def cmd_pixbg(fm, ut, stage, workers):
    jobs = [('mouse_userTrain', c, str(stage)) for c in sorted(ut['sequences'])]
    jobs += [('mouse_submission', c, str(stage)) for c in sorted(fm) if c not in EXCLUDE]
    with Pool(workers) as p:
        B = dict(p.map(_pixbg_one, jobs, chunksize=4))
    utc = sorted(ut['sequences'])
    glob = np.median(np.stack([B[c] for c in utc]), 0).round().astype(np.uint8)
    np.save(PLAN / 'pixbg_global.npy', glob)
    np.savez_compressed(PLAN / 'pixbg_clips.npz', clips=np.array(list(B)), bg=np.stack(list(B.values())))
    st = {}
    for c, b in B.items():
        diff = b.astype(int) - glob.astype(int)
        st[c] = {'frac_absdiff_gt40': float((np.abs(diff) > 40).mean()),
                 'frac_clip_darker_by40_and_dark': float(((diff < -40) & (b < 60)).mean()),
                 'corr': float(np.corrcoef(b.ravel(), glob.ravel())[0, 1])}
    v = np.array([s['frac_absdiff_gt40'] for s in st.values()])
    info = {'n_clips': len(st), 'global_from': 'median over userTrain clips of the per-clip 0.98-quantile (30 frames)',
            'frac_absdiff_gt40_quantiles': np.quantile(v, [0, .5, .9, .99, 1]).round(4).tolist(),
            'corr_quantiles': np.quantile([s['corr'] for s in st.values()], [0, .01, .1, .5]).round(4).tolist(),
            'clip_darker_dark_quantiles': np.quantile([s['frac_clip_darker_by40_and_dark'] for s in st.values()],
                                                      [.5, .9, .99, 1]).round(4).tolist(),
            'global_mean_grey': float(glob.mean()), 'per_clip': st}
    return info


def make_train(ut):
    rng = np.random.default_rng(0)
    rows = []
    for c in sorted(ut['sequences']):
        i = np.sort(rng.choice(GRID_I, 16, replace=False))
        role = np.array(['train'] * 13 + ['calib'] * 3)
        rng.shuffle(role)
        rows += [{'clip': c, 'i': int(a), 'role': r} for a, r in zip(i, role)]
    return pd.DataFrame(rows)


def make_eval(sub, fm):
    """Stratified sample of the 5 fps grid of every submission clip with video (inclusion probabilities known):
        S_oral  grid frames with an oral_* positive within +-6 raw frames (0.2 s): ALL of them (cap 7,000)
        S_aw    not S_oral, an approaches / watching / chases positive within +-6 raw frames: 2,500 uniform
        S_hud   neither, huddles positive at the frame: 1,500 uniform
        S_rest  the rest: 10,000 uniform
    weight w = |S| / n_S (Horvitz-Thompson); every metric uses w."""
    vocab = list(sub['vocabulary'])
    L = sub['label_array']
    R = {b: vocab.index(b) for b in BEHAVIOURS}
    clips = sorted(c for c in fm if c not in EXCLUDE)
    rows = []
    for c in clips:
        a, b = fm[c]
        lab = {k: L[r, a:b] == 1 for k, r in R.items()}
        near = lambda m: np.convolve(m.astype(float), np.ones(13), 'same')[GRID_I] > 0  # noqa: E731
        oral = near(lab['oral_oral_contact'] | lab['oral_genital_contact'] | lab['oral_ear_contact'])
        aw = near(lab['approaches'] | lab['watching'] | lab['chases'])
        hud = lab['huddles'][GRID_I]
        s = np.where(oral, 'oral', np.where(aw, 'aw', np.where(hud, 'hud', 'rest')))
        for j, i in enumerate(GRID_I):
            rows.append({'clip': c, 'i': int(i), 'stratum': s[j], **{k: bool(v[i]) for k, v in lab.items()}})
    U = pd.DataFrame(rows)
    want = {'oral': 7000, 'aw': 2500, 'hud': 1500, 'rest': 10000}
    rng = np.random.default_rng(0)
    parts, info = [], {'population': 'submission clips with video x label index 30..1794 step 6', 'N_population': len(U),
                       'strata': {}}
    for s, n in want.items():
        P = U[U.stratum == s]
        n = min(n, len(P))
        take = P.iloc[np.sort(rng.choice(len(P), n, replace=False))].copy()
        take['w'] = len(P) / n
        parts.append(take)
        info['strata'][s] = {'N': int(len(P)), 'n': int(n), 'w': float(len(P) / n)}
    E = pd.concat(parts).sort_values(['clip', 'i']).reset_index(drop=True)
    info['n_frames'] = len(E)
    info['n_clips'] = int(E['clip'].nunique())
    info['weighted_base_rate'] = {b: float(np.average(E[b], weights=E.w)) for b in BEHAVIOURS}
    info['population_base_rate'] = {b: float(U[b].mean()) for b in BEHAVIOURS}
    info['raw_base_rate_in_sample'] = {b: float(E[b].mean()) for b in BEHAVIOURS}
    info['n_pos'] = {b: int(E[b].sum()) for b in BEHAVIOURS}
    info['n_clips_with_pos'] = {b: int(E.groupby('clip')[b].any().sum()) for b in BEHAVIOURS}
    # clips with an oral positive anywhere in the labels vs clips contributing an oral positive / an S_oral frame
    any_oral = set()
    for c in clips:
        a, b = fm[c]
        if ((L[R['oral_oral_contact'], a:b] == 1) | (L[R['oral_genital_contact'], a:b] == 1)
                | (L[R['oral_ear_contact'], a:b] == 1)).any():
            any_oral.add(c)
    info['clips_with_oral_pos_in_labels'] = len(any_oral)
    info['of_which_contribute_S_oral_frames'] = int(len(any_oral & set(E[E.stratum == 'oral']['clip'])))
    return E, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=16)
    args = ap.parse_args()
    t0 = time.time()
    PLAN.mkdir(parents=True, exist_ok=True)
    stage = stage_dir()
    sub, fm = load_labels()
    sk = np.load(RAW / 'mouse_submission_keypoints.npy', allow_pickle=True).item()['sequences']
    ut = np.load(RAW / 'mouse_user_train.npy', allow_pickle=True).item()
    log(f'loaded labels {sub["label_array"].shape}, {len(sk)} kp clips, {len(ut["sequences"])} userTrain clips')
    if (PLAN / 'pixbg_stats.json').exists() and (PLAN / 'offset.json').exists():
        log('[SKIP] labels / offset / pixbg (done)')
        return finish(sub, fm, ut, t0)
    lab = cmd_labels(sub, fm, sk)
    (PLAN / 'labels_check.json').write_text(json.dumps(lab, indent=1))
    log('labels: ' + json.dumps({k: v for k, v in lab.items() if k != 'rows'})[:3000])
    for r in lab['rows']:
        log(f'  row {r}')
    off = cmd_offset(sub, fm, sk, ut, stage, args.workers)
    (PLAN / 'offset.json').write_text(json.dumps(off, indent=1))
    log('offset: ' + json.dumps(off))
    pb = cmd_pixbg(fm, ut, stage, args.workers)
    (PLAN / 'pixbg_stats.json').write_text(json.dumps(pb, indent=1))
    log('pixbg: ' + json.dumps({k: v for k, v in pb.items() if k != 'per_clip'}))
    finish(sub, fm, ut, t0)


def _offset_fast(job):
    """grey under the keypoints of MOVING mice only (keypoint centroid speed > 2 px / frame at 224, kp at 512 scale,
    (y, x) order = the order found by cmd_offset), offsets -2..2."""
    set_, clip, kp, stage = job
    g = decode_clip(set_, clip, stage, 'gray').astype(np.float32)
    k = kp.astype(np.float32) * 224 / 512
    x, y = k[..., 1], k[..., 0]  # (T, 3, 12)
    cx, cy = x.mean(2), y.mean(2)
    sp = np.zeros_like(cx)
    sp[1:-1] = np.hypot(cx[2:] - cx[:-2], cy[2:] - cy[:-2]) / 2
    xi, yi = np.clip(np.round(x).astype(int), 0, 223), np.clip(np.round(y).astype(int), 0, 223)
    T = len(k)
    out = {}
    for d in range(-2, 3):
        vals = []
        for t, m in zip(*np.nonzero(sp > 2)):
            if 0 <= t + d < len(g):
                vals.append(g[t + d, yi[t, m], xi[t, m]].mean())
        out[d] = (float(np.mean(vals)) if vals else float('nan'), len(vals))
    return clip, out


def finish(sub, fm, ut, t0):
    offj = json.loads((PLAN / 'offset.json').read_text())
    if 'fast_motion' not in offj:
        sk = np.load(RAW / 'mouse_submission_keypoints.npy', allow_pickle=True).item()['sequences']
        stage = stage_dir()
        jobs = [('mouse_submission', c, sk[c]['keypoints'], str(stage)) for c in offj['clips']]
        with Pool(16) as p:
            R = dict(p.map(_offset_fast, jobs))
        m = {d: float(np.nanmean([R[c][d][0] for c in R])) for d in range(-2, 3)}
        best = [min(range(-2, 3), key=lambda d: R[c][d][0]) for c in R]
        offj['fast_motion'] = {'mean_grey_under_moving_mouse_kp': m, 'n_mouse_frames': sum(R[c][0][1] for c in R),
                               'per_clip_best_offset_counts': {d: best.count(d) for d in range(-2, 3)},
                               'best': min(m, key=m.get)}
        (PLAN / 'offset.json').write_text(json.dumps(offj, indent=1))
        log('offset fast-motion check: ' + json.dumps(offj['fast_motion']))
    T = make_train(ut)
    T.to_parquet(PLAN / 'train_frames.parquet')
    log(f'train plan: {T.role.value_counts().to_dict()}, {T["clip"].nunique()} clips')
    E, info = make_eval(sub, fm)
    E.to_parquet(PLAN / 'eval_frames.parquet')
    (PLAN / 'eval_plan.json').write_text(json.dumps(info, indent=1))
    log('eval plan: ' + json.dumps(info))
    log(f'done in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
