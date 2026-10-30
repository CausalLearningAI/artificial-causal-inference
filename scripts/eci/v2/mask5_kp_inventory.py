"""
Keypoint / track inventory of the mouse benchmarks for a keypoint-guided store mask (CPU, Slurm defaultp).
Over the 5 fps store grid of EVERY clip that goes into the benchmark store (bench_store/size.json rule: video frame k
with k % step == 0): coverage (all parts / every animal present), missing rate per part, CalMS21 confidence scores,
jitter (frame-to-frame displacement at the source frame rate, native px) and an identity-swap proxy.
Swap proxy: at source frame t, the animal assignment that minimises the summed centroid displacement from t-1 is not
the identity (and beats it by > 50%). -> results/vision/eci_v2_0B/mask_approval_v3/kp_inventory.json
"""
import json
import sys
from itertools import permutations
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'scripts/eci/v2_mask2'))
sys.path.insert(0, str(REPO / 'scripts/benchmark/crim13'))
OUT = REPO / 'results/vision/eci_v2_0B/mask_approval_v3'


def swap_stats(c):
    """c (T, A, 2) centroids (NaN allowed) -> (frames evaluated, swap frames)."""
    A = c.shape[1]
    perms = [p for p in permutations(range(A)) if p != tuple(range(A))]
    ok = np.isfinite(c).all((1, 2))
    v = ok[1:] & ok[:-1]
    a, b = c[:-1][v], c[1:][v]
    d0 = np.linalg.norm(b - a, axis=-1).sum(1)
    dmin = np.min([np.linalg.norm(b[:, list(p)] - a, axis=-1).sum(1) for p in perms], 0)
    return int(v.sum()), int(((dmin < 0.5 * d0) & (d0 > 5)).sum())


def disp_stats(x):
    """x (T, ...parts..., 2) -> displacement samples (finite consecutive)."""
    d = np.linalg.norm(x[1:] - x[:-1], axis=-1)
    return d[np.isfinite(d)]


def q(v, qs=(0.5, 0.9, 0.99)):
    return [float(x) for x in np.quantile(v, qs)] if len(v) else []


def mabe():
    import common as CM
    sys.path.insert(0, str(REPO / 'scripts/benchmark/mabe22'))
    import eci_common as EC
    off = EC.offset()
    parts = CM.MAB_PARTS
    res = {}
    for st, f in (('mouse_userTrain', 'mouse_user_train.npy'), ('mouse_submission', 'mouse_submission_keypoints.npy')):
        seqs = np.load(CM.MAB / 'raw' / f, allow_pickle=True).item()['sequences']
        vids = {p.stem for p in (CM.MAB / 'videos' / st).iterdir() if p.suffix == '.avi'}
        n_grid = n_all = n_every = n_oob = n_ii = 0
        miss = np.zeros(12)
        dsp, cen_d = {p: [] for p in ('nose', 'center', 'tail_tip')}, []
        sw_n = sw_s = 0
        clips = [c for c in sorted(vids) if c not in EC.EXCLUDE]
        no_kp = [c for c in clips if c not in seqs]
        for c in clips:
            if c not in seqs:
                continue
            k = seqs[c]['keypoints'].astype(np.float64)  # (T, 3, 12, 2) (y, x) on 512
            T = len(k)
            m = (k == 0).all(-1)
            oob = ((k < 0) | (k > 512)).any(-1) & ~m
            k[m | oob] = np.nan
            k = k * 224 / 512  # native 224 px
            grid = [v for v in range(0, T + off, 6) if 0 <= v - off < T]
            ii = np.array([v - off for v in grid])
            n_grid += len(range(0, T + off, 6))
            mg = m[ii] | oob[ii]
            n_all += int((~mg).all((1, 2)).sum())
            n_every += int((~mg).any(2).all(1).sum())
            n_oob += int(oob[ii].any((1, 2)).sum())
            miss += mg.mean(1).sum(0)
            n_ii += len(ii)
            for nm in dsp:
                if len(dsp[nm]) < 2_000_000:
                    dsp[nm].append(disp_stats(k[:, :, parts.index(nm)]))
            cen = np.nanmean(k[:, :, [3, 6, 9]], 2)
            a, b = swap_stats(cen)
            sw_n += a
            sw_s += b
        res[st] = {'clips_in_store': len(clips), 'clips_without_keypoints': len(no_kp), 'grid_frames': n_grid,
                   'grid_frames_with_keypoint_row': n_ii,
                   'share_grid_all_36_parts': n_all / n_grid, 'share_grid_every_mouse_has_a_part': n_every / n_grid,
                   'share_grid_with_out_of_range_coord': n_oob / n_grid,
                   'missing_rate_per_part': {p: float(miss[j] / max(n_ii, 1)) for j, p in enumerate(parts)},
                   'disp_native_px_q50_q90_q99': {nm: q(np.concatenate(v)) for nm, v in dsp.items()},
                   'swap_proxy_rate': sw_s / max(sw_n, 1), 'swap_proxy_frames': sw_n,
                   'confidence_scores': 'not released (keypoints only, 0 = missing)'}
        print(st, json.dumps(res[st]), flush=True)
    return res


def calms():
    import common as CM
    res = {}
    for sp in ('train', 'test'):
        d = np.load(CM.CAL / f'derived/task1_{sp}.npz', allow_pickle=True)
        names, lens = list(d['names']), d['lens']
        kp = d['keypoints'].transpose(0, 1, 3, 2).astype(np.float64)  # (N, 2, 7, 2)
        sc = d['scores']  # (N, 2, 7)
        st = np.r_[0, np.cumsum(lens)]
        grid = np.concatenate([st[j] + np.arange(0, lens[j], 6) for j in range(len(names))])
        ok = np.isfinite(kp).all(-1) & (kp >= 0).all(-1)
        dsp = {}
        sw_n = sw_s = 0
        for mi, an in enumerate(('resident', 'intruder')):
            for pj, p in enumerate(CM.CAL_PARTS):
                dd = np.concatenate([disp_stats(kp[st[j]:st[j + 1], mi, pj]) for j in range(len(names))])
                dsp[f'{an}_{p}'] = q(dd)
        for j in range(len(names)):
            a, b = swap_stats(kp[st[j]:st[j + 1], :, 3])
            sw_n += a
            sw_s += b
        g = sc[grid]
        res[sp] = {'videos': len(names), 'grid_frames': int(len(grid)), 'share_grid_all_14_parts': float(ok[grid].all((1, 2)).mean()),
                   'score_q10_q50_q90': {f'{an}_{p}': q(g[:, mi, pj], (0.1, 0.5, 0.9)) for mi, an in enumerate(('resident', 'intruder'))
                                         for pj, p in enumerate(CM.CAL_PARTS)},
                   'share_grid_score_lt_0.5': {an: float((g[:, mi] < 0.5).mean()) for mi, an in enumerate(('resident', 'intruder'))},
                   'share_grid_mouse_mean_score_lt_0.5': {an: float((g[:, mi].mean(1) < 0.5).mean()) for mi, an in enumerate(('resident', 'intruder'))},
                   'disp_native_px_q50_q90_q99': dsp, 'swap_proxy_rate_neck': sw_s / max(sw_n, 1), 'swap_proxy_frames': sw_n}
        print(sp, json.dumps(res[sp]), flush=True)
    return res


def crim():
    from align_common import ROOT, load_track
    I = json.loads((REPO / 'results/vision/benchmark_crim13/inventory/inventory.json').read_text())['videos']
    res = {}
    for sp in ('train', 'test'):
        vs = [v for v in I if v['split'] == sp]
        n = n0 = n1 = n2 = 0
        dr, di = [], []
        sw_n = sw_s = 0
        for v in vs:
            pos, nm = load_track(ROOT / v['dir'] / f'{v["name"]}_t-track.mat')
            g = np.arange(0, len(nm), 5)
            n += len(g)
            n0 += int((nm[g] == 0).sum())
            n1 += int((nm[g] == 1).sum())
            n2 += int((nm[g] == 2).sum())
            dr.append(disp_stats(pos[:, 0]))
            di.append(disp_stats(pos[:, 1]))
            p2 = pos.copy()
            p2[nm < 2] = np.nan
            a, b = swap_stats(p2)
            sw_n += a
            sw_s += b
        res[sp] = {'videos': len(vs), 'grid_frames': n, 'share_grid_0_mice': n0 / n, 'share_grid_1_mouse': n1 / n,
                   'share_grid_2_mice': n2 / n,
                   'disp_native_px_q50_q90_q99': {'resident': q(np.concatenate(dr)), 'intruder': q(np.concatenate(di))},
                   'swap_proxy_rate': sw_s / max(sw_n, 1), 'swap_proxy_frames': sw_n,
                   'points_per_mouse': 1, 'confidence_scores': 'none (x, y only)'}
        print(sp, json.dumps(res[sp]), flush=True)
    return res


if __name__ == '__main__':
    out = {'mabe22': mabe(), 'calms21': calms(), 'crim13': crim()}
    (OUT / 'kp_inventory.json').write_text(json.dumps(out, indent=1))
