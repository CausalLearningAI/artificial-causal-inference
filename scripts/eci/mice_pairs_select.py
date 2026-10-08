"""
Per-mouse / per-pair token checks (mice v1, ECI representation), step 1 (CPU): choose the target frames and find each
one's anchor frame for the PRE-REGISTERED combined splitter (fixed 2026-10-08, before any of these frames was looked at):

    For each target frame t (5 fps), the anchor is the nearest frame s with |s - t| <= 100 frames (+-20 s) where the
    dark-blob detector sees exactly 4 blobs, each in the single-mouse range (split_test_common.is_clean, the rule of
    split_test_select.py). Search order 0, -1, +1, -2, +2, ... (at equal distance the EARLIER frame wins), only inside
    the video (frames 0 .. n_frames - 1). SAM 2.1 masks are propagated from s to t, forward (s < t) or backward
    (s > t) - mice_pairs_sam.py. No anchor within +-20 s -> per-frame SAM with point + negative prompts (B1npk of the
    split test) on t. The branch of every frame is recorded.

Target sets (--set):
    check1  a FRESH evaluation set: 1000 contact + 1000 non-contact store frames (1 fps DINOv2 fg448 store) of the 144
            annotated videos, the first / last 30 s excluded, equal quota per pool (split_test_select.quota_sample),
            seed 1, disjoint from every frame of results/vision/eci_split_test/mice/eval.parquet (incl. pilot-only).
            contact = nose_nose (Y_nn or Y_np) or nose_tail (Y_nt); non-contact = all labels 0.
    bsub    the 20,000-frame B subset of the patch diagnostic (results/vision/eci_repr_diag/mice/b/subset.parquet,
            same order). No edge exclusion (the subset has none); edge frames are flagged.

Frame source: reading ~10^5 single JPGs from NFS runs at ~18 files/s (split-test select, b_extract staging), so each
video's standardised 5 fps mp4 (data/mice/v1/observations/full/<video>.mp4, one sequential read) is decoded on the
job's local disk with EXACTLY src/dataset/get_frames.py's ffmpeg command (fps=5, rgb24, -q:v 2, frame_%06d.jpg), and
the decoded frames are checked against dataset/mice/v1/frames/full (byte equality and max pixel difference of 3
frames per video, frame count = annotations.csv); select.json reports the check. Both sets are processed in one job
(--set both) so every video is decoded once.

Output results/vision/eci_mice_pairs/<set>/: targets.parquet (one row per target: row, obs, frame_idx, frame_path,
labels, anchor_d = s - t or NaN), targets.tar (the 512 px target JPGs), clips.tar (every frame between anchor and
target, both included), select.json.
Usage: python scripts/eci/mice_pairs_select.py --set both --workers 16
"""
import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts/eci'))
from split_test_common import DOMAINS, blob_labels, dark_clean, is_clean  # noqa: E402
from split_test_select import quota_sample  # noqa: E402

DS = REPO / 'dataset'
OUT = REPO / 'results/vision/eci_mice_pairs'
WIN = 100     # +-20 s at 5 fps
EDGE = 150    # first / last 30 s (check1 only)
log = lambda s: print(s, flush=True)  # noqa: E731


def video_job(job):
    """(obs, [(tid, t)], frame paths of the video (index = frame_idx), blob params, lo, hi, stage) ->
    (obs, [(tid, d)], {path: bytes} of every frame between anchor and target, decode check)."""
    obs, targets, fps, p, lo, hi, stage = job
    vdir = Path(fps[0]).parent.name
    loc = Path(stage) / 'dec' / vdir
    loc.mkdir(parents=True, exist_ok=True)
    mp4 = REPO / f'data/mice/v1/observations/full/{vdir}.mp4'
    subprocess.run(['ffmpeg', '-nostdin', '-loglevel', 'error', '-y', '-i', str(mp4), '-vf', 'fps=5,format=rgb24',
                    '-start_number', '0', '-q:v', '2', str(loc / 'frame_%06d.jpg')], check=True)
    n = len(fps)
    n_dec = len(list(loc.glob('frame_*.jpg')))
    chk = {'obs': obs, 'n_ann': n, 'n_decoded': n_dec, 'bytes_equal': 0, 'max_abs_px': 0}
    for f in sorted({0, n // 2, int(targets[0][1])}):
        a, b = (DS / fps[f]).read_bytes(), (loc / Path(fps[f]).name).read_bytes()
        chk['bytes_equal'] += int(a == b)
        chk['max_abs_px'] = max(chk['max_abs_px'], int(np.abs(np.asarray(Image.open(io.BytesIO(a)), np.int16)
                                                               - np.asarray(Image.open(io.BytesIO(b)), np.int16)).max()))
    pb = np.load(DOMAINS['mice']['bg'] / f'{obs}.npz')['pix_bg']
    clean, raw = {}, {}

    def ok(f):
        if f not in clean:
            b = (loc / Path(fps[f]).name).read_bytes()
            raw[f] = b
            g = np.asarray(Image.open(io.BytesIO(b)).convert('L'))
            _, areas = blob_labels(dark_clean(g, pb, p), p)
            clean[f] = is_clean(areas, 4, lo, hi)
        return clean[f]

    out, need = [], set()
    for tid, t in targets:
        d = None
        for k in range(WIN + 1):
            for s in ((t,) if k == 0 else (t - k, t + k)):
                if 0 <= s < min(n, n_dec) and ok(s):
                    d = s - t
                    break
            if d is not None:
                break
        ok(t)  # target bytes always
        out.append((tid, d))
        need.update(range(min(t, t + d), max(t, t + d) + 1) if d is not None else (t,))
    shutil.rmtree(loc)
    chk['n_read'] = len(raw)
    return obs, out, {fps[f]: raw[f] for f in need}, chk


def write_tar(path, items):
    with tarfile.open(path, 'w') as tf:
        for fp, b in items:
            ti = tarfile.TarInfo(fp)
            ti.size = len(b)
            tf.addfile(ti, io.BytesIO(b))


def build_targets(name, ann, nper, args, t0):
    if name == 'check1':
        from spatial_sae_pilot import StoreIndex
        idx = StoreIndex('mice')
        tr, ev, lab, beh, train_v, eval_v = idx.split('mice', DOMAINS['mice']['n_train'], 0)
        rows = idx.rows[ev]
        E = pd.DataFrame({'row': rows, 'obs': ann.observation_id.values[rows], 'frame_idx': ann.frame_idx.values[rows],
                          'frame_path': ann.frame_path.values[rows]})
        for c in beh:
            E[c] = lab[c].values
        E['n_frames'] = nper.loc[E.obs].values
        E = E[(E.frame_idx >= EDGE) & (E.frame_idx < E.n_frames - EDGE)]
        old = pd.read_parquet(REPO / 'results/vision/eci_split_test/mice/eval.parquet')
        n0 = len(E)
        E = E[~E.row.isin(set(old.row))].reset_index(drop=True)
        E['contact'] = E.nose_nose | E.nose_tail
        E['noncontact'] = ~(E.nose_nose | E.nose_tail | E.nn_mutual | E.np_directional)
        E['unit'] = E.obs.str.rsplit('_', n=2).str[0]
        log(f'check1 candidates: {n0:,} store frames of {E.obs.nunique()} annotated videos after edge exclusion, '
            f'{n0 - len(E)} removed as split-test eval frames; contact {E.contact.sum():,}, non-contact '
            f'{E.noncontact.sum():,} [{time.time() - t0:.0f}s]')
        rng = np.random.default_rng(1)
        pc = quota_sample(E[E.contact], 'unit', args.n_contact, rng)
        pn = quota_sample(E[E.noncontact], 'unit', args.n_noncontact, rng)
        S = E.loc[np.sort(np.r_[pc, pn])].copy()
        S['set'] = np.where(S.index.isin(pc), 'contact', 'noncontact')
        S = S.reset_index(drop=True)
        assert not S.row.isin(set(old.row)).any()
    else:
        S = pd.read_parquet(REPO / 'results/vision/eci_repr_diag/mice/b/subset.parquet')
        S['frame_idx'] = ann.frame_idx.values[S.row.values]
        S['frame_path'] = ann.frame_path.values[S.row.values]
        S['n_frames'] = nper.loc[S.obs].values
        S['edge'] = (S.frame_idx < EDGE) | (S.frame_idx >= S.n_frames - EDGE)
        S['set'] = np.where(S.nose_nose | S.nose_tail, 'contact', 'noncontact')
    if args.max_targets:
        S = S.groupby('set', group_keys=False).apply(lambda g: g.head(args.max_targets // 2)).reset_index(drop=True)
    log(f'{name}: ' + S.set.value_counts().to_dict().__repr__())
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', required=True, choices=['check1', 'bsub', 'both'])
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--n-contact', type=int, default=1000)
    ap.add_argument('--n-noncontact', type=int, default=1000)
    ap.add_argument('--max-targets', type=int, default=0, help='smoke test: first N targets per set only')
    ap.add_argument('--max-videos', type=int, default=0, help='smoke test: first N videos only')
    args = ap.parse_args()
    names = ['check1', 'bsub'] if args.set == 'both' else [args.set]
    t0 = time.time()
    ann = pd.read_csv(DOMAINS['mice']['ann'], usecols=['observation_id', 'frame_idx', 'frame_path'])
    nper = ann.groupby('observation_id').size()
    cal = json.loads((REPO / 'results/vision/eci_split_test/mice/calib.json').read_text())
    p, lo, hi = cal['blob'], cal['single_lo'], cal['single_hi']
    Ss = {nm: build_targets(nm, ann, nper, args, t0) for nm in names}
    if args.max_videos:
        keep = sorted(set.intersection(*[set(S.obs) for S in Ss.values()]))[:args.max_videos]
        Ss = {nm: S[S.obs.isin(keep)].reset_index(drop=True) for nm, S in Ss.items()}
    A = pd.concat([S[['obs', 'frame_idx']].assign(tset=nm, k=np.arange(len(S))) for nm, S in Ss.items()],
                  ignore_index=True)
    A['tid'] = np.arange(len(A))
    fp_by = {o: g.sort_values('frame_idx').frame_path.values
             for o, g in ann[ann.observation_id.isin(set(A.obs))].groupby('observation_id')}
    stage = Path(os.environ.get('STAGE', OUT / 'stage'))
    stage.mkdir(parents=True, exist_ok=True)
    jobs = [(o, list(zip(g.tid, g.frame_idx.astype(int))), fp_by[o], p, lo, hi, str(stage))
            for o, g in A.groupby('obs')]
    jobs.sort(key=lambda j: -len(j[1]))
    d_all = np.full(len(A), np.nan)
    clips, checks = {}, []
    t1 = time.time()
    log(f'{len(A)} targets in {len(jobs)} videos; ffmpeg {subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.splitlines()[0]}')
    with Pool(args.workers) as pool:
        for k, (o, res, got, chk) in enumerate(pool.imap_unordered(video_job, jobs)):
            for tid, d in res:
                if d is not None:
                    d_all[tid] = d
            clips.update(got)
            checks.append(chk)
            if k % 10 == 0:
                log(f'  video {k + 1}/{len(jobs)} {o}: {len(res)} targets, {chk} [{time.time() - t1:.0f}s]')
    C = pd.DataFrame(checks)
    dec = {'n_videos': len(C), 'frames_checked_bytes_equal': int(C.bytes_equal.sum()),
           'max_abs_px_diff': int(C.max_abs_px.max()), 'videos_frame_count_mismatch': int((C.n_ann != C.n_decoded).sum()),
           'n_frames_read': int(C.n_read.sum())}
    log(f'decode check: {dec}')
    for nm, S in Ss.items():
        out = OUT / nm
        out.mkdir(parents=True, exist_ok=True)
        d = d_all[A.tid[A.tset == nm].values]
        S['anchor_d'] = d
        S['branch'] = np.where(np.isfinite(d), 'prop', 'b1npk')
        S['tid'] = np.arange(len(S))
        need = set()
        for o, t, dd in zip(S.obs, S.frame_idx.astype(int), d):
            r = range(min(t, t + int(dd)), max(t, t + int(dd)) + 1) if np.isfinite(dd) else (t,)
            need.update(fp_by[o][f] for f in r)
        write_tar(stage / f'{nm}_targets.tar', ((fp, clips[fp]) for fp in S.frame_path.unique()))
        write_tar(stage / f'{nm}_clips.tar', ((fp, clips[fp]) for fp in sorted(need)))
        for f in ('targets.tar', 'clips.tar'):
            shutil.copy(stage / f'{nm}_{f}', out / f)
            os.remove(stage / f'{nm}_{f}')
        S.to_parquet(out / 'targets.parquet')
        ad = np.abs(d[np.isfinite(d)])
        info = {'set': nm, 'n_targets': len(S), 'window_frames': WIN, 'edge_frames': EDGE if nm == 'check1' else None,
                'coverage': float(np.isfinite(d).mean()),
                'coverage_by_set': S.groupby('set').anchor_d.apply(lambda x: float(np.isfinite(x).mean())).to_dict(),
                'frac_anchor_before': float((d < 0).mean()), 'frac_anchor_after': float((d > 0).mean()),
                'frac_target_clean': float((d == 0).mean()),
                'abs_d_median': float(np.median(ad)) if len(ad) else None,
                'abs_d_mean': float(ad.mean()) if len(ad) else None, 'n_clip_frames': len(need),
                'tar_mb': {f: round(os.path.getsize(out / f) / 1e6, 1) for f in ('targets.tar', 'clips.tar')},
                'decode_check': dec, 'blob': p, 'single_lo': lo, 'single_hi': hi, 'wall_s': round(time.time() - t0, 1)}
        (out / 'select.json').write_text(json.dumps(info, indent=1))
        log(json.dumps(info))
    C.to_csv(OUT / 'decode_check.csv', index=False)


if __name__ == '__main__':
    main()
