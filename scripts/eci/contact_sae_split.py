"""
Contact-token SAE (ECI, mice v1), step 1 (GPU): the PRE-REGISTERED combined instance splitter of mice_pairs_*.py on
UNLABELLED training frames, then the label-free automatic usability rule and the patch-level mask geometry.

Training frames (fixed before any is looked at): the levers / native_res training split = the 100 videos WITHOUT
behaviour annotation (spatial_sae_pilot.StoreIndex.split('mice', 100, seed 0)), their fg448 store frames (1 fps, >= 1
foreground patch). Per video, the sorted store frames are cut into consecutive non-overlapping windows of 30 frames
(30 s); 10 windows are drawn at random without replacement (seed 0), so 300 frames per video, 30,000 in all. Windows
(not isolated frames) because their store tokens are read in contiguous runs and neighbouring targets share SAM clips.

Splitter (identical to mice_pairs_select.py + mice_pairs_sam.py, imported where the code is a function):
    anchor   nearest frame s with |s - t| <= 100 (5 fps, +-20 s) where the dark-blob detector sees exactly 4 single-mouse
             blobs; search 0, -1, +1, ... (mice_pairs_select.video_job: the video's standardised 5 fps mp4 decoded on
             the job's local disk with get_frames.py's ffmpeg command, 3 frames per video compared to the dataset JPGs)
    prop     SAM 2.1-large video predictor from s to t (blob point + box prompts at s), targets sharing an anchor and a
             direction share one clip; per-pixel highest logit, largest component (split_test_sam.resolve)
    b1npk    no anchor: per-frame SAM image predictor, peaks prompts, positive own point + 3 negatives
Automatic rule (split_test_common.score_frame, label-free part only): count_ok & area_ok & dark_ok with A1_m = the
check1 value (results/vision/eci_mice_pairs/check1/table.json, the value the B subset was scored with). Usable frames
get mice_pairs_score.mask_geometry (debris-cleaned masks): cnt (4, 32, 32) pixels of each mouse per 16 x 16 patch of
the 512 px frame (= the 14 px patch of the 448 px DINOv2 input), stats, mind.

Output results/vision/eci_contact_sae/split/: targets.parquet (all shards, written by every shard identically),
shard_<k>of<n>.npz (tid, correct, branch, anchor_d, runtime_s, n_masks; for correct frames u_tid, cnt, stats, mind),
shard_<k>of<n>.json.
Usage (scripts/eci/contact_sae.sh STEP=split): python scripts/eci/contact_sae_split.py --shard 0 --n-shards 8
"""
import argparse
import json
import os
import shutil
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts/eci'))
from mice_pairs_score import mask_geometry  # noqa: E402
from mice_pairs_select import video_job  # noqa: E402
from split_test_common import DOMAINS, blob_labels, clean_labels, dark_clean, decode, score_frame  # noqa: E402
from split_test_sam import CKPTS, blob_points, mice_points_peaks, resolve  # noqa: E402

OUT = REPO / 'results/vision/eci_contact_sae/split'
N_WIN, WIN_LEN, SEED = 10, 30, 0
log = lambda s: print(time.strftime('%H:%M:%S'), s, flush=True)  # noqa: E731


def train_targets():
    """-> DataFrame of the 30,000 training targets (frame = store frame index, row, obs, frame_idx, frame_path)."""
    import spatial_sae_pilot as ssp
    idx = ssp.StoreIndex('mice')
    tr, _, _, _, train_v, _ = idx.split('mice', 100, 0)
    tr = np.sort(tr)
    rng = np.random.default_rng(SEED)
    pick = []
    for v in sorted(train_v):
        f = tr[idx.obs[tr] == v]
        nwin = len(f) // WIN_LEN
        for w in np.sort(rng.choice(nwin, min(N_WIN, nwin), replace=False)):
            pick.append(f[w * WIN_LEN:(w + 1) * WIN_LEN])
    frames = np.concatenate(pick)
    ann = pd.read_csv(DOMAINS['mice']['ann'], usecols=['observation_id', 'frame_idx', 'frame_path'])
    rows = idx.rows[frames]
    T = pd.DataFrame({'frame': frames, 'row': rows, 'obs': idx.obs[frames], 'frame_idx': ann.frame_idx.values[rows],
                      'frame_path': ann.frame_path.values[rows], 'n_fg': idx.nfg[frames]})
    T['win'] = np.repeat(np.arange(len(pick)), [len(p) for p in pick])
    return T, ann


def sam_video(S, frames, fps, bg, cal, a1d, pred, vp, stage):
    """mice_pairs_sam.py main() for the targets S of ONE video (same code path) -> labels (n, 512, 512), runtime."""
    p = cal['blob']
    labels = np.zeros((len(S), 512, 512), np.uint8)
    rt = np.zeros(len(S))
    def fpath(f):
        return fps[f]

    def geom(fp):
        dark = dark_clean(decode(frames[fp], 'L'), bg, p)
        return blob_labels(dark, p)

    fb = np.flatnonzero(S.branch.values == 'b1npk')

    def choose(m3, sc, pts, j):
        best, ba = None, -1
        for c in range(m3.shape[0]):
            mk = m3[c] > 0
            area = int(mk.sum())
            if area == 0 or area > 3 * a1d:
                continue
            if any(mk[min(max(int(round(q[0])), 0), 511), min(max(int(round(q[1])), 0), 511)]
                   for k, q in enumerate(pts) if k != j):
                continue
            if area > ba:
                best, ba = c, area
        return m3[int(np.argmax(sc)) if best is None else best]
    for i in fb:
        r = S.iloc[i]
        blobs, areas = geom(r.frame_path)
        pts = mice_points_peaks(blobs, areas, a1d, 4) if len(areas) else []
        n = len(pts)
        if n == 0:
            continue
        a = time.time()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            pred.set_image(decode(frames[r.frame_path]))
            xy = np.array([[q[1], q[0]] for q in pts], np.float32).reshape(n, 2)
            m, sc, _ = pred.predict(point_coords=np.repeat(xy[None], n, 0), point_labels=np.eye(n, dtype=np.int32),
                                    multimask_output=True, return_logits=True)
        m = np.asarray(m, np.float32).reshape(n, 3, 512, 512)
        sc = np.asarray(sc).reshape(n, 3)
        labels[i] = resolve(np.stack([choose(m[j], sc[j], pts, j) for j in range(n)]), pts)
        rt[i] = time.time() - a

    P = S[S.branch == 'prop'].copy()
    P['s'] = (P.frame_idx + P.anchor_d).astype(int)
    P['dir'] = np.where(P.anchor_d < 0, 1, np.where(P.anchor_d > 0, -1, 0))
    groups = []
    for s, g in P.groupby('s'):
        fw, bw, z = g[g.dir == 1], g[g.dir == -1], g[g.dir == 0]
        if len(fw) or len(z):
            groups.append((s, 1, pd.concat([z, fw])))
        if len(bw):
            groups.append((s, -1, bw))
    clip = stage / 'clip'
    n_cf = 0
    for s, sgn, g in groups:
        L = int(np.abs(g.anchor_d).max())
        n_cf += L + 1
        blobs_s, areas_s = geom(fpath(s))
        pts = blob_points(blobs_s, len(areas_s))
        if clip.exists():
            shutil.rmtree(clip)
        clip.mkdir()
        for k in range(L + 1):
            (clip / f'{k:05d}.jpg').write_bytes(frames[fpath(s + sgn * k)])
        want = {}
        for i, d in zip(g.index, g.anchor_d):
            want.setdefault(int(abs(d)), []).append(i)
        torch.cuda.synchronize()
        a = time.time()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            st = vp.init_state(str(clip))
            for j, q in enumerate(pts):
                yy, xx = np.nonzero(blobs_s == j + 1)
                vp.add_new_points_or_box(st, frame_idx=0, obj_id=j + 1, points=np.array([[q[1], q[0]]], np.float32),
                                         labels=np.ones(1, np.int32),
                                         box=np.array([xx.min() - 2, yy.min() - 2, xx.max() + 2, yy.max() + 2], np.float32))
            for fi, oids, lg in vp.propagate_in_video(st):
                if fi in want:
                    lg = lg[:, 0].float().cpu().numpy()
                    Lm = resolve(lg[np.argsort(list(oids))], None)
                    for i in want[fi]:
                        labels[i] = Lm
            vp.reset_state(st)
        torch.cuda.synchronize()
        rt[g.index.values] = (time.time() - a) / len(g)
    if clip.exists():
        shutil.rmtree(clip)
    return labels, rt, len(groups), n_cf


def score_video(job):
    """(labels, target JPG bytes, obs, cal, a1m) -> per target (correct, n_masks, geometry or None)."""
    labels, tb, obs, cal, a1m = job
    bg = np.load(DOMAINS['mice']['bg'] / f'{obs}.npz')['pix_bg']
    c2 = dict(cal, a1_m=a1m)
    out = []
    for L, b in zip(labels, tb):
        dark = dark_clean(decode(b, 'L'), bg, cal['blob'])
        blobs, areas = blob_labels(dark, cal['blob'])
        s = score_frame(L.astype(np.int32), dark, blobs, areas, c2, 'mice', contact=None)
        geo = None
        if s['correct']:
            L2, _ = clean_labels(L.astype(np.int32), a1m)
            geo = mask_geometry(L2, a1m)
        out.append((bool(s['correct']), int(s['n_masks']), geo))
    return obs, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--n-shards', type=int, default=1)
    ap.add_argument('--workers', type=int, default=6, help='decode / anchor-search processes')
    ap.add_argument('--max-videos', type=int, default=0, help='smoke test')
    ap.add_argument('--max-per-video', type=int, default=0, help='smoke test: first N targets per video')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    stage = Path(os.environ.get('STAGE', '/tmp'))
    t0 = time.time()
    T, ann = train_targets()
    if args.shard == 0 and not args.max_videos:
        T.to_parquet(OUT / 'targets.parquet')
    T['tid'] = np.arange(len(T))
    vids = sorted(T.obs.unique())
    mine = vids[args.shard::args.n_shards]
    if args.max_videos:
        mine = mine[:args.max_videos]
    T = T[T.obs.isin(mine)]
    if args.max_per_video:
        T = T.groupby('obs', group_keys=False).head(args.max_per_video)
    log(f'shard {args.shard}/{args.n_shards}: {len(mine)} videos, {len(T)} targets [{time.time() - t0:.0f}s]')
    cal = json.loads((REPO / 'results/vision/eci_split_test/mice/calib.json').read_text())
    a1m = json.loads((REPO / 'results/vision/eci_mice_pairs/check1/table.json').read_text())['a1_m']
    p, lo, hi, a1d = cal['blob'], cal['single_lo'], cal['single_hi'], cal['a1_dark']
    a2 = ann[ann.observation_id.isin(set(mine))]
    fp_by = {o: g.sort_values('frame_idx').frame_path.values for o, g in a2.groupby('observation_id')}
    for o, v in fp_by.items():
        g = a2[a2.observation_id == o]
        assert np.array_equal(np.sort(g.frame_idx.values), np.arange(len(v))), o
    del ann, a2
    jobs = [(o, list(zip(g.tid, g.frame_idx.astype(int))), fp_by[o], p, lo, hi, str(stage)) for o, g in T.groupby('obs')]

    spool = Pool(2)  # scoring; both pools forked before CUDA is initialised
    pool = Pool(args.workers)
    from sam2.build_sam import build_sam2, build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    cfg, ck = CKPTS['large']
    dev = torch.device('cuda')
    pred = SAM2ImagePredictor(build_sam2(cfg, str(ck), device=dev))
    vp = build_sam2_video_predictor(cfg, str(ck), device=dev)
    res = {}
    checks, stats_v = [], []
    tt = time.time()
    gpu_s = 0.0
    pend = []
    if True:
        for k, (o, out, got, chk) in enumerate(pool.imap_unordered(video_job, jobs)):
            S = T[T.obs == o].copy()
            dmap = dict(out)
            S['anchor_d'] = [np.nan if dmap[t] is None else dmap[t] for t in S.tid]
            S['branch'] = np.where(np.isfinite(S.anchor_d), 'prop', 'b1npk')
            S = S.reset_index(drop=True)
            bg = np.load(DOMAINS['mice']['bg'] / f'{o}.npz')['pix_bg']
            a = time.time()
            labels, rt, n_clips, n_cf = sam_video(S, got, fp_by[o], bg, cal, a1d, pred, vp, stage)
            gpu_s += time.time() - a
            for i, r in S.iterrows():
                res[int(r.tid)] = {'branch': r.branch, 'anchor_d': r.anchor_d, 'runtime_s': float(rt[i])}
            pend.append((S.tid.values.copy(), spool.apply_async(
                score_video, ((labels, [got[fp] for fp in S.frame_path], o, cal, a1m),))))
            checks.append(chk)
            stats_v.append({'obs': o, 'n': len(S), 'prop': int((S.branch == 'prop').sum()), 'clips': n_clips,
                            'clip_frames': n_cf, 'sam_s': round(time.time() - a, 1)})
            log(f'  video {k + 1}/{len(jobs)} {o}: {stats_v[-1]} decode-check {chk} [{time.time() - tt:.0f}s]')
            del got, labels
    geo = {}
    for tids, ar in pend:
        _, out = ar.get()
        for tid, (c, nm, g) in zip(tids, out):
            res[int(tid)].update(correct=c, n_masks=nm)
            if g is not None:
                geo[int(tid)] = g
    spool.close()
    pool.close()
    tids = np.array(sorted(res))
    R = pd.DataFrame([res[t] for t in tids])
    utid = np.array(sorted(geo))
    sfx = f'_{args.shard}of{args.n_shards}' + ('_smoke' if args.max_videos else '')
    tmp = stage / f'shard{sfx}.npz'
    np.savez_compressed(tmp, tid=tids, correct=R.correct.values.astype(bool), branch=R.branch.values.astype(str),
                        anchor_d=R.anchor_d.values.astype(np.float32), runtime_s=R.runtime_s.values.astype(np.float32),
                        n_masks=R.n_masks.values.astype(np.int16), u_tid=utid,
                        cnt=np.stack([geo[t][0] for t in utid]) if len(utid) else np.zeros((0, 4, 32, 32), np.uint16),
                        stats=np.stack([geo[t][1] for t in utid]) if len(utid) else np.zeros((0, 4, 5), np.float32),
                        mind=np.stack([geo[t][2] for t in utid]) if len(utid) else np.zeros((0, 4, 4), np.float32))
    shutil.copy(tmp, OUT / tmp.name)
    C = pd.DataFrame(checks)
    meta = {'shard': args.shard, 'n_shards': args.n_shards, 'n_videos': len(mine), 'n_targets': len(R),
            'frac_correct': float(R.correct.mean()), 'frac_prop': float((R.branch == 'prop').mean()),
            'frac_correct_by_branch': R.groupby('branch').correct.mean().to_dict(),
            'gpu_sam_s': round(gpu_s, 1), 'sam_s_per_target': gpu_s / max(len(R), 1), 'wall_s': round(time.time() - t0, 1),
            'gpu': torch.cuda.get_device_name(0), 'a1_m': a1m,
            'decode_check': {'frames_bytes_equal': int(C.bytes_equal.sum()), 'max_abs_px': int(C.max_abs_px.max()),
                             'videos_count_mismatch': int((C.n_ann != C.n_decoded).sum())},
            'videos': stats_v}
    (OUT / f'shard{sfx}.json').write_text(json.dumps(meta, indent=1))
    log(json.dumps({k: v for k, v in meta.items() if k != 'videos'}))


if __name__ == '__main__':
    main()
