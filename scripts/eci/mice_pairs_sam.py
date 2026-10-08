"""
Per-mouse / per-pair token checks, step 2 (GPU): the PRE-REGISTERED combined splitter (mice_pairs_select.py docstring)
on the targets of one set. SAM 2.1 hiera large (tools/sam2), 512 px frames, no training, no labels.

Branch 'prop' (anchor s within +-20 s): SAM 2 video predictor on the clip s .. t (backward clips are written in
reversed frame order, so the anchor is always clip frame 0 and propagation runs forward in the file order). Prompts at
the anchor = exactly split_test_sam.py B2: one positive point per dark blob (its distance-transform maximum) + the
blob's bounding box (+2 px). Targets of one video that share the same anchor and direction share one clip (the output
at clip frame k depends only on frames 0 .. k, so this changes nothing but the runtime); a target that is itself clean
(d = 0) is read at clip frame 0 of its anchor's forward clip. Masks at the target frame, overlaps resolved per pixel
by the highest logit (> 0), each object keeps its largest component (split_test_sam.resolve).
Branch 'b1npk' (no anchor): split_test_sam.py B1n with the 'peaks' prompts (mice_points_peaks): per-frame SAM image
predictor, each mouse's point positive and the other 3 negative, of the 3 candidate masks the largest that holds no
other prompt and is <= 3 A1_dark, else SAM's best predicted IoU.

Output results/vision/eci_mice_pairs/<set>/labels[_s<k>of<n>].npz: labels (F, 512, 512) uint8, idx (row of
targets.parquet), branch, anchor_d, runtime_s (per clip, divided over its targets; b1npk per frame); sam_meta[...].json.
--shard k --n-shards n: only the videos k, k + n, ... of the sorted video ids (parallel jobs; mice_pairs_score.py
merges the shard files).
Usage: python scripts/eci/mice_pairs_sam.py --set check1 [--shard 0 --n-shards 3]
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts/eci'))
from split_test_common import DOMAINS, blob_labels, dark_clean, decode, read_tar_frames  # noqa: E402
from split_test_sam import CKPTS, blob_points, mice_points_peaks, resolve  # noqa: E402

OUT = REPO / 'results/vision/eci_mice_pairs'
log = lambda s: print(s, flush=True)  # noqa: E731


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', required=True, choices=['check1', 'bsub'])
    ap.add_argument('--ckpt', default='large', choices=list(CKPTS))
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--n-shards', type=int, default=1, help='videos split round-robin (sorted ids) over shard jobs')
    ap.add_argument('--sub', type=int, default=0)
    ap.add_argument('--n-sub', type=int, default=1, help="split one shard's videos again, round-robin")
    args = ap.parse_args()
    out = OUT / args.set
    stage = Path(os.environ.get('STAGE', '/tmp'))
    S = pd.read_parquet(out / 'targets.parquet')
    vids = sorted(S.obs.unique())
    mine = set(vids[args.shard::args.n_shards][args.sub::args.n_sub])
    S = S[S.obs.isin(mine)]  # keeps the targets.parquet index (= global target id)
    gidx = S.index.values
    S = S.reset_index(drop=True)
    sfx = '' if args.n_shards == 1 else f'_s{args.shard}of{args.n_shards}'
    sfx += '' if args.n_sub == 1 else f'_sub{args.sub}of{args.n_sub}'
    cal = json.loads((REPO / 'results/vision/eci_split_test/mice/calib.json').read_text())
    p, a1d = cal['blob'], cal['a1_dark']
    t0 = time.time()
    frames = read_tar_frames(stage / 'targets.tar')
    frames.update(read_tar_frames(stage / 'clips.tar'))
    ann = pd.read_csv(DOMAINS['mice']['ann'], usecols=['observation_id', 'frame_idx', 'frame_path'])
    ann = ann[ann.observation_id.isin(set(S.obs))]
    fp_of = {(o, f): fp for o, f, fp in zip(ann.observation_id, ann.frame_idx, ann.frame_path)}
    log(f'{len(S)} targets, {len(frames):,} frames from tar [{time.time() - t0:.0f}s]; branch counts '
        f'{S.branch.value_counts().to_dict()}')
    bgs = {}

    def geom(fp, o):
        if o not in bgs:
            if len(bgs) > 16:
                bgs.pop(next(iter(bgs)))
            bgs[o] = np.load(DOMAINS['mice']['bg'] / f'{o}.npz')['pix_bg']
        dark = dark_clean(decode(frames[fp], 'L'), bgs[o], p)
        return blob_labels(dark, p)

    from sam2.build_sam import build_sam2, build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    cfg, ck = CKPTS[args.ckpt]
    dev = torch.device('cuda')
    labels = np.zeros((len(S), 512, 512), np.uint8)
    rt = np.zeros(len(S))
    meta = {'ckpt': str(ck), 'cfg': cfg, 'gpu': torch.cuda.get_device_name(0), 'shard': args.shard,
            'n_shards': args.n_shards, 'n_targets': len(S), 'n_videos': len(mine)}

    # ---- branch b1npk: per-frame image predictor
    fb = np.flatnonzero(S.branch.values == 'b1npk')
    if len(fb):
        pred = SAM2ImagePredictor(build_sam2(cfg, str(ck), device=dev))

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
        tt = time.time()
        for k, i in enumerate(fb):
            r = S.iloc[i]
            blobs, areas = geom(r.frame_path, r.obs)
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
            if k % 500 == 0:
                log(f'  b1npk {k}/{len(fb)} [{time.time() - tt:.0f}s]')
        del pred
        torch.cuda.empty_cache()

    # ---- branch prop: grouped clips
    P = S[S.branch == 'prop'].copy()
    P['s'] = (P.frame_idx + P.anchor_d).astype(int)
    P['dir'] = np.where(P.anchor_d < 0, 1, np.where(P.anchor_d > 0, -1, 0))  # +1 forward clip, -1 backward
    groups = []
    for (o, s), g in P.groupby(['obs', 's']):
        fw, bw, z = g[g.dir == 1], g[g.dir == -1], g[g.dir == 0]
        if len(fw) or len(z):
            groups.append((o, s, 1, pd.concat([z, fw])))
        if len(bw):
            groups.append((o, s, -1, bw))
    n_prop_frames = sum(int(np.abs(g.anchor_d).max()) + 1 for *_, g in groups)
    log(f'prop: {len(P)} targets in {len(groups)} clips, {n_prop_frames:,} clip frames to propagate')
    vp = build_sam2_video_predictor(cfg, str(ck), device=dev)
    clip = stage / 'clip'
    tt = time.time()
    n_box = 0
    for gi, (o, s, sgn, g) in enumerate(groups):
        L = int(np.abs(g.anchor_d).max())
        blobs_s, areas_s = geom(fp_of[(o, s)], o)
        pts = blob_points(blobs_s, len(areas_s))
        if clip.exists():
            shutil.rmtree(clip)
        clip.mkdir()
        for k in range(L + 1):
            (clip / f'{k:05d}.jpg').write_bytes(frames[fp_of[(o, s + sgn * k)]])
        want = {int(abs(d)): [] for d in g.anchor_d}
        for i, d in zip(g.index, g.anchor_d):
            want[int(abs(d))].append(i)
        torch.cuda.synchronize(); a = time.time()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            st = vp.init_state(str(clip))
            for j, q in enumerate(pts):
                yy, xx = np.nonzero(blobs_s == j + 1)
                vp.add_new_points_or_box(st, frame_idx=0, obj_id=j + 1, points=np.array([[q[1], q[0]]], np.float32),
                                         labels=np.ones(1, np.int32),
                                         box=np.array([xx.min() - 2, yy.min() - 2, xx.max() + 2, yy.max() + 2], np.float32))
                n_box += 1
            for fi, oids, lg in vp.propagate_in_video(st):
                if fi in want:
                    lg = lg[:, 0].float().cpu().numpy()
                    Lm = resolve(lg[np.argsort(list(oids))], None)
                    for i in want[fi]:
                        labels[i] = Lm
            vp.reset_state(st)
        torch.cuda.synchronize()
        dt = time.time() - a
        rt[g.index.values] = dt / len(g)
        if gi % 200 == 0:
            log(f'  clip {gi}/{len(groups)} {o} s={s} dir={sgn} len={L + 1} targets={len(g)} {dt:.2f}s '
                f'[{time.time() - tt:.0f}s]')
    if clip.exists():
        shutil.rmtree(clip)
    meta.update(n_clips=len(groups), n_clip_frames=n_prop_frames, n_objects_with_box=n_box,
                prop_wall_s=round(time.time() - tt, 1), wall_s=round(time.time() - t0, 1),
                runtime_s_per_target={b: float(rt[S.branch.values == b].mean()) for b in ('prop', 'b1npk')
                                      if (S.branch.values == b).any()})
    tmp = stage / f'labels{sfx}.npz'
    np.savez_compressed(tmp, labels=labels, idx=gidx, branch=S.branch.values.astype(str), anchor_d=S.anchor_d.values,
                        runtime_s=rt)
    shutil.copy(tmp, out / tmp.name)
    (out / f'sam_meta{sfx}.json').write_text(json.dumps(meta, indent=1))
    log(json.dumps(meta))


if __name__ == '__main__':
    main()
