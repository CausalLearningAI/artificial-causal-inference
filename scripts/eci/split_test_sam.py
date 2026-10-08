"""
Instance-split GATE test, method B (GPU): SAM 2.1 (official repo, tools/sam2; checkpoint --ckpt, default hiera large)
on the 512 px eval frames of split_test_select.py. No training, no labels.

Prompts (label-free, from the dark mask of split_test_common.py and, for ants, the tracking's colour dots):
    mice  blob-count prompts: the 4 points are shared out over the dark blobs in proportion to area (each blob >= 1,
          largest remainder); a blob with k points is split by k-means of its pixel coordinates and each cluster gets
          the point deepest inside its own region. History (looked at on eval frames, no labels used): the 4
          strongest distance-transform peaks of the frame put 2 points on one long mouse and none on a small one
          (smoke run); then k_b = round(area / A1_dark) per blob with greedy 35 px-separated peaks over-allocated
          to blobs with tails / shadows and lined points up inside a 3-mouse huddle, leaving a mouse without a
          prompt (first full run). Both are kept (--mice-prompts peaks | kmeans; outputs renamed B1pk / B1npk and
          B1km / B1nkm): the k-means version scored LOWER on the automatic rule, so the change did not help.
    ants  the yellow dot and the blue dot (raw detection; the tracker's gap-filled colour centroid when missing) and a
          focal point = the dark-blob pixel (distance >= 1.5 px from the blob edge) farthest from both dots
B-native (--native, mice): the same prompts (from the 512 px dark mask) x 2 on the 1024 px frames decoded from the
2064 px source (split_test_native.py, main contact / non-contact sets only), SAM logits averaged 2 x 2 back to 512.
Input crop: ants = a square zoom crop around all dark animal pixels and prompts (+ 24 px margin, side >= 128 px; SAM
resizes it to 1024, so a ~40 px ant becomes >= ~300 px), mice = the whole 512 frame. A first smoke run without the
crop and with SAM's own best-IoU mask returned only the colour-dot / gaster part of the ants.
Mask choice (B1, B1n): of SAM's 3 multimask candidates, the LARGEST one that holds no other animal's prompt point and is
<= 3 A1_dark pixels (whole animal, not the group / arena); if none qualifies, SAM's best predicted IoU.
Methods:
    B1    one positive point per animal
    B1n   the animal's point positive + the other animals' points negative
    B1z / B1nz  (ants) as B1 / B1n but each animal gets its own zoom crop of --zoom-px (128) px centred on its prompt
          (other animals' points inside the crop as negatives for B1nz), one image embedding per animal: added after
          the B1n sheets showed gaster-only masks even on isolated ants whenever the ants were spread out (the shared
          crop then is the whole arena, so no zoom)
    B2    short-range video propagation: SAM 2 video predictor on the 5 fps frames s..t, s = the nearest earlier frame
          within 10 s (50 frames) where the animals are cleanly separate (split_test_select.py b2_d); prompts at s =
          one positive point per animal (one per dark blob, its distance-transform maximum, when the detector sees N
          single-size blobs at s - always for mice; else, ants only, dots + focal point as B1) + the box of the clean blob holding the point (when no other object shares
          that blob); same zoom crop for every frame of the clip (around the animals at s and t); masks read at t. Frames with
          no clean start frame within 10 s get an empty label map (they count as failures; coverage reported).
Overlaps resolved per pixel by the highest mask logit among objects with logit > 0; each object keeps the connected
component holding its prompt point (B1 / B1n) or its largest component (B2).

Output results/vision/eci_split_test/<domain>/labels_{B1,B1n,B2}.npz (labels (F, 512, 512) uint8 in eval.parquet
order, prompts, runtime) and sam_meta.json.
Usage: python scripts/eci/split_test_sam.py --domain mice [--ckpt large] [--methods B1 B1n B2] [--max-frames 50]
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
from scipy import ndimage as ndi

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts/eci'))
from split_test_common import DOMAINS, OUT, blob_labels, dark_clean, decode, is_clean, load_calib, read_tar_frames  # noqa: E402

TOOLS = Path('/nfs/scistore19/locatgrp/rcadei/tools')
CKPTS = {'large': ('configs/sam2.1/sam2.1_hiera_l.yaml', TOOLS / 'sam2_ckpt/sam2.1_hiera_large.pt'),
         'base_plus': ('configs/sam2.1/sam2.1_hiera_b+.yaml', TOOLS / 'sam2_ckpt/sam2.1_hiera_base_plus.pt')}
log = lambda s: print(s, flush=True)  # noqa: E731


def mice_points_peaks(blobs, areas, a1, n=4, sep=35.0, min_dt=3.0):
    """Run-1 prompts ('peaks', methods B1pk / B1npk): blobs in decreasing area get k_b = clip(round(area / A1_dark), 1,
    n) points each until n points are placed; within a blob the k_b points are its greedy distance-transform peaks
    >= sep px apart."""
    pts = []
    for b in np.argsort(-areas) + 1:
        kb = min(int(np.clip(np.round(areas[b - 1] / a1), 1, n)), n - len(pts))
        if kb <= 0:
            break
        m = blobs == b
        ys, xs = np.nonzero(m)
        y0, x0 = ys.min(), xs.min()
        sub = m[y0:ys.max() + 1, x0:xs.max() + 1]
        dt = ndi.distance_transform_edt(sub)
        yy, xx = np.mgrid[:sub.shape[0], :sub.shape[1]]
        got = 0
        while got < kb:
            i = int(np.argmax(dt))
            y, x = divmod(i, sub.shape[1])
            if dt[y, x] < min_dt and got > 0:
                break
            pts.append((float(y + y0), float(x + x0)))
            got += 1
            dt[(yy - y) ** 2 + (xx - x) ** 2 < sep * sep] = 0
    return pts


def mice_points(blobs, areas, a1, n=4, iters=15):
    """Blob-count prompts: the n points are shared out over the dark blobs in proportion to their area (every blob >= 1
    while points remain, largest remainder first; blobs beyond n in decreasing area get none); a blob with k points is
    split by k-means of its pixel coordinates (farthest-point init) and each cluster's point is its pixel deepest
    inside the blob (distance-transform maximum)."""
    order = np.argsort(-areas)[:n]
    k = np.ones(len(order), int)
    share = areas[order] / areas[order].sum() * n
    rem = share - k
    while k.sum() < n:
        j = int(np.argmax(rem))
        k[j] += 1
        rem[j] -= 1
    pts = []
    for b, kb in zip(order + 1, k):
        m = blobs == b
        ys, xs = np.nonzero(m)
        y0, x0 = ys.min(), xs.min()
        sub = m[y0:ys.max() + 1, x0:xs.max() + 1]
        dt = ndi.distance_transform_edt(sub)
        yy, xx = np.nonzero(sub)
        X = np.stack([yy, xx], 1).astype(np.float32)
        C = [X[int(np.argmax(dt[yy, xx]))]]
        for _ in range(1, kb):
            d = np.min([((X - c) ** 2).sum(1) for c in C], 0)
            C.append(X[int(np.argmax(d))])
        C = np.array(C)
        for _ in range(iters):
            lab = np.argmin(((X[:, None] - C[None]) ** 2).sum(-1), 1)
            C = np.array([X[lab == c].mean(0) if (lab == c).any() else C[c] for c in range(kb)])
        lab = np.argmin(((X[:, None] - C[None]) ** 2).sum(-1), 1)
        for c in range(kb):
            sel = lab == c
            if not sel.any():
                continue
            # depth inside this cluster's own region, so the point sits on the body of that cluster's mouse
            reg = np.zeros_like(sub)
            reg[yy[sel], xx[sel]] = True
            dtc = ndi.distance_transform_edt(reg)
            i = int(np.argmax(dtc))
            y, x = divmod(i, sub.shape[1])
            pts.append((float(y + y0), float(x + x0)))
    return pts


def blob_points(blobs, n_blobs):
    pts = []
    for b in range(1, n_blobs + 1):
        m = blobs == b
        dt = ndi.distance_transform_edt(m)
        y, x = divmod(int(np.argmax(dt)), m.shape[1])
        pts.append((y, x))
    return pts


def ants_points(dark_blobs, row):
    def dot(c):
        y, x = row[f'raw_{c}_y'], row[f'raw_{c}_x']
        if not (np.isfinite(x) and np.isfinite(y)):
            y, x = row[f'{c}_y'], row[f'{c}_x']
        return (float(y), float(x)) if np.isfinite(x) and np.isfinite(y) else None
    yel, blu = dot('yellow'), dot('blue')
    pts = [p for p in (yel, blu) if p is not None]
    m = dark_blobs > 0
    dt = ndi.distance_transform_edt(m)
    yy, xx = np.nonzero(m & (dt >= 1.5))
    if len(yy) == 0:
        yy, xx = np.nonzero(m)
    if len(yy):
        if pts:
            dmin = np.min([np.hypot(yy - p[0], xx - p[1]) for p in pts], 0)
            i = int(np.argmax(dmin))
        else:
            i = int(np.argmax(dt[yy, xx]))
        pts.append((float(yy[i]), float(xx[i])))
    return pts


def resolve(logits, pts=None):
    """logits (n, H, W) -> label map uint8; each object keeps the component with its point (else largest)."""
    if len(logits) == 0:
        return np.zeros((512, 512), np.uint8)
    mx = logits.max(0)
    L = np.where(mx > 0, logits.argmax(0) + 1, 0).astype(np.uint8)
    out = np.zeros_like(L)
    for i in range(len(logits)):
        cc, n = ndi.label(L == i + 1, structure=np.ones((3, 3), bool))
        if n == 0:
            continue
        keep = 0
        if pts is not None:
            y, x = int(round(pts[i][0])), int(round(pts[i][1]))
            keep = cc[min(max(y, 0), 511), min(max(x, 0), 511)]
        if keep == 0:
            keep = int(np.argmax(np.bincount(cc.ravel())[1:])) + 1
        out[cc == keep] = i + 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--domain', required=True, choices=list(DOMAINS))
    ap.add_argument('--ckpt', default='large', choices=list(CKPTS))
    ap.add_argument('--methods', nargs='+', default=['B1', 'B1n', 'B2'])
    ap.add_argument('--max-frames', type=int, default=0)
    ap.add_argument('--suffix', default='')
    ap.add_argument('--mice-prompts', default='kmeans', choices=['kmeans', 'peaks'],
                    help="mice B1 prompts; outputs were renamed B1km / B1nkm (kmeans) and B1pk / B1npk (peaks)")
    ap.add_argument('--native', action='store_true', help='mice: SAM on the 1024 px native-decoded frames (B-native)')
    ap.add_argument('--zoom-px', type=int, default=128, help='B1z / B1nz: side of the per-animal zoom crop (512 px frame)')
    args = ap.parse_args()
    dom = DOMAINS[args.domain]
    N, p = dom['N'], dom['blob']
    stage = Path(os.environ.get('STAGE', '/tmp'))
    E = pd.read_parquet(OUT / args.domain / 'eval.parquet')
    if args.max_frames:
        E = E.groupby('set', group_keys=False).apply(lambda g: g.head(args.max_frames)).sort_index()
    cal = load_calib(args.domain)
    t0 = time.time()
    frames = read_tar_frames(stage / 'frames.tar')
    nat, F = {}, 1
    if args.native:  # mice: 1024 px frames decoded from the 2064 px source (split_test_native.py)
        nat, F = read_tar_frames(stage / 'native.tar'), 2
        log(f'{len(nat):,} native 1024 px frames')

    def down(lg):  # logits at F x 512 -> 512 (mean over F x F blocks)
        if F == 1:
            return lg
        sh = lg.shape
        return lg.reshape(sh[:-2] + (sh[-2] // F, F, sh[-1] // F, F)).mean((-3, -1))
    log(f'{len(frames):,} frames from tar [{time.time() - t0:.0f}s]; {len(E)} eval frames')
    from sam2.build_sam import build_sam2, build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    cfg, ck = CKPTS[args.ckpt]
    dev = torch.device('cuda')
    bgs = {}

    def pix_bg(o):
        if o not in bgs:
            bgs[o] = np.load(dom['bg'] / f'{o}.npz')['pix_bg']
        return bgs[o]

    def geom(fp, o):
        grey = decode(frames[fp], 'L')
        dark = dark_clean(grey, pix_bg(o), p)
        blobs, areas = blob_labels(dark, p)
        return dark, blobs, areas

    def prompts(blobs, areas, dark, row, clean=False):
        if args.domain == 'mice':
            if clean:
                return blob_points(blobs, len(areas))
            f = mice_points if args.mice_prompts == 'kmeans' else mice_points_peaks
            return f(blobs, areas, cal['a1_dark'], N) if len(areas) else []
        if clean and is_clean(areas, N, cal['single_lo'], cal['single_hi']):
            return blob_points(blobs, len(areas))  # B2 start frame: one point per separate ant blob
        return ants_points(blobs, row)

    def crop_of(masks, pts):
        """ants: square zoom crop around the animal pixels and prompts (SAM sees ~40 px ants at 1024 otherwise as
        ~80 px); mice: the whole frame."""
        if args.domain == 'mice':
            return 0, 0, 512
        yy, xx = [], []
        for m in masks:
            a, b = np.nonzero(m)
            yy += list(a); xx += list(b)
        yy += [q[0] for q in pts]; xx += [q[1] for q in pts]
        if not yy:
            return 0, 0, 512
        y0, y1, x0, x1 = min(yy), max(yy), min(xx), max(xx)
        side = int(min(512, max(y1 - y0, x1 - x0, 128 - 48) + 48))
        cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
        return int(min(max(cy - side / 2, 0), 512 - side)), int(min(max(cx - side / 2, 0), 512 - side)), side

    def paste(lg, box):
        y0, x0, sd = box
        full = np.full(lg.shape[:-2] + (512, 512), -50.0, np.float32)
        full[..., y0:y0 + sd, x0:x0 + sd] = lg
        return full

    def choose(m3, sc, pts, j):
        """3 candidate masks of object j -> the largest that holds no other prompt point and is <= 3 A1_dark;
        else the best predicted IoU (SAM's own choice)."""
        best, ba = None, -1
        for c in range(m3.shape[0]):
            mk = m3[c] > 0
            area = int(mk.sum())
            if area == 0 or area > 3 * cal['a1_dark']:
                continue
            if any(mk[min(max(int(round(q[0])), 0), 511), min(max(int(round(q[1])), 0), 511)]
                   for k, q in enumerate(pts) if k != j):
                continue
            if area > ba:
                best, ba = c, area
        return m3[int(np.argmax(sc)) if best is None else best]

    res = {m: np.zeros((len(E), 512, 512), np.uint8) for m in args.methods}
    rt = {m: np.zeros(len(E)) for m in args.methods}
    P = []
    meta = {'ckpt': str(ck), 'cfg': cfg, 'gpu': torch.cuda.get_device_name(0)}
    if {'B1', 'B1n', 'B1z', 'B1nz'} & set(args.methods):
        pred = SAM2ImagePredictor(build_sam2(cfg, str(ck), device=dev))
        tt = time.time()
        for i, (_, r) in enumerate(E.iterrows()):
            dark, blobs, areas = geom(r.frame_path, r.obs)
            pts = prompts(blobs, areas, dark, r)
            P.append(pts)
            box = crop_of([blobs > 0], pts)
            y0, x0, sd = box
            if args.native:
                key = f'{r.obs}/{int(r.frame_idx):06d}.jpg'
                if key not in nat:
                    continue
                rgb = decode(nat[key])  # whole frame (mice crop is the whole frame)
            else:
                rgb = np.ascontiguousarray(decode(frames[r.frame_path])[y0:y0 + sd, x0:x0 + sd])
            n = len(pts)
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                torch.cuda.synchronize(); a = time.time()
                pred.set_image(rgb)
                torch.cuda.synchronize(); t_emb = time.time() - a
                xy = F * np.array([[q[1] - x0, q[0] - y0] for q in pts], np.float32).reshape(n, 2)
                for meth in ('B1', 'B1n'):  # (B1z / B1nz below)
                    if meth not in args.methods or n == 0:
                        continue
                    a = time.time()
                    if meth == 'B1':
                        co, lab = xy[:, None, :], np.ones((n, 1), np.int32)
                    else:  # own point positive, the other animals' points negative
                        co, lab = np.repeat(xy[None], n, 0), np.eye(n, dtype=np.int32)
                    m, sc, _ = pred.predict(point_coords=co, point_labels=lab, multimask_output=True, return_logits=True)
                    m = paste(down(np.asarray(m, np.float32).reshape(n, 3, F * sd, F * sd)), box)
                    sc = np.asarray(sc).reshape(n, 3)
                    lg = np.stack([choose(m[j], sc[j], pts, j) for j in range(n)])
                    res[meth][i] = resolve(lg, pts)
                    rt[meth][i] = t_emb + time.time() - a
            zm = [m_ for m_ in ('B1z', 'B1nz') if m_ in args.methods]
            if zm and n:
                full = decode(frames[r.frame_path])
                Z = args.zoom_px
                lgs = {m_: [] for m_ in zm}
                a = time.time()
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                    for j in range(n):  # one zoom crop (and image embedding) per animal, centred on its prompt
                        zy = int(min(max(pts[j][0] - Z / 2, 0), 512 - Z)); zx = int(min(max(pts[j][1] - Z / 2, 0), 512 - Z))
                        pred.set_image(np.ascontiguousarray(full[zy:zy + Z, zx:zx + Z]))
                        own = np.array([[pts[j][1] - zx, pts[j][0] - zy]], np.float32)
                        oth = np.array([[q[1] - zx, q[0] - zy] for k, q in enumerate(pts) if k != j
                                        and 0 <= q[0] - zy < Z and 0 <= q[1] - zx < Z], np.float32).reshape(-1, 2)
                        for m_ in zm:
                            co = own if m_ == 'B1z' else np.r_[own, oth]
                            lab = np.r_[np.ones(1, np.int32), np.zeros(len(co) - 1, np.int32)]
                            mm, sc, _ = pred.predict(point_coords=co, point_labels=lab, multimask_output=True,
                                                     return_logits=True)
                            mm = paste(np.asarray(mm, np.float32).reshape(3, Z, Z), (zy, zx, Z))
                            lgs[m_].append(choose(mm, np.asarray(sc).reshape(3), pts, j))
                for m_ in zm:
                    res[m_][i] = resolve(np.stack(lgs[m_]), pts)
                    rt[m_][i] = (time.time() - a) / len(zm)
            if i % 200 == 0:
                log(f'  B1 {i}/{len(E)} [{time.time() - tt:.0f}s]')
        del pred
        torch.cuda.empty_cache()
    if 'B2' in args.methods:
        from PIL import Image
        vp = build_sam2_video_predictor(cfg, str(ck), device=dev)
        clip = stage / 'clip'
        ann = pd.read_csv(dom['ann'], usecols=['observation_id', 'frame_idx', 'frame_path'])
        ann = ann[ann.observation_id.isin(set(E.obs))]
        fp_of = {(o, f): fp for o, f, fp in zip(ann.observation_id, ann.frame_idx, ann.frame_path)}
        if args.domain == 'ants':
            exp = pd.read_csv(REPO / 'dataset/ants/eci/experiment.csv').set_index('observation_id')
            tracks = {}
        n_box = 0
        tt = time.time()
        for i, (_, r) in enumerate(E.iterrows()):
            d = int(r.b2_d)
            if d < 0:
                continue
            s = int(r.frame_idx) - d
            if args.domain == 'ants':
                if r.obs not in tracks:
                    if len(tracks) > 16:
                        tracks.pop(next(iter(tracks)))
                    tracks[r.obs] = pd.read_csv(REPO / f'dataset/ants/{exp.loc[r.obs, "experiment"]}/tracking/{r.obs}.csv').set_index('frame_idx')
                rs = tracks[r.obs].loc[s]
            else:
                rs = None
            dark_s, blobs_s, areas_s = geom(fp_of[(r.obs, s)], r.obs)
            dark_t, blobs_t, _ = geom(r.frame_path, r.obs)
            pts = prompts(blobs_s, areas_s, dark_s, rs, clean=True)
            n = len(pts)
            # each object's box = the bbox of the clean-frame blob holding its point (none if two objects share one)
            bid = [int(blobs_s[min(max(int(round(q[0])), 0), 511), min(max(int(round(q[1])), 0), 511)]) for q in pts]
            for j, q in enumerate(pts):
                if bid[j] == 0:  # nearest blob within 6 px
                    yy, xx = np.nonzero(blobs_s)
                    if len(yy):
                        k = int(np.argmin((yy - q[0]) ** 2 + (xx - q[1]) ** 2))
                        if (yy[k] - q[0]) ** 2 + (xx[k] - q[1]) ** 2 <= 36:
                            bid[j] = int(blobs_s[yy[k], xx[k]])
            box = crop_of([blobs_s > 0, blobs_t > 0], pts)
            y0, x0, sd = box
            if clip.exists():
                shutil.rmtree(clip)
            clip.mkdir()
            if args.native and any(f'{r.obs}/{s + k:06d}.jpg' not in nat for k in range(d + 1)):
                continue
            for k in range(d + 1):
                b = nat[f'{r.obs}/{s + k:06d}.jpg'] if args.native else frames[fp_of[(r.obs, s + k)]]
                if sd == 512:
                    (clip / f'{k:05d}.jpg').write_bytes(b)
                else:
                    Image.fromarray(np.ascontiguousarray(decode(b)[y0:y0 + sd, x0:x0 + sd])).save(clip / f'{k:05d}.jpg', quality=95)
            torch.cuda.synchronize(); a = time.time()
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                st = vp.init_state(str(clip))
                for j in range(n):
                    kw = {}
                    if bid[j] > 0 and bid.count(bid[j]) == 1:
                        yy, xx = np.nonzero(blobs_s == bid[j])
                        kw['box'] = F * np.array([xx.min() - x0 - 2, yy.min() - y0 - 2, xx.max() - x0 + 2, yy.max() - y0 + 2], np.float32)
                        n_box += 1
                    vp.add_new_points_or_box(st, frame_idx=0, obj_id=j + 1,
                                             points=F * np.array([[pts[j][1] - x0, pts[j][0] - y0]], np.float32),
                                             labels=np.ones(1, np.int32), **kw)
                last = None
                for fi, oids, lg in vp.propagate_in_video(st):
                    if fi == d:
                        last = (list(oids), lg[:, 0].float().cpu().numpy())
            torch.cuda.synchronize()
            rt['B2'][i] = time.time() - a
            if last is not None:
                oids, lg = last
                res['B2'][i] = resolve(paste(down(lg[np.argsort(oids)]), box), None)
            if i % 100 == 0:
                log(f'  B2 {i}/{len(E)} d={d} {rt["B2"][i]:.2f}s [{time.time() - tt:.0f}s]')
        if clip.exists():
            shutil.rmtree(clip)
        meta['b2_mean_d_frames'] = float(E.b2_d[E.b2_d >= 0].mean())
        meta['b2_objects_with_box'] = n_box
    for m in args.methods:
        tmp = stage / f'labels_{m}{args.suffix}.npz'
        ok = rt[m] > 0
        np.savez_compressed(tmp, labels=res[m], runtime_s=rt[m], level='pixel',
                            prompts=np.array([json.dumps(q) for q in P]) if P else np.array([]))
        shutil.copy(tmp, OUT / args.domain / tmp.name)
        meta[m] = {'runtime_s_per_frame_median': float(np.median(rt[m][ok])) if ok.any() else None,
                   'runtime_s_per_frame_mean': float(rt[m][ok].mean()) if ok.any() else None, 'n_run': int(ok.sum())}
        log(f'{m}: {ok.sum()} frames run, median {meta[m]["runtime_s_per_frame_median"]} s/frame')
    (OUT / args.domain / f'sam_meta{args.suffix}.json').write_text(json.dumps(meta, indent=1))
    log(f'done [{time.time() - t0:.0f}s]')


if __name__ == '__main__':
    main()
