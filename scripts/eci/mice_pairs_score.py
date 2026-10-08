"""
Per-mouse / per-pair token checks, step 3 (CPU): score the combined splitter's masks with EXACTLY the automatic rule of
split_test_common.score_frame, per branch; contact sheets for the visual check; for the B subset, the mask geometry
of the frames whose split passes the automatic rule (the only label-free usability filter).

A1_m (the method's single-mouse mask area): check1 = median area of its masks (>= 0.1 A1_dark) on the CLEAN
non-contact frames of the check1 set (the split-test rule, split_test_score.py); bsub reuses the check1 value (fixed
before the B subset is scored).
Sheets (check1, seed 0): 50 random contact frames (2 sheets x 25) and 25 random non-contact frames; tiles carry only an
index; sheets/index.json records the frames, their branch and automatic verdicts.
Geometry (bsub, usable frames; masks after split_test_common.clean_labels, i.e. debris < 0.2 A1_m dropped, 4 masks):
    cnt    (U, 4, 32, 32) uint16 pixels of mouse m inside each 16 x 16 patch of the 512 px frame = the 14 px patch of
           the 448 px DINOv2 input (the HF processor resizes 512 -> 448 without crop, so patch (r, c) = frame pixels
           [16 r, 16 r + 16) x [16 c, 16 c + 16))
    stats  (U, 4, 5)  area / A1_m, centroid y / 512, centroid x / 512, principal-axis angle theta (rad, from the
           pixel second moments, defined mod pi: head / tail are not distinguishable without keypoints), elongation
           (sqrt of the eigenvalue ratio)
    mind   (U, 4, 4)  minimum mask-to-mask pixel distance / 16 (in patches; 0 = touching)
Output results/vision/eci_mice_pairs/<set>/: scores.parquet, table.json, sheets/ (check1), geom.npz (bsub).
Usage: python scripts/eci/mice_pairs_score.py --set check1 --workers 8
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts/eci'))
from split_test_common import (DOMAINS, blob_labels, clean_labels, crop_box, dark_clean, decode, is_clean,  # noqa: E402
                               read_tar_frames, score_frame, sheet, tile)

OUT = REPO / 'results/vision/eci_mice_pairs'
log = lambda s: print(s, flush=True)  # noqa: E731
G = {}


def frame_geom(i):
    r = G['S'].iloc[i]
    pb = np.load(DOMAINS['mice']['bg'] / f'{r.obs}.npz')['pix_bg']
    dark = dark_clean(decode(G['frames'][r.frame_path], 'L'), pb, G['cal']['blob'])
    blobs, areas = blob_labels(dark, G['cal']['blob'])
    return dark, blobs, areas


def work_areas(i):
    _, _, areas = frame_geom(i)
    c = np.bincount(G['labels'][i].ravel())[1:]
    cal = G['cal']
    return is_clean(areas, 4, cal['single_lo'], cal['single_hi']), c[c >= 0.1 * cal['a1_dark']].tolist()


def mask_geometry(L2, a1m):
    cnt = np.zeros((4, 32, 32), np.uint16)
    stats = np.zeros((4, 5), np.float32)
    mind = np.zeros((4, 4), np.float32)
    dts = []
    for m in range(4):
        mk = L2 == m + 1
        cnt[m] = mk.reshape(32, 16, 32, 16).sum((1, 3))
        yy, xx = np.nonzero(mk)
        cy, cx = yy.mean(), xx.mean()
        C = np.cov(np.stack([yy - cy, xx - cx]))
        ev, evec = np.linalg.eigh(C)
        v = evec[:, 1]  # major axis (y, x)
        stats[m] = (mk.sum() / a1m, cy / 512, cx / 512, np.arctan2(v[0], v[1]) % np.pi,
                    np.sqrt(ev[1] / max(ev[0], 1e-6)))
        dts.append(ndi.distance_transform_edt(~mk))
    for i in range(4):
        for j in range(4):
            if i != j:
                mind[i, j] = dts[j][L2 == i + 1].min() / 16
    return cnt, stats, mind


def work_score(i):
    r = G['S'].iloc[i]
    dark, blobs, areas = frame_geom(i)
    L = G['labels'][i].astype(np.int32)
    cal = dict(G['cal'], a1_m=G['a1m'])
    s = score_frame(L, dark, blobs, areas, cal, 'mice', contact=r.set == 'contact')
    s.update(i=i, merged_blob=bool((areas > G['cal']['single_hi']).any()), n_dark_blobs=len(areas),
             clean=is_clean(areas, 4, G['cal']['single_lo'], G['cal']['single_hi']))
    geo = None
    if G['geom'] and s['correct']:
        L2, _ = clean_labels(L, G['a1m'])
        geo = mask_geometry(L2, G['a1m'])
    return s, geo


def summ(d):
    c = d.correct.astype(bool)
    pr = d.pair_ok.dropna().astype(bool)
    return {'n': int(len(d)), 'auto_correct': round(float(c.mean()), 4) if len(d) else None,
            'pair_ok': round(float(pr.mean()), 4) if len(pr) else None, 'n_pair': int(len(pr)),
            'count_ok': round(float(d.count_ok.mean()), 4) if len(d) else None,
            'area_ok': round(float(d.area_ok.mean()), 4) if len(d) else None,
            'dark_ok': round(float(d.dark_ok.mean()), 4) if len(d) else None,
            'mean_n_masks': round(float(d.n_masks.mean()), 3) if len(d) else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', required=True, choices=['check1', 'bsub'])
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()
    out = OUT / args.set
    stage = Path(os.environ.get('STAGE', out))
    t0 = time.time()
    S = pd.read_parquet(out / 'targets.parquet')
    labels = np.zeros((len(S), 512, 512), np.uint8)
    seen = np.zeros(len(S), bool)
    files = sorted(out.glob('labels*.npz'))
    for f in files:
        z = np.load(f)
        labels[z['idx']] = z['labels']
        assert not seen[z['idx']].any(), f'{f}: targets labelled twice'
        seen[z['idx']] = True
        assert (z['branch'] == S.branch.values[z['idx']]).all()
    assert seen.all(), f'{(~seen).sum()} targets without masks ({[f.name for f in files]})'
    cal = json.loads((REPO / 'results/vision/eci_split_test/mice/calib.json').read_text())
    G.update(S=S, labels=labels, frames=read_tar_frames(stage / 'targets.tar'), cal=cal, geom=args.set == 'bsub')
    log(f'{len(S)} targets [{time.time() - t0:.0f}s]')
    if args.set == 'check1':
        with Pool(args.workers) as pool:
            A = list(pool.imap(work_areas, range(len(S)), chunksize=16))
        nc = (S.set == 'noncontact').values
        ar = [a for (c, aa), n in zip(A, nc) if c and n for a in aa]
        a1m = float(np.median(ar))
        n_clean_nc = int(sum(c for (c, _), n in zip(A, nc) if n))
        log(f'A1_m = {a1m:.1f} px ({a1m / cal["a1_dark"]:.2f} A1_dark) from {n_clean_nc} clean non-contact frames')
    else:
        a1m = json.loads((OUT / 'check1/table.json').read_text())['a1_m']
        n_clean_nc = None
    G['a1m'] = a1m
    with Pool(args.workers) as pool:
        R = list(pool.imap(work_score, range(len(S)), chunksize=8))
    D = pd.DataFrame([s for s, _ in R])
    D = pd.concat([S.reset_index(drop=True), D.drop(columns='i')], axis=1)
    D['correct'] = D.correct.astype(bool)
    D.to_parquet(out / 'scores.parquet')
    table = {'set': args.set, 'a1_m': a1m, 'a1_dark': cal['a1_dark'], 'n_clean_noncontact': n_clean_nc}
    for st in ('contact', 'noncontact'):
        d = D[D.set == st]
        t = {'all': summ(d), 'coverage_prop': round(float((d.branch == 'prop').mean()), 4)}
        for b in ('prop', 'b1npk'):
            t[b] = summ(d[d.branch == b])
        for nm, m in (('prop_anchor_before', d.anchor_d < 0), ('prop_anchor_after', d.anchor_d > 0),
                      ('prop_target_clean', d.anchor_d == 0)):
            t[nm] = summ(d[m])
        t['merged_blob'], t['separate'] = summ(d[d.merged_blob]), summ(d[~d.merged_blob])
        ad = np.abs(d.anchor_d.dropna())
        for lo_, hi_ in ((1, 10), (11, 50), (51, 100)):
            t[f'prop_absd_{lo_}_{hi_}'] = summ(d[(np.abs(d.anchor_d) >= lo_) & (np.abs(d.anchor_d) <= hi_)])
        t['abs_d_median'] = float(ad.median()) if len(ad) else None
        table[st] = t
    if args.set == 'bsub':
        w = D.w.values
        u = D.correct.values
        table['usable'] = {'n_usable': int(u.sum()), 'n': len(D), 'frac_usable': round(float(u.mean()), 4),
                           'frac_usable_weighted': round(float(np.average(u, weights=w)), 4),
                           'frac_usable_contact': round(float(u[D.set == 'contact'].mean()), 4),
                           'frac_usable_noncontact': round(float(u[D.set == 'noncontact'].mean()), 4)}
        for b in ('nose_nose', 'nose_tail'):
            y = D[b].values.astype(float)
            table['usable'][f'base_rate_{b}_all'] = round(float(np.average(y, weights=w)), 5)
            table['usable'][f'base_rate_{b}_usable'] = round(float(np.average(y[u], weights=w[u])), 5)
            table['usable'][f'n_pos_{b}_usable'] = int(y[u].sum())
            table['usable'][f'n_pos_{b}_all'] = int(y.sum())
        table['usable']['n_fg_mean_all'] = float(np.average(D.n_fg, weights=w))
        table['usable']['n_fg_mean_usable'] = float(np.average(D.n_fg[u], weights=w[u]))
        table['usable']['frac_edge_frames_usable'] = float(D.edge[u].mean())
        ui = np.flatnonzero(u)
        cnt = np.stack([R[i][1][0] for i in ui])
        stats = np.stack([R[i][1][1] for i in ui])
        mind = np.stack([R[i][1][2] for i in ui])
        np.savez_compressed(out / 'geom.npz', idx=ui, cnt=cnt, stats=stats, mind=mind)
        log(f'geometry of {len(ui)} usable frames -> {out / "geom.npz"}')
    (out / 'table.json').write_text(json.dumps(table, indent=1))
    log(json.dumps(table, indent=1))

    if args.set == 'check1':
        sd = out / 'sheets'
        sd.mkdir(exist_ok=True)
        rng = np.random.default_rng(0)
        vis_c = np.sort(rng.choice(np.flatnonzero((S.set == 'contact').values), 50, replace=False))
        vis_n = np.sort(rng.choice(np.flatnonzero((S.set == 'noncontact').values), 25, replace=False))
        index = {}
        for name, ids in (('contact', vis_c), ('noncontact', vis_n)):
            tiles = []
            for k, i in enumerate(ids):
                dark, blobs, _ = frame_geom(i)
                L = G['labels'][i].astype(np.int32)
                rgb = decode(G['frames'][S.frame_path.iloc[i]])
                tiles.append(tile(rgb, L, crop_box('mice', dark | (L > 0)), px=300, text=str(k)))
            for part in range(0, len(tiles), 25):
                sheet(tiles[part:part + 25], 5, sd / f'combined_{name}_{part // 25}.jpg',
                      title=f'mice combined splitter {name} tiles {part}-{min(part + 25, len(tiles)) - 1}')
            index[name] = {'i': ids.tolist(), 'branch': S.branch.values[ids].tolist(),
                           'anchor_d': [None if not np.isfinite(x) else int(x) for x in S.anchor_d.values[ids]],
                           'auto_correct': D.correct.values[ids].tolist(), 'obs': S.obs.values[ids].tolist(),
                           'frame_idx': S.frame_idx.values[ids].astype(int).tolist()}
        (sd / 'index.json').write_text(json.dumps(index, indent=1))
        log(f'sheets -> {sd}')
    log(f'done [{time.time() - t0:.0f}s]')


if __name__ == '__main__':
    main()
