"""
Instance-split GATE test, scoring (CPU): the pre-registered automatic rule and pair check of split_test_common.py for
every method's label maps, the table, and the contact sheets for the visual check.

Methods read: blob (the dark-blob connected components, computed here = the existing baseline), A_* (slots,
labels_A_*.npz, 32 x 32 -> 16 px blocks), B1 / B1n / B2 (SAM 2, labels_B*.npz), C (AMADEUS oriented boxes of the
pilot videos, results/tracking_pilot/<domain>/<id>/amadeus/tracks.parquet, source frame = 6 x frame_idx (30 -> 5 fps,
start_frame 0), scaled by 512 / source width; pixels in overlapping boxes go to the nearest box centre; reference
only: AMADEUS is long-term tracking with identity correction). Slots are not judged on train-video frames (the ants
pilots are train videos).

A1_m (each method's single-animal mask area) = median area of its masks (>= 0.1 A1_dark) on the CLEAN non-contact
eval frames (dark-blob detector: exactly N blobs, each in the single range).
Sheets (same frames for every method, seed 0): 50 random contact frames (ants: with both raw dots detected) and 25
random non-contact frames, mice 25 tiles per sheet (the arena), ants 12 larger tiles per sheet (a square crop around
the dark pixels, the method's masks and the dots); tiles carry only an index (no automatic verdict), dots drawn for ants
(yellow / blue circles). sheets/index.json records the frames and the automatic verdicts for the comparison.

Output results/vision/eci_split_test/<domain>/: scores.parquet (one row per method x frame), table.json, sheets/.
Usage: python scripts/eci/split_test_score.py --domain mice [--methods blob A_k5 B1 ...]
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

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts/eci'))
from split_test_common import (DOMAINS, OUT, blob_labels, crop_box, dark_clean, decode, is_clean, load_calib,  # noqa: E402
                               read_tar_frames, score_frame, sheet, tile, upsample_patch_labels)

log = lambda s: print(s, flush=True)  # noqa: E731
G = {}


def dots_of(r):
    if 'raw_yellow_x' not in r:
        return None
    f = lambda c: (float(r[f'raw_{c}_y']), float(r[f'raw_{c}_x'])) if np.isfinite(r[f'raw_{c}_x']) else None  # noqa: E731
    return (f('yellow'), f('blue'))


def frame_geom(i):
    r = G['E'].iloc[i]
    grey = decode(G['frames'][r.frame_path], 'L')
    pb = np.load(G['dom']['bg'] / f'{r.obs}.npz')['pix_bg']
    dark = dark_clean(grey, pb, G['p'])
    blobs, areas = blob_labels(dark, G['p'])
    return dark, blobs, areas


def get_labels(m, i, blobs=None):
    if m == 'blob':
        return blobs
    L = G['labels'][m][i]
    return upsample_patch_labels(L) if L.shape[0] == 32 else L.astype(np.int32)


def work_areas(i):
    dark, blobs, areas = frame_geom(i)
    out = {}
    for m in G['methods']:
        if m not in G['avail'][i]:
            continue
        L = get_labels(m, i, blobs)
        c = np.bincount(L.ravel())[1:]
        out[m] = c[c >= 0.1 * G['cal']['a1_dark']].tolist()
    return i, is_clean(areas, G['dom']['N'], G['cal']['single_lo'], G['cal']['single_hi']), out


def work_score(i):
    r = G['E'].iloc[i]
    dark, blobs, areas = frame_geom(i)
    dom = G['domain']
    if dom == 'ants':
        contact = (bool(r.groom_yellow), bool(r.groom_blue)) if r.contact else None
        dots = dots_of(r)
    else:
        contact, dots = bool(r.contact), None
    rows = []
    for m in G['methods']:
        if m not in G['avail'][i]:
            continue
        L = get_labels(m, i, blobs)
        cal = dict(G['cal'], a1_m=G['a1'][m])
        s = score_frame(L, dark, blobs, areas, cal, dom, dots=dots, contact=contact)
        s.update(method=m, i=i)
        rows.append(s)
    merged = bool((areas > G['cal']['single_hi']).any())
    return i, merged, len(areas), rows


def amadeus_labels(E, domain):
    """pilot frames -> {i: label map} from the AMADEUS oriented boxes (source px -> 512 px)."""
    out = {}
    yy, xx = np.mgrid[:512, :512].astype(np.float32)
    for v in set(E.pilot) - {''}:
        d = REPO / f'results/tracking_pilot/{domain}/{v}/amadeus'
        T = pd.read_parquet(d / 'tracks.parquet', columns=['frame_src', 'track_id', 'cx', 'cy', 'w', 'h', 'axis_angle'])
        T = T.set_index('frame_src')
        meta = json.loads((d / 'meta.json').read_text())
        if meta['window']['start_frame'] != 0 or meta['source_ffprobe']['width'] != meta['source_ffprobe']['height']:
            raise RuntimeError(f'{v}: unexpected AMADEUS window / frame shape')
        s = 512 / meta['source_ffprobe']['width']
        for i in np.flatnonzero(E.pilot.values == v):
            fs = 6 * int(E.frame_idx.iloc[i])
            if fs not in T.index:
                continue
            B = T.loc[[fs]]
            L = np.zeros((512, 512), np.int32)
            best = np.full((512, 512), np.inf, np.float32)
            for k, b in enumerate(B.itertuples()):
                if not np.isfinite(b.cx):
                    continue
                cx, cy, w, h, a = b.cx * s, b.cy * s, b.w * s, b.h * s, b.axis_angle
                u, v_ = (xx - cx) * np.cos(a) + (yy - cy) * np.sin(a), -(xx - cx) * np.sin(a) + (yy - cy) * np.cos(a)
                inside = (np.abs(u) <= w / 2) & (np.abs(v_) <= h / 2)
                d = u * u + v_ * v_
                take = inside & (d < best)
                L[take] = k + 1
                best[take] = d[take]
            out[i] = L.astype(np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--domain', required=True, choices=list(DOMAINS))
    ap.add_argument('--methods', nargs='*', default=None)
    ap.add_argument('--workers', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    args = ap.parse_args()
    dom = DOMAINS[args.domain]
    out = OUT / args.domain
    stage = Path(os.environ.get('STAGE', out))
    t0 = time.time()
    E = pd.read_parquet(out / 'eval.parquet')
    cal = load_calib(args.domain)
    frames = read_tar_frames(stage / 'frames.tar')
    labels, runtime = {}, {}
    methods = ['blob']
    for f in sorted(out.glob('labels_*.npz')):
        m = f.stem[len('labels_'):]
        z = np.load(f)
        labels[m] = z['labels']
        rt = z['runtime_s'] if 'runtime_s' in z.files else None
        runtime[m] = float(np.median(rt[rt > 0])) if rt is not None and (rt > 0).any() else \
            float(z['runtime_s_per_frame']) if 'runtime_s_per_frame' in z.files else None
        methods.append(m)
    avail = [set(methods) for _ in range(len(E))]
    E['train_video'] = E['train_video'] if 'train_video' in E else False
    for i in np.flatnonzero((E.set == 'pilot_only').values):  # native-decoded (B-native) frames: main sets only
        avail[i] -= {m for m in methods if m.endswith('_nat')}
    for i in np.flatnonzero(E.train_video.values):  # slots were trained on these videos: not judged there
        avail[i] -= {m for m in methods if m.startswith('A_')}
    C = amadeus_labels(E, args.domain)
    if C:
        lab_c = np.zeros((len(E), 512, 512), np.uint8)
        for i, L in C.items():
            lab_c[i] = L
            avail[i].add('C')
        labels['C'] = lab_c
        methods.append('C')
    if args.methods:
        methods = [m for m in methods if m in args.methods]
    G.update(E=E, frames=frames, dom=dom, p=dom['blob'], cal=cal, labels=labels, methods=methods, avail=avail,
             domain=args.domain)
    log(f'{len(E)} eval frames, methods {methods} [{time.time() - t0:.0f}s]')
    with Pool(args.workers) as pool:
        A = list(pool.imap(work_areas, range(len(E)), chunksize=16))
    clean = np.zeros(len(E), bool)
    areas_m = {m: [] for m in methods}
    for i, c, o in A:
        clean[i] = c
        if c and E.set.iloc[i] == 'noncontact':
            for m, a in o.items():
                areas_m[m] += a
    if 'C' in methods:  # C exists only on pilot frames: calibrate on its clean non-contact pilot frames
        for i, c, o in A:
            if c and E.noncontact.iloc[i] and E.set.iloc[i] != 'noncontact' and 'C' in o:
                areas_m['C'] += o['C']
    a1 = {m: float(np.median(v)) if v else float('nan') for m, v in areas_m.items()}
    log(f'clean non-contact frames: {(clean & (E.set == "noncontact").values).sum()}; A1_m: '
        + ', '.join(f'{m} {v:.0f} ({v / cal["a1_dark"]:.2f} A1_dark)' for m, v in a1.items()))
    G['a1'] = a1
    with Pool(args.workers) as pool:
        R = list(pool.imap(work_score, range(len(E)), chunksize=8))
    rows, merged, nblob = [], np.zeros(len(E), bool), np.zeros(len(E), int)
    for i, mg, nb, rr in R:
        merged[i], nblob[i] = mg, nb
        rows += rr
    S = pd.DataFrame(rows)
    E['merged_blob'], E['n_dark_blobs'], E['clean'] = merged, nblob, clean
    S = S.merge(E[['set', 'pilot', 'train_video', 'contact', 'merged_blob', 'n_dark_blobs', 'clean', 'obs', 'frame_idx']
                  + (['n_blobs'] if 'n_blobs' in E else []) + (['b2_d'] if 'b2_d' in E else [])],
                left_on='i', right_index=True)
    S.to_parquet(out / 'scores.parquet')
    log(f'scored {len(S):,} (method, frame) [{time.time() - t0:.0f}s]')

    # table
    def summ(d):
        c = d.correct.dropna().astype(bool)
        pr = d.pair_ok.dropna().astype(bool) if 'pair_ok' in d else pd.Series([], dtype=bool)
        r = {'n': int(len(d)), 'n_scored': int(len(c)), 'auto_correct': round(float(c.mean()), 4) if len(c) else None,
             'pair_ok': round(float(pr.mean()), 4) if len(pr) else None, 'n_pair': int(len(pr)),
             'count_ok': round(float(d.count_ok.mean()), 4), 'area_ok': round(float(d.area_ok.mean()), 4),
             'dark_ok': round(float(d.dark_ok.mean()), 4), 'mean_n_masks': round(float(d.n_masks.mean()), 3)}
        if 'dots_ok' in d:
            dd = d.dots_ok.dropna().astype(bool)
            r['dots_ok'] = round(float(dd.mean()), 4) if len(dd) else None
            r['frac_both_dots'] = round(float(d.both_dots.mean()), 4)
        return r
    table = {'domain': args.domain, 'a1_m': a1, 'a1_dark': cal['a1_dark'], 'runtime_s_per_frame': runtime,
             'n_clean_noncontact': int((clean & (E.set == 'noncontact').values).sum()), 'methods': {}}
    main_ = S[S.set.isin(['contact', 'noncontact'])]
    for m in methods:
        dm = main_[main_.method == m]
        t = {}
        for st in ('contact', 'noncontact'):
            d = dm[dm.set == st]
            if len(d) == 0:
                continue
            t[st] = summ(d)
            if st == 'contact':
                if args.domain == 'ants':
                    t['contact_merged'] = summ(d[d.n_blobs < 3])
                    t['contact_separate'] = summ(d[d.n_blobs == 3])
                else:
                    t['contact_merged'] = summ(d[d.merged_blob])
                    t['contact_separate'] = summ(d[~d.merged_blob])
                if 'b2_d' in d and m == 'B2':
                    t['contact_b2_covered'] = summ(d[d.b2_d >= 0])
                    t['b2_coverage'] = round(float((d.b2_d >= 0).mean()), 4)
                    t['b2_median_d_frames'] = float(d.b2_d[d.b2_d >= 0].median())
            elif m == 'B2':
                t['noncontact_b2_covered'] = summ(d[d.b2_d >= 0])
                t['b2_coverage_noncontact'] = round(float((d.b2_d >= 0).mean()), 4)
        dp = S[(S.method == m) & (S.pilot != '')]
        if len(dp):
            t['pilot_contact'] = summ(dp[dp.contact]) if dp.contact.any() else None
            t['pilot_noncontact'] = summ(dp[~dp.contact]) if (~dp.contact).any() else None
        table['methods'][m] = t
    (out / 'table.json').write_text(json.dumps(table, indent=1))
    for m, t in table['methods'].items():
        c, n = t.get('contact', {}), t.get('noncontact', {})
        log(f'{m:12s} contact auto {c.get("auto_correct")} pair {c.get("pair_ok")} (n {c.get("n_scored")}) | '
            f'merged {t.get("contact_merged", {}).get("auto_correct")} sep {t.get("contact_separate", {}).get("auto_correct")} | '
            f'non-contact auto {n.get("auto_correct")} | pilot c {(t.get("pilot_contact") or {}).get("auto_correct")} '
            f'nc {(t.get("pilot_noncontact") or {}).get("auto_correct")}')

    # sheets
    sd = out / 'sheets'
    sd.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    cand = np.flatnonzero((E.set == 'contact').values & (
        (E.raw_yellow_x.notna() & E.raw_blue_x.notna()).values if args.domain == 'ants' else True))
    vis_c = np.sort(rng.choice(cand, 50, replace=False))
    vis_n = np.sort(rng.choice(np.flatnonzero((E.set == 'noncontact').values), 25, replace=False))
    px, per, cols = (390, 12, 4) if args.domain == 'ants' else (300, 25, 5)  # ants are ~40 px long: bigger tiles
    index = {'contact': vis_c.tolist(), 'noncontact': vis_n.tolist(), 'auto': {}}
    for m in methods:
        if m == 'C':
            continue
        for name, ids in (('contact', vis_c), ('noncontact', vis_n)):
            tiles = []
            for k, i in enumerate(ids):
                r = E.iloc[i]
                dark, blobs, _ = frame_geom(i)
                L = get_labels(m, i, blobs)
                dots = dots_of(r) if args.domain == 'ants' else None
                rgb = decode(frames[r.frame_path])
                tiles.append(tile(rgb, L, crop_box(args.domain, dark | (L > 0), dots), px=px, dots=dots, text=str(k)))
            for part in range(0, len(tiles), per):
                sheet(tiles[part:part + per], cols, sd / f'{m}_{name}_{part // per}.jpg',
                      title=f'{args.domain} {m} {name} tiles {part}-{min(part + per, len(tiles)) - 1}')
            sm = S[(S.method == m)].set_index('i')
            index['auto'][f'{m}_{name}'] = [None if pd.isna(sm.correct.get(i)) else bool(sm.correct.get(i)) for i in ids]
    (sd / 'index.json').write_text(json.dumps(index, indent=1))
    log(f'sheets -> {sd} [{time.time() - t0:.0f}s]')


if __name__ == '__main__':
    main()
