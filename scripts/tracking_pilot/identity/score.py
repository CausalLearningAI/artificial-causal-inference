"""Mark-reading pass: decode the source video at the read stride, crop every ISOLATED track row
(see assign.isolated: detected, touching no other box, not oversize, not tiny) and compute mark
features. Non-isolated rows are never read (they are 'unreadable' by construction).

Outputs in results/tracking_pilot/{domain}/{video}/identity/{tag}/:
    feats.parquet   frame_src, track_id, variant, mask_ok, hand/colour features
    dino.npy        (mice) DINOv2 flip-averaged features, row-aligned with feats.parquet
    crops/          a JPEG of every `dump_every`-th read (for labelling and audit)
apply_scorer() turns features into reads (readable, p_<id>) with a trained scorer.
"""
import argparse
import pickle
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import marks_ants
import marks_mice
from assign import DEFAULTS, frame_geometry, isolated
from common import (IDENTITIES, N_ANIMALS, RESULTS, annotated_window, first_frame_index, frame_stride,
                    iter_frames, load_tracks,
                    out_dir, video_path)
from standin import MICE_SCALE, background

READ_S = {'mice': 0.5, 'ants': 0.2}


def get_background(domain, vid):
    d = out_dir(domain, vid)
    p = d / f'background_{domain}.npy'
    if p.exists():
        return np.load(p)
    s, e = annotated_window(domain, vid)
    path = video_path(domain, vid)
    bg = background(path, MICE_SCALE, s, e) if domain == 'mice' else \
        background(path, 1.0, max(s, 1), e, gray=False, q=0.85)
    np.save(p, bg)
    return bg


def read_pass(domain, vid, tracks, tag, dump_every=4, device='cuda'):
    od = out_dir(domain, vid) / tag
    (od / 'crops').mkdir(parents=True, exist_ok=True)
    bg = get_background(domain, vid)
    path = video_path(domain, vid)
    rows, dino_crops = [], []
    t0 = time.time()
    for variant, tr in tracks.groupby('variant'):
        g = frame_geometry(tr, DEFAULTS, N_ANIMALS[domain])
        g['iso'] = isolated(g)
        med_area_src = float(np.median((g.w * g.h)[g.detected]))
        st = frame_stride(tr)
        rs = max(st, int(round(READ_S[domain] * 30 / st)) * st)
        # first track frame that exists in the video (ant mkvs start at frame 1)
        fv = first_frame_index(path)
        f0 = int(tr.frame_src[tr.frame_src >= fv].min())
        iso = g[g.iso]
        iso = iso[(iso.frame_src - f0) % rs == 0]
        by_f = {f: d for f, d in iso.groupby('frame_src')}
        last = int(iso.frame_src.max()) + 1 if len(iso) else f0
        # median single-animal mask area in crop pixels (mice): body area scales with (224/560)^2
        k = 0
        for f, img in iter_frames(path, stride=rs, gray=(domain == 'mice'), start=f0, end=last):
            if f not in by_f:
                continue
            for r in by_f[f].itertuples():
                if domain == 'mice':
                    c, m, _, ang = marks_mice.aligned_crop(img, bg, MICE_SCALE, r.cx, r.cy)
                    hf = marks_mice.hand_features(c, m)
                    rec = {'mask_area': int(m.sum()), 'border': float(np.r_[m[0], m[-1], m[:, 0], m[:, -1]].mean()),
                           'angle': ang, **{f'h{i}': v for i, v in enumerate(hf)}}
                    dino_crops.append(c)
                    im = c
                else:
                    F, c = marks_ants.features(img, bg, r.cx, r.cy)
                    rec = {'blue_px': F[0], 'yellow_px': F[1], 'fg_area': F[2], 'core_area': F[3]}
                    im = c
                fn = ''
                if k % dump_every == 0:
                    fn = f'{variant}_{f:06d}_{r.track_id}.jpg'
                    cv2.imwrite(str(od / 'crops' / fn), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
                rows.append({'variant': variant, 'frame_src': f, 'track_id': r.track_id,
                             'crop_file': fn, 'n_det': int(r.n_det), **rec})
                k += 1
            if f % 9000 < rs:
                print(f'  {domain}/{vid}/{variant} frame {f} reads {k} ({time.time() - t0:.0f}s)', flush=True)
    feats = pd.DataFrame(rows)
    N = N_ANIMALS[domain]
    area_col = 'mask_area' if domain == 'mice' else 'core_area'
    med = ref_area(feats, area_col, N)
    hi = 1.6 if domain == 'mice' else marks_ants.CORE_MAX
    feats['too_big'] = feats[area_col] > hi * med  # probably two animals in one box
    if domain == 'mice':
        feats['mask_ok'] = (feats.mask_area >= 0.5 * med) & ~feats.too_big & (feats.border < 0.15)
        X = marks_mice.dino_features(np.stack(dino_crops), device=device) if dino_crops else \
            np.zeros((0, 1536), np.float32)
        np.save(od / 'dino.npy', X.astype(np.float16))
    else:
        feats['mask_ok'] = (feats.core_area >= 0.5 * med) & ~feats.too_big
    feats.to_parquet(od / 'feats.parquet')
    print(f'{domain}/{vid}/{tag}: {len(feats)} reads, {time.time() - t0:.0f}s')
    return feats


def ref_area(feats, col, N):
    """Single-animal reference area of the crop mask: median over reads from frames where all N
    animals are separate boxes (>= 50 reads), else the 25th percentile (see assign._median_size)."""
    if not len(feats):
        return 0.0
    ref = feats[feats.n_det == N]
    if len(ref) >= 50:
        return float(ref[col].median())
    return float(feats[col].quantile(0.25))


def scorer_path(domain):
    return RESULTS / domain / 'scorer' / 'scorer.pkl'


def design_matrix(domain, feats, dino, kind):
    if domain == 'ants':
        return marks_ants.design(feats[['blue_px', 'yellow_px', 'fg_area']].values)  # core_area: gating only
    if kind == 'dino':
        return np.asarray(dino, np.float32)
    return feats[[c for c in feats.columns if c.startswith('h') and c[1:].isdigit()]].values


def apply_scorer(domain, vid, tag):
    od = out_dir(domain, vid) / tag
    feats = pd.read_parquet(od / 'feats.parquet')
    if 'too_big' not in feats.columns:  # feats written before the too_big gate existed (mice)
        med = ref_area(feats, 'mask_area', N_ANIMALS[domain])
        feats['too_big'] = feats.mask_area > 1.6 * med
    dino = np.load(od / 'dino.npy') if domain == 'mice' else None
    with open(scorer_path(domain), 'rb') as fh:
        S = pickle.load(fh)
    ids = IDENTITIES[domain]
    X = design_matrix(domain, feats, dino, S['kind'])
    P = S['model'].predict_proba(X) if len(X) else np.zeros((0, len(ids)))
    # model classes are identity names; reorder to ids
    order = [list(S['model'].classes_).index(i) for i in ids]
    P = P[:, order]
    reads = feats[['variant', 'frame_src', 'track_id']].copy()
    for a, i in enumerate(ids):
        reads[f'p_{i}'] = P[:, a]
    reads['readable'] = feats.mask_ok.values & (P.max(1) >= S['read_p'])
    reads['too_big'] = feats.too_big.values
    reads.to_parquet(od / 'reads.parquet')
    return reads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--domain', required=True)
    ap.add_argument('--video', required=True)
    ap.add_argument('--tracks', required=True)
    ap.add_argument('--tag', required=True, help='output subfolder, e.g. standin or amadeus')
    ap.add_argument('--dump_every', type=int, default=4)
    a = ap.parse_args()
    tracks = load_tracks(a.tracks)
    read_pass(a.domain, a.video, tracks, a.tag, a.dump_every)


if __name__ == '__main__':
    main()
