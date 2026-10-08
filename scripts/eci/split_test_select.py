"""
Instance-split GATE test, step 1 (CPU): choose the evaluation frames, calibrate the dark-blob areas, find the B2
(SAM 2 short-range propagation) start frames, and pack every needed 512 px frame into one tar.

Video split = the earlier pilots' (scripts/eci/spatial_sae_pilot.py StoreIndex.split, split seed 0): mice train = 100
unannotated videos, eval = the 144 annotated videos; ants train = 128 random videos, eval = the other 128. Only frames
of the 1 fps DINOv2 foreground token store are used (so the slot model can be read on the very same frames).
Frames in the first / last 30 s of a video are excluded (experimenter's hand, white card).

Eval sets (column 'set'), stratified, seed 0:
    mice  contact     ~1000 frames with nose_nose (Y_nn or Y_np) or nose_tail (Y_nt); equal quota per pool (24 pools)
          noncontact  ~1000 frames with all three labels 0, equal quota per pool
    ants  contact     ~1000 frames with Y_Y2F or Y_B2F: 500 with merged tracker blobs (n_blobs < 3) + 500 with
                      n_blobs == 3, each stratum with an equal quota per eval video
          noncontact  ~1000 frames with neither label, equal quota per eval video
    pilot (flag)      up to 100 contact + 100 non-contact frames of each AMADEUS pilot video (method C reference);
                      frames of the main sets that fall in a pilot video carry the flag too. The mice pilots are eval
                      videos; the ants pilots (3_1_1, 3_6_6, 3_21_2) are TRAIN videos of the split, so their frames
                      ('pilot_only', train_video True) are never in the main sets and slots are not judged on them
B2 start frame (per eval frame t at 5 fps): the nearest s in [t - 50, t] (10 s) where the animals are cleanly
separate. ants: tracking n_blobs == 3 and both raw dots detected (tracking csv, no image read); mice: the dark-blob
detector sees exactly 4 blobs, each in the single-mouse range (frames read backwards from t). d = t - s, or -1.
Ants single-ant dark area: median blob area in frames where the detector sees exactly 3 blobs (40 train videos x
30 frames), single range [0.5, 1.5] x that (the crops_select.py recipe; mice reuse dataset/mice/v1/eci/crops/config.json).

Output results/vision/eci_split_test/<domain>/: eval.parquet, calib.json, frames.tar (eval frames + B2 windows).
Usage: python scripts/eci/split_test_select.py --domain mice --workers 8
"""
import argparse
import io
import json
import os
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
from split_test_common import DOMAINS, OUT, blob_labels, dark_clean, is_clean  # noqa: E402

DS = REPO / 'dataset'
WIN = 50
EDGE = 150
log = lambda s: print(s, flush=True)  # noqa: E731
AMADEUS = REPO / 'results/tracking_pilot'
ANTS_PILOTS = ['3_1_1', '3_6_6', '3_21_2']
_BG = {}


def pix_bg(domain, obs):
    k = (domain, obs)
    if k not in _BG:
        if len(_BG) > 8:
            _BG.pop(next(iter(_BG)))
        _BG[k] = np.load(DOMAINS[domain]['bg'] / f'{obs}.npz')['pix_bg']
    return _BG[k]


def quota_sample(df, key, n, rng):
    """equal quota per key value, remainder redistributed to keys with frames left."""
    groups = {k: g.index.values for k, g in df.groupby(key)}
    take = {k: 0 for k in groups}
    left = n
    while left > 0:
        open_ = [k for k in groups if take[k] < len(groups[k])]
        if not open_:
            break
        q = max(1, left // len(open_))
        for k in open_:
            a = min(q, len(groups[k]) - take[k], left)
            take[k] += a
            left -= a
            if left == 0:
                break
    out = [rng.choice(groups[k], take[k], replace=False) for k in groups if take[k]]
    return np.sort(np.concatenate(out)) if out else np.zeros(0, int)


def mice_search(job):
    """(t_paths newest first, obs, p, lo, hi) -> (d, {path: bytes}) reading back from t until a clean frame."""
    paths, obs, p, lo, hi = job
    got = {}
    d = -1
    for k, fp in enumerate(paths):
        b = (DS / fp).read_bytes()
        got[fp] = b
        g = np.asarray(Image.open(io.BytesIO(b)).convert('L'))
        _, areas = blob_labels(dark_clean(g, pix_bg('mice', obs), p), p)
        if is_clean(areas, 4, lo, hi):
            d = k
            break
    if d < 0:
        got = {paths[0]: got[paths[0]]}
    return d, got


def read_paths(paths):
    return {fp: (DS / fp).read_bytes() for fp in paths}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--domain', required=True, choices=list(DOMAINS))
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--n-contact', type=int, default=1000)
    ap.add_argument('--n-noncontact', type=int, default=1000)
    ap.add_argument('--n-pilot', type=int, default=100)
    args = ap.parse_args()
    dom = DOMAINS[args.domain]
    out = OUT / args.domain
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    t0 = time.time()
    from spatial_sae_pilot import StoreIndex
    idx = StoreIndex(args.domain)
    tr, ev, lab, beh, train_v, eval_v = idx.split(args.domain, dom['n_train'], 0)
    log(f'{args.domain}: {len(train_v)} train videos, {len(eval_v)} eval videos, {len(ev):,} eval store frames '
        f'[{time.time() - t0:.0f}s]')
    ann = pd.read_csv(dom['ann'], usecols=['observation_id', 'frame_idx', 'frame_path'])
    nper = ann.groupby('observation_id').size()
    if args.domain == 'ants':  # the AMADEUS ants pilots are TRAIN videos: their frames join as flagged extras only
        from spatial_sae_pilot import labelled_frames
        ptr = tr[np.isin(idx.obs[tr], ANTS_PILOTS)]
        ev = np.r_[ev, ptr]
        lab = labelled_frames('ants')[0].set_index('row').loc[idx.rows[ev]].reset_index()
    rows = idx.rows[ev]
    E = pd.DataFrame({'row': rows, 'obs': ann.observation_id.values[rows], 'frame_idx': ann.frame_idx.values[rows],
                      'frame_path': ann.frame_path.values[rows]})
    for c in beh:
        E[c] = lab[c].values
    E['train_video'] = E.obs.isin(set(train_v))
    E['n_frames'] = nper.loc[E.obs].values
    E = E[(E.frame_idx >= EDGE) & (E.frame_idx < E.n_frames - EDGE)].reset_index(drop=True)
    if args.domain == 'mice':
        E['contact'] = E.nose_nose | E.nose_tail
        E['noncontact'] = ~(E.nose_nose | E.nose_tail | E.nn_mutual | E.np_directional)
        E['unit'] = E.obs.str.rsplit('_', n=2).str[0]
        exp = pd.read_csv(REPO / 'data/mice/v1/experiment.csv')
        f2o = dict(zip(exp.observation_file, exp.observation_id))
        pilots = {}
        for v in ('rd25', 'rd32', 'rd18'):
            m = json.loads((AMADEUS / 'mice' / v / 'amadeus/meta.json').read_text())
            pilots[f2o[Path(m['source_path']).name]] = v
    else:
        E['contact'] = E.groom_yellow | E.groom_blue
        E['noncontact'] = ~E['contact']
        E['unit'] = E.obs
        exp = pd.read_csv(REPO / 'dataset/ants/eci/experiment.csv').set_index('observation_id')
        tracks = {}
        for o in sorted(E.obs.unique()):
            t = pd.read_csv(DS / f'ants/{exp.loc[o, "experiment"]}/tracking/{o}.csv').set_index('frame_idx')
            tracks[o] = t
        cols = ['n_blobs', 'raw_yellow_x', 'raw_yellow_y', 'raw_blue_x', 'raw_blue_y', 'yellow_x', 'yellow_y',
                'blue_x', 'blue_y', 'focal_x', 'focal_y']
        T = pd.concat([t[cols].assign(obs=o) for o, t in tracks.items()]).reset_index()
        E = E.merge(T, on=['obs', 'frame_idx'], how='left', validate='one_to_one')
        if E.n_blobs.isna().any():
            raise RuntimeError(f'{E.n_blobs.isna().sum()} eval frames without a tracking row')
        pilots = {v: v for v in ANTS_PILOTS if v in set(E.obs)}
    log(f'candidates: {len(E):,} frames, contact {E.contact.sum():,}, non-contact {E.noncontact.sum():,}; '
        f'AMADEUS pilots: {pilots}')

    picks = {}
    M = E[~E.train_video]
    C = M[M.contact]
    if args.domain == 'ants':
        h = args.n_contact // 2
        picks['contact'] = np.r_[quota_sample(C[C.n_blobs < 3], 'unit', h, rng),
                                 quota_sample(C[C.n_blobs == 3], 'unit', args.n_contact - h, rng)]
    else:
        picks['contact'] = quota_sample(C, 'unit', args.n_contact, rng)
    picks['noncontact'] = quota_sample(M[M.noncontact], 'unit', args.n_noncontact, rng)
    pil = []
    for o in pilots:
        for flag in ('contact', 'noncontact'):
            P = E[(E.obs == o) & E[flag]]
            pil.append(rng.choice(P.index.values, min(args.n_pilot, len(P)), replace=False))
    pil = np.concatenate(pil) if pil else np.zeros(0, int)
    keep = np.unique(np.r_[picks['contact'], picks['noncontact'], pil])
    S = E.loc[keep].copy()
    S['set'] = np.where(S.index.isin(picks['contact']), 'contact',
                        np.where(S.index.isin(picks['noncontact']), 'noncontact', 'pilot_only'))
    S['pilot'] = S.obs.map(pilots).fillna('')
    S = S.reset_index(drop=True)
    log(S.groupby(['set', 'pilot']).size().to_string())

    # calibration
    p = dict(dom['blob'])
    if args.domain == 'mice':
        cfg = json.loads((DS / 'mice/v1/eci/crops/config.json').read_text())['params']
        a1, lo, hi = 2 * cfg['single_lo'], cfg['single_lo'], cfg['single_hi']
        calib_src = 'dataset/mice/v1/eci/crops/config.json'
    else:
        ranges = {}
        o_all = ann.observation_id.values
        tv = rng.choice(train_v, 40, replace=False)
        areas_all, nb = [], []
        for o in tv:
            r = np.flatnonzero(o_all == o)
            for rr in np.linspace(r[0] + EDGE, r[-1] - EDGE, 30).astype(int):
                g = np.asarray(Image.open(DS / ann.frame_path.values[rr]).convert('L'))
                _, ar = blob_labels(dark_clean(g, pix_bg('ants', o), p), p)
                areas_all.append(ar); nb.append(len(ar))
        nb = np.array(nb)
        a3 = np.concatenate([a for a, n in zip(areas_all, nb) if n == 3])
        a1 = float(np.median(a3)); lo, hi = 0.5 * a1, 1.5 * a1
        calib_src = f'40 train videos x 30 frames; blob-count histogram {np.bincount(nb).tolist()}'
    cal = {'domain': args.domain, 'blob': p, 'a1_dark': a1, 'single_lo': lo, 'single_hi': hi, 'calib_src': calib_src,
           'train_videos': [str(v) for v in train_v], 'eval_videos': [str(v) for v in eval_v], 'pilots': pilots,
           'window_frames': WIN, 'edge_frames': EDGE}
    log(f'calibration: {json.dumps({k: cal[k] for k in ("a1_dark", "single_lo", "single_hi", "calib_src")})}')

    # B2 start frames and frame bytes
    fp_by = {o: g.set_index('frame_idx').frame_path for o, g in ann[ann.observation_id.isin(set(S.obs))].groupby('observation_id')}
    frames = {}
    d_all = np.full(len(S), -1)
    t1 = time.time()
    if args.domain == 'mice':
        jobs = []
        for o, f in zip(S.obs, S.frame_idx):
            jobs.append(([fp_by[o].loc[t] for t in range(f, max(f - WIN, 0) - 1, -1)], o, p, lo, hi))
        with Pool(args.workers) as pool:
            for i, (d, got) in enumerate(pool.imap(mice_search, jobs, chunksize=4)):
                d_all[i] = d
                frames.update(got)
                if i % 200 == 0:
                    log(f'  search {i}/{len(jobs)} [{time.time() - t1:.0f}s]')
    else:
        jobs = []
        for i, (o, f) in enumerate(zip(S.obs, S.frame_idx)):
            t = tracks[o]
            for k in range(0, WIN + 1):
                s = f - k
                if s < 0:
                    break
                r = t.loc[s]
                if r.n_blobs == 3 and np.isfinite(r.raw_yellow_x) and np.isfinite(r.raw_blue_x):
                    d_all[i] = k
                    break
            jobs.append([fp_by[o].loc[s] for s in range(f - max(d_all[i], 0), f + 1)])
        with Pool(args.workers) as pool:
            for got in pool.imap(read_paths, jobs, chunksize=8):
                frames.update(got)
    S['b2_d'] = d_all
    log(f'B2 start found within {WIN} frames: {(d_all >= 0).mean():.3f} (contact {(d_all[S.set == "contact"] >= 0).mean():.3f}); '
        f'median d {np.median(d_all[d_all >= 0]) if (d_all >= 0).any() else -1}; {len(frames):,} frames read '
        f'[{time.time() - t1:.0f}s]')
    stage = Path(os.environ.get('STAGE', out))
    stage.mkdir(parents=True, exist_ok=True)
    tpath = stage / 'frames.tar'
    with tarfile.open(tpath, 'w') as tf:
        for fp, b in frames.items():
            ti = tarfile.TarInfo(fp)
            ti.size = len(b)
            tf.addfile(ti, io.BytesIO(b))
    if stage != out:
        os.system(f'cp {tpath} {out}/frames.tar')
    S.to_parquet(out / 'eval.parquet')
    (out / 'calib.json').write_text(json.dumps(cal, indent=1))
    log(f'done -> {out} ({len(S)} eval frames, tar {os.path.getsize(out / "frames.tar") / 1e6:.0f} MB) '
        f'[{time.time() - t0:.0f}s]')


if __name__ == '__main__':
    main()
