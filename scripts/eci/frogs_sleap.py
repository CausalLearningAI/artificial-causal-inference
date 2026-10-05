"""
SLEAP poses of the frogs v1 videos at the 5 fps frames (one npz per video), for the frog foreground mask
(src/eci/foreground.py rule 'frogs', scripts/eci/frogs_background.py).

Input: the re-predicted SLEAP analysis files (data/frogs/v1/experiment.csv 'sleap_h5'; one model for every video),
tracks (1 track, xy, 22 nodes, every 60 fps frame) in 800 px source coordinates.
Output: dataset/frogs/v1/sleap/<observation_id>.npz
    nodes        (n5, 22, 2) float32  x, y of every node at source frames 5, 17, 29, ... (the 5 fps frames of
                                      dataset/frogs/v1/frames; NaN = node not predicted)
    node_names   (22,) str
    source_frames  int                60 fps frames of the video (checked equal to experiment.csv)

Run with a Python that has h5py (the crl environment has none; /usr/bin/python3 does). numpy + h5py only.
Usage: /usr/bin/python3 scripts/eci/frogs_sleap.py [--overwrite]

--stage 44-48 (tadpoles; the default --stage Juv above is unchanged): rows of data/frogs/v1_tadpole/experiment.csv
(scripts/eci/frogs_prepare.py --stage 44-48 --step source), the ORIGINAL SLEAP analysis files (11 tadpole nodes:
Left_Eye, Heart_Center, Tail_Stem, Tail_1 .. Tail_6, Tail_Tip, Right_Eye; source px of the raw frame, which differs per
video). Output dataset/frogs/v1_tadpole/sleap/<observation_id>.npz:
    nodes (n5, 11, 2) float32, scores (n5, 11) float32 (SLEAP point scores, NaN where not predicted or where the
    instance score < MIN_INSTANCE), instance (n5,) float32 (instance score, NaN likewise), node_names, source_frames, width, height,
    centre (n5, 2) float32: body centre for the crop = mean over the 60 fps frames 12 i + 3 .. 12 i + 7 (+-33 ms, no lag)
        of the centroid of the predicted nodes of frames with >= CENTRE_MIN_NODES of the 11 nodes (NaN: none)
at source frames 5, 17, 29, ... (n5 = the number of those frames), and dataset/frogs/v1_tadpole/sleap/quality.csv, one
row per video, measured on ALL 60 fps frames: frac_missing (no node predicted), frac_low (instance score < 0.9, the
lab's TRACK_SELECT_THRES, or missing), frac_ghost (predicted, instance score < MIN_INSTANCE), mean_point_score, mean_instance_score, frac_missing_heart, the SLEAP model
(provenance, centroid model folder name) and the median body length (px, sum of the skeleton segments eyes-midpoint ->
Heart_Center -> Tail_Stem -> Tail_1 .. Tail_6 -> Tail_Tip over frames with all those nodes).
Usage: /usr/bin/python3 scripts/eci/frogs_sleap.py --stage 44-48
"""

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# 60 fps source -> 5 fps frames: frame i of the standardized video (ffmpeg fps=5) is source frame 12 i + 5 (measured:
# the extracted frames match source frame 12 i + 5 best among offsets -12..12 whenever the frog moves)
STEP, OFFSET = 12, 5


# MIN_INSTANCE: frames with a SLEAP instance score below it are treated as not predicted (nodes NaN in the npz, no
# centre). Looked at (crops of frames binned by instance score, 7 videos): below ~0.3 the nodes often sit on nothing
# (FoxP1_304_6: a 'ghost' instance at 0.21-0.27 in half of its frames, the tadpole elsewhere); 0.3-0.7 is mostly the
# real tadpole at the dish rim. 3.2% of all predicted frames are < 0.5; the scores are not bimodal, so this is a
# judgement, not a clean separation.
CENTRE_MIN_NODES, CENTRE_HALF, MIN_INSTANCE = 6, 2, 0.3
AXIS = ['Heart_Center', 'Tail_Stem', 'Tail_1', 'Tail_2', 'Tail_3', 'Tail_4', 'Tail_5', 'Tail_6', 'Tail_Tip']


def main_tadpole(overwrite):
    import json
    out = ROOT / 'dataset/frogs/v1_tadpole/sleap'
    out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(ROOT / 'data/frogs/v1_tadpole/experiment.csv')))
    qual = []
    for r in rows:
        f = out / f'{r["observation_id"]}.npz'
        with h5py.File(r['sleap_h5'], 'r') as h:
            tr = h['tracks'][:]
            names = [n.decode() for n in h['node_names'][:]]
            ps = h['point_scores'][:]
            inst = h['instance_scores'][:]
            prov = json.loads(h['provenance'][()])
        n_src = int(r['source_frames'])
        if tr.shape[:3] != (1, 2, 11) or tr.shape[3] != n_src or inst.shape != (1, n_src):
            raise SystemExit(f'{r["observation_id"]}: tracks {tr.shape}, expected (1, 2, 11, {n_src})')
        xy = np.transpose(tr[0], (2, 1, 0))  # (n_src, 11, 2)
        sc = np.where(np.isnan(xy[..., 0]), np.nan, ps[0].T)
        ins = inst[0]
        miss = np.isnan(xy[..., 0]).all(1)
        ghost = ~miss & ~(ins >= MIN_INSTANCE)
        eyes = (xy[:, names.index('Left_Eye')] + xy[:, names.index('Right_Eye')]) / 2
        chain = np.stack([eyes] + [xy[:, names.index(n)] for n in AXIS], 1)
        bl = np.linalg.norm(np.diff(chain, axis=1), axis=-1).sum(1)
        model = Path(prov.get('model_paths', [''])[0]).parent.name
        qual.append({'observation_id': r['observation_id'], 'group': r['group'], 'session': r['session'],
                     'width': r['width'], 'height': r['height'], 'model': model,
                     'frac_missing': round(float(miss.mean()), 5),
                     'frac_low': round(float((miss | ~(ins >= 0.9)).mean()), 5),
                     'frac_ghost': round(float(ghost.mean()), 5),
                     'frac_missing_heart': round(float(np.isnan(xy[:, names.index('Heart_Center'), 0]).mean()), 5),
                     'mean_point_score': round(float(np.nanmean(sc)), 4),
                     'mean_instance_score': round(float(np.nanmean(np.where(miss, np.nan, ins))), 4),
                     'body_len_px': round(float(np.nanmedian(bl)), 1)})
        print(qual[-1], flush=True)
        if f.exists() and not overwrite:
            continue
        sel = slice(OFFSET, None, STEP)
        xy[ghost] = np.nan
        sc[ghost] = np.nan
        ok = np.isfinite(xy[..., 0]).sum(1) >= CENTRE_MIN_NODES
        cen = np.where(ok[:, None], np.nanmean(np.where(np.isfinite(xy), xy, np.nan), 1), np.nan)
        idx = np.arange(OFFSET, n_src, STEP)
        win = np.clip(idx[:, None] + np.arange(-CENTRE_HALF, CENTRE_HALF + 1)[None], 0, n_src - 1)
        cw = cen[win]  # (n5, 5, 2)
        cnt = np.isfinite(cw[..., 0]).sum(1)
        centre = np.where(cnt[:, None] > 0, np.nansum(np.nan_to_num(cw, nan=0.0), 1) / np.maximum(cnt, 1)[:, None], np.nan)
        tmp = out / f'{r["observation_id"]}.tmp.npz'
        np.savez(tmp, centre=centre.astype(np.float32), nodes=xy[sel].astype(np.float32), scores=sc[sel].astype(np.float32),
                 instance=np.where(miss | ghost, np.nan, ins)[sel].astype(np.float32), node_names=np.array(names),
                 source_frames=n_src, width=int(r['width']), height=int(r['height']))
        tmp.rename(f)
    keys = list(qual[0])
    with open(out / 'quality.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        w.writerows(qual)
    print(f'done: {len(rows)} videos in {out}, quality.csv')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--stage', default='Juv', choices=('Juv', '44-48'))
    args = ap.parse_args()
    if args.stage == '44-48':
        main_tadpole(args.overwrite)
        return
    out = ROOT / 'dataset/frogs/v1/sleap'
    out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(ROOT / 'data/frogs/v1/experiment.csv')))
    for r in rows:
        f = out / f'{r["observation_id"]}.npz'
        if f.exists() and not args.overwrite:
            continue
        with h5py.File(r['sleap_h5'], 'r') as h:
            tr = h['tracks'][:]
            names = np.array([n.decode() for n in h['node_names'][:]])
        n_src = int(r['source_frames'])
        if tr.shape[:3] != (1, 2, len(names)) or tr.shape[3] != n_src:
            raise SystemExit(f'{r["observation_id"]}: tracks {tr.shape}, expected (1, 2, {len(names)}, {n_src})')
        nodes = np.transpose(tr[0][:, :, OFFSET::STEP], (2, 1, 0)).astype(np.float32)  # (n5, nodes, xy)
        tmp = out / f'{r["observation_id"]}.tmp.npz'
        np.savez(tmp, nodes=nodes, node_names=names, source_frames=n_src)
        tmp.rename(f)
        miss = np.isnan(nodes[..., 0])
        print(f'{r["observation_id"]}: {len(nodes)} frames, all nodes missing in {miss.all(1).mean():.4f}, '
              f'mean nodes visible {(~miss).sum(1).mean():.1f}/{len(names)}', flush=True)
    print(f'done: {len(rows)} videos in {out}')


if __name__ == '__main__':
    main()
