"""
CalMS21 - mouse body length in NATIVE pixels, label-free, and the patches-per-body-length of every arm (addendum (1) of
eci_res.sh; CPU job). Keypoints are used ONLY as a check, never to select frames or set the value.

Label-free measure (resident = the black mouse, the animal the dark pixel cue of the mask isolates): per video, the
per-pixel median background of 32 evenly spaced frames; on 24 plan frames per video the dark-cue pixels (grey < 60 and
> 40 below the background, the mask's pixel cue at native resolution), a 7 x 7 binary opening (removes the tail and
bedding specks), connected components; a frame counts when it holds exactly ONE component >= 800 px and that
component's area is within 0.6-1.4 x the video's median such area (an isolated, unmerged, unoccluded resident).
Body length = full major axis of the component's moment ellipse = 4 sqrt(largest eigenvalue of the pixel-coordinate
covariance). Check: keypoint nose -> tail base straight distance of the resident on the same frames.
patches per body length = L_native x (input side / 1024 canvas) / patch px, per arm (canvas = native scale).
The low-detail control's source side (addendum (2)): lo_side = round(1024 x 40 / L_native), so that the resident is
~40 px long in the shrunk source -> plan/lo_side.txt. Outputs: OUT/bodylen/bodylen.json, plan/lo_side.txt
"""
import json
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calms_common import DER, OUT, PLAN, decode_frames, log, stage_dir, video_path  # noqa: E402

ARMS = {'D2-32': (448, 14), 'D2-64': (896, 14), 'D3-32': (512, 16), 'D3-64': (1024, 16), 'VJ-32': (512, 16),
        'VJ-64': (1024, 16), 'D3-129 (stress)': (2064, 16)}
TARGET = 40.0


def one(job):
    split, q, idx, nfr, stage = job
    loc = Path(stage) / f'{q}_bl.mp4'
    shutil.copyfile(video_path(split, q), loc)
    bgi = np.linspace(0, nfr - 1, 32).round().astype(int)
    fr, _ = decode_frames(loc, set(bgi.tolist()) | set(idx.tolist()), gray=True, threads=2)
    loc.unlink()
    bg = np.median(np.stack([fr[x] for x in bgi]).astype(np.float32), 0)
    out = []
    for i in idx:
        g = fr[i].astype(np.float32)
        dk = ndi.binary_opening((g < 60) & (bg - g > 40), np.ones((7, 7), bool))
        lab, n = ndi.label(dk)
        if n == 0:
            out.append((int(i), 0, np.nan, 0))
            continue
        area = ndi.sum(dk, lab, range(1, n + 1))
        big = np.flatnonzero(area >= 800)
        if len(big) != 1:
            out.append((int(i), len(big), np.nan, 0))
            continue
        yy, xx = np.nonzero(lab == big[0] + 1)
        ev = np.linalg.eigvalsh(np.cov(np.stack([xx, yy]).astype(np.float64)))
        out.append((int(i), 1, float(4 * np.sqrt(ev[-1])), int(area[big[0]])))
    return split, q, out


def main():
    stage = stage_dir()
    inv = json.loads((OUT / 'inventory' / 'inventory.json').read_text())
    jobs = []
    for set_, split in (('eval', 'test'), ('train', 'train')):
        P = pd.read_parquet(PLAN / f'{set_}_frames.parquet')
        vids = sorted(P.video.unique())
        if split == 'train':
            vids = list(np.random.default_rng(0).choice(vids, 12, replace=False))
        for q in vids:
            ii = P[P.video == q].i.values
            ii = ii[np.linspace(0, len(ii) - 1, min(24, len(ii))).round().astype(int)]
            jobs.append((split, q, np.unique(ii), inv['videos'][split]['per_video'][q]['frames'], str(stage)))
    with Pool(16) as p:
        R = p.map(one, jobs)
    rows = []
    for split, q, out in R:
        Z = np.load(DER / f'task1_{split}.npz')
        names = list(Z['names'])
        st = np.r_[0, np.cumsum(Z['lens'])][names.index(q)]
        for i, nb, L, a in out:
            k = Z['keypoints'][st + i, 0]  # resident (2, 7): x / y rows
            kp = float(np.hypot(k[0, 0] - k[0, 6], k[1, 0] - k[1, 6]))
            rows.append({'split': split, 'video': q, 'i': i, 'n_big': nb, 'len_major': L, 'area': a, 'kp_nose_tail': kp})
    D = pd.DataFrame(rows)
    D['ok'] = D.len_major.notna()
    med_area = D[D.ok].groupby('video').area.median()
    D['iso'] = D.ok & (D.area / D.video.map(med_area)).between(0.6, 1.4)
    S = D[D.iso]
    L = float(S.len_major.median())
    lo = int(round(1024 * TARGET / L))
    res = {'n_frames_tried': len(D), 'n_isolated': int(len(S)), 'n_videos': int(S.video.nunique()),
           'body_len_native_px_median': L, 'body_len_iqr': S.len_major.quantile([.25, .75]).round(1).tolist(),
           'kp_nose_tail_base_px_median_same_frames': float(S.kp_nose_tail.median()),
           'kp_nose_tail_base_iqr': S.kp_nose_tail.quantile([.25, .75]).round(1).tolist(),
           'kp_nose_tail_base_px_median_all_frames': float(D.kp_nose_tail.median()),
           'corr_major_vs_kp': float(np.corrcoef(S.len_major, S.kp_nose_tail)[0, 1]),
           'per_video_median': S.groupby('video').len_major.median().round(1).to_dict(),
           'patches_per_body_length': {a: L * s / 1024 / p for a, (s, p) in ARMS.items()},
           'patches_per_body_length_kp': {a: float(S.kp_nose_tail.median()) * s / 1024 / p for a, (s, p) in ARMS.items()},
           'lo_target_px': TARGET, 'lo_side': lo,
           'lo_source_body_len_px': L * lo / 1024,
           'mabe22_reference': 'MABe22 r3/bodylen.json: nose-tail base straight 44.6 px on the 512 keypoint scale = '
                               '19.5 px in the released 224-px videos; the 40 px target is the coordinator\'s value'}
    (OUT / 'bodylen').mkdir(parents=True, exist_ok=True)
    (OUT / 'bodylen' / 'bodylen.json').write_text(json.dumps(res, indent=1))
    D.to_csv(OUT / 'bodylen' / 'frames.csv', index=False)
    (PLAN / 'lo_side.txt').write_text(f'{lo}\n# bodylen.py: round(1024 x {TARGET} / {L:.1f})\n')
    log(json.dumps(res))


if __name__ == '__main__':
    main()
