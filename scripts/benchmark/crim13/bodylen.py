"""CRIM13 body length, second pass (the inventory's |I - bg| blob merged with the head cable, the dark cage walls and
moved bedding: median 170 px, implausible). Label-free, top view, frames with exactly one tracked mouse (resident alone):
background = per-pixel median of 40 frames; foreground = pixels DARKER than the background by > 40 grey levels (the
resident is dark on pale bedding), restricted to the arena (pixels whose background is brighter than the image
median, which drops the dark walls), 11 x 11 elliptical opening (drops tail and cable), the component under the
track point (or nearest within 25 px). Body length = major axis 4 sqrt(largest eigenvalue) and, as a check, the
longest extent of the component along that axis. Videos whose resident is pale would give no blob and are counted.
usage: python -I bodylen.py <data root> <local dir> <out.json>"""
import glob
import json
import os
import shutil
import sys
import zlib
from multiprocessing import Pool

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import seqio  # noqa: E402
import scipy.io as sio  # noqa: E402

ROOT, LOC, OUT = sys.argv[1], sys.argv[2], sys.argv[3]


def load_track(p):  # same as inventory.py: Y{1,1,t} -> (T, 2, 2) positions, number of mice per frame
    Y = sio.loadmat(p)['Y']
    pos = np.zeros((Y.shape[2], 2, 2))
    nm = np.zeros(Y.shape[2], np.int8)
    for k in range(Y.shape[2]):
        a = np.asarray(Y[0, 0, k])
        if a.size:
            a = a.reshape(-1, 2)
            nm[k] = min(len(a), 2)
            pos[k, :nm[k]] = a[:2]
    return pos, nm


def work(job):
    sdir, name = job
    tmp = os.path.join(LOC, name + '_t.seq')
    shutil.copyfile(os.path.join(ROOT, sdir, name + '_t.seq'), tmp)
    try:
        pos, nm = load_track(os.path.join(ROOT, sdir, name + '_t-track.mat'))
        rd = seqio.Reader(tmp)
        N = min(len(rd), len(nm))
        rng = np.random.default_rng(zlib.crc32(name.encode()))
        bg = np.median(np.stack([rd.gray(i) for i in np.linspace(0, N - 1, 40).astype(int)]), 0).astype(np.int16)
        arena = bg > np.median(bg)
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        one = np.where(nm[:N] == 1)[0]
        maj, ext, area = [], [], []
        for i in rng.choice(one, size=min(25, len(one)), replace=False) if len(one) else []:
            g = rd.gray(i).astype(np.int16)
            fg = (((bg - g) > 40) & arena).astype(np.uint8)
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, ker)
            n, lab, st, cen = cv2.connectedComponentsWithStats(fg, connectivity=8)
            x, y = pos[i, 0]
            if not (0 <= int(y) < fg.shape[0] and 0 <= int(x) < fg.shape[1]):
                continue
            k = lab[int(y), int(x)]
            if k == 0:
                dd = [np.hypot(*(cen[j] - [x, y])) if st[j, cv2.CC_STAT_AREA] > 300 else 1e9 for j in range(1, n)]
                if not dd or min(dd) > 25:
                    continue
                k = 1 + int(np.argmin(dd))
            ys, xs = np.nonzero(lab == k)
            if len(xs) < 300:
                continue
            P = np.stack([xs, ys]).astype(float)
            w, v = np.linalg.eigh(np.cov(P))
            proj = v[:, -1] @ (P - P.mean(1, keepdims=True))
            maj.append(float(4 * np.sqrt(w[-1])))
            ext.append(float(proj.max() - proj.min()))
            area.append(int(len(xs)))
        rd.close()
    finally:
        os.remove(tmp)
    return dict(name=name, split=sdir, n=len(maj), major=maj, extent=ext, area=area,
                bg_arena_median=float(np.median(bg[arena])))


if __name__ == '__main__':
    os.makedirs(LOC, exist_ok=True)
    jobs = [(s, os.path.basename(p)[:-6]) for s in ['train', 'test'] for p in sorted(glob.glob(os.path.join(ROOT, s, '*_t.seq')))]
    with Pool(int(os.environ.get('NWORK', 12))) as pool:
        res = pool.map(work, jobs, chunksize=1)
    m = [x for r in res for x in r['major']]
    e = [x for r in res for x in r['extent']]
    per = [np.median(r['major']) for r in res if r['n']]
    summ = dict(videos=len(res), videos_with_blobs=sum(r['n'] > 0 for r in res), blobs=len(m),
                major_median=float(np.median(m)), major_iqr=[float(np.percentile(m, 25)), float(np.percentile(m, 75))],
                extent_median=float(np.median(e)), per_video_major_median_iqr=[float(np.percentile(per, 25)),
                                                                              float(np.median(per)), float(np.percentile(per, 75))])
    json.dump(dict(summary=summ, videos=res), open(OUT, 'w'))
    print(json.dumps(summ))
