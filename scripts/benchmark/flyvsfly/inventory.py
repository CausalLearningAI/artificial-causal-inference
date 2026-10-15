"""Fly-vs-Fly inventory (Eyjolfsdottir et al. 2014; CaltechDATA zrznw-w7386).

Run inside a CPU job (inventory.sh) on a copy staged to local disk: python -I inventory.py <staged root> <out dir>
Measures, from the files only:
  - videos: ffprobe stream info + frame count = demuxed packets (ffprobe -count_packets; one packet per frame for mpeg4 / mjpeg), per context and split;
  - labels: <movie>_actions.mat (bouts per fly x behaviour, [start end col3], 1-based inclusive frame indices as
    the release's visualizer.m reads them), <movie>_second_actions.mat (a second annotation, where present),
    traintest.mat (released train / test movie ids, 1-based into movienames);
  - per behaviour x split: frames positive for EITHER fly (the frame-level target), positive rate, bouts (runs of the
    either-fly mask) and per-fly annotated bouts, bout-length median, movies with a positive;
  - alignment: (a) frame counts video vs track vs last labelled frame; (b) video vs track: blob centroids segmented
    from decoded frames matched to trk.data pos x/y at lags -5..+5, scored on the faster fly in frames where the flies are apart and it moves fast (so a 1-frame shift shows);
    (c) labels vs track: bout-onset-triggered averages of track signals (speed for lunge / charge, wing angle for
    wing extension / wing threat) at offsets -15..+15 frames;
  - body length: label-free, the major axis (4 sqrt(largest eigenvalue)) of dark-body blobs from a background-subtracted
    threshold, on frames where the two blobs are well separated; compared with trk 'major axis len'.
Writes inventory.json + example frames (examples/*.png); inventory_md.py renders the markdown."""
import glob
import json
import os
import subprocess
import sys

import cv2
import numpy as np
import scipy.io as sio
from scipy.optimize import linear_sum_assignment

ROOT, OUT = sys.argv[1], sys.argv[2]
BIN = '/nfs/scistore19/locatgrp/rcadei/.conda/envs/crl/bin'
os.makedirs(os.path.join(OUT, 'examples'), exist_ok=True)
RNG = np.random.default_rng(0)
CONTEXTS = {'Aggression': 'Aggression', 'Courtship': 'Courtship', 'BoyMeetsBoy': 'BoyMeetsBoy'}


def mstr(x):
    return str(np.asarray(x).ravel()[0]) if np.asarray(x).size else ''


NCOL = {}  # bout-array column counts seen, per file kind (actions.mat / second_actions.mat)


def load_actions(p):
    d = sio.loadmat(p)
    ncol = NCOL.setdefault(os.path.basename(p).split('_', 1)[1], set())
    behs = [mstr(b) for b in d['behs'].ravel()]
    B = d['bouts']
    bouts = {}
    for f in range(B.shape[0]):
        for j, b in enumerate(behs):
            if j >= B.shape[1]:
                continue
            a = np.asarray(B[f, j])
            a = a.reshape(-1, a.shape[1]) if a.size else np.zeros((0, 3))
            ncol.add(a.shape[1])
            if a.shape[1] == 2:  # some files have [start end] only; visualizer.m pads a zero third column the same way
                a = np.hstack([a, np.zeros((len(a), 1))])
            bouts[(f, b)] = a.astype(np.int64)
    return behs, B.shape, bouts


def ffprobe(p):
    r = subprocess.run([f'{BIN}/ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_packets', '-show_entries',
                        'stream=codec_name,pix_fmt,width,height,r_frame_rate,avg_frame_rate,nb_frames,nb_read_packets,'
                        'duration', '-show_entries', 'format=duration,size,format_name', '-of', 'json', p],
                       capture_output=True, text=True, check=True)
    j = json.loads(r.stdout)
    s, f = j['streams'][0], j['format']
    return dict(codec=s.get('codec_name'), pix_fmt=s.get('pix_fmt'), width=s['width'], height=s['height'],
                r_frame_rate=s.get('r_frame_rate'), avg_frame_rate=s.get('avg_frame_rate'),
                nb_frames_header=s.get('nb_frames'), frames=int(s['nb_read_packets']),
                duration_s=float(f['duration']), bytes=int(f['size']), container=f.get('format_name'))


def runs(mask):
    m = np.concatenate([[0], mask.astype(np.int8), [0]])
    d = np.diff(m)
    s, e = np.where(d == 1)[0], np.where(d == -1)[0]
    return s, e - s


def read_frames(path, idx):
    """Decode the frames in idx (0-based) as grayscale by ONE sequential pass from frame 0 (no seeking: container seeks
    can land on a neighbouring frame, which would fake or hide an alignment offset). Returns ({i: gray}, frames read)."""
    want = set(int(i) for i in idx)
    last = max(want) if want else -1
    cap = cv2.VideoCapture(path)
    out, i = {}, 0
    while i <= last:
        if not cap.grab():
            break
        if i in want:
            ok, fr = cap.retrieve()
            out[i] = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        i += 1
    cap.release()
    return out, i


def blobs(gray, bg, thr):
    """Dark-on-bright foreground blobs: (centroid x, y, area, major axis 4 sqrt(l1), minor 4 sqrt(l2))."""
    fg = (bg.astype(np.int16) - gray.astype(np.int16)) > thr
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(fg, connectivity=8)
    res = []
    for k in range(1, n):
        if st[k, cv2.CC_STAT_AREA] < 30:
            continue
        ys, xs = np.nonzero(lab == k)
        c = np.cov(np.stack([xs, ys]))
        ev = np.sort(np.linalg.eigvalsh(c))[::-1]
        res.append((cen[k][0], cen[k][1], int(st[k, cv2.CC_STAT_AREA]), 4 * np.sqrt(max(ev[0], 0)),
                    4 * np.sqrt(max(ev[1], 0))))
    return sorted(res, key=lambda r: -r[2])


inv = dict(source='CaltechDATA zrznw-w7386 (old id 1893 redirects there), CC0', root=ROOT, contexts={})
# downloads + extraction reports (verification record)
raw = os.path.join(ROOT, 'raw')
rec = json.load(open(os.path.join(raw, 'record_files_zrznw-w7386.json')))
inv['files'] = []
for e in rec['entries']:
    okp = os.path.join(raw, e['key'] + '.ok')
    ok = open(okp).read().split() if os.path.exists(okp) else None
    inv['files'].append(dict(key=e['key'], bytes=int(e['size']), md5_record=e['checksum'].split(':')[1],
                             md5_verified=bool(ok and ok[1] == e['checksum'].split(':')[1])))
inv['extraction'] = {}
for p in sorted(glob.glob(os.path.join(raw, 'extract_*.json'))):
    r = json.load(open(p))
    inv['extraction'][r['zip']] = dict(members=r['members'], extracted=len(r['extracted']), skipped=len(r['skipped']),
                                       bad=r['bad'], deleted_zip=r['deleted_zip'])

for ctx in CONTEXTS:
    cdir = os.path.join(ROOT, ctx)
    tt = sio.loadmat(os.path.join(cdir, 'traintest.mat'))
    names = [mstr(x) for x in tt['movienames'].ravel()]
    split = {}
    for k, s in (('trainids', 'train'), ('testids', 'test')):
        for i in np.asarray(tt[k]).ravel():
            split[names[int(i) - 1]] = s
    C = dict(traintest=dict(behs=[mstr(b) for b in tt['behs'].ravel()], movienames=names,
                            trainids=np.asarray(tt['trainids']).ravel().tolist(),
                            testids=np.asarray(tt['testids']).ravel().tolist(),
                            n_flies=int(np.asarray(tt['n_flies']).ravel()[0]), movie_ext=mstr(tt['movie_ext'])),
             movies={})
    for name in names:
        mdir = os.path.join(cdir, name)
        base = os.path.basename(name)
        vids = sorted(glob.glob(os.path.join(mdir, base + '.mp4')) + glob.glob(os.path.join(mdir, base + '.avi')))
        M = dict(split=split.get(name, 'none'), video=os.path.relpath(vids[0], ROOT) if vids else None)
        if vids:
            M['probe'] = ffprobe(vids[0])
        trk = sio.loadmat(os.path.join(mdir, base + '_track.mat'))['trk']
        tdata = trk['data'][0, 0]
        flag = np.asarray(trk['flag_frames'][0, 0]).ravel()
        tnames = [mstr(x) for x in trk['names'][0, 0].ravel()]
        M['track'] = dict(shape=list(tdata.shape), names=tnames, flag_frames_nonzero=int((flag != 0).sum()),
                          flag_values=np.unique(flag).tolist()[:10], nan_frac=float(np.isnan(tdata).mean()))
        behs, bshape, bouts = load_actions(os.path.join(mdir, base + '_actions.mat'))
        M['behs'] = behs
        M['bouts_cell_shape'] = list(bshape)
        T = tdata.shape[1]
        nfr = M['probe']['frames'] if vids else T
        allb = np.concatenate([b for b in bouts.values() if len(b)]) if any(len(b) for b in bouts.values()) else None
        M['label_frame_range'] = [int(allb[:, 0].min()), int(allb[:, 1].max())] if allb is not None else None
        M['col3_values'] = sorted({int(v) for b in bouts.values() for v in (b[:, 2] if b.shape[1] > 2 else [])})
        M['bad_bouts'] = int(sum(((b[:, 1] < b[:, 0]) | (b[:, 0] < 1) | (b[:, 1] > nfr)).sum()
                                 for b in bouts.values() if len(b)))
        # frame-level masks per behaviour: label frame L (1-based) -> 0-based frame L-1 (visualizer.m convention)
        lab = {}
        for b in behs:
            per_fly = []
            m_any = np.zeros(nfr, bool)
            for f in range(bshape[0]):
                bb = bouts.get((f, b), np.zeros((0, 3), int))
                per_fly.append(len(bb))
                for s, e in bb[:, :2]:
                    m_any[max(s - 1, 0):min(e, nfr)] = True
            st, ln = runs(m_any)
            lens = [int(e - s + 1) for f in range(bshape[0]) for s, e in bouts.get((f, b), np.zeros((0, 3), int))[:, :2]]
            lab[b] = dict(frames=int(m_any.sum()), bouts_any=len(st), bouts_per_fly=per_fly, bout_len_annot=lens,
                          bout_len_any=ln.tolist())
        M['labels'] = lab
        # timeline: first / last labelled frame per behaviour, third-column value counts
        M['timeline'] = {b: [int(min(bb[:, 0].min() for bb in [bouts.get((f, b)) for f in range(bshape[0])] if bb is not None and len(bb))),
                             int(max(bb[:, 1].max() for bb in [bouts.get((f, b)) for f in range(bshape[0])] if bb is not None and len(bb)))]
                         for b in behs if any(len(bouts.get((f, b), [])) for f in range(bshape[0]))}
        M['col3_counts'] = {b: {str(v): int(sum((bouts.get((f, b), np.zeros((0, 3), int))[:, 2] == v).sum() for f in range(bshape[0])))
                                for v in M['col3_values']} for b in behs}
        if 'copulation' in behs:  # where do the other behaviours fall relative to the (long) copulation bout(s)?
            cop = np.concatenate([bouts.get((f, 'copulation'), np.zeros((0, 3), int)) for f in range(bshape[0])])
            if len(cop):
                c0, c1 = int(cop[:, 0].min()), int(cop[:, 1].max())
                rel = {}
                for b in behs:
                    if b == 'copulation':
                        continue
                    bb = np.concatenate([bouts.get((f, b), np.zeros((0, 3), int)) for f in range(bshape[0])])
                    rel[b] = dict(before=int((bb[:, 1] < c0).sum()) if len(bb) else 0,
                                  during=int(((bb[:, 0] >= c0) & (bb[:, 1] <= c1)).sum()) if len(bb) else 0,
                                  after=int((bb[:, 0] > c1).sum()) if len(bb) else 0)
                M['copulation'] = dict(bouts=cop[:, :2].tolist(), first_start=c0, last_end=c1, others=rel)
        sp = os.path.join(mdir, base + '_second_actions.mat')
        if os.path.exists(sp):
            b2, s2, bouts2 = load_actions(sp)
            agree = {}
            for b in b2:
                m1 = np.zeros(nfr, bool)
                m2 = np.zeros(nfr, bool)
                for f in range(min(bshape[0], s2[0])):
                    for s, e in bouts.get((f, b), np.zeros((0, 3), int))[:, :2]:
                        m1[s - 1:e] = True
                    for s, e in bouts2.get((f, b), np.zeros((0, 3), int))[:, :2]:
                        m2[s - 1:e] = True
                inter, uni = int((m1 & m2).sum()), int((m1 | m2).sum())
                agree[b] = dict(frames_first=int(m1.sum()), frames_second=int(m2.sum()), iou=inter / uni if uni else None,
                                col3_second=sorted({int(v) for f in range(s2[0]) for v in bouts2.get((f, b), np.zeros((0, 3), int))[:, 2]}),
                                bouts_second=[len(bouts2.get((f, b), [])) for f in range(s2[0])])
            M['second_annotation'] = dict(behs=b2, agreement=agree)
        C['movies'][name] = M
        print(ctx, name, M.get('probe', {}).get('frames'), 'track', tdata.shape, flush=True)
    inv['contexts'][ctx] = C


# ---------- alignment (b) + body length on a sample of movies ----------
def speed(tdata):
    xy = tdata[:, :, :2]
    v = np.linalg.norm(np.diff(xy, axis=1), axis=2)
    return np.concatenate([v[:, :1] * np.nan, v], axis=1)  # speed[f, t] = |pos[t] - pos[t-1]|


ALIGN = {}
BODY = {}
for ctx, C in inv['contexts'].items():
    names = C['traintest']['movienames']
    pick = [names[i] for i in sorted(RNG.choice(len(names), size=min(3, len(names)), replace=False))]
    for name in pick:
        mdir = os.path.join(ROOT, ctx, name)
        base = os.path.basename(name)
        vid = os.path.join(ROOT, C['movies'][name]['video'])
        tdata = sio.loadmat(os.path.join(mdir, base + '_track.mat'))['trk']['data'][0, 0]
        nfr = C['movies'][name]['probe']['frames']
        sp = speed(tdata)
        T = min(tdata.shape[1], nfr)
        nf = tdata.shape[0]
        xy = tdata[:, :T, :2]
        maj = np.nanmax(tdata[:, :T, 3], axis=0)
        dist = np.linalg.norm(xy[0] - xy[1], axis=1) if nf == 2 else np.full(T, np.inf)
        vmax = np.nanmax(sp[:, :T], axis=0)
        ok = np.all(np.isfinite(xy[:, :, 0]), axis=0) & (np.arange(T) > 50) & (np.arange(T) < T - 50)
        # alignment frames: flies apart (dist > 1.5 x body length) and the faster fly in the movie's top 5 % of speed;
        # 15 windows of 40 consecutive frames around them, only qualifying frames are scored
        sep_ok = ok & (dist > 1.5 * maj)
        fastf = np.where(sep_ok & (vmax > np.nanpercentile(vmax[sep_ok], 95)))[0] if sep_ok.any() else np.array([], int)
        starts = sorted(RNG.choice(fastf, size=min(15, len(fastf)), replace=False)) if len(fastf) else []
        idx = sorted({int(s) + k for s in starts for k in range(-20, 20)} & set(fastf.tolist()))
        # body-length frames: flies far apart (dist > 2.5 x body length), random, up to 200
        farf = np.where(ok & (dist > 2.5 * maj))[0]
        bidx = sorted(RNG.choice(farf, size=min(200, len(farf)), replace=False).tolist()) if len(farf) else []
        bgi = [int(x) for x in np.linspace(100, nfr - 100, 30)]
        frames, _ = read_frames(vid, set(idx) | set(bgi) | set(bidx))
        # background: per-pixel max over 30 frames spread over the movie (flies are dark and move)
        bg = cv2.medianBlur(np.max(np.stack([frames[i] for i in bgi]), 0), 5)
        # foreground threshold: 0.5 x the median darkness (bg - frame) at the track positions (fly centres)
        dark = []
        for i in (idx + bidx)[::5]:
            for f in range(nf):
                x, y = tdata[f, i, :2]
                if np.isfinite(x) and np.isfinite(y) and 0 <= int(y) < bg.shape[0] and 0 <= int(x) < bg.shape[1]:
                    dark.append(int(bg[int(y), int(x)]) - int(frames[i][int(y), int(x)]))
        thr = max(10, 0.5 * float(np.median(dark))) if dark else 30
        LG = range(-5, 6)
        errs = {k: [] for k in LG}
        for i in idx:
            bl = blobs(frames[i], bg, thr)
            if len(bl) < nf:
                continue
            P = np.array([[b[0], b[1]] for b in bl[:nf]])
            fly = int(np.nanargmax(sp[:, i]))  # score the faster fly: its position changes most between frames
            for k in LG:
                j = i + k
                if 0 <= j < T and np.all(np.isfinite(tdata[:, j, :2])):
                    D = np.linalg.norm(P[:, None] - tdata[None, :, j, :2], axis=2)
                    r, c = linear_sum_assignment(D)
                    errs[k].append(float(D[r, c][list(c).index(fly)]))
        body, trk_major = [], []
        for i in bidx:
            bl = blobs(frames[i], bg, thr)
            # each track position must fall within 0.5 body length of its own blob centroid
            for f in range(nf):
                x, y = tdata[f, i, :2]
                dd = [np.hypot(b[0] - x, b[1] - y) for b in bl]
                if dd and min(dd) < 0.5 * maj[i]:
                    b = bl[int(np.argmin(dd))]
                    body.append(b[3])
                    trk_major.append(float(tdata[f, i, 3]))
        med = {k: (float(np.median(v)) if v else None) for k, v in errs.items()}
        good = {k: v for k, v in med.items() if v is not None}
        ALIGN[f'{ctx}/{name}'] = dict(threshold=thr, frames_scored=len(errs[0]), median_err_px_by_lag=med,
                                      best_lag=min(good, key=good.get) if good else None,
                                      median_speed_px_per_frame_scored_fly=float(np.median(vmax[idx])) if idx else None)
        BODY[f'{ctx}/{name}'] = dict(n_blobs=len(body), blob_major_median=float(np.median(body)) if body else None,
                                     blob_major_iqr=[float(np.percentile(body, 25)), float(np.percentile(body, 75))] if body else None,
                                     trk_major_median=float(np.median(trk_major)) if trk_major else None)
        print('align', ctx, name, ALIGN[f'{ctx}/{name}'], BODY[f'{ctx}/{name}'], flush=True)
inv['alignment_video_vs_track'] = ALIGN
inv['body_length'] = BODY

# whole-dataset trk major axis length (pixels), per context
inv['trk_major_axis_len'] = {}
for ctx, C in inv['contexts'].items():
    v = []
    for name in C['traintest']['movienames']:
        t = sio.loadmat(os.path.join(ROOT, ctx, name, os.path.basename(name) + '_track.mat'))['trk']['data'][0, 0]
        v.append(t[:, ::10, 3].ravel())
    v = np.concatenate(v)
    v = v[np.isfinite(v)]
    inv['trk_major_axis_len'][ctx] = dict(median=float(np.median(v)), p10=float(np.percentile(v, 10)),
                                          p90=float(np.percentile(v, 90)))

# ---------- alignment (c): labels vs track, onset-triggered averages ----------
SIG = {'lunge': 'speed', 'charge': 'speed', 'wing_threat': 'wing', 'wing_extension': 'wing', 'tussle': 'speed',
       'circle': 'speed', 'touch': 'mindist'}
OFF = np.arange(-15, 16)
ETA = {}
for ctx, C in inv['contexts'].items():
    acc = {}
    for name in C['traintest']['movienames']:
        mdir = os.path.join(ROOT, ctx, name)
        base = os.path.basename(name)
        trk = sio.loadmat(os.path.join(mdir, base + '_track.mat'))['trk']
        tdata = trk['data'][0, 0]
        tn = [mstr(x) for x in trk['names'][0, 0].ravel()]
        behs, bshape, bouts = load_actions(os.path.join(mdir, base + '_actions.mat'))
        sp = speed(tdata)
        wing = np.nanmax(np.abs(tdata[:, :, [tn.index('wing l ang'), tn.index('wing r ang')]]), axis=2)
        mind = tdata[:, :, tn.index('min fg dist')] if 'min fg dist' in tn else None
        T = tdata.shape[1]
        for (f, b), bb in bouts.items():
            if b not in SIG or f >= tdata.shape[0] or not len(bb):
                continue
            s = {'speed': sp, 'wing': wing, 'mindist': mind}[SIG[b]]
            if s is None:
                continue
            for st in bb[:, 0]:
                t0 = int(st) - 1  # label (1-based) -> track index (0-based)
                if t0 + OFF[0] >= 0 and t0 + OFF[-1] < T:
                    acc.setdefault(b, []).append(s[f, t0 + OFF])
    ETA[ctx] = {}
    for b, a in acc.items():
        a = np.nanmean(np.stack(a), 0)
        d = np.diff(a)
        ETA[ctx][b] = dict(signal=SIG[b], n_onsets=len(acc[b]), offsets=OFF.tolist(),
                           mean=[float(x) for x in a],
                           steepest_change_between_offsets=[int(OFF[np.argmax(np.abs(d))]), int(OFF[np.argmax(np.abs(d)) + 1])])
        print('eta', ctx, b, ETA[ctx][b]['steepest_change_between_offsets'], np.round(a, 2).tolist(), flush=True)
inv['alignment_labels_vs_track'] = ETA

# ---------- example frames (one per context, with track overlay) ----------
for ctx, C in inv['contexts'].items():
    name = C['traintest']['movienames'][0]
    vid = os.path.join(ROOT, C['movies'][name]['video'])
    tdata = sio.loadmat(os.path.join(ROOT, ctx, name, os.path.basename(name) + '_track.mat'))['trk']['data'][0, 0]
    i = int(C['movies'][name]['probe']['frames'] // 3)
    fr = cv2.cvtColor(read_frames(vid, [i])[0][i], cv2.COLOR_GRAY2BGR)
    for f in range(tdata.shape[0]):
        x, y = tdata[f, i, :2]
        if np.isfinite(x):
            cv2.circle(fr, (int(x), int(y)), 4, (0, 0, 255) if f == 0 else (255, 0, 0), -1)
    cv2.imwrite(os.path.join(OUT, 'examples', f'{ctx}_{name.replace("/", "_")}_frame{i}.png'), fr)
inv['bout_array_columns'] = {k: sorted(v) for k, v in NCOL.items()}
json.dump(inv, open(os.path.join(OUT, 'inventory.json'), 'w'), indent=1, default=int)
print('DONE', flush=True)
