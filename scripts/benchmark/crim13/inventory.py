"""CRIM13 inventory (Burgos-Artizzu et al. 2012; CaltechDATA 4emt5-b0t10), top-view videos only.

Run inside a CPU job (inventory.sh): python -I inventory.py <data root on BeeGFS> <local staging dir> <out dir>
Small files (labels .txt, -track.mat, -seek.mat, dinfo.mat) are staged once; each top-view .seq is copied to local
disk by its worker, measured, and deleted. Measures, from the files only:
  - videos: seq header, frame records walked to EOF (count), timestamps (fps, gaps), vs -seek.mat and dinfo.mat;
  - labels (<name>_t.txt, Caltech Behavior Annotator format): vocabulary, streams (annotators), intervals
    [start end] 1-based inclusive, coverage of 1..nFrames (dense or not);
  - tracks (<name>_t-track.mat, Y{1,1,t} = 0, 1 or 2 rows of x, y; row 1 = resident per CRIM13GetInfo.m);
  - alignment: (a) counts video vs track vs labels; (b) video vs track: per-frame motion energy (mean |I_t - I_t-1|,
    1/4-size decode) vs track speed, Pearson r at lags -5..+5 for every video; foreground hit rate at track positions
    vs random positions; (c) labels vs video: bout-onset-triggered motion energy, offsets -15..+15
    (label frame L -> 0-based frame L - 1, the CodeExamples.m / seqIo convention);
  - body length: label-free; background = per-pixel median of 40 frames, |I - bg| > thr, 7x7 opening (drops tail and
    cable), the blob at the track position on frames with exactly one tracked mouse (resident alone); major axis
    4 sqrt(largest eigenvalue) in native pixels.
Writes inventory.json + 3 example frames."""
import glob
import json
import os
import re
import shutil
import zlib
import sys
from multiprocessing import Pool

import cv2
import numpy as np
import scipy.io as sio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import seqio  # noqa: E402

ROOT, LOC, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(os.path.join(OUT, 'examples'), exist_ok=True)
OFF = np.arange(-15, 16)
LAGS = np.arange(-5, 6)


def parse_txt(p):
    lines = open(p, errors='replace').read().splitlines()
    vocab, streams, cur, i = [], {}, None, 0
    while i < len(lines) and not lines[i].startswith('Configuration file'):
        i += 1
    i += 1
    while i < len(lines) and lines[i].strip():
        parts = lines[i].split()
        vocab.append(parts[0])
        i += 1
    for ln in lines[i:]:
        m = re.match(r'^(S\d+):', ln.strip())
        if m:
            cur = m.group(1)
            streams[cur] = []
            continue
        parts = ln.split()
        if cur and len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
            streams[cur].append((int(parts[0]), int(parts[1]), parts[2]))
    return vocab, streams


def runs(mask):
    m = np.concatenate([[0], mask.astype(np.int8), [0]])
    d = np.diff(m)
    s, e = np.where(d == 1)[0], np.where(d == -1)[0]
    return s, e - s


def load_track(p):
    Y = sio.loadmat(p)['Y']
    T = Y.shape[2]
    pos = np.zeros((T, 2, 2))
    nm = np.zeros(T, np.int8)
    for k in range(T):
        a = np.asarray(Y[0, 0, k])
        if a.size:
            a = a.reshape(-1, 2)
            nm[k] = min(len(a), 2)
            pos[k, :nm[k]] = a[:2]
    return pos, nm


def work(job):
    split, sdir, name = job
    seq = os.path.join(ROOT, sdir, name + '_t.seq')
    small = os.path.join(LOC, 'small', sdir)
    tmp = os.path.join(LOC, 'v', name + '_t.seq')
    shutil.copyfile(seq, tmp)
    R = dict(split=split, dir=sdir, name=name, bytes=os.path.getsize(tmp))
    try:
        R['header'] = seqio.header(tmp)
        offs, ns, ts, tail = seqio.index(tmp)
        R['frames'] = int(len(offs))
        R['trailing_bytes'] = int(tail)
        dt = np.diff(ts)
        R['ts'] = dict(fps_from_ts=float(1 / np.median(dt)), dt_median_ms=float(np.median(dt) * 1e3),
                       dt_max_ms=float(dt.max() * 1e3), gaps_gt_1p5x=int((dt > 1.5 * np.median(dt)).sum()),
                       nonmonotonic=int((dt <= 0).sum()), duration_s=float(ts[-1] - ts[0] + np.median(dt)))
        sk = os.path.join(small, name + '_t-seek.mat')
        if os.path.exists(sk):
            so = sio.loadmat(sk)['seek'].ravel().astype(np.int64)
            R['seek_mat'] = dict(n=int(len(so)), equal_to_walk=bool(len(so) == len(offs) and np.all(so == offs)))
        pos, nm = load_track(os.path.join(small, name + '_t-track.mat'))
        R['track'] = dict(T=int(len(nm)), frames_0=int((nm == 0).sum()), frames_1=int((nm == 1).sum()),
                          frames_2=int((nm == 2).sum()),
                          x_range=[float(pos[nm > 0, 0, 0].min()), float(pos[nm > 0, 0, 0].max())] if (nm > 0).any() else None,
                          y_range=[float(pos[nm > 0, 0, 1].min()), float(pos[nm > 0, 0, 1].max())] if (nm > 0).any() else None)
        vocab, streams = parse_txt(os.path.join(small, name + '_t.txt'))
        R['vocab'] = vocab
        R['streams'] = {k: len(v) for k, v in streams.items()}
        N = R['frames']
        S = streams.get('S1', [])
        cov = np.zeros(N + 2, np.int16)
        lab = np.full(N, -1, np.int16)
        for s, e, b in S:
            cov[max(s, 0):min(e, N) + 1] += 1
            if b in vocab:
                lab[max(s - 1, 0):min(e, N)] = vocab.index(b)
        R['labels'] = dict(first=S[0][0] if S else None, last=max(e for _, e, _ in S) if S else None,
                           intervals=len(S), frames_uncovered=int((cov[1:N + 1] == 0).sum()),
                           frames_multiply_covered=int((cov[1:N + 1] > 1).sum()),
                           unknown_names=sorted({b for _, _, b in S if b not in vocab}),
                           beyond_video=int(sum(e > N for _, e, _ in S)))
        per = {}
        for k, b in enumerate(vocab):
            st, ln = runs(lab == k)
            per[b] = dict(frames=int((lab == k).sum()), bouts=len(st), lens=ln.tolist(), onsets=st.tolist())
        R['per_beh'] = per
        # ---- decode every frame at 1/4 size: motion energy ----
        rd = seqio.Reader(tmp, offs, ns)
        prev, me = None, np.full(N, np.nan, np.float32)
        for i in range(N):
            g = rd.gray(i, 4).astype(np.int16)
            if prev is not None and g.shape == prev.shape:
                me[i] = np.abs(g - prev).mean()
            prev = g
        # track speed (resident and intruder summed, only where both frames have the mouse)
        T = min(len(nm), N)
        spd = np.full(N, np.nan)
        for m in range(2):
            ok = (nm[1:T] > m) & (nm[:T - 1] > m)
            v = np.linalg.norm(pos[1:T, m] - pos[:T - 1, m], axis=1)
            v[~ok] = np.nan
            spd[1:T] = np.where(np.isnan(spd[1:T]), v, spd[1:T] + np.nan_to_num(v))
        corr = {}
        for L in LAGS:  # r(track speed at t, motion energy at t + L)
            a = spd[max(0, -L):N - max(0, L)]
            b = me[max(0, L):N - max(0, -L) if L < 0 else N][:len(a)]
            a = a[:len(b)]
            ok = np.isfinite(a) & np.isfinite(b) & (a < 50)  # drop tracker jumps
            corr[int(L)] = float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 100 else None
        R['align_track_vs_video'] = dict(r_by_lag=corr, best_lag=int(max(corr, key=lambda k: corr[k] if corr[k] is not None else -9)))
        # onset-triggered motion energy (label frame L, 1-based -> frame L-1)
        eta = {}
        for b, d in per.items():
            rows = [me[o + OFF] for o in d['onsets'] if o + OFF[0] >= 1 and o + OFF[-1] < N]
            if rows:
                eta[b] = [len(rows), np.nansum(np.stack(rows), 0).tolist()]
        R['eta_sum'] = eta
        # ---- full-res frames: background, foreground hit test, body length ----
        rng = np.random.default_rng(zlib.crc32(name.encode()))  # deterministic per video
        bgi = np.linspace(0, N - 1, 40).astype(int)
        bg = np.median(np.stack([rd.gray(i) for i in bgi]), 0).astype(np.int16)
        two = np.where(nm[:T] == 2)[0]
        one = np.where(nm[:T] == 1)[0]
        hits, rand = [], []
        for i in rng.choice(two, size=min(30, len(two)), replace=False) if len(two) else []:
            g = cv2.GaussianBlur(np.abs(rd.gray(i).astype(np.int16) - bg).astype(np.uint8), (9, 9), 0)
            for m in range(2):
                x, y = pos[i, m]
                if 0 <= int(y) < g.shape[0] and 0 <= int(x) < g.shape[1]:
                    hits.append(int(g[int(y), int(x)]))
                rand.append(int(g[rng.integers(g.shape[0]), rng.integers(g.shape[1])]))
        R['fg_at_track'] = dict(n=len(hits), median_absdiff_at_track=float(np.median(hits)) if hits else None,
                                median_absdiff_random=float(np.median(rand)) if rand else None)
        body, area = [], []
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        for i in rng.choice(one, size=min(25, len(one)), replace=False) if len(one) else []:
            d = np.abs(rd.gray(i).astype(np.int16) - bg)
            fg = cv2.morphologyEx((d > 30).astype(np.uint8), cv2.MORPH_OPEN, ker)
            n, labm, st, cen = cv2.connectedComponentsWithStats(fg, connectivity=8)
            x, y = pos[i, 0]
            if not (0 <= int(y) < fg.shape[0] and 0 <= int(x) < fg.shape[1]):
                continue
            k = labm[int(y), int(x)]
            if k == 0:  # track point not on a blob: take the nearest blob centroid within 25 px
                dd = [np.hypot(*(cen[j] - [x, y])) if st[j, cv2.CC_STAT_AREA] > 300 else 1e9 for j in range(1, n)]
                if not dd or min(dd) > 25:
                    continue
                k = 1 + int(np.argmin(dd))
            ys, xs = np.nonzero(labm == k)
            if len(xs) < 300:
                continue
            ev = np.sort(np.linalg.eigvalsh(np.cov(np.stack([xs, ys]))))[::-1]
            body.append(4 * np.sqrt(ev[0]))
            area.append(len(xs))
        R['body'] = dict(n=len(body), major=[float(b) for b in body], area=[int(a) for a in area])
        rd.close()
    finally:
        os.remove(tmp)
    print(split, name, R.get('frames'), R.get('align_track_vs_video', {}).get('best_lag'), flush=True)
    return R


if __name__ == '__main__':
    inv = dict(source='CaltechDATA 4emt5-b0t10 (old id 1892 redirects there), CC-BY', root=ROOT)
    raw = os.path.join(ROOT, 'raw')
    rec = json.load(open(os.path.join(raw, 'record_files_4emt5-b0t10.json')))
    inv['files'] = []
    for e in rec['entries']:
        okp = os.path.join(raw, e['key'] + '.ok')
        ok = open(okp).read().split() if os.path.exists(okp) else None
        inv['files'].append(dict(key=e['key'], bytes=int(e['size']), md5_record=e['checksum'].split(':')[1],
                                 downloaded=bool(ok), md5_verified=bool(ok and ok[1] == e['checksum'].split(':')[1])))
    inv['extraction'] = {}
    crc_main = {}
    val_seq = []
    for p in sorted(glob.glob(os.path.join(raw, 'extract_*.json'))):
        r = json.load(open(p))
        inv['extraction'][r['zip']] = dict(members=r['members'], extracted=len(r['extracted']),
                                           skipped=len(r['skipped']), bad=r['bad'], deleted_zip=r['deleted_zip'],
                                           skip_regex=r['skip_regex'],
                                           skipped_bytes=int(sum(s[1] for s in r['skipped'])))
        for m in r['extracted'] + r['skipped']:
            if r['zip'] != 'validation.zip':
                crc_main[os.path.basename(m[0])] = (m[-2] if len(m) == 4 else m[1], m[-1])
        if r['zip'] == 'validation.zip':
            val_seq = [(m[0], m[1], m[2]) for m in r['skipped'] if m[0].endswith('.seq')]
    vs = [dict(member=n, size=s, crc=c, main=crc_main.get(os.path.basename(n))) for n, s, c in val_seq]
    inv['validation_seq_vs_main'] = dict(n=len(vs), identical=sum(1 for v in vs if v['main'] and v['main'][0] == v['size']
                                                                    and v['main'][1] == v['crc']),
                                         missing_in_main=[v['member'] for v in vs if not v['main']])
    # stage the small files once
    for sdir in ['train', 'test', 'validation/train', 'validation/test']:
        os.makedirs(os.path.join(LOC, 'small', sdir), exist_ok=True)
        for p in glob.glob(os.path.join(ROOT, sdir, '*')):
            if not p.endswith('.seq') and os.path.isfile(p):
                shutil.copy(p, os.path.join(LOC, 'small', sdir))
    os.makedirs(os.path.join(LOC, 'v'), exist_ok=True)
    jobs = []
    inv['splits'] = {}
    for split in ['train', 'test']:
        seqs = sorted(glob.glob(os.path.join(ROOT, split, '*_t.seq')))
        txts = sorted(glob.glob(os.path.join(LOC, 'small', split, '*_t.txt')))
        names = [os.path.basename(s)[:-6] for s in seqs]
        inv['splits'][split] = dict(top_seq=len(seqs), label_txt=len(txts),
                                    txt_without_seq=sorted(set(os.path.basename(t)[:-6] for t in txts) - set(names)),
                                    seq_without_txt=sorted(set(names) - set(os.path.basename(t)[:-6] for t in txts)))
        jobs += [(split, split, n) for n in names if os.path.exists(os.path.join(LOC, 'small', split, n + '_t.txt'))]
        dp = os.path.join(LOC, 'small', split, 'dinfo.mat')
        if os.path.exists(dp):
            d = sio.loadmat(dp, squeeze_me=True, struct_as_record=False)['d']
            nt = [str(x) for x in np.atleast_1d(d.videos.NamesTop)]
            inv['splits'][split]['dinfo'] = dict(num_videos=int(d.num_videos), num_behaviors=int(d.num_behaviors),
                                                 behaviors=[str(x) for x in np.atleast_1d(d.behaviors.Names)],
                                                 names_top=nt,
                                                 num_frames=[int(x) for x in np.atleast_1d(d.videos.NumFrames)],
                                                 not_on_disk=sorted(set(n[:-6] for n in nt) - set(names)))
    for sub in ['train', 'test']:
        inv['splits'][f'validation/{sub}'] = sorted(os.path.basename(p)[:-6] for p in
                                                   glob.glob(os.path.join(LOC, 'small', 'validation', sub, '*_t.txt')))
        # are the validation label files identical to the main ones?
        same = []
        for n in inv['splits'][f'validation/{sub}']:
            a = open(os.path.join(LOC, 'small', 'validation', sub, n + '_t.txt'), 'rb').read()
            m = [os.path.join(LOC, 'small', s, n + '_t.txt') for s in ['train', 'test']]
            m = [x for x in m if os.path.exists(x)]
            same.append([n, os.path.basename(os.path.dirname(m[0])) if m else None,
                         bool(m and open(m[0], 'rb').read() == a)])
        inv['splits'][f'validation/{sub}_vs_main'] = same
    with Pool(int(os.environ.get('NWORK', 12))) as pool:
        res = pool.map(work, jobs, chunksize=1)
    inv['videos'] = res
    # example frames: train video with 2 mice, test video with 2 mice, one resident-alone frame
    rng = np.random.default_rng(0)
    ex = [(r, 2) for r in [x for x in res if x['split'] == 'train'][:1] + [x for x in res if x['split'] == 'test'][:1]]
    ex += [([x for x in res if x['split'] == 'train'][1], 1)]
    for r, nmice in ex:
        seq = os.path.join(ROOT, r['dir'], r['name'] + '_t.seq')
        pos, nm = load_track(os.path.join(LOC, 'small', r['dir'], r['name'] + '_t-track.mat'))
        cand = np.where(nm == nmice)[0]
        i = int(cand[len(cand) // 2]) if len(cand) else r['frames'] // 2
        rd = seqio.Reader(seq)
        img = cv2.cvtColor(rd.gray(i), cv2.COLOR_GRAY2BGR)
        rd.close()
        for m in range(nm[i]):
            cv2.circle(img, (int(pos[i, m, 0]), int(pos[i, m, 1])), 6, (0, 0, 255) if m == 0 else (255, 0, 0), -1)
        cv2.imwrite(os.path.join(OUT, 'examples', f"{r['split']}_{r['name']}_frame{i}.png"), img)
    json.dump(inv, open(os.path.join(OUT, 'inventory.json'), 'w'), default=int)
    print('DONE', flush=True)
