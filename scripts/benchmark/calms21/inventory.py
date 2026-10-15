"""
CalMS21 inventory (CPU Slurm job; inventory.sh). Label-free except the label statistics themselves.

1 files      sizes / md5 vs the record (download.py markers), unzip report
2 videos     ffprobe every task1 MP4: codec, size, fps, exact packet count, duration
3 labels     task1 train / test JSONs -> derived/task1_{split}.npz (keypoints float32 (T, 2, 2, 7), scores, labels);
             per label: frames, positive rate, bouts (count, median / mean / p90 length), sequences with a positive;
             keypoint frames vs video frames per sequence
4 tasks 2/3  per behaviour / annotator: sequences, frames, positives; which task1 VIDEO each task2 / task3 sequence is
             (keypoint fingerprint lookup: nose + neck of both mice, rounded to 0.01 px, at 64 evenly spaced frames)
5 alignment  which video frame holds keypoint / label index i: 8 videos (4 test, 4 train), first 9,000 frames decoded
             grey at 512 x 285; grey under the RESIDENT (black) mouse keypoints and under the INTRUDER (white) mouse
             keypoints for video frame i + d, d = -3..3, only frames where that mouse moves > 3 px / frame (full-res
             px). The right d is darkest under the black mouse and brightest under the white mouse.
6 appearance floor / mouse grey levels (per-pixel median background of 30 frames per video; grey under keypoints)
7 examples   3 test frames (investigation, attack, mount; middle of a bout) as JPG, plain and with keypoints
Outputs: OUT/inventory/{inventory.json, inventory.md, examples/*.jpg}, OUT/plan/offset.json, ROOT/derived/*.npz
"""
import json
import os
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calms_common import (ANN, DER, FFMPEG, FFPROBE, H, OUT, PARTS, PLAN, ROOT, TASK1, VID, W, decode_frames,  # noqa
                          log, video_path)
import subprocess  # noqa: E402

INV = OUT / 'inventory'


def probe(path):
    r = subprocess.run([FFPROBE, '-v', 'error', '-select_streams', 'v:0', '-count_packets', '-show_entries',
                        'stream=codec_name,pix_fmt,width,height,r_frame_rate,avg_frame_rate,nb_frames,nb_read_packets,'
                        'duration:format=duration,size', '-of', 'json', str(path)], capture_output=True, text=True)
    j = json.loads(r.stdout)
    s = j['streams'][0]
    s['format_duration'] = float(j['format'].get('duration', 'nan'))
    s['file_size'] = int(j['format'].get('size', 0))
    return s


def bouts(y):
    """runs of True -> lengths"""
    y = np.r_[False, y.astype(bool), False]
    d = np.diff(y.astype(np.int8))
    return np.flatnonzero(d == -1) - np.flatnonzero(d == 1)


def load_json(path):
    log(f'loading {path.name} ({path.stat().st_size / 1e9:.2f} GB)')
    with open(path) as f:
        return json.load(f)


def seq_arrays(d):
    kp = np.asarray(d['keypoints'], np.float32)
    sc = np.asarray(d['scores'], np.float32) if 'scores' in d else None
    an = np.asarray(d['annotations'], np.int16) if 'annotations' in d else None
    return kp, sc, an


def label_stats(seqs, vocab):
    """seqs: {name: labels (T,)} -> per label stats"""
    out = {}
    tot = sum(len(a) for a in seqs.values())
    for name, k in vocab.items():
        b = np.concatenate([bouts(a == k) for a in seqs.values()]) if seqs else np.array([])
        npos = int(sum((a == k).sum() for a in seqs.values()))
        out[name] = {'frames': npos, 'pos_rate': npos / max(tot, 1), 'bouts': int(len(b)),
                     'bout_len_median': float(np.median(b)) if len(b) else None,
                     'bout_len_mean': float(b.mean()) if len(b) else None,
                     'bout_len_p90': float(np.percentile(b, 90)) if len(b) else None,
                     'seqs_with_pos': int(sum((a == k).any() for a in seqs.values()))}
    return out


def fingerprint(kp, idx):
    a = np.round(kp[np.asarray(list(idx))][..., [0, 3]].reshape(-1, 8).astype(np.float64), 2)
    return list(map(tuple, a.tolist()))


def _align_one(job):
    split, seq, kp = job
    n = min(9000, len(kp))
    fr, nd = decode_frames(video_path(split, seq), range(n + 3), size=(512, 285), gray=True, threads=2)
    g = np.stack([fr[i] for i in range(min(nd, n + 3))]).astype(np.float32)
    sx, sy = 512 / W, 285 / H
    res = {'n_decoded': int(len(g))}
    for m, name in ((0, 'resident_black'), (1, 'intruder_white')):
        x, y = kp[:n, m, 0, :] * sx, kp[:n, m, 1, :] * sy  # (n, 7)
        cx, cy = kp[:n, m, 0, :].mean(1), kp[:n, m, 1, :].mean(1)
        sp = np.zeros(n)
        sp[1:-1] = np.hypot(cx[2:] - cx[:-2], cy[2:] - cy[:-2]) / 2
        xi, yi = np.clip(np.round(x).astype(int), 0, 511), np.clip(np.round(y).astype(int), 0, 284)
        t = np.flatnonzero(sp > 3)
        r = {}
        for d in range(-3, 4):
            ok = t[(t + d >= 0) & (t + d < len(g))]
            r[d] = float(g[(ok + d)[:, None], yi[ok], xi[ok]].mean()) if len(ok) else float('nan')
        res[name] = r
        res[f'{name}_n_moving_frames'] = int(len(t))
    # appearance: per-pixel median background of 30 frames, grey under each mouse (all frames)
    bi = np.linspace(0, len(g) - 1, 30).round().astype(int)
    bg = np.median(g[bi], 0)
    res['floor_bg_median_grey'] = float(np.median(bg))
    res['floor_bg_grey_q'] = np.percentile(bg, [5, 25, 50, 75, 95]).round(1).tolist()
    for m, name in ((0, 'resident_black'), (1, 'intruder_white')):
        x = np.clip(np.round(kp[:n, m, 0, :] * sx).astype(int), 0, 511)
        y = np.clip(np.round(kp[:n, m, 1, :] * sy).astype(int), 0, 284)
        res[f'{name}_grey_under_kp_median'] = float(np.median(g[np.arange(n)[:, None], y, x]))
    return seq, res


def main():
    INV.mkdir(parents=True, exist_ok=True)
    (INV / 'examples').mkdir(exist_ok=True)
    PLAN.mkdir(parents=True, exist_ok=True)
    DER.mkdir(parents=True, exist_ok=True)
    inv = {}
    # 1 files
    rec = {e['key']: e for e in json.load(open(ROOT / 'raw' / 'record_files.json'))['entries']}
    inv['files'] = {k: {'record_size': e['size'], 'record_md5': e['checksum'],
                        'verified_marker': (ROOT / 'raw' / f'{k}.ok').exists()} for k, e in rec.items()}
    rep = {}
    for f in sorted((ROOT / 'raw').glob('unzip_report_*.json')):
        rep.update(json.load(open(f)))
    inv['unzip'] = {k: ({kk: vv for kk, vv in v.items() if kk != 'files'} if isinstance(v, dict) else v)
                    for k, v in rep.items()}
    # 2 videos
    vids = {s: sorted(p.stem for p in (VID / s).glob('*.mp4')) for s in ('train', 'test')}
    with Pool(8) as p:
        pr = {s: dict(zip(v, p.map(probe, [video_path(s, q) for q in v]))) for s, v in vids.items()}
    inv['videos'] = {}
    for s, d in pr.items():
        nf = np.array([int(r['nb_read_packets']) for r in d.values()])
        inv['videos'][s] = {
            'n': len(d), 'width': dict(Counter(r['width'] for r in d.values())),
            'height': dict(Counter(r['height'] for r in d.values())),
            'codec': dict(Counter(r['codec_name'] for r in d.values())),
            'pix_fmt': dict(Counter(r['pix_fmt'] for r in d.values())),
            'r_frame_rate': dict(Counter(r['r_frame_rate'] for r in d.values())),
            'avg_frame_rate': dict(Counter(r['avg_frame_rate'] for r in d.values())),
            'frames_total': int(nf.sum()), 'frames_min': int(nf.min()), 'frames_median': float(np.median(nf)),
            'frames_max': int(nf.max()),
            'duration_s_total': float(sum(r['format_duration'] for r in d.values())),
            'duration_min_median': float(np.median([r['format_duration'] for r in d.values()]) / 60),
            'per_video': {q: {'frames': int(r['nb_read_packets']), 'duration_s': r['format_duration'],
                              'fps': r['avg_frame_rate'], 'bytes': r['file_size']} for q, r in d.items()}}
        log(f'videos {s}: ' + json.dumps({k: v for k, v in inv['videos'][s].items() if k != 'per_video'}))
    # 3 task1 labels
    t1 = {}
    fp_index = {}
    for s in ('train', 'test'):
        J = load_json(ANN / 'task1_classic_classification' / f'calms21_task1_{s}.json')
        (grp,) = J.keys()
        kps, scs, ans, names, meta = [], [], [], [], {}
        for key, d in J[grp].items():
            kp, sc, an = seq_arrays(d)
            q = key.split('/')[-1]
            names.append(q)
            kps.append(kp)
            scs.append(sc)
            ans.append(an)
            meta[q] = d.get('metadata', {})
        del J
        lens = np.array([len(a) for a in ans])
        np.savez(DER / f'task1_{s}.npz', names=np.array(names), lens=lens, keypoints=np.concatenate(kps),
                 scores=np.concatenate(scs), labels=np.concatenate(ans))
        vocab = meta[names[0]].get('vocab', {n: i for i, n in enumerate(TASK1)})
        assert all(m.get('vocab', vocab) == vocab for m in meta.values())
        seqs = dict(zip(names, ans))
        st = label_stats(seqs, vocab)
        vf = {q: inv['videos'][s]['per_video'].get(q, {}).get('frames') for q in names}
        t1[s] = {'group_key': grp, 'n_seqs': len(names), 'frames': int(lens.sum()), 'vocab': vocab,
                 'kp_shape_per_frame': list(kps[0].shape[1:]), 'score_shape_per_frame': list(scs[0].shape[1:]),
                 'label_values': sorted(np.unique(np.concatenate(ans)).tolist()),
                 'annotator_ids': dict(Counter(str(m.get('annotator-id', m.get('annotator_id'))) for m in meta.values())),
                 'labels': st,
                 'kp_x_range': [float(min(k[:, :, 0].min() for k in kps)), float(max(k[:, :, 0].max() for k in kps))],
                 'kp_y_range': [float(min(k[:, :, 1].min() for k in kps)), float(max(k[:, :, 1].max() for k in kps))],
                 'video_minus_kp_frames': dict(Counter(int(vf[q] - len(a)) if vf[q] is not None else None
                                                       for q, a in seqs.items())),
                 'seqs_without_video': [q for q in names if vf[q] is None],
                 'videos_without_seq': [q for q in inv['videos'][s]['per_video'] if q not in seqs],
                 'per_seq': {q: {'frames': int(len(a)), **{n: int((a == k).sum()) for n, k in vocab.items()}}
                             for q, a in seqs.items()}}
        log(f'task1 {s}: ' + json.dumps({k: v for k, v in t1[s].items() if k != 'per_seq'}))
        for q, kp in zip(names, kps):
            for j, f in enumerate(fingerprint(kp, range(len(kp)))):
                fp_index[f] = (s, q, j)
        if s == 'test':
            kp_test = dict(zip(names, kps))
            an_test = seqs
        else:
            kp_train = dict(zip(names, kps))
    inv['task1'] = t1
    # 4 tasks 2 / 3
    other = {}
    for task, zdir, files in (('task2', 'task2_annotation_styles', ('train', 'test')),
                              ('task3', 'task3_new_behaviors', ('train', 'test'))):
        for s in files:
            J = load_json(ANN / zdir / f'calms21_{task}_{s}.json')
            for grp, D in J.items():
                rows = {}
                for key, d in D.items():
                    kp, sc, an = seq_arrays(d)
                    idx = np.linspace(0, len(kp) - 1, 64).round().astype(int)
                    hits = [fp_index.get(f) for f in fingerprint(kp, idx)]
                    hv = [(h[0], h[1], h[2] - i) for h, i in zip(hits, idx) if h is not None]
                    best = Counter(hv).most_common(1)
                    vocab = d.get('metadata', {}).get('vocab')
                    rows[key.split('/')[-1]] = {
                        'frames': int(len(kp)), 'vocab': vocab,
                        'pos': {n: int((an == k).sum()) for n, k in (vocab or {}).items()} if an is not None else None,
                        'fingerprint_hits': len(hv), 'match': (list(best[0][0]) + [best[0][1]]) if best else None}
                other.setdefault(task, {}).setdefault(grp, {})[s] = rows
                nm = sum(1 for r in rows.values() if r['match'] and r['match'][3] >= 48)
                log(f'{task} {grp} {s}: {len(rows)} seqs, {sum(r["frames"] for r in rows.values())} frames, '
                    f'{nm} matched to a task1 video (>= 48 / 64 fingerprints)')
            del J
    inv['tasks23'] = other
    # 5 + 6 alignment and appearance
    rng = np.random.default_rng(0)
    pick = [('test', q, kp_test[q]) for q in rng.choice(sorted(kp_test), 4, replace=False)]
    pick += [('train', q, kp_train[q]) for q in rng.choice(sorted(kp_train), 4, replace=False)]
    with Pool(8) as p:
        A = dict(p.map(_align_one, pick))
    darkest = Counter(min(r['resident_black'], key=r['resident_black'].get) for r in A.values())
    bright = Counter(max(r['intruder_white'], key=r['intruder_white'].get) for r in A.values())
    mb = {d: float(np.mean([r['resident_black'][d] for r in A.values()])) for d in range(-3, 4)}
    mw = {d: float(np.mean([r['intruder_white'][d] for r in A.values()])) for d in range(-3, 4)}
    off = min(mb, key=mb.get)
    info = {'offset': int(off), 'rule': 'video frame = keypoint / label index + offset',
            'mean_grey_under_moving_black_kp': mb, 'mean_grey_under_moving_white_kp': mw,
            'best_white': int(max(mw, key=mw.get)), 'per_video_darkest_black': dict(darkest),
            'per_video_brightest_white': dict(bright), 'videos': {q: A[q] for q in A}}
    info['agree'] = info['best_white'] == off
    (PLAN / 'offset.json').write_text(json.dumps(info, indent=1))
    inv['alignment'] = info
    log('alignment: ' + json.dumps({k: v for k, v in info.items() if k != 'videos'}))
    # 7 examples
    from PIL import Image, ImageDraw
    ex = []
    for name, k in (('investigation', 1), ('attack', 0), ('mount', 2)):
        cand = []
        for q, a in sorted(an_test.items()):
            b = np.r_[False, a == k, False].astype(np.int8)
            st_, en = np.flatnonzero(np.diff(b) == 1), np.flatnonzero(np.diff(b) == -1)
            cand += [(int(e - s_), q, int((s_ + e) // 2)) for s_, e in zip(st_, en)]
        L_, q, i = sorted(cand)[-len(cand) // 2]  # a median-long bout
        fr, _ = decode_frames(video_path('test', q), [i + off])
        im = Image.fromarray(fr[i + off])
        im.save(INV / 'examples' / f'{name}_{q}_kp{i}.jpg', quality=92)
        dr = ImageDraw.Draw(im)
        for m, col in ((0, (255, 60, 60)), (1, (60, 160, 255))):
            for pt in range(7):
                x, y = kp_test[q][i, m, 0, pt], kp_test[q][i, m, 1, pt]
                dr.ellipse([x - 4, y - 4, x + 4, y + 4], outline=col, width=2)
        dr.text((6, 6), f'{q} kp index {i} (video frame {i + off}) label {name}; red = resident, blue = intruder',
                fill=(255, 255, 0))
        im.save(INV / 'examples' / f'{name}_{q}_kp{i}_overlay.jpg', quality=92)
        ex.append({'label': name, 'seq': q, 'kp_index': i, 'video_frame': i + off, 'bout_len': L_})
    inv['examples'] = ex
    (INV / 'inventory.json').write_text(json.dumps(inv, indent=1, default=str))
    log('done')


if __name__ == '__main__':
    main()
