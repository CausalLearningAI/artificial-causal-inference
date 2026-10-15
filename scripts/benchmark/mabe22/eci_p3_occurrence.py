"""
P3 (analysis only): how often does one MABe22 mouse-triplet frame hold TWO simultaneous occurrences of a pair behaviour?
Keypoints sk[clip]['keypoints'] (1800, 3, 12, 2), 512-px scale, (y, x); kp 0 = nose, kp 9 = tail base. The keypoint row
index IS the label index i (label_array[:, fm[clip][0] + i]); no shift is applied (the "+1" of video = label + 1 only
concerns the 1,801 video frames, not keypoints). Missing keypoint = NaN or exactly (0, 0); a frame is excluded from every
denominator unless all 3 noses (and, for the oral-genital definition, all 3 tail bases) are valid.
Groups: oral_oral_contact==1, oral_genital_contact==1, all frames, negative (that behaviour ==0), negative_all (all 9
BEHAVIOURS ==0). NaN labels are excluded from label-defined groups. All submission clips are used (no video needed).
Run in a Slurm CPU job: python -I scripts/benchmark/mabe22/eci_p3_occurrence.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eci_common import BEHAVIOURS, OUT, RAW, log  # noqa: E402

TS = [5, 7, 10]
OO, OG = 'oral_oral_contact', 'oral_genital_contact'
PAIRS = [(0, 1), (0, 2), (1, 2)]
ORD = [(a, b) for a in range(3) for b in range(3) if a != b]


def valid(p):  # p (T, 3, 2)
    return np.isfinite(p).all(-1) & ~(p == 0).all(-1)


def main():
    sub = np.load(RAW / 'mouse_submission_labels.npy', allow_pickle=True).item()
    fm = np.load(RAW / 'mouse_frame_number_map.npy', allow_pickle=True).item()
    sk = np.load(RAW / 'mouse_submission_keypoints.npy', allow_pickle=True).item()['sequences']
    vocab = list(sub['vocabulary'])
    L = sub['label_array']
    log(f'vocab {vocab}; label_array {L.shape}; n clips fm {len(fm)} kp {len(sk)}')
    R = {b: vocab.index(b) for b in BEHAVIOURS}
    cols = {'nn': [], 'og_ord': [], 'okn': [], 'okt': []}
    lab = {b: [] for b in BEHAVIOURS}
    for c in sorted(fm):
        a, b = fm[c]
        k = sk[c]['keypoints'].astype(np.float64)
        assert k.shape == (b - a, 3, 12, 2), (c, k.shape, a, b)
        nose, tb = k[:, :, 0], k[:, :, 9]
        vn, vt = valid(nose), valid(tb)
        dnn = np.stack([np.linalg.norm(nose[:, p] - nose[:, q], axis=-1) for p, q in PAIRS], 1)
        dog = np.stack([np.linalg.norm(nose[:, p] - tb[:, q], axis=-1) for p, q in ORD], 1)
        cols['nn'].append(dnn); cols['og_ord'].append(dog)
        cols['okn'].append(vn.all(1)); cols['okt'].append(vn.all(1) & vt.all(1))
        for bn in BEHAVIOURS:
            lab[bn].append(L[R[bn], a:b])
    D = {k: np.concatenate(v) for k, v in cols.items()}
    lab = {k: np.concatenate(v) for k, v in lab.items()}
    n = len(D['okn'])
    log(f'total frames {n}')
    allfin = np.all([np.isfinite(lab[b]) for b in BEHAVIOURS], 0)
    neg_all = allfin & np.all([lab[b] == 0 for b in BEHAVIOURS], 0)
    groups = {'oral_oral': lab[OO] == 1, 'oral_genital': lab[OG] == 1, 'all': np.ones(n, bool),
              'negative_oral_oral': lab[OO] == 0, 'negative_oral_genital': lab[OG] == 0, 'negative_all9': neg_all}
    res = {'labels_used': [OO, OG], 'vocab': vocab, 'n_frames_total': n,
           'indexing': 'keypoint row i == label index i (fm[clip][0]+i); no shift; coords order irrelevant for distances',
           'missing_rule': 'NaN or (0,0); frame excluded unless all 3 noses valid (and all 3 tail bases for oral_genital)',
           'groups': {}}
    unord_idx = [[ORD.index((p, q)), ORD.index((q, p))] for p, q in PAIRS]
    for g, m in groups.items():
        r = {'n_labelled': int(m.sum())}
        for defn, ok, key in (('nose_nose', 'okn', 'nn'), ('nose_tailbase', 'okt', 'og_ord')):
            mm = m & D[ok]
            r[defn] = {'n_used': int(mm.sum()), 'n_excluded_missing': int((m & ~D[ok]).sum())}
            for T in TS:
                if key == 'nn':
                    cnt = (D['nn'][mm] < T).sum(1)
                    variants = {'unordered_pairs': cnt}
                else:
                    d = D['og_ord'][mm]
                    variants = {'ordered_pairs': (d < T).sum(1),
                                'unordered_pairs': np.stack([(d[:, i] < T).any(1) for i in unord_idx], 1).sum(1)}
                for vn_, cnt in variants.items():
                    nn_ = max(int(mm.sum()), 1)
                    maxc = 3 if vn_ == 'unordered_pairs' else 6
                    r[defn].setdefault(f'T{T}', {})[vn_] = {
                        'share_ge1': float((cnt >= 1).sum() / nn_), 'share_ge2': float((cnt >= 2).sum() / nn_),
                        'dist': {str(j): float((cnt == j).sum() / nn_) for j in range(maxc + 1)},
                        'n_ge2': int((cnt >= 2).sum())}
        res['groups'][g] = r
        log(f'{g}: n={r["n_labelled"]} nn T7 {r["nose_nose"]["T7"]} | og T7 {r["nose_tailbase"]["T7"]}')
    out = OUT / 'p3'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'occurrence.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
