"""
CalMS21 ECI - frame plans (CPU, seconds; run inside a job). Rules: eci_res.sh.

Grid: label / keypoint index i on a 5 fps grid (i % 6 == 0) whose video frame v = i + offset leaves room for the 2 s lag
and the V-JEPA clip (v - 60 >= 0, v - 56 >= 0, v + 64 <= last video frame) and i < the sequence's label length.
    train   task1 TRAIN videos (labels never read): NTRAIN + NCALIB grid frames drawn uniformly over all train videos'
            grid frames (rng 0), role 'train' / 'calib' assigned at random.
    eval    task1 TEST videos, stratified with known inclusion probabilities (inverse-probability weights w, as MABe22):
                S_att   an attack positive within +-6 frames (0.2 s): all, capped at the stratum size
                S_mnt   not S_att, a mount positive within +-6 frames
                S_inv   neither, investigation positive at the frame
                S_rest  the rest (other)
            label columns = task1 labels at the frame (attack, investigation, mount, other).
Outputs: OUT/plan/{train_frames.parquet, eval_frames.parquet, eval_plan.json}
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calms_common import DER, OUT, PLAN, TASK1, log, offset  # noqa: E402

NTRAIN, NCALIB = 10000, 2000
WANT = {'att': 2500, 'mnt': 2500, 'inv': 2500, 'rest': 2500}
SCORED = ['attack', 'investigation', 'mount']   # pre-registered after the inventory (eci_res.sh)


def grid(split, off, nfr):
    Z = np.load(DER / f'task1_{split}.npz')
    names, lens = list(Z['names']), Z['lens']
    st = np.r_[0, np.cumsum(lens)]
    rows = []
    for k, q in enumerate(names):
        N = nfr[q]['frames']
        i = np.arange(0, lens[k], 6)
        v = i + off
        ok = (v - 60 >= 0) & (v + 64 <= N - 1)
        lab = Z['labels'][st[k]:st[k + 1]]
        for a in i[ok]:
            rows.append((q, int(a), int(lab[a])))
    return pd.DataFrame(rows, columns=['video', 'i', 'lab']), Z, names, st


def main():
    PLAN.mkdir(parents=True, exist_ok=True)
    off = offset()
    inv = json.loads((OUT / 'inventory' / 'inventory.json').read_text())
    vocab = inv['task1']['test']['vocab']
    # train
    G, _, _, _ = grid('train', off, inv['videos']['train']['per_video'])
    rng = np.random.default_rng(0)
    take = np.sort(rng.choice(len(G), NTRAIN + NCALIB, replace=False))
    T = G.iloc[take][['video', 'i']].reset_index(drop=True)
    role = np.array(['train'] * NTRAIN + ['calib'] * NCALIB)
    rng.shuffle(role)
    T['role'] = role
    T = T.sort_values(['video', 'i']).reset_index(drop=True)
    T.to_parquet(PLAN / 'train_frames.parquet')
    log(f'train plan: {T.role.value_counts().to_dict()} from {T.video.nunique()} videos (grid {len(G)})')
    # eval
    U, Z, names, st = grid('test', off, inv['videos']['test']['per_video'])
    near = {}
    for b in ('attack', 'mount'):
        m = np.zeros(len(U), bool)
        for k, q in enumerate(names):
            lab = Z['labels'][st[k]:st[k + 1]] == vocab[b]
            nb = np.convolve(lab.astype(float), np.ones(13), 'same') > 0
            sel = (U.video == q).values
            m[sel] = nb[U.i.values[sel]]
        near[b] = m
    for b in TASK1:
        U[b] = U.lab.values == vocab[b]
    U['stratum'] = np.where(near['attack'], 'att', np.where(near['mount'], 'mnt',
                                                              np.where(U.investigation, 'inv', 'rest')))
    parts, info = [], {'population': 'task1 test videos x 5 fps grid with lag / clip room', 'N_population': len(U),
                       'offset': off, 'strata': {}}
    for s, n in WANT.items():
        Pp = U[U.stratum == s]
        n = min(n, len(Pp))
        tk = Pp.iloc[np.sort(rng.choice(len(Pp), n, replace=False))].copy()
        tk['w'] = len(Pp) / n
        parts.append(tk)
        info['strata'][s] = {'N': int(len(Pp)), 'n': int(n), 'w': float(len(Pp) / n)}
    E = pd.concat(parts).sort_values(['video', 'i']).reset_index(drop=True).drop(columns='lab')
    E.to_parquet(PLAN / 'eval_frames.parquet')
    info.update(n_frames=len(E), n_videos=int(E.video.nunique()), scored_labels=SCORED,
                weighted_base_rate={b: float(np.average(E[b], weights=E.w)) for b in TASK1},
                population_base_rate={b: float(U[b].mean()) for b in TASK1},
                n_pos={b: int(E[b].sum()) for b in TASK1},
                n_videos_with_pos={b: int(E.groupby('video')[b].any().sum()) for b in TASK1},
                train={'n_train': NTRAIN, 'n_calib': NCALIB, 'n_videos': int(T.video.nunique()),
                       'grid_population': int(len(G))})
    (PLAN / 'eval_plan.json').write_text(json.dumps(info, indent=1))
    log('eval plan: ' + json.dumps(info))


if __name__ == '__main__':
    main()
