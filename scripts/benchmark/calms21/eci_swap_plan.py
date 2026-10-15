"""
CalMS21 ECI - ROLE SWAP (rules: eci_swap.sh) - frame plans (CPU job; reads the 2 GB task-3 JSONs). Run with
CALMS_OUT=results/vision/benchmark_calms21/swap so PLAN = swap/plan.

SET NAMES ARE SWAPPED ON PURPOSE. The shared encoders (eci_encode.py, eci_p2_encode.py) map --set train -> task1 TRAIN
videos and anything else -> task1 TEST videos, so in this experiment:
    PLAN/eval_frames.parquet   = the SAE POOL: task1 TEST videos (19), labels never read; 10,000 'train' + 2,000 'calib'
                                 frames uniform over the 5 fps grid (i % 6 == 0) with lag / clip room (rng 0), as the
                                 grid's plan; plus orbit_pool (bool) = a fixed random subset of 4,000 train + 1,000 calib
                                 (rng 1), the reduced pool of the orbit arm S4 and its matched control S1r.
    PLAN/train_frames.parquet  = the EVAL frames: task1 TRAIN videos (70), every video frame i (30 fps grid) with
                                 60 <= i and i + 64 <= N - 1, stratified with inverse-probability weights w.
Labels on eval frames: task1 attack / investigation / mount (every frame of every train video), and the task-3 labels
sniff_face, approach, mount_attempt, intromission: in_<b> = the frame lies in a task-3 <b> sequence (each sequence
is one whole task1 train video, offset 0, fingerprint-matched in the inventory); <b> = its binary label there (False
elsewhere; a <b> label is only ever scored on its in_<b> frames).
Strata (disjoint, first match wins; 'near' = a positive within +-6 frames):
    ma_near 1500, in_near 1500, ap_near 1500, sf_near 2500   (task-3 labels, rarest first)
    at_near 1500, mo_near 1500, inv 1500                    (task1 attack-near, mount-near, investigation positive)
    ma_rest 1000, in_rest 1000, ap_rest 1000, sf_rest 1000   (rest of the frames of each label's own videos)
    rest 1000                                               (everything else)
    n = min(cap, stratum size), w = N_stratum / n.
Outputs: PLAN/{train_frames.parquet, eval_frames.parquet, swap_plan.json}, plus offset.json / mask_rule.txt copied
from the grid's plan and inventory/inventory.json copied (the encoders read them under OUT).
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calms_common import ANN, DER, OUT, PLAN, REPO, log  # noqa: E402

GRID_OUT = REPO / 'results/vision/benchmark_calms21'
T3 = ['sniff_face', 'approach', 'mount_attempt', 'intromission']
T1 = ['attack', 'investigation', 'mount']
SHORT = {'mount_attempt': 'ma', 'intromission': 'in', 'approach': 'ap', 'sniff_face': 'sf'}
CAP_NEAR = {'mount_attempt': 1500, 'intromission': 1500, 'approach': 1500, 'sniff_face': 2500}
CAP_T1 = {'at_near': 1500, 'mo_near': 1500, 'inv': 1500}
CAP_REST = 1000
NTRAIN, NCALIB, NORB_T, NORB_C = 10000, 2000, 4000, 1000


def near(y):
    return np.convolve(y.astype(float), np.ones(13), 'same') > 0


def main():
    assert str(OUT).endswith('benchmark_calms21/swap'), OUT
    PLAN.mkdir(parents=True, exist_ok=True)
    for f in ('offset.json', 'mask_rule.txt'):
        shutil.copyfile(GRID_OUT / 'plan' / f, PLAN / f)
    (OUT / 'inventory').mkdir(parents=True, exist_ok=True)
    shutil.copyfile(GRID_OUT / 'inventory' / 'inventory.json', OUT / 'inventory' / 'inventory.json')
    inv = json.loads((OUT / 'inventory' / 'inventory.json').read_text())
    assert json.loads((PLAN / 'offset.json').read_text())['offset'] == 0
    vocab = inv['task1']['train']['vocab']
    # ------------------------------------------------------------------ SAE pool: task1 TEST videos, labels unread
    Zt = np.load(DER / 'task1_test.npz')
    nft = inv['videos']['test']['per_video']
    rows = []
    for q, L in zip(Zt['names'], Zt['lens']):
        i = np.arange(0, L, 6)
        ok = (i - 60 >= 0) & (i + 64 <= nft[q]['frames'] - 1)
        rows += [(q, int(a)) for a in i[ok]]
    G = pd.DataFrame(rows, columns=['video', 'i'])
    rng = np.random.default_rng(0)
    take = np.sort(rng.choice(len(G), NTRAIN + NCALIB, replace=False))
    S = G.iloc[take].reset_index(drop=True)
    role = np.array(['train'] * NTRAIN + ['calib'] * NCALIB)
    rng.shuffle(role)
    S['role'] = role
    r1 = np.random.default_rng(1)
    orb = np.zeros(len(S), bool)
    for r, n in (('train', NORB_T), ('calib', NORB_C)):
        idx = np.flatnonzero(S.role.values == r)
        orb[r1.choice(idx, n, replace=False)] = True
    S['orbit_pool'] = orb
    S = S.sort_values(['video', 'i']).reset_index(drop=True)
    S.to_parquet(PLAN / 'eval_frames.parquet')
    log(f'SAE pool (test videos): {S.role.value_counts().to_dict()}, orbit pool {int(S.orbit_pool.sum())}, '
        f'{S.video.nunique()} videos, grid {len(G)}')
    # ------------------------------------------------------------------ task-3 labels on train videos
    t3 = inv['tasks23']['task3']
    want = {}  # (json split, seq key) -> (label, task1 train video)
    for b in T3:
        for sp, D in t3[b].items():
            for k, v in D.items():
                m = v.get('match')
                if m and m[0] == 'train':
                    assert m[2] == '0' and m[3] >= 48, (b, k, m)
                    want[(sp, k)] = (b, m[1])
    lab3 = {}  # (label, video) -> (T,) bool
    for sp in ('train', 'test'):
        log(f'loading task3 {sp}')
        with open(ANN / 'task3_new_behaviors' / f'calms21_task3_{sp}.json') as f:
            J = json.load(f)
        for grp, D in J.items():
            for key, d in D.items():
                k = key.split('/')[-1]
                if (sp, k) not in want:
                    continue
                b, q = want[(sp, k)]
                voc = d['metadata']['vocab']
                an = np.asarray(d['annotations'], np.int16)
                assert (b, q) not in lab3, (b, q)
                lab3[(b, q)] = an == voc[b]
        del J
    assert len(lab3) == len(want), (len(lab3), len(want))
    # ------------------------------------------------------------------ eval population: task1 TRAIN videos, 30 fps
    Z = np.load(DER / 'task1_train.npz')
    nf = inv['videos']['train']['per_video']
    st = np.r_[0, np.cumsum(Z['lens'])]
    parts = []
    for k, q in enumerate(Z['names']):
        L = int(Z['lens'][k])
        assert L == nf[q]['frames'], q
        lab = Z['labels'][st[k]:st[k + 1]]
        i = np.arange(L)
        ok = (i - 60 >= 0) & (i + 64 <= L - 1)
        D = {'video': q, 'i': i}
        for b in T1:
            y = lab == vocab[b]
            D[b] = y
            D['near_' + b] = near(y)
        for b in T3:
            if (b, q) in lab3:
                y = lab3[(b, q)]
                assert len(y) == L, (b, q, len(y), L)
                D['in_' + b], D[b], D['near_' + b] = np.ones(L, bool), y, near(y)
            else:
                D['in_' + b], D[b], D['near_' + b] = (np.zeros(L, bool),) * 3
        parts.append(pd.DataFrame(D)[ok])
    U = pd.concat(parts).reset_index(drop=True)
    stratum = np.full(len(U), '', object)
    order = [(f'{SHORT[b]}_near', U['near_' + b].values) for b in ['mount_attempt', 'intromission', 'approach', 'sniff_face']]
    order += [('at_near', U.near_attack.values), ('mo_near', U.near_mount.values), ('inv', U.investigation.values)]
    order += [(f'{SHORT[b]}_rest', U['in_' + b].values) for b in ['mount_attempt', 'intromission', 'approach', 'sniff_face']]
    order += [('rest', np.ones(len(U), bool))]
    for s, m in order:
        stratum[(stratum == '') & m] = s
    U['stratum'] = stratum
    caps = {f'{SHORT[b]}_near': c for b, c in CAP_NEAR.items()} | CAP_T1 | {f'{SHORT[b]}_rest': CAP_REST for b in T3}
    caps['rest'] = CAP_REST
    rng = np.random.default_rng(0)
    sel, info = [], {}
    for s, _ in order:
        Pp = U[U.stratum == s]
        n = min(caps[s], len(Pp))
        tk = Pp.iloc[np.sort(rng.choice(len(Pp), n, replace=False))].copy()
        tk['w'] = len(Pp) / n
        sel.append(tk)
        info[s] = {'N': int(len(Pp)), 'n': int(n), 'w': float(len(Pp) / n)}
    E = pd.concat(sel).sort_values(['video', 'i']).reset_index(drop=True)
    E = E[['video', 'i', 'stratum', 'w'] + T1 + T3 + ['in_' + b for b in T3]]
    E.to_parquet(PLAN / 'train_frames.parquet')
    labs = {}
    for b in T1 + T3:
        m = np.ones(len(E), bool) if b in T1 else E['in_' + b].values
        Um = np.ones(len(U), bool) if b in T1 else U['in_' + b].values
        y = E[b].values[m]
        pv = E[m].groupby('video')[b].any()
        labs[b] = {'scored_on': 'all eval frames' if b in T1 else f'in_{b} frames (its own sequences)',
                   'n_frames': int(m.sum()), 'n_pos': int(y.sum()),
                   'weighted_base_rate': float(np.average(y, weights=E.w.values[m])),
                   'population_base_rate': float(U[b].values[Um].mean()), 'n_videos': int(E.video[m].nunique()),
                   'n_videos_with_pos': int(pv.sum())}
    out = {'eval_population': 'task1 train videos, every frame i with 60 <= i, i + 64 <= N - 1', 'N_population': len(U),
           'n_eval_frames': len(E), 'strata': info, 'labels': labs,
           'sae_pool': {'videos': int(S.video.nunique()), 'roles': S.role.value_counts().to_dict(),
                        'orbit_pool': int(S.orbit_pool.sum()), 'grid_population': len(G)}}
    (PLAN / 'swap_plan.json').write_text(json.dumps(out, indent=1))
    log(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
