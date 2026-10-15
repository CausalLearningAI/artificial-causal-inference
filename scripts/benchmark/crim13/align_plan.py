"""
CRIM13 alignment test - frame plans (CPU job, minutes). Rules: eci_align.sh.

Grid: 0-based video frame v on a 5 fps grid (v % 5 == 0) with room for the 2 s lag (v - 50 >= 0), v < the video's frames.
Per HALF (the released train / test videos; each half is the SAE half of one fold and the scoring half of the other):
    pool   label-free: 10,000 'train' + 2,000 'calib' (theta*) grid frames drawn uniformly over the half's grid frames
           (rng 0; labels never read for these draws), disjoint from the half's eval frames.
    eval   15,000 grid frames stratified by the frame's own S1 label (labels are dense and mutually exclusive):
           sniff 1,500; each other scored behaviour 1,000 (capped at the stratum size); rest (other + human) takes the
           remainder; weight w = N_stratum / n_stratum (inverse inclusion probability).
Outputs: OUT/plan/{frames.parquet (half, video, v, role, label columns, w), videos.parquet, plan.json}
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from align_common import INV, LAG, PLAN, ROOT, SCORED, STEP, VOCAB, log, meta_of, read_labels  # noqa: E402

NPOOL, NCAL, NEVAL = 10000, 2000, 15000
WANT = {b: (1500 if b == 'sniff' else 1000) for b in SCORED}


def main():
    PLAN.mkdir(parents=True, exist_ok=True)
    inv = json.loads(INV.read_text())
    V, G = [], []
    for r in inv['videos']:
        q, half, n = r['name'], r['split'], r['frames']
        res, intr, strain = meta_of(q)
        V.append({'video': q, 'half': half, 'frames': n, 'W': r['header']['width'], 'H': r['header']['height'],
                  'resident': res, 'intruder': intr, 'strain': strain})
        lab = read_labels(ROOT / r['dir'] / f'{q}_t.txt', n)
        assert (lab >= 0).all(), q
        v = np.arange(LAG, n, STEP)
        v = v[v % STEP == 0]
        G.append(pd.DataFrame({'half': half, 'video': q, 'v': v, 'lab': lab[v]}))
    V = pd.DataFrame(V)
    G = pd.concat(G, ignore_index=True)
    rng = np.random.default_rng(0)
    parts, info = [], {'grid': f'v % {STEP} == 0, v >= {LAG}', 'halves': {}}
    for half in ('train', 'test'):
        U = G[G.half == half].reset_index(drop=True)
        name = U.lab.map(lambda k: VOCAB[k])
        stratum = np.where(name.isin(SCORED), name, 'rest')
        U['stratum'] = stratum
        ev, hi = [], {'N_grid': len(U), 'strata': {}}
        left = NEVAL
        for s in SCORED:
            P_ = U[U.stratum == s]
            n = min(WANT[s], len(P_))
            left -= n
            tk = P_.iloc[np.sort(rng.choice(len(P_), n, replace=False))].copy()
            tk['w'] = len(P_) / n
            ev.append(tk)
            hi['strata'][s] = {'N': int(len(P_)), 'n': int(n), 'w': float(len(P_) / n)}
        P_ = U[U.stratum == 'rest']
        tk = P_.iloc[np.sort(rng.choice(len(P_), left, replace=False))].copy()
        tk['w'] = len(P_) / left
        ev.append(tk)
        hi['strata']['rest'] = {'N': int(len(P_)), 'n': int(left), 'w': float(len(P_) / left)}
        E = pd.concat(ev)
        E['role'] = 'eval'
        rest = U.drop(index=E.index)
        take = rng.choice(len(rest), NPOOL + NCAL, replace=False)
        Pl = rest.iloc[np.sort(take)].copy()
        role = np.array(['train'] * NPOOL + ['calib'] * NCAL)
        rng.shuffle(role)
        Pl['role'] = role
        Pl['w'] = np.nan
        Pl['stratum'] = ''   # label-free pool: the label column is dropped below
        D = pd.concat([E, Pl])
        for b in VOCAB:
            D[b] = (D.lab.values == VOCAB.index(b)) & (D.role.values == 'eval')
        parts.append(D.drop(columns='lab'))
        Ee = D[D.role == 'eval']
        hi.update(n_eval=int(len(Ee)), n_pool=int((D.role == 'train').sum()), n_calib=int((D.role == 'calib').sum()),
                  n_videos=int(U.video.nunique()), n_eval_videos=int(Ee.video.nunique()),
                  weighted_base_rate={b: float(np.average(Ee[b], weights=Ee.w)) for b in VOCAB},
                  grid_base_rate={b: float((U.lab == VOCAB.index(b)).mean()) for b in VOCAB},
                  n_pos={b: int(Ee[b].sum()) for b in VOCAB},
                  n_videos_with_pos={b: int(Ee.groupby('video')[b].any().sum()) for b in VOCAB})
        info['halves'][half] = hi
        log(f'{half}: ' + json.dumps({k: v for k, v in hi.items() if k in ('N_grid', 'n_eval', 'n_pool', 'n_calib')}))
    F = pd.concat(parts).sort_values(['half', 'video', 'v']).reset_index(drop=True)
    F = F.merge(V[['video', 'resident', 'strain']], on='video', how='left')
    F.to_parquet(PLAN / 'frames.parquet')
    V.to_parquet(PLAN / 'videos.parquet')
    info['scored'] = SCORED
    (PLAN / 'plan.json').write_text(json.dumps(info, indent=1))
    log(json.dumps(info['halves']['train']['strata']))
    log(json.dumps(info['halves']['test']['strata']))


if __name__ == '__main__':
    main()
