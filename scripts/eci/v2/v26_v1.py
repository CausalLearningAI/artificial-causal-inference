"""
ECI v2.6 addition (coordinator, after the pre-registration): is v2 significantly better than v1? Mice v1 seeds 1-10 have
codes only on the 648,000 atlas frames, so this script rescores EVERY compared model on the same frame set: the
edge-excluded ext table restricted to part == 'atlas' (647,712 frames). Per model: selection on the SG pools (rule given per
arm; eligibility on all frames of the set), held-out non-SG metrics: AUROC (v22_eval.held / metrics) and precision@top3/video
(v24_top3, raw frame code ranking). Bootstrap: the v2.5 non-SG pool weights (rng 20261104, 1,000 replicates; the SAME pool
draw for every model = paired over pools) x seeds resampled independently per model (rng 20261109).
usage: v26_v1.py --arms 'NAME:KEYPREFIX:SEEDS:RULE;...'  (v1 rule auc; readout v1 = raw code, v2 = HMM P(on) for AUROC)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v26_eval as W6  # noqa: E402  (patches: v2.6 paths, edge-excluded mice table, dist rule)

EV = W6.EV
C, E, VF, T3 = EV.C, EV.E, EV.VF, EV.T3
NB = 1000


def model_cells(key, Fa, idx, rule, W, units, uidx, sel, ev):
    t0 = time.time()
    if key.startswith('v1_') and key != 'v1_s0':
        VF.BEEF = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_2/full')  # v1 seeds 1-10 parts (v2.2 full table)
        Xf = VF.assemble(key)  # 648,000 full-table rows
        R = np.asarray(Xf[Fa.full_row.values], np.float32)
        del Xf
        X = R
    else:
        pr, pp = C.cache_paths(key)
        R = np.asarray(np.load(pr, mmap_mode='r')[idx], np.float32)
        X = R if key.startswith('v1_') else np.asarray(np.load(pp, mmap_mode='r')[idx], np.float32)
    el = (X > 0).mean(0) >= C.ELIG
    st = C.neuron_stats_multi(X, Fa, [np.flatnonzero(sel)], None)[0]
    st['fire'] = np.where(el, 1.0, 0.0)
    pk = C.select(st, rule)
    units_ev = sorted(set(Fa.unit.values[ev]))
    wb = W[:, [uidx[u] for u in units_ev]]
    wv = lambda vids, us: W[:, [uidx[u] for u in us]]  # noqa: E731
    rows = np.flatnonzero(ev)
    out = {'picks': pk}
    rng = np.random.default_rng(0)
    for b in C.BEH:
        j = pk[b]
        h = E.held(Fa, X, R, None, j, X[:, j], sel, ev, b, units_ev, rng)
        out[f'{b}|auc'] = E.metrics(h, wb)['auc']
        c = T3.top3_videos(Fa.obs.values[rows], Fa[f'y_{b}'].values[rows], Fa[f'v_{b}'].values[rows], R[rows, j], X[rows, j], Fa.unit.values[rows])
        t3 = T3.top3_rep(c, wv)
        out[f'{b}|top3'], out[f'{b}|chance'] = t3['top3'], t3['chance']
    print(f'{key} rule {rule}: picks {pk} ({time.time() - t0:.0f}s)', flush=True)
    return out


def setup():
    F, _ = C.mice_table()
    atl = F.part.values == 'atlas'
    idx = np.flatnonzero(atl)
    Fa = F.iloc[idx].reset_index(drop=True)
    units = sorted(set(F.unit.values))
    uidx = {u: i for i, u in enumerate(units)}
    non = sorted(set(F.unit.values[F.nonsg.values]))
    rng = np.random.default_rng(20261104)
    W = np.ones((NB + 1, len(units)))
    W[1:, [units.index(u) for u in non]] = np.stack([np.bincount(rng.integers(0, len(non), len(non)), minlength=len(non)) for _ in range(NB)])
    return F, Fa, idx, W, units, uidx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', required=True)
    ap.add_argument('--out', default='mice_atlas.json')
    ap.add_argument('--cell', default='')  # 'KEY:RULE' -> compute one model's cells into work/ (array task)
    a = ap.parse_args()
    import pickle
    wd = W6.T6.OUT26 / 'v2_vs_v1' / 'work'
    wd.mkdir(parents=True, exist_ok=True)
    if a.cell:
        key, rule = a.cell.split(':')
        F, Fa, idx, W, units, uidx = setup()
        c = model_cells(key, Fa, idx, rule, W, units, uidx, Fa.sg.values, Fa.nonsg.values)
        pickle.dump(c, open(wd / f'{key}_{rule}.pkl', 'wb'))
        return
    F, _ = C.mice_table()
    atl = F.part.values == 'atlas'
    idx = np.flatnonzero(atl)
    Fa = F.iloc[idx].reset_index(drop=True)
    units = sorted(set(F.unit.values))
    uidx = {u: i for i, u in enumerate(units)}
    non = sorted(set(F.unit.values[F.nonsg.values]))
    rng = np.random.default_rng(20261104)
    W = np.ones((NB + 1, len(units)))
    W[1:, [units.index(u) for u in non]] = np.stack([np.bincount(rng.integers(0, len(non), len(non)), minlength=len(non)) for _ in range(NB)])
    sel, ev = Fa.sg.values, Fa.nonsg.values
    per, picks = {}, {}
    for spec in a.arms.split(';'):
        name, pre, seeds, rule = spec.split(':')
        cells = [pickle.load(open(wd / f'{pre}_s{s}_{rule}.pkl', 'rb')) for s in seeds.split(',')]
        per[name] = {m: np.stack([c[m] for c in cells]) for m in cells[0] if m != 'picks'}
        picks[name] = [c['picks'] for c in cells]
    r2 = np.random.default_rng(20261109)
    res = {'frame_set': f'edge-excluded ext frames, part == atlas: {len(Fa):,} frames, held-out non-SG {int(ev.sum()):,}',
           'levels': {}, 'contrasts': {}, 'picks': picks}
    for n, d in per.items():
        res['levels'][n] = {}
        for m, v in d.items():
            ix = r2.integers(0, len(v), (NB, len(v)))
            bb = np.nanmean(v[:, 1:][ix, np.arange(NB)[:, None]], 1)
            res['levels'][n][m] = {'mean': float(np.nanmean(v[:, 0])), 'ci': np.nanpercentile(bb, [2.5, 97.5]).tolist()}
    names = list(per)
    for x in names:
        for y in names:
            if x == y or not y.startswith('v1'):
                continue
            r = {}
            for m in per[x]:
                if m.endswith('chance'):
                    continue
                ax, ay = per[x][m][:, 1:], per[y][m][:, 1:]
                q = np.arange(NB)[:, None]
                ix, iy = r2.integers(0, len(ax), (NB, len(ax))), r2.integers(0, len(ay), (NB, len(ay)))
                dd = np.nanmean(ax[ix, q], 1) - np.nanmean(ay[iy, q], 1)
                r[m] = {'diff': float(np.nanmean(per[x][m][:, 0]) - np.nanmean(per[y][m][:, 0])), 'ci': np.nanpercentile(dd, [2.5, 97.5]).tolist()}
            res['contrasts'][f'{x} - {y}'] = r
    od = W6.T6.OUT26 / 'v2_vs_v1'
    od.mkdir(parents=True, exist_ok=True)
    (od / a.out).write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res['contrasts'], indent=1))


if __name__ == '__main__':
    main()
