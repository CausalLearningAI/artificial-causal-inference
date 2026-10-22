"""
Behaviour alignment of the atlas models (v0 = original ECI recipe, v1 = deployed 'online', v2 = R4 'new') on the
annotated videos, for the atlas page (scripts/eci/build_atlas_v37.py reads the output). CPU only.

Frames: one shared frame set per series, so the three models are scored on exactly the same frames: the labelled
frames of the series on the 1 fps grid (frame_idx % 5 == 0 of the 5 fps grid; mice habituation: its last 15 min, the
window of v0 and v2), restricted to frames every model has codes for (v0's 2 fps grid holds every 5th 5 fps frame;
ants v2's v0 frames are 30 fps source frames s = 15 i, 5 fps frame k = round((s - 2) / 6)).
Neuron value per frame: v1: codes_max of the v1 SAE; v2 = the FROZEN v2 (scripts/eci/v2/), one run per page Outcome
input (V2_READOUT): continuous = <sae>_smooth (0.6 s smoothed frame codes, no firing threshold), binary = <sae> (the
same codes zeroed below the fnorm threshold), prob = <sae>_prob (P(on)), binary_map = <sae>_map; v0: the code of the
mean-pooled token. Neurons searched: v1 and v2 at each page prefix p in PREFIXES (the first p neurons of the Matryoshka
dictionary, the page's Neurons selector), v0 its full 4608 codes (no nested blocks). Output key per model run:
'<model>|<outcome input>|<prefix>'.
Behaviours: ants Y2F (grooming by yellow), B2F (by blue), groom_any = Y2F or B2F, YOL / BOL (yellow / blue on the
lid), onlid_any = YOL or BOL (frames where the label exists); mice nn, np, nn_or_np, nt.
Metrics per neuron and half: R~ = normalised R-precision (share of positives among the top N frames, N = positives,
ties pro rata; (P - base) / (1 - base)), primary; AUROC (Mann-Whitney, ties averaged). Cross-fitting: units (ants
videos, mice pools) split into two halves (sorted unit i -> half i % 2, as scripts/eci/rollout_r4.py cmd_align); the
TOP_K neurons by R~ on one half are scored on the other half, both directions; a listed neuron's R~cf / AUROCcf = the
mean of its test-half values over the directions that selected it. cf_top1 = the R4 definition (mean over the two
directions of the test R~ of each direction's best neuron), the cross-check against results/vision/eci_rollout_r4/
<dom>/align.json (same rule, other frame set).
Also per video: the time fraction of every behaviour over its labelled frames (all labelled 5 fps frames of the page
window), for the annotation check of the page (build_atlas_v37.py).

Output: results/vision/eci_atlas_v37/align/<series>.json
Usage: sbatch scripts/eci/atlas_align.sh   (SERIES=mice,ants_v3 for a subset)
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / 'results/vision/eci_atlas_v37/align'
BIG = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_v0')
V2SAE = {'mice': 'v2_mice_c0_w2048_s7_smooth', 'ants': 'v2_ants_c0_w2048_s6_smooth'}  # the frozen v2, primary input
V1SAE = {'mice': 'matryoshka_btk_1024_k16_fg448al_s0', 'ants': 'matryoshka_btk_1024_k16_antsfg_s0'}
# series -> (domain, experiments, v1 layout, v2 layout, v0 results dir, v0 big dir)
SERIES = {'ants_v2': ('ants', ('v2',), 'eci30', 'eci_r4v2p', 'results/vision/ants/eci_v0', 'ants_v2'),
          'ants_v3': ('ants', ('v3',), 'eci30', 'eci_r4v3p', 'results/vision/ants/eci_v0_ants_v3', 'ants_v3'),
          'ants_v4': ('ants', ('v4', 'v4_infested'), 'eci_c1v4_noday', 'eci_r4v4p',
                      'results/vision/ants/eci_v0_ants_v4', 'ants_v4'),
          'mice': ('mice', None, None, None, 'results/vision/mice/eci_v0_mice_v1', 'mice_v1')}
BEH = {'ants': {'Y2F': ['Y_Y2F'], 'B2F': ['Y_B2F'], 'groom_any': ['Y_Y2F', 'Y_B2F'], 'YOL': ['Y_YOL'],
                'BOL': ['Y_BOL'], 'onlid_any': ['Y_YOL', 'Y_BOL']},
       'mice': {'nn': ['Y_nn'], 'np': ['Y_np'], 'nn_or_np': ['Y_nn', 'Y_np'], 'nt': ['Y_nt']}}
TOP_K = 5
PREFIXES = (128, 256)  # the two smallest page prefixes of v1 and v2 (the others are not shown, page size)
# v2 outcome inputs (page Outcome selector value -> code set suffix of the frozen v2 SAE)
V2_READOUT = {'continuous': '_smooth', 'binary': '', 'prob': '_prob', 'binary_map': '_map'}


def log(*x):
    print(time.strftime('[%H:%M:%S]'), *x, flush=True)


def ann_of(ser):
    dom, exps, l1, l2, *_ = SERIES[ser]
    f = REPO / ('dataset/mice/v1/annotations.csv' if dom == 'mice' else f'dataset/ants/{l2}/annotations.csv')
    cols = sorted({c for v in BEH[dom].values() for c in v})
    a = pd.read_csv(f, usecols=['observation_id', 'frame_idx'] + cols + ([] if dom == 'mice' else ['experiment']),
                    dtype={'observation_id': str})
    a['row'] = np.arange(len(a))
    if exps:
        a = a[a.experiment.isin(exps)]
    if dom == 'mice':  # habituation: its last 15 min (frames 4500..8999), the window of v0 and v2
        a = a[~a.observation_id.str.endswith('_H') | (a.frame_idx >= 4500)]
    return a


def labels(a, dom):
    """{behaviour: (y bool, valid bool)} over the rows of a."""
    out = {}
    for b, cols in BEH[dom].items():
        V = a[cols].notna().all(1).to_numpy()
        y = (a[cols].fillna(0).to_numpy() > 0).any(1)
        out[b] = (y, V)
    return out


def rows_of(layout_ann, keys):
    """Row numbers of (obs, frame_idx) keys in a layout's annotations.csv (-1 = missing)."""
    idx = pd.Series(layout_ann['row'].to_numpy(), index=pd.MultiIndex.from_arrays(
        [layout_ann.observation_id, layout_ann.frame_idx]))
    r = idx.reindex(keys).to_numpy()
    return np.where(np.isnan(r), -1, r).astype(np.int64)


def v0_frames(ser):
    """(obs, 5 fps frame_idx) keys of v0's frames and a function code(rows) -> dense (len, 4608) float32 values."""
    dom, *_, rd, bd = SERIES[ser]
    fr = pd.read_csv(BIG / bd / 'frames.csv', dtype={'observation_id': str})
    if ser == 'ants_v2':
        k = np.round((fr['source_frame'].to_numpy() - 2) / 6).astype(int)
        X = np.load(BIG / bd / 'frame_codes_meanpool.npy', mmap_mode='r')
        get = lambda r: np.asarray(X[r], np.float32)  # noqa: E731
    else:
        k = fr['frame_idx'].to_numpy()
        ti = np.load(BIG / bd / 'frame_codes_top20_idx.npy', mmap_mode='r')
        tv = np.load(BIG / bd / 'frame_codes_top20_val.npy', mmap_mode='r')

        def get(r):
            out = np.zeros((len(r), 4608), np.float32)
            ii, vv = np.asarray(ti[r], np.int64), np.asarray(tv[r], np.float32)
            np.put_along_axis(out, ii, np.maximum(vv, 0), 1)
            return out
    return pd.MultiIndex.from_arrays([fr['observation_id'].to_numpy(), k]), get


def rprec(X, y):
    """R~ per column of X (frames x neurons) for labels y, and the base rate."""
    n, m = X.shape
    Np = int(y.sum())
    pi = Np / max(n, 1)
    if Np == 0 or Np == n:
        return np.full(m, np.nan), pi
    cut = np.partition(X, n - Np, axis=0)[n - Np]
    above, tied = X > cut, X == cut
    na, nt = above.sum(0), tied.sum(0)
    yf = y.astype(np.float32)
    pa, pt = yf @ above, yf @ tied
    P = (pa + pt * (Np - na) / np.maximum(nt, 1)) / Np
    return (P - pi) / (1 - pi), pi


def auroc(X, y):
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return np.full(X.shape[1], np.nan)
    R = rankdata(X, axis=0)
    return (R[y].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)


def half_metrics(X, y, valid, half):
    """{half: (R~ per column, base rate, AUROC per column)} of one model x behaviour."""
    out = {}
    for h in (0, 1):
        m = valid & (half == h)
        out[h] = rprec(X[m], y[m]) + (auroc(X[m], y[m]),)
    return out


def score(res, y, valid, ids, cols=None):
    """Cross-fitted top list of one model x behaviour from half_metrics res, restricted to the columns cols (the
    neurons of a Matryoshka prefix; None = all)."""
    cols = np.arange(len(ids)) if cols is None else np.asarray(cols)
    dirs, pick = [], {}
    for s, t in ((0, 1), (1, 0)):
        sc = np.where(np.isfinite(res[s][0][cols]), res[s][0][cols], -np.inf)
        top = cols[np.argsort(-sc, kind='stable')[:TOP_K]]
        dirs.append({'select_half': s, 'best': ids[int(top[0])], 'test_rprec': float(res[t][0][top[0]])})
        for j in top:
            pick.setdefault(int(j), []).append((float(res[t][0][j]), float(res[t][2][j])))
    lst = sorted(([ids[j], float(np.mean([a for a, _ in v])), float(np.mean([b for _, b in v])), len(v)]
                  for j, v in pick.items()), key=lambda x: -x[1])[:TOP_K]
    return {'base_rate': float(y[valid].mean()), 'n_frames': int(valid.sum()),
            'base_half': [float(res[0][1]), float(res[1][1])],
            'cf_top1_rprec': float(np.mean([d['test_rprec'] for d in dirs])), 'dirs': dirs,
            'top': [[int(j), round(r, 4), round(a, 4), k] for j, r, a, k in lst]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--series', default=','.join(SERIES))
    a_ = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for ser in a_.series.split(','):
        t0 = time.time()
        dom, exps, l1, l2, rd, bd = SERIES[ser]
        A = ann_of(ser)
        L = labels(A, dom)
        # per-video behaviour time fractions over all labelled frames of the window (annotation check)
        pv = {}
        for b, (y, V) in L.items():
            s = pd.DataFrame({'o': A.observation_id.to_numpy()[V], 'y': y[V]}).groupby('o')['y'].agg(['mean', 'size'])
            pv[b] = {o: [round(float(m), 6), int(n)] for o, m, n in zip(s.index, s['mean'], s['size'])}
        lab_any = np.zeros(len(A), bool)
        for y, V in L.values():
            lab_any |= V
        S = A[lab_any & (A.frame_idx.to_numpy() % 5 == 0)]
        keys = pd.MultiIndex.from_arrays([S.observation_id.to_numpy(), S.frame_idx.to_numpy()])
        # the rows of each model's codes for these frames
        k0, get0 = v0_frames(ser)
        r0 = pd.Series(np.arange(len(k0)), index=k0)
        r0 = r0[~r0.index.duplicated()].reindex(keys).to_numpy()
        if dom == 'mice':
            c1 = REPO / 'dataset/mice/v1/eci/codes' / V1SAE[dom]
            c2 = REPO / 'dataset/mice/v1/eci/codes' / V2SAE[dom]
            rr1 = rr2 = S['row'].to_numpy()
        else:
            a1 = pd.read_csv(REPO / f'dataset/ants/{l1}/annotations.csv', usecols=['observation_id', 'frame_idx'],
                             dtype={'observation_id': str})
            a1['row'] = np.arange(len(a1))
            rr1 = rows_of(a1, keys)
            rr2 = S['row'].to_numpy()
            c1 = REPO / f'dataset/ants/{l1}/codes' / V1SAE[dom]
            c2 = REPO / f'dataset/ants/{l2}/codes' / V2SAE[dom]
        ok = np.isfinite(r0) & (rr1 >= 0)
        log(f'{ser}: {len(S)} labelled 1 fps frames, {int(ok.sum())} with codes in every model '
            f'(missing v0 {int((~np.isfinite(r0)).sum())}, v1 {int((rr1 < 0).sum())})')
        S, keys = S[ok], keys[ok]
        r0, rr1, rr2 = r0[ok].astype(np.int64), rr1[ok], rr2[ok]
        pos = A.index.get_indexer(S.index)
        LS = {b: (y[pos], V[pos]) for b, (y, V) in L.items()}
        obs = S.observation_id.to_numpy()
        unit = pd.Series(obs).str.rsplit('_', n=2).str[0].to_numpy() if dom == 'mice' else obs
        units = sorted(set(unit))
        half = pd.Series(unit).map({u: i % 2 for i, u in enumerate(units)}).to_numpy()
        def load_codes(c, rr):
            return lambda: np.asarray(np.load(c / 'codes_max.npy', mmap_mode='r')[np.sort(rr)],
                                      np.float32)[np.argsort(np.argsort(rr))]
        # (model key, loader, source, prefixes): v1 / v2 at their two smallest page prefixes (the Matryoshka prefix =
        # the first p neurons), every v2 outcome input; v0 = its full 4608-code representation (no nested blocks)
        runs = [('v1|continuous', load_codes(c1, rr1), f'{c1}/codes_max.npy', PREFIXES)]
        for ro, suf in V2_READOUT.items():
            cv = c2.with_name(c2.name[:-len('_smooth')] + suf)
            runs.append((f'v2|{ro}', load_codes(cv, rr2), f'{cv}/codes_max.npy', PREFIXES))
        runs.append(('v0|continuous', lambda: get0(np.sort(r0))[np.argsort(np.argsort(r0))],
                     f'{BIG / bd} frame codes (mean-pooled token)', None))
        models = {}
        for nm, load, source, pxs in runs:
            t1 = time.time()
            X = load()
            width = X.shape[1]
            if pxs:
                X = X[:, :max(pxs)]
            alive = np.flatnonzero((X > 0).any(0))
            Xa = X[:, alive]
            ids = (alive + 1).tolist() if nm.startswith('v0') else alive.tolist()  # page neuron ids (v0: N_<j+1>)
            res = {b: (half_metrics(Xa, y, V, half) if V.sum() >= 100 and y[V].sum() >= 10 else None)
                   for b, (y, V) in LS.items()}
            for p in (pxs or [width]):
                cols = np.flatnonzero(alive < p)
                key = f'{nm}|{p}'
                models[key] = {'n_neurons': int(width), 'prefix': int(p), 'n_alive': int(len(cols)), 'source': source,
                               'beh': {b: (score(r, LS[b][0], LS[b][1], ids, cols) if r is not None and len(cols) else None)
                                       for b, r in res.items()}}
                log(f'{ser} {key}: {len(cols)} alive of the first {p}, {time.time() - t1:.0f} s: ' + ', '.join(
                    f'{b} top {v["top"][0][0]} R~cf {v["top"][0][1]:.3f} AUROC {v["top"][0][2]:.3f}' if v else f'{b} -'
                    for b, v in models[key]['beh'].items()))
            del X, Xa
        out = {'series': ser, 'domain': dom, 'n_frames': int(len(S)), 'n_units': len(units),
               'units': 'pools' if dom == 'mice' else 'videos', 'top_k': TOP_K,
               'behaviours': {b: BEH[dom][b] for b in BEH[dom]}, 'models': models, 'per_video_time': pv,
               'prefixes': list(PREFIXES), 'version': 2,
               'frames': 'labelled frames on the 1 fps grid shared by v0, v1 and v2' +
                         (' (habituation: last 15 min)' if dom == 'mice' else '')}
        (OUT / f'{ser}.json').write_text(json.dumps(out, separators=(',', ':')))
        log(f'{ser}: done in {time.time() - t0:.0f} s -> {OUT / f"{ser}.json"}')


if __name__ == '__main__':
    main()
