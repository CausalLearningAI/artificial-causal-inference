"""
ECI v2-temporary NES: per-frame max codes of ONE v2 dictionary on EVERY frame of the NES layouts (CPU only).

Why: the v2.9 / v2.7 scoring codes cover only the annotated videos (mice: 144 of 432 v1 videos; ants: v3 + v4 first
10 min, no v2, no v4_infested). NES needs every video of the v1 NES layouts:
  mice  dataset/mice/v1/annotations.csv   (432 videos, 2,592,000 frames; the v1 NES row order)
  ants  dataset/ants/eci30/annotations.csv (232 videos, 1,824,000 frames; the v1 30-min NES row order)
The codes are computed with the v2.9 scoring code itself (scripts/eci/v2/v29_eval.py -> v25_train.cmd_codes: tokens
from the v2_full token store, the arm's input features, sae.encode(mode='threshold'), max over the frame's M_t tokens),
only on a different row table. Edge frames (the first and last L = 2 frames of a video, which lack the t +- 2 neighbour
the input needs) get the code of the nearest computed frame (4 of 3,000-9,000 frames per video).

Dictionaries: mice PREDk16_s2901 (v2.9), ants PREDk_s2703 (v2.7; the v2.9 tree links its checkpoint).

Steps:
  rows   --dom D --key K            row table (all non-edge frames of the layout) + layout.parquet + sae link
  codes  --dom D --key K --task t --ntask n   = v25_train.cmd_codes on that row table (parts/)
  layout --dom D --key K --ntask n  parts -> <BEE>/<dom>/nes_codes/<name>/{codes_max.npy (layout order), n_fg.npy,
                                    config.json, identity.json, DONE}; identity check vs the stored scoring codes
Output root: /mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_nes/
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v29_eval as E9  # noqa: E402  (registers the PREDk16 / PREDk arms exactly as the v2.9 scoring)

TR = E9.TR
REPO = Path('/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference')
BEEN = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_nes')
ANN = {'mice': REPO / 'dataset/mice/v1/annotations.csv', 'ants': REPO / 'dataset/ants/eci30/annotations.csv'}
SAE_SRC = {'mice': Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_9/mice/sae'),
           'ants': Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_7/ants/sae')}
REF = {'mice': Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_9/mice/codes'),  # scoring codes (identity check)
       'ants': Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_7/ants/codes')}
L = TR.L


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def nes_name(key):
    return f'v2tmp_{key}'


def series_of(dom, exp):
    if dom == 'mice':
        return 'v1'
    return {'v4_infested': 'v4'}.get(exp, exp)  # the token store keeps v4_infested under series v4


def cmd_rows(a):
    dom = a.dom
    cols = ['observation_id', 'frame_idx', 'T'] + (['experiment'] if dom == 'ants' else [])
    A = pd.read_csv(ANN[dom], usecols=cols)
    n = len(A)
    starts = np.flatnonzero(np.r_[True, A.observation_id.values[1:] != A.observation_id.values[:-1]])
    ends = np.r_[starts[1:], n]
    import v23_train as V23
    man = {(s, o): t for s, o, t in V23.store_videos(dom)}
    lay, rows = [], []
    for lo, hi in zip(starts, ends):
        obs = A.observation_id.values[lo]
        ser = series_of(dom, A.experiment.values[lo] if dom == 'ants' else None)
        T = hi - lo
        fi = A.frame_idx.values[lo:hi]
        assert (fi == np.arange(T)).all(), f'{obs}: frame_idx is not 0..T-1'
        assert man.get((ser, obs)) == T, f'{ser}/{obs}: store frames {man.get((ser, obs))} vs layout {T}'
        lay.append((ser, obs, int(lo), int(T)))
        f = np.arange(L, T - L)
        rows.append(pd.DataFrame({'obs': obs, 'series': ser, 'T': T, 'fi': f, 'k': f, 'ann_row': lo + f}))
    assert len(lay) == len(set((s, o) for s, o, _, _ in lay)), 'duplicate (series, obs) in the layout'
    R = pd.concat(rows, ignore_index=True)
    R['xrow'] = np.arange(len(R))
    R['edge'] = False
    od = BEEN / dom / 'codes'
    od.mkdir(parents=True, exist_ok=True)
    R.to_parquet(od / 'rows.parquet')
    pd.DataFrame(lay, columns=['series', 'obs', 'ann_lo', 'T']).to_parquet(BEEN / dom / 'layout.parquet')
    sd = BEEN / dom / 'sae'
    sd.mkdir(parents=True, exist_ok=True)
    if not (sd / a.key).exists():
        (sd / a.key).symlink_to((SAE_SRC[dom] / a.key).resolve())
    log(f'rows {dom}: {len(lay)} videos, {n:,} layout frames, {len(R):,} non-edge rows; sae {a.key} -> {(sd / a.key).resolve()}')


def cmd_codes(a):
    TR.BEE25 = BEEN  # rows / sae / parts under the v2_nes tree; alpha from results/vision/eci_v2_9/<dom>_count.json
    a.keys = a.key
    TR.cmd_codes(a)


def n_fg_of(dom, ser, obs, T):
    import v23_train as V23
    z = np.load(V23.STORE / dom / V23.VER / 'videos' / ser / obs / 'index.npz')
    ptr, fl = z['ptr'], z['flag']
    assert len(z['k']) == T
    m = ((fl & 1) > 0).astype(np.int64)
    c = np.r_[0, np.cumsum(m)]
    return (c[ptr[1:]] - c[ptr[:-1]]).astype(np.int16)


def cmd_layout(a):
    dom, key = a.dom, a.key
    R = pd.read_parquet(BEEN / dom / 'codes' / 'rows.parquet')
    Lay = pd.read_parquet(BEEN / dom / 'layout.parquet')
    N = int(Lay.ann_lo.iloc[-1] + Lay['T'].iloc[-1])
    pd_ = BEEN / dom / 'codes' / 'parts'
    out = BEEN / dom / 'nes_codes' / nes_name(key)
    loc = out.with_name(out.name + '.tmp')  # written in place on beegfs (21 GB for mice), renamed when complete
    shutil.rmtree(loc, ignore_errors=True)
    loc.mkdir(parents=True)
    W = np.load(pd_ / f'{key}_t0.npy', mmap_mode='r').shape[1]
    Y = np.lib.format.open_memmap(loc / 'codes_max.npy', 'w+', np.float16, (N, W))
    seen = np.zeros(N, bool)
    t0 = time.time()
    for t in range(a.ntask):
        r = np.load(pd_ / f'rows_t{t}.npy')
        if not len(r):
            continue
        X = np.load(pd_ / f'{key}_t{t}.npy')
        ar = R.ann_row.values[r]
        assert not seen[ar].any()
        order = np.argsort(ar)
        Y[ar[order]] = X[order]
        seen[ar] = True
    for ser, obs, lo, T in Lay.itertuples(index=False):  # edge frames <- nearest computed frame
        for f in range(L):
            Y[lo + f] = Y[lo + L]
            Y[lo + T - 1 - f] = Y[lo + T - 1 - L]
            seen[[lo + f, lo + T - 1 - f]] = True
    assert seen.all(), f'{(~seen).sum()} layout rows missing'
    Y.flush()
    log(f'layout {dom} {key}: {N:,} x {W} filled in {time.time() - t0:.0f}s; frames with any code > 0: '
        f'{float((np.asarray(Y[::97]).max(1) > 0).mean()):.4f} (every 97th frame)')
    nfg = np.zeros(N, np.int16)
    for ser, obs, lo, T in Lay.itertuples(index=False):
        nfg[lo:lo + T] = n_fg_of(dom, ser, obs, T)
    np.save(loc / 'n_fg.npy', nfg)
    # identity check: the same frames in the stored scoring codes (computed by the same code on its own row table)
    ref = {}
    rp = REF[dom] / f'{key}_max.npy'
    if rp.exists():
        Rr = pd.read_parquet(REF[dom] / 'rows.parquet')
        Xr = np.load(rp, mmap_mode='r')
        lo_of = {(s, o): lo for s, o, lo, _ in Lay.itertuples(index=False)}
        ser_r = Rr.series.map(lambda s: s).values
        ann = np.array([lo_of.get((s, o), -1) for s, o in zip(ser_r, Rr.obs.values)]) + Rr.fi.values
        ok = np.flatnonzero((ann >= Rr.fi.values) & (Rr.fi.values >= L) & (Rr.fi.values < Rr['T'].values - L))
        rng = np.random.default_rng(0)
        pick = np.sort(rng.choice(ok, min(20000, len(ok)), replace=False))
        a_ = np.asarray(Y[ann[pick]], np.float32)
        b_ = np.asarray(Xr[pick], np.float32)
        ref = {'ref': str(rp), 'ref_rows_in_layout': int(len(ok)), 'ref_rows_total': int(len(Rr)), 'checked_rows': int(len(pick)),
               'max_abs_diff': float(np.abs(a_ - b_).max()), 'frac_rows_identical': float((a_ == b_).all(1).mean())}
        log('identity vs stored scoring codes:', json.dumps(ref))
    else:
        ref = {'ref': str(rp), 'missing': True}
        log(f'identity: no stored scoring codes at {rp}')
    cfg = {'sae': nes_name(key), 'key': key, 'domain': dom, 'layout': str(ANN[dom]), 'frames': N, 'latents': W,
           'videos': int(len(Lay)), 'sae_checkpoint': str((BEEN / dom / 'sae' / key / 'sae_theta.pt').resolve()),
           'token_store': 'eci_tokens/v2_full/<dom>/v2.1_m5',
           'pooling': {'codes_max': 'max over the M_t tokens of the frame of the threshold-mode SAE codes (v2.9 scoring '
                                    'code, scripts/eci/v2/v25_train.py cmd_codes)',
                       'n_fg': 'number of M_t (mask) tokens of the frame'},
           'edge_frames': f'first / last {L} frames of each video copy the nearest computed frame',
           'identity': ref, 'time': time.strftime('%Y-%m-%dT%H:%M:%S')}
    (loc / 'config.json').write_text(json.dumps(cfg, indent=1))
    (loc / 'identity.json').write_text(json.dumps(ref, indent=1))
    del Y
    shutil.rmtree(out, ignore_errors=True)
    loc.rename(out)
    (out / 'DONE').touch()
    for t in range(a.ntask):  # parts merged -> removed
        for f in (pd_ / f'{key}_t{t}.npy', pd_ / f'rows_t{t}.npy'):
            if f.exists():
                os.remove(f)
    log(f'layout {dom} {key} -> {out}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=('rows', 'codes', 'layout'))
    p.add_argument('--dom', required=True, choices=('mice', 'ants'))
    p.add_argument('--key', required=True)
    p.add_argument('--task', type=int, default=0)
    p.add_argument('--ntask', type=int, default=1)
    a = p.parse_args()
    {'rows': cmd_rows, 'codes': cmd_codes, 'layout': cmd_layout}[a.step](a)
