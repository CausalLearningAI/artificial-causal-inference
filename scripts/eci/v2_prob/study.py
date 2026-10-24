"""
ECI v2 per-neuron PROBABILITY READOUT and METRIC AUDIT: data steps (probe set L, token encodes, frame codes).

PROBE SET L (label-free choice of frames; fixed 2026-10-25 before any token, code or score of this study existed).
    Contiguous 5 fps windows of the LABELLED videos, so that time models (HMM, bouts) can be fitted and scored:
      mice  all 144 annotated mice v1 videos, one window of 600 consecutive frames (2 min) per video, centred in the
            atlas window (habituation '_H' videos: frames 4500..8999 -> k0 = 6450; odour videos (4500 frames) -> k0 =
            1950). 86,400 frames.
      ants  v2: all 44 videos, 600 frames centred in the labelled 0..2999 -> k0 = 1200 (26,400 frames);
            v3: all 107 labelled videos, 300 frames centred -> k0 = 1350 (32,100 frames). v4 not used (cost).
    The windows are fixed by position only (no label is read to place them).
ENCODES (label-free): the frozen v2 encoder and the M0 | N2 union of scripts/eci/v2_behav/behav.py cmd_enc2, v1's exact
    pipeline (behav.py cmd_enc1), reused unchanged; the invariance copies are switched off (INV_EVERY = 10^9: only
    each window's first frame gets the four transformed copies, never used here). Stores ->
    /mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_prob/<dom>/{v1,v2}/dense/s<i>/.
DICTIONARIES (no SAE is trained):
      v1      deployed matryoshka_btk_1024_k16_{fg448al | antsfg}_s0 and v1's recipe retrained, seeds 1-10
              (eci_codes/v2_diag/v1retrain/<dom>/V1R_full_s<s>/sae.pt);
      V2F_M0  seeds 900-919, V2F_N2 seeds 900-919 (eci_codes/v2_diag/<dom>/sae/);
      v2dep   the frozen deployed v2 (mice s7, ants s6) = the bridge to the full-video codes (identity check).
    Frame code = max over the kept tokens of the threshold-mode code (behav.py cmd_codes, unsmoothed), fp16, rows in
    probe order -> BEE/<dom>/codes/<dict>.npy. Identity checks (reported): v1 s0 vs its stored codes_max rows; v2dep vs
    the frozen per-video codes_max.

Steps (scripts/eci/v2_prob/job.sh, EXTRA = arguments):
    plan                          (CPU)  -> results/vision/eci_v2_prob/<dom>/{videos,probe}.parquet
    enc2 --dom D --shard i --nshard n   (GPU)
    enc1 --dom D --shard i --nshard n   (GPU)
    codes --dom D                 (GPU)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / 'results/vision/eci_v2_prob'
BEE = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_prob')
EC = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes')
os.environ['BEHAV_OUT'] = str(OUT)
os.environ['BEHAV_BEE'] = str(BEE)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts/eci/v2_behav'))
import behav as BH  # noqa: E402

BH.INV_EVERY = 10 ** 9
assert BH.OUT == OUT and BH.BEE == BEE
V1SAE = {'mice': 'dataset/mice/v1/eci/sae/matryoshka_btk_1024_k16_fg448al_s0/sae.pt',
         'ants': 'dataset/ants/eci/sae/matryoshka_btk_1024_k16_antsfg_s0/sae.pt'}
DEP = {'mice': 7, 'ants': 6}
V2F_SEEDS = range(900, 920)


def log(s):
    print(time.strftime('%H:%M:%S'), s, flush=True)


def dicts_of(dom):
    """[(name, sae path, W, mask, family)]"""
    out = [('v1_s0', REPO / V1SAE[dom], 1024, 'v1', 'v1')]
    out += [(f'v1_s{s}', EC / f'v2_diag/v1retrain/{dom}/V1R_full_s{s}/sae.pt', 1024, 'v1', 'v1') for s in range(1, 11)]
    for a, m in (('V2F_M0', 'M0'), ('V2F_N2', 'N2')):
        out += [(f'{a}_s{s}', EC / f'v2_diag/{dom}/sae/{a}_s{s}/sae_theta.pt', 2048, m, 'v2') for s in V2F_SEEDS]
    out.append((f'v2dep_s{DEP[dom]}', EC / f'v2/{dom}/sae/v2_{dom}_c0_w2048_s{DEP[dom]}/sae_theta.pt', 2048, 'M0', 'v2'))
    return out


def cmd_plan(args):
    rows = []
    a = pd.read_csv(REPO / 'dataset/mice/v1/annotations.csv', usecols=['observation_id', 'frame_idx', 'Y_nn'])
    lab = sorted(a.loc[a.Y_nn.notna(), 'observation_id'].unique())
    n = a.groupby('observation_id').size()
    for o in lab:
        k0 = 6450 if o.endswith('_H') else 1950
        assert n[o] in (4500, 9000) and (n[o] == 9000) == o.endswith('_H')
        rows.append(dict(dom='mice', series='v1', obs=o, n=int(n[o]), k0=k0, nwin=600))
    a2 = pd.read_csv(REPO / 'dataset/ants/eci30/annotations.csv', dtype={'observation_id': str},
                     usecols=['observation_id', 'frame_idx', 'experiment', 'Y_Y2F'])
    for ser, nw, k0 in (('v2', 600, 1200), ('v3', 300, 1350)):
        g = a2[(a2.experiment == ser) & a2.Y_Y2F.notna()]
        for o in sorted(g.observation_id.unique()):
            assert g[g.observation_id == o].frame_idx.max() == 2999
            rows.append(dict(dom='ants', series=ser, obs=o, n=int((a2.observation_id == o).sum()), k0=k0, nwin=nw))
    R = pd.DataFrame(rows)
    for dom, Vd in R.groupby('dom'):
        Vd = Vd.drop(columns='dom').reset_index(drop=True)
        Vd['vid'] = np.arange(len(Vd))
        Fr = pd.DataFrame([dict(vid=r.vid, series=r.series, obs=r.obs, k=int(r.k0 + i), inv=(i == 0))
                           for r in Vd.itertuples() for i in range(r.nwin)])
        (OUT / dom).mkdir(parents=True, exist_ok=True)
        Vd.to_parquet(OUT / dom / 'videos.parquet')
        Fr.to_parquet(OUT / dom / 'probe.parquet')
        log(f'plan {dom}: {len(Vd)} videos {Vd.series.value_counts().to_dict()}, {len(Fr):,} frames')


def cmd_codes(args):
    import torch
    from src.eci.sae import load_sae
    dev = torch.device('cuda')
    dom = args.dom
    Pr = pd.read_parquet(OUT / dom / 'probe.parquet')
    key = lambda F_: pd.MultiIndex.from_arrays([F_.vid.values.astype(int), F_.k.values.astype(int)])  # noqa: E731
    pidx = pd.Series(np.arange(len(Pr)), index=key(Pr))
    od = BEE / dom / 'codes'
    od.mkdir(parents=True, exist_ok=True)
    D = dicts_of(dom)
    assert all(p.exists() for _, p, *_ in D), [n for n, p, *_ in D if not p.exists()]
    meta = {}
    for fam in ('v1', 'v2'):
        St = BH.Store(dom, fam, 'dense', 1024 if fam == 'v2' else 768)
        fidx = pidx.reindex(key(St.F)).values
        assert not np.isnan(fidx).any() and len(np.unique(fidx)) == len(Pr), (fam, 'store != probe set')
        fidx = fidx.astype(np.int64)
        dd = [d for d in D if d[4] == fam]
        keeps = {m: St.keep(m) for m in {d[3] for d in dd}}
        meta[f'tokens_per_frame_{fam}'] = {m: float(k.sum() / St.nf) for m, k in keeps.items()}
        for g0 in range(0, len(dd), 12):
            grp = dd[g0:g0 + 12]
            saes = {}
            for name, p, W, m, _ in grp:
                sae, norm, _ = load_sae(p, dev)
                saes[name] = (sae, norm, W, m)
            X = {n_: torch.zeros((len(Pr), v[2]), dtype=torch.float16, device=dev) for n_, v in saes.items()}
            fp = 400 if fam == 'v2' else 800
            t0 = time.time()
            for f0 in range(0, St.nf, fp):
                f1 = min(St.nf, f0 + fp)
                a, b = int(St.starts[f0]), int(St.starts[f1])
                if a == b:
                    continue
                z = torch.from_numpy(St.tok(a, b)).to(dev)
                fl_all = torch.from_numpy(St.tf[a:b] - f0).to(dev)
                rows_t = torch.from_numpy(fidx[f0:f1]).to(dev)
                for n_, (sae, norm, W, m) in saes.items():
                    sel = torch.from_numpy(keeps[m][a:b]).to(dev)
                    with torch.no_grad():
                        c = sae.encode(norm(z[sel].float()), mode='threshold').float()
                        fm = torch.zeros((f1 - f0, W), device=dev).index_reduce_(0, fl_all[sel], c, 'amax',
                                                                                 include_self=True)
                        X[n_][rows_t] = fm.half()
            for n_, x in X.items():
                np.save(od / f'{n_}.npy', x.cpu().numpy())
            log(f'codes {dom} {fam}: saved {list(X)} ({time.time() - t0:.0f}s); {BH.gpu_line()}')
            del X, saes
            torch.cuda.empty_cache()
    # identity checks
    Fv = BH.Store(dom, 'v1', 'dense', 768).F
    row = pd.Series(Fv.row.values, index=key(Fv)).reindex(key(Pr)).values.astype(np.int64)
    X = np.load(od / 'v1_s0.npy').astype(np.float32)
    ref = np.load(REPO / BH.V1[dom]['codes'], mmap_mode='r')
    o = np.argsort(row)
    R = np.empty_like(X)
    R[o] = np.asarray(ref[row[o]]).astype(np.float32)
    meta['identity_v1_s0'] = {'fire_agree': float(((X > 0) == (R > 0)).mean()),
                              'max_abs_diff': float(np.abs(X - R).max()), 'mean_abs_diff': float(np.abs(X - R).mean())}
    X = np.load(od / f'v2dep_s{DEP[dom]}.npy').astype(np.float32)
    ag, md = [], 0.0
    for (ser, obs), g in Pr.groupby(['series', 'obs'], sort=False):
        ref = np.load(EC / f'v2/{dom}/videos/{ser}/{obs}.npz')['codes_max'][g.k.values].astype(np.float32)
        x = X[g.index.values]
        ag.append(((x > 0) == (ref > 0)).mean())
        md = max(md, float(np.abs(x - ref).max()))
    meta['identity_v2dep'] = {'fire_agree_mean': float(np.mean(ag)), 'fire_agree_min': float(np.min(ag)),
                              'max_abs_diff': md}
    (OUT / dom / 'codes_meta.json').write_text(json.dumps(meta, indent=1))
    log(json.dumps(meta))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('step')
    p.add_argument('--dom', default='')
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--nshard', type=int, default=1)
    p.add_argument('--selftest', action='store_true')
    a = p.parse_args()
    if a.step == 'enc2':
        BH.cmd_enc2(a)
    elif a.step == 'enc1':
        BH.cmd_enc1(a)
    else:
        {'plan': cmd_plan, 'codes': cmd_codes}[a.step](a)


if __name__ == '__main__':
    main()
