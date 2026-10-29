"""
ECI v2.0 Part B variant DYN (pre-registered in scripts/eci/v2/v20_gate.py): SAE input [z_t, z_t - z_(t - 1 s)] at the same
grid position (difference 0 where that position is not kept 1 s earlier), the v2.0 recipe otherwise (Matryoshka
BatchTopK 2048, 128/256/512/2048, k16, 30k x 4096, per-block TokenNorm), seeds 970-972. No re-encoding: stores with all
N2 tokens of frames 1 s apart.
    ants: train on the V2-full 'full1' store (1 fps, every video, N2 arm, keep p = 1; lag = 1 frame = 5 grid steps);
          eval codes on its labelled frames (eval.py frame table f1_idx); dynamic share on the suite's 5 fps probe store
          (lag 5 frames).
    mice: train on the behaviour-suite 5 fps probe store (72 videos x 1500 frames, N2; lag 5); eval codes on the v2_prob
          probe store (the eval frames, N2, lag 5); dynamic share on the suite probe store.
Train tokens capped at 16M per domain (uniform, rng 0; host memory), calib / eval 1.5M.
Steps: run --dom D (GPU: train + all codes) ; score --dom D (CPU: gate + dynamic share)
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
for p in ('', 'scripts/eci/v2', 'scripts/eci/v2_eval', 'scripts/eci/v2_prob', 'scripts/eci/v2_behav', 'scripts/eci/v2_diag'):
    sys.path.insert(0, str(REPO / p))
EC = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes')
BEE = EC / 'v2_0B/dynsae'
OUT = REPO / 'results/vision/eci_v2_0B/dynsae'
SEEDS = (970, 971, 972)
CAP = {'train': 16_000_000, 'calib': 1_500_000, 'eval': 1_500_000}
log = lambda s: print(time.strftime('%H:%M:%S'), s, flush=True)  # noqa: E731


class Tok:
    """frames F (obs, k), token frame tf, pos, keep mask, get(idx) -> (n, 1024) fp16 tokens; prev index at lag L."""

    def __init__(self, F, tf, pos, get, keep, lag):
        self.F, self.get = F, get
        self.ix = np.flatnonzero(keep)
        o = pd.factorize(F.obs.values)[0].astype(np.int64)
        k = F.k.values.astype(np.int64)
        key = lambda kk: (o[tf[self.ix]] << 40) | (kk << 16) | pos[self.ix]  # noqa: E731
        cur = pd.Series(np.arange(len(self.ix)), index=key(k[tf[self.ix]]))
        want = key(k[tf[self.ix]] - lag)
        self.prev = cur.reindex(want).fillna(-1).values.astype(np.int64)  # index into self.ix, -1 = none
        self.tf = tf[self.ix]
        self.pos = pos[self.ix]
        log(f'Tok: {len(self.ix):,} tokens, previous-second token present {(self.prev >= 0).mean():.3f}')

    def x(self, sel):
        """inputs [z, z - z_prev] (n, 2048) fp16 of kept-token numbers sel (sorted)."""
        z = self.get(self.ix[sel]).astype(np.float32)
        pv = self.prev[sel]
        d = np.zeros_like(z)
        m = pv >= 0
        if m.any():
            u, inv = np.unique(pv[m], return_inverse=True)
            zp = self.get(self.ix[u]).astype(np.float32)[inv]
            d[m] = z[m] - zp
        return np.concatenate([z, d], 1).astype(np.float16)


def full1_ants():
    import diag as D
    T2 = D.T2
    old = T2.BEE
    T2.BEE = D.BEE
    T2.KINDS['ants'].setdefault('full1', {'series': D.F1_SERIES['ants'][0]})
    try:
        U = T2.Union('ants', 'full1')
    finally:
        T2.BEE = old
    keep = U.arm('N2')
    offs = np.cumsum([0] + [p['n'] for p in U.parts])
    mms = [np.memmap(p['root'] / 'tok.f16', np.float16, 'r', shape=(p['n'], 1024)) for p in U.parts]

    def get(idx):
        out = np.empty((len(idx), 1024), np.float16)
        o = np.argsort(idx)
        si = idx[o]
        for i, mm in enumerate(mms):
            a, b = np.searchsorted(si, [offs[i], offs[i + 1]])
            if b > a:
                out[o[a:b]] = mm[si[a:b] - offs[i]]
        return out
    return U.F, U.tf, U.pos, get, keep


def bh_store(root):
    import behav as BH
    BH.BEE = root
    St = BH.Store(DOM, 'v2', 'dense', 1024)
    bnd = np.array([b for b in St.bounds])

    def get(idx):
        out = np.empty((len(idx), 1024), np.float16)
        o = np.argsort(idx)
        si = idx[o]
        for i, (a0, b0) in enumerate(bnd):
            a, b = np.searchsorted(si, [a0, b0])
            if b > a:
                out[o[a:b]] = St.mm[i][si[a:b] - a0]
        return out
    F = St.F.copy()
    if 'obs' not in F:
        F['obs'] = F.vid.astype(str)
    return F, St.tf, St.pos, get, St.keep('N2'), St


def roles_of(dom, obs):
    import v20 as V
    c = V.dcfg(dom)
    r = {}
    for sc in c['series']:
        for v in json.loads((REPO / sc['plan']).read_text())['videos']:
            r[v['obs']] = v['role']
    return np.array([r.get(o, 'train') for o in obs])


def frame_codes(sae, norm, T, frames_of, nF, dev):
    """frame-max codes (nF, 2048) fp16 of the kept tokens whose frame maps (frames_of[tf]) >= 0."""
    import torch
    fo = frames_of[T.tf]
    sel = np.flatnonzero(fo >= 0)
    X = torch.zeros((nF, sae.n_latents), device=dev)
    for c0 in range(0, len(sel), 300_000):
        s = sel[c0:c0 + 300_000]
        x = torch.from_numpy(T.x(s)).to(dev)
        with torch.no_grad():
            c = sae.encode(norm(x.float()), mode='threshold').float()
        X.index_reduce_(0, torch.from_numpy(fo[s]).to(dev), c, 'amax', include_self=True)
    return X.half().cpu().numpy()


DOM = None


def cmd_run(a):
    global DOM
    import torch
    from src.eci.v2 import sae as SA
    import v20 as V
    import eval as EV
    DOM = a.dom
    dev = torch.device('cuda')
    s = V.R.spec()
    t0 = time.time()
    if DOM == 'ants':
        F, tf, pos, get, keep = full1_ants()
        T = Tok(F, tf, pos, get, keep, lag=5)
        roles = roles_of('ants', F.obs.values)
    else:
        F, tf, pos, get, keep, _ = bh_store(EC / 'v2_behav')
        T = Tok(F, tf, pos, get, keep, lag=5)
        roles = roles_of('mice', F.obs.values)
    tr = roles[T.tf]
    rng = np.random.default_rng(0)
    X = {}
    info = {'tokens_total': int(len(T.ix)), 'prev_present': float((T.prev >= 0).mean())}
    for r in ('train', 'calib', 'eval'):
        ii = np.flatnonzero(tr == r)
        ii = np.sort(rng.choice(ii, min(len(ii), CAP[r]), replace=False))
        info[f'tokens_{r}'] = int(len(ii))
        x = np.concatenate([T.x(ii[c:c + 1_000_000]) for c in range(0, len(ii), 1_000_000)])
        X[r] = V.HostTok(torch.from_numpy(x), dev) if r == 'train' else torch.from_numpy(x)
        log(f'{DOM} {r}: {len(ii):,} tokens ({time.time() - t0:.0f}s)')
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = {**s['sae'], 'context_window': 1}
    recs = {}
    for seed in SEEDS:
        od = BEE / DOM / 'sae' / f'dyn_{DOM}_s{seed}'
        recs[seed] = SA.train(cfg, seed, X['train'], X['calib'], X['eval'], od, dev)
        log(f'{od.name}: eval FVE {recs[seed]["eval_fve"]:.5f} L0 {recs[seed]["eval_l0"]:.2f} ({recs[seed]["train_time_s"]:.0f}s)')
        (OUT / f'{DOM}_train.json').write_text(json.dumps({'data': info, 'seeds': recs}, indent=1, default=float))
    del X
    # eval-frame codes
    od = BEE / DOM / 'codes'
    od.mkdir(parents=True, exist_ok=True)
    if DOM == 'ants':
        Fe = pd.read_parquet(EV.BEE / 'ants' / 'frames.parquet')
        fmap = np.full(len(F), -1, np.int64)
        fmap[Fe.f1_idx.values] = np.arange(len(Fe))
        Te, nE = T, len(Fe)
    else:
        Pr = pd.read_parquet(REPO / 'results/vision/eci_v2_prob/mice/probe.parquet')  # (study.py OUT; not imported: it re-points behav)
        Fp, tfp, posp, getp, keepp, Stp = bh_store(EC / 'v2_prob')
        Te = Tok(Fp, tfp, posp, getp, keepp, lag=5)
        key = lambda F_: pd.MultiIndex.from_arrays([F_.vid.values.astype(int), F_.k.values.astype(int)])  # noqa: E731
        fmap = pd.Series(np.arange(len(Pr)), index=key(Pr)).reindex(key(Fp)).values.astype(np.int64)
        nE = len(Pr)
    # probe (dynamic share) codes: suite 5 fps probe store
    if DOM == 'ants':
        Fb, tfb, posb, getb, keepb, _ = bh_store(EC / 'v2_behav')
        Tb = Tok(Fb, tfb, posb, getb, keepb, lag=5)
    else:
        Fb, Tb = F, T
    import behav as BH
    Prb = pd.read_parquet(REPO / f'results/vision/eci_v2_behav/{DOM}/probe.parquet')
    key = lambda F_: pd.MultiIndex.from_arrays([F_.vid.values.astype(int), F_.k.values.astype(int)])  # noqa: E731
    bmap = pd.Series(np.arange(len(Prb)), index=key(Prb)).reindex(key(Fb)).values.astype(np.int64)
    for seed in SEEDS:
        sae, norm = SA.load(BEE / DOM / 'sae' / f'dyn_{DOM}_s{seed}' / 'sae_theta.pt', dev)
        np.save(od / f'dyn_s{seed}_eval.npy', frame_codes(sae, norm, Te, fmap, nE, dev))
        np.save(od / f'dyn_s{seed}_probe.npy', frame_codes(sae, norm, Tb, bmap, len(Prb), dev))
        log(f'{DOM} s{seed}: codes saved')


def cmd_score(a):
    import analysis as AN
    import eval as EV
    import v20_gate as G
    dom = a.dom
    F = EV.frame_table(dom)
    SP = AN.splits(F.unit.values, EV.SPLITS)
    starts = AN.seq_starts(F)
    res = {}
    for seed in SEEDS:
        X = np.load(BEE / dom / 'codes' / f'dyn_s{seed}_eval.npy').astype(np.float32)
        if dom == 'mice':
            X = AN.smooth_seq(X[F.code_row.values], starts)
        res[seed] = G.score(dom, X, np.ones(len(F), bool), F, SP)
        log(f'{dom} dyn s{seed} scored')
    S = json.loads((G.OUT / f'{dom}_scores.json').read_text())
    gate = {}
    for seed in SEEDS:
        gate[seed] = G.gate_rows(dom, S['v1'], res[seed], S['eligibility'])
    # dynamic share @256 (v20_vs_v1 'dyn' rule)
    from scipy.stats import rankdata
    z = np.load(REPO / f'results/vision/eci_v2_0B/v1_vs_v2/me_{dom}.npz')
    ME, AR = z['me'], z['area']
    ok = np.isfinite(ME)
    zs = lambda r: (r - r.mean()) / r.std()  # noqa: E731
    zM, zA = zs(rankdata(ME[ok])), zs(rankdata(AR[ok]))
    dyn = {}
    for seed in SEEDS:
        C = np.load(BEE / dom / 'codes' / f'dyn_s{seed}_probe.npy').astype(np.float32)
        n = e = 0
        for j in range(256):
            c = C[:, j]
            if (c > 0).mean() < 0.01:
                continue
            rc = zs(rankdata(c[ok]))
            rm, ra = float((rc * zM).mean()), float((rc * zA).mean())
            e += 1
            n += int(rm >= 0.10 and rm > abs(ra))
        dyn[seed] = {'dynamic_256': n, 'eligible_256': e}
    out = {'gate': gate, 'dyn256': dyn,
           'n_pass_eligible': {s: sum(r['PASS'] for r in g.values() if r['eligible']) for s, g in gate.items()}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'{dom}_score.json').write_text(json.dumps(out, indent=1, default=float))
    log(json.dumps({'n_pass': out['n_pass_eligible'], 'dyn256': dyn}))
    for s, g in gate.items():
        log(f's{s}: ' + ', '.join(f'{c} {"P" if r["PASS"] else "F"} AUROC {r["v2_AUROC"]:.3f} top1 {r["v2_top1_prec"]:.3f}'
                                   for c, r in g.items() if r['eligible']))


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=('run', 'score'))
    p.add_argument('--dom', choices=('mice', 'ants'))
    a = p.parse_args()
    {'run': cmd_run, 'score': cmd_score}[a.step](a)


if __name__ == '__main__':
    main()
