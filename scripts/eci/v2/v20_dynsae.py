"""
ECI v2.0 Part B variant DYN3 (pre-registered in scripts/eci/v2/v20_gate.py AMENDMENT 1 (a)): SAE input
[z_t, z_t - z_(t-1), std_(t-2..t+2)(z)] at the same grid position, 5 fps consecutive frames (difference 0 when the
position is not kept at t-1; std over the window frames where the position is kept, 0 if only t), per-block TokenNorm
(3 x 1024), the v2.0 recipe otherwise (Matryoshka BatchTopK 2048, 128/256/512/2048, k16, 30k x 4096), seeds 970-972.
Trained on the suite's contiguous all-token 5 fps probe stores (eci_codes/v2_behav/<dom>/v2/dense, N2 tokens), train
tokens capped at 12M (uniform, rng 0), calib / eval 1.5M. No re-encoding, no 30 fps.
Codes: eval frames = the v2_prob Tier B probe stores (mice = the eval.py mice frames; ants = the 5 fps ants probe table
of v20_prox.py), probe frames of the suite (dynamic share).
Steps: run --dom D (GPU) ; score --dom D (CPU)
"""
import json
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
CAP = {'train': 12_000_000, 'calib': 1_500_000, 'eval': 1_500_000}
LAGS = (-2, -1, 1, 2)
log = lambda s: print(time.strftime('%H:%M:%S'), s, flush=True)  # noqa: E731


class Tok:
    """kept tokens of a behav-style store with their same-position neighbours at frames t-2, t-1, t+1, t+2."""

    def __init__(self, St, keep):
        self.St = St
        F = St.F
        self.ix = np.flatnonzero(keep)
        o = pd.factorize(F.vid.values)[0].astype(np.int64)
        k = F.k.values.astype(np.int64)
        tf, pos = St.tf[self.ix], St.pos[self.ix]
        key = lambda d: o[tf] * (1 << 40) + (k[tf] + d + 8) * (1 << 16) + pos  # noqa: E731 (no OR of negatives)
        cur = pd.Series(np.arange(len(self.ix)), index=key(0))
        self.nb = {d: cur.reindex(key(d)).fillna(-1).values.astype(np.int64) for d in LAGS}
        self.tf = tf
        bnd = St.bounds
        self.bstart = np.array([b[0] for b in bnd] + [bnd[-1][1]])
        log(f'Tok: {len(self.ix):,} tokens; neighbour present ' + ', '.join(f'{d}: {(v >= 0).mean():.3f}' for d, v in self.nb.items()))

    def get(self, idx):
        out = np.empty((len(idx), 1024), np.float16)
        o = np.argsort(idx)
        si = idx[o]
        for i in range(len(self.bstart) - 1):
            a, b = np.searchsorted(si, [self.bstart[i], self.bstart[i + 1]])
            if b > a:
                out[o[a:b]] = self.St.mm[i][si[a:b] - self.bstart[i]]
        return out

    def x(self, sel):
        z = self.get(self.ix[sel]).astype(np.float32)
        nbz, nbm = {}, {}
        for d in LAGS:
            pv = self.nb[d][sel]
            m = pv >= 0
            zz = np.zeros_like(z)
            if m.any():
                u, inv = np.unique(pv[m], return_inverse=True)
                zz[m] = self.get(self.ix[u]).astype(np.float32)[inv]
            nbz[d], nbm[d] = zz, m
        diff = np.where(nbm[-1][:, None], z - nbz[-1], 0)
        cnt = 1 + sum(nbm[d].astype(np.float32) for d in LAGS)
        s1 = z + sum(nbz[d] for d in LAGS)
        s2 = z * z + sum(nbz[d] ** 2 for d in LAGS)
        mu = s1 / cnt[:, None]
        sd = np.sqrt(np.maximum(s2 / cnt[:, None] - mu * mu, 0))
        sd[cnt == 1] = 0
        return np.concatenate([z, diff, sd], 1).astype(np.float16)


def store(root, dom):
    import behav as BH
    BH.BEE = root
    return BH.Store(dom, 'v2', 'dense', 1024)


def roles_of(dom, obs):
    import v20 as V
    c = V.dcfg(dom)
    r = {}
    for sc in c['series']:
        for v in json.loads((REPO / sc['plan']).read_text())['videos']:
            r[v['obs']] = v['role']
    return np.array([r.get(str(o), 'train') for o in obs])


def frame_codes(sae, norm, T, frames_of, nF, dev):
    import torch
    fo = frames_of[T.tf]
    sel = np.flatnonzero(fo >= 0)
    X = torch.zeros((nF, sae.n_latents), device=dev)
    for c0 in range(0, len(sel), 200_000):
        s = sel[c0:c0 + 200_000]
        x = torch.from_numpy(T.x(s)).to(dev)
        with torch.no_grad():
            c = sae.encode(norm(x.float()), mode='threshold').float()
        X.index_reduce_(0, torch.from_numpy(fo[s]).to(dev), c, 'amax', include_self=True)
    return X.half().cpu().numpy()


def pmap(Pr, Fs):
    key = lambda F_: pd.MultiIndex.from_arrays([F_.vid.values.astype(int), F_.k.values.astype(int)])  # noqa: E731
    return pd.Series(np.arange(len(Pr)), index=key(Pr)).reindex(key(Fs)).fillna(-1).values.astype(np.int64)


def cmd_run(a):
    import torch
    from src.eci.v2 import sae as SA
    import v20 as V
    dom = a.dom
    dev = torch.device('cuda')
    s = V.R.spec()
    t0 = time.time()
    St = store(EC / 'v2_behav', dom)
    T = Tok(St, St.keep('N2'))
    tr = roles_of(dom, St.F.obs.values)[T.tf]
    rng = np.random.default_rng(0)
    X, info = {}, {'tokens_total': int(len(T.ix)), 'neighbour_present': {d: float((v >= 0).mean()) for d, v in T.nb.items()}}
    for r in ('train', 'calib', 'eval'):
        ii = np.flatnonzero(tr == r)
        ii = np.sort(rng.choice(ii, min(len(ii), CAP[r]), replace=False))
        info[f'tokens_{r}'] = int(len(ii))
        x = np.concatenate([T.x(ii[c:c + 500_000]) for c in range(0, len(ii), 500_000)])
        X[r] = V.HostTok(torch.from_numpy(x), dev) if r == 'train' else torch.from_numpy(x)
        log(f'{dom} {r}: {len(ii):,} tokens ({time.time() - t0:.0f}s)')
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = {**s['sae'], 'context_window': 0, 'input': 'DYN3 [z, dz, std5]'}
    recs = {}
    for seed in SEEDS:
        od = BEE / dom / 'sae' / f'dyn3_{dom}_s{seed}'
        recs[seed] = SA.train(cfg, seed, X['train'], X['calib'], X['eval'], od, dev)
        log(f'{od.name}: eval FVE {recs[seed]["eval_fve"]:.5f} L0 {recs[seed]["eval_l0"]:.2f} ({recs[seed]["train_time_s"]:.0f}s)')
        (OUT / f'{dom}_train.json').write_text(json.dumps({'data': info, 'seeds': recs}, indent=1, default=float))
    del X
    od = BEE / dom / 'codes'
    od.mkdir(parents=True, exist_ok=True)
    Prb = pd.read_parquet(REPO / f'results/vision/eci_v2_behav/{dom}/probe.parquet')
    bmap = pmap(Prb, St.F)
    Se = store(EC / 'v2_prob', dom)
    Te = Tok(Se, Se.keep('N2'))
    Pre = pd.read_parquet(REPO / f'results/vision/eci_v2_prob/{dom}/probe.parquet')
    emap = pmap(Pre, Se.F)
    for seed in SEEDS:
        sae, norm = SA.load(BEE / dom / 'sae' / f'dyn3_{dom}_s{seed}' / 'sae_theta.pt', dev)
        np.save(od / f'dyn3_s{seed}_eval.npy', frame_codes(sae, norm, Te, emap, len(Pre), dev))
        np.save(od / f'dyn3_s{seed}_probe.npy', frame_codes(sae, norm, T, bmap, len(Prb), dev))
        log(f'{dom} s{seed}: codes saved')


def cmd_score(a):
    import analysis as AN
    import eval as EV
    import v20_gate as G
    import v20_prox as PX
    dom = a.dom
    if dom == 'mice':
        F = EV.frame_table('mice')
        SP = AN.splits(F.unit.values, EV.SPLITS)
        starts = AN.seq_starts(F)
        S = json.loads((G.OUT / 'mice_scores.json').read_text())
        v1, elig = S['v1'], S['eligibility']
        proxy = np.load(G.OUT / 'proxy_mice.npy')
        take = lambda X: AN.smooth_seq(X[F.code_row.values], starts)  # noqa: E731
    else:
        F, Pr = PX.ants_probe_table()
        SP = AN.splits(F.unit.values, EV.SPLITS)
        S = json.loads((G.OUT / 'antsprobe_scores.json').read_text())
        v1 = {str(k): v for k, v in S['v1'].items()}
        elig = S['eligibility']
        proxy = np.load(G.OUT / 'proxy_antsprobe.npy')
        take = lambda X: X  # noqa: E731
    res, gate, prox = {}, {}, {}
    use = np.ones(len(F), bool)
    for seed in SEEDS:
        X = take(np.load(BEE / dom / 'codes' / f'dyn3_s{seed}_eval.npy').astype(np.float32))
        res[seed] = G.score(dom, X, use, F, SP)
        gate[seed] = G.gate_rows(dom, v1, res[seed], elig)
        prox[seed] = PX.strat(F, X, use, SP, res[seed], proxy)
        log(f'{dom} dyn3 s{seed}: PASS {sum(r["PASS"] for r in gate[seed].values() if r["eligible"])} / '
            f'{sum(r["eligible"] for r in gate[seed].values())}')
    from scipy.stats import rankdata
    z = np.load(REPO / f'results/vision/eci_v2_0B/v1_vs_v2/me_{dom}.npz')
    ME, AR = z['me'], z['area']
    ok = np.isfinite(ME)
    zs = lambda r: (r - r.mean()) / r.std()  # noqa: E731
    zM, zA = zs(rankdata(ME[ok])), zs(rankdata(AR[ok]))
    dyn = {}
    for seed in SEEDS:
        C = np.load(BEE / dom / 'codes' / f'dyn3_s{seed}_probe.npy').astype(np.float32)
        n = e = 0
        for j in range(256):
            c = C[:, j]
            if (c > 0).mean() < 0.01:
                continue
            rc = zs(rankdata(c[ok]))
            rm, ra = float((rc * zM).mean()), float((rc * zA).mean())
            e += 1
            n += int(rm >= 0.10 and rm > abs(ra))
        dyn[seed] = {'dynamic_256': n, 'eligible_256': e, 'share': n / max(e, 1)}
    out = {'gate': gate, 'prox': prox, 'dyn256': dyn}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'{dom}_score.json').write_text(json.dumps(out, indent=1, default=float))
    log(json.dumps(dyn))


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=('run', 'score'))
    p.add_argument('--dom', choices=('mice', 'ants'))
    a = p.parse_args()
    {'run': cmd_run, 'score': cmd_score}[a.step](a)


if __name__ == '__main__':
    main()
