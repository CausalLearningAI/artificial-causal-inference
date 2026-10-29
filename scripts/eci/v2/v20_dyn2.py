"""
ECI v2.0 dynamic-input arms (pre-registered in scripts/eci/v2/v20_gate.py AMENDMENT 3): X0 = [x], A = [x, U1 c],
B = [x, U1 c, sqrt(mean_k (U2 r_k)^2)] on 5 fps same-position token sequences (N2 mask), blocks rescaled to the mean
squared norm of x, one global TokenNorm; Matryoshka BatchTopK 2048 (128/256/512/2048) k16 30k, seeds 990-992.
Steps: run --dom D (GPU: PCA, train 3 arms x 3 seeds, codes) ; score --dom D (CPU)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
for p in ('', 'scripts/eci/v2', 'scripts/eci/v2_eval', 'scripts/eci/v2_prob', 'scripts/eci/v2_behav'):
    sys.path.insert(0, str(REPO / p))
EC = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes')
BEE = EC / 'v2_0B/dyn2'
OUT = REPO / 'results/vision/eci_v2_0B/dyn2'
SEEDS = (990, 991, 992)
ARMS = {'X0': 1024, 'A': 1152, 'B': 1280}
WARMS = {'A10': (1152, 0.10), 'A20': (1152, 0.20), 'B10': (1280, 0.10), 'B20': (1280, 0.20)}  # AMENDMENT 6


def wscale(x, arm):
    """equal-norm features [x, b1, b2] -> motion blocks rescaled to a share w of the mean squared norm (AMENDMENT 6)."""
    d, w = WARMS[arm]
    nb = 1 if d == 1152 else 2
    f = (w / (1 - w) / nb) ** 0.5  # each equal-norm motion block has the msn of x
    y = x[:, :d].clone()
    y[:, 1024:] *= f
    return y
CAP = {'train': 12_000_000, 'calib': 1_500_000, 'eval': 1_500_000}
NPCA, KP = 1_000_000, 128
log = lambda s: print(time.strftime('%H:%M:%S'), s, flush=True)  # noqa: E731


class Seq:
    """kept N2 tokens of a behav-style store and their same-position tokens at t-2..t+2."""

    def __init__(self, St):
        self.St = St
        self.ix = np.flatnonzero(St.keep('N2'))
        o = pd.factorize(St.F.vid.values)[0].astype(np.int64)
        k = St.F.k.values.astype(np.int64)
        tf, pos = St.tf[self.ix], St.pos[self.ix]
        key = lambda d: o[tf] * (1 << 40) + (k[tf] + d + 8) * (1 << 16) + pos  # noqa: E731
        cur = pd.Series(np.arange(len(self.ix)), index=key(0))
        self.nb = {d: cur.reindex(key(d)).fillna(-1).values.astype(np.int64) for d in (-2, -1, 1, 2)}
        self.tf = tf
        b = St.bounds
        self.bs = np.array([x[0] for x in b] + [b[-1][1]])
        log(f'Seq: {len(self.ix):,} tokens; neighbours ' + ', '.join(f'{d}: {(v >= 0).mean():.3f}' for d, v in self.nb.items()))

    def get(self, idx):
        out = np.empty((len(idx), 1024), np.float16)
        o = np.argsort(idx)
        si = idx[o]
        for i in range(len(self.bs) - 1):
            a, b = np.searchsorted(si, [self.bs[i], self.bs[i + 1]])
            if b > a:
                out[o[a:b]] = self.St.mm[i][si[a:b] - self.bs[i]]
        return out

    def raw(self, sel, dev):
        """-> x (n, 1024), c (n, 1024), r list of ((n, 1024), mask) for k in -2..2, on dev (float32)."""
        import torch
        x = torch.from_numpy(self.get(self.ix[sel])).to(dev).float()
        nbx, nbm = {}, {}
        for d in (-2, -1, 1, 2):
            pv = self.nb[d][sel]
            m = pv >= 0
            z = torch.zeros_like(x)
            if m.any():
                u, inv = np.unique(pv[m], return_inverse=True)
                zz = torch.from_numpy(self.get(self.ix[u])).to(dev).float()[torch.from_numpy(inv).to(dev)]
                z[torch.from_numpy(m).to(dev)] = zz
            nbx[d], nbm[d] = z, torch.from_numpy(m).to(dev)
        mp, mm_ = nbm[1][:, None], nbm[-1][:, None]
        c = torch.where(mp & mm_, (nbx[1] - nbx[-1]) / 2, torch.where(mp, nbx[1] - x, torch.where(mm_, x - nbx[-1], 0 * x)))
        cnt = 1 + sum(nbm[d].float() for d in nbx)
        mean = (x + sum(nbx[d] * nbm[d][:, None] for d in nbx)) / cnt[:, None]
        rs = [(x - mean, torch.ones_like(nbm[1]))] + [((nbx[d] - mean) * nbm[d][:, None], nbm[d]) for d in nbx]
        return x, c, rs, cnt


def fit_pca(S, sel, dev):
    import torch
    C1 = torch.zeros((1024, 1024), device=dev, dtype=torch.float64)
    C2 = torch.zeros_like(C1)
    n1 = n2 = 0
    for a in range(0, len(sel), 100_000):
        x, c, rs, cnt = S.raw(sel[a:a + 100_000], dev)
        C1 += (c.T.double() @ c.double())
        n1 += len(c)
        for r, m in rs:
            rm = r[m]
            C2 += rm.T.double() @ rm.double()
            n2 += len(rm)
    out = {}
    for nm, C, n in (('U1', C1, n1), ('U2', C2, n2)):
        w, v = torch.linalg.eigh(C / n)
        out[nm] = v[:, -KP:].flip(1).float().T.contiguous()  # (128, 1024), descending variance
        out[nm + '_var_share'] = float(w[-KP:].sum() / w.sum())
    return out


def feats(S, sel, P, dev):
    """-> (n, 1280) fp16 [x, s1 U1 c, s2 sqrt(mean_k (U2 r_k)^2)] (scales s1, s2 in P; 1 before they are set)."""
    import torch
    x, c, rs, cnt = S.raw(sel, dev)
    b1 = c @ P['U1'].T
    acc = torch.zeros((len(x), KP), device=dev)
    for r, m in rs:
        acc += ((r @ P['U2'].T) ** 2) * m[:, None]
    b2 = torch.sqrt(acc / cnt[:, None])
    return torch.cat([x, P.get('s1', 1.0) * b1, P.get('s2', 1.0) * b2], 1)


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


def pmap(Pr, Fs):
    key = lambda F_: pd.MultiIndex.from_arrays([F_.vid.values.astype(int), F_.k.values.astype(int)])  # noqa: E731
    return pd.Series(np.arange(len(Pr)), index=key(Pr)).reindex(key(Fs)).fillna(-1).values.astype(np.int64)


def cmd_run(a):
    import torch
    from src.eci.v2 import sae as SA
    import v20 as V
    dom = a.dom
    dev = torch.device('cuda')
    t0 = time.time()
    S = Seq(store(EC / 'v2_behav', dom))
    tr = roles_of(dom, S.St.F.obs.values)[S.tf]
    rng = np.random.default_rng(0)
    trn = np.flatnonzero(tr == 'train')
    P = fit_pca(S, np.sort(rng.choice(trn, min(NPCA, len(trn)), replace=False)), dev)
    smp = np.sort(rng.choice(trn, 200_000, replace=False))
    f = feats(S, smp, P, dev)
    ms = [float((f[:, a_:b_] ** 2).sum(1).mean()) for a_, b_ in ((0, 1024), (1024, 1152), (1152, 1280))]
    P['s1'], P['s2'] = float(np.sqrt(ms[0] / ms[1])), float(np.sqrt(ms[0] / ms[2]))
    info = {'tokens_total': int(len(S.ix)), 'U1_var_share': P['U1_var_share'], 'U2_var_share': P['U2_var_share'],
            'mean_sq_norm_before': ms, 's1': P['s1'], 's2': P['s2'],
            'neighbour_present': {d: float((v >= 0).mean()) for d, v in S.nb.items()}}
    BEE.mkdir(parents=True, exist_ok=True)
    torch.save({k: v for k, v in P.items()}, BEE / f'{dom}_pca.pt')
    log(f'{dom} PCA: {json.dumps(info)} ({time.time() - t0:.0f}s)')
    X = {}
    for r in ('train', 'calib', 'eval'):
        ii = np.flatnonzero(tr == r)
        ii = np.sort(rng.choice(ii, min(len(ii), CAP[r]), replace=False))
        X[r] = torch.cat([feats(S, ii[c:c + 200_000], P, dev).half().cpu() for c in range(0, len(ii), 200_000)])
        info[f'tokens_{r}'] = int(len(ii))
        log(f'{dom} {r}: {len(ii):,} ({time.time() - t0:.0f}s)')
    OUT.mkdir(parents=True, exist_ok=True)
    s = V.R.spec()
    recs = {}
    for arm, d in ARMS.items():
        Xtr = V.HostTok(X['train'][:, :d].contiguous(), dev)
        Xca, Xev = X['calib'][:, :d].contiguous(), X['eval'][:, :d].contiguous()
        for seed in SEEDS:
            od = BEE / dom / 'sae' / f'{arm}_s{seed}'
            r_ = SA.train({**s['sae'], 'context_window': 0, 'norm': 'global', 'arm': arm}, seed, Xtr, Xca, Xev, od, dev)
            recs[f'{arm}_s{seed}'] = r_
            log(f'{dom} {arm} s{seed}: FVE {r_["eval_fve"]:.5f} ({r_["train_time_s"]:.0f}s)')
            (OUT / f'{dom}_train.json').write_text(json.dumps({'data': info, 'runs': recs}, indent=1, default=float))
        del Xtr
    del X
    codes(dom, S, P, dev, ('eval', 'probe'), t0)


def codes(dom, S, P, dev, which, t0, arms=None):
    """frame-max codes of every arm / seed on the eval frames (v2_prob store) and the suite probe frames (training store)."""
    import torch
    from src.eci.v2 import sae as SA
    od = BEE / dom / 'codes'
    od.mkdir(parents=True, exist_ok=True)
    sets = {'probe': (S, pd.read_parquet(REPO / f'results/vision/eci_v2_behav/{dom}/probe.parquet'))}
    if 'eval' in which:
        sets['eval'] = (Seq(store(EC / 'v2_prob', dom)), pd.read_parquet(REPO / f'results/vision/eci_v2_prob/{dom}/probe.parquet'))
    for nm, (Sq, Pr) in sets.items():
        if nm not in which:
            continue
        fo = pmap(Pr, Sq.St.F)[Sq.tf]
        sel = np.flatnonzero(fo >= 0)
        arms_ = arms or ARMS
        saes = {f'{arm}_s{seed}': SA.load(BEE / dom / 'sae' / f'{arm}_s{seed}' / 'sae_theta.pt', dev)
                for arm in arms_ for seed in SEEDS}
        Xc = {k: torch.zeros((len(Pr), 2048), device=dev) for k in saes}
        for c0 in range(0, len(sel), 50_000):
            s_ = sel[c0:c0 + 50_000]
            f = feats(Sq, s_, P, dev)
            fl = torch.from_numpy(fo[s_]).to(dev)
            for k, (sae, norm) in saes.items():
                arm = k.split('_')[0]
                with torch.no_grad():
                    xin = wscale(f, arm) if arm in WARMS else f[:, :ARMS[arm]]
                    cc = sae.encode(norm(xin), mode='threshold').float()
                Xc[k].index_reduce_(0, fl, cc, 'amax', include_self=True)
        for k, x in Xc.items():
            np.save(od / f'{k}_{nm}.npy', x.half().cpu().numpy())
        log(f'{dom} {nm}: codes saved ({time.time() - t0:.0f}s)')
        del Xc


def cmd_runw(a):
    """AMENDMENT 6: train the down-weighted arms on the same token sample, then their codes."""
    import torch
    from src.eci.v2 import sae as SA
    import v20 as V
    dom = a.dom
    dev = torch.device('cuda')
    t0 = time.time()
    S = Seq(store(EC / 'v2_behav', dom))
    P = torch.load(BEE / f'{dom}_pca.pt', map_location=dev)
    tr = roles_of(dom, S.St.F.obs.values)[S.tf]
    rng = np.random.default_rng(0)
    trn = np.flatnonzero(tr == 'train')
    rng.choice(trn, min(NPCA, len(trn)), replace=False)  # replay the run step's draws (PCA sample, scale sample)
    rng.choice(trn, 200_000, replace=False)
    X = {}
    for r in ('train', 'calib', 'eval'):
        ii = np.flatnonzero(tr == r)
        ii = np.sort(rng.choice(ii, min(len(ii), CAP[r]), replace=False))
        X[r] = torch.cat([feats(S, ii[c:c + 100_000], P, dev).half().cpu() for c in range(0, len(ii), 100_000)])
        log(f'{dom} {r}: {len(ii):,} ({time.time() - t0:.0f}s)')
    s = V.R.spec()
    recs = {}
    for arm in WARMS:
        Xtr = V.HostTok(wscale(X['train'], arm).contiguous(), dev)
        Xca, Xev = wscale(X['calib'], arm), wscale(X['eval'], arm)
        for seed in SEEDS:
            od = BEE / dom / 'sae' / f'{arm}_s{seed}'
            r_ = SA.train({**s['sae'], 'context_window': 0, 'norm': 'global', 'arm': arm}, seed, Xtr, Xca, Xev, od, dev)
            recs[f'{arm}_s{seed}'] = r_
            log(f'{dom} {arm} s{seed}: FVE {r_["eval_fve"]:.5f} ({r_["train_time_s"]:.0f}s)')
            (OUT / f'{dom}_trainw.json').write_text(json.dumps(recs, indent=1, default=float))
        del Xtr
    del X
    codes(dom, S, P, dev, ('eval', 'probe'), t0, arms=WARMS)


def cmd_pcodes(a):
    """resume: probe-frame codes only (the eval codes exist), PCA from BEE/<dom>_pca.pt."""
    import torch
    dev = torch.device('cuda')
    S = Seq(store(EC / 'v2_behav', a.dom))
    P = torch.load(BEE / f'{a.dom}_pca.pt', map_location=dev)
    codes(a.dom, S, P, dev, tuple(a.which.split(',')), time.time())


def cmd_score(a):
    import torch
    import analysis as AN
    import eval as EV
    import v20_gate as G
    import v20_prox as PX
    from scipy.stats import rankdata
    dom = a.dom
    if dom == 'mice':
        F = EV.frame_table('mice')
        S = json.loads((G.OUT / 'mice_scores.json').read_text())
        v1, elig = S['v1'], S['eligibility']
        proxy = np.load(G.OUT / 'proxy_mice.npy')
        starts = AN.seq_starts(F)
        take = lambda X: AN.smooth_seq(X[F.code_row.values], starts)  # noqa: E731
    else:
        F, _ = PX.ants_probe_table()
        S = json.loads((G.OUT / 'antsprobe_scores.json').read_text())
        v1, elig = {str(k): v for k, v in S['v1'].items()}, S['eligibility']
        proxy = np.load(G.OUT / 'proxy_antsprobe.npy')
        take = lambda X: X  # noqa: E731
    SP = AN.splits(F.unit.values, EV.SPLITS)
    use = np.ones(len(F), bool)
    z = np.load(REPO / f'results/vision/eci_v2_0B/v1_vs_v2/me_{dom}.npz')
    ok = np.isfinite(z['me'])
    zs = lambda r: (r - r.mean()) / r.std()  # noqa: E731
    zM, zA = zs(rankdata(z['me'][ok])), zs(rankdata(z['area'][ok]))
    out = {'v2.0': {k: G.gate_rows(dom, v1, S['v2'][k], elig) for k in ('hmm', 'smooth')}}
    for arm in (WARMS if a.weighted else ARMS):
        for seed in SEEDS:
            k = f'{arm}_s{seed}'
            X = take(np.load(BEE / dom / 'codes' / f'{k}_eval.npy').astype(np.float32))
            res = G.score(dom, X, use, F, SP)
            gate = {c: r for c, r in G.gate_rows(dom, v1, res, elig).items() if c.split('|')[1] not in G.ONLID}
            prox = PX.strat(F, X, use, SP, res, proxy)
            C = np.load(BEE / dom / 'codes' / f'{k}_probe.npy').astype(np.float32)
            n = e = 0
            for j in range(256):
                cj = C[:, j]
                if (cj > 0).mean() < 0.01:
                    continue
                rc = zs(rankdata(cj[ok]))
                rm, ra = float((rc * zM).mean()), float((rc * zA).mean())
                e += 1
                n += int(rm >= 0.10 and rm > abs(ra))
            ck = torch.load(BEE / dom / 'sae' / k / 'sae_theta.pt', map_location='cpu', weights_only=False)
            Wd = [v for kk, v in ck['state_dict'].items() if 'W_dec' in kk][0].float()
            Wd = Wd if Wd.shape[0] == 2048 else Wd.T
            share = (Wd[:, 1024:] ** 2).sum(1) / (Wd ** 2).sum(1)
            out[k] = {'gate': gate, 'prox': prox, 'dyn256': {'dynamic': n, 'eligible': e, 'share': n / max(e, 1)},
                      'n_dec_share_gt_0.25': int((share > 0.25).sum()), 'n_dec_share_gt_0.25_256': int((share[:256] > 0.25).sum()),
                      'n_pass': sum(r['PASS'] for r in gate.values() if r['eligible']),
                      'n_pass_am4': sum(r['PASS_am4'] for r in gate.values() if r['eligible'])}
            log(f'{dom} {k}: PASS {out[k]["n_pass"]} / am4 {out[k]["n_pass_am4"]} of {sum(r["eligible"] for r in gate.values())}; '
                f'dyn256 {n}/{e}; dec share>0.25 {out[k]["n_dec_share_gt_0.25"]}; '
                + ', '.join(f'{c} AUROC {r["v2_AUROC"]:.3f} top1 {r["v2_top1_prec"]:.3f}' for c, r in gate.items() if r['eligible']))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'{dom}_score{"w" if a.weighted else ""}.json').write_text(json.dumps(out, indent=1, default=float))


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=('run', 'score', 'pcodes', 'runw'))
    p.add_argument('--dom', choices=('mice', 'ants'))
    p.add_argument('--which', default='probe')
    p.add_argument('--weighted', action='store_true')
    a = p.parse_args()
    {'run': cmd_run, 'score': cmd_score, 'pcodes': cmd_pcodes, 'runw': cmd_runw}[a.step](a)


if __name__ == '__main__':
    main()
