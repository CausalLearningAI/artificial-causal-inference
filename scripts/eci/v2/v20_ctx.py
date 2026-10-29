"""
ECI v2.0 Part B variant CTX (pre-registered in scripts/eci/v2/v20_gate.py AMENDMENT 2): the v2.0 N2 token set dilated
by one token (3 x 3 on the DINOv3 grid), frozen v2.0 encoder / resolution / canonical frame.
Steps (scripts/eci/v2/v20.sh is not used; GPU jobs log GPU util):
    encode --dom D --set probe|eval --shard i --nshard n  (GPU) tokens of the frame set -> BEE/<D>/<set>/s<i>/
    train  --dom D                                        (GPU) SAE seeds 980-982 on the probe tokens (16M cap)
    codes  --dom D                                        (GPU) frame-max codes of the eval and probe frame sets
    score  --dom D                                        (CPU) gate + proximity test + dynamic share @256
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
for p in ('', 'scripts/eci/v2', 'scripts/eci/v2_eval', 'scripts/eci/v2_prob', 'scripts/eci/v2_behav'):
    sys.path.insert(0, str(REPO / p))
EC = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes')
BEE = EC / 'v2_0B/ctx'
OUT = REPO / 'results/vision/eci_v2_0B/ctx'
SEEDS = (980, 981, 982)
CAP = {'train': 16_000_000, 'calib': 1_500_000, 'eval': 1_500_000}
log = lambda s: print(time.strftime('%H:%M:%S'), s, flush=True)  # noqa: E731


def frame_set(dom, which):
    """DataFrame series (plan series), obs, k in the order the codes are returned."""
    import v20 as V
    if which == 'probe':
        F = pd.read_parquet(REPO / f'results/vision/eci_v2_behav/{dom}/probe.parquet')[['series', 'obs', 'k']]
    else:
        import eval as EV
        F = EV.frame_table(dom)[['series', 'obs', 'k']].copy()
    c = V.dcfg(dom)
    have = {(sc['name'], v['obs']) for sc in c['series'] for v in json.loads((REPO / sc['plan']).read_text())['videos']}
    F['series'] = [s if (s, o) in have else 'v4' for s, o in zip(F.series.values, F.obs.values)]  # v4_infested -> v4 plan
    assert all((s, o) in have for s, o in zip(F.series.values, F.obs.values))
    return F.reset_index(drop=True)


def cmd_encode(a):
    import torch
    import torch.nn.functional as Fn
    import v20 as V
    from src.eci.v2.encode import Encoder
    from src.eci.v2.video import Prefetch
    dev = torch.device('cuda')
    dom = a.dom
    c = V.dcfg(dom)
    NS = V.patch_series(dom)
    enc = Encoder(dev)
    SE = {sc['name']: NS(sc, dev, encoder=enc) for sc in c['series']}
    F = frame_set(dom, a.set)
    F['fid'] = np.arange(len(F))
    vids = sorted({(s, o) for s, o in zip(F.series.values, F.obs.values)})[a.shard::a.nshard]
    jobs, meta = [], []
    for s, o in vids:
        g = F[(F.series == s) & (F.obs == o)].sort_values('k')
        ks, fids = g.k.values, g.fid.values
        br = np.r_[0, np.flatnonzero(np.diff(ks) != (np.diff(ks)[0] if len(ks) > 1 else 1)) + 1, len(ks)]
        for b0, b1 in zip(br[:-1], br[1:]):
            step = int(ks[b0 + 1] - ks[b0]) if b1 - b0 > 1 else 1
            for c0 in range(b0, b1, 240):
                c1 = min(b1, c0 + 240)
                jobs.append(((len(meta),), SE[s].job(SE[s].V[o], int(ks[c0]), c1 - c0, step)))
                meta.append((s, o, ks[c0:c1], fids[c0:c1]))
    loc = Path(os.environ['LOCAL_DIR']) / f'ctx_{dom}_{a.set}_s{a.shard}'
    loc.mkdir(parents=True, exist_ok=True)
    ftok = open(loc / 'tok.f16', 'wb')
    tfr, tps, recs = [], [], []
    t0 = time.time()
    n_n2 = n_ctx = 0
    pf = Prefetch(jobs, max(4, int(os.environ.get('SLURM_CPUS_PER_TASK', 16)) - 4), fn=SE[vids[0][0]].run_job)
    for (j,), frm in pf:
        s, o, kk, ff = meta[j]
        se = SE[s]
        bs = 16 if se.S >= 1000 else 32
        for b in range(0, len(kk), bs):
            with torch.no_grad():
                x, fd = se.frames(se.V[o], frm[b:b + bs], kk[b:b + bs])
                z, keep = se.tokens(x, fd)
                G = se.G
                dil = Fn.max_pool2d(keep.float().view(-1, 1, G, G), 3, 1, 1).flatten(1) > 0.5
            n_n2 += int(keep.sum())
            n_ctx += int(dil.sum())
            for ii in range(len(dil)):
                pt = torch.nonzero(dil[ii]).flatten()
                ftok.write(z[ii, pt].cpu().numpy().tobytes())
                tfr.append(np.full(len(pt), len(recs), np.int32))
                tps.append(pt.cpu().numpy().astype(np.int16))
                recs.append({'series': s, 'obs': o, 'k': int(kk[b + ii]), 'fid': int(ff[b + ii]), 'G': G, 'n': int(len(pt)),
                             'role': se.V[o]['role']})
        if j % 50 == 0:
            el = time.time() - t0
            log(f'ctx {dom} {a.set} s{a.shard}: {j}/{len(jobs)} runs, {len(recs):,} frames, {len(recs) / max(el, 1):.1f} fps '
                f'(GPU {enc.gpu_s:.0f}s); tokens N2 {n_n2:,} -> CTX {n_ctx:,}')
    ftok.close()
    np.save(loc / 'tok_frame.npy', np.concatenate(tfr))
    np.save(loc / 'tok_pos.npy', np.concatenate(tps))
    pd.DataFrame(recs).to_parquet(loc / 'frames.parquet')
    info = {'frames': len(recs), 'tokens_N2': n_n2, 'tokens_CTX': n_ctx, 'wall_s': time.time() - t0, 'gpu_s': enc.gpu_s,
            'gpu': torch.cuda.get_device_name(0)}
    log(json.dumps(info))
    (OUT / dom).mkdir(parents=True, exist_ok=True)
    (OUT / dom / f'encode_{a.set}_s{a.shard}.json').write_text(json.dumps(info, indent=1))
    V.R.atomic_copy(loc, BEE / dom / a.set / f's{a.shard}')
    import shutil
    shutil.rmtree(loc)


class Store:
    def __init__(self, dom, which):
        sh = sorted([p for p in (BEE / dom / which).glob('s*') if not p.name.endswith('.tmp')], key=lambda p: int(p.name[1:]))
        self.F, self.tf, self.pos, self.mm, self.b = [], [], [], [], [0]
        o = 0
        for p in sh:
            F_ = pd.read_parquet(p / 'frames.parquet')
            tf = np.load(p / 'tok_frame.npy')
            self.F.append(F_)
            self.tf.append(tf + o)
            self.pos.append(np.load(p / 'tok_pos.npy'))
            self.mm.append(np.memmap(p / 'tok.f16', np.float16, 'r', shape=(len(tf), 1024)))
            self.b.append(self.b[-1] + len(tf))
            o += len(F_)
        self.F = pd.concat(self.F, ignore_index=True)
        self.tf = np.concatenate(self.tf)

    def get(self, idx):
        out = np.empty((len(idx), 1024), np.float16)
        o = np.argsort(idx)
        si = idx[o]
        for i, mm in enumerate(self.mm):
            a, b = np.searchsorted(si, [self.b[i], self.b[i + 1]])
            if b > a:
                out[o[a:b]] = mm[si[a:b] - self.b[i]]
        return out


def cmd_train(a):
    import torch
    from src.eci.v2 import sae as SA
    import v20 as V
    dev = torch.device('cuda')
    St = Store(a.dom, 'probe')
    role = St.F.role.values[St.tf]
    rng = np.random.default_rng(0)
    X, info = {}, {'tokens_total': int(len(St.tf)), 'frames': int(len(St.F))}
    for r in ('train', 'calib', 'eval'):
        ii = np.flatnonzero(role == r)
        ii = np.sort(rng.choice(ii, min(len(ii), CAP[r]), replace=False))
        x = torch.from_numpy(np.concatenate([St.get(ii[c:c + 2_000_000]) for c in range(0, len(ii), 2_000_000)]))
        X[r] = V.HostTok(x, dev) if r == 'train' else x
        info[f'tokens_{r}'] = int(len(ii))
        log(f'{a.dom} {r}: {len(ii):,}')
    s = V.R.spec()
    recs = {}
    OUT.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        od = BEE / a.dom / 'sae' / f'ctx_{a.dom}_s{seed}'
        recs[seed] = SA.train({**s['sae'], 'context_window': 0, 'mask': 'CTX'}, seed, X['train'], X['calib'], X['eval'], od, dev)
        log(f'{od.name}: FVE {recs[seed]["eval_fve"]:.5f} ({recs[seed]["train_time_s"]:.0f}s)')
        (OUT / f'{a.dom}_train.json').write_text(json.dumps({'data': info, 'seeds': recs}, indent=1, default=float))


def cmd_codes(a):
    import torch
    from src.eci.v2 import sae as SA
    dev = torch.device('cuda')
    od = BEE / a.dom / 'codes'
    od.mkdir(parents=True, exist_ok=True)
    for which in ('eval', 'probe'):
        St = Store(a.dom, which)
        nF = int(St.F.fid.max()) + 1
        assert len(St.F) == nF and St.F.fid.nunique() == nF, (which, len(St.F), nF)
        fo = St.F.fid.values[St.tf]
        for seed in SEEDS:
            sae, norm = SA.load(BEE / a.dom / 'sae' / f'ctx_{a.dom}_s{seed}' / 'sae_theta.pt', dev)
            X = torch.zeros((nF, sae.n_latents), device=dev)
            for c0 in range(0, len(St.tf), 500_000):
                ii = np.arange(c0, min(len(St.tf), c0 + 500_000))
                x = torch.from_numpy(St.get(ii)).to(dev)
                with torch.no_grad():
                    cc = sae.encode(norm(x.float()), mode='threshold').float()
                X.index_reduce_(0, torch.from_numpy(fo[ii]).to(dev), cc, 'amax', include_self=True)
            np.save(od / f'ctx_s{seed}_{which}.npy', X.half().cpu().numpy())
            log(f'{a.dom} {which} s{seed}: codes saved')


def cmd_score(a):
    import analysis as AN
    import eval as EV
    import v20_gate as G
    import v20_prox as PX
    from scipy.stats import rankdata
    dom = a.dom
    F = EV.frame_table(dom)
    SP = AN.splits(F.unit.values, EV.SPLITS)
    S = json.loads((G.OUT / f'{dom}_scores.json').read_text())
    proxy = np.load(G.OUT / f'proxy_{dom}.npy')
    use = np.ones(len(F), bool)
    z = np.load(REPO / f'results/vision/eci_v2_0B/v1_vs_v2/me_{dom}.npz')
    ok = np.isfinite(z['me'])
    zs = lambda r: (r - r.mean()) / r.std()  # noqa: E731
    zM, zA = zs(rankdata(z['me'][ok])), zs(rankdata(z['area'][ok]))
    out = {}
    for seed in SEEDS:
        X = np.load(BEE / dom / 'codes' / f'ctx_s{seed}_eval.npy').astype(np.float32)
        if dom == 'mice':
            X = AN.smooth_seq(X, AN.seq_starts(F))
        res = G.score(dom, X, use, F, SP)
        gate = G.gate_rows(dom, S['v1'], res, S['eligibility'])
        prox = PX.strat(F, X, use, SP, res, proxy)
        C = np.load(BEE / dom / 'codes' / f'ctx_s{seed}_probe.npy').astype(np.float32)
        n = e = 0
        for j in range(256):
            c = C[:, j]
            if (c > 0).mean() < 0.01:
                continue
            rc = zs(rankdata(c[ok]))
            rm, ra = float((rc * zM).mean()), float((rc * zA).mean())
            e += 1
            n += int(rm >= 0.10 and rm > abs(ra))
        gate = {c_: r for c_, r in gate.items() if c_.split('|')[1] not in G.ONLID}
        out[seed] = {'gate': gate, 'prox': prox, 'dyn256': {'dynamic': n, 'eligible': e, 'share': n / max(e, 1)},
                     'n_pass': sum(r['PASS'] for r in gate.values() if r['eligible'])}
        log(f'{dom} ctx s{seed}: PASS {out[seed]["n_pass"]} / {sum(r["eligible"] for r in gate.values())}; dyn256 '
            f'{n}/{e}; ' + ', '.join(f'{c_} {"P" if r["PASS"] else "F"} AUROC {r["v2_AUROC"]:.3f} top1 {r["v2_top1_prec"]:.3f}'
                                      for c_, r in gate.items() if r['eligible']))
    (OUT / f'{dom}_score.json').write_text(json.dumps(out, indent=1, default=float))


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=('encode', 'train', 'codes', 'score'))
    p.add_argument('--dom', choices=('mice', 'ants'))
    p.add_argument('--set', default='probe')
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--nshard', type=int, default=1)
    a = p.parse_args()
    {'encode': cmd_encode, 'train': cmd_train, 'codes': cmd_codes, 'score': cmd_score}[a.step](a)


if __name__ == '__main__':
    main()
