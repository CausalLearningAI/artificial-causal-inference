"""
ECI v2.1: the v2 SAE retrained on the v2.1 store (approved mask v5) in four input arms; alignment vs v1 on mice / ants.

==================================================================================================================
PRE-REGISTRATION (written 2026-10-31 BEFORE any v2.1 SAE was trained and before any score; the user's design relayed by
the coordinator).
STORE: eci_tokens/v2_full/<dom>/v2.1_m5 (per frame U_t = union of M_{t-2..t+2}; flag bit0 = in M_t (mask v5)). Training /
    readout tokens = the M_t tokens of each frame t. 5 fps, canonical orientation, frozen v2.0 encoder tokens.
ARMS (same tokens and positions; x_t = the token at an M_t position at frame t; neighbours read at the SAME position in
    frames t-2..t+2, all present inside a video by construction of U):
    S  x (1024)
    A  [x, s1 * U1 c_t] (1152), c_t = (x_{t+1} - x_{t-1}) / 2, U1 = top-128 PCA directions of the centred c (per domain)
    B  [x, s1 * U1 c_t, s2 * sqrt(mean_k (U2 (x_{t+k} - m_t))^2)] (1280), k = -2..2, m_t = mean of the window frames,
       U2 = top-128 PCA directions of the deviations (per domain)
    V  x + c_t (1024), weight 1, no PCA. Reported: median ||c_t|| / ||x_t||.
    EDGE RULE (all arms, first / last 2 frames of a video): only frames that exist are used; c_t = (x_{t+1} - x_{t-1}) / 2
    when both neighbours exist, the one-sided difference x_{t+1} - x_t or x_t - x_{t-1} when one exists, 0 when none;
    m_t and the block-2 mean over the window frames that exist (block 2 = 0 for a single frame).
    SCALING (A, B: the approved design of AMENDMENT 3 of v20_gate.py): each motion block rescaled to the mean squared norm
    of x on the PCA sample (s1, s2), then one global TokenNorm; S and V use the v2.0 TokenNorm (one 1024 block).
    PCA: ~1M random training-role M_t tokens per domain (rng 0).
SAE: the frozen v2.0 recipe: Matryoshka BatchTopK 2048, k16, prefixes 128/256/512/2048, 30k steps x 4096, lr 5e-4,
    theta* on calibration-role tokens, FVE on evaluation-role tokens; train tokens: up to 16M M_t tokens of training-role
    videos (windows of 25 frames sampled uniformly per video, rng 0); seeds 1001-1003 per arm and domain (mice, ants).
READOUTS (as v2.0): frame max over the M_t tokens; continuous = 0.6 s centred mean; P(on) = per-neuron 3-state HMM
    (src/eci/v2/hmm.py) fitted label-free per domain on the scoring windows themselves (mice: the 144 x 600-frame eval
    windows; ants: the 5 fps span of every labelled video) - a STATED change from v2.0 (calibration videos), because only
    these windows are decoded here.
METRICS per arm (median and range over the 3 seeds), each against v1 (11 seeds, eval.py frames / scorer / splits):
    1 AUROC vs v1 (t CI of the v1 seed mean); 2 top-1% precision vs v1 and the gate threshold; 3 nAP and F1max for P(on)
    and continuous; 4 proximity-matched AUROC (ants, v20_prox strat, the existing per-frame proxy); 5 dynamic share @256
    (v20_vs_v1 'dyn' rule on the suite probe frames); 6 capacity @256 / full: the behaviour suite needs transformed-frame
    tokens under mask v5 that the store does not hold - NOT computed here (stated); 7 SAE eval FVE; 8 best neuron per label.
    Reference rows: v1 (11 seeds) and the current v2 (PASS B, old store, N2 mask).
LABELS: ants Y2F / B2F / groom_any on every series eval.py scores (on-lid removed); mice nn and nt (and np, nose_nose
    reported).
DECISION: the deployment gate of results/vision/eci_v2_0B/gate/gate.md, AMENDMENT 5 rule (PASS_final), per seed; an arm's
    count = median over its seeds of the eligible cells passed (mice + ants). PICK: among the arms with the highest count,
    the higher median AUROC across seeds (mean over eligible cells); a motion arm (A / B / V) wins over S only if it passes
    at least as many cells AND has a higher median dynamic share @256.
==================================================================================================================
Steps: prep --dom D (GPU) | train --dom D --arm X --seed s (GPU) | codes --dom D (GPU) | score --dom D (CPU) | report (CPU)
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
os.environ.setdefault('V2_OUT', str(REPO / 'results/vision/eci_v2_0B'))
os.environ.setdefault('V2_BEE', '/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_0B')
STORE = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_tokens/v2_full')
VER = 'v2.1_m5'
BEE = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_1')
OUT = REPO / 'results/vision/eci_v2_1'
ARMS = {'S': 1024, 'A': 1152, 'B': 1280, 'V': 1024}
SEEDS = (1001, 1002, 1003)
CAP = {'train': 16_000_000, 'calib': 1_500_000, 'eval': 1_500_000}
WIN, NPCA, KP = 25, 1_000_000, 128
log = lambda s: print(time.strftime('%H:%M:%S'), s, flush=True)  # noqa: E731


# ---------------------------------------------------------------------------------------------- store access
class Video:
    def __init__(self, dom, ser, obs):
        d = STORE / dom / VER / 'videos' / ser / obs
        z = np.load(d / 'index.npz')
        self.k, self.ptr, self.idx, self.flag = z['k'], z['ptr'], z['idx'].astype(np.int64), z['flag']
        self.tok = np.memmap(d / 'tok.f16', np.float16, 'r', shape=(int(self.ptr[-1]), 1024))
        self.G = int(z['G'])

    def window(self, i0, i1):
        """frames i0..i1-1 (positions in self.k) with 2-frame margins -> dict of arrays for feature building."""
        a0, a1 = max(0, i0 - 2), min(len(self.k), i1 + 2)
        r0, r1 = int(self.ptr[a0]), int(self.ptr[a1])
        tok = np.asarray(self.tok[r0:r1])  # one contiguous read
        fi = np.repeat(np.arange(a0, a1), np.diff(self.ptr[a0:a1 + 1]))
        return {'tok': tok, 'fi': fi, 'pos': self.idx[r0:r1], 'flag': self.flag[r0:r1], 'a0': a0, 'a1': a1, 'i0': i0, 'i1': i1}


def centre_rows(W):
    """rows of the M_t tokens of frames i0..i1-1 and the rows of the same positions at frames t-2..t+2 (-1 if absent)."""
    fi, pos = W['fi'], W['pos']
    key = fi * 65536 + pos
    s = pd.Series(np.arange(len(fi)), index=key)
    c = np.flatnonzero(((W['flag'] & 1) > 0) & (fi >= W['i0']) & (fi < W['i1']))
    nb = {d: s.reindex(key[c] + d * 65536).fillna(-1).values.astype(np.int64) for d in (-2, -1, 1, 2)}
    return c, nb


def raw_feats(tok, c, nb, dev):
    """-> x, cdiff, list of (deviation, mask), count (torch, float32) for the centre rows c."""
    import torch
    T = lambda r: torch.from_numpy(tok[np.where(r >= 0, r, 0)]).to(dev).float()  # noqa: E731
    x = torch.from_numpy(tok[c]).to(dev).float()
    nx = {d: T(nb[d]) for d in nb}
    nm = {d: torch.from_numpy(nb[d] >= 0).to(dev) for d in nb}
    mp, mm = nm[1][:, None], nm[-1][:, None]
    cd = torch.where(mp & mm, (nx[1] - nx[-1]) / 2, torch.where(mp, nx[1] - x, torch.where(mm, x - nx[-1], torch.zeros_like(x))))
    cnt = 1 + sum(nm[d].float() for d in nm)
    mean = (x + sum(nx[d] * nm[d][:, None] for d in nx)) / cnt[:, None]
    rs = [(x - mean, torch.ones_like(nm[1]))] + [((nx[d] - mean) * nm[d][:, None], nm[d]) for d in nx]
    return x, cd, rs, cnt


def arm_inputs(x, cd, rs, cnt, P):
    """-> {arm: (n, d) float32} for the four arms (P = PCA + scales)."""
    import torch
    b1 = cd @ P['U1'].T
    acc = torch.zeros((len(x), KP), device=x.device)
    for r, m in rs:
        acc += ((r @ P['U2'].T) ** 2) * m[:, None]
    b2 = torch.sqrt(acc / cnt[:, None])
    return {'S': x, 'A': torch.cat([x, P['s1'] * b1], 1), 'B': torch.cat([x, P['s1'] * b1, P['s2'] * b2], 1), 'V': x + cd}


def videos_of(dom):
    import v20 as V
    c = json.loads((REPO / 'configs/eci_v2/mice_v20.json').read_text()) if dom == 'mice' else V.dcfg(dom)
    out = []
    for sc in c['series']:
        for v in json.loads((REPO / sc['plan']).read_text())['videos']:
            if (STORE / dom / VER / 'videos' / sc['name'] / v['obs'] / 'index.npz').exists():
                out.append((sc['name'], v['obs'], v['role']))
    return out


# ---------------------------------------------------------------------------------------------- prep
def cmd_prep(a):
    """PCA (U1, U2), block scales, and the train / calib / eval token sample -> BEE/<dom>/data/{<role>_B.npy (scaled
    [x, b1, b2], 1280), <role>_C.npy (c_t, 1024), pca.pt}; arm inputs: S = B[:, :1024], A = B[:, :1152], B, V = x + C."""
    import torch
    dev = torch.device('cuda')
    dom = a.dom
    rng = np.random.default_rng(0)
    vids = videos_of(dom)
    t0 = time.time()
    m_per_frame = np.mean([((Video(dom, v[0], v[1]).flag & 1) > 0).sum() / len(Video(dom, v[0], v[1]).k)
                           for v in vids[::max(1, len(vids) // 6)][:6]])
    plan = []
    for r in CAP:
        vs = [v for v in vids if v[2] == r]
        n_win = max(1, int(np.ceil(1.1 * CAP[r] / m_per_frame / (WIN * max(1, len(vs))))))
        for ser, obs, role in vs:
            nf = len(Video(dom, ser, obs).k)
            starts = np.sort(rng.choice(max(1, nf - WIN), min(n_win, max(1, nf - WIN)), replace=False))
            plan += [(ser, obs, role, int(s_)) for s_ in starts]
    log(f'{dom}: {len(plan)} windows of {WIN} frames from {len(vids)} videos; M tokens / frame ~{m_per_frame:.0f}')
    cache = {}

    def win(ser, obs, s_):
        if (ser, obs) not in cache:
            cache.clear()
            cache[(ser, obs)] = Video(dom, ser, obs)
        V_ = cache[(ser, obs)]
        W = V_.window(s_, min(len(V_.k), s_ + WIN))
        c, nb = centre_rows(W)
        return (raw_feats(W['tok'], c, nb, dev) if len(c) else None)
    # pass 1: PCA sample from training windows (random order, stop at NPCA)
    pca_c, pca_r, n1 = [], [], 0
    for jj in rng.permutation([i for i, x in enumerate(plan) if x[2] == 'train']):
        ser, obs, role, s_ = plan[jj]
        r_ = win(ser, obs, s_)
        if r_ is None:
            continue
        x, cd, rs, cnt = r_
        pca_c.append(cd.cpu())
        pca_r.append(torch.cat([r[m] for r, m in rs]).cpu()[::5])
        n1 += len(cd)
        if n1 >= NPCA:
            break
    P = {}
    for nm, L in (('U1', pca_c), ('U2', pca_r)):
        Xp = torch.cat(L)[:NPCA].to(dev).double()
        Xp -= Xp.mean(0)
        w, v = torch.linalg.eigh(Xp.T @ Xp / len(Xp))
        P[nm] = v[:, -KP:].flip(1).float().T.contiguous()
        P[nm + '_var_share'] = float(w[-KP:].sum() / w.sum())
    del pca_c, pca_r
    P['s1'] = P['s2'] = 1.0
    log(f'{dom} PCA done ({time.time() - t0:.0f}s)')
    # pass 2: all windows -> unscaled [x, b1, b2] and c
    data = {r: {'B': [], 'C': []} for r in CAP}
    ms, ratio = {'x': 0.0, 'b1': 0.0, 'b2': 0.0, 'n': 0}, []
    for j, (ser, obs, role, s_) in enumerate(plan):
        r_ = win(ser, obs, s_)
        if r_ is None:
            continue
        x, cd, rs, cnt = r_
        f = arm_inputs(x, cd, rs, cnt, P)['B']
        data[role]['B'].append(f.half().cpu())
        data[role]['C'].append(cd.half().cpu())
        if role == 'train' and ms['n'] < 2_000_000:
            ms['x'] += float((x ** 2).sum())
            ms['b1'] += float((f[:, 1024:1152] ** 2).sum())
            ms['b2'] += float((f[:, 1152:] ** 2).sum())
            ms['n'] += len(x)
            ratio.append((cd.norm(dim=1) / x.norm(dim=1)).cpu().numpy())
        if j % 1000 == 0:
            log(f'{dom} prep {j}/{len(plan)} windows ({time.time() - t0:.0f}s)')
    P['s1'] = float(np.sqrt(ms['x'] / ms['b1']))
    P['s2'] = float(np.sqrt(ms['x'] / ms['b2']))
    od = BEE / dom / 'data'
    od.mkdir(parents=True, exist_ok=True)
    info = {'windows': len(plan), 'M_tokens_per_frame_est': float(m_per_frame), 'U1_var_share': P['U1_var_share'],
            'U2_var_share': P['U2_var_share'], 's1': P['s1'], 's2': P['s2'],
            'median_norm_ratio_dx_over_x': float(np.median(np.concatenate(ratio))), 'tokens': {}}
    rng2 = np.random.default_rng(1)
    import shutil
    loc = Path(os.environ['LOCAL_DIR'])
    for r in CAP:
        B_ = torch.cat(data[r]['B'])
        C_ = torch.cat(data[r]['C'])
        data[r] = None
        keep = torch.from_numpy(np.sort(rng2.choice(len(B_), min(len(B_), CAP[r]), replace=False)))
        info['tokens'][r] = {'available': int(len(B_)), 'used': int(len(keep))}
        B_ = B_[keep]
        B_[:, 1024:1152] *= P['s1']
        B_[:, 1152:] *= P['s2']
        for nm, X in (('B', B_), ('C', C_[keep])):
            np.save(loc / f'{r}_{nm}.npy', X.numpy())
            shutil.copyfile(loc / f'{r}_{nm}.npy', od / f'{r}_{nm}.npy')
            (loc / f'{r}_{nm}.npy').unlink()
        del B_, C_
    torch.save(P, od / 'pca.pt')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'{dom}_prep.json').write_text(json.dumps(info, indent=1))
    log(f'{dom} prep done: {json.dumps(info)} ({time.time() - t0:.0f}s)')


def cmd_train(a):
    import torch
    import v20 as V
    from src.eci.v2 import sae as SA
    dev = torch.device('cuda')
    od = BEE / a.dom / 'data'
    X = {}
    for r in CAP:
        B_ = np.load(od / f'{r}_B.npy', mmap_mode='r')
        if a.arm == 'V':
            X[r] = torch.from_numpy(np.asarray(B_[:, :1024], np.float32) + np.load(od / f'{r}_C.npy').astype(np.float32)).half()
        else:
            X[r] = torch.from_numpy(np.ascontiguousarray(B_[:, :ARMS[a.arm]]))
    s = V.R.spec()
    cfg = {**s['sae'], 'context_window': 0, 'arm': a.arm, 'store': VER}
    if a.arm in ('A', 'B'):
        cfg['norm'] = 'global'
    rec = SA.train(cfg, a.seed, V.HostTok(X['train'], dev), X['calib'], X['eval'],
                   BEE / a.dom / 'sae' / f'{a.arm}_s{a.seed}', dev)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'{a.dom}_train_{a.arm}_s{a.seed}.json').write_text(json.dumps(rec, indent=1, default=float))
    log(f'{a.dom} {a.arm} s{a.seed}: eval FVE {rec["eval_fve"]:.5f} L0 {rec["eval_l0"]:.2f} ({rec["train_time_s"]:.0f}s)')


# ---------------------------------------------------------------------------------------------- codes
def frame_sets(dom):
    """{'eval': DataFrame series/obs/k (eval.py order) + the decode spans; 'probe': suite probe frames}."""
    import eval as EV
    F = EV.frame_table(dom)
    E = F[['series', 'obs', 'k']].copy()
    if dom == 'mice':
        E['series'] = 'v1'
    else:
        E['series'] = E.series.where(E.series != 'v4_infested', 'v4')
    Pr = pd.read_parquet(REPO / f'results/vision/eci_v2_behav/{dom}/probe.parquet')[['series', 'obs', 'k']]
    return {'eval': E.reset_index(drop=True), 'probe': Pr.reset_index(drop=True)}


def cmd_codes(a):
    """frame-max codes over the M_t tokens of every arm / seed: the 5 fps spans of the eval videos (for the readouts) and
    the suite probe frames -> BEE/<dom>/codes/<arm>_s<seed>_{span,probe}.npy + span index."""
    import torch
    from src.eci.v2 import sae as SA
    dev = torch.device('cuda')
    dom = a.dom
    P = torch.load(BEE / dom / 'data' / 'pca.pt', map_location=dev)
    saes = {f'{arm}_s{s}': SA.load(BEE / dom / 'sae' / f'{arm}_s{s}' / 'sae_theta.pt', dev) for arm in ARMS for s in SEEDS}
    od = BEE / dom / 'codes'
    od.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for nm, Fs in frame_sets(dom).items():
        spans = []
        for (ser, obs), g in Fs.groupby(['series', 'obs'], sort=False):
            if nm == 'eval':
                spans.append((ser, obs, int(g.k.min()), int(g.k.max()) + 1))  # full 5 fps span (readouts need it)
            else:
                spans.append((ser, obs, int(g.k.min()), int(g.k.max()) + 1))
        rows = []
        for ser, obs, k0, k1 in spans:
            rows += [(ser, obs, k) for k in range(k0, k1)]
        R = pd.DataFrame(rows, columns=['series', 'obs', 'k'])
        Xc = {k: np.zeros((len(R), 2048), np.float16) for k in saes}
        base = 0
        for (ser, obs, k0, k1) in spans:
            V_ = Video(dom, ser, obs)
            kpos = {int(k): i for i, k in enumerate(V_.k)}
            i0, i1 = kpos[k0], kpos[k1 - 1] + 1
            for b0 in range(i0, i1, 600):
                W = V_.window(b0, min(i1, b0 + 600))
                c, nb = centre_rows(W)
                if len(c):
                    x, cd, rs, cnt = raw_feats(W['tok'], c, nb, dev)
                    f = arm_inputs(x, cd, rs, cnt, P)
                    fl = torch.from_numpy(W['fi'][c] - i0).to(dev)
                    nfr = min(i1, b0 + 600) - b0
                    for k, (sae, norm) in saes.items():
                        with torch.no_grad():
                            cc = sae.encode(norm(f[k.split('_')[0]]), mode='threshold').float()
                        m = torch.zeros((i1 - i0, 2048), device=dev).index_reduce_(0, fl, cc, 'amax', include_self=True)
                        Xc[k][base + b0 - i0: base + b0 - i0 + nfr] = m[b0 - i0: b0 - i0 + nfr].half().cpu().numpy()
            base += i1 - i0
        R.to_parquet(od / f'{nm}_rows.parquet')
        for k, x in Xc.items():
            np.save(od / f'{k}_{nm}.npy', x)
        log(f'{dom} {nm}: {len(R):,} frames, codes of {len(saes)} SAEs saved ({time.time() - t0:.0f}s)')


# ---------------------------------------------------------------------------------------------- score
def cmd_score(a):
    import analysis as AN
    import eval as EV
    import v20_gate as G
    import v20_prox as PX
    from multiprocessing import get_context
    from scipy.stats import rankdata
    from src.eci.v2 import readout as RO
    import v20 as V20
    dom = a.dom
    F = EV.frame_table(dom)
    SP = AN.splits(F.unit.values, EV.SPLITS)
    S = json.loads((G.OUT / f'{dom}_scores.json').read_text())
    v1, elig = S['v1'], S['eligibility']
    od = BEE / dom / 'codes'
    R = pd.read_parquet(od / 'eval_rows.parquet')
    E = frame_sets(dom)['eval']
    rix = pd.Series(np.arange(len(R)), index=pd.MultiIndex.from_frame(R)).reindex(pd.MultiIndex.from_frame(E)).values
    assert not np.isnan(rix).any()
    rix = rix.astype(np.int64)
    st = np.r_[0, np.flatnonzero((R.obs.values[1:] != R.obs.values[:-1]) | (R.series.values[1:] != R.series.values[:-1])) + 1, len(R)]
    proxy = np.load(G.OUT / f'proxy_{dom}.npy') if dom == 'ants' else None
    z = np.load(REPO / f'results/vision/eci_v2_0B/v1_vs_v2/me_{dom}.npz')
    ok = np.isfinite(z['me'])
    zs = lambda r: (r - r.mean()) / r.std()  # noqa: E731
    zM, zA = zs(rankdata(z['me'][ok])), zs(rankdata(z['area'][ok]))
    Rp = pd.read_parquet(od / 'probe_rows.parquet')
    Pr = frame_sets(dom)['probe']
    pix = pd.Series(np.arange(len(Rp)), index=pd.MultiIndex.from_frame(Rp)).reindex(pd.MultiIndex.from_frame(Pr)).values.astype(np.int64)
    use = np.ones(len(F), bool)
    out = {}
    for arm in ARMS:
        for s in SEEDS:
            k = f'{arm}_s{s}'
            f_ = OUT / dom / f'{k}.json'
            if f_.exists() and not a.force:
                out[k] = json.loads(f_.read_text())
                continue
            raw = np.load(od / f'{k}_eval.npy').astype(np.float32)
            sm = np.concatenate([RO.smooth(raw[a0:a1], 1) for a0, a1 in zip(st[:-1], st[1:])])
            # HMM P(on): fitted per neuron on these spans (label-free), applied per span
            from src.eci.v2 import hmm as HM
            Xt = np.ascontiguousarray(raw.T)
            V20._G.update(Xc=Xt, sc=st.astype(np.int64))
            with get_context('fork').Pool(int(os.environ.get('SLURM_CPUS_PER_TASK', 8))) as pool:
                prm = [None] * 2048
                for j, p in pool.imap_unordered(V20._fit_one, range(2048), chunksize=16):
                    prm[j] = p
            pon = np.zeros_like(raw)
            for j in range(2048):
                pon[:, j] = HM.apply(Xt[j], st, prm[j])[0]
            res = {}
            for rd, X in (('hmm', pon), ('smooth', sm), ('raw', raw)):
                Xe = X[rix]
                sc = G.score(dom, Xe, use, F, SP)
                gate = {c: r for c, r in G.gate_rows(dom, v1, sc, elig).items() if c.split('|')[1] not in G.ONLID}
                res[rd] = {'gate': gate, 'cells': {c: {m_: v.get(m_) for m_ in ('AUROC', 'nAP', 'F1max', 'best_neuron')}
                                                   for c, v in sc.items()}}
                if proxy is not None and rd != 'raw':
                    res[rd]['prox'] = PX.strat(F, Xe, use, SP, sc, proxy)
            C = np.load(od / f'{k}_probe.npy').astype(np.float32)[pix]
            n = e = 0
            for j in range(256):
                cj = C[:, j]
                if (cj > 0).mean() < 0.01:
                    continue
                rc = zs(rankdata(cj[ok]))
                rm, ra = float((rc * zM).mean()), float((rc * zA).mean())
                e += 1
                n += int(rm >= 0.10 and rm > abs(ra))
            res['dyn256'] = {'dynamic': n, 'eligible': e, 'share': n / max(e, 1)}
            res['fve'] = json.loads((OUT / f'{dom}_train_{k}.json').read_text())['eval_fve']
            (OUT / dom).mkdir(parents=True, exist_ok=True)
            f_.write_text(json.dumps(res, default=float))
            out[k] = res
            el = {c: r for c, r in res['hmm']['gate'].items() if r['eligible']}
            log(f'{dom} {k}: PASS_final P(on) {sum(r["PASS_final"] for r in el.values())} / {len(el)}; dyn256 {n}/{e}; FVE {res["fve"]:.4f}')


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=('prep', 'train', 'codes', 'score'))
    p.add_argument('--dom', choices=('mice', 'ants'))
    p.add_argument('--arm', default='S')
    p.add_argument('--seed', type=int, default=1001)
    p.add_argument('--force', action='store_true')
    a = p.parse_args()
    {'prep': cmd_prep, 'train': cmd_train, 'codes': cmd_codes, 'score': cmd_score}[a.step](a)


if __name__ == '__main__':
    main()
