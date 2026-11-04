"""
ECI v2.2 token-level steps: 2x2-pooled training tokens, and eval-frame codes under several readouts for any V-input SAE.

Input of every SAE here = arm V's input x_t + (x_{t+1} - x_{t-1}) / 2 (v21_train.raw_feats edge rule) on the mask-v5
(M_t, flag bit0) tokens of the v2.1 store.
2x2 POOLED token = the mean of the V inputs of the M_t tokens inside one 2x2 block of the patch grid (grid G -> G/2; mice
    64 -> 32, i.e. 32-px effective patches at 1024 px), per frame; blocks with no M_t token do not exist.

Steps
  prep  --dom D        (CPU) the v21_train.cmd_prep window plan (rng 0, identical windows / roles), every window's M_t
                       tokens pooled 2x2 -> BEE22/<dom>/data/<role>_P2.npy (fp16, all available pooled tokens; the
                       native V tokens of the same windows are v2_1/<dom>/data/<role>_{B,C}.npy)
  codes --dom D --models K1,K2,..   (GPU) eval-span codes, readouts per SAE:
        max     frame max over native tokens (identity vs the stored codes for stored V models)
        box3    native per-patch code map, 3x3 box mean (zeros outside the mask, / 9), then frame max
        coarse  frame max over the 2x2-pooled tokens encoded with the same SAE
        ms      max(max, coarse) (multi-scale: native and pooled tokens of the frame together)
     -> BEE22/<dom>/codes/<KEY>_<readout>_eval.npy + eval_rows.parquet (identical to the stored rows, checked)
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
sys.path.insert(0, str(REPO / 'scripts/eci/v2'))
import v21_ladder as L  # noqa: E402

T = L.T
BEE22 = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_2')
BEE23 = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_3')  # v2.3 Vall_* SAEs and their codes
OUT22 = REPO / 'results/vision/eci_v2_2'
log = L.log


def pool2(f, fi, pos, G):
    """V inputs f (n, d) torch of M_t tokens at frame fi (n,) and grid position pos (n,) -> pooled (m, d), their frame
    (m,), pooled grid position (m,) on the (G/2) grid, and the block index of each input token (n,)."""
    import torch
    r, c = pos // G, pos % G
    g2 = (G + 1) // 2
    key = fi.astype(np.int64) * (g2 * g2) + (r // 2) * g2 + (c // 2)
    u, inv = np.unique(key, return_inverse=True)
    inv_t = torch.from_numpy(inv).to(f.device)
    s = torch.zeros((len(u), f.shape[1]), device=f.device, dtype=f.dtype).index_add_(0, inv_t, f)
    n = torch.bincount(inv_t, minlength=len(u)).to(f.dtype)
    return s / n[:, None], (u // (g2 * g2)), (u % (g2 * g2)), inv


# ---------------------------------------------------------------------------------------------- prep
def cmd_prep(a):
    import torch
    dom = a.dom
    dev = torch.device('cpu')
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    rng = np.random.default_rng(0)  # = v21_train.cmd_prep plan
    vids = T.videos_of(dom)
    t0 = time.time()
    m_per_frame = np.mean([((T.Video(dom, v[0], v[1]).flag & 1) > 0).sum() / len(T.Video(dom, v[0], v[1]).k)
                           for v in vids[::max(1, len(vids) // 6)][:6]])
    plan = []
    for r in T.CAP:
        vs = [v for v in vids if v[2] == r]
        n_win = max(1, int(np.ceil(1.1 * T.CAP[r] / m_per_frame / (T.WIN * max(1, len(vs))))))
        for ser, obs, role in vs:
            nf = len(T.Video(dom, ser, obs).k)
            starts = np.sort(rng.choice(max(1, nf - T.WIN), min(n_win, max(1, nf - T.WIN)), replace=False))
            plan += [(ser, obs, role, int(s_)) for s_ in starts]
    log(f'{dom}: {len(plan)} windows (same plan as v21_train prep); M tokens / frame ~{m_per_frame:.0f}')
    data = {r: [] for r in T.CAP}
    nnat = {r: 0 for r in T.CAP}
    cache = {}
    for j, (ser, obs, role, s_) in enumerate(plan):
        if (ser, obs) not in cache:
            cache.clear()
            cache[(ser, obs)] = T.Video(dom, ser, obs)
        V_ = cache[(ser, obs)]
        W = V_.window(s_, min(len(V_.k), s_ + T.WIN))
        c, nb = T.centre_rows(W)
        if not len(c):
            continue
        x, cd, rs, cnt = T.raw_feats(W['tok'], c, nb, dev)
        p, _, _, _ = pool2(x + cd, W['fi'][c], W['pos'][c].astype(np.int64), V_.G)
        data[role].append(p.half())
        nnat[role] += len(c)
        if j % 1000 == 0:
            log(f'{dom} prep {j}/{len(plan)} ({time.time() - t0:.0f}s)')
    od = BEE22 / dom / 'data'
    od.mkdir(parents=True, exist_ok=True)
    info = {'windows': len(plan), 'tokens': {}}
    loc = Path(os.environ.get('LOCAL_DIR', '/tmp'))
    import shutil
    for r in T.CAP:
        X = torch.cat(data[r]).numpy()
        info['tokens'][r] = {'pooled': int(len(X)), 'native_in_windows': int(nnat[r]), 'native_per_pooled': nnat[r] / max(1, len(X))}
        np.save(loc / f'{r}_P2.npy', X)
        shutil.copyfile(loc / f'{r}_P2.npy', od / f'{r}_P2.npy')
        (loc / f'{r}_P2.npy').unlink()
    OUT22.mkdir(parents=True, exist_ok=True)
    (OUT22 / f'{dom}_prep_P2.json').write_text(json.dumps(info, indent=1))
    log(f'{dom} prep done {json.dumps(info)} ({time.time() - t0:.0f}s)')


# ---------------------------------------------------------------------------------------------- codes
def sae_path(dom, key):
    if key.startswith('Vall_'):
        return BEE23 / dom / 'sae' / key / 'sae_theta.pt'
    p = BEE22 / dom / 'sae' / key / 'sae_theta.pt'
    return p if p.exists() else T.BEE / dom / 'sae' / key / 'sae_theta.pt'


RO = None  # --readouts override


def readouts_of(key):
    """pooled-only SAEs (key starts with 'VP2') are read out on pooled tokens only."""
    if key.startswith('VP2'):
        return ('coarse',)
    return RO or ('max', 'box3', 'coarse', 'ms')


def box3(cc, fi, pos, G):
    """per-patch codes cc (n, W) at frame fi / grid pos -> 3x3 box mean (zeros where no token, / 9) at each token."""
    import torch
    key = torch.from_numpy(fi.astype(np.int64) * (G * G) + pos).to(cc.device)
    o = torch.argsort(key)
    ks = key[o]
    r, c = torch.from_numpy(pos // G).to(cc.device), torch.from_numpy(pos % G).to(cc.device)
    acc = torch.zeros_like(cc)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            rr, c2 = r + dy, c + dx
            ok = (rr >= 0) & (rr < G) & (c2 >= 0) & (c2 < G)
            q = key + dy * G + dx
            i = torch.searchsorted(ks, q).clamp(max=len(ks) - 1)
            hit = ok & (ks[i] == q)
            acc[hit] += cc[o[i[hit]]]
    return acc / 9


def cmd_codes(a):
    import torch
    from src.eci.v2 import sae as SA
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    dom = a.dom
    keys = a.models.split(',')
    saes = {k: SA.load(sae_path(dom, k), dev) for k in keys}
    Wd = {k: s[0].n_latents for k, s in saes.items()}
    Fs = T.frame_sets(dom)['eval']
    spans = [(ser, obs, int(g.k.min()), int(g.k.max()) + 1) for (ser, obs), g in Fs.groupby(['series', 'obs'], sort=False)]
    R = pd.DataFrame([(ser, obs, k) for ser, obs, k0, k1 in spans for k in range(k0, k1)], columns=['series', 'obs', 'k'])
    R0 = pd.read_parquet(T.BEE / dom / 'codes' / 'eval_rows.parquet')
    assert len(R0) == len(R) and (R0.values == R.values).all(), 'eval rows differ from the stored rows'
    Xc = {(k, r): np.zeros((len(R), Wd[k]), np.float16) for k in keys for r in readouts_of(k)}
    t0, base, ntok, npool = time.time(), 0, 0, 0
    for si, (ser, obs, k0, k1) in enumerate(spans):
        V_ = T.Video(dom, ser, obs)
        G = V_.G
        kpos = {int(k): i for i, k in enumerate(V_.k)}
        i0, i1 = kpos[k0], kpos[k1 - 1] + 1
        for b0 in range(i0, i1, 600):
            W = V_.window(b0, min(i1, b0 + 600))
            c, nb = T.centre_rows(W, 1)
            if not len(c):
                continue
            x, cd, rs, cnt = T.raw_feats(W['tok'], c, nb, dev)
            f = x + cd
            fi = W['fi'][c] - i0
            pos = W['pos'][c].astype(np.int64)
            fp, pfi, ppos, _ = pool2(f, fi, pos, G)
            ntok += len(c)
            npool += len(fp)
            fl = torch.from_numpy(fi).to(dev)
            pfl = torch.from_numpy(pfi).to(dev)
            nfr = min(i1, b0 + 600) - b0
            sl = slice(base + b0 - i0, base + b0 - i0 + nfr)
            for k, (sae, norm) in saes.items():
                ro = readouts_of(k)
                out = {}
                with torch.no_grad():
                    if 'max' in ro or 'box3' in ro:
                        cc = sae.encode(norm(f), mode='threshold').float()
                        out['max'] = torch.zeros((i1 - i0, Wd[k]), device=dev).index_reduce_(0, fl, cc, 'amax', include_self=True)
                        if 'box3' in ro:
                            out['box3'] = torch.zeros((i1 - i0, Wd[k]), device=dev).index_reduce_(0, fl, box3(cc, fi, pos, G), 'amax', include_self=True)
                        del cc
                    if 'coarse' in ro or 'ms' in ro:
                        cp = sae.encode(norm(fp), mode='threshold').float()
                        out['coarse'] = torch.zeros((i1 - i0, Wd[k]), device=dev).index_reduce_(0, pfl, cp, 'amax', include_self=True)
                    if 'ms' in ro:
                        out['ms'] = torch.maximum(out['max'], out['coarse'])
                    if 'top10' in ro:  # = v21_train.cmd_codes 'top10': mean of the 10 largest token codes of the frame
                        cc = sae.encode(norm(f), mode='threshold').float()
                        lf = torch.from_numpy(fi - (b0 - i0)).to(dev)
                        o_ = torch.argsort(lf, stable=True)
                        lfs = lf[o_]
                        cnt_ = torch.bincount(lfs, minlength=nfr)
                        st_ = torch.cumsum(cnt_, 0) - cnt_
                        rk = torch.arange(len(lfs), device=dev) - st_[lfs]
                        D = torch.zeros((nfr, int(cnt_.max()), Wd[k]), device=dev)
                        D[lfs, rk] = cc[o_]
                        v_ = D.topk(min(10, D.shape[1]), dim=1).values.sum(1) / cnt_.clamp(min=1).float()[:, None].clamp(max=10)
                        out['top10'] = torch.zeros((i1 - i0, Wd[k]), device=dev)
                        out['top10'][b0 - i0: b0 - i0 + nfr] = v_
                        del D, cc
                for r in ro:
                    Xc[(k, r)][sl] = out[r][b0 - i0: b0 - i0 + nfr].half().cpu().numpy()
        base += i1 - i0
        if si % 50 == 0:
            log(f'{dom} codes span {si}/{len(spans)} ({time.time() - t0:.0f}s)')
    od = (BEE23 if keys[0].startswith('Vall_') else BEE22) / dom / 'codes'
    od.mkdir(parents=True, exist_ok=True)
    R.to_parquet(od / 'eval_rows.parquet')
    idc = {'tokens': ntok, 'pooled_tokens': npool, 'native_per_pooled': ntok / max(1, npool)}
    for k in keys:  # identity of the native frame max vs the stored codes of a stored model
        ref = T.BEE / dom / 'codes' / f'{k}_eval.npy'
        if ref.exists() and 'max' in readouts_of(k):
            r_ = np.load(ref, mmap_mode='r')
            mx = 0.0
            for b in range(0, len(R), 50_000):
                mx = max(mx, float(np.abs(Xc[(k, 'max')][b:b + 50_000].astype(np.float32) - np.asarray(r_[b:b + 50_000], np.float32)).max()))
            idc[f'{k}_max_vs_stored_max_abs_diff'] = mx
    for (k, r), x in Xc.items():
        np.save(od / f'{k}_{r}_eval.npy', x)
    import socket
    idc.update(node=socket.gethostname(), gpu=torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu', seconds=time.time() - t0)
    oj = (REPO / 'results/vision/eci_v2_3') if keys[0].startswith('Vall_') else OUT22
    (oj / 'codes').mkdir(parents=True, exist_ok=True)
    (oj / 'codes' / f'{dom}_{a.tag or keys[0]}.json').write_text(json.dumps(idc, indent=1))
    log(f'{dom} codes done {json.dumps(idc)}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=('prep', 'codes'))
    p.add_argument('--dom', choices=('mice', 'ants'))
    p.add_argument('--models', default='')
    p.add_argument('--tag', default='')
    p.add_argument('--readouts', default='')
    a = p.parse_args()
    if a.readouts:
        RO = tuple(a.readouts.split(','))
    {'prep': cmd_prep, 'codes': cmd_codes}[a.step](a)
