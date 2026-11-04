"""
ECI v2.4: SAEs on average-pooled DINOv3 tokens with weighted motion (pre-registration scripts/eci/v2/v24_prereg.md).

Input of each arm, from the pooled store of v24_pool.py (x = pooled token at t, d = pooled Delta_t = x_{t+2} - x_{t-2}):
  W  x + alpha d            (1024)   arms P2W (P = 2), P4W (P = 4)
  N  x                      (1024)   arm P2N (no motion control)
  C  [x, alpha d]           (2048)   arm P2C (concatenation)
  alpha = the per-domain, per-P scalar of OUT24/<dom>_pool.json (motion energy = appearance energy).
  Arithmetic: alpha d = float32(d) * float32(alpha); x + alpha d in float32; fp16 to the SAE (two separate elementwise
  kernels, no fused multiply-add: identical on CPU and GPU, checked).
SAE recipe = V / Vall (src/eci/sae: Matryoshka BatchTopK 2048 latents, k 16, prefixes 128/256/512/2048, AuxK, Adam lr
  5e-4, 500 warm-up steps, cosine to 10% over the run, TokenNorm = one scalar scale over the whole input (one block,
  also for C, so alpha keeps its meaning), geometric-median b_dec from 100k tokens), batch 4096.
Data stream: every pooled token of every store video. The store rows are cut into segments of SEG rows (per task file);
  per pass the segments are put in a random order (rng WIN_RNG + pass) and grouped in chunks of CW segments; each chunk
  is permuted per seed (own torch generator) and consumed in batches of 4096; the tail is carried into the next chunk;
  the final tail (< 4096) is dropped. Steps = OUT24/<dom>_pool.json (>= 1 pass and >= ~50k steps).
  All arms of one P train in ONE job on ONE stream; a seed's batch is the same token set for every arm (paired).
theta* (calibration) / FVE: two fixed label-free samples of 1.5M pooled tokens each (rng 7 / 8, random segments); they
  are training tokens too (all tokens are training tokens; stated).
Codes (end of the training job, or step `codes`): frame max over the pooled cells of a frame of encode(norm(feature),
  threshold mode); mice = the 864,000 ext frames (v2_3/ext/frames.parquet order); ants = the v2.1 eval rows
  (eci_codes/v2_1/ants/codes/eval_rows.parquet order) -> BEE24/<dom>/codes/<KEY>_max.npy (fp16).
Steps
  train --dom D --P p --arms W,N,C --seeds a,b,c   (GPU)
  codes --dom D --keys K1,K2                       (GPU or CPU) from the saved checkpoints
  featcheck --dom D --P p                          (CPU) loader features == independent recomputation, 50 windows
"""
import argparse
import json
import os
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
for p in ('', 'scripts/eci/v2'):
    sys.path.insert(0, str(REPO / p))
import v24_pool as PL  # noqa: E402

V23, BEE24, OUT24, log = PL.V23, PL.BEE24, PL.OUT24, PL.log
BS, WIN_RNG = 4096, 20261104
SEG, CW = int(os.environ.get('V24_SEG', 8192)), int(os.environ.get('V24_CW', 320))
NCAL = int(os.environ.get('V24_NCAL', 1_500_000))
DIN = {'W': 1024, 'N': 1024, 'C': 2048}


def key_of(P, arm, seed):
    return f'P{P}{arm}_s{seed}'


def parse_key(k):
    a, s = k.split('_s')
    return int(a[1]), a[2], int(s)


def alpha_of(dom, P):
    return float(json.loads((OUT24 / f'{dom}_pool.json').read_text())['P'][str(P)]['alpha'])


def feats_t(x, d, arm, alpha):
    """torch features (x, d fp16 tensors) -> fp16 tensor."""
    import torch
    if arm == 'N':
        return x
    ad = d.float() * torch.tensor(alpha, dtype=torch.float32, device=d.device)
    if arm == 'W':
        return (x.float() + ad).half()
    return torch.cat([x, ad.half()], 1)


def feats_np(x, d, arm, alpha):
    """numpy reference of feats_t."""
    if arm == 'N':
        return x
    ad = d.astype(np.float32) * np.float32(alpha)
    if arm == 'W':
        return (x.astype(np.float32) + ad).astype(np.float16)
    return np.concatenate([x, ad.astype(np.float16)], 1)


def segments(sh):
    return [(t, r0, min(sh.rows[t], r0 + SEG)) for t in sh.tasks for r0 in range(0, sh.rows[t], SEG)]


def sample(sh, n, seed, need_d=True):
    """label-free sample of ~n pooled tokens: whole random segments (rng seed) -> (x, d) fp16 numpy."""
    segs = segments(sh)
    o = np.random.default_rng(seed).permutation(len(segs))
    xs, ds, tot = [], [], 0
    for i in o:
        t, r0, r1 = segs[i]
        xs.append(sh.read(t, 'x', r0, r1))
        ds.append(sh.read(t, 'd', r0, r1) if need_d else None)
        tot += r1 - r0
        if tot >= n:
            break
    x = np.concatenate(xs)[:n]
    return x, (np.concatenate(ds)[:n] if need_d else None)


def gpu_util():
    return V23.gpu_util()


# ---------------------------------------------------------------------------------------------- train
def cmd_train(a):
    import torch
    from concurrent.futures import ThreadPoolExecutor
    from src.eci.sae import MatryoshkaBatchTopKSAE, TokenNorm, _train_step, geometric_median, save_checkpoint
    from src.eci.sae_calibrate import calibrate_threshold, set_threshold, sparsity_stats
    import v20 as V
    dom, P = a.dom, a.P
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seeds = [int(s) for s in a.seeds.split(',')]
    arms = a.arms.split(',')
    need_d = any(x != 'N' for x in arms)
    pj = json.loads((OUT24 / f'{dom}_pool.json').read_text())['P'][str(P)]
    NP, alpha, passes, total = pj['pooled_tokens'], pj['alpha'], pj['passes'], pj['steps']
    steps = int(a.max_steps) if a.max_steps else total
    sh = PL.Shards(dom, P)
    assert sum(sh.rows.values()) == NP
    segs = segments(sh)
    chunks = []
    for ps in range(passes):
        o = np.random.default_rng(WIN_RNG + ps).permutation(len(segs))
        chunks += [[segs[i] for i in o[c0:c0 + CW]] for c0 in range(0, len(o), CW)]
    spec = V.R.spec()['sae']
    lr, warmup = spec['lr'], 500
    nthr = int(os.environ.get('SLURM_CPUS_PER_TASK', 16))
    ex = ThreadPoolExecutor(max(4, nthr - 2))
    log(f'{dom} P{P}: {NP:,} pooled tokens, alpha {alpha:.6f}, {passes} passes, {len(segs):,} segments/pass, {len(chunks)} chunks, '
        f'steps {steps:,} (schedule {total:,}), arms {arms}, seeds {seeds}, {nthr} CPUs, {dev}')

    def load(seg, keep=False):
        t, r0, r1 = seg
        x = sh.read(t, 'x', r0, r1)
        d = sh.read(t, 'd', r0, r1) if need_d else None
        xg = torch.from_numpy(x).to(dev)
        dg = torch.from_numpy(d).to(dev) if need_d else None
        return xg, dg, x.nbytes * (2 if need_d else 1), ((x, d) if keep else None)
    fcheck = {'segments': 0, 'tokens': 0, 'max_abs_diff': 0.0}

    def gather(fs):
        wt, xs, ds, nb = 0.0, [], [], 0
        for f in fs:
            t0_ = time.time()
            xg, dg, nb_, keep = f.result()
            wt += time.time() - t0_
            nb += nb_
            xs.append(xg)
            ds.append(dg)
            if keep is not None:  # identity: GPU features == numpy features, every arm
                for arm in arms:
                    g = feats_t(xg, dg, arm, alpha).cpu().numpy()
                    r = feats_np(keep[0], keep[1], arm, alpha)
                    fcheck['max_abs_diff'] = max(fcheck['max_abs_diff'], float(np.abs(g.astype(np.float32) - r.astype(np.float32)).max()))
                    if not np.array_equal(g.view(np.uint16), r.view(np.uint16)):
                        raise SystemExit(f'GPU FEATURE IDENTITY FAILED {arm} {fcheck}')
                fcheck['segments'] += 1
                fcheck['tokens'] += len(keep[0])
        return torch.cat(xs), (torch.cat(ds) if need_d else None), nb, wt
    pending = [ex.submit(load, s, i < 16) for i, s in enumerate(chunks[0])]
    Xn, Dn, nb, wait = gather(pending)
    log(f'GPU feature identity vs numpy (bit-exact, all arms): {fcheck}')
    pending = [ex.submit(load, s) for s in chunks[1]] if len(chunks) > 1 else []
    torch.backends.cuda.matmul.allow_tf32 = True
    M = {}
    for arm in arms:
        for s in seeds:
            torch.manual_seed(s)
            g = np.random.default_rng(s)
            ii = torch.from_numpy(np.sort(g.choice(len(Xn), min(250_000, len(Xn)), replace=False))).to(dev)
            init = feats_t(Xn[ii], Dn[ii] if need_d else None, arm, alpha).float()
            norm = TokenNorm(DIN[arm]).to(dev).fit_blocks(init, [DIN[arm]], seed=s)
            sae = MatryoshkaBatchTopKSAE(d_in=DIN[arm], n_latents=spec['W'], prefixes=tuple(spec['prefixes']), k=spec['k'], seed=s).to(dev)
            with torch.no_grad():
                sae.b_dec.copy_(geometric_median(norm(init[:100_000])))
            del init
            M[(arm, s)] = {'sae': sae, 'norm': norm, 'opt': torch.optim.Adam(sae.parameters(), lr=lr, betas=(0.9, 0.999)), 'hist': []}
    gens = {s: torch.Generator(device=dev).manual_seed(s) for s in seeds}
    step, ci, cx, cd, used = 0, 0, None, None, 0
    t0 = time.time()
    io = {'wait_s': wait, 'bytes': nb, 'tokens': len(Xn)}
    tlog, slog, util = time.time(), 0, []
    k0 = (arms[0], seeds[0])
    while step < steps:
        Xc = Xn if cx is None else torch.cat([cx, Xn])
        Dc = (Dn if cd is None else torch.cat([cd, Dn])) if need_d else None
        del Xn, Dn
        n = (len(Xc) // BS) * BS
        last = ci == len(chunks) - 1
        cx = None if last else Xc[n:].clone()
        cd = None if (last or not need_d) else Dc[n:].clone()
        Xg, Dg = Xc[:n], (Dc[:n] if need_d else None)
        del Xc, Dc
        perms = {s: torch.randperm(n, generator=gens[s], device=dev) for s in seeds}
        for b in range(n // BS):
            if step >= steps:
                break
            for s in seeds:
                ib = perms[s][b * BS:(b + 1) * BS]
                xb, db = Xg[ib], (Dg[ib] if need_d else None)
                for arm in arms:
                    m = M[(arm, s)]
                    logs, _ = _train_step(m['sae'], m['opt'], m['norm'](feats_t(xb, db, arm, alpha)), step, total, lr, warmup, 1.0)
                    if step % 5000 == 0 or step == steps - 1:
                        logs.update(step=step, elapsed_s=round(time.time() - t0, 1))
                        m['hist'].append(logs)
            used += BS
            step += 1
            if step % 2000 == 0:
                u, mem = gpu_util()
                util.append(u)
                el = time.time() - tlog
                log(f'step {step:,}/{steps:,} chunk {ci + 1}/{len(chunks)}: {(step - slog) / el:.1f} steps/s '
                    f'({(step - slog) * BS / el:,.0f} tokens/s per SAE, {len(M)} SAEs), GPU util {u}% mem {mem} MiB, '
                    f'loader wait so far {io["wait_s"]:.0f}s of {time.time() - t0:.0f}s, read {io["bytes"] / (time.time() - t0) / 1e9:.2f} GB/s, '
                    f'fve {k0} {M[k0]["hist"][-1]["fve"] if M[k0]["hist"] else float("nan"):.4f}, ETA {(steps - step) / ((step - slog) / el) / 60:.0f} min')
                tlog, slog = time.time(), step
        del Xg, Dg, perms
        if step >= steps or last:
            break
        ci += 1
        Xn, Dn, nb, wait = gather(pending)
        io['wait_s'] += wait
        io['bytes'] += nb
        io['tokens'] += len(Xn)
        pending = [ex.submit(load, s_) for s_ in chunks[ci + 1]] if ci + 1 < len(chunks) else []
    tt = time.time() - t0
    for f in pending:
        f.cancel()
    ex.shutdown(wait=False, cancel_futures=True)
    rec0 = {'node': socket.gethostname(), 'gpu': torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu', 'steps_run': step,
            'steps_schedule': total, 'tokens_used_per_sae': used, 'pooled_tokens_total': NP, 'passes': used / NP,
            'passes_planned': passes, 'train_time_s': tt, 'steps_per_s': step / tt, 'n_saes_in_job': len(M),
            'loader_wait_s': io['wait_s'], 'store_bytes_read': io['bytes'], 'store_GB_per_s': io['bytes'] / tt / 1e9,
            'gpu_util_samples': util, 'gpu_util_median': float(np.median(util)) if util else None, 'chunks_consumed': ci + 1,
            'chunks_total': len(chunks), 'gpu_feature_identity': fcheck, 'tokens_loaded': io['tokens'], 'alpha': alpha,
            'segment_rows': SEG, 'chunk_segments': CW, 'window_rng': WIN_RNG}
    log(f'train loop done: {json.dumps(rec0)}')
    (OUT24 / 'train').mkdir(parents=True, exist_ok=True)
    if a.max_steps:
        (OUT24 / 'train' / f'{dom}_P{P}_bench_{socket.gethostname()}.json').write_text(json.dumps(rec0, indent=1))
        if not a.save_bench:
            return
    xc, dc = sample(sh, NCAL, 7, need_d)
    xe, de = sample(sh, NCAL, 8, need_d)
    saved = {}
    for (arm, s), m in M.items():
        sae, norm = m['sae'], m['norm']
        sae.eval()
        torch.backends.cuda.matmul.allow_tf32 = False
        Xc_ = torch.from_numpy(feats_np(xc, dc, arm, alpha))
        Xe_ = torch.from_numpy(feats_np(xe, de, arm, alpha))
        th, _ = calibrate_threshold(sae, Xc_, norm=norm)
        set_threshold(sae, th)
        sst = sparsity_stats(sae, Xe_, th, norm=norm)
        fired = torch.zeros(sae.n_latents, dtype=torch.long, device=dev)
        with torch.no_grad():
            for b0 in range(0, len(Xe_), 65536):
                fired += (sae.encode(norm(Xe_[b0:b0 + 65536].to(dev).float()), mode='threshold') > 0).sum(0)
        key = key_of(P, arm, s)
        cfg = {**spec, 'steps': total, 'arm': f'P{P}{arm}', 'pool': P, 'input': arm, 'd_in': DIN[arm], 'alpha': alpha,
               'lag_frames': PL.LAG, 'warmup': warmup, 'tokennorm_blocks': [DIN[arm]]}
        rec = {'cfg': cfg, 'seed': s, 'key': key, **rec0, 'n_calib_tokens': len(Xc_), 'n_eval_tokens': len(Xe_), 'theta_star': float(th),
               'eval_fve': float(sst['fve']), 'eval_l0': float(sst['l0']), 'dead_at_end_train': int((sae.tokens_since_fired >= sae.dead_tokens).sum()),
               'dead_frac_at_end_train': float((sae.tokens_since_fired >= sae.dead_tokens).float().mean()),
               'never_fire_on_eval_frac': float((fired == 0).float().mean()), 'history': m['hist'],
               'calib_eval_note': 'calib / eval samples are pooled training tokens (every token is a training token)'}
        out = BEE24 / dom / 'sae' / key
        out.mkdir(parents=True, exist_ok=True)
        save_checkpoint(out / 'sae_theta.pt', sae, norm, extra={'config': 'eci_v2_4', 'cfg': cfg, 'seed': s, 'theta_star': float(th)})
        (out / 'train.json').write_text(json.dumps(rec, indent=1, default=float))
        (OUT24 / 'train' / f'{dom}_{key}.json').write_text(json.dumps(rec, indent=1, default=float))
        log(f'{dom} {key}: theta* {th:.4f} eval FVE {sst["fve"]:.5f} L0 {sst["l0"]:.2f} dead(train) {rec["dead_at_end_train"]} '
            f'never-fire(eval) {rec["never_fire_on_eval_frac"]:.4f}')
        saved[key] = (sae, norm)
    del xc, dc, xe, de
    if not a.no_codes:
        codes(dom, saved, dev, alpha_of_P={P: alpha})


# ---------------------------------------------------------------------------------------------- codes
def eval_rows(dom):
    """-> DataFrame (store vid index, k) in the scoring order + the row table itself."""
    vids = PL.videos(dom)
    vix = {(s, o): i for i, (s, o, _) in enumerate(vids)}
    if dom == 'mice':
        R = pd.read_parquet(Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_3/ext/frames.parquet'), columns=['obs', 'k'])
        R['series'] = 'v1'
    else:
        R = pd.read_parquet(V23.BEE21 / 'ants' / 'codes' / 'eval_rows.parquet')
    R['vid'] = [vix[(s, str(o))] for s, o in zip(R.series.values, R.obs.values)]
    return R


def codes(dom, saved, dev, alpha_of_P):
    """saved {key: (sae, norm)} -> BEE24/<dom>/codes/<key>_max.npy (frame max over pooled cells), scoring row order."""
    import torch
    t0 = time.time()
    R = eval_rows(dom)
    byP = {}
    for k in saved:
        byP.setdefault(parse_key(k)[0], []).append(k)
    od = BEE24 / dom / 'codes'
    od.mkdir(parents=True, exist_ok=True)
    meta = {}
    for P, keys in byP.items():
        sh = PL.Shards(dom, P)
        alpha = alpha_of_P[P]
        need_d = any(parse_key(k)[1] != 'N' for k in keys)
        out = {k: np.zeros((len(R), saved[k][0].n_latents), np.float16) for k in keys}
        ncell, nfr_empty = 0, 0
        for vi, g in R.groupby('vid', sort=False):
            t, r0, r1 = sh.vrow[vi]
            ix = sh.index(t)
            kk = ix['k'][r0:r1]
            want = g.k.values.astype(np.int64)
            sel = np.flatnonzero(np.isin(kk, want))
            if not len(sel):
                nfr_empty += len(g)
                continue
            a0, a1 = int(sel[0]), int(sel[-1]) + 1
            x = sh.read(t, 'x', r0 + a0, r0 + a1)
            d = sh.read(t, 'd', r0 + a0, r0 + a1) if need_d else None
            m = np.isin(kk[a0:a1], want)
            x, d = x[m], (d[m] if need_d else None)
            fk = kk[a0:a1][m]
            frow = pd.Series(np.arange(len(g)), index=want).reindex(fk).values
            assert not np.isnan(frow).any()
            ncell += len(x)
            nfr_empty += len(g) - len(np.unique(fk))
            fl = torch.from_numpy(frow.astype(np.int64)).to(dev)
            xt = torch.from_numpy(x).to(dev)
            dt = torch.from_numpy(d).to(dev) if need_d else None
            rows = g.index.values
            for k in keys:
                sae, norm = saved[k]
                arm = parse_key(k)[1]
                with torch.no_grad():
                    mx = torch.zeros((len(g), sae.n_latents), device=dev)
                    for b0 in range(0, len(x), 65536):
                        f = feats_t(xt[b0:b0 + 65536], dt[b0:b0 + 65536] if need_d else None, arm, alpha)
                        cc = sae.encode(norm(f.float()), mode='threshold').float()
                        mx.index_reduce_(0, fl[b0:b0 + 65536], cc, 'amax', include_self=True)
                out[k][rows] = mx.half().cpu().numpy()
        for k in keys:
            np.save(od / f'{k}_max.npy', out[k])
        meta[f'P{P}'] = {'keys': keys, 'frames': len(R), 'pooled_cells_encoded': ncell, 'cells_per_frame': ncell / len(R),
                         'frames_without_cells': nfr_empty}
        log(f'{dom} codes P{P} {keys}: {meta[f"P{P}"]} ({time.time() - t0:.0f}s)')
    (OUT24 / 'codes').mkdir(parents=True, exist_ok=True)
    (OUT24 / 'codes' / f'{dom}_{"_".join(sorted(saved))[:120]}.json').write_text(json.dumps(meta, indent=1))
    return meta


def cmd_codes(a):
    import torch
    from src.eci.v2 import sae as SA
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    keys = a.keys.split(',')
    saved = {k: SA.load(BEE24 / a.dom / 'sae' / k / 'sae_theta.pt', dev) for k in keys}
    codes(a.dom, saved, dev, {P: alpha_of(a.dom, P) for P in {parse_key(k)[0] for k in keys}})


# ---------------------------------------------------------------------------------------------- featcheck
def cmd_featcheck(a):
    """loader path (Shards.read rows of a window + feats_t on CPU) == independent recomputation from the native store
    (v24_pool.recompute_frame python loop + feats_np), bit-exact, every arm; 40 random windows of 3 frames + 10 windows
    at video edges (first / last frames, one-sided Delta)."""
    import torch
    dom, P = a.dom, a.P
    sh = PL.Shards(dom, P)
    vids = PL.videos(dom)
    alpha = alpha_of(dom, P)
    rng = np.random.default_rng(5)
    win = []
    for vi in rng.choice(len(vids), 40, replace=True):
        nf = vids[vi][2]
        i0 = int(rng.integers(0, nf - 3))
        win.append((int(vi), i0, i0 + 3))
    for j, vi in enumerate(rng.choice(len(vids), 10, replace=False)):
        nf = vids[vi][2]
        win.append((int(vi), 0, 3) if j % 2 == 0 else (int(vi), nf - 3, nf))
    res = {'P': P, 'alpha': alpha, 'windows': 0, 'video_edge_windows': 10, 'cells': 0, 'max_abs_diff': 0.0, 'exact': True}
    for vi, i0, i1 in win:
        ser, obs, nf = vids[vi]
        t, r0, r1 = sh.vrow[vi]
        ix = sh.index(t)
        fi = ix['fi'][r0:r1]
        m = np.flatnonzero((fi >= i0) & (fi < i1))
        if not len(m):
            continue
        x, d = sh.read(t, 'x', r0 + int(m[0]), r0 + int(m[-1]) + 1), sh.read(t, 'd', r0 + int(m[0]), r0 + int(m[-1]) + 1)
        assert (np.diff(m) == 1).all()
        for i in range(i0, i1):
            ref, _, _ = PL.recompute_frame(dom, ser, obs, i, P)
            q = np.flatnonzero(fi[m] == i)
            pos = ix['pos'][r0 + m[q]].astype(int)
            assert sorted(pos.tolist()) == sorted(ref), 'cell set differs'
            rx = np.stack([ref[p][0] for p in pos])
            rd = np.stack([ref[p][1] for p in pos])
            for arm in ('W', 'N', 'C'):
                g = feats_t(torch.from_numpy(x[q]), torch.from_numpy(d[q]), arm, alpha).numpy()
                r = feats_np(rx, rd, arm, alpha)
                res['max_abs_diff'] = max(res['max_abs_diff'], float(np.abs(g.astype(np.float32) - r.astype(np.float32)).max()))
                res['exact'] &= bool(np.array_equal(g.view(np.uint16), r.view(np.uint16)))
            res['cells'] += len(q)
        res['windows'] += 1
    (OUT24 / f'{dom}_P{P}_featcheck.json').write_text(json.dumps(res, indent=1))
    log(f'featcheck {dom} P{P}: {res}')
    if not res['exact']:
        raise SystemExit('FEATURE IDENTITY FAILED')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=('train', 'codes', 'featcheck'))
    p.add_argument('--dom', required=True, choices=('mice', 'ants'))
    p.add_argument('--P', type=int, default=2)
    p.add_argument('--arms', default='W')
    p.add_argument('--seeds', default='2401,2402,2403')
    p.add_argument('--keys', default='')
    p.add_argument('--max_steps', type=int, default=0)
    p.add_argument('--save_bench', action='store_true')
    p.add_argument('--no_codes', action='store_true')
    a = p.parse_args()
    {'train': cmd_train, 'codes': cmd_codes, 'featcheck': cmd_featcheck}[a.step](a)
