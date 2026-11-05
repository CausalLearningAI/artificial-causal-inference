"""
ECI v2.5: native-resolution SAEs with a proper temporal-context test, video edges excluded (pre-registration
scripts/eci/v2/v25_prereg.md).

Tokens: the v2.1 store (eci_tokens/v2_full/<dom>/v2.1_m5): DINOv3-L/16 block 20, native grid, 5 fps; frame t holds
  U_t = union of M_s (mask v5, flag bit0) for s in t-2..t+2, so every M_t position is stored at t-2..t+2.
Lag L = 2 frames (+-0.4 s). EDGE RULE: a frame t is used (training AND scoring) iff L <= t < T - L (T = frames of the
  store video); every used token therefore has its full temporal context and no one-sided difference is ever formed.
x_t = the M_t token; Delta_t = fp16(float32(x_{t+L}) - float32(x_{t-L})) at the same grid position.
Arms (input to the SAE, float32 arithmetic, fp16 to the SAE; v24_train.feats_t / feats_np):
  N0  x_t                          (1024)  static control
  W1  x_t + alpha Delta_t          (1024)  equal-energy weighted sum
  Wh  x_t + (alpha / 2) Delta_t    (1024)  lighter sum
  C1  [x_t, alpha Delta_t]         (2048)  concatenation
  alpha = sqrt(E||x - mean x||^2 / E||Delta||^2) per domain on a fixed label-free sample (step count), OUT25/<dom>_count.json.
SAE recipe = V / Vall / v2.4 (Matryoshka BatchTopK 2048 latents, k 16, prefixes 128/256/512/2048, AuxK, Adam lr 5e-4,
  500 warm-up steps, cosine to 10%, TokenNorm one scalar over the whole input, geometric-median b_dec), batch 4096.
Stream: every used M_t token of every store video (mice 648, ants 232); videos cut into windows of WF centre frames inside
  [L, T - L); per pass the windows in a random order (rng WIN_RNG + pass), chunks of CW windows, each chunk permuted per
  seed; tail carried; final tail dropped. One job per domain trains ALL arms x seeds from ONE store read; within a seed
  every arm sees the same token batch (paired). passes = max(1, ceil(50,000 x 4096 / N)), steps = floor(passes N / 4096).
Steps
  count --dom D      (CPU) per-video token counts with / without the edge rule, alpha, passes, steps -> OUT25/<dom>_count.json
  featcheck --dom D  (CPU) loader features == an independent per-token recomputation (bit-exact), 50 windows incl. 10 at
                     the edge boundary; window coverage == exactly the frames [L, T - L) of every video
  train --dom D      (GPU) -> BEE25/<dom>/sae/<arm>_s<seed>/, OUT25/train/
  codes --dom D --task i --ntask n   (CPU) frame max over the M_t tokens of every non-edge scored frame -> parts
  merge --dom D      (CPU) parts -> BEE25/<dom>/codes/<key>_max.npy (row order = BEE25/<dom>/codes/rows.parquet)
"""
import argparse
import json
import math
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
for p in ('', 'scripts/eci/v2'):
    sys.path.insert(0, str(REPO / p))
import v21_train as T  # noqa: E402
import v23_train as V23  # noqa: E402
import v24_train as T24  # noqa: E402  (feats_t / feats_np only)

STORE, VER = T.STORE, T.VER
BEE25 = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_5')
OUT25 = REPO / 'results/vision/eci_v2_5'
L = 2
WF, BS = 100, 4096
CW = int(os.environ.get('V25_CW', 256))
WIN_RNG = 20261105
NCAL = 1_500_000
ARMS = {'N0': ('N', 0.0), 'W1': ('W', 1.0), 'Wh': ('W', 0.5), 'C1': ('C', 1.0)}
DIN = {'N0': 1024, 'W1': 1024, 'Wh': 1024, 'C1': 2048}
SEEDS = (2501, 2502, 2503)
log = T.log


def key_of(arm, seed):
    return f'{arm}_s{seed}'


def arm_alpha(arm, alpha):
    return float(alpha) * ARMS[arm][1]


def feats_t(x, d, arm, alpha):
    return T24.feats_t(x, d, ARMS[arm][0], arm_alpha(arm, alpha))


def feats_np(x, d, arm, alpha):
    return T24.feats_np(x, d, ARMS[arm][0], arm_alpha(arm, alpha))


def videos(dom):
    return V23.store_videos(dom)  # [(series, obs, T)], asserted = the training-set rule


def windows(dom):
    out = []
    for ser, obs, nf in videos(dom):
        out += [(ser, obs, i0, min(nf - L, i0 + WF), nf) for i0 in range(L, nf - L, WF)]
    return out


def centre_index(W, nf):
    """centre rows c (M_t tokens of frames i0..i1-1, all with L <= t < T - L) and the rows of the same position at t - L
    and t + L (must exist: fail loud)."""
    fi, pos = W['fi'], W['pos']
    if not (W['i0'] >= L and W['i1'] <= nf - L):
        raise SystemExit(f'EDGE FRAME IN WINDOW {W["i0"]} {W["i1"]} T={nf}')
    key = fi.astype(np.int64) * 65536 + pos
    assert (np.diff(key) > 0).all(), 'store rows not sorted by (frame, position)'
    c = np.flatnonzero(((W['flag'] & 1) > 0) & (fi >= W['i0']) & (fi < W['i1']))
    out = {'c': c}
    for s in (-L, L):
        q = key[c] + s * 65536
        j = np.minimum(np.searchsorted(key, q), len(key) - 1)
        if not (key[j] == q).all():
            raise SystemExit(f'MISSING NEIGHBOUR at lag {s}: {(key[j] != q).sum()} tokens')
        out[s] = j
    return out


def xd_np(tok, I):
    return tok[I['c']], (tok[I[L]].astype(np.float32) - tok[I[-L]].astype(np.float32)).astype(np.float16)


def xd_torch(tok, I, dev):
    import torch
    t = torch.from_numpy(tok).to(dev)
    c, jp, jm = (torch.from_numpy(I[k]).to(dev) for k in ('c', L, -L))
    return t[c], (t[jp].float() - t[jm].float()).half()


class Store(V23.Store):
    def load_xd(self, w, dev=None, keep=False):
        """window -> (x, d) (torch on dev, or numpy if dev is None), bytes read, centre frame index, raw W if keep."""
        ser, obs, i0, i1, nf = w
        W = self.video(ser, obs).window(i0, i1)
        I = centre_index(W, nf)
        if dev is None:
            x, d = xd_np(W['tok'], I)
        else:
            x, d = xd_torch(W['tok'], I, dev)
        return x, d, int(W['tok'].nbytes), W['fi'][I['c']], ((W, I) if keep else None)


# ---------------------------------------------------------------------------------------------- count / alpha
def cmd_count(a):
    from concurrent.futures import ThreadPoolExecutor
    dom = a.dom
    vids = videos(dom)

    def one(v):
        z = np.load(STORE / dom / VER / 'videos' / v[0] / v[1] / 'index.npz')
        ptr, fl = z['ptr'], z['flag']
        assert len(z['k']) == v[2]
        m = (fl & 1) > 0
        allM = int(m.sum())
        r0, r1 = int(ptr[L]), int(ptr[v[2] - L])
        used = int(m[r0:r1].sum())
        return {'series': v[0], 'obs': v[1], 'frames': v[2], 'frames_used': v[2] - 2 * L, 'M_all': allM, 'M_used': used}
    with ThreadPoolExecutor(16) as ex:
        rows = list(ex.map(one, vids))
    tot = {k: int(sum(r[k] for r in rows)) for k in ('frames', 'frames_used', 'M_all', 'M_used')}
    ws = windows(dom)
    cover = sum(w[3] - w[2] for w in ws)
    assert cover == tot['frames_used'], (cover, tot)
    # alpha on a fixed label-free sample of whole random windows (rng 6), >= 2M tokens
    st = Store(dom)
    rng = np.random.default_rng(6)
    xs, ds, n = [], [], 0
    for i in rng.permutation(len(ws)):
        x, d, _, _, _ = st.load_xd(ws[i])
        xs.append(x)
        ds.append(d)
        n += len(x)
        if n >= 2_000_000:
            break
    X = np.concatenate(xs).astype(np.float64)
    D = np.concatenate(ds).astype(np.float64)
    ex_ = float(((X - X.mean(0)) ** 2).sum(1).mean())
    ed = float((D ** 2).sum(1).mean())
    alpha = math.sqrt(ex_ / ed)
    N = tot['M_used']
    passes = max(1, math.ceil(50_000 * BS / N))
    steps = (passes * N) // BS
    res = {'dom': dom, 'lag_frames': L, 'edge_rule': f'frames {L} <= t < T - {L} of every store video', 'videos': len(rows), **tot,
           'M_removed_by_edge_rule': tot['M_all'] - tot['M_used'], 'windows': len(ws), 'alpha': alpha,
           'E_centred_x_sq': ex_, 'E_delta_sq': ed, 'alpha_sample_tokens': int(len(X)), 'alpha_sample_windows_rng': 6,
           'passes': passes, 'steps': steps, 'batch': BS, 'node': socket.gethostname(), 'videos_list': rows}
    OUT25.mkdir(parents=True, exist_ok=True)
    (OUT25 / f'{dom}_count.json').write_text(json.dumps(res, indent=1))
    log(f'count {dom}: {json.dumps({k: v for k, v in res.items() if k != "videos_list"})}')


# ---------------------------------------------------------------------------------------------- featcheck
def cmd_featcheck(a):
    """loader path (Store.load_xd on CPU torch + feats_t) == independent per-token recomputation (T.Video memmap reader,
    python dict of (frame, position) -> row, numpy float32), bit-exact, every arm; 40 random windows of 3 frames + 10 at
    the edge boundary (first used frames L.. / last used frames ..T-L-1)."""
    import torch
    dom = a.dom
    alpha = json.loads((OUT25 / f'{dom}_count.json').read_text())['alpha']
    vids = videos(dom)
    rng = np.random.default_rng(5)
    win = []
    for vi in rng.choice(len(vids), 40, replace=True):
        nf = vids[vi][2]
        i0 = int(rng.integers(L, nf - L - 3))
        win.append((int(vi), i0, i0 + 3))
    for j, vi in enumerate(rng.choice(len(vids), 10, replace=False)):
        nf = vids[vi][2]
        win.append((int(vi), L, L + 3) if j % 2 == 0 else (int(vi), nf - L - 3, nf - L))
    st = Store(dom)
    res = {'alpha': alpha, 'windows': 0, 'edge_boundary_windows': 10, 'tokens': 0, 'max_abs_diff': 0.0, 'exact': True}
    for vi, i0, i1 in win:
        ser, obs, nf = vids[vi]
        x, d, _, fi_c, (W, I) = st.load_xd((ser, obs, i0, i1, nf), dev=torch.device('cpu'), keep=True)
        Vd = T.Video(dom, ser, obs)
        W0 = Vd.window(i0, i1)
        rows = {(int(f), int(p)): r for r, (f, p) in enumerate(zip(W0['fi'], W0['pos']))}
        rx, rd = [], []
        for r in range(len(W0['fi'])):
            f, p = int(W0['fi'][r]), int(W0['pos'][r])
            if not (W0['flag'][r] & 1) or f < i0 or f >= i1:
                continue
            assert L <= f < nf - L
            rx.append(W0['tok'][r])
            rd.append((W0['tok'][rows[(f + L, p)]].astype(np.float32) - W0['tok'][rows[(f - L, p)]].astype(np.float32)).astype(np.float16))
        rx, rd = np.stack(rx), np.stack(rd)
        assert len(rx) == len(x)
        for arm in ARMS:
            g = feats_t(x, d, arm, alpha).numpy()
            r_ = feats_np(rx, rd, arm, alpha)
            res['max_abs_diff'] = max(res['max_abs_diff'], float(np.abs(g.astype(np.float32) - r_.astype(np.float32)).max()))
            res['exact'] &= bool(np.array_equal(g.view(np.uint16), r_.view(np.uint16)))
        res['windows'] += 1
        res['tokens'] += len(rx)
    # window coverage: exactly the frames [L, T - L) of every video, no edge frame
    ws = windows(dom)
    cov = {}
    for ser, obs, i0, i1, nf in ws:
        cov.setdefault((ser, obs), []).append((i0, i1, nf))
    ok = all(sorted(v)[0][0] == L and sorted(v)[-1][1] == v[0][2] - L and
             all(b[0] == a_[1] for a_, b in zip(sorted(v), sorted(v)[1:])) for v in cov.values())
    res['coverage_exact'] = bool(ok and len(cov) == len(vids))
    OUT25.mkdir(parents=True, exist_ok=True)
    (OUT25 / f'{dom}_featcheck.json').write_text(json.dumps(res, indent=1))
    log(f'featcheck {dom}: {res}')
    if not (res['exact'] and res['coverage_exact']):
        raise SystemExit('FEATURE IDENTITY / COVERAGE FAILED')


# ---------------------------------------------------------------------------------------------- train
def sample(st, ws, n, seed):
    """label-free sample of ~n used tokens: whole random windows (rng seed) -> (x, d) numpy fp16."""
    xs, ds, tot = [], [], 0
    for i in np.random.default_rng(seed).permutation(len(ws)):
        x, d, _, _, _ = st.load_xd(ws[i])
        xs.append(x)
        ds.append(d)
        tot += len(x)
        if tot >= n:
            break
    return np.concatenate(xs)[:n], np.concatenate(ds)[:n]


def cmd_train(a):
    import torch
    from concurrent.futures import ThreadPoolExecutor
    from src.eci.sae import MatryoshkaBatchTopKSAE, TokenNorm, _train_step, geometric_median, save_checkpoint
    from src.eci.sae_calibrate import calibrate_threshold, set_threshold, sparsity_stats
    import v20 as V
    dom = a.dom
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seeds = [int(s) for s in a.seeds.split(',')]
    arms = a.arms.split(',')
    cj = json.loads((OUT25 / f'{dom}_count.json').read_text())
    N, alpha, passes, total = cj['M_used'], cj['alpha'], cj['passes'], cj['steps']
    steps = int(a.max_steps) if a.max_steps else total
    ws = windows(dom)
    chunks = []
    for ps in range(passes):
        o = np.random.default_rng(WIN_RNG + ps).permutation(len(ws))
        chunks += [[ws[i] for i in o[c0:c0 + CW]] for c0 in range(0, len(o), CW)]
    spec = V.R.spec()['sae']
    lr, warmup = spec['lr'], 500
    nthr = int(os.environ.get('SLURM_CPUS_PER_TASK', 16))
    st = Store(dom)
    ex = ThreadPoolExecutor(max(4, nthr - 2))
    log(f'{dom}: {N:,} used M_t tokens, alpha {alpha:.6f}, {passes} passes, {len(ws):,} windows/pass, {len(chunks)} chunks, '
        f'steps {steps:,} (schedule {total:,}), arms {arms}, seeds {seeds}, {nthr} CPUs, {dev}')
    fcheck = {'windows': 0, 'tokens': 0, 'max_abs_diff': 0.0, 'min_centre_frame_margin': 10 ** 9}

    def gather(fs):
        wt, xs, ds, nb = 0.0, [], [], 0
        for f in fs:
            t0_ = time.time()
            x, d, nb_, fi_c, keep = f.result()
            wt += time.time() - t0_
            nb += nb_
            if len(x):
                xs.append(x)
                ds.append(d)
            if keep is not None and len(x):  # identity: GPU features == numpy features, every arm
                W, I = keep
                rx, rd = xd_np(W['tok'], I)
                for arm in arms:
                    g = feats_t(x, d, arm, alpha).cpu().numpy()
                    r_ = feats_np(rx, rd, arm, alpha)
                    fcheck['max_abs_diff'] = max(fcheck['max_abs_diff'], float(np.abs(g.astype(np.float32) - r_.astype(np.float32)).max()))
                    if not np.array_equal(g.view(np.uint16), r_.view(np.uint16)):
                        raise SystemExit(f'GPU FEATURE IDENTITY FAILED {arm} {fcheck}')
                fcheck['windows'] += 1
                fcheck['tokens'] += len(rx)
        return torch.cat(xs), torch.cat(ds), nb, wt
    pending = [ex.submit(st.load_xd, w, dev, i < 16) for i, w in enumerate(chunks[0])]
    Xn, Dn, nb, wait = gather(pending)
    log(f'GPU feature identity vs numpy (bit-exact, all arms): {fcheck}')
    pending = [ex.submit(st.load_xd, w, dev) for w in chunks[1]] if len(chunks) > 1 else []
    torch.backends.cuda.matmul.allow_tf32 = True
    M = {}
    for arm in arms:
        for s in seeds:
            torch.manual_seed(s)
            g = np.random.default_rng(s)
            ii = torch.from_numpy(np.sort(g.choice(len(Xn), min(250_000, len(Xn)), replace=False))).to(dev)
            init = feats_t(Xn[ii], Dn[ii], arm, alpha).float()
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
        Dc = Dn if cd is None else torch.cat([cd, Dn])
        del Xn, Dn
        n = (len(Xc) // BS) * BS
        last = ci == len(chunks) - 1
        cx = None if last else Xc[n:].clone()
        cd = None if last else Dc[n:].clone()
        Xg, Dg = Xc[:n], Dc[:n]
        del Xc, Dc
        perms = {s: torch.randperm(n, generator=gens[s], device=dev) for s in seeds}
        for b in range(n // BS):
            if step >= steps:
                break
            for s in seeds:
                ib = perms[s][b * BS:(b + 1) * BS]
                xb, db = Xg[ib], Dg[ib]
                for arm in arms:
                    m = M[(arm, s)]
                    logs, _ = _train_step(m['sae'], m['opt'], m['norm'](feats_t(xb, db, arm, alpha)), step, total, lr, warmup, 1.0)
                    if step % 5000 == 0 or step == steps - 1:
                        logs.update(step=step, elapsed_s=round(time.time() - t0, 1))
                        m['hist'].append(logs)
            used += BS
            step += 1
            if step % 2000 == 0:
                u, mem = V23.gpu_util()
                util.append(u)
                el = time.time() - tlog
                log(f'step {step:,}/{steps:,} chunk {ci + 1}/{len(chunks)}: {(step - slog) / el:.2f} steps/s ({len(M)} SAEs), '
                    f'GPU util {u}% mem {mem} MiB, loader wait so far {io["wait_s"]:.0f}s of {time.time() - t0:.0f}s, '
                    f'read {io["bytes"] / (time.time() - t0) / 1e9:.2f} GB/s, fve {k0} '
                    f'{M[k0]["hist"][-1]["fve"] if M[k0]["hist"] else float("nan"):.4f}, ETA {(steps - step) / ((step - slog) / el) / 60:.0f} min')
                tlog, slog = time.time(), step
        del Xg, Dg, perms
        if step >= steps or last:
            break
        ci += 1
        Xn, Dn, nb, wait = gather(pending)
        io['wait_s'] += wait
        io['bytes'] += nb
        io['tokens'] += len(Xn)
        pending = [ex.submit(st.load_xd, w, dev) for w in chunks[ci + 1]] if ci + 1 < len(chunks) else []
    tt = time.time() - t0
    for f in pending:
        f.cancel()
    ex.shutdown(wait=False, cancel_futures=True)
    rec0 = {'node': socket.gethostname(), 'gpu': torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu', 'steps_run': step,
            'steps_schedule': total, 'tokens_used_per_sae': used, 'M_used_total': N, 'passes': used / N, 'passes_planned': passes,
            'train_time_s': tt, 'steps_per_s': step / tt, 'n_saes_in_job': len(M), 'loader_wait_s': io['wait_s'],
            'store_bytes_read': io['bytes'], 'store_GB_per_s': io['bytes'] / tt / 1e9, 'gpu_util_samples': util,
            'gpu_util_median': float(np.median(util)) if util else None, 'chunks_consumed': ci + 1, 'chunks_total': len(chunks),
            'gpu_feature_identity': fcheck, 'tokens_loaded': io['tokens'], 'alpha': alpha, 'lag_frames': L,
            'window_frames': WF, 'chunk_windows': CW, 'window_rng': WIN_RNG}
    log(f'train loop done: {json.dumps(rec0)}')
    (OUT25 / 'train').mkdir(parents=True, exist_ok=True)
    if a.max_steps and not a.save_bench:
        (OUT25 / 'train' / f'{dom}_bench_{socket.gethostname()}.json').write_text(json.dumps(rec0, indent=1))
        return
    xc, dc = sample(st, ws, NCAL, 7)
    xe, de = sample(st, ws, NCAL, 8)
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
        key = key_of(arm, s)
        cfg = {**spec, 'steps': total, 'arm': arm, 'input': ARMS[arm][0], 'alpha_mult': ARMS[arm][1], 'd_in': DIN[arm],
               'alpha': alpha, 'alpha_eff': arm_alpha(arm, alpha), 'lag_frames': L, 'warmup': warmup, 'tokennorm_blocks': [DIN[arm]],
               'edge_rule': f'{L} <= t < T - {L}'}
        rec = {'cfg': cfg, 'seed': s, 'key': key, **rec0, 'n_calib_tokens': len(Xc_), 'n_eval_tokens': len(Xe_), 'theta_star': float(th),
               'eval_fve': float(sst['fve']), 'eval_l0': float(sst['l0']),
               'dead_at_end_train': int((sae.tokens_since_fired >= sae.dead_tokens).sum()),
               'dead_frac_at_end_train': float((sae.tokens_since_fired >= sae.dead_tokens).float().mean()),
               'never_fire_on_eval_frac': float((fired == 0).float().mean()), 'history': m['hist'],
               'calib_eval_note': 'calib (rng 7) / eval (rng 8) samples are whole random windows of training tokens'}
        out = BEE25 / dom / 'sae' / key
        out.mkdir(parents=True, exist_ok=True)
        save_checkpoint(out / 'sae_theta.pt', sae, norm, extra={'config': 'eci_v2_5', 'cfg': cfg, 'seed': s, 'theta_star': float(th)})
        (out / 'train.json').write_text(json.dumps(rec, indent=1, default=float))
        (OUT25 / 'train' / f'{dom}_{key}.json').write_text(json.dumps(rec, indent=1, default=float))
        log(f'{dom} {key}: theta* {th:.4f} eval FVE {sst["fve"]:.5f} L0 {sst["l0"]:.2f} dead(train) {rec["dead_at_end_train"]} '
            f'never-fire(eval) {rec["never_fire_on_eval_frac"]:.4f}')
    log('ALL CHECKPOINTS SAVED')


# ---------------------------------------------------------------------------------------------- codes
def eval_rows(dom):
    """the scored frames of a domain, edge-excluded -> DataFrame (series, obs, k, fi, T, xrow) in scoring order.
    mice: v2.3 ext frames (864,000; xrow = row there); ants: v2.1 eval_rows (5 fps spans) of the 147 v3 + v4 videos of
    v23_ants.table (xrow = row in eval_rows.parquet)."""
    vids = {(s, o): nf for s, o, nf in videos(dom)}
    if dom == 'mice':
        R = pd.read_parquet(Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_3/ext/frames.parquet'), columns=['obs', 'k'])
        R['series'] = 'v1'
    else:
        import v23_ants as AN
        R = pd.read_parquet(T.BEE / 'ants' / 'codes' / 'eval_rows.parquet')
        A = AN.table()
        keep = set(zip(A.series.values, A.obs.values))
        R = R[[(s, str(o)) in keep for s, o in zip(R.series.values, R.obs.values)]]
        assert len({(s, o) for s, o in zip(R.series.values, R.obs.values)}) == 147
    R = R.reset_index().rename(columns={'index': 'xrow'})
    R['obs'] = R.obs.astype(str)
    R['T'] = [vids[(s, o)] for s, o in zip(R.series.values, R.obs.values)]
    fi = np.empty(len(R), np.int64)
    for (s, o), g in R.groupby(['series', 'obs'], sort=False):
        kk = np.load(STORE / dom / VER / 'videos' / s / o / 'index.npz')['k']
        pos = pd.Series(np.arange(len(kk)), index=kk).reindex(g.k.values).values
        assert not np.isnan(pos).any()
        fi[g.index.values] = pos.astype(np.int64)
    R['fi'] = fi
    R['edge'] = (R.fi.values < L) | (R.fi.values >= R['T'].values - L)
    return R


def cmd_rows(a):
    R = eval_rows(a.dom)
    od = BEE25 / a.dom / 'codes'
    od.mkdir(parents=True, exist_ok=True)
    R.to_parquet(od / 'rows_all.parquet')
    Rk = R[~R.edge.values].reset_index(drop=True)
    Rk.to_parquet(od / 'rows.parquet')
    log(f'rows {a.dom}: {len(R):,} scored frames, {int(R.edge.sum())} edge frames removed -> {len(Rk):,}')


def cmd_codes(a):
    import torch
    from src.eci.v2 import sae as SA
    dom = a.dom
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    dev = torch.device('cpu')
    alpha = json.loads((OUT25 / f'{dom}_count.json').read_text())['alpha']
    keys = a.keys.split(',') if a.keys else [key_of(arm, s) for arm in ARMS for s in SEEDS]
    saved = {k: SA.load(BEE25 / dom / 'sae' / k / 'sae_theta.pt', dev) for k in keys}
    R = pd.read_parquet(BEE25 / dom / 'codes' / 'rows.parquet')
    assert not R.edge.any()
    gv = list(dict.fromkeys(zip(R.series.values, R.obs.values)))[a.task::a.ntask]
    st = Store(dom)
    t0 = time.time()
    rows_all, outs = [], {k: [] for k in keys}
    for ser, obs in gv:
        g = R[(R.series.values == ser) & (R.obs.values == obs)]
        nf = int(g['T'].values[0])
        fis = g.fi.values.astype(np.int64)
        assert (np.diff(fis) > 0).all() and fis.min() >= L and fis.max() < nf - L
        out = {k: np.zeros((len(g), 2048), np.float16) for k in keys}
        lo, hi = int(fis.min()), int(fis.max()) + 1
        for i0 in range(lo, hi, WF):
            i1 = min(hi, i0 + WF)
            want = (fis >= i0) & (fis < i1)
            if not want.any():
                continue
            x, d, _, fi_c, _ = st.load_xd((ser, obs, i0, i1, nf), dev=dev)
            fr = pd.Series(np.flatnonzero(want), index=fis[want]).reindex(fi_c).values  # frame row of every centre token
            mk = ~np.isnan(fr)
            x, d, fr = x[torch.from_numpy(mk)], d[torch.from_numpy(mk)], torch.from_numpy(fr[mk].astype(np.int64))
            sl = np.flatnonzero(want)
            for k in keys:
                sae, norm = saved[k]
                arm = k.split('_s')[0]
                with torch.no_grad():
                    mx = torch.zeros((len(g), sae.n_latents))
                    for b0 in range(0, len(x), 32768):
                        cc = sae.encode(norm(feats_t(x[b0:b0 + 32768], d[b0:b0 + 32768], arm, alpha).float()), mode='threshold')
                        mx.index_reduce_(0, fr[b0:b0 + 32768], cc, 'amax', include_self=True)
                out[k][sl] = mx[sl].half().numpy()
        rows_all.append(g.index.values)
        for k in keys:
            outs[k].append(out[k])
        log(f'codes {dom} {ser}/{obs}: {len(g)} frames ({time.time() - t0:.0f}s)')
    od = BEE25 / dom / 'codes' / 'parts'
    od.mkdir(parents=True, exist_ok=True)
    np.save(od / f'rows_t{a.task}.npy', np.concatenate(rows_all) if rows_all else np.zeros(0, np.int64))
    for k in keys:
        np.save(od / f'{k}_t{a.task}.npy', np.concatenate(outs[k]) if outs[k] else np.zeros((0, 2048), np.float16))
    log(f'codes task {a.task}/{a.ntask} done: {len(gv)} videos ({time.time() - t0:.0f}s)')


def cmd_merge(a):
    dom = a.dom
    R = pd.read_parquet(BEE25 / dom / 'codes' / 'rows.parquet')
    od = BEE25 / dom / 'codes'
    keys = a.keys.split(',') if a.keys else [key_of(arm, s) for arm in ARMS for s in SEEDS]
    rr = [np.load(od / 'parts' / f'rows_t{t}.npy') for t in range(a.ntask)]
    for k in keys:
        out, seen = np.zeros((len(R), 2048), np.float16), np.zeros(len(R), bool)
        for t in range(a.ntask):
            r = rr[t]
            assert not seen[r].any()
            out[r], seen[r] = np.load(od / 'parts' / f'{k}_t{t}.npy'), True
        assert seen.all(), f'{k}: {(~seen).sum()} rows missing'
        np.save(od / f'{k}_max.npy', out)
        log(f'merge {dom} {k}: {len(R):,} rows, frames with any code > 0: {(out.max(1) > 0).mean():.4f}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=('count', 'featcheck', 'train', 'rows', 'codes', 'merge'))
    p.add_argument('--dom', required=True, choices=('mice', 'ants'))
    p.add_argument('--arms', default='N0,W1,Wh,C1')
    p.add_argument('--seeds', default=','.join(map(str, SEEDS)))
    p.add_argument('--keys', default='')
    p.add_argument('--max_steps', type=int, default=0)
    p.add_argument('--save_bench', action='store_true')
    p.add_argument('--task', type=int, default=0)
    p.add_argument('--ntask', type=int, default=1)
    a = p.parse_args()
    {'count': cmd_count, 'featcheck': cmd_featcheck, 'train': cmd_train, 'rows': cmd_rows, 'codes': cmd_codes,
     'merge': cmd_merge}[a.step](a)
