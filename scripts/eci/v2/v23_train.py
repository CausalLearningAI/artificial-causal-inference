"""
ECI v2.3: the V SAE retrained on ALL M_t tokens of ALL videos of a domain, about one pass (pre-registration
scripts/eci/v2/v23_prereg.md). Recipe = V (scripts/eci/v2/v21_train.py arm V, src/eci/v2/sae.train) except the data:
  input   x_t + (x_{t+1} - x_{t-1}) / 2 on the mask-v5 tokens (flag bit0) of the v2.1 store, V's edge rule
          (one-sided difference at a video's first / last frame), float32 arithmetic, stored fp16
  data    every M_t token of every store video (mice v1 + v2, 648 videos; ants v2 + v3 + v4, 232 videos), streamed from
          the store (no copy): videos cut into windows of WF frames, windows in one random order (rng WIN_RNG, shared by
          the seeds), grouped into chunks of CW windows; each chunk is permuted per seed (its own torch generator) and
          consumed in batches of 4096; the tail (< 4096 tokens, store order) of a chunk is carried into the next chunk;
          the final tail (< 4096) is dropped. Steps = floor(N_M / 4096) = one pass.
  seeds   several SAEs trained in ONE job on the same stream (one store read for all seeds; per-seed init, TokenNorm,
          geometric-median b_dec and batch composition, as SA.train)
  lr      5e-4, 500 warm-up steps, cosine decay to the new step count (src/eci/sae._lr_at with total = steps)
  theta*, FVE  on the v2.1 calib / eval V token samples (1.5M each, BEE21/<dom>/data/{calib,eval}_{B,C}.npy = the same
          tokens V used); their videos are now ALSO training videos (overlap stated)
Steps
  count --dom D        (CPU) per-video frames / U rows / M_t tokens -> OUT23/<dom>_tokens.json; read throughput bench
  train --dom D --seeds a,b,c [--max_steps n --bench]   (GPU) -> BEE23/<dom>/sae/Vall_s<seed>, OUT23/train/
  featcheck --dom D    (CPU) streamed V features == v22_full.feats_v on sample windows (identity, fail loud)
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

REPO = Path(__file__).resolve().parents[3]
for p in ('', 'scripts/eci/v2'):
    sys.path.insert(0, str(REPO / p))
import v21_train as T  # noqa: E402

STORE, VER = T.STORE, T.VER
BEE21 = T.BEE
BEE23 = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_3')
OUT23 = REPO / 'results/vision/eci_v2_3'
WF, CW, BS = 100, int(os.environ.get('V23_CW', 256)), 4096  # V23_CW: smoke tests only
WIN_RNG = 20261104
log = T.log


ANTS_RULE = {'v2': None, 'v3': (2, 6, 8), 'v4': (2, 6), 'v4_infested': (2, 6)}  # AMENDMENT 1 (user): valid videos, these treatments


def ants_rule_videos():
    """the ants training set by rule (data/ants/<series>/experiment.csv): all valid v2; v3 treatments 2, 6, 8; v4 and
    v4_infested treatments 2, 6 -> {(store series, obs)} (the store keeps v4_infested under series 'v4')."""
    import pandas as pd
    out = set()
    for ser, tr in ANTS_RULE.items():
        e = pd.read_csv(REPO / 'data/ants' / ser / 'experiment.csv')
        e = e[e.valid == 1] if tr is None else e[e.treatment.isin(tr) & (e.valid == 1)]
        out |= {('v4' if ser == 'v4_infested' else ser, str(o)) for o in e.observation_id}
    return out


def store_videos(dom):
    m = json.loads((STORE / dom / VER / 'manifest.json').read_text())
    assert not m['missing_videos'] and m['videos'] == m['expected_videos'] == len(m['videos_list'])
    vids = [(v['series'], v['obs'], int(v['frames'])) for v in m['videos_list']]
    if dom == 'ants':
        rule = ants_rule_videos()
        assert {(s_, o) for s_, o, _ in vids} == rule and len(rule) == 232, 'ants store videos != the training-set rule'
    else:
        assert len(vids) == 648
    return vids


def windows(dom):
    out = []
    for ser, obs, nf in store_videos(dom):
        out += [(ser, obs, i0, min(nf, i0 + WF)) for i0 in range(0, nf, WF)]
    return out


def nb_index(W):
    """centre rows c (M_t tokens of frames i0..i1-1) and the rows of the same position at t-1 / t+1 (ok = exists)."""
    fi, pos = W['fi'], W['pos']
    key = fi.astype(np.int64) * 65536 + pos
    assert (np.diff(key) > 0).all(), 'store rows not sorted by (frame, position)'
    c = np.flatnonzero(((W['flag'] & 1) > 0) & (fi >= W['i0']) & (fi < W['i1']))
    out = {'c': c}
    for d in (-1, 1):
        q = key[c] + d * 65536
        j = np.minimum(np.searchsorted(key, q), len(key) - 1)
        ok = key[j] == q
        out[d] = (np.where(ok, j, 0), ok)
    return out


def feats_gpu(tok, I, dev):
    """= feats_np on the GPU (same float32 elementwise ops, exact): tok (m, 1024) fp16 numpy, I = nb_index -> (n, 1024) fp16."""
    import torch
    t = torch.from_numpy(tok).to(dev, non_blocking=False)
    c = torch.from_numpy(I['c']).to(dev)
    x = t[c].float()
    jm, okm = (torch.from_numpy(v).to(dev) for v in I[-1])
    jp, okp = (torch.from_numpy(v).to(dev) for v in I[1])
    nm, np_ = t[jm].float(), t[jp].float()
    mp, mm = okp[:, None], okm[:, None]
    cd = torch.where(mp & mm, (np_ - nm) / 2, torch.where(mp, np_ - x, torch.where(mm, x - nm, torch.zeros_like(x))))
    return (x + cd).half()


def feats_np(W):
    """V input of the M_t tokens of frames W['i0']..W['i1']-1 (v22_full.feats_v, numpy float32) -> (n, 1024) float32."""
    fi, pos, tok = W['fi'], W['pos'], W['tok']
    key = fi.astype(np.int64) * 65536 + pos
    assert (np.diff(key) > 0).all(), 'store rows not sorted by (frame, position)'
    c = np.flatnonzero(((W['flag'] & 1) > 0) & (fi >= W['i0']) & (fi < W['i1']))
    x = tok[c].astype(np.float32)
    nx, ok = {}, {}
    for d in (-1, 1):
        q = key[c] + d * 65536
        j = np.minimum(np.searchsorted(key, q), len(key) - 1)
        ok[d] = key[j] == q
        nx[d] = tok[np.where(ok[d], j, 0)].astype(np.float32)
    mp, mm = ok[1][:, None], ok[-1][:, None]
    cd = np.where(mp & mm, (nx[1] - nx[-1]) / np.float32(2), np.where(mp, nx[1] - x, np.where(mm, x - nx[-1], np.float32(0))))
    return x + cd


class Vid:
    """T.Video with the index kept in its stored dtypes (all videos of a domain stay cached; T.Video holds idx as int64)."""

    def __init__(self, dom, ser, obs):
        d = STORE / dom / VER / 'videos' / ser / obs
        z = np.load(d / 'index.npz')
        self.k, self.ptr, self.idx, self.flag = z['k'], z['ptr'], z['idx'], z['flag']
        self.tok = np.memmap(d / 'tok.f16', np.float16, 'r', shape=(int(self.ptr[-1]), 1024))

    def window(self, i0, i1):
        """= T.Video.window (same rows, same dict)."""
        a0, a1 = max(0, i0 - 2), min(len(self.k), i1 + 2)
        r0, r1 = int(self.ptr[a0]), int(self.ptr[a1])
        tok = np.asarray(self.tok[r0:r1])
        fi = np.repeat(np.arange(a0, a1), np.diff(self.ptr[a0:a1 + 1]))
        return {'tok': tok, 'fi': fi, 'pos': self.idx[r0:r1].astype(np.int64), 'flag': self.flag[r0:r1], 'a0': a0, 'a1': a1,
                'i0': i0, 'i1': i1}


class Store:
    def __init__(self, dom):
        self.dom, self.cache, self.lock = dom, {}, threading.Lock()

    def video(self, ser, obs):
        with self.lock:
            v = self.cache.get((ser, obs))
        if v is None:
            v = Vid(self.dom, ser, obs)
            with self.lock:
                self.cache[(ser, obs)] = v
        return v

    def load(self, w):
        ser, obs, i0, i1 = w
        W = self.video(ser, obs).window(i0, i1)
        return feats_np(W).astype(np.float16), int(W['tok'].nbytes)

    def load_raw(self, w):
        """raw window rows + neighbour index (CPU part; the features are built on the GPU by feats_gpu)."""
        ser, obs, i0, i1 = w
        W = self.video(ser, obs).window(i0, i1)
        return W, nb_index(W)


# ---------------------------------------------------------------------------------------------- count
def cmd_count(a):
    from concurrent.futures import ThreadPoolExecutor
    dom = a.dom
    vids = store_videos(dom)
    t0 = time.time()

    def one(v):
        z = np.load(STORE / dom / VER / 'videos' / v[0] / v[1] / 'index.npz')
        fl = z['flag']
        k = z['k']
        return {'series': v[0], 'obs': v[1], 'frames': int(len(k)), 'U_rows': int(z['ptr'][-1]), 'M_tokens': int(((fl & 1) > 0).sum()),
                'k_min': int(k.min()), 'k_max': int(k.max())}
    with ThreadPoolExecutor(16) as ex:
        rows = list(ex.map(one, vids))
    tot = {key: int(sum(r[key] for r in rows)) for key in ('frames', 'U_rows', 'M_tokens')}
    ser = {}
    for r in rows:
        s = ser.setdefault(r['series'], {'videos': 0, 'frames': 0, 'M_tokens': 0, 'U_rows': 0})
        s['videos'] += 1
        for key in ('frames', 'M_tokens', 'U_rows'):
            s[key] += r[key]
    res = {'dom': dom, 'videos': len(rows), **tot, 'M_per_frame': tot['M_tokens'] / tot['frames'],
           'U_over_M': tot['U_rows'] / tot['M_tokens'], 'steps_one_pass': tot['M_tokens'] // BS,
           'V_fp16_TB': tot['M_tokens'] * 2048 / 1e12, 'by_series': ser, 'count_seconds': time.time() - t0,
           'windows': len(windows(dom)), 'videos_list': rows}
    # read + feature throughput: random windows, 1 thread then 16 threads (BeeGFS, cold)
    rng = np.random.default_rng(7)
    ws = windows(dom)
    pick = [ws[i] for i in rng.choice(len(ws), 160, replace=False)]
    st = Store(dom)
    b = {}
    for nthr, sub in ((1, pick[:32]), (16, pick[32:])):
        t1 = time.time()
        with ThreadPoolExecutor(nthr) as ex:
            out = list(ex.map(st.load, sub))
        el = time.time() - t1
        ntok = sum(len(o[0]) for o in out)
        nb = sum(o[1] for o in out)
        b[f'threads_{nthr}'] = {'windows': len(sub), 'seconds': el, 'U_GB_per_s': nb / el / 1e9, 'M_tokens_per_s': ntok / el}
    # read only (no features), 16 threads, other windows
    sub = [ws[i] for i in rng.choice(len(ws), 128, replace=False)]
    t1 = time.time()
    with ThreadPoolExecutor(16) as ex:
        nb = sum(ex.map(lambda w: int(st.video(w[0], w[1]).window(w[2], w[3])['tok'].nbytes), sub))
    el = time.time() - t1
    b['read_only_threads_16'] = {'windows': len(sub), 'seconds': el, 'U_GB_per_s': nb / el / 1e9}
    res['bench'] = b
    res['node'] = socket.gethostname()
    OUT23.mkdir(parents=True, exist_ok=True)
    (OUT23 / f'{dom}_tokens.json').write_text(json.dumps(res, indent=1))
    log(f'{dom}: {json.dumps({k: v for k, v in res.items() if k != "videos_list"})}')


def cmd_featcheck(a):
    """streamed features (feats_np) == v22_full.feats_v (torch CPU float32) on random windows; exact equality."""
    import torch
    import v22_full as VF
    rng = np.random.default_rng(3)
    ws = windows(a.dom)
    st = Store(a.dom)
    res = {'windows': 0, 'tokens': 0, 'max_abs_diff': 0.0}
    first = [i for i, w in enumerate(ws) if w[2] == 0]
    last = [i for i, w in enumerate(ws) if i + 1 == len(ws) or ws[i + 1][2] == 0]
    pick = list(rng.choice(len(ws), 40, replace=False)) + list(rng.choice(first, 5, replace=False)) + list(rng.choice(last, 5, replace=False))
    res['video_edge_windows'] = 10
    for i in pick:
        ser, obs, i0, i1 = ws[i]
        W = st.video(ser, obs).window(i0, i1)
        W0 = T.Video(a.dom, ser, obs).window(i0, i1)
        assert all(np.array_equal(W[k_], W0[k_]) for k_ in ('tok', 'fi', 'pos', 'flag')), 'window != T.Video.window'
        f1 = feats_np(W)
        f2, _ = VF.feats_v(W, torch.device('cpu'))
        f2 = f2.numpy()
        assert f1.shape == f2.shape
        res['windows'] += 1
        res['tokens'] += len(f1)
        res['max_abs_diff'] = max(res['max_abs_diff'], float(np.abs(f1 - f2).max()) if len(f1) else 0.0)
    OUT23.mkdir(parents=True, exist_ok=True)
    (OUT23 / f'{a.dom}_featcheck.json').write_text(json.dumps(res, indent=1))
    log(f'featcheck {a.dom}: {res}')
    if res['max_abs_diff'] != 0.0:
        raise SystemExit('FEATURE IDENTITY FAILED')


# ---------------------------------------------------------------------------------------------- train
def gpu_util():
    try:
        import subprocess
        o = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=10).stdout.strip().split('\n')[0]
        u, m = o.split(',')
        return int(u), int(m)
    except Exception:
        return -1, -1


def cmd_train(a):
    import torch
    from concurrent.futures import ThreadPoolExecutor
    from src.eci.sae import MatryoshkaBatchTopKSAE, TokenNorm, _train_step, geometric_median, save_checkpoint
    from src.eci.sae_calibrate import calibrate_threshold, set_threshold, sparsity_stats
    import v20 as V
    dom = a.dom
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # cpu = smoke test only
    seeds = [int(s) for s in a.seeds.split(',')]
    cnt = json.loads((OUT23 / f'{dom}_tokens.json').read_text())
    NM = cnt['M_tokens']
    steps = NM // BS if not a.max_steps else int(a.max_steps)
    total_sched = NM // BS  # the schedule always spans one pass (a bench run stops early on the same schedule)
    ws = windows(dom)
    order = np.random.default_rng(WIN_RNG).permutation(len(ws))
    chunks = [[ws[i] for i in order[c0:c0 + CW]] for c0 in range(0, len(order), CW)]
    spec = V.R.spec()['sae']
    cfg = {**spec, 'steps': total_sched, 'context_window': 0, 'arm': 'Vall', 'store': VER, 'data': 'all M_t tokens, all videos',
           'window_frames': WF, 'chunk_windows': CW, 'window_rng': WIN_RNG, 'warmup': 500}
    lr, warmup = spec['lr'], 500
    nthr = int(os.environ.get('SLURM_CPUS_PER_TASK', 16))
    st = Store(dom)
    ex = ThreadPoolExecutor(max(4, nthr - 2))
    log(f'{dom}: {NM:,} M_t tokens, {len(ws):,} windows in {len(chunks)} chunks, steps {steps:,} (schedule {total_sched:,}), '
        f'seeds {seeds}, {nthr} CPUs, device {dev}')
    pending = [ex.submit(st.load_raw, w) for w in chunks[0]]
    fcheck = {'windows': 0, 'tokens': 0, 'max_abs_diff': 0.0}

    def gather(fs):
        """-> (n, 1024) fp16 GPU tensor of the chunk (window order), bytes read, seconds waited for the loader."""
        wt, parts, nb_ = 0.0, [], 0
        for f in fs:
            t = time.time()
            W, I = f.result()
            wt += time.time() - t
            nb_ += W['tok'].nbytes
            if len(I['c']):
                parts.append(feats_gpu(W['tok'], I, dev))
                if fcheck['windows'] < 16:  # identity: GPU features == feats_np (== v22_full.feats_v, featcheck step)
                    ref = feats_np(W).astype(np.float16)
                    fcheck['max_abs_diff'] = max(fcheck['max_abs_diff'], float(np.abs(parts[-1].cpu().numpy().astype(np.float32) - ref.astype(np.float32)).max()))
                    fcheck['windows'] += 1
                    fcheck['tokens'] += len(ref)
                    if fcheck['max_abs_diff'] != 0.0:
                        raise SystemExit(f'GPU FEATURE IDENTITY FAILED {fcheck}')
        return torch.cat(parts), nb_, wt
    Xn, nb, wait = gather(pending)
    log(f'GPU feature identity vs numpy: {fcheck}')
    pending = [ex.submit(st.load_raw, w) for w in chunks[1]] if len(chunks) > 1 else []
    torch.backends.cuda.matmul.allow_tf32 = True
    M = {}
    for s in seeds:
        torch.manual_seed(s)
        g = np.random.default_rng(s)
        ii = torch.from_numpy(np.sort(g.choice(len(Xn), min(250_000, len(Xn)), replace=False))).to(dev)
        init = Xn[ii].float()
        norm = TokenNorm(1024).to(dev).fit_blocks(init, [1024], seed=s)
        sae = MatryoshkaBatchTopKSAE(d_in=1024, n_latents=spec['W'], prefixes=tuple(spec['prefixes']), k=spec['k'], seed=s).to(dev)
        with torch.no_grad():
            sae.b_dec.copy_(geometric_median(norm(init[:100_000])))
        del init
        M[s] = {'sae': sae, 'norm': norm, 'opt': torch.optim.Adam(sae.parameters(), lr=lr, betas=(0.9, 0.999)),
                'gen': torch.Generator(device=dev).manual_seed(s), 'hist': []}
    step, ci, carry, used = 0, 0, None, 0
    t0 = time.time()
    io = {'wait_s': wait, 'bytes': nb, 'tokens': len(Xn)}
    tlog, slog = time.time(), 0
    util = []
    while step < steps:
        Xc = Xn if carry is None else torch.cat([carry, Xn])
        del Xn
        n = (len(Xc) // BS) * BS
        last = ci == len(chunks) - 1
        carry = None if last else Xc[n:].clone()
        Xg = Xc[:n]
        del Xc
        perms = {s: torch.randperm(n, generator=M[s]['gen'], device=dev) for s in seeds}
        for b in range(n // BS):
            if step >= steps:
                break
            for s in seeds:
                m = M[s]
                logs, _ = _train_step(m['sae'], m['opt'], m['norm'](Xg[perms[s][b * BS:(b + 1) * BS]]), step, total_sched, lr, warmup, 1.0)
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
                    f'({(step - slog) * BS / el:,.0f} tokens/s per seed, {len(seeds)} SAEs), GPU util {u}% mem {mem} MiB, '
                    f'loader wait so far {io["wait_s"]:.0f}s of {time.time() - t0:.0f}s, '
                    f'fve s{seeds[0]} {M[seeds[0]]["hist"][-1]["fve"] if M[seeds[0]]["hist"] else float("nan"):.4f}, '
                    f'ETA {(steps - step) / ((step - slog) / el) / 60:.0f} min')
                tlog, slog = time.time(), step
        del Xg, perms
        if step >= steps or last:
            break
        ci += 1
        Xn, nb, wait = gather(pending)
        io['wait_s'] += wait
        io['bytes'] += nb
        io['tokens'] += len(Xn)
        pending = [ex.submit(st.load_raw, w) for w in chunks[ci + 1]] if ci + 1 < len(chunks) else []
    tt = time.time() - t0
    for f in pending:
        f.cancel()
    ex.shutdown(wait=False, cancel_futures=True)
    rec0 = {'node': socket.gethostname(), 'gpu': torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu', 'steps_run': step, 'steps_schedule': total_sched,
            'tokens_used_per_seed': used, 'M_tokens_total': NM, 'passes': used / NM, 'train_time_s': tt,
            'steps_per_s': step / tt, 'tokens_per_s_per_seed': used / tt, 'n_seeds_in_job': len(seeds),
            'loader_wait_s': io['wait_s'], 'store_bytes_read': io['bytes'], 'store_GB_per_s': io['bytes'] / tt / 1e9,
            'gpu_util_samples': util, 'gpu_util_median': float(np.median(util)) if util else None,
            'chunks_consumed': ci + 1, 'chunks_total': len(chunks), 'gpu_feature_identity': fcheck,
            'tokens_loaded': io['tokens']}
    log(f'train loop done: {json.dumps(rec0)}')
    (OUT23 / 'train').mkdir(parents=True, exist_ok=True)
    if a.bench:
        (OUT23 / 'train' / f'{dom}_bench_{socket.gethostname()}.json').write_text(json.dumps(rec0, indent=1))
        return
    # theta* on the calib sample, FVE on the eval sample (the V tokens of v21_train.cmd_train arm V)
    od = BEE21 / dom / 'data'
    X = {}
    for r in ('calib', 'eval'):
        B_ = np.load(od / f'{r}_B.npy', mmap_mode='r')
        X[r] = torch.from_numpy(np.asarray(B_[:, :1024], np.float32) + np.load(od / f'{r}_C.npy').astype(np.float32)).half()
    for s in seeds:
        m = M[s]
        sae, norm = m['sae'], m['norm']
        sae.eval()
        torch.backends.cuda.matmul.allow_tf32 = False
        th, _ = calibrate_threshold(sae, X['calib'], norm=norm)
        set_threshold(sae, th)
        sst = sparsity_stats(sae, X['eval'], th, norm=norm)
        rec = {'cfg': cfg, 'seed': s, **rec0, 'n_calib_tokens': len(X['calib']), 'n_eval_tokens': len(X['eval']),
               'theta_star': float(th), 'eval_fve': float(sst['fve']), 'eval_l0': float(sst['l0']),
               'dead_at_end_train': int((sae.tokens_since_fired >= sae.dead_tokens).sum()), 'history': m['hist'],
               'calib_eval_overlap': 'the calib / eval tokens come from v2.1 calib / eval-role videos, which are training videos here'}
        out = BEE23 / dom / 'sae' / f'Vall_s{s}'
        out.mkdir(parents=True, exist_ok=True)
        save_checkpoint(out / 'sae_theta.pt', sae, norm, extra={'config': 'eci_v2_3', 'cfg': cfg, 'seed': s, 'theta_star': float(th)})
        (out / 'train.json').write_text(json.dumps(rec, indent=1, default=float))
        (OUT23 / 'train' / f'{dom}_Vall_s{s}.json').write_text(json.dumps(rec, indent=1, default=float))
        log(f'{dom} Vall_s{s}: theta* {th:.4f} eval FVE {sst["fve"]:.5f} L0 {sst["l0"]:.2f} dead {rec["dead_at_end_train"]}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=('count', 'train', 'featcheck'))
    p.add_argument('--dom', required=True, choices=('mice', 'ants'))
    p.add_argument('--seeds', default='2001,2002,2003')
    p.add_argument('--max_steps', type=int, default=0)
    p.add_argument('--bench', action='store_true')
    a = p.parse_args()
    {'count': cmd_count, 'train': cmd_train, 'featcheck': cmd_featcheck}[a.step](a)
