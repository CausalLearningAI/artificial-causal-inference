"""
ECI v2.4 pooled token store (pre-registration scripts/eci/v2/v24_prereg.md).

From the v2.1 token store (DINOv3-L/16 block 20, mask v5; U_t = union of M_{t-2..t+2}, so every M_t position is stored
at frames t-2..t+2 inside a video) build, for EVERY frame of EVERY store video, the average-pooled tokens on a PxP grid
(P = 2 and 4):
  cell rule (= VP2, scripts/eci/v2/v22_encode.pool2): a pooled cell of frame t exists iff >= 1 M_t token (flag bit0)
      lies in its PxP block of the native G x G grid (grid G -> ceil(G / P); a border block of an odd-size grid is
      averaged over the tokens it has, exactly as pool2 with (G + 1) // 2); cell value = mean over the M_t tokens of
      the block; n = their number.
  x  = mean over the block's M_t positions of the token at frame t
  d  = mean over the SAME positions of Delta = x_{t+L} - x_{t-L}, L = 2 frames (+-0.4 s at 5 fps, the largest
       symmetric lag the store holds). Edge rule (a video's first / last 2 frames): one-sided 2 (x_{t+L} - x_t) or
       2 (x_t - x_{t-L}) when only one neighbour exists (same scale as the central difference), 0 when none.
  arithmetic: float64 sums of fp16 tokens (exact, so independent of summation order), mean in float64, cast float64 ->
      float32 -> float16 (numpy). x and d are stored separately (fp16), so P2W (x + alpha d), P2N (x) and P2C
      ([x, alpha d]) all read the same store.
Layout: BEE24/<dom>/pool/P<P>/t<task>.{x,d}.f16 (rows of the task's videos in video / frame / cell order) +
  t<task>.npz (vid = index into v23_train.store_videos(dom), fi = frame position in the store video, k, pos = pooled
  cell (row * ceil(G / P) + col), n = native M_t tokens in the cell; vrow = per-video row ranges) + t<task>.json
  (counts and energy sums for alpha).
Steps
  pool --dom D --task i --ntask n   (CPU) videos[i::n]
  check --dom D                     (CPU) 3 random frames per P: stored x / d == an independent per-cell numpy loop
                                    (exact) and the cell set == v22_encode.pool2 (P = 2, VP2's rule); value diff vs
                                    pool2's float32 mean reported
  alpha --dom D                     (CPU) combine the task stats -> OUT24/<dom>_pool.json: pooled token counts,
                                    native per pooled, alpha = sqrt(E||x - mean x||^2 / E||d||^2) per P over ALL pooled
                                    tokens (label-free), steps / passes per the pre-registered rule
"""
import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
for p in ('', 'scripts/eci/v2'):
    sys.path.insert(0, str(REPO / p))
import v23_train as V23  # noqa: E402

STORE, VER = V23.STORE, V23.VER
BEE24 = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_4')
OUT24 = REPO / 'results/vision/eci_v2_4'
LAG = 2
POOLS = (2, 4)
WF = 100
BS, MIN_STEPS = 4096, 50_000
log = V23.log


def videos(dom):
    return V23.store_videos(dom)


def lag_rows(W, c):
    """rows of the same position at t-L / t+L for centre rows c (ok = exists)."""
    key = W['fi'].astype(np.int64) * 65536 + W['pos']
    assert (np.diff(key) > 0).all(), 'store rows not sorted by (frame, position)'
    out = {}
    for s in (-1, 1):
        q = key[c] + s * LAG * 65536
        j = np.minimum(np.searchsorted(key, q), len(key) - 1)
        out[s] = (np.where(key[j] == q, j, 0), key[j] == q)
    return out


def cell_keys(fi, pos, G, P):
    g = (G + P - 1) // P
    return fi.astype(np.int64) * (g * g) + (pos // G // P) * g + (pos % G // P), g


def pool_window(W, G, Ps=POOLS):
    """-> {P: (x fp16 (m, 1024), d fp16 (m, 1024), fi (m,), cell pos (m,), n (m,))} for frames i0..i1-1 of window W.
    Cell sums as sparse 0 / +-1 / +-2 matrices times the float64 tokens: every fp16 value is an integer multiple of 2^-24,
    so these float64 sums are EXACT (order-independent); P = 4 sums = sums of the P = 2 sums (exact)."""
    from scipy.sparse import csr_matrix
    c = np.flatnonzero(((W['flag'] & 1) > 0) & (W['fi'] >= W['i0']) & (W['fi'] < W['i1']))
    out = {}
    if not len(c):
        return out, 0
    nb = lag_rows(W, c)
    (jm, okm), (jp, okp) = nb[-1], nb[1]
    tok = W['tok'].astype(np.float64)
    nr = len(tok)
    fi, pos = W['fi'][c], W['pos'][c].astype(np.int64)
    key2, g2 = cell_keys(fi, pos, G, 2)
    u2, inv2, n2 = np.unique(key2, return_inverse=True, return_counts=True)
    m2 = len(u2)
    Sx = csr_matrix((np.ones(len(c)), (inv2, c)), shape=(m2, nr))
    both, onlyp, onlym = okp & okm, okp & ~okm, okm & ~okp
    rr = np.concatenate([inv2[both], inv2[both], inv2[onlyp], inv2[onlyp], inv2[onlym], inv2[onlym]])
    cc = np.concatenate([jp[both], jm[both], jp[onlyp], c[onlyp], c[onlym], jm[onlym]])
    vv = np.concatenate([np.full(both.sum(), 1.0), np.full(both.sum(), -1.0), np.full(onlyp.sum(), 2.0), np.full(onlyp.sum(), -2.0),
                         np.full(onlym.sum(), 2.0), np.full(onlym.sum(), -2.0)])
    Sd = csr_matrix((vv, (rr, cc)), shape=(m2, nr))
    sx, sd, n = {2: Sx @ tok}, {2: Sd @ tok}, {2: n2.astype(np.float64)}
    u = {2: u2}
    if 4 in Ps:
        fr, rc = u2 // (g2 * g2), u2 % (g2 * g2)
        g4 = (G + 3) // 4
        assert g4 == (g2 + 1) // 2
        key4 = fr * (g4 * g4) + (rc // g2 // 2) * g4 + (rc % g2 // 2)
        u4, inv4 = np.unique(key4, return_inverse=True)
        S4 = csr_matrix((np.ones(m2), (inv4, np.arange(m2))), shape=(len(u4), m2))
        sx[4], sd[4], n[4], u[4] = S4 @ sx[2], S4 @ sd[2], S4 @ n[2], u4
    for P in Ps:
        g = (G + P - 1) // P
        nn = n[P][:, None]
        out[P] = ((sx[P] / nn).astype(np.float32).astype(np.float16), (sd[P] / nn).astype(np.float32).astype(np.float16),
                  (u[P] // (g * g)).astype(np.int32), (u[P] % (g * g)).astype(np.uint16), n[P].astype(np.uint8))
    return out, len(c)


# ---------------------------------------------------------------------------------------------- pool
def cmd_pool(a):
    dom = a.dom
    vids = videos(dom)
    mine = list(range(len(vids)))[a.task::a.ntask]
    t0 = time.time()
    od = {P: BEE24 / dom / 'pool' / f'P{P}' for P in POOLS}
    fh, idx, stats = {}, {}, {}
    for P in POOLS:
        od[P].mkdir(parents=True, exist_ok=True)
        fh[P] = {s: open(od[P] / f't{a.task}.{s}.f16.part', 'wb', buffering=64 << 20) for s in ('x', 'd')}
        idx[P] = {k: [] for k in ('vid', 'fi', 'k', 'pos', 'n')}
        stats[P] = {'rows': 0, 'native': 0, 'sum_x': np.zeros(1024), 'sum_x2': 0.0, 'sum_d': np.zeros(1024), 'sum_d2': 0.0,
                    'vrow': {}}
    nat = 0
    for vi in mine:
        ser, obs, nf = vids[vi]
        V = V23.Vid(dom, ser, obs)
        G = int(np.load(STORE / dom / VER / 'videos' / ser / obs / 'index.npz')['G'])
        r0 = {P: stats[P]['rows'] for P in POOLS}
        for i0 in range(0, nf, WF):
            W = V.window(i0, min(nf, i0 + WF))
            pw, nc = pool_window(W, G)
            nat += nc
            for P, (x, d, fi, pos, n) in pw.items():
                fh[P]['x'].write(x.tobytes())
                fh[P]['d'].write(d.tobytes())
                for k_, v_ in (('vid', np.full(len(fi), vi, np.int16)), ('fi', fi), ('k', V.k[fi].astype(np.int32)), ('pos', pos), ('n', n)):
                    idx[P][k_].append(v_)
                s = stats[P]
                s['rows'] += len(x)
                s['native'] += int(n.sum())
                x64, d64 = x.astype(np.float64), d.astype(np.float64)
                s['sum_x'] += x64.sum(0)
                s['sum_d'] += d64.sum(0)
                s['sum_x2'] += float((x64 * x64).sum())
                s['sum_d2'] += float((d64 * d64).sum())
        for P in POOLS:
            stats[P]['vrow'][vi] = [r0[P], stats[P]['rows']]
        el = time.time() - t0
        log(f'{dom} task {a.task}: video {mine.index(vi) + 1}/{len(mine)} {ser}/{obs}: {nat:,} native tokens, {el:.0f}s, '
            f'{nat / el:,.0f} native tokens/s, P2 rows {stats[2]["rows"]:,}, P4 rows {stats[4]["rows"]:,}')
    for P in POOLS:
        for s_ in ('x', 'd'):
            fh[P][s_].close()
            os.replace(od[P] / f't{a.task}.{s_}.f16.part', od[P] / f't{a.task}.{s_}.f16')
        np.savez(od[P] / f't{a.task}.npz', **{k: np.concatenate(v) for k, v in idx[P].items()},
                 vrow=np.array([[vi, *stats[P]['vrow'][vi]] for vi in mine], np.int64))
        s = stats[P]
        meta = {'task': a.task, 'ntask': a.ntask, 'videos': mine, 'rows': s['rows'], 'native_tokens': s['native'],
                'native_tokens_all_P': nat, 'sum_x': s['sum_x'].tolist(), 'sum_d': s['sum_d'].tolist(), 'sum_x2': s['sum_x2'],
                'sum_d2': s['sum_d2'], 'seconds': time.time() - t0, 'node': socket.gethostname()}
        assert s['native'] == nat, 'pooled cells do not cover every M_t token'
        (od[P] / f't{a.task}.json').write_text(json.dumps(meta))
    log(f'{dom} task {a.task} done: {len(mine)} videos, {nat:,} M_t tokens, P2 {stats[2]["rows"]:,} / P4 {stats[4]["rows"]:,} '
        f'pooled rows ({time.time() - t0:.0f}s)')


# ---------------------------------------------------------------------------------------------- shard reader
class Shards:
    """the pooled store of one domain and P: per task files; per video row range -> (task, r0, r1)."""

    def __init__(self, dom, P):
        self.d = BEE24 / dom / 'pool' / f'P{P}'
        self.tasks = sorted(int(p.stem[1:]) for p in self.d.glob('t*.json'))
        self.meta = {t: json.loads((self.d / f't{t}.json').read_text()) for t in self.tasks}
        nt = self.meta[self.tasks[0]]['ntask']
        assert self.tasks == list(range(nt)), f'{self.d}: tasks {len(self.tasks)} of {nt}'
        self.rows = {t: self.meta[t]['rows'] for t in self.tasks}
        self.vrow = {}
        for t in self.tasks:
            z = np.load(self.d / f't{t}.npz')
            for vi, r0, r1 in z['vrow']:
                self.vrow[int(vi)] = (t, int(r0), int(r1))
        self._idx = {}

    def index(self, t):
        if t not in self._idx:
            z = np.load(self.d / f't{t}.npz')
            self._idx[t] = {k: z[k] for k in ('vid', 'fi', 'k', 'pos', 'n')}
        return self._idx[t]

    def read(self, t, which, r0, r1):
        """rows r0..r1-1 of task t, which in ('x', 'd') -> fp16 (r1 - r0, 1024); os.preadv (releases the GIL)."""
        out = np.empty((r1 - r0, 1024), np.float16)
        fd = os.open(self.d / f't{t}.{which}.f16', os.O_RDONLY)
        try:
            mv, off, done = memoryview(out).cast('B'), r0 * 2048, 0
            while done < out.nbytes:
                n = os.preadv(fd, [mv[done:]], off + done)
                assert n > 0, 'short read'
                done += n
        finally:
            os.close(fd)
        return out


# ---------------------------------------------------------------------------------------------- check
def recompute_frame(dom, ser, obs, i, P):
    """independent per-cell recomputation of frame i (explicit python loop over cells, float64) -> {pos: (x, d, n)}."""
    T = V23.T
    Vd = T.Video(dom, ser, obs)
    G = Vd.G
    W = Vd.window(i, i + 1)
    fi, pos, tok, fl = W['fi'], W['pos'], W['tok'], W['flag']
    rows = {(int(f), int(p)): r for r, (f, p) in enumerate(zip(fi, pos))}
    g = (G + P - 1) // P
    cells = {}
    for r in np.flatnonzero(((fl & 1) > 0) & (fi == i)):
        p = int(pos[r])
        cells.setdefault((p // G // P) * g + (p % G // P), []).append(r)
    out = {}
    for cp, rr in cells.items():
        sx, sd = np.zeros(1024), np.zeros(1024)
        for r in rr:
            x = tok[r].astype(np.float64)
            p = int(pos[r])
            jp, jm = rows.get((i + LAG, p)), rows.get((i - LAG, p))
            if jp is not None and jm is not None:
                dd = tok[jp].astype(np.float64) - tok[jm].astype(np.float64)
            elif jp is not None:
                dd = 2 * (tok[jp].astype(np.float64) - x)
            elif jm is not None:
                dd = 2 * (x - tok[jm].astype(np.float64))
            else:
                dd = np.zeros(1024)
            sx += x
            sd += dd
        out[cp] = ((sx / len(rr)).astype(np.float32).astype(np.float16), (sd / len(rr)).astype(np.float32).astype(np.float16), len(rr))
    return out, W, G


def cmd_check(a):
    import torch
    import v22_encode as EN
    dom = a.dom
    vids = videos(dom)
    rng = np.random.default_rng(24)
    res = {'frames': [], 'all_exact': True}
    picks = [(int(v), int(rng.integers(0, vids[v][2]))) for v in rng.choice(len(vids), 3, replace=False)]
    picks.append((picks[0][0], 0))  # a video-edge frame (one-sided Delta)
    picks.append((picks[1][0], vids[picks[1][0]][2] - 1))
    for P in POOLS:
        sh = Shards(dom, P)
        for vi, i in picks:
            ser, obs, nf = vids[vi]
            t, r0, r1 = sh.vrow[vi]
            ix = sh.index(t)
            m = np.flatnonzero(ix['fi'][r0:r1] == i) + r0
            ref, W, G = recompute_frame(dom, ser, obs, i, P)
            x, d = sh.read(t, 'x', int(m[0]), int(m[-1]) + 1), sh.read(t, 'd', int(m[0]), int(m[-1]) + 1)
            assert (np.diff(m) == 1).all()
            pos = ix['pos'][m].astype(int)
            same_cells = sorted(pos.tolist()) == sorted(ref)
            ex = same_cells and all(np.array_equal(x[q], ref[p][0]) and np.array_equal(d[q], ref[p][1]) and ix['n'][m[q]] == ref[p][2]
                                    for q, p in enumerate(pos))
            rec = {'P': P, 'video': f'{ser}/{obs}', 'frame': i, 'cells': len(pos), 'same_cells': same_cells, 'exact_x_d_n': bool(ex)}
            if P == 2:  # VP2's rule (v22_encode.pool2) on the same frame: cell set and float32 mean of x
                c = np.flatnonzero(((W['flag'] & 1) > 0) & (W['fi'] == i))
                fp, _, ppos, _ = EN.pool2(torch.from_numpy(W['tok'][c].astype(np.float32)), W['fi'][c], W['pos'][c].astype(np.int64), G)
                o = np.argsort(pos)
                rec['pool2_same_cells'] = bool(np.array_equal(np.sort(ppos), pos[o]))
                rec['pool2_x_max_abs_diff'] = float(np.abs(fp.numpy()[np.argsort(ppos)] - x[o].astype(np.float32)).max())
                rec['pool2_x_equal_after_fp16_cast'] = bool(np.array_equal(fp.numpy()[np.argsort(ppos)].astype(np.float16), x[o]))
            res['frames'].append(rec)
            res['all_exact'] &= bool(ex) and rec.get('pool2_same_cells', True)
            log(f'check {dom}: {rec}')
    OUT24.mkdir(parents=True, exist_ok=True)
    (OUT24 / f'{dom}_poolcheck.json').write_text(json.dumps(res, indent=1))
    if not res['all_exact']:
        raise SystemExit('POOL IDENTITY FAILED')


# ---------------------------------------------------------------------------------------------- alpha / counts
def plan_steps(n):
    """pre-registered rule: >= 1 full pass and >= ~50k steps -> passes = max(1, ceil(50,000 * 4096 / N)),
    steps = floor(passes * N / 4096)."""
    passes = max(1, int(np.ceil(MIN_STEPS * BS / n)))
    return passes, (passes * n) // BS


def cmd_alpha(a):
    dom = a.dom
    cnt = json.loads((V23.OUT23 / f'{dom}_tokens.json').read_text())
    res = {'dom': dom, 'lag_frames': LAG, 'lag_seconds': LAG / 5, 'M_tokens_store': cnt['M_tokens'], 'P': {}}
    for P in POOLS:
        sh = Shards(dom, P)
        M = [sh.meta[t] for t in sh.tasks]
        N = sum(m['rows'] for m in M)
        nat = sum(m['native_tokens'] for m in M)
        assert nat == cnt['M_tokens'], f'P{P}: pooled cells cover {nat:,} M_t tokens, store has {cnt["M_tokens"]:,}'
        assert len(sh.vrow) == len(videos(dom))
        sx = np.sum([m['sum_x'] for m in M], 0)
        ex2 = sum(m['sum_x2'] for m in M) / N
        ed2 = sum(m['sum_d2'] for m in M) / N
        mu = sx / N
        var_x = ex2 - float(mu @ mu)
        alpha = float(np.sqrt(var_x / ed2))
        passes, steps = plan_steps(N)
        res['P'][P] = {'pooled_tokens': N, 'native_per_pooled': nat / N, 'E_x_minus_mean_sq': var_x, 'E_x_sq': ex2, 'E_d_sq': ed2,
                       'E_d_minus_mean_sq': ed2 - float((np.sum([m['sum_d'] for m in M], 0) / N) @ (np.sum([m['sum_d'] for m in M], 0) / N)),
                       'alpha': alpha, 'passes': passes, 'steps': steps, 'tokens_used': steps * BS,
                       'passes_exact': steps * BS / N, 'tasks': len(M), 'pool_seconds_max': max(m['seconds'] for m in M)}
        log(f'{dom} P{P}: {json.dumps(res["P"][P])}')
    OUT24.mkdir(parents=True, exist_ok=True)
    (OUT24 / f'{dom}_pool.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=('pool', 'check', 'alpha'))
    p.add_argument('--dom', required=True, choices=('mice', 'ants'))
    p.add_argument('--task', type=int, default=0)
    p.add_argument('--ntask', type=int, default=1)
    a = p.parse_args()
    {'pool': cmd_pool, 'check': cmd_check, 'alpha': cmd_alpha}[a.step](a)
