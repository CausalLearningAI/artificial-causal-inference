"""
ECI v2.1 full token store with the approved mask v5 (configs/eci_v2/store_v21.json).

Per frame t (5 fps, native side S, canonical orientation, frozen v2.0 encoder):
    M_t = probe mask (DINOv3 block-20 token probe, per-domain threshold t / core t_core, mask5_val.postproc)
    U_t = union of M_s for s in t-2..t+2 (the frames that exist)
    stored once per frame: patch indices of U_t (uint16), tokens fp16 (n, 1024), flag uint8 (bit0 = in M_t,
    bit1 = in N2 at t), probe probability uint8 (round(255 p)).
U_{t+k} contains M_t for |k| <= 2, so every token at a position of M_t is present in frames t-2..t+2 (checked in
`check` and per video by verify_video).

Layout: <root>/<domain>/<version>/videos/<series>/<obs>/{tok.f16, index.npz}; written in /localhome staging and moved
atomically (directory rename); a video is complete iff index.npz exists (resumable). Manifest: `manifest` step.

Steps:
  encode --domain D --task i --ntask n     (GPU) videos i, i + n, ... of the domain (all series), skipping complete ones
  check --domain D                         (GPU) the pre-launch checks (tokens vs v2.0 store5, mask IoU vs 2080 Ti,
                                           no-duplication property, throughput) -> results/.../store_v21/check_<D>.json
  manifest --domain D                      (CPU) manifest.json (videos, frames, tokens, bytes, crc32 per file)
  setup                                    (CPU) copy the approved probes + thresholds into <root>/<domain>/<version>/_probe
"""
import argparse
import json
import os
import shutil
import sys
import time
import zlib
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
os.environ.setdefault('V2_OUT', str(REPO / 'results/vision/eci_v2_0B'))  # the mask capture / N2 path of mask v4 / v5
os.environ.setdefault('V2_BEE', '/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_0B')
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts/eci/v2'))
CFG = json.loads((REPO / 'configs/eci_v2/store_v21.json').read_text())
RES = REPO / 'results/vision/eci_v2_0B/store_v21'
# token gate vs the v2.0 store5 (recalibrated 2026-10-30 on the mice / ants per-token cosine distributions, see
# check_<dom>.json 'controls': a one-frame shift or a 90 degree rotation must fail it)
COS_P1, COS_MIN, MAD_MAX = 0.999, 0.995, 0.02
log = lambda s: print(time.strftime('%H:%M:%S'), s, flush=True)  # noqa: E731


def dcfg(dom):
    return CFG['domains'][dom]


def vroot(dom):
    return Path(CFG['root']) / dom / CFG['version']


def probe_file(dom):
    d = dcfg(dom)
    return Path(d['probe']) if d.get('probe') else vroot(dom) / '_probe' / 'probe.npz'


def load_probe(dom, dev):
    import torch
    P = np.load(probe_file(dom))
    t = lambda k: torch.from_numpy(np.asarray(P[k], np.float32)).to(dev)  # noqa: E731
    return {'mu': t('mu'), 'sd': t('sd'), 'w': t('w'), 'b': float(P['b'])}


def encb(enc, x):
    """encode in sub-batches that fit the GPU (11 GB cards: 1 frame at 1024 px, 2 below; >= 40 GB: up to 16)."""
    import torch
    mem = torch.cuda.get_device_properties(0).total_memory / 2**30
    S = x.shape[-1]
    eb = int(os.environ.get('EBS', 0)) or (1 if mem < 16 and S >= 1000 else 2 if mem < 16 else 4 if mem < 30 else 16)
    return torch.cat([enc(x[i:i + eb]) for i in range(0, len(x), eb)])


def probe_gpu(P, z):
    """z (B, N, 1024) fp16 cuda -> p (B, N) float32 (same arithmetic as mask5_val.probe_p, in torch)."""
    import torch
    x = (z.float() - P['mu']) / P['sd']
    return torch.sigmoid(x @ P['w'] + P['b'])


def mask_of(p, G, d):
    import mask5_val as M5
    return M5.postproc(p, G, d['t'], d['t_core'], None, d['smooth'])


BENCH_NAME = {'mab': 'mabe22', 'cal': 'calms21', 'crim': 'crim13'}


class BenchSeries:
    """design-benchmark frame source with the lab Series interface (job / run_job / frames / S / G / videos).
    Frames: scripts/eci/v2_temporal/temporal.py BENCHES (decode, canvas at the benchmark store side S: MABe22 512,
    CalMS21 1024, CRIM13 640), 5 fps grid = source frames k * step. Video list: scripts/eci/v2/v20_bench_store.py
    video_universe (all videos / clips). Flag bit1 for benchmarks = the temporal-study tight flag (B.fd >= 0.25 with
    the per-clip background of B.mask_prep), because N2 is not defined at the benchmark store side."""

    geometry = 'bench'

    def __init__(self, dom, dev, encoder=None):
        sys.path.insert(0, str(REPO / 'scripts/eci/v2_temporal'))
        import v20_bench_store as BS
        self.B, U = BS.video_universe(BENCH_NAME[dom])
        self.dev, self.S, self.step = dev, self.B.S, self.B.step
        self.G = self.S // 16
        self.videos = [{'obs': q, 'split': sp, 'N': int(N), 'n': -(-int(N) // self.step), 'path': None} for q, sp, N in U]
        self.V = {v['obs']: v for v in self.videos}
        self._prep = None

    def job(self, v, k0, n, threads=4):
        return {'q': v['obs'], 'split': v['split'], 'want': [int(k) * self.step for k in range(k0, k0 + n)]}

    def run_job(self, q, split, want):
        B = self.B
        if BENCH_NAME_REV.get(type(B).__name__) == 'mab':
            fr = B.EC.decode_clip(split, q, Path(os.environ['LOCAL_DIR']))
            return [fr[i] for i in want]
        fr, _ = B.decode(q, set(want), None)
        return [fr[i] for i in want]

    def frames(self, v, fr, ks):
        import torch
        B = self.B
        B._cur = v['obs']
        if self._prep != v['obs']:
            bgi = B.bg_index(v['N'])
            bg = self.run_job(v['obs'], v['split'], [int(i) for i in bgi])
            B.mask_prep({int(i): f for i, f in zip(bgi, bg)}, v['N'], self.dev, self.S, B.cells(v['obs']))
            self._prep = v['obs']
        import temporal as T
        x = T.canvas_at(B, list(fr), self.dev, self.S)
        fd = B.fd(list(fr), self.dev, self.S)
        return x, (fd >= 0.25).float() if fd.dtype != torch.bool else fd.float()


class BenchSource:
    """design-benchmark source on the APPROVED mask-v5 benchmark path (the benchmark agent's code, imported read-only):
    decode + canvas = scripts/eci/v2/mask5_bench.Data (MABe22 512 bicubic, CalMS21 1024 accurate-seek padded canvas,
    CRIM13 640 canvas_gpu), 5 fps grid = source frames k * STEP; N2 (flag bit1) = mask5_bench.n2_masks with the pooled
    background pack (Data.packs); final mask M_t (flag bit0) = mask5_tune.kg_fast(probe mask, keypoints); probe-only
    mask (flag bit2) = mask5_val.postproc of the probe."""

    geometry = 'bench'

    def __init__(self, dom, dev, encoder=None):
        for q in ('scripts/eci/v2_mask2', 'scripts/benchmark/crim13', 'scripts/eci/v2_temporal'):
            if str(REPO / q) not in sys.path:
                sys.path.insert(0, str(REPO / q))
        import mask5_bench as MB
        import v20_bench_store as BS
        self.MB, self.kind, self.dev = MB, dom, dev
        self.D = MB.Data(dom, dev)
        self.S, self.step, self.h = MB.SIDE[dom], MB.STEP[dom], MB.h_of(dom)
        self.G = self.S // 16
        _, U = BS.video_universe(MB.NAME[dom])
        self.videos = [{'obs': q, 'split': sp, 'N': int(N), 'n': -(-int(N) // self.step), 'path': None} for q, sp, N in U]
        self.V = {v['obs']: v for v in self.videos}
        self.cur, self.kpcache = None, {}

    def job(self, v, k0, n, threads=4):
        return {'vid': v['obs'], 'split': v['split'], 'N': v['N'], 'idx': [int(k) * self.step for k in range(k0, k0 + n)]}

    def run_job(self, vid, split, N, idx):
        CM = self.D.CM
        idx = np.minimum(np.asarray(idx), N - 1)
        if self.kind == 'mab':
            arr = CM.mab_decode(split, vid)
            return arr[np.clip(idx, 0, len(arr) - 1)]
        if self.kind == 'cal':
            return cal_decode_seq(CM, vid, split, idx)
        from align_common import ROOT, load_seek, read_frames
        v = self.D.inv[vid]
        d = ROOT / v['dir']
        offs = load_seek(d / f'{vid}_t-seek.mat')
        fr, _ = read_frames(d / f'{vid}_t.seq', offs, set(idx.tolist()), threads=2)
        return np.stack([fr[int(i)] for i in idx])

    def _prep(self, v):
        if self.cur == v['obs']:
            return
        dec = self.D.decode(v['obs'], v['split'], np.array([0]), v['N'])
        self.W, self.H = dec.get('W'), dec.get('H')
        _, self.pack, self.pooled = self.D.packs(v['obs'], dec)
        self.kpf = self._kps_loader(v)
        self.cur = v['obs']

    def _kps_loader(self, v):
        CM = self.D.CM
        q, sp = v['obs'], v['split']
        if self.kind == 'mab':
            if sp not in self.kpcache:
                self.kpcache = {sp: CM.mab_keypoints(sp)}
            seq = self.kpcache[sp].get(q)

            def f(s):
                i = int(s) - 1  # keypoint index = video frame - 1
                if seq is None or i < 0 or i >= len(seq['keypoints']):
                    return np.full((3, 12, 2), np.nan, np.float32), 1.0
                k = CM.mab_kp(seq, i, 512 / 224)
                k[((k < 0) | (k > 512)).any(-1)] = np.nan
                return k, 1.0
            return f
        if self.kind == 'cal':
            if sp not in self.kpcache:
                self.kpcache = {sp: CM.CalKP(sp)}
            K = self.kpcache[sp]
            T = int(K.lens[K.names.index(q)])
            return lambda s: (K.get(q, int(min(s, T - 1))), 1.0)
        from align_common import ROOT, geom, load_track
        iv = self.D.inv[q]
        pos, nm = load_track(ROOT / iv['dir'] / f'{q}_t-track.mat')
        W, H = int(iv['header']['width']), int(iv['header']['height'])
        pt, sc = geom(W, H)
        xy = np.stack([pos[..., 0] * sc, (pos[..., 1] + pt) * sc], -1)
        xy[nm < 2] = np.nan
        return lambda s: (xy[int(min(s, len(nm) - 1))], 640 / W)

    def frames(self, v, fr, ks):
        self._prep(v)
        x = self.D.canvas(fr, self.W, self.H) if self.kind == 'crim' else self.D.canvas(fr)
        n2 = np.stack([r['keep'] for r in self.MB.n2_masks(x, self.pack, self.h)])
        import torch
        return x, torch.from_numpy(n2.astype(np.float32))

    def final_mask(self, v, k, p, d):
        """-> (M_t final, probe-only mask, fallback flag)."""
        import mask5_tune as MT
        B = mask_of(p, self.G, d)
        if d.get('arm', 'probe') in ('KG', 'KGt'):
            kps, scale = self.kpf(int(k) * self.step)
            M, fb = MT.kg_fast(self.kind, B, kps, self.S, self.G, self.h, scale)
            return M, B, bool(fb)
        return B, B, True


BENCH_NAME_REV = {'MABe22': 'mab', 'CalMS21': 'cal', 'CRIM13': 'crim'}


def lab_videos(dom):
    """-> [(series, video dict)] over the domain's series (plan order); benchmarks: [(dom, clip)]."""
    if dcfg(dom)['kind'] == 'bench':
        return [(dom, v) for v in BenchSource(dom, None).videos]
    import v20 as V
    c = V.dcfg(dom)
    out = []
    for ser in dcfg(dom)['series']:
        sc = V.R.series_of(c, ser)
        P = json.loads((REPO / sc['plan']).read_text())
        out += [(ser, v) for v in P['videos']]
    return out


class Writer:
    """rolling 5-frame buffer: push frames in order, records for frame t are emitted once M_{t+2} exists."""

    def __init__(self, n, G):
        self.n, self.G = n, G
        self.buf = {}  # k -> dict(z (N, 1024) cuda fp16, p (N,) np, M (N,) bool, n2 (N,) bool)
        self.idx, self.flag, self.prob, self.cnt, self.ks = [], [], [], [], []
        self.tok = []
        self.nfb = 0  # benchmark frames where the keypoint step fell back to the probe mask

    def push(self, k, z, p, M, n2, B=None):
        self.buf[k] = {'z': z, 'p': p, 'M': M, 'n2': n2, 'B': B}
        self.flush(k - 2)

    def flush(self, t):
        if t < 0 or t not in self.buf:
            return
        U = np.zeros(self.G * self.G, bool)
        for s in range(t - 2, t + 3):
            if s in self.buf:
                U |= self.buf[s]['M']
        b = self.buf[t]
        ii = np.flatnonzero(U).astype(np.uint16)
        import torch
        self.tok.append(b['z'][torch.from_numpy(ii.astype(np.int64)).to(b['z'].device)].cpu().numpy())
        self.idx.append(ii)
        fl = b['M'][ii].astype(np.uint8) | (b['n2'][ii].astype(np.uint8) << 1)
        if b['B'] is not None:
            fl |= b['B'][ii].astype(np.uint8) << 2
        self.flag.append(fl)
        self.prob.append(np.round(255 * b['p'][ii]).astype(np.uint8))
        self.cnt.append(len(ii))
        self.ks.append(t)
        b['z'] = None
        for s in [q for q in self.buf if q < t - 1]:  # U_{t+1} needs M_{t-1} .. M_{t+3}
            del self.buf[s]

    def finish(self):
        for t in sorted(self.buf):
            if self.buf.get(t) is not None and self.buf[t]['z'] is not None:
                self.flush(t)


def encode_video(Se, enc, P, d, v, n, ch, bs, local, par=4):
    """one pass over video v -> (tok array list, index dict) written to local/<obs>/."""
    import torch
    from src.eci.v2.video import Prefetch
    jobs = [((k0,), Se.job(v, k0, min(ch, n - k0), threads=4)) for k0 in range(0, n, ch)]
    pf = Prefetch(jobs, par, fn=Se.run_job)
    W = Writer(n, Se.G)
    od = local / v['obs']
    if od.exists():
        shutil.rmtree(od)
    od.mkdir(parents=True)
    ftok = open(od / 'tok.f16', 'wb')
    nbytes = 0
    for (k0,), fr in pf:
        for b0 in range(0, len(fr), bs):
            kk = k0 + b0 + np.arange(len(fr[b0:b0 + bs]))
            with torch.no_grad():
                x, fd = Se.frames(v, fr[b0:b0 + bs], kk)
                z = encb(enc, x)
                p = probe_gpu(P, z).cpu().numpy()
            n2 = (fd >= 0.25).cpu().numpy()
            for i, k in enumerate(kk):
                if hasattr(Se, 'final_mask'):
                    M, B, fb = Se.final_mask(v, k, p[i], d)
                    W.nfb += int(fb)
                    W.push(int(k), z[i], p[i], M, n2[i], B)
                else:
                    W.push(int(k), z[i], p[i], mask_of(p[i], Se.G, d), n2[i])
                while W.tok:
                    a = W.tok.pop(0)
                    ftok.write(a.tobytes())
                    nbytes += a.nbytes
    W.finish()
    while W.tok:
        a = W.tok.pop(0)
        ftok.write(a.tobytes())
        nbytes += a.nbytes
    ftok.close()
    cnt = np.array(W.cnt, np.int64)
    assert W.ks == list(range(n)), (W.ks[:5], n)
    assert nbytes == int(cnt.sum()) * 2048, (nbytes, cnt.sum())
    ptr = np.r_[0, np.cumsum(cnt)]
    np.savez(od / 'index.npz', k=np.array(W.ks, np.int32), ptr=ptr, idx=np.concatenate(W.idx) if W.idx else np.zeros(0, np.uint16),
             flag=np.concatenate(W.flag) if W.flag else np.zeros(0, np.uint8),
             prob=np.concatenate(W.prob) if W.prob else np.zeros(0, np.uint8), G=Se.G, S=Se.S, obs=v['obs'],
             version=CFG['version'], probe_t=d['t'], probe_t_core=d['t_core'], probe_smooth=d['smooth'],
             fps=5, encoder=CFG['encoder'], arm=d.get('arm', 'probe'), arm_label=d.get('label', ''),
             kg_fallback_frames=W.nfb, flag_bits='bit0 M_t final | bit1 N2 at t | bit2 probe-only mask (benchmarks)')
    return od, int(cnt.sum()), nbytes


def verify_video(od):
    """no-duplication property on a written video: every position of M_t (flag bit0) is in U_s for |s - t| <= 2."""
    z = np.load(od / 'index.npz')
    ptr, idx, flag = z['ptr'], z['idx'], z['flag']
    n = len(ptr) - 1
    sets = [set(idx[ptr[t]:ptr[t + 1]].tolist()) for t in range(n)]
    bad, checked = 0, 0
    for t in range(n):
        Mt = idx[ptr[t]:ptr[t + 1]][(flag[ptr[t]:ptr[t + 1]] & 1) == 1]
        for s in range(max(0, t - 2), min(n, t + 3)):
            checked += len(Mt)
            bad += sum(1 for q in Mt.tolist() if q not in sets[s])
    size_ok = (od / 'tok.f16').stat().st_size == int(ptr[-1]) * 2048
    return {'frames': n, 'pairs_checked': checked, 'missing': bad, 'size_ok': bool(size_ok)}


def series_obj(dom, ser, dev, enc):
    if dcfg(dom)['kind'] == 'bench':
        return BenchSource(dom, dev)
    import v20 as V
    NS = V.make_series_class(dom)
    sc = V.R.series_of(V.dcfg(dom), ser)
    return NS(sc, dev, encoder=enc)


def stage_video(v, local):
    """copy the source video to /localhome (inputs staged per video); returns a patched video dict. Benchmarks: their
    decoders stage the clip themselves (temporal.py decode copies to /localhome)."""
    if v.get('path') is None:
        return v, Path(local) / '_none'
    src = Path(v['path'])
    dst = local / '_src' / src.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src, dst)
    return {**v, 'path': str(dst)}, dst


def cmd_encode(a):
    import torch
    from src.eci.v2.encode import Encoder
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False
    d = dcfg(a.domain)
    if d['kind'] == 'bench':
        assert d.get('enabled') and d.get('probe'), f'{a.domain}: benchmark mask not approved yet (configs/eci_v2/store_v21.json)'
        ecfg = {'chunk': 400, 'bs': 16}
    else:
        import v20 as V
        ecfg = V.dcfg(a.domain)['encode']
    ch, bs = ecfg['chunk'], min(ecfg['bs'], 16 if a.domain == 'mice' else ecfg['bs'])
    enc = Encoder(dev)
    P = load_probe(a.domain, dev)
    local = Path(os.environ['LOCAL_DIR'])
    vids = lab_videos(a.domain)[a.task::a.ntask]
    SE = {}
    T0, nf_tot = time.time(), 0
    for ser, v in vids:
        dst = vroot(a.domain) / 'videos' / ser / v['obs']
        if (dst / 'index.npz').exists():
            log(f'skip {ser} {v["obs"]} (complete)')
            continue
        if ser not in SE:
            SE[ser] = series_obj(a.domain, ser, dev, enc)
        Se = SE[ser]
        n = min(v['n'], a.max_frames) if a.max_frames else v['n']
        t0, g0 = time.time(), enc.gpu_s
        vs, srcf = stage_video(v, local)
        od, ntok, nbytes = encode_video(Se, enc, P, d, vs, n, ch, bs, local)
        srcf.unlink(missing_ok=True)
        ver = verify_video(od)
        assert ver['missing'] == 0 and ver['size_ok'], ver
        tmp = dst.with_name(dst.name + '.tmp')
        if tmp.exists():
            shutil.rmtree(tmp)
        shutil.copytree(od, tmp)
        if dst.exists():
            shutil.rmtree(dst)
        os.rename(tmp, dst)
        shutil.rmtree(od)
        dt = time.time() - t0
        nf_tot += n
        log(f'{ser} {v["obs"]}: {n} frames {dt:.0f}s ({n / dt:.1f} fps, GPU {enc.gpu_s - g0:.0f}s) | U tok/frame '
            f'{ntok / n:.1f} | {nbytes / 1e9:.2f} GB | verify {ver}')
    log(f'task {a.task}/{a.ntask} DONE: {nf_tot} frames in {time.time() - T0:.0f}s on {torch.cuda.get_device_name(0)}')


def cmd_setup(a):
    """copy the approved mask-v5 probes (eci_mask5/<dom>/probe.npz as refitted by the extended run) + thresholds."""
    for dom in ('ants', 'mice', 'frogs'):
        d = dcfg(dom)
        pd_ = vroot(dom) / '_probe'
        pd_.mkdir(parents=True, exist_ok=True)
        src = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_mask5') / dom / 'probe.npz'
        shutil.copy2(src, pd_ / 'probe.npz')
        (pd_ / 'thresholds.json').write_text(json.dumps({k: d[k] for k in ('t', 't_core', 'smooth', 'or_n2')} |
                                                        {'version': CFG['version'], 'probe_crc32': zlib.crc32((pd_ / 'probe.npz').read_bytes())}))
        log(f'{dom}: probe -> {pd_}')


def cmd_check(a):
    """pre-launch checks on the encode GPU."""
    import torch
    import pandas as pd
    import mask5_val as M5
    from src.eci.v2.encode import Encoder
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False
    dom = a.domain
    d = dcfg(dom)
    enc = Encoder(dev)
    P = load_probe(dom, dev)
    Pn = {k: np.load(vroot(dom) / '_probe' / 'probe.npz')[k] for k in ('mu', 'sd', 'w', 'b')}
    Pn['b'] = float(Pn['b'])
    out = {'gpu': torch.cuda.get_device_name(0), 'domain': dom}
    # 1. tokens vs the v2.0 store5 tokens (same frames, same positions)
    st = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_0') / dom / 'store5'
    diffs, coss, nt = [], [], 0
    shards = sorted(p for p in st.glob('*s*') if (p / 'tok.f16').exists()) if st.exists() else []
    if shards:
        F = pd.read_parquet(shards[0] / 'frames.parquet')
        tf = np.load(shards[0] / 'tok_frame.npy')
        tp = np.load(shards[0] / 'tok_pos.npy')
        tok = np.memmap(shards[0] / 'tok.f16', np.float16, 'r').reshape(-1, 1024)
        ser, obs = F.series.iloc[0], F.obs.iloc[0]
        import v20 as V
        Se = series_obj(dom, ser, dev, enc)
        v = Se.V[obs]
        rows = np.flatnonzero((F.obs == obs).values)[:a.nframes]
        ks = F.k.values[rows]
        fr = Se.run_job(**Se.job(v, int(ks[0]), int(ks[-1] - ks[0] + 2), threads=4))  # +1 frame for the shift control
        arms = {'same': [], 'ctl_next_frame': [], 'ctl_rot90': []}
        dd_all = []

        def cosv(a_, b_):
            return (a_ * b_).sum(1) / (np.linalg.norm(a_, axis=1) * np.linalg.norm(b_, axis=1))
        with torch.no_grad():
            for i0 in range(0, len(ks), 8):
                kk = ks[i0:i0 + 8]
                x, _ = Se.frames(v, fr[kk - ks[0]], kk)
                z = encb(enc, x).float().cpu().numpy()
                xn, _ = Se.frames(v, fr[kk - ks[0] + 1], kk + 1)
                zn = encb(enc, xn).float().cpu().numpy()
                zr = encb(enc, torch.rot90(x, 1, dims=(2, 3)).contiguous()).float().cpu().numpy()
                for j, r in enumerate(rows[i0:i0 + 8]):
                    sel = np.flatnonzero(tf == r)
                    if not len(sel):
                        continue
                    ref = tok[sel].astype(np.float32)
                    pos = tp[sel].astype(np.int64)
                    dd_all.append(np.abs(z[j, pos] - ref).ravel())
                    arms['same'].append(cosv(z[j, pos], ref))
                    arms['ctl_next_frame'].append(cosv(zn[j, pos], ref))
                    arms['ctl_rot90'].append(cosv(zr[j, pos], ref))
                    nt += len(sel)
        dd = np.concatenate(dd_all)
        stats = {}
        for nm, L in arms.items():
            cc = np.concatenate(L)
            stats[nm] = {'cos_min': float(cc.min()), 'cos_p0.1': float(np.quantile(cc, 0.001)),
                         'cos_p1': float(np.quantile(cc, 0.01)), 'cos_p5': float(np.quantile(cc, 0.05)),
                         'cos_median': float(np.median(cc)), 'cos_mean': float(cc.mean()),
                         'share_below_0.999': float((cc < 0.999).mean()), 'share_below_0.99': float((cc < 0.99).mean())}
        gate = lambda st_: st_['cos_p1'] >= COS_P1 and st_['cos_min'] >= COS_MIN  # noqa: E731
        out['tokens_vs_store5'] = {'shard': shards[0].name, 'obs': obs, 'frames': int(len(ks)), 'tokens': int(nt),
                                   'max_abs_diff': float(dd.max()), 'mean_abs_diff': float(dd.mean()), **stats['same'],
                                   'controls': {k_: v_ for k_, v_ in stats.items() if k_ != 'same'},
                                   'gate': f'cos_p1 >= {COS_P1} and cos_min >= {COS_MIN} and mean_abs_diff <= {MAD_MAX}',
                                   'pass': bool(gate(stats['same']) and dd.mean() <= MAD_MAX),
                                   'controls_fail_gate': bool(not any(gate(stats[c_]) for c_ in ('ctl_next_frame', 'ctl_rot90')))}
    else:
        out['tokens_vs_store5'] = 'no v2_0 store5 for this domain'
    log(json.dumps(out))
    # 2. probe mask IoU vs the 2080 Ti masks (mask5 windows), same frames
    W = sorted((Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_mask5') / dom).glob('win_*.npz'))[:a.nwin]
    inter = union = 0
    zd = []
    for f in W:
        zz = np.load(f)
        ser, rest = f.stem[4:].split('_', 1)
        obs, k = rest.rsplit('_', 1)
        Se = series_obj(dom, ser, dev, enc)
        ks = [int(q) for q in zz['ks']]
        fr = Se.run_job(**Se.job(Se.V[obs], ks[0], len(ks), threads=4))
        with torch.no_grad():
            x, _ = Se.frames(Se.V[obs], fr, ks)
            z = encb(enc, x)
            p = probe_gpu(P, z).cpu().numpy()
        G = int(zz['G'])
        for i in range(len(ks)):
            m_new = mask_of(p[i], G, d)
            m_old = mask_of(M5.probe_p(Pn, zz['z'][i]), G, d)
            inter += int((m_new & m_old).sum())
            union += int((m_new | m_old).sum())
        zd.append(float(np.abs(z.float().cpu().numpy() - zz['z'].astype(np.float32)).mean()))
    out['mask_iou_vs_2080ti'] = {'windows': len(W), 'token_iou': inter / max(union, 1), 'mean_abs_token_diff': float(np.mean(zd))}
    log(json.dumps(out['mask_iou_vs_2080ti']))
    # 3 + 4. full pass on the first short video(s) of the domain (max_frames), verify, throughput
    local = Path(os.environ['LOCAL_DIR'])
    vids = lab_videos(dom)[:a.nvid]
    thr = []
    import v20 as V
    ecfg = V.dcfg(dom)['encode']
    for ser, v in vids:
        Se = series_obj(dom, ser, dev, enc)
        n = min(v['n'], a.max_frames)
        vs, srcf = stage_video(v, local)
        t0 = time.time()
        od, ntok, nb = encode_video(Se, enc, P, d, vs, n, ecfg['chunk'], min(ecfg['bs'], 16), local)
        dt = time.time() - t0
        srcf.unlink()
        ver = verify_video(od)
        thr.append({'obs': v['obs'], 'frames': n, 's': dt, 'fps': n / dt, 'U_tok_per_frame': ntok / n, **ver})
        shutil.rmtree(od)
        log(json.dumps(thr[-1]))
    out['videos'] = thr
    t5 = out['tokens_vs_store5']
    out['pass'] = bool((isinstance(t5, str) or (t5['pass'] and t5['controls_fail_gate']))
                       and out['mask_iou_vs_2080ti']['token_iou'] >= 0.98 and all(t['missing'] == 0 and t['size_ok'] for t in thr))
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f'check_{dom}.json').write_text(json.dumps(out, indent=1))
    log(f'CHECK {dom} pass={out["pass"]}')
    if not out['pass']:
        raise SystemExit(f'check failed for {dom}')


def cmd_manifest(a):
    rows, tot = [], {'videos': 0, 'frames': 0, 'tokens': 0, 'bytes': 0}
    root = vroot(a.domain) / 'videos'
    for ser in (dcfg(a.domain).get('series') or [a.domain]):
        for od in sorted((root / ser).glob('*')):
            if od.name.endswith('.tmp') or not (od / 'index.npz').exists():
                continue
            z = np.load(od / 'index.npz')
            nb = (od / 'tok.f16').stat().st_size
            crc = 0
            with open(od / 'tok.f16', 'rb') as f:
                while True:
                    blk = f.read(1 << 26)
                    if not blk:
                        break
                    crc = zlib.crc32(blk, crc)
            rows.append({'series': ser, 'obs': od.name, 'frames': int(len(z['k'])), 'tokens': int(z['ptr'][-1]), 'bytes': nb,
                         'crc32_tok': crc, 'crc32_index': zlib.crc32((od / 'index.npz').read_bytes())})
            for k in ('frames', 'tokens', 'bytes'):
                tot[k] += rows[-1][k]
            tot['videos'] += 1
    allv = lab_videos(a.domain)
    done = {(r['series'], r['obs']) for r in rows}
    missing = [f'{ser}/{v["obs"]}' for ser, v in allv if (ser, v['obs']) not in done]
    exp = len(allv)
    man = {'domain': a.domain, 'version': CFG['version'], 'expected_videos': exp, **tot, 'TB': tot['bytes'] / 1e12,
           'missing_videos': missing, 'videos_list': rows}
    (vroot(a.domain) / 'manifest.json').write_text(json.dumps(man, indent=1))
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f'manifest_{a.domain}.json').write_text(json.dumps(man, indent=1))
    log(json.dumps({k: (v if k != 'missing_videos' else len(v)) for k, v in man.items() if k != 'videos_list'}))


def cmd_benchtest(a):
    """smoke test of the benchmark path (NOT a store write): first clip, a.max_frames frames, with the mice probe as a
    stand-in for the not-yet-approved benchmark probe; output stays in LOCAL_DIR and is deleted."""
    import torch
    from src.eci.v2.encode import Encoder
    dev = torch.device('cuda')
    enc = Encoder(dev)
    P = load_probe('mice', dev)
    d = dict(dcfg('mice'))
    Se = BenchSeries(a.domain, dev)
    v = Se.videos[0]
    n = min(v['n'], a.max_frames or 60)
    t0 = time.time()
    od, ntok, nb = encode_video(Se, enc, P, d, v, n, 30, 8, Path(os.environ['LOCAL_DIR']))
    ver = verify_video(od)
    out = {'bench': a.domain, 'clip': v['obs'], 'S': Se.S, 'G': Se.G, 'frames': n, 'fps': n / (time.time() - t0),
           'U_tok_per_frame': ntok / n, **ver, 'gpu': torch.cuda.get_device_name(0)}
    shutil.rmtree(od)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f'benchtest_{a.domain}.json').write_text(json.dumps(out, indent=1))
    log(json.dumps(out))


def cmd_benchcheck(a):
    """benchmark pre-launch check: probe probabilities / masks vs the benchmark agent's 2080 Ti outputs (eval files),
    no-duplication + throughput on the first video(s)."""
    import torch
    from src.eci.v2.encode import Encoder
    dom = a.domain
    d = dcfg(dom)
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False
    enc = Encoder(dev)
    P = load_probe(dom, dev)
    Se = BenchSource(dom, dev)
    out = {'gpu': torch.cuda.get_device_name(0), 'domain': dom, 'arm': d['arm']}
    ev = sorted((Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_mask5_bench') / dom).glob('eval_*.npz'))[:a.nwin]
    dp, ip, ia, ua, ik, uk, nfr = [], 0, 0, 0, 0, 0, 0
    for f in ev:
        z = np.load(f)
        vid = str(z['vid'])
        v = Se.V[vid]
        ks = np.asarray(z['ks'])
        fr = Se.run_job(vid, v['split'], v['N'], ks)
        with torch.no_grad():
            x, _ = Se.frames(v, fr, ks // Se.step)
            zt = encb(enc, x)
            p = probe_gpu(P, zt).cpu().numpy()
        for i in range(len(ks)):
            pa = z['pB'][z['inv'][i][2]].astype(np.float32)
            dp.append(np.abs(p[i] - pa))
            m1, m2 = mask_of(p[i], Se.G, d), mask_of(pa, Se.G, d)
            ia += int((m1 & m2).sum()); ua += int((m1 | m2).sum())
            k5 = ks[i] // Se.step
            f1, _, _ = Se.final_mask(v, k5, p[i], d)
            f2, _, _ = Se.final_mask(v, k5, pa, d)
            ik += int((f1 & f2).sum()); uk += int((f1 | f2).sum())
            nfr += 1
    dd = np.concatenate(dp)
    out['vs_bench_agent_2080ti'] = {'eval_files': len(ev), 'frames': nfr, 'max_abs_dp': float(dd.max()), 'mean_abs_dp': float(dd.mean()),
                                    'probe_mask_iou': ia / max(ua, 1), 'final_mask_iou': ik / max(uk, 1)}
    log(json.dumps(out))
    local = Path(os.environ['LOCAL_DIR'])
    thr = []
    for v in Se.videos[:a.nvid]:
        n = min(v['n'], a.max_frames)
        t0 = time.time()
        od, ntok, nb = encode_video(Se, enc, P, d, v, n, 400, 16, local)
        dt = time.time() - t0
        ver = verify_video(od)
        zi = np.load(od / 'index.npz')
        thr.append({'obs': v['obs'], 'frames': n, 'fps': n / dt, 'U_tok_per_frame': ntok / n, 'kg_fallback_frames': int(zi['kg_fallback_frames']),
                    'M_tok_per_frame': float(((zi['flag'] & 1) > 0).sum() / n), **ver})
        shutil.rmtree(od)
        log(json.dumps(thr[-1]))
    out['videos'] = thr
    vb = out['vs_bench_agent_2080ti']
    out['pass'] = bool(vb['probe_mask_iou'] >= 0.98 and vb['final_mask_iou'] >= 0.98 and all(t['missing'] == 0 and t['size_ok'] for t in thr))
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f'check_{dom}.json').write_text(json.dumps(out, indent=1))
    log(f'CHECK {dom} pass={out["pass"]}')
    if not out['pass']:
        raise SystemExit(f'check failed for {dom}')


def cal_decode_seq(CM, name, split, idx, threads=4):
    """CalMS21 frames idx (an every-STEP run, possibly repeated at the clip end) in ONE ffmpeg pass: accurate seek to the
    first frame exactly as common.cal_decode does for a single frame, then keep every 6th decoded frame (select), same
    pad filter -> (n, 3, 1024, 1024) uint8, pixel-identical to common.cal_decode (verified by `decodecheck`). Any other
    index pattern falls back to common.cal_decode."""
    import subprocess
    idx = np.asarray(idx, int)
    a0 = int(idx[0])
    rel = idx - a0
    if (rel < 0).any() or (rel % 6).any() or (np.diff(idx) < 0).any():
        return CM.cal_decode(name, split, idx, threads=2)
    cnt = int(rel.max() // 6 + 1)
    path = CM.CAL / f'videos/task1_videos_mp4/{split}/{name}.mp4'
    cmd = [CM.FFMPEG, '-nostdin', '-v', 'error', '-threads', str(threads)]
    if a0 > 0:
        cmd += ['-ss', f'{(a0 - 0.5) / 30:.6f}']
    cmd += ['-i', str(path), '-vf', 'select=not(mod(n\\,6)),pad=1024:1024:0:227:color=0x7C7468', '-vsync', '0',
            '-frames:v', str(cnt), '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-']
    p = subprocess.run(cmd, capture_output=True)
    a = np.frombuffer(p.stdout, np.uint8)
    got = a.size // (1024 * 1024 * 3)
    if got < cnt:  # clip end: fewer frames decoded than requested (indices clipped to N - 1 repeat the last one)
        if got == 0:
            return CM.cal_decode(name, split, idx, threads=2)
    a = a[:got * 1024 * 1024 * 3].reshape(got, 1024, 1024, 3)
    j = np.minimum(rel // 6, got - 1)
    return a[j].transpose(0, 3, 1, 2).copy()


def cmd_decodecheck(a):
    """pixel identity of cal_decode_seq vs common.cal_decode on one CalMS21 video (first a.max_frames 5 fps frames, two
    chunks), plus decode timings."""
    for q in ('scripts/eci/v2_mask2',):
        sys.path.insert(0, str(REPO / q))
    import common as CM
    Se = BenchSource('cal', None)
    v = Se.videos[a.task]
    n = min(v['n'], a.max_frames)
    out = {'video': v['obs'], 'split': v['split'], 'N': v['N'], 'frames': n, 'chunks': []}
    for k0 in (0, n // 2):
        idx = np.minimum(np.arange(k0, min(n, k0 + n // 2)) * 6, v['N'] - 1)
        t0 = time.time()
        x1 = CM.cal_decode(v['obs'], v['split'], idx, threads=2)
        t1 = time.time()
        x2 = cal_decode_seq(CM, v['obs'], v['split'], idx)
        t2 = time.time()
        out['chunks'].append({'k0': int(k0), 'frames': int(len(idx)), 'identical': bool(np.array_equal(x1, x2)),
                              'n_diff_px': int((x1 != x2).any(1).sum()), 'old_fps': len(idx) / (t1 - t0),
                              'new_fps': len(idx) / (t2 - t1)})
        log(json.dumps(out['chunks'][-1]))
    # the clip end (indices clipped to N - 1)
    idx = np.minimum(np.arange(v['n'] - 5, v['n'] + 1) * 6, v['N'] - 1)
    out['end_identical'] = bool(np.array_equal(CM.cal_decode(v['obs'], v['split'], idx, threads=2), cal_decode_seq(CM, v['obs'], v['split'], idx)))
    log(json.dumps(out))
    RES.mkdir(parents=True, exist_ok=True)
    (RES / 'decodecheck_cal.json').write_text(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('step')
    ap.add_argument('--domain', default='ants')
    ap.add_argument('--task', type=int, default=int(os.environ.get('SLURM_ARRAY_TASK_ID', 0)))
    ap.add_argument('--ntask', type=int, default=1)
    ap.add_argument('--max_frames', type=int, default=0)
    ap.add_argument('--nframes', type=int, default=24)
    ap.add_argument('--nwin', type=int, default=4)
    ap.add_argument('--nvid', type=int, default=1)
    a = ap.parse_args()
    if a.step in ('check', 'benchcheck') and not a.max_frames:
        a.max_frames = 300
    globals()['cmd_' + a.step](a)


if __name__ == '__main__':
    main()
