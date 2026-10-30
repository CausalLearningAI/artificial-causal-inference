"""
Mask v5 on the mouse benchmarks (MABe22, CalMS21, CRIM13). Pre-registration: scripts/eci/v2/mask5_bench_prereg.md.
Arms: T = approved mice probe as it is; B = per-benchmark probe (SAM B_l_full labels offline, 120 training frames
from the benchmark's training split); N2 (pooled / per-clip background). Scored with keypoints only.

Steps (scripts/eci/v2/mask5_bench.sh):
  run   --set cal|mab|crim  (GPU) training frames + SAM labels -> probe B; held-out frames + 5-frame windows: tokens,
                            probabilities of T and B, N2 -> BEE/<set>/
  score --set ...           (CPU) metrics, decision rules, store projection -> OUT/bench_<set>_v5.json, bench_v5.md
  sheet --set ...           (CPU) OUT/bench_<name>.png
"""
import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts/eci/v2'))
sys.path.insert(0, str(REPO / 'scripts/eci/v2_mask2'))
sys.path.insert(0, str(REPO / 'scripts/benchmark/crim13'))
OUT = REPO / 'results/vision/eci_v2_0B/mask_approval_v3'
OUT4 = REPO / 'results/vision/eci_v2_0B/mask_approval_v2'
BEE = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_mask5_bench')
BG = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_bench_bg')
MICE_PROBE = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_mask5/mice/probe.npz')
CFG = {'t': 0.9, 't_core': 0.8, 'or_n2': False, 'smooth': False}  # approved mice setting, used for T and B
NAME = {'mab': 'mabe22', 'cal': 'calms21', 'crim': 'crim13'}
STEP = {'mab': 6, 'cal': 6, 'crim': 5}
SIDE = {'mab': 512, 'cal': 1024, 'crim': 640}
log = lambda s: print(time.strftime('%H:%M:%S'), s, flush=True)  # noqa: E731


def h_of(kind):
    h = json.loads((OUT4 / f'bench_{kind}.json').read_text())['h']
    return h * 512 / 896 if kind == 'mab' else h


def resize_to(x, S):
    import torch
    import torch.nn.functional as F
    return F.interpolate(x.float(), size=(S, S), mode='bicubic', align_corners=False).round().clamp(0, 255).to(torch.uint8)


def load_probe(path):
    z = np.load(path)
    P = {k: z[k] for k in ('mu', 'sd', 'w', 'b')}
    P['b'] = float(P['b'])
    return P


# ------------------------------------------------------------------------------------------------ data (per video)
class Data:
    """decodes frames on the store canvas (uint8 (n, 3, S, S) torch on dev) and background packs."""

    def __init__(self, kind, dev):
        import common as CM
        self.CM, self.kind, self.dev = CM, kind, dev
        self.groups = json.loads((BG / NAME[kind] / 'groups.json').read_text())
        self.pk_cache = {}
        if kind == 'crim':
            self.inv = {v['name']: v for v in json.loads((REPO / 'results/vision/benchmark_crim13/inventory/inventory.json')
                                                          .read_text())['videos']}

    # raw decode (CPU threads) -> dict of numpy arrays
    def decode(self, vid, split, idx, N=None):
        """idx: sorted unique source indices -> native frames list in the same order (+ 48 bg frames)."""
        CM = self.CM
        bgi = np.linspace(0, N - 1, 48).round().astype(int)
        if self.kind == 'mab':
            arr = CM.mab_decode(split, vid)
            return {'x': arr[np.clip(idx, 0, len(arr) - 1)], 'bg': arr[np.linspace(0, len(arr) - 1, 48).round().astype(int)]}
        if self.kind == 'cal':
            return {'x': CM.cal_decode(vid, split, idx, threads=2), 'bg': CM.cal_decode(vid, split, bgi, threads=2)}
        from align_common import ROOT, load_seek, read_frames
        v = self.inv[vid]
        d = ROOT / v['dir']
        offs = load_seek(d / f'{vid}_t-seek.mat')
        fr, _ = read_frames(d / f'{vid}_t.seq', offs, set(np.asarray(idx).tolist()) | set(bgi.tolist()), threads=2)
        return {'x': np.stack([fr[int(i)] for i in idx]), 'bg': np.stack([fr[int(i)] for i in bgi]),
                'W': int(v['header']['width']), 'H': int(v['header']['height'])}

    def canvas(self, a, W=None, H=None):
        import torch
        if self.kind == 'mab':
            return torch.cat([resize_to(torch.from_numpy(a[i:i + 64]).to(self.dev), 512) for i in range(0, len(a), 64)])
        if self.kind == 'cal':
            return torch.from_numpy(np.ascontiguousarray(a)).to(self.dev)
        from align_common import canvas_gpu
        return torch.cat([canvas_gpu(list(a[i:i + 64]), W, H, self.dev).to(torch.uint8) for i in range(0, len(a), 64)])

    def packs(self, vid, dec):
        """-> (per-clip pack, pooled pack or the per-clip one when the clip has no setup, pooled flag)."""
        from src.eci.v2 import mask2 as M2
        pc = M2.background_pack(self.canvas(dec['bg'], dec.get('W'), dec.get('H')))
        g = self.groups['clip_group'].get(vid)
        if g is None:
            return pc, pc, False
        if g not in self.pk_cache:
            self.pk_cache.clear()
            st = np.load(BG / NAME[self.kind] / f'pool_g{g}.npy')
            shp = self.groups['groups'][str(g)]['shape']
            import torch
            try:
                x = self.canvas(st, shp[1], shp[0]) if self.kind == 'crim' else self.canvas(st)
                self.pk_cache[g] = M2.background_pack(x)
            except torch.cuda.OutOfMemoryError:
                x = None
                torch.cuda.empty_cache()
                dev0, self.dev = self.dev, torch.device('cpu')
                x = self.canvas(st, shp[1], shp[0]) if self.kind == 'crim' else self.canvas(st)
                self.dev = dev0
                self.pk_cache[g] = {k: v.to(dev0) if torch.is_tensor(v) else v for k, v in M2.background_pack(x).items()}
                log(f'pooled pack g{g}: GPU OOM -> computed on the CPU')
            del x
            torch.cuda.empty_cache()
        return pc, self.pk_cache[g], True


def prefetch(ex, fn, items, k=3):
    """ordered map with at most k results decoded ahead (bounded memory)."""
    from collections import deque
    q = deque()
    it = iter(items)
    for x in it:
        q.append(ex.submit(fn, x))
        if len(q) >= k:
            break
    while q:
        r = q.popleft().result()
        nx = next(it, None)
        if nx is not None:
            q.append(ex.submit(fn, nx))
        yield r


def n2_masks(x, pack, h):
    """validate.masks_all N2 (+ core0, app) on canvas frames x."""
    import validate as V
    from src.eci.v2 import mask2 as M2
    out = []
    for b0 in range(0, len(x), 8):
        xx = x[b0:b0 + 8]
        ms, r = V.masks_all(xx, pack, h)
        core0 = V.dark_cue(xx, pack['q98'])
        _, app = M2.appendage(r['pix'], h)
        for i in range(len(xx)):
            out.append({'keep': ms['N2'][0][i].cpu().numpy(), 'pix': r['pix'][i].cpu().numpy(),
                        'core0': core0[i].cpu().numpy(), 'app': app[i].cpu().numpy()})
    return out


# ------------------------------------------------------------------------------------------------ frame lists
def train_items(kind):
    g = np.random.default_rng(0)
    if kind == 'cal':
        import common as CM
        names, lens = CM.cal_names('train')
        out = []
        for j in sorted(g.choice(len(names), 30, replace=False)):
            T = int(lens[j])
            out.append((names[j], 'train', np.sort(g.choice(np.arange(30, T - 30), 4, replace=False)), T))
        return out
    if kind == 'mab':
        sys.path.insert(0, str(REPO / 'scripts/benchmark/mabe22'))
        import eci_common as EC
        import common as CM
        clips = sorted(p.stem for p in (CM.MAB / 'videos/mouse_userTrain').iterdir() if p.suffix == '.avi')
        clips = [c for c in clips if c not in EC.EXCLUDE]
        out = []
        for c in sorted(g.choice(clips, 60, replace=False)):
            out.append((c, 'mouse_userTrain', np.sort(g.choice(np.arange(30, 1800, 6), 2, replace=False)) + 1, 1801))
        return out
    from align_common import meta_of
    I = json.loads((REPO / 'results/vision/benchmark_crim13/inventory/inventory.json').read_text())['videos']
    out = []
    for strain in ('BC', 'C57'):
        vs = sorted([v for v in I if v['split'] == 'train' and meta_of(v['name'])[2] == strain], key=lambda v: v['name'])
        for v in [vs[i] for i in sorted(g.choice(len(vs), 20, replace=False))]:
            N = int(v['frames'])
            out.append((v['name'], 'train', np.sort(g.choice(np.arange(50, N - 50, 5), 3, replace=False)), N))
    return out


def eval_items(kind):
    """validate.py held-out frames -> list of (vid, split, centre source idx, N, kps per frame, extra)."""
    import validate as V
    if kind == 'cal':
        return [(nm, 'test', ks, T, kps, None) for nm, T, ks, kps in V.cal_videos('test', 24, 0)]
    if kind == 'mab':
        return [(c, 'mouse_submission', ks + 1, 1801, [k * 512 / 896 for k in kps], None) for c, ks, kps in V.mab_clips(120, 8, 0)]
    from align_common import geom
    out = []
    for nm, strain, d, N, W, H, ks, pos in V.crim_videos(12, 16, 0):
        pt, sc = geom(W, H)
        xy = np.stack([pos[..., 0] * sc, (pos[..., 1] + pt) * sc], -1)
        out.append((nm, 'test', ks, N, [xy[i] for i in range(len(ks))], strain))
    return out


# ------------------------------------------------------------------------------------------------ run (GPU)
def cmd_run(a):
    import torch
    import mask5_val as MV
    from src.eci.v2 import mask4 as M4
    from src.eci.v2.encode import Encoder
    import validate as V
    dev = torch.device('cuda')
    kind = a.set
    D = BEE / kind
    D.mkdir(parents=True, exist_ok=True)
    h = h_of(kind)
    S = SIDE[kind]
    G = S // 16
    data = Data(kind, dev)
    enc = Encoder(dev)
    if not torch.cuda.is_bf16_supported():  # RTX 2080 Ti (debug_gpu): fp32 encode, as mask5_val build
        import src.eci.v2.encode as EN

        def call32(self, x):
            xin = (x.float() / 255 - self.mean) / self.std
            with torch.no_grad():
                hh = self.m(pixel_values=xin).last_hidden_state
            z = hh[:, EN.N_PRE:].float()
            assert torch.isfinite(z).all()
            return z.half()
        EN.Encoder.__call__ = call32
        log('encoder: fp32 (bf16 unsupported on this GPU)')
    EB = 1 if S >= 1024 else 4  # encode batch (11 GB 2080 Ti in fp32)
    t0 = time.time()
    log(f'{kind}: S {S} G {G} h {h:.3f} on {torch.cuda.get_device_name(0)}')
    # ---------------- training frames for B
    if a.part in ('train', 'all') and not (D / 'probe.npz').exists():
        sam, proc = M4.load_sam('large', dev)
        items = train_items(kind)
        Zs, Ys, stats = [], [], []
        gneg = np.random.default_rng(0)
        with ThreadPoolExecutor(4) as ex:
            decs = prefetch(ex, lambda it: data.decode(it[0], it[1], it[2], it[3]), items)
            for (vid, sp, ks, N), dec in zip(items, decs):
                x = data.canvas(dec['x'], dec.get('W'), dec.get('H'))
                _, pp, _ = data.packs(vid, dec)
                n2 = n2_masks(x, pp, h)
                z = torch.cat([enc(x[i:i + EB]) for i in range(0, len(x), EB)])
                for i in range(len(ks)):
                    q = n2[i]
                    core = q['core0'] | (q['pix'] & ~q['app'])
                    objs = M4.objects(q['pix'], core, h)
                    ch = M4.sam_full(sam, proc, x[i].permute(1, 2, 0).cpu().numpy(), objs, dev)
                    u, _ = M4.assemble(ch, objs, q['pix'], core, h, rule='B', app=q['app'])
                    y = M4.tokens(u, h, dev=dev)
                    neg = np.flatnonzero(~y)
                    sel = np.concatenate([np.flatnonzero(y), gneg.choice(neg, min(len(neg), max(4 * int(y.sum()), 200)), replace=False)])
                    Zs.append(z[i].cpu().numpy()[sel])
                    Ys.append(y[sel])
                    stats.append({'vid': vid, 'k': int(ks[i]), 'lab_tok': int(y.sum()), 'n2_tok': int(q['keep'].sum())})
                del x, z
                if len(stats) % 20 < len(ks):
                    log(f'train {len(stats)} frames {time.time() - t0:.0f}s')
        Z, Y = np.concatenate(Zs), np.concatenate(Ys).astype(np.float32)
        P = MV.fit_probe(Z, Y)
        np.savez(D / 'probe.npz', **{k: np.asarray(v) for k, v in P.items()})
        (D / 'train.json').write_text(json.dumps({'frames': stats, 'n_tokens': int(len(Y)), 'n_pos': int(Y.sum())}))
        log(f'{kind}: probe B on {len(stats)} frames, {len(Y)} tokens ({int(Y.sum())} pos) {time.time() - t0:.0f}s')
        del sam, proc
        torch.cuda.empty_cache()
    if a.part == 'train':
        return
    PB, PT = load_probe(D / 'probe.npz'), load_probe(MICE_PROBE)
    Pt = {k: (torch.tensor(np.asarray(v), dtype=torch.float32, device=dev) if k != 'b' else v) for k, v in PT.items()}
    Pb = {k: (torch.tensor(np.asarray(v), dtype=torch.float32, device=dev) if k != 'b' else v) for k, v in PB.items()}

    def prob(P, z):
        return torch.sigmoid(((z.float() - P['mu']) / P['sd']) @ P['w'] + P['b'])
    # ---------------- held-out frames
    items = [(vi, it) for vi, it in enumerate(eval_items(kind)) if vi % a.nshard == a.shard]
    st = STEP[kind]
    tenc = []
    with ThreadPoolExecutor(4) as ex:
        def dec_one(it):
            vid, sp, ks, N, kps, extra = it[1]
            win = np.clip(np.asarray(ks)[:, None] + st * np.arange(-2, 3)[None], 0, N - 1)
            u, inv = np.unique(win, return_inverse=True)
            d = data.decode(vid, sp, u, N)
            d['inv'] = inv.reshape(win.shape)
            d['cpos'] = np.searchsorted(u, np.asarray(ks))
            return d
        for (vi, it), dec in zip(items, prefetch(ex, dec_one, items)):
            vid, sp, ks, N, kps, extra = it
            x = data.canvas(dec['x'], dec.get('W'), dec.get('H'))
            pc, pp, pooled = data.packs(vid, dec)
            xc = x[dec['cpos']]
            n2c, n2p = n2_masks(xc, pc, h), n2_masks(xc, pp, h)
            torch.cuda.synchronize()
            t1 = time.time()
            z = torch.cat([enc(x[i:i + EB]) for i in range(0, len(x), EB)])
            tenc.append((time.time() - t1) / len(x))
            pt_, pb_ = prob(Pt, z).cpu().numpy(), prob(Pb, z).cpu().numpy()
            ident = []
            if kind in ('cal', 'crim'):  # validate.py N2 identity (per-clip background)
                import torch as _t
                for i in range(len(ks)):
                    kk = kps[i] if kind == 'cal' else (kps[i], extra)
                    ident.append(V.score_frame(_t.from_numpy(n2c[i]['keep']), _t.from_numpy(n2c[i]['pix']), kk, S, G, h, kind))
            np.savez_compressed(
                D / f'eval_{vi:03d}.npz', vid=vid, ks=np.asarray(ks), inv=dec['inv'], pT=pt_.astype(np.float16),
                pB=pb_.astype(np.float16), n2c=np.stack([q['keep'] for q in n2c]), n2p=np.stack([q['keep'] for q in n2p]),
                n2pix=np.packbits(np.stack([q['pix'] for q in n2p]), -1), img=xc.permute(0, 2, 3, 1).cpu().numpy(),
                kps=np.stack([np.asarray(k, np.float32) for k in kps]), strain=str(extra), pooled=pooled,
                ident=json.dumps(ident))
            del x, z
            if vi % 10 == 0:
                log(f'eval video {vi} ({len(items)} in shard) {time.time() - t0:.0f}s')
    (D / f'run_{a.shard}.json').write_text(json.dumps({'enc_s_per_frame_median': float(np.median(tenc)), 'gpu': torch.cuda.get_device_name(0),
                                           'h': h, 'S': S, 'wall_s': time.time() - t0}))
    log(f'{kind}: done {time.time() - t0:.0f}s, encode {np.median(tenc):.4f} s/frame')


# ------------------------------------------------------------------------------------------------ score (CPU)
PARTS = {'cal': ['nose', 'ear_l', 'ear_r', 'neck', 'hip_l', 'hip_r', 'tail_base'],
         'mab': ['nose', 'ear_l', 'ear_r', 'neck', 'paw_fl', 'paw_fr', 'center', 'paw_hl', 'paw_hr', 'tail_base',
                 'tail_mid', 'tail_tip']}


def region_tokens(kind, kps, S, G, h):
    """(G*G,) bool: token contains a pixel of the animal region (prereg)."""
    import cv2
    m = np.zeros((S, S), np.uint8)
    if kind == 'crim':
        for p in kps:
            cv2.circle(m, (int(round(p[0])), int(round(p[1]))), int(round(6 * h)), 1, -1)
    else:
        M = int(round(1.5 * h))
        for k in kps:
            ok = np.isfinite(k).all(1)
            hull = np.zeros((S, S), np.uint8)
            if ok.sum() >= 3:
                cv2.fillConvexPoly(hull, cv2.convexHull(k[ok].round().astype(np.int32)), 1)
            for p in k[ok]:
                cv2.circle(hull, (int(round(p[0])), int(round(p[1]))), 1, 1, -1)
            hull = cv2.dilate(hull, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * M + 1, 2 * M + 1)))
            m |= hull
            if kind == 'cal' and ok[0] and ok[6]:
                cv2.circle(m, (int(round(k[6][0])), int(round(k[6][1]))), int(round(np.linalg.norm(k[0] - k[6]))), 1, -1)
    return m.reshape(G, S // G, G, S // G).max((1, 3)).ravel() > 0


def tok_of(p, G, S):
    p = np.clip(p, 0, S - 1e-3)
    return (p[..., 1] // (S / G)).astype(int) * G + (p[..., 0] // (S / G)).astype(int)


def frame_metrics(kind, keep, kps, strain, S, G, h, reg):
    import cv2
    o = {}
    K = keep.reshape(G, G)
    if kind == 'crim':
        c = tok_of(np.asarray(kps), G, S)
        nb = [bool(K[max(0, x // G - 1):x // G + 2, max(0, x % G - 1):x % G + 2].any()) for x in c]
        o['kp/res_point'] = [float(keep[c[0]])]
        o['kp/res_point_tol1'] = [float(nb[0])]
        o[f'kp/int_{strain}_point'] = [float(keep[c[1]])]
        o[f'kp/int_{strain}_point_tol1'] = [float(nb[1])]
        o['all_kp'] = [float(keep[c[0]]), float(keep[c[1]])]
        o['miss/resident'] = [float(not nb[0])]
        o[f'miss/intruder_{strain}'] = [float(not nb[1])]
        o['miss/all'] = [float(not nb[0]), float(not nb[1])]
    else:
        names = PARTS[kind]
        o['all_kp'], o['miss/all'] = [], []
        for ai, k in enumerate(kps):
            ok = np.isfinite(k).all(1)
            hit = keep[tok_of(np.where(ok[:, None], k, 0), G, S)] & ok
            an = ('resident' if ai == 0 else 'intruder_white') if kind == 'cal' else 'mouse'
            pre = ('res_' if ai == 0 else 'int_') if kind == 'cal' else ''
            for j, nm in enumerate(names):
                if ok[j]:
                    o.setdefault(f'kp/{pre}{nm}', []).append(float(hit[j]))
                    o['all_kp'].append(float(hit[j]))
            o.setdefault(f'miss/{an}', []).append(float(not hit.any()))
            o['miss/all'].append(float(not hit.any()))
    bg = keep & ~reg
    n, lab = cv2.connectedComponents(K.astype(np.uint8), connectivity=8)
    R = reg.reshape(G, G)
    o['bg_tok'] = [float(bg.sum())]
    o['iso_bg_comp'] = [float(sum(1 for c in range(1, n) if not R[lab == c].any()))]
    o['tok'] = [float(keep.sum())]
    return o


def agg(rows):
    acc = {}
    for r in rows:
        for k, v in r.items():
            acc.setdefault(k, []).extend(v)
    return {k: float(np.mean(v)) for k, v in acc.items()} | {'n_kp': len(acc['all_kp'])}


def cmd_score(a):
    import cv2
    import mask5_val as MV
    kind = a.set
    D = BEE / kind
    runs = [json.loads(f.read_text()) for f in sorted(D.glob('run_*.json'))]
    run = {'S': runs[0]['S'], 'h': runs[0]['h'], 'gpu': sorted({r['gpu'] for r in runs}),
           'enc_s_per_frame_median': float(np.median([r['enc_s_per_frame_median'] for r in runs]))}
    S, h = run['S'], run['h']
    G = S // 16
    pp = lambda p: MV.postproc(p.astype(np.float32), G, CFG['t'], CFG['t_core'], None, CFG['smooth'])  # noqa: E731
    rows = {k: [] for k in ('N2_pool', 'N2_clip', 'T', 'B')}
    ident, uni = [], {'T': [], 'B': [], 'T_dil': [], 'B_dil': [], 'T_t': [], 'B_t': []}
    per_frame = []
    files = sorted(D.glob('eval_*.npz'))
    for f in files:
        z = np.load(f)
        strain = str(z['strain'])
        ident += json.loads(str(z['ident']))
        for i in range(len(z['ks'])):
            kps = z['kps'][i]
            reg = region_tokens(kind, kps, S, G, h)
            ms = {'N2_pool': z['n2p'][i].astype(bool), 'N2_clip': z['n2c'][i].astype(bool)}
            for arm, key in (('T', 'pT'), ('B', 'pB')):
                win = [pp(z[key][j]) for j in z['inv'][i]]
                ms[arm] = win[2]
                U = np.any(win, 0)
                uni[arm].append(U.sum())
                uni[arm + '_t'].append(win[2].sum())
                uni[arm + '_dil'].append((cv2.dilate(U.reshape(G, G).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0).sum())
            fm = {}
            for arm, keep in ms.items():
                m = frame_metrics(kind, keep, kps, strain, S, G, h, reg)
                rows[arm].append(m)
                fm[arm] = {'kp_miss': float(len(m['all_kp']) - sum(m['all_kp'])), 'animal_miss': float(sum(m['miss/all'])),
                           'bg': m['bg_tok'][0]}
            per_frame.append({'file': f.name, 'i': i, **fm})
    res = {'set': kind, 'bench': NAME[kind], 'S': S, 'G': G, 'h': h, 'cfg': CFG, 'n_frames': len(per_frame),
           'n_videos': len(files), 'arms': {arm: agg(R) for arm, R in rows.items()},
           'enc_s_per_frame_median': run['enc_s_per_frame_median'], 'gpu': run['gpu']}
    tr = json.loads((D / 'train.json').read_text())
    res['train'] = {'frames': len(tr['frames']), 'tokens': tr['n_tokens'], 'pos': tr['n_pos'],
                    'videos': len({x['vid'] for x in tr['frames']}),
                    'mean_label_tok': float(np.mean([x['lab_tok'] for x in tr['frames']])),
                    'mean_n2_tok': float(np.mean([x['n2_tok'] for x in tr['frames']]))}
    if ident:
        ref = json.loads((OUT4 / f'bench_{kind}.json').read_text())['N2']
        mine = {k: float(np.mean([r[k] for r in ident if k in r])) for k in ref if k not in ('tokens_per_frame', 'leak')}
        res['identity_N2_clip_vs_validate_max_abs_diff'] = max(abs(mine[k] - ref[k]) for k in mine)
    A = res['arms']
    T, B, N = A['T'], A['B'], A['N2_pool']
    chooseB = bool(B['all_kp'] > T['all_kp'] and B['miss/all'] <= T['miss/all'] and B['bg_tok'] <= T['bg_tok'] + max(1, 0.1 * T['bg_tok']))
    ch = 'B' if chooseB else 'T'
    C = A[ch]
    res['chosen'] = ch
    res['pass'] = bool(C['all_kp'] > N['all_kp'] and C['miss/all'] <= N['miss/all'] and C['bg_tok'] <= N['bg_tok'])
    sz = json.loads((REPO / 'results/vision/eci_v2_0B/bench_store/size.json').read_text())['bench'][NAME[kind]]
    res['store'] = {}
    for arm in ('T', 'B'):
        ut, ud = float(np.mean(uni[arm])), float(np.mean(uni[arm + '_dil']))
        res['store'][arm] = {'tok_at_t': float(np.mean(uni[arm + '_t'])), 'union5_tok': ut, 'union5_over_t': ut / max(np.mean(uni[arm + '_t']), 1e-9),
                             'union5_dil1_tok': ud, 'frames_5fps': sz['frames_5fps'],
                             'TB_union5': ut * 2048 * sz['frames_5fps'] / 1e12, 'TB_union5_dil1': ud * 2048 * sz['frames_5fps'] / 1e12,
                             'TB_all_patches': sz['TB']}
    res['worst'] = {arm: sorted(per_frame, key=lambda r: (-r[arm]['animal_miss'], -r[arm]['kp_miss']))[:3] for arm in ('T', 'B')}
    (OUT / f'bench_{kind}_v5.json').write_text(json.dumps(res, indent=1))
    log(json.dumps({k: v for k, v in res.items() if k not in ('worst',)}, indent=1))
    # table
    keys = sorted(A['T'])
    order = ['all_kp'] + [k for k in keys if k.startswith('kp/')] + [k for k in keys if k.startswith('miss/')] + ['bg_tok', 'iso_bg_comp', 'tok']
    L = [f'## {NAME[kind]} ({res["n_frames"]} held-out frames, {res["n_videos"]} videos; canvas {S}, h {h:.2f} px; '
         f'background margin 1.5 h = {1.5 * h:.1f} px{" + 6 h disk" if kind == "crim" else ""})', '',
         f'chosen arm: **{ch}**; pre-registered PASS vs N2 (pooled): **{res["pass"]}**', '',
         '| metric | N2 pooled bg | N2 per-clip bg | T (mice probe) | B (bench probe) |', '|---|---|---|---|---|']
    for k in order:
        L.append(f'| {k} | ' + ' | '.join(f'{A[arm].get(k, float("nan")):.3f}' for arm in ('N2_pool', 'N2_clip', 'T', 'B')) + ' |')
    L += ['', f'store (union t-2..t+2 at 5 fps, fp16 1024-d, {sz["frames_5fps"]:,} frames): ' +
          '; '.join(f'{arm} {res["store"][arm]["union5_tok"]:.1f} tok/frame = {res["store"][arm]["TB_union5"]:.3f} TB '
                    f'(dilated 1 token {res["store"][arm]["TB_union5_dil1"]:.3f} TB)' for arm in ('T', 'B')) +
          f'; all patches {sz["TB"]:.3f} TB', '']
    (OUT / f'bench_{kind}_v5.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


# ------------------------------------------------------------------------------------------------ sheet (CPU)
def cmd_sheet(a):
    import cv2
    import mask5_val as MV
    from PIL import Image, ImageDraw
    kind = a.set
    D = BEE / kind
    res = json.loads((OUT / f'bench_{kind}_v5.json').read_text())
    S, h, ch = res['S'], res['h'], res['chosen']
    G = S // 16
    files = sorted(D.glob('eval_*.npz'))
    g = np.random.default_rng(0)
    picks = [(files[j].name, int(g.integers(0, 8))) for j in g.choice(len(files), 6, replace=False)]
    picks += [(w['file'], w['i']) for w in res['worst'][ch][:2]]
    T = 448
    sh = Image.new('RGB', (5 * T, T * len(picks)))
    for r, (fn, i) in enumerate(picks):
        z = np.load(D / fn)
        i = min(i, len(z['ks']) - 1)
        im = z['img'][i].copy()
        kps = z['kps'][i]
        pp = lambda p: MV.postproc(p.astype(np.float32), G, res['cfg']['t'], res['cfg']['t_core'], None, False)  # noqa
        kT, kB, kN = pp(z['pT'][z['inv'][i][2]]), pp(z['pB'][z['inv'][i][2]]), z['n2p'][i].astype(bool)
        kc = kB if ch == 'B' else kT
        pts = np.asarray(kps).reshape(-1, 2)
        imk = im.copy()
        for p in pts:
            if np.isfinite(p).all():
                cv2.circle(imk, (int(p[0]), int(p[1])), max(2, S // 256), (255, 0, 255), -1)
        # zoom: MABe22 tail tip of mouse 0, CalMS21 white intruder nose, CRIM13 intruder point
        c = kps[0][11] if kind == 'mab' else (kps[1][0] if kind == 'cal' else kps[1])
        side = int(min(S, max(96, 12 * h)))
        x0, y0 = int(np.clip(c[0] - side / 2, 0, S - side)), int(np.clip(c[1] - side / 2, 0, S - side))
        both = MV.grid_tint(imk, kN & kc, (255, 255, 255), 0.25)
        both = MV.grid_tint(both, kc & ~kN, (0, 220, 0), 0.5)
        both = MV.grid_tint(both, kN & ~kc, (255, 0, 0), 0.5)
        tiles = [(imk, f'{"worst " + ch if r >= 6 else "random"} | {kind} {z["vid"]} k{z["ks"][i]}'),
                 (MV.grid_tint(im, kN, (0, 90, 255)), f'N2 pooled {int(kN.sum())} tok'),
                 (MV.grid_tint(im, kT, (0, 220, 0)), f'T mice probe {int(kT.sum())} tok'),
                 (MV.grid_tint(im, kB, (0, 220, 0)), f'B bench probe {int(kB.sum())} tok'),
                 (both[y0:y0 + side, x0:x0 + side], f'zoom {ch} vs N2: green {ch} only, red N2 only, white both')]
        for j, (t, lab) in enumerate(tiles):
            tt = Image.fromarray(t).resize((T, T), Image.NEAREST if j == 4 else Image.BILINEAR)
            dr = ImageDraw.Draw(tt)
            dr.rectangle([0, 0, T, 15], fill=(0, 0, 0))
            dr.text((4, 2), lab[:74], fill=(255, 255, 0))
            sh.paste(tt, (j * T, r * T))
    p = OUT / f'bench_{NAME[kind]}.png'
    sh.save(p)
    log(f'{p}: {len(picks)} rows')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('step')
    ap.add_argument('--set', default='cal', choices=('cal', 'mab', 'crim'))
    ap.add_argument('--part', default='all', choices=('all', 'train', 'eval'))
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshard', type=int, default=1)
    a = ap.parse_args()
    globals()['cmd_' + a.step](a)


if __name__ == '__main__':
    main()
