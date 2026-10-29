"""
ECI v2.0 design-benchmark store (CRIM13, CalMS21, MABe22): size, and the POOLED-BACKGROUND mask fix.

Encoder (frozen v2.0, unchanged): src/eci/v2/encode.Encoder (DINOv3-L/16 block 20, rev ea8dc28, bf16, fp16 tokens) at
each benchmark's v2 side S (scripts/eci/v2_temporal/temporal.py BENCHES: MABe22 512, CalMS21 1024, CRIM13 640).
5 fps grid = source frames k with k % B.step == 0, 0 <= k < N. temporal.py and scripts/eci/v2_mask2/ are imported
read-only (other studies).

STEP 1 (2026-10-29): size of a store of ALL patches = 5.101 TB > 5 TB -> not built (size.json). Scope changed by the
caller: fix the benchmark mask first (below); the store (one-token-dilated superset of the fixed mask + per-token flags,
like the lab store) waits for a go-ahead.

==================================================================================================================
PRE-REGISTRATION: POOLED BACKGROUND (written 2026-10-29 before any pooled background, grouping or score existed)
Problem: an animal that stays still for most of a short clip is part of the per-clip median (or quantile) background,
so the 'changed from background' / 'darker than background' cues lose it.
POOLED BACKGROUND = per-pixel statistics over frames sampled from MANY clips of one camera / arena SETUP.
  frames    per clip the 48 validation background frames (linspace(0, N - 1, 48), scripts/eci/v2_mask2/validate.py
            N_BG); the pooled SAMPLE of a clip = 8 of them (every 6th: 0, 6, ..., 42), the per-clip median = all 48.
  SETUP     no benchmark has camera / arena metadata except the CRIM13 frame size (640 x 480 vs 656 x 490 = two camera
            configurations). Rule: hard split by native frame size; within it, cluster the per-clip median
            backgrounds: grey (PIL weights) of the per-clip median frame on the native video area (CalMS21: the 570
            video rows, no pad), area-averaged to a 32 x 32 grid; distance(a, b) = median over the 1024 cells of
            |a - b| (grey levels; robust to the animals / small shifts); COMPLETE linkage, cut at 10 grey levels (a
            quarter of the dark rule's 40-level drop). A group with >= 8 clips is a setup; clips in smaller groups
            keep their per-clip background (counted and reported).
  POOL      per setup: the 8-frame samples of at most 64 clips of the setup (rng 0 choice when more) -> <= 512 frames,
            converted to each rule's canvas exactly as the frames it masks. The clip being masked may be among them.
  RULES recomputed with the pooled frames instead of the clip's own background frames (same reducers as each rule):
    M0  validate.py frozen dark cue: grey < 60 AND q98 - grey > 40, q98 = per-pixel 0.98 grey quantile -> pooled q98.
    N2  src/eci/v2/mask2 thin-part rule (r25t) with background_pack (median, MAD, q02, q98, ghost) -> pack of the
        pooled frames; core = the M0 pixels with the pooled q98. h = validate's per-set body half-width computed from
        the PER-CLIP packs (unchanged between backgrounds, so only the background differs).
    T   the temporal-study tight flag (= the store's flag (a), temporal.py B.fd >= 0.25): MABe22 native 224, kthvalue 29
        of 30 frames (= the 29/30 quantile) with the arena-floor waiver (quad fitted on it) and 7 px cells (32 grid);
        CalMS21 / CRIM13 per-pixel median grey of 32 frames (+ CRIM13 padding cells). Pooled: the same order statistic
        (rank round(29/30 n)) / median over the pooled frames; MABe22 quad refitted on the pooled quantile.
        Deviation (stated): T is applied to validate.py's decoded frames (CalMS21 accurate-seek ffmpeg pad; CRIM13 / MABe22
        the same decoders as temporal.py).
    N2s MABe22 only: N2 at the store side 512 (bicubic resize of the 224 frames as temporal.MABe22.canvas), h scaled by
        512 / 896, per-clip pack of the 48 background frames at 512; scored on the 32 grid.
  CHECKS (validate.py score_frame on validate.py's frames, every arm with the per-clip AND the pooled background):
    MABe22 tail_seg_tok (+ tail_base_tok, body_tok); CalMS21 pale_tok, pale_missed (+ body_tok, tail_base_tok_*);
    CRIM13 pale_tok, pale_tok_tol1 (+ res_tok, c57_tok); leak and tokens / frame for all.
    KEPT-TOKEN CHANGE pooled vs per-clip per arm: added = |pool & ~clip| / |clip|, removed = |clip & ~pool| / |clip|
    (summed over the scored frames). IDENTITY: the per-clip M0 / N1 / N2 numbers must equal validate.py's
    results/vision/eci_v2_mask2/validate/{mab,cal,crim}_final.json.
  No adoption rule is pre-registered here (the caller decides from the numbers).
==================================================================================================================

Steps (scripts/eci/v2/v20_bench_store.sh in Slurm):
    size                          (CPU) -> OUT/size.json
    bgs   --bench B --shard i --nshard n   (CPU) per-clip median (48 frames) + 8-frame sample -> BG/<B>/s<i>.npz
    group --bench B               (CPU) setups + pooled frame stacks -> BG/<B>/groups.json, BG/<B>/pool_g<j>.npy
    check --bench B               (GPU) masks with both backgrounds + validate checks -> OUT/check_<B>.json
"""
import argparse
import json
import math
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts/eci/v2_temporal'))
OUT = REPO / 'results/vision/eci_v2_0B/bench_store'
TOKROOT = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_tokens/v2_full')
BG = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_bench_bg')
BYTES_PER_TOKEN = 1024 * 2  # fp16 x 1024
LIMIT_TB = 5.0
N_BG, SAMPLE_EVERY, CUT, MIN_GROUP, MAX_CLIPS, GRID_C = 48, 6, 10.0, 8, 64, 32
VSET = {'mabe22': 'mab', 'calms21': 'cal', 'crim13': 'crim'}
log = lambda s: print(time.strftime('%H:%M:%S'), s, flush=True)  # noqa: E731


def local_dir():
    d = Path(os.environ.get('LOCAL_DIR') or '')
    assert str(d).startswith('/localhome') or str(d).startswith(os.environ.get('TMPDIR') or '/nonexistent'), d
    d.mkdir(parents=True, exist_ok=True)
    return d


def video_universe(name):
    """-> (bench, list of (video, set / split, N source frames)) for ALL videos / clips of the benchmark."""
    import temporal as T
    B = T.BENCHES[name]()
    if name == 'mabe22':
        EC = B.EC
        out = []
        for st in ('mouse_userTrain', 'mouse_submission'):
            clips = sorted(p.stem for p in (EC.VID / st).iterdir() if p.suffix == '.avi')
            out += [(c, st, 1801) for c in clips if c not in EC.EXCLUDE]
        return B, out
    if name == 'calms21':
        return B, [(q, sp, n) for q, (sp, n) in sorted(B.lens.items())]
    inv = json.loads(B.AC.INV.read_text())['videos']
    return B, [(v['name'], v['split'], int(v['frames'])) for v in sorted(inv, key=lambda v: v['name'])]


def cmd_size(a):
    res = {'unit': 'bytes; TB = 1e12 bytes, TiB = 2^40', 'bytes_per_token': BYTES_PER_TOKEN, 'limit_TB': LIMIT_TB,
           'rule_5fps': 'k % step == 0, 0 <= k < N', 'bench': {}}
    tot = 0
    for name in ('crim13', 'calms21', 'mabe22'):
        B, U = video_universe(name)
        G = B.S // 16
        fr = int(sum(math.ceil(n / B.step) for _, _, n in U))
        by = fr * G * G * BYTES_PER_TOKEN
        sets = {}
        for _, s, n in U:
            sets.setdefault(s, [0, 0])
            sets[s][0] += 1
            sets[s][1] += math.ceil(n / B.step)
        res['bench'][name] = {'S': B.S, 'G': G, 'step': B.step, 'videos': len(U), 'source_frames': int(sum(n for *_, n in U)),
                              'frames_5fps': fr, 'tokens': fr * G * G, 'bytes': by, 'TB': by / 1e12, 'TiB': by / 2**40,
                              'per_set': {k: {'videos': v[0], 'frames_5fps': v[1]} for k, v in sets.items()},
                              'flag_bytes_per_flag_file': fr * G * G // 8}
        tot += by
        log(f'{name}: {len(U)} videos, {fr:,} 5 fps frames, G {G}, {by / 1e12:.3f} TB')
    res['total_bytes'] = tot
    res['total_TB'] = tot / 1e12
    res['total_TiB'] = tot / 2**40
    res['above_limit'] = bool(tot / 1e12 > LIMIT_TB)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'size.json').write_text(json.dumps(res, indent=1))
    log(f'TOTAL {tot / 1e12:.3f} TB ({tot / 2**40:.3f} TiB); above {LIMIT_TB} TB: {res["above_limit"]}')


# ================================================================================================ pooled background
def v2m2():
    p = str(REPO / 'scripts/eci/v2_mask2')
    if p not in sys.path:
        sys.path.insert(0, p)
    p = str(REPO / 'scripts/benchmark/crim13')
    if p not in sys.path:
        sys.path.insert(0, p)
    import common as CM
    return CM


def crim_inv():
    I = json.loads((REPO / 'results/vision/benchmark_crim13/inventory/inventory.json').read_text())['videos']
    return {v['name']: v for v in I}


def bg_frames_native(name, vid, split, N):
    """the 48 background frames of one clip in the form each check decodes them: MABe22 (48, 3, 224, 224) RGB;
    CalMS21 (48, 3, 1024, 1024) padded canvas (validate cal_decode); CRIM13 (48, H, W) grey native."""
    CM = v2m2()
    bgi = np.linspace(0, N - 1, N_BG).round().astype(int)
    if name == 'mabe22':
        a = CM.mab_decode(split, vid)
        return a[bgi]
    if name == 'calms21':
        return CM.cal_decode(vid, split, bgi, threads=2)
    from align_common import ROOT, load_seek, read_frames
    v = crim_inv()[vid]
    d = ROOT / v['dir']
    offs = load_seek(d / f'{vid}_t-seek.mat')
    fr, _ = read_frames(d / f'{vid}_t.seq', offs, set(bgi.tolist()), threads=2)
    return np.stack([fr[i] for i in bgi])


def cmd_bgs(a):
    _, U = video_universe(a.bench)
    U = [u for j, u in enumerate(U) if j % a.nshard == a.shard]
    loc = local_dir() / f'bgs_{a.bench}_s{a.shard}.npz'
    t0 = time.time()

    def one(u):
        vid, sp, N = u
        x = bg_frames_native(a.bench, vid, sp, N)
        # lower median (as mask2.median_background) for an even count
        med = np.sort(x, axis=0)[(len(x) - 1) // 2]
        return vid, sp, med.astype(np.uint8), x[::SAMPLE_EVERY].copy()

    arrs, meta = {}, []
    with ThreadPoolExecutor(a.workers) as ex:
        for k, (vid, sp, med, smp) in enumerate(ex.map(one, U)):
            arrs[f'med__{vid}'] = med
            arrs[f'smp__{vid}'] = smp
            meta.append({'video': vid, 'set': sp, 'shape': list(med.shape)})
            if k % 50 == 0:
                log(f'{k + 1}/{len(U)} {time.time() - t0:.0f}s')
    np.savez(loc, **arrs)
    (BG / a.bench).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(loc, BG / a.bench / f'.s{a.shard}.npz.tmp')
    os.replace(BG / a.bench / f'.s{a.shard}.npz.tmp', BG / a.bench / f's{a.shard}.npz')
    (BG / a.bench / f's{a.shard}.json').write_text(json.dumps({'videos': meta, 'wall_s': time.time() - t0}, indent=1))
    loc.unlink()
    log(f'bgs {a.bench} s{a.shard}: {len(U)} clips, {time.time() - t0:.0f}s')


def grey_np(x):
    """(..., 3, H, W) uint8 RGB or (..., H, W) grey -> float grey (PIL 'L' weights, as src/eci/v2/mask.grey_of)."""
    if x.ndim >= 3 and x.shape[-3] == 3:
        t = x.astype(np.float64)
        return np.floor((t[..., 0, :, :] * 19595 + t[..., 1, :, :] * 38470 + t[..., 2, :, :] * 7471 + 32768) / 65536)
    return x.astype(np.float64)


def small_grey(name, med):
    g = grey_np(med)
    if name == 'calms21':
        g = g[227:227 + 570]
    H, W = g.shape
    ys = np.linspace(0, H, GRID_C + 1).round().astype(int)
    xs = np.linspace(0, W, GRID_C + 1).round().astype(int)
    return np.array([[g[ys[i]:ys[i + 1], xs[j]:xs[j + 1]].mean() for j in range(GRID_C)] for i in range(GRID_C)])


def load_bgs(bench):
    meta, med, smp = [], {}, {}
    for jf in sorted((BG / bench).glob('s*.json')):
        z = np.load(jf.with_suffix('.npz'))
        for m in json.loads(jf.read_text())['videos']:
            meta.append(m)
            med[m['video']] = z[f'med__{m["video"]}']
            smp[m['video']] = z[f'smp__{m["video"]}']
    return meta, med, smp


def cmd_group(a):
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    meta, med, smp = load_bgs(a.bench)
    _, U = video_universe(a.bench)
    assert sorted(m['video'] for m in meta) == sorted(u[0] for u in U), 'bgs shards incomplete'
    vids = sorted(med)
    shape = {v: tuple(med[v].shape) for v in vids}
    groups, gid, info = {}, 0, {'rule': f'split by native frame size; complete linkage on median |grey diff| of '
                                       f'{GRID_C}x{GRID_C} per-clip medians, cut {CUT}; setup iff >= {MIN_GROUP} clips',
                               'by_shape': {}}
    rng = np.random.default_rng(0)
    for shp in sorted(set(shape.values())):
        vs = [v for v in vids if shape[v] == shp]
        X = np.stack([small_grey(a.bench, med[v]).ravel() for v in vs])
        Dm = np.zeros((len(vs), len(vs)))
        for i in range(len(vs)):
            Dm[i] = np.median(np.abs(X - X[i]), axis=1)
        Dm = (Dm + Dm.T) / 2
        np.fill_diagonal(Dm, 0)
        lab = fcluster(linkage(squareform(Dm, checks=False), 'complete'), CUT, 'distance') if len(vs) > 1 else np.ones(1, int)
        iu = np.triu_indices(len(vs), 1)
        info['by_shape'][str(shp)] = {'clips': len(vs), 'clusters': int(lab.max()),
                                      'dist_quantiles_5_25_50_75_95': np.quantile(Dm[iu], [.05, .25, .5, .75, .95]).tolist()
                                      if len(vs) > 1 else []}
        for c in np.unique(lab):
            mem = [vs[i] for i in np.flatnonzero(lab == c)]
            if len(mem) < MIN_GROUP:
                for v in mem:
                    groups[v] = None
                continue
            pick = sorted(rng.choice(mem, MAX_CLIPS, replace=False).tolist()) if len(mem) > MAX_CLIPS else mem
            st = np.concatenate([smp[v] for v in pick])
            np.save(BG / a.bench / f'pool_g{gid}.npy', st)
            ii = [vs.index(v) for v in mem]
            info.setdefault('groups', {})[str(gid)] = {
                'shape': list(shp), 'clips': len(mem), 'pooled_clips': len(pick), 'pooled_frames': int(len(st)),
                'max_within_dist': float(Dm[np.ix_(ii, ii)].max())}
            for v in mem:
                groups[v] = gid
            gid += 1
    info['n_clips'] = len(vids)
    info['n_clips_pooled'] = int(sum(g is not None for g in groups.values()))
    info['n_clips_unpooled'] = int(sum(g is None for g in groups.values()))
    info['clip_group'] = groups
    (BG / a.bench / 'groups.json').write_text(json.dumps(info, indent=1))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'groups_{a.bench}.json').write_text(json.dumps({k: v for k, v in info.items() if k != 'clip_group'},
                                                           indent=1))
    log(json.dumps({k: v for k, v in info.items() if k != 'clip_group'}))


# ------------------------------------------------------------------------------------------------ check (GPU)
class TRule:
    """the temporal-study tight flag on the check frames (temporal.py MABe22.fd / Bench.fd >= 0.25)."""

    def __init__(self, bench, dev):
        import temporal as T
        self.T, self.bench, self.dev = T, bench, dev

    def prep(self, frames_bg, W=None, H=None, pooled=False):
        """frames_bg: MABe22 (n, 3, 224, 224) RGB native; CalMS21 (n, 3, 1024, 1024) canvas; CRIM13 (n, H, W) grey."""
        import torch
        T = self.T
        if self.bench == 'mabe22':
            sys.path.insert(0, str(REPO / 'scripts/benchmark/mabe22'))
            import eci_encode as ME
            g = np.stack([T.MABe22._grey(f.transpose(1, 2, 0)) for f in frames_bg])
            gt = torch.from_numpy(g).to(self.dev).float()
            kth = int(round(29 / 30 * len(g))) if pooled else 29
            assert pooled or len(g) == 30
            pix = gt.kthvalue(kth, dim=0).values
            quad = ME.arena_quad(pix.cpu().numpy() > 100)
            self.pix, self.hole, self.quad_ok = pix, torch.from_numpy(quad['inside']).to(self.dev), bool(quad['ok'])
            return
        x = self.canvas(frames_bg, W, H)
        g = T.grey_t(x)
        self.bg = g.float().median(0).values
        self.cells = None
        if self.bench == 'crim13':
            from align_common import video_cells
            self.cells = torch.from_numpy(video_cells(W, H).astype(np.float32)).to(self.dev)

    def canvas(self, fr, W=None, H=None):
        import torch
        if self.bench == 'crim13':
            from align_common import canvas_gpu
            return canvas_gpu(list(fr), W, H, self.dev).to(torch.uint8)
        return torch.as_tensor(fr).to(self.dev)

    def keep(self, frames, W=None, H=None):
        """frames as prep (MABe22 native RGB) -> (B, G*G) bool, pixel map."""
        import torch
        import torch.nn.functional as F
        T = self.T
        if self.bench == 'mabe22':
            gg = torch.from_numpy(np.stack([T.MABe22._grey(f.transpose(1, 2, 0)) for f in frames])).to(self.dev).float()
            dk = (gg < T.DARK_ABS) & (((self.pix[None] - gg) > T.DARK_REL) | self.hole[None])
            return F.avg_pool2d(dk.float()[:, None], 7).flatten(1) >= T.THR, dk
        x = self.canvas(frames, W, H)
        g = T.grey_t(x).float()
        px = (g < T.DARK_ABS) & ((self.bg[None] - g) > T.DARK_REL)
        return T.fraction(x, self.bg, self.cells) >= T.THR, px


def tbg_index(bench, N):
    return np.linspace(0, N - 1, 30 if bench == 'mabe22' else 32).round().astype(int)


def resize_to(x, S):
    import torch
    import torch.nn.functional as F
    return F.interpolate(x.float(), size=(S, S), mode='bicubic', align_corners=False).round().clamp(0, 255).to(torch.uint8)


def cmd_check(a):
    import torch
    import torch.nn.functional as F
    CM = v2m2()
    import validate as V
    from src.eci.v2 import mask2 as M2
    dev = torch.device('cuda')
    bench, kind = a.bench, VSET[a.bench]
    G_ = json.loads((BG / bench / 'groups.json').read_text())
    cg = G_['clip_group']
    t00 = time.time()
    # ---- frames exactly as validate.run
    if kind == 'cal':
        items = V.cal_videos('test', 24, 0)
        with ThreadPoolExecutor(6) as ex:
            loaded = list(ex.map(lambda it: V.load_cal(it, 'test'), items))
        S, G = 1024, 64
        vids = [(nm, torch.from_numpy(xb).to(dev), torch.from_numpy(x).to(dev), ks, kps) for nm, xb, x, ks, kps in loaded]
        Tn = {nm: T for nm, T, _, _ in items}
        extra = {}
        for nm, T, _, _ in items:
            extra[nm] = CM.cal_decode(nm, 'test', tbg_index(bench, T), threads=2)
        geo = {nm: (None, None) for nm, *_ in vids}
    elif kind == 'crim':
        from align_common import load_seek, read_frames
        items = V.crim_videos(12, 16, 0)
        with ThreadPoolExecutor(6) as ex:
            vids0 = list(ex.map(lambda it: V.load_crim(it, dev), items))
        vids = [(nm, xb, x, ks, [(xy[i], strain) for i in range(len(ks))]) for nm, strain, xb, x, ks, xy in vids0]
        S, G = 640, 40
        extra, geo, natx = {}, {}, {}
        for nm, strain, d, N, W, H, ks, pos in items:
            offs = load_seek(d / f'{nm}_t-seek.mat')
            tb = tbg_index(bench, N)
            fr, _ = read_frames(d / f'{nm}_t.seq', offs, set(tb.tolist()) | set(ks.tolist()), threads=2)
            extra[nm] = np.stack([fr[i] for i in tb])
            natx[nm] = np.stack([fr[i] for i in ks])
            geo[nm] = (W, H)
    else:
        items = V.mab_clips(120, 8, 0)
        vids, extra, natx, nat48 = [], {}, {}, {}
        for c, ks, kps in items:
            arr = CM.mab_decode('mouse_submission', c)
            ib = np.linspace(0, len(arr) - 1, N_BG).round().astype(int)
            vids.append((c, V.up4(arr[ib], dev), V.up4(arr[ks + 1], dev), ks, kps))
            extra[c] = arr[tbg_index(bench, len(arr))]
            natx[c] = arr[ks + 1]
            nat48[c] = arr[ib]
        S, G = 896, 56
        geo = {c: (None, None) for c, *_ in vids}
    log(f'{kind}: {len(vids)} videos loaded {time.time() - t00:.0f}s')
    packs = [M2.background_pack(xb) for _, xb, _, _, _ in vids]
    h = V.body_h([c.cpu().numpy() for (_, xb, _, _, _), pk in zip(vids, packs) for c in V.dark_cue(xb[::6], pk['q98'])])
    log(f'{kind}: h {h:.3f}')
    # ---- pooled stacks per group, converted lazily to each canvas
    pool_cache = {}

    def pooled(g, form):
        key = (g, form)
        if key in pool_cache:
            return pool_cache[key]
        pool_cache.clear() if len(pool_cache) > 6 else None
        st = np.load(BG / bench / f'pool_g{g}.npy')
        if form == 'val':  # validate canvas (the M0 / N2 masks)
            if kind == 'mab':
                x = torch.cat([V.up4(st[i:i + 64], dev) for i in range(0, len(st), 64)])
            elif kind == 'cal':
                x = torch.from_numpy(st).to(dev)
            else:
                Wd, Hd = G_['groups'][str(g)]['shape'][1], G_['groups'][str(g)]['shape'][0]
                from align_common import canvas_gpu
                x = torch.cat([canvas_gpu(list(st[i:i + 64]), Wd, Hd, dev).to(torch.uint8) for i in range(0, len(st), 64)])
            r = M2.background_pack(x)
            del x
        elif form == 's512':
            x = torch.cat([resize_to(torch.from_numpy(st[i:i + 64]).to(dev), 512) for i in range(0, len(st), 64)])
            r = M2.background_pack(x)
            del x
        else:  # raw frames for the T rule
            r = st
        pool_cache[key] = r
        torch.cuda.empty_cache()
        return r

    arms = ['M0', 'N1', 'N2', 'T'] + (['N2s'] if kind == 'mab' else [])
    rows = {(arm, b): [] for arm in arms for b in ('clip', 'pool')}
    chg = {arm: {'clip': 0, 'add': 0, 'rem': 0} for arm in arms}
    TR = TRule(bench, dev)
    h512 = h * 512 / 896
    n_pooled_vid = 0
    for (nm, xb, x, ks, kps), pk in zip(vids, packs):
        g = cg.get(nm)
        n_pooled_vid += int(g is not None)
        W, H = geo[nm]
        res = {}
        for b in ('clip', 'pool'):
            pkb = pk if (b == 'clip' or g is None) else pooled(g, 'val')
            out = {'M0': [], 'N1': [], 'N2': [], 'T': [], 'N2s': []}
            for b0 in range(0, len(x), 8):
                ms, _ = V.masks_all(x[b0:b0 + 8], pkb, h)
                for arm in ('M0', 'N1', 'N2'):
                    out[arm] += [(ms[arm][0][i], ms[arm][1][i]) for i in range(len(ms[arm][0]))]
            # T rule
            if kind == 'mab':
                TR.prep(extra[nm] if (b == 'clip' or g is None) else pooled(g, 'raw'), pooled=(b == 'pool' and g is not None))
                kt, pt = TR.keep(natx[nm])
            elif kind == 'cal':
                TR.prep(extra[nm] if (b == 'clip' or g is None) else pooled(g, 'raw'))
                kt, pt = TR.keep(x)
            else:
                TR.prep(extra[nm] if (b == 'clip' or g is None) else pooled(g, 'raw'), W, H)
                kt, pt = TR.keep(natx[nm], W, H)
            out['T'] = [(kt[i], pt[i]) for i in range(len(kt))]
            if kind == 'mab':  # N2 at the store side 512
                x5 = resize_to(torch.from_numpy(natx[nm]).to(dev), 512)
                if b == 'clip' or g is None:
                    p5 = M2.background_pack(resize_to(torch.from_numpy(nat48[nm]).to(dev), 512))
                else:
                    p5 = pooled(g, 's512')
                core0 = V.dark_cue(x5, p5['q98'])
                r = M2.cue(x5, p5, core0, h512)
                _, app = M2.appendage(r['pix'], h512)
                fr, na = M2.token_fractions(r['pix'], app)
                kk = M2.keep(fr, na, 'r25t')
                out['N2s'] = [(kk[i], r['pix'][i]) for i in range(len(kk))]
            res[b] = out
        for arm in arms:
            for i in range(len(ks)):
                kc, kp = res['clip'][arm][i][0], res['pool'][arm][i][0]
                chg[arm]['clip'] += int(kc.sum())
                chg[arm]['add'] += int((kp & ~kc).sum())
                chg[arm]['rem'] += int((kc & ~kp).sum())
                for b in ('clip', 'pool'):
                    kept, pix = res[b][arm][i]
                    if arm in ('M0', 'N1', 'N2'):
                        Sa, Ga, kq, ha = S, G, kps[i], h
                    elif arm == 'T':
                        if kind == 'mab':
                            Sa, Ga, ha = 224, 32, h / 4
                            kq = [k / 4 for k in kps[i]]
                        else:
                            Sa, Ga, kq, ha = S, G, kps[i], h
                    else:
                        Sa, Ga, ha = 512, 32, h512
                        kq = [k * 512 / 896 for k in kps[i]]
                    rows[(arm, b)].append({'video': nm, 'k': int(ks[i]), **V.score_frame(kept, pix, kq, Sa, Ga, ha, kind)})
    out = {'set': kind, 'bench': bench, 'S_validate': S, 'G_validate': G, 'h': h, 'n_videos': len(vids),
           'n_videos_pooled': n_pooled_vid, 'n_frames': len(rows[('M0', 'clip')]), 'wall_s': time.time() - t00,
           'gpu': torch.cuda.get_device_name(0), 'arms': {}}
    for arm in arms:
        out['arms'][arm] = {}
        for b in ('clip', 'pool'):
            R = rows[(arm, b)]
            keys = sorted({k for r in R for k in r if k not in ('video', 'k', 'tokens', 'leak_n')})
            m = {k: float(np.mean([r[k] for r in R if k in r])) for k in keys}
            m['tokens_per_frame'] = float(np.mean([r['tokens'] for r in R]))
            m['leak'] = float(sum(r['leak_n'] for r in R) / max(sum(r['tokens'] for r in R), 1))
            out['arms'][arm][b] = m
        c = chg[arm]
        out['arms'][arm]['change'] = {'kept_clip': c['clip'], 'added': c['add'], 'removed': c['rem'],
                                      'added_share': c['add'] / max(c['clip'], 1),
                                      'removed_share': c['rem'] / max(c['clip'], 1)}
        log(f'{kind} {arm}: ' + json.dumps(out['arms'][arm]))
    ref = json.loads((REPO / f'results/vision/eci_v2_mask2/validate/{kind}_final.json').read_text())
    out['identity_vs_validate'] = {arm: {k: out['arms'][arm]['clip'].get(k, float('nan')) - v for k, v in ref[arm].items()}
                                   for arm in ('M0', 'N1', 'N2')}
    out['identity_h_diff'] = h - ref['h']
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'check_{bench}.json').write_text(json.dumps(out, indent=1))
    log(f'identity: h diff {h - ref["h"]:.2e}; max |diff| ' +
        str(max(abs(d) for v in out['identity_vs_validate'].values() for d in v.values())))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=('size', 'bgs', 'group', 'check'))
    p.add_argument('--bench', default='crim13', choices=('crim13', 'calms21', 'mabe22'))
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--nshard', type=int, default=1)
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    {'size': cmd_size, 'bgs': cmd_bgs, 'group': cmd_group, 'check': cmd_check}[a.step](a)


if __name__ == '__main__':
    main()
