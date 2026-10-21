"""
ECI v2 MASK TEST (pre-registered): does a mask that keeps THIN body parts (mouse tails, ant antennae / legs, frog
limbs) beat the frozen v2 TIGHT mask (token kept iff >= 25% of its 16 x 16 patch are animal pixels)?

==================================================================================================================
PRE-REGISTRATION (written 2026-10-21, before any token of this test was encoded and before any score existed).

ARMS (identical encoder tokens: DINOv3-L/16 block 20 of the frozen v2 pipeline, src/eci/v2/pipeline.py Series, same
canonical frames, same per-domain pixel cue; only the token SELECTION differs; the encoder sees the whole frame, so
a token's value never depends on the mask)
    M0  v2 tight: animal-pixel fraction of the patch >= 0.25.
    M1  any animal pixel in the patch (> 0).
    M2  animal pixel mask dilated by a disk of radius r_dil PIXELS (4 native px converted to the encoder input side S:
        mice 1024 / 2064 -> 2 px, ants / frogs 4 px), then the per-patch >= 0.25 rule.
    M3  (reference, coverage / leak numbers only, no SAE) loose v1 / r4 style ring: 2 x 2 token cells (the old 32-type
        coarse grid) with animal fraction > 0.15 form the core, 1-cell (3 x 3) dilation, projected back to tokens.
SAE (every arm): the frozen v2 recipe (freeze.py cfg_of('F0', 2048) -> r3.train_arm): Matryoshka BatchTopK 2048
    (prefixes 128 / 512 / 2048), k 16, 30,000 steps x 4096 tokens, Adam 5e-4, theta* on calibration videos. One SAE per
    domain (ants v2 + v3 pooled). Seeds 60-65 (mice, ants; never used before), 60-62 (frogs). Deployment seed (sheets)
    = highest held-out FVE among seeds 60-62 (label-free, as v2).
FRAMES (fixed subsets of the freeze-round frame lists; the stored v2 tokens contain only M0 tokens, so every frame is
    re-encoded once and the union M1 | M2 is stored):
    mice v1   train pool = round-2 pool (nnnt r2), every second burst per video (8 of 16) -> ~31k frames;
              scored = round-2 diagnosis set, every second frame per video -> 43,200 frames (144 annotated videos).
    ants      pools v2 / v3 (freeze), every second burst per video; scored = the freeze scored frames, all
              (v2 13,200, v3 32,100).
    frogs juv pool_fj (freeze), every second burst per video (no labels; calib videos select, eval videos show).
THIN PARTS (label-free, from the animal pixel mask at the encoder input side S): body half-width h = median over
    connected components (>= 100 px) of the maximum Euclidean distance transform inside the component, on ~240 frames
    of training videos per domain ('calib' step, written to results/vision/eci_v2_mask/calib.json before encoding).
    Thin pixels = animal pixels removed by a morphological opening with a disk of radius r_open = round(body width / 4)
    = round(h / 2), i.e. parts narrower than about half a body width (tails, antennae, legs, limbs; also the narrowest
    tips of snouts and the corners of the outline).
METRICS (per domain and arm)
    coverage   thin coverage = thin pixels inside kept tokens / all thin pixels; animal-pixel recall likewise; tokens
               per frame; background leak = share of kept tokens with < 5% animal pixels.
    alignment  R~ (cross-fitted normalised R-precision, src/eci/v2/score.py, 10 random unit halves) and AUROC (cf);
               mice nose_nose / nose_tail on all frames and on the consistent annotators SG / SD / MR / CP; mice
               primary = mean(R~ nn, R~ nt); ants groom_any / groom_yellow / groom_blue (v2, v3) and on_lid_any (v3);
               ants primary = mean over v2 / v3 of R~(groom_any). Welch 95% CIs over seeds (6 vs 6).
    background neurons (as scripts/eci/diag_r4bg/diag.py H3, off-animal = < 5% animal pixels): for each neuron and
               each selection-half frame where it fires (frame max > 0), the class of its argmax token; background
               neuron = off-animal argmax in > 50% of >= 20 firing frames. Share of such neurons.
    thin-part neurons: thin share = sum_t a_t * thin_px_t / sum_t a_t * animal_px_t over all firing tokens of the
               selection-half frames; thin-part neuron = thin share > 0.5 with >= 20 firing frames. Contact sheets
               (24 top frames of the OTHER half, <= 1 per video, real per-token activation maps) of the top 3 thin
               neurons per domain (deployment seed of M1 and M2; of M0 too when it has any) -> results/vision/
               eci_v2_mask/sheets/. Halves: mice pools / ants videos of split rnd0; frogs calib (select) / eval (show).
DECISION RULE: adopt M1 or M2 over M0 only if ALL of
    (a) mice primary R~: Welch 95% CI lower bound of mean(arm) - mean(M0) >= -0.01;
    (b) ants primary R~: Welch 95% CI lower bound of mean(arm) - mean(M0) >= -0.01;
    (c) background leak (share of kept tokens with < 5% animal pixels) <= 0.10 in every domain (mice, ants v2, v3, frogs);
    (d) thin coverage >= M0's + 0.15 (absolute) in mice AND ants (v2 and v3).
    If both pass: the one with the larger (mice primary delta + ants primary delta); exact tie -> M2 (fewer tokens).
    Otherwise keep M0 and report which thin-part behaviours / neurons are lost. Background-neuron shares, AUROC, the
    consistent-annotator set, per-label numbers and frogs are reported and never decide.
==================================================================================================================

Steps (scripts/eci/v2_mask/mask.sh runs each in Slurm; staging in $LOCAL_DIR):
    calib                           (GPU) body widths -> OUT/calib.json
    prep --dom D --kind K --shard i --nshard n   (GPU) union M1|M2 token store shard -> BEE/<D>/<K>/s<i>/
    train --dom D --arm A [--seeds]  (GPU) SAEs -> BEE/<D>/sae/<A>_s<seed>, OUT/<D>/train_<A>.json
    score --dom D --arm A            (GPU) alignment -> OUT/<D>/score_<A>.json
    neurons --dom D --arm A          (GPU) background / thin neurons (+ sheets) -> OUT/<D>/neurons_<A>.json
    decide                           (CPU) -> OUT/decision.json, OUT/table.md
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
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts/eci'))
sys.path.insert(0, str(REPO / 'scripts/eci/nnnt'))
sys.path.insert(0, str(REPO / 'scripts/eci/v2'))
import freeze as FZ  # noqa: E402  (read-only: TokStore, labels, splits, sheets, train recipe)

N, R2, R3, R4 = FZ.N, FZ.R2, FZ.R3, FZ.R4
from src.eci.v2.score import crossfit, splits_of, welch  # noqa: E402

OUT = Path(os.environ.get('V2M_OUT') or REPO / 'results/vision/eci_v2_mask')
BEE = Path(os.environ.get('V2M_BEE') or '/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_mask')
B0 = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes')
log = N.log
D = 1024
ARMS = ('M0', 'M1', 'M2')
SEEDS = {'mice': tuple(range(60, 66)), 'ants': tuple(range(60, 66)), 'frogs': (60, 61, 62)}
DEPLOY_FROM = (60, 61, 62)
LEAK, THIN_NEURON, MIN_FIRE = 0.05, 0.5, 20
CFG = {d: json.loads((REPO / f'configs/eci_v2/{d}.json').read_text()) for d in ('mice', 'ants', 'frogs')}
KINDS = {
    'mice': {'pool': dict(series='v1', ref=B0 / 'nnnt/r2/pool', sub='half_bursts'),
             'diag': dict(series='v1', ref=B0 / 'nnnt/r2/diag', sub='every2')},
    'ants': {'pool_v2': dict(series='v2', ref=B0 / 'v2/freeze/ants/pool_v2', sub='half_bursts'),
             'pool_v3': dict(series='v3', ref=B0 / 'v2/freeze/ants/pool_v3', sub='half_bursts'),
             'scored_v2': dict(series='v2', ref=B0 / 'v2/freeze/ants/scored_v2', sub='all'),
             'scored_v3': dict(series='v3', ref=B0 / 'v2/freeze/ants/scored_v3', sub='all')},
    'frogs': {'pool_fj': dict(series='juv', ref=B0 / 'v2/freeze/frogs/pool_fj', sub='half_bursts')}}
POOLS = {'mice': ('pool',), 'ants': ('pool_v2', 'pool_v3'), 'frogs': ('pool_fj',)}


def series_cfg(dom, name):
    return next(s for s in CFG[dom]['series'] if s['name'] == name)


def dev_of():
    import torch
    return torch.device('cuda')


def gpu_line():
    import torch
    return f'GPU {torch.cuda.get_device_name(0)}, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB'


# ---------------------------------------------------------------------------------------------- frame subsets
def frame_list(dom, kind):
    K = KINDS[dom][kind]
    Fr = pd.read_parquet(K['ref'] / 'frames.parquet').reset_index(drop=True)
    Fr['ref_f'] = np.arange(len(Fr))
    if K['sub'] == 'all':
        return Fr
    keep = np.zeros(len(Fr), bool)
    for o, g in Fr.groupby('obs', sort=False):
        g = g.sort_values('k')
        if K['sub'] == 'every2':
            keep[g.index.values[::2]] = True
        else:
            bid = np.cumsum(np.r_[0, np.diff(g.k.values) > 2])
            keep[g.index.values[bid % 2 == 0]] = True
    return Fr[keep].reset_index(drop=True)


def runs_of(ks):
    """Sorted 5 fps frame indices of one video -> decode runs (k0, n, every, wanted ks)."""
    ks = np.sort(np.asarray(ks))
    out, sing = [], []
    grp = np.split(ks, np.flatnonzero(np.diff(ks) > 2) + 1)
    for g in grp:
        if len(g) >= 2:
            out.append((int(g[0]), int(g[-1] - g[0] + 1), 1, g))
        else:
            sing.append(int(g[0]))
    i = 0
    while i < len(sing):
        j = i + 1
        d = sing[j] - sing[i] if j < len(sing) else 1
        while j < len(sing) and sing[j] - sing[j - 1] == d and j - i < 60:
            j += 1
        g = np.array(sing[i:j])
        out.append((int(g[0]), len(g), int(d) if len(g) > 1 else 1, g))
        i = j
    return out


# ---------------------------------------------------------------------------------------------- morphology
def disk(r, dev):
    import torch
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return torch.from_numpy((x * x + y * y <= r * r).astype(np.float32))[None, None].to(dev)


def dilate(px, K):
    import torch.nn.functional as F
    r = K.shape[-1] // 2
    return F.conv2d(px.float()[:, None], K, padding=r)[:, 0] > 0.5


def erode(px, K):
    import torch.nn.functional as F
    r = K.shape[-1] // 2
    return F.conv2d(px.float()[:, None], K, padding=r)[:, 0] > float(K.sum()) - 0.5


def masks_of(px, G, Kd, Ko):
    """px (B, S, S) bool animal pixels -> per token (B, G*G): raw / dilated / thin fractions, M3 bool."""
    import torch.nn.functional as F
    thin = px & ~dilate(erode(px, Ko), Ko)
    pool = lambda m: F.avg_pool2d(m.float()[:, None], 16).flatten(1)  # noqa: E731
    raw, dil, thn = pool(px), pool(dilate(px, Kd)), pool(thin)
    B = px.shape[0]
    cell = F.avg_pool2d(raw.view(B, 1, G, G), 2) > 0.15
    cell = F.max_pool2d(cell.float(), 3, 1, 1) > 0
    m3 = F.interpolate(cell.float(), scale_factor=2, mode='nearest').flatten(1) > 0.5
    return raw, dil, thn, m3, thin


# ---------------------------------------------------------------------------------------------- calib
def cmd_calib(args):
    import torch
    from scipy import ndimage
    from src.eci.v2.pipeline import Series
    dev = dev_of()
    out = {}
    for dom, kind in (('mice', 'pool'), ('ants', 'pool_v2'), ('ants', 'pool_v3'), ('frogs', 'pool_fj')):
        K = KINDS[dom][kind]
        se = Series(series_cfg(dom, K['series']), dev, load_encoder=False)
        Fr = frame_list(dom, kind)
        Fr = Fr[Fr.role == 'train'] if 'role' in Fr else Fr
        g = np.random.default_rng(0)
        obs = g.choice(Fr.obs.unique(), min(24, Fr.obs.nunique()), replace=False)
        hmax, npx = [], []
        for o in obs:
            v = se.V[o]
            ks = np.sort(g.choice(Fr[Fr.obs == o].k.values, 10, replace=False))
            for k in ks:
                fr = se.run_job(**se.job(v, int(k), 1))
                x, _ = se.frames(v, fr, [int(k)])
                px = se.mask.pixels(x, v['src_lo'] + np.array([int(k)]))[0].cpu().numpy()
                npx.append(int(px.sum()))
                lab, n = ndimage.label(px)
                if n == 0:
                    continue
                edt = ndimage.distance_transform_edt(px)
                area = np.bincount(lab.ravel(), minlength=n + 1)
                mx = ndimage.maximum(edt, lab, np.arange(1, n + 1))
                hmax += [float(m) for m, a in zip(mx, area[1:]) if a >= 100]
        h = float(np.median(hmax))
        w_nat = float(se.videos[0]['probe']['w'])
        out[f'{dom}/{K["series"]}'] = {'S': se.S, 'native_w': w_nat, 'n_frames': len(npx), 'n_components': len(hmax),
                                       'body_halfwidth_px': h, 'body_width_px': 2 * h, 'r_open': max(1, int(round(h / 2))),
                                       'r_dil': max(1, int(round(4 * se.S / w_nat))),
                                       'animal_px_per_frame_median': float(np.median(npx))}
        log(f'calib {dom}/{K["series"]}: ' + json.dumps(out[f'{dom}/{K["series"]}']))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'calib.json').write_text(json.dumps(out, indent=1))


def calib_of(dom, series):
    c = json.loads((OUT / 'calib.json').read_text())
    return c[f'{dom}/{series}']


# ---------------------------------------------------------------------------------------------- prep
def mask_panels(items, path, G):
    """items [(x (3,S,S) uint8, {arm: kept (G*G,) bool}, thin (S,S) bool, title)] -> JPEG rows x (M0 M1 M2 M3)."""
    from PIL import Image, ImageDraw
    tile = 384
    sh = Image.new('RGB', (4 * tile, len(items) * tile))
    for r, (x, kept, thin, title) in enumerate(items):
        base = x.transpose(1, 2, 0).astype(np.float32)
        for c, arm in enumerate(('M0', 'M1', 'M2', 'M3')):
            im = base.copy()
            Kp = np.kron(kept[arm].reshape(G, G), np.ones((16, 16), bool))
            im[Kp] = 0.6 * im[Kp] + 0.4 * np.array([0, 255, 0])
            im[thin] = np.array([255, 0, 255])
            t = Image.fromarray(im.clip(0, 255).astype(np.uint8)).resize((tile, tile), Image.BILINEAR)
            ImageDraw.Draw(t).text((4, 4), f'{arm} {int(kept[arm].sum())} tok | {title}', fill=(255, 255, 0))
            sh.paste(t, (c * tile, r * tile))
    sh.save(path, quality=80)


def cmd_prep(args):
    import torch
    from src.eci.v2.encode import Encoder
    from src.eci.v2.pipeline import Series
    from src.eci.v2.video import Prefetch
    dev = dev_of()
    dom, kind = args.dom, args.kind
    K = KINDS[dom][kind]
    cal = calib_of(dom, K['series'])
    enc = Encoder(dev)
    se = Series(series_cfg(dom, K['series']), dev, encoder=enc)
    G, S = se.G, se.S
    Kd, Ko = disk(cal['r_dil'], dev), disk(cal['r_open'], dev)
    Fr = frame_list(dom, kind)
    obs_all = list(dict.fromkeys(Fr.obs.values))
    mine = set(obs_all[args.shard::args.nshard])
    Fr = Fr[Fr.obs.isin(mine)].reset_index(drop=True)
    items = []
    for o, g in Fr.groupby('obs', sort=False):
        idx = dict(zip(g.k.values, g.index.values))
        for k0, n, every, ks in runs_of(g.k.values):
            items.append((o, k0, n, every, ks, [idx[k] for k in ks]))
    if args.max_runs:
        items = items[:args.max_runs]
    jobs = [((i,), se.job(se.V[o], k0, n, every)) for i, (o, k0, n, every, ks, _) in enumerate(items)]
    loc = N.local_dir() / f'{dom}_{kind}_s{args.shard}'
    loc.mkdir(parents=True, exist_ok=True)
    ftok = open(loc / 'tok.f16', 'wb')
    meta = {k: [] for k in ('tok_frame', 'tok_pos', 'fr_raw', 'fr_dil', 'fr_thin')}
    recs, panels, t0 = [], [], time.time()
    bs = 16 if S >= 1000 else 32
    pf = Prefetch(jobs, max(4, int(os.environ.get('SLURM_CPUS_PER_TASK', 16)) - 4), fn=se.run_job)
    for (i,), fr in pf:
        o, k0, n, every, ks, fidx = items[i]
        v = se.V[o]
        grid = k0 + every * np.arange(n)
        sel = np.isin(grid, ks)
        fr = fr[sel]
        assert len(fr) == len(ks)
        for b in range(0, len(ks), bs):
            kb = ks[b:b + bs]
            with torch.no_grad():
                x, _ = se.frames(v, fr[b:b + bs], kb)
                px = se.mask.pixels(x, v['src_lo'] + kb)
                raw, dil, thn, m3, thin = masks_of(px, G, Kd, Ko)
                z = enc(x)
            arm = {'M0': raw >= 0.25, 'M1': raw > 0, 'M2': dil >= 0.25, 'M3': m3}
            store = arm['M1'] | arm['M2']
            npx, nthin = px.flatten(1).sum(1).float(), thin.flatten(1).sum(1).float()
            st = {'animal_px': npx, 'thin_px': nthin}
            for a, m in arm.items():
                st[f'n_{a}'] = m.sum(1).float()
                st[f'leak_{a}'] = (m & (raw < LEAK)).sum(1).float()
                st[f'thin_{a}'] = (thn * m).sum(1) * 256
                st[f'anim_{a}'] = (raw * m).sum(1) * 256
            st = {k_: v_.cpu().numpy() for k_, v_ in st.items()}
            for ii in range(len(kb)):
                pos = torch.nonzero(store[ii]).flatten()
                fid = len(recs)
                ftok.write(z[ii, pos].cpu().numpy().tobytes())
                meta['tok_frame'].append(np.full(len(pos), fid, np.int32))
                meta['tok_pos'].append(pos.cpu().numpy().astype(np.int16))
                for nm, t_ in (('fr_raw', raw), ('fr_dil', dil), ('fr_thin', thn)):
                    meta[nm].append(t_[ii, pos].cpu().numpy().astype(np.float16))
                r0 = Fr.iloc[fidx[b + ii]]
                rec = {'obs': o, 'k': int(kb[ii]), 'ref_f': int(r0.ref_f), 'n_store': int(len(pos))}
                for c_ in ('row', 'role', 'series'):
                    if c_ in Fr:
                        rec[c_] = r0[c_]
                rec.update({k_: float(v_[ii]) for k_, v_ in st.items()})
                recs.append(rec)
                if len(panels) < 4 and args.shard == 0 and o not in {p[3].split(' ')[0] for p in panels} and \
                        fid % 7 == 3:
                    panels.append((x[ii].cpu().numpy(), {a: m[ii].cpu().numpy() for a, m in arm.items()},
                                   np.asarray(thin[ii].cpu().numpy()), f'{o} k{int(kb[ii])}'))
        if i % 200 == 0:
            el = time.time() - t0
            log(f'prep {dom} {kind} s{args.shard}: run {i}/{len(items)}, {len(recs):,} frames, {el:.0f}s '
                f'({len(recs) / max(el, 1):.1f} fps; GPU busy {enc.gpu_s:.0f}s; decode thread-s {pf.dec_s:.0f}); {gpu_line()}')
    ftok.close()
    for k_, x_ in meta.items():
        np.save(loc / f'{k_}.npy', np.concatenate(x_))
    Fm = pd.DataFrame(recs)
    Fm.to_parquet(loc / 'frames.parquet')
    el = time.time() - t0
    info = {'dom': dom, 'kind': kind, 'shard': args.shard, 'nshard': args.nshard, 'S': S, 'G': G, 'calib': cal,
            'n_frames': len(Fm), 'n_stored_tokens': int(Fm.n_store.sum()), 'wall_s': el, 'gpu_s': enc.gpu_s,
            'fps_wall': len(Fm) / el, 'gpu': torch.cuda.get_device_name(0),
            **{f'tokens_per_frame_{a}': float(Fm[f'n_{a}'].mean()) for a in ('M0', 'M1', 'M2', 'M3')}}
    if args.shard == 0:
        info['check_vs_v2_store'] = identity_check(loc, K['ref'], Fm, G)
    log(json.dumps(info, default=float))
    od = OUT / dom / 'prep'
    od.mkdir(parents=True, exist_ok=True)
    tag = '_selftest' if args.max_runs else ''
    (od / f'{kind}_s{args.shard}{tag}.json').write_text(json.dumps(info, indent=1, default=float))
    if panels:
        mask_panels(panels, od / f'mask_panels_{kind}{tag}.jpg', G)
    R4.atomic_copy(loc, BEE / dom / kind / f's{args.shard}')  # selftests set V2M_BEE / V2M_OUT elsewhere


def identity_check(loc, ref, Fm, G, n=48):
    """M0 tokens of the first n frames vs the stored v2 (freeze / r2) tokens of the same frames: positions and values."""
    tf = np.load(loc / 'tok_frame.npy')
    pos = np.load(loc / 'tok_pos.npy')
    raw = np.load(loc / 'fr_raw.npy').astype(np.float32)
    tok = np.memmap(loc / 'tok.f16', np.float16, 'r', shape=(len(tf), D))
    rtf = np.load(ref / 'tok_frame.npy', mmap_mode='r')
    rpos = np.load(ref / 'tok_pos.npy', mmap_mode='r')
    rtok = np.memmap(ref / 'tok.f16', np.float16, 'r', shape=(len(rtf), D))
    same_pos, n_new, n_ref, dmax, dmean = 0, 0, 0, 0.0, []
    st = np.r_[0, np.cumsum(np.bincount(tf, minlength=len(Fm)))]
    for f in range(min(n, len(Fm))):
        a, b = st[f], st[f + 1]
        m0 = raw[a:b] >= 0.25
        p_new = pos[a:b][m0].astype(np.int64)
        rf = int(Fm.ref_f.values[f])
        ra, rb = np.searchsorted(rtf, rf), np.searchsorted(rtf, rf, side='right')
        p_ref = np.asarray(rpos[ra:rb]).astype(np.int64)
        n_new += len(p_new)
        n_ref += len(p_ref)
        same_pos += int(set(p_new.tolist()) == set(p_ref.tolist()))
        com, i1, i2 = np.intersect1d(p_new, p_ref, return_indices=True)
        if len(com):
            d = np.abs(np.asarray(tok[a:b][m0][i1], np.float32) - np.asarray(rtok[ra:rb][i2], np.float32))
            dmax = max(dmax, float(d.max()))
            dmean.append(float(d.mean()))
    return {'frames': min(n, len(Fm)), 'frames_same_M0_positions': same_pos, 'M0_tokens_new': n_new,
            'tokens_ref': n_ref, 'max_abs_diff': dmax, 'mean_abs_diff': float(np.mean(dmean)) if dmean else None}


# ---------------------------------------------------------------------------------------------- stores
class Union:
    """All shards of one (dom, kind) union store, staged to LOCAL_DIR."""

    def __init__(self, dom, kind):
        self.dom, self.kind = dom, kind
        root = BEE / dom / kind
        shards = sorted(root.glob('s*'), key=lambda p: int(p.name[1:]))
        assert shards, root
        self.parts, Fs, o = [], [], 0
        for sp in shards:
            dst = N.local_dir() / f'{dom}_{kind}_{sp.name}'
            if not dst.exists():
                R4.atomic_copy(sp, dst)
            tf = np.load(dst / 'tok_frame.npy')
            F_ = pd.read_parquet(dst / 'frames.parquet')
            self.parts.append(dict(root=dst, tf=tf + o, n=len(tf)))
            Fs.append(F_)
            o += len(F_)
        self.F = pd.concat(Fs, ignore_index=True)
        self.tf = np.concatenate([p['tf'] for p in self.parts])
        cat = lambda nm, dt: np.concatenate([np.load(p['root'] / f'{nm}.npy').astype(dt) for p in self.parts])  # noqa
        self.pos = cat('tok_pos', np.int64)
        self.raw, self.dil, self.thin = cat('fr_raw', np.float32), cat('fr_dil', np.float32), cat('fr_thin', np.float32)
        self.G = KINDS_G(dom, kind)

    def arm(self, a):
        return {'M0': self.raw >= 0.25, 'M1': self.raw > 0, 'M2': self.dil >= 0.25}[a]

    def tokens(self, m):
        out, o = [], 0
        for p in self.parts:
            mm = np.memmap(p['root'] / 'tok.f16', np.float16, 'r', shape=(p['n'], D))
            out.append(np.asarray(mm[m[o:o + p['n']]]))
            o += p['n']
        return np.concatenate(out)


def KINDS_G(dom, kind):
    return series_cfg(dom, KINDS[dom][kind]['series'])['S'] // 16


def arm_store(U, a, dev, on_gpu):
    """A freeze TokStore view of the union store restricted to arm a's tokens (frame order kept)."""
    m = U.arm(a)
    St = object.__new__(FZ.TokStore)
    St.F = U.F
    St.tf, St.pos = U.tf[m], U.pos[m]
    St.fd, St.thin = U.raw[m], U.thin[m]
    St.tok = U.tokens(m)
    St.nf, St.G, St.dev = len(U.F), U.G, dev
    St.starts = np.r_[0, np.cumsum(np.bincount(St.tf, minlength=St.nf))]
    St.dist = np.zeros(len(St.tf), np.int8)
    import torch
    St.T = R2.tokens_to(torch, St.tok, dev) if on_gpu else None
    return St


# ---------------------------------------------------------------------------------------------- train
def sae_dir(dom, a, s):
    return BEE / dom / 'sae' / f'{a}_s{s}'


def load_sae(dom, a, s, dev):
    from src.eci.sae import load_sae as ls
    sae, norm, _ = ls(sae_dir(dom, a, s) / 'sae_theta.pt', dev)
    return sae, norm


def cmd_train(args):
    import torch
    dev = dev_of()
    dom, a = args.dom, args.arm
    seeds = FZ.parse_seeds(args.seeds, SEEDS[dom])
    todo = [s for s in seeds if not (sae_dir(dom, a, s) / 'sae_theta.pt').exists()]
    if not todo:
        return
    sts = {k: arm_store(Union(dom, k), a, dev, on_gpu=False) for k in POOLS[dom]}
    X = {}
    for r in ('train', 'calib', 'eval'):
        X[r] = torch.cat([St.inputs(0, St.F.role.values[St.tf] == r) for St in sts.values()])
    X['calib'], X['eval'] = X['calib'].cpu(), X['eval'].cpu()
    log(f'{dom} {a}: ' + ', '.join(f'{r} {len(x):,}' for r, x in X.items()) + f' tokens; {gpu_line()}')
    res_f = OUT / dom / f'train_{a}.json'
    res = json.loads(res_f.read_text()) if res_f.exists() else {}
    for s in todo:
        cfg = FZ.cfg_of('F0', 2048) | {'arm_mask': a}
        if args.steps:
            cfg['steps'] = args.steps
        ti = R3.train_arm(torch, cfg, s, X['train'], X['calib'], X['eval'], sae_dir(dom, a, s), dev)
        res[f'{a}_s{s}'] = {k: ti[k] for k in ('train_time_s', 'n_train_tokens', 'theta_star', 'eval_fve', 'eval_l0',
                                                'dead_at_end_train')}
        log(f'{dom} {a}_s{s}: {ti["train_time_s"]:.0f}s, theta* {ti["theta_star"]:.4f}, eval FVE {ti["eval_fve"]:.4f}, '
            f'L0 {ti["eval_l0"]:.2f}, dead {ti["dead_at_end_train"]}; {gpu_line()}')
        res_f.parent.mkdir(parents=True, exist_ok=True)
        res_f.write_text(json.dumps(res, indent=1, default=float))


def fve_of(dom, a, s):
    f = OUT / dom / f'train_{a}.json'
    d = json.loads(f.read_text()) if f.exists() else {}
    return d.get(f'{a}_s{s}', {}).get('eval_fve', np.nan)


# ---------------------------------------------------------------------------------------------- score
def mice_sets(St):
    Y, unit, ann = FZ.mice_labels(St.F)
    sc = json.loads((N.OUT / 'score.json').read_text())
    units, SPL = sc['units'], {k_: np.array(v_) for k_, v_ in sc['splits'].items()}
    uidx = pd.Series(range(len(units)), index=units).loc[unit].values
    sets = {'all': np.arange(St.nf), 'consistent': np.flatnonzero(pd.Series(ann).isin(FZ.CONSISTENT).values)}
    half = {k: {s_: h[uidx[ix]] for s_, h in SPL.items()} for k, ix in sets.items()}
    return Y, sets, half, uidx, SPL


def ants_set(St, ser):
    ok, lid_ok, Y, _ = FZ.ants_labels(ser)
    rows, obs = St.F.row.values.astype(int), St.F.obs.values
    units, SPL = splits_of(set(obs))
    uidx = pd.Series(range(len(units)), index=units).loc[obs].values
    labs = list(FZ.ALAB) if lid_ok[rows].all() else list(FZ.ALAB[:3])
    return Y[rows][:, :len(labs)], {k: h[uidx] for k, h in SPL.items()}, labs


def cmd_score(args):
    import torch
    dev = dev_of()
    dom, a = args.dom, args.arm
    f = OUT / dom / f'score_{a}.json'
    res = json.loads(f.read_text()) if f.exists() else {}
    if dom == 'mice':
        St = arm_store(Union('mice', 'diag'), a, dev, on_gpu=True)
        Y, sets, half, _, _ = mice_sets(St)
        Yg = {k: torch.from_numpy(Y[ix]).to(dev) for k, ix in sets.items()}
        ixg = {k: torch.from_numpy(ix).to(dev) for k, ix in sets.items()}
        res['_frames'] = {k: int(len(ix)) for k, ix in sets.items()}
        for s in SEEDS[dom]:
            name = f'{a}_s{s}'
            if name in res or not (sae_dir(dom, a, s) / 'sae_theta.pt').exists():
                continue
            sae, norm = load_sae(dom, a, s, dev)
            X = St.codes(sae, norm, 0, 'all')
            out = {}
            for k in sets:
                o = crossfit(X if k == 'all' else X.index_select(0, ixg[k]), Yg[k], half[k], FZ.MLAB)
                o['primary'] = float(np.mean([o['mean']['nose_nose']['R'], o['mean']['nose_tail']['R']]))
                out[k] = FZ.compact(o) | {'primary': o['primary']}
            out['eval_fve'] = fve_of(dom, a, s)
            res[name] = out
            m = out['all']['mean']
            log(f'mice {name}: primary R~ {out["all"]["primary"]:.4f} | nn R~ {m["nose_nose"]["R"]:.4f} AUROC '
                f'{m["nose_nose"]["A"]:.4f} | nt R~ {m["nose_tail"]["R"]:.4f} AUROC {m["nose_tail"]["A"]:.4f} | '
                f'consistent primary R~ {out["consistent"]["primary"]:.4f}; {gpu_line()}')
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(res, indent=1, default=float))
            del X
            torch.cuda.empty_cache()
    elif dom == 'ants':
        sts = {ser: arm_store(Union('ants', f'scored_{ser}'), a, dev, on_gpu=True) for ser in ('v2', 'v3')}
        lab = {}
        for ser, St in sts.items():
            Y, h, labs = ants_set(St, ser)
            lab[ser] = (torch.from_numpy(Y).to(dev), h, labs)
        for s in SEEDS[dom]:
            name = f'{a}_s{s}'
            if name in res or not (sae_dir(dom, a, s) / 'sae_theta.pt').exists():
                continue
            sae, norm = load_sae(dom, a, s, dev)
            out = {}
            for ser, St in sts.items():
                Yg, h, labs = lab[ser]
                out[ser] = FZ.compact(crossfit(St.codes(sae, norm, 0, 'all'), Yg, h, labs))
            out['primary'] = float(np.mean([out[x]['mean']['groom_any']['R'] for x in ('v2', 'v3')]))
            out['lid_any'] = out['v3']['mean']['on_lid_any']['R']
            out['eval_fve'] = fve_of(dom, a, s)
            res[name] = out
            log(f'ants {name}: primary R~ {out["primary"]:.4f} | v2 groom_any R~ {out["v2"]["mean"]["groom_any"]["R"]:.4f} '
                f'| v3 groom_any R~ {out["v3"]["mean"]["groom_any"]["R"]:.4f} | v3 on_lid_any R~ {out["lid_any"]:.4f}; '
                f'{gpu_line()}')
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(res, indent=1, default=float))


# ---------------------------------------------------------------------------------------------- neurons
def token_stats(St, sae, norm, fsel):
    """Selection frames fsel (nf,) bool -> per neuron: firing frames, argmax-token class counts (off < 5% animal px,
    body >= 25%), thin-share numerator / denominator, plus frame codes (nf, W) fp16 (all frames)."""
    import torch
    W = sae.n_latents
    dev = St.dev
    acc = {k: torch.zeros(W, dtype=torch.float64, device=dev) for k in ('fire', 'hit', 'off', 'body', 'num', 'den')}
    X = torch.zeros((St.nf, W), dtype=torch.float16, device=dev)
    fs = torch.from_numpy(fsel).to(dev)
    fp = max(4, min(48, int(2**27 // W // 160)))
    with torch.no_grad():
        for f0, f1, a, b, z in St.chunks(fp):
            fl = torch.from_numpy(St.tf[a:b] - f0).to(dev)
            c = sae.encode(norm(z.float()), mode='threshold').float()
            fm = torch.zeros((f1 - f0, W), device=dev).index_reduce_(0, fl, c, 'amax', include_self=True)
            X[f0:f1] = fm.half()
            ok = fs[f0:f1]
            if not ok.any():
                continue
            tok_ok = ok[fl]
            raw = torch.from_numpy(St.fd[a:b]).to(dev)
            thin = torch.from_numpy(St.thin[a:b]).to(dev)
            hit = (c > 0) & (c == fm[fl]) & tok_ok[:, None]
            acc['fire'] += ((fm > 0) & ok[:, None]).sum(0).double()
            acc['hit'] += hit.sum(0).double()
            acc['off'] += hit[raw < LEAK].sum(0).double()
            acc['body'] += hit[raw >= 0.25].sum(0).double()
            ct = c * tok_ok[:, None]
            acc['num'] += (ct * thin[:, None]).sum(0).double()
            acc['den'] += (ct * raw[:, None]).sum(0).double()
    return {k: v.cpu().numpy() for k, v in acc.items()}, X


def neuron_summary(acc):
    ok = acc['fire'] >= MIN_FIRE
    off = np.where(ok, acc['off'] / np.maximum(acc['hit'], 1), np.nan)
    body = np.where(ok, acc['body'] / np.maximum(acc['hit'], 1), np.nan)
    thin = np.where(ok & (acc['den'] > 0), acc['num'] / np.maximum(acc['den'], 1e-12), np.nan)
    return off, body, thin, ok


def cmd_neurons(args):
    import torch
    dev = dev_of()
    dom, a = args.dom, args.arm
    # stores (selection half A / sheet half B)
    if dom == 'mice':
        sts = {'v1': arm_store(Union('mice', 'diag'), a, dev, on_gpu=True)}
        Y, sets, half, uidx, SPL = mice_sets(sts['v1'])
        hA = {'v1': SPL['rnd0'][uidx] == 0}
    elif dom == 'ants':
        sts = {ser: arm_store(Union('ants', f'scored_{ser}'), a, dev, on_gpu=True) for ser in ('v2', 'v3')}
        hA = {ser: ants_set(St, ser)[1]['rnd0'] == 0 for ser, St in sts.items()}
    else:
        U = Union('frogs', 'pool_fj')
        sts = {'juv': arm_store(U, a, dev, on_gpu=True)}
        hA = {'juv': sts['juv'].F.role.values == 'calib'}
    hB = {k: (~v) if dom != 'frogs' else (sts[k].F.role.values == 'eval') for k, v in hA.items()}
    tot_thin = sum(float(St.F.thin_px.sum()) for St in sts.values())
    tot_anim = sum(float(St.F.animal_px.sum()) for St in sts.values())
    f = OUT / dom / f'neurons_{a}.json'
    res = json.loads(f.read_text()) if f.exists() else {}
    res['_thin_px_share_of_animal_px'] = tot_thin / tot_anim
    fv = {s: fve_of(dom, a, s) for s in DEPLOY_FROM}
    dep = max(fv, key=lambda s: fv[s] if np.isfinite(fv[s]) else -1)
    res['_deploy_seed'] = dep
    for s in SEEDS[dom]:
        name = f'{a}_s{s}'
        if (name in res and s != dep) or not (sae_dir(dom, a, s) / 'sae_theta.pt').exists():
            continue
        sae, norm = load_sae(dom, a, s, dev)
        accs, Xs = {}, {}
        for ser, St in sts.items():
            accs[ser], Xs[ser] = token_stats(St, sae, norm, hA[ser])
        acc = {k: sum(accs[ser][k] for ser in sts) for k in accs[next(iter(sts))]}
        off, body, thin, ok = neuron_summary(acc)
        r = {'n_scored': int(ok.sum()), 'n_background': int(np.nansum(off > 0.5)),
             'share_background': float(np.nansum(off > 0.5) / max(ok.sum(), 1)),
             'argmax_off_share_median': float(np.nanmedian(off)), 'argmax_body_share_median': float(np.nanmedian(body)),
             'n_thin': int(np.nansum(thin > THIN_NEURON)), 'n_thin_gt_0.3': int(np.nansum(thin > 0.3)),
             'thin_share_median': float(np.nanmedian(thin)), 'thin_share_p99': float(np.nanquantile(thin, 0.99)),
             'top_thin': [[int(j), float(thin[j]), float(acc['fire'][j])] for j in np.argsort(-np.nan_to_num(thin, nan=-1))[:10]]}
        log(f'{dom} {name}: background neurons {r["n_background"]}/{r["n_scored"]}, thin neurons (share > 0.5) '
            f'{r["n_thin"]}, thin share p99 {r["thin_share_p99"]:.3f}, top {r["top_thin"][:3]}; {gpu_line()}')
        if s == dep:
            r['sheets'] = sheets(dom, a, s, sae, norm, sts, Xs, hB, r['top_thin'][:3])
        res[name] = r
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(res, indent=1, default=float))
        del Xs
        torch.cuda.empty_cache()


def sheets(dom, a, s, sae, norm, sts, Xs, hB, top):
    from src.eci.v2.pipeline import Series
    import torch
    out = {}
    (OUT / 'sheets').mkdir(parents=True, exist_ok=True)
    ser_cfg = {'v1': ('mice', 'v1'), 'v2': ('ants', 'v2'), 'v3': ('ants', 'v3'), 'juv': ('frogs', 'juv')}
    SE = {ser: Series(series_cfg(*ser_cfg[ser]), St.dev, load_encoder=False) for ser, St in sts.items()}
    labf = {}
    if dom == 'mice':
        ann = pd.read_csv(REPO / N.C['ann'], usecols=['Y_nn', 'Y_np', 'Y_nt']).values
        labf['v1'] = lambda f, St=sts['v1']: ''.join(c for c, y in zip(('nn', 'np', 'nt'), ann[int(St.F.row.values[f])])
                                                    if y > 0) or '-'
    elif dom == 'ants':
        for ser, St in sts.items():
            _, _, Y, _ = FZ.ants_labels(ser)
            labf[ser] = lambda f, St=St, Y=Y: ''.join(c for c, y in zip(('G', 'gY', 'gB', 'L'), Y[int(St.F.row.values[f])][:4])
                                                      if y) or '-'
    else:
        labf['juv'] = lambda f: '-'
    want = {'mice': ('nn', 'np', 'nt'), 'ants': ('G', 'L'), 'frogs': ('x',)}[dom]
    for j, ts, nfire in top:
        cand = []
        for ser, St in sts.items():
            v = Xs[ser][:, j].float().cpu().numpy()
            c = pd.DataFrame({'f': np.arange(St.nf), 'v': v, 'obs': St.F.obs.values})[hB[ser]]
            c = c[c.v > 0].sort_values('v', ascending=False).drop_duplicates('obs').head(24)
            cand += [(r.v, ser, int(r.f)) for r in c.itertuples()]
        cand = sorted(cand, reverse=True)[:24]
        recs_all, summ_all = [], []
        for ser in sts:
            fr = [f_ for _, s_, f_ in cand if s_ == ser]
            if not fr:
                continue
            St, se = sts[ser], SE[ser]

            def dec(f, St=St, se=se):
                o, k = St.F.obs.values[f], int(St.F.k.values[f])
                v = se.V[o]
                x, _ = se.frames(v, se.run_job(**se.job(v, k, 1)), [k])
                return x[0].cpu().numpy()
            fn = OUT / 'sheets' / f'{dom}_{a}_s{s}_n{j}_{ser}'
            summ, recs = FZ.frame_sheet(St, fr, sae, norm, 0, 'all', j, labf[ser], want, dec, fn,
                                        f'v2 mask test {dom} {a} s{s} thin-part neuron n{j} (thin share {ts:.2f}, '
                                        f'selected on half A; top held-out half-B frames, <= 1 per video) [{ser}]')
            thin_tok = []
            for f_ in fr:
                a0, b0 = int(St.starts[f_]), int(St.starts[f_ + 1])
                zt = torch.from_numpy(np.ascontiguousarray(St.tok[a0:b0])).to(St.dev)
                with torch.no_grad():
                    act = sae.encode(norm(zt.float()), mode='threshold')[:, j].float().cpu().numpy()
                pk = int(np.argmax(act))
                thin_tok.append([float(St.thin[a0 + pk]), float(St.fd[a0 + pk])])
            summ['peak_token_thin_frac_mean'] = float(np.mean([t[0] for t in thin_tok]))
            summ['peak_token_animal_frac_mean'] = float(np.mean([t[1] for t in thin_tok]))
            summ_all.append({'series': ser, 'n_frames': len(fr), **summ, 'sheet': str(fn.relative_to(REPO))})
            recs_all += recs
        out[str(j)] = {'thin_share': ts, 'fire_frames_A': nfire, 'per_series': summ_all}
        log(f'sheet {dom} {a} s{s} n{j}: ' + json.dumps(summ_all, default=float)[:600])
    return out


# ---------------------------------------------------------------------------------------------- decide
def coverage(dom):
    kinds = {'mice': ('diag', 'pool'), 'ants': ('scored_v2', 'pool_v2', 'scored_v3', 'pool_v3'), 'frogs': ('pool_fj',)}
    out = {}
    for kind in kinds[dom]:
        root = BEE / dom / kind
        Fs = [pd.read_parquet(p / 'frames.parquet') for p in sorted(root.glob('s*'))]
        if not Fs:
            continue
        Fm = pd.concat(Fs, ignore_index=True)
        key = {'diag': 'mice', 'pool': 'mice_pool', 'pool_fj': 'frogs'}.get(kind, kind)
        r = {'n_frames': len(Fm), 'thin_px_share_of_animal_px': float(Fm.thin_px.sum() / Fm.animal_px.sum())}
        for a in ('M0', 'M1', 'M2', 'M3'):
            r[a] = {'thin_coverage': float(Fm[f'thin_{a}'].sum() / Fm.thin_px.sum()),
                    'animal_px_recall': float(Fm[f'anim_{a}'].sum() / Fm.animal_px.sum()),
                    'tokens_per_frame': float(Fm[f'n_{a}'].mean()),
                    'leak_share': float(Fm[f'leak_{a}'].sum() / max(Fm[f'n_{a}'].sum(), 1))}
        out[key] = r
    return out


def cmd_decide(args):
    dec = {'rule': 'scripts/eci/v2_mask/mask_test.py docstring (pre-registered 2026-10-21)',
           'calib': json.loads((OUT / 'calib.json').read_text())}
    cov = {d: coverage(d) for d in ('mice', 'ants', 'frogs')}
    dec['coverage'] = cov
    M, A = {}, {}
    for a in ARMS:
        for d, D_ in (('mice', M), ('ants', A)):
            f = OUT / d / f'score_{a}.json'
            if f.exists():
                D_.update(json.loads(f.read_text()))
    mq = lambda a, q, st='all': [M[f'{a}_s{s}'][st]['primary'] if q == 'primary' else  # noqa: E731
                                 M[f'{a}_s{s}'][st]['mean'][q[0]][q[1]] for s in SEEDS['mice'] if f'{a}_s{s}' in M]
    aq = lambda a, q: [A[f'{a}_s{s}']['primary'] if q == 'primary' else A[f'{a}_s{s}'][q[0]]['mean'][q[1]][q[2]]  # noqa
                       for s in SEEDS['ants'] if f'{a}_s{s}' in A]
    tab = {}
    for a in ARMS:
        t = {'mice primary R~': FZ.mean_sd(mq(a, 'primary')),
             'mice consistent primary R~': FZ.mean_sd(mq(a, 'primary', 'consistent'))}
        for lb in ('nose_nose', 'nose_tail'):
            t[f'mice {lb} R~'] = FZ.mean_sd(mq(a, (lb, 'R')))
            t[f'mice {lb} AUROC'] = FZ.mean_sd(mq(a, (lb, 'A')))
            t[f'mice consistent {lb} R~'] = FZ.mean_sd(mq(a, (lb, 'R'), 'consistent'))
        t['ants primary R~'] = FZ.mean_sd(aq(a, 'primary'))
        for ser in ('v2', 'v3'):
            for lb in ('groom_any', 'groom_yellow', 'groom_blue') + (('on_lid_any',) if ser == 'v3' else ()):
                t[f'ants {ser} {lb} R~'] = FZ.mean_sd(aq(a, (ser, lb, 'R')))
                t[f'ants {ser} {lb} AUROC'] = FZ.mean_sd(aq(a, (ser, lb, 'A')))
        for d in ('mice', 'ants', 'frogs'):
            f = OUT / d / f'neurons_{a}.json'
            if f.exists():
                nr = json.loads(f.read_text())
                ss = [k for k in nr if not k.startswith('_')]
                t[f'{d} background-neuron share'] = FZ.mean_sd([nr[k]['share_background'] for k in ss])
                t[f'{d} thin-part neurons (share > 0.5)'] = FZ.mean_sd([nr[k]['n_thin'] for k in ss])
                t[f'{d} thin-part neurons (share > 0.3)'] = FZ.mean_sd([nr[k]['n_thin_gt_0.3'] for k in ss])
        tab[a] = t
    dec['table'] = tab
    crit = {}
    for a in ('M1', 'M2'):
        mw, aw = welch(mq(a, 'primary'), mq('M0', 'primary')), welch(aq(a, 'primary'), aq('M0', 'primary'))
        leak = {f'{d}/{k}': cov[d][k][a]['leak_share'] for d in cov for k in cov[d] if 'pool' not in k}
        thin = {f'{d}/{k}': cov[d][k][a]['thin_coverage'] - cov[d][k]['M0']['thin_coverage']
                for d in ('mice', 'ants') for k in cov[d] if 'pool' not in k}
        c = {'mice_primary_minus_M0': mw, 'ants_primary_minus_M0': aw, 'leak': leak, 'thin_coverage_gain': thin,
             'a': bool(mw[1] >= -0.01), 'b': bool(aw[1] >= -0.01), 'c': bool(max(leak.values()) <= 0.10),
             'd': bool(min(thin.values()) >= 0.15)}
        c['pass'] = c['a'] and c['b'] and c['c'] and c['d']
        crit[a] = c
    dec['criteria'] = crit
    passing = [a for a in ('M1', 'M2') if crit[a]['pass']]
    if not passing:
        dec['DECISION'] = 'keep M0'
    else:
        sc = {a: crit[a]['mice_primary_minus_M0'][0] + crit[a]['ants_primary_minus_M0'][0] for a in passing}
        best = max(sc.values())
        dec['DECISION'] = 'adopt ' + ('M2' if sc.get('M2') == best else max(sc, key=sc.get))
    (OUT / 'decision.json').write_text(json.dumps(dec, indent=1, default=float))
    lines = ['| metric | ' + ' | '.join(ARMS) + ' |', '|---|' + '---|' * len(ARMS)]
    for k in tab['M0']:
        lines.append(f'| {k} | ' + ' | '.join(f'{tab[a][k][0]:.4f} ± {tab[a][k][1]:.4f} (n={tab[a][k][2]})'
                                              if a in tab and k in tab[a] else '-' for a in ARMS) + ' |')
    lines += ['', '| coverage (domain/frames) | arm | thin coverage | animal-px recall | tokens / frame | leak share (< 5% animal px) |',
              '|---|---|---|---|---|---|']
    for d in cov:
        for k in cov[d]:
            for a in ('M0', 'M1', 'M2', 'M3'):
                r = cov[d][k][a]
                lines.append(f'| {d}/{k} | {a} | {r["thin_coverage"]:.3f} | {r["animal_px_recall"]:.3f} | '
                             f'{r["tokens_per_frame"]:.1f} | {r["leak_share"]:.3f} |')
    lines += ['', f'DECISION: {dec["DECISION"]}', '', '```', json.dumps(crit, indent=1, default=float), '```']
    (OUT / 'table.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=('calib', 'prep', 'train', 'score', 'neurons', 'decide'))
    p.add_argument('--dom', default='mice')
    p.add_argument('--kind', default='diag')
    p.add_argument('--arm', default='M0')
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--nshard', type=int, default=1)
    p.add_argument('--max-runs', type=int, default=0)
    p.add_argument('--seeds', default='')
    p.add_argument('--steps', type=int, default=0)
    args = p.parse_args()
    {'calib': cmd_calib, 'prep': cmd_prep, 'train': cmd_train, 'score': cmd_score, 'neurons': cmd_neurons,
     'decide': cmd_decide}[args.step](args)


if __name__ == '__main__':
    main()
