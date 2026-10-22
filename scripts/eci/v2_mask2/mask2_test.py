"""
ECI v2 THIN-PART MASK TEST (release blocker): a better label-free animal PIXEL CUE so the representation captures mouse
TAILS and ant ANTENNAE / LEGS while keeping the nose-nose / nose-tail and grooming alignment and no background neurons.

==================================================================================================================
PRE-REGISTRATION (written 2026-10-22 before any held-out benchmark score (step 1) and before any token of step 2 was
encoded. Development of the cue used ONLY CalMS21 TRAIN videos, MABe22 userTrain clips and lab TRAINING videos,
scripts/eci/v2_mask2/explore.py; parameter changes after this point are listed as dated amendments.)

CUE (src/eci/v2/mask2.py, module docstring = the exact definition): frozen dark cue = CORE; plus pixels that differ
    from the per-video median background (blur sigma 1 px, max over RGB, threshold max(20, 4 x per-frame robust noise))
    and are connected to a core pixel through such pixels within 6 body half-widths (geodesic growth), plus body-wide
    strong-change blobs (opening of d > 40 by a square of side 2 round(h/2) + 1) as extra core (pale animals).
ARMS (identical DINOv3-L/16 block-20 tokens of the frozen v2 encoder; only token SELECTION differs)
    M0  frozen v2 tight mask: frozen cue pixel fraction >= 0.25 per 16 x 16 patch.
    N1  new cue, same >= 0.25 rule.
    N2  new cue, >= 0.25 OR >= 16 GROWN (non-core) pixels in the patch (a thin attached part crossing it).

STEP 1  pixel-cue validation on held-out public benchmarks (scripts/eci/v2_mask2/validate.py docstring = metrics):
    MABe22 submission (black mice; tail_base / tail_mid / tail_tip keypoints), CalMS21 test (black resident + WHITE
    intruder), CRIM13 test (BC albino or C57 intruder; one track point per mouse).
    Step-1 PASS for an arm N (N1 / N2) iff ALL of
      (1a) MABe22 tail-segment token recall (tail_seg_tok) >= M0's + 0.20 (absolute);
      (1b) pale-mouse recall improves: CalMS21 pale_tok >= M0's + 0.10 AND CRIM13 BC pale_tok >= M0's + 0.10;
      (1c) body keypoints are not lost: MABe22 and CalMS21 body_tok >= M0's - 0.01;
      (1d) leak (kept tokens with no keypoint-region pixel) <= 0.05 on MABe22, CalMS21 and CRIM13.
    The ONE arm carried to step 2 = the passing arm with the higher MABe22 tail_seg_tok; within 0.02 -> N1 (fewer
    tokens). No arm passes -> stop and report (no step 2).

STEP 2  representation check (same frames / splits / labels / SAE recipe as the previous mask test,
    scripts/eci/v2_mask/mask_test.py; tokens re-encoded once, the union store holds N2 tokens, which contain M0 and N1).
    SAE: frozen v2 recipe (Matryoshka BatchTopK 2048, prefixes 128/512/2048, k 16, 30k steps), one per domain (ants v2 +
    v3 pooled), M0 and the step-1 arm, SAME seeds.
    SEEDS (user request: the minimum the non-inferiority test can resolve): from the previous test's 6-seed sds (mice
    primary R~ pooled sd 0.0151, ants 0.0199, Welch two-sample 95% CI), the CI half-width at a true difference 0 is
    <= the 0.01 margin from n = 20 seeds per arm (mice, 0.0099) and n = 33 (ants, 0.0099); n = 3 would give 0.035 /
    0.045 and n = 6 0.020 / 0.026, i.e. the rule could not pass even for an arm identical to M0. Seeds 70-89 (mice),
    70-102 (ants); deployment seed for sheets = highest held-out FVE among seeds 70-72.
    LAB COVERAGE (per domain, frame-averaged ratios): appendage pixels = new-cue pixels outside the body (body = opening
    of the new-cue pixels by a disk of radius round(h/2), dilated by a disk of radius max(1, round(h/4))); appendage
    coverage = appendage px in kept tokens / all appendage px; new-cue animal px recall; tokens / frame; leak = share of
    kept tokens with < 5% of their pixels in (frozen cue | raw change d > T_LO) (an independent off-animal test that
    does not use the growth / connectivity step).
    ALIGNMENT: R~ (cross-fitted normalised R-precision, 10 random unit halves) + AUROC; mice nose_nose / nose_tail, all
    annotators and the consistent annotators; mice primary = mean(R~ nn, R~ nt); ants groom_any / yellow / blue (v2,
    v3), on_lid_any (v3); ants primary = mean over v2 / v3 of R~ groom_any.
    NEURONS (deployment seed AND all seeds for counts; selection half A, sheets from half B, 24 frames, <= 1 per video,
    real per-token heat maps): background neuron = argmax token off-animal (as LEAK above) in > 50% of >= 20 firing
    frames; tail / antenna (appendage) neuron = activation-weighted appendage share sum_t a_t app_t / sum_t a_t anim_t
    > 0.5 with >= 20 firing frames; odour-box neuron (mice) = argmax token centre within 0.5 x near_radius of the
    canonical odour corner (dataset/mice/v1/eci/odor_corner.csv corner, rotated with the frame; top right) in > 50% of
    >= 20 firing frames. Sheets: top 3 appendage neurons and top 3 odour-box neurons by firing frames (both arms).
DECISION RULE (adopt the step-1 arm N over M0 iff ALL):
    (i)   step 1 passed (above);
    (ii)  mice primary R~ and ants primary R~: Welch 95% CI lower bound of mean(N) - mean(M0) >= -0.01 (each);
    (iii) lab leak <= 0.05 in mice, ants v2, ants v3 AND background-neuron share <= 0.01 in mice and ants (all seeds,
          mean);
    (iv)  appendage neurons exist (>= 5 per domain, mean over seeds, in mice AND ants) and the top-3 sheets of the
          deployment seed are visibly on tails / antennae / legs (the visual judgement is reported per sheet with the
          measured peak-token appendage fraction).
    Otherwise keep M0. If adopted: prepare the rollout config (configs/eci_v2/*.json mask cue 'thin') so the full
    rollout is one command.
==================================================================================================================

Steps (scripts/eci/v2_mask2/job.sh runs each in Slurm; staging in $LOCAL_DIR):
    bgs --dom D --kind K --shard i --nshard n   (CPU) per-video (per-window) median backgrounds -> BEE/bg/<dom>_<series>/
    prep --dom D --kind K --shard i --nshard n  (GPU) N2 token store + per-token fractions -> BEE/<D>/<K>/s<i>/
    train --dom D --arm A --seeds a-b           (GPU) SAEs -> BEE/<D>/sae/<A>_s<seed>, OUT/<D>/train_<A>.json
    score --dom D --arm A                       (GPU) alignment -> OUT/<D>/score_<A>.json
    neurons --dom D --arm A                     (GPU) background / appendage / odour neurons (+ sheets)
    decide                                      (CPU) -> OUT/decision.json, OUT/table.md
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
sys.path.insert(0, str(REPO / 'scripts/eci/v2_mask'))
import freeze as FZ  # noqa: E402
import mask_test as MT  # noqa: E402  (read-only: frame lists, runs, labels, splits, helpers)

N, R2, R3, R4 = FZ.N, FZ.R2, FZ.R3, FZ.R4
from src.eci.v2.score import crossfit, splits_of, welch  # noqa: E402
from src.eci.v2 import mask2 as M2  # noqa: E402

OUT = Path(os.environ.get('V2M2_OUT') or REPO / 'results/vision/eci_v2_mask2')
BEE = Path(os.environ.get('V2M2_BEE') or '/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_mask2')
log = N.log
D = 1024
SEEDS = {'mice': tuple(range(70, 90)), 'ants': tuple(range(70, 103))}
DEPLOY_FROM = (70, 71, 72)
LEAK, APP_NEURON, MIN_FIRE, ODOR_FRAC = 0.05, 0.5, 20, 0.5
N_BG = 48
KINDS, POOLS = MT.KINDS, {'mice': ('pool',), 'ants': ('pool_v2', 'pool_v3')}
CALIB = json.loads((REPO / 'results/vision/eci_v2_mask/calib.json').read_text())  # previous test's body widths


def series_cfg(dom, name):
    return MT.series_cfg(dom, name)


def h_of(dom, series):
    return float(CALIB[f'{dom}/{series}']['body_halfwidth_px'])


def arm_of_step1():
    f = OUT / 'validate' / 'step1.json'
    return json.loads(f.read_text())['carried_arm'] if f.exists() else 'N2'


def window_of(dom, k):
    return int(k) // 3000 if dom == 'ants' else 0


def bg_path(dom, series, obs, w):
    return BEE / 'bg' / f'{dom}_{series}' / (f'{obs}.npz' if w == 0 else f'{obs}__w{w}.npz')


# ---------------------------------------------------------------------------------------------- backgrounds
def cmd_bgs(args):
    """Per (video, window) median background of N_BG frames evenly spread over the window, canonical, side S."""
    import torch
    from concurrent.futures import ThreadPoolExecutor
    from src.eci.v2.pipeline import Series
    dom = args.dom
    todo = []
    for kind in KINDS[dom]:
        K = KINDS[dom][kind]
        Fr = MT.frame_list(dom, kind)
        for o, g in Fr.groupby('obs', sort=False):
            for w in sorted({window_of(dom, k) for k in g.k.values}):
                todo.append((K['series'], o, w))
    todo = sorted(set(todo))[args.shard::args.nshard]
    SE = {}
    dev = torch.device('cpu')
    torch.set_num_threads(4)

    def one(t):
        ser, o, w = t
        p = bg_path(dom, ser, o, w)
        if p.exists():
            return 0
        se = SE.setdefault(ser, Series(series_cfg(dom, ser), dev, load_encoder=False))
        v = se.V[o]
        lo, hi = w * 3000 if dom == 'ants' else 0, min(v['n'], (w + 1) * 3000) if dom == 'ants' else v['n']
        ks = np.unique(np.linspace(lo, hi - 1, N_BG).round().astype(int))
        xs = [se.frames(v, se.run_job(**se.job(v, int(k), 1, threads=1)), [int(k)])[0] for k in ks]
        xs = torch.cat(xs)
        bgf = M2.median_background(xs)
        mads = {f'mad_s{int(sg * 10)}': M2.background_mad(xs, bgf, sigma=sg).half().numpy() for sg in (1.0, 1.5, 2.0)}
        q02, q98 = M2.grey_quantiles(xs)
        mads['q02'], mads['q98'] = q02.round().to(torch.uint8).numpy(), q98.round().to(torch.uint8).numpy()
        bg = bgf.round().clamp(0, 255).to(torch.uint8).numpy()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.stem + '.tmp.npz')
        np.savez(tmp, bg=bg, n=len(ks), **mads)
        tmp.rename(p)
        return len(ks)
    for ser in {t[0] for t in todo}:
        SE[ser] = Series(series_cfg(dom, ser), dev, load_encoder=False)
    t0 = time.time()
    with ThreadPoolExecutor(max(2, int(os.environ.get('SLURM_CPUS_PER_TASK', 16)) // 4)) as ex:
        for i, n in enumerate(ex.map(one, todo)):
            if i % 10 == 0:
                log(f'bgs {dom} shard {args.shard}: {i + 1}/{len(todo)} ({time.time() - t0:.0f}s)')
    log(f'bgs {dom} shard {args.shard}: done {len(todo)} in {time.time() - t0:.0f}s')


def load_pack(dom, series, o, w, dev):
    """stored background (cmd_bgs) -> the src/eci/v2/mask2.background_pack dict (ghost with the series' dark rule)."""
    import torch
    z = np.load(bg_path(dom, series, o, w))
    m = series_cfg(dom, series)['mask']
    bg = torch.from_numpy(z['bg']).to(dev).float()
    q98 = torch.from_numpy(z['q98']).to(dev).float()
    gb = M2.grey_f(bg[None])[0]
    return {'bg': bg, 'mad': torch.from_numpy(z[f'mad_s{int(round(M2.SIGMA * 10))}']).to(dev).float(),
            'q02': torch.from_numpy(z['q02']).to(dev).float(), 'q98': q98,
            'ghost': (gb < m.get('dark_abs', 60)) & ((q98 - gb) > m.get('dark_rel', 40))}


# ---------------------------------------------------------------------------------------------- odour corner
def odour_corner(S):
    """obs -> (row, col, radius) of the canonical odour corner at side S (top right after the v2 rotation)."""
    from src.eci.foreground import ODOR_ROT90
    oc = pd.read_csv(REPO / 'dataset/mice/v1/eci/odor_corner.csv').set_index('observation_id')
    out = {}
    for o, r in oc.iterrows():
        y, x = float(r.corner_y_px), float(r.corner_x_px)
        for _ in range(int(ODOR_ROT90[r.odor_corner])):  # torch.rot90 k=1 dims (H, W): out[i, j] = in[j, W-1-i]
            y, x = 511.0 - x, y
        out[o] = (y * S / 512, x * S / 512, 0.5 * float(r.near_radius_px) * S / 512)
    return out


# ---------------------------------------------------------------------------------------------- prep
def cmd_prep(args):
    import torch
    import torch.nn.functional as F
    from src.eci.v2.encode import Encoder
    from src.eci.v2.pipeline import Series
    from src.eci.v2.video import Prefetch
    dev = torch.device('cuda')
    dom, kind = args.dom, args.kind
    K = KINDS[dom][kind]
    h = h_of(dom, K['series'])
    enc = Encoder(dev)
    se = Series(series_cfg(dom, K['series']), dev, encoder=enc)
    G, S = se.G, se.S
    Ko = MT.disk(max(1, int(round(h / 2))), dev)
    Kt = MT.disk(max(1, int(round(h / 4))), dev)
    Fr = MT.frame_list(dom, kind)
    obs_all = list(dict.fromkeys(Fr.obs.values))
    mine = set(obs_all[args.shard::args.nshard])
    Fr = Fr[Fr.obs.isin(mine)].reset_index(drop=True)
    items = []
    for o, g in Fr.groupby('obs', sort=False):
        idx = dict(zip(g.k.values, g.index.values))
        for w in sorted({window_of(dom, k) for k in g.k.values}):
            kw = np.array([k for k in g.k.values if window_of(dom, k) == w])
            for k0, n, every, ks in MT.runs_of(kw):
                items.append((o, w, k0, n, every, ks, [idx[k] for k in ks]))
    if args.max_runs:
        items = items[:args.max_runs]
    jobs = [((i,), se.job(se.V[o], k0, n, every)) for i, (o, w, k0, n, every, ks, _) in enumerate(items)]
    loc = N.local_dir() / f'{dom}_{kind}_s{args.shard}'
    loc.mkdir(parents=True, exist_ok=True)
    ftok = open(loc / 'tok.f16', 'wb')
    meta = {k: [] for k in ('tok_frame', 'tok_pos', 'fr_raw', 'fr_new', 'n_grown', 'fr_app', 'fr_any')}
    recs, panels, t0 = [], [], time.time()
    bs = 16 if S >= 1000 else 32
    bgc = {}
    pf = Prefetch(jobs, max(4, int(os.environ.get('SLURM_CPUS_PER_TASK', 16)) - 4), fn=se.run_job)
    for (i,), fr in pf:
        o, w, k0, n, every, ks, fidx = items[i]
        v = se.V[o]
        if (o, w) not in bgc:
            bgc.clear()
            bgc[(o, w)] = load_pack(dom, K['series'], o, w, dev)
        pack = bgc[(o, w)]
        grid = k0 + every * np.arange(n)
        fr = fr[np.isin(grid, ks)]
        assert len(fr) == len(ks)
        for b in range(0, len(ks), bs):
            kb = ks[b:b + bs]
            with torch.no_grad():
                x, _ = se.frames(v, fr[b:b + bs], kb)
                core0 = se.mask.pixels(x, v['src_lo'] + kb)
                r = M2.cue(x, pack, core0, h)
                pix, grown = r['pix'], r['grown']
                body = MT.dilate(MT.dilate(MT.erode(pix, Ko), Ko), Kt)
                app = pix & ~body
                anyp = core0 | (r['d'] > M2.T_LO)
                pool = lambda m: F.avg_pool2d(m.float()[:, None], 16).flatten(1)  # noqa: E731
                raw, frn, fapp, fany = pool(core0), pool(pix), pool(app), pool(anyp)
                ng = pool(grown) * 256
                z = enc(x)
            arm = {'M0': raw >= 0.25, 'N1': M2.keep(frn, ng, 'r25'), 'N2': M2.keep(frn, ng, 'r25t')}
            store = arm['N2'] | arm['M0']
            napp, nanim = app.flatten(1).sum(1).float(), pix.flatten(1).sum(1).float()
            st = {'animal_px': nanim, 'app_px': napp, 'core_px': core0.flatten(1).sum(1).float(),
                  'grown_px': grown.flatten(1).sum(1).float(), 'thr': r['thr']}
            for a, m in arm.items():
                st[f'n_{a}'] = m.sum(1).float()
                st[f'leak_{a}'] = (m & (fany < LEAK)).sum(1).float()
                st[f'app_{a}'] = (fapp * m).sum(1) * 256
                st[f'anim_{a}'] = (frn * m).sum(1) * 256
            st = {k_: v_.cpu().numpy() for k_, v_ in st.items()}
            for ii in range(len(kb)):
                pos = torch.nonzero(store[ii]).flatten()
                fid = len(recs)
                ftok.write(z[ii, pos].cpu().numpy().tobytes())
                meta['tok_frame'].append(np.full(len(pos), fid, np.int32))
                meta['tok_pos'].append(pos.cpu().numpy().astype(np.int16))
                for nm, t_ in (('fr_raw', raw), ('fr_new', frn), ('fr_app', fapp), ('fr_any', fany)):
                    meta[nm].append(t_[ii, pos].cpu().numpy().astype(np.float16))
                meta['n_grown'].append(ng[ii, pos].round().cpu().numpy().astype(np.int16))
                r0 = Fr.iloc[fidx[b + ii]]
                rec = {'obs': o, 'k': int(kb[ii]), 'ref_f': int(r0.ref_f), 'n_store': int(len(pos))}
                for c_ in ('row', 'role', 'series'):
                    if c_ in Fr:
                        rec[c_] = r0[c_]
                rec.update({k_: float(v_[ii]) for k_, v_ in st.items()})
                recs.append(rec)
                if len(panels) < 6 and args.shard == 0 and fid % 11 == 5 and \
                        o not in {p[-1].split(' ')[0] for p in panels}:
                    panels.append((x[ii].cpu().numpy(), {a: m[ii].cpu().numpy() for a, m in arm.items()},
                                   grown[ii].cpu().numpy(), app[ii].cpu().numpy(), f'{o} k{int(kb[ii])}'))
        if i % 200 == 0:
            el = time.time() - t0
            log(f'prep {dom} {kind} s{args.shard}: run {i}/{len(items)}, {len(recs):,} frames, {el:.0f}s '
                f'({len(recs) / max(el, 1):.1f} fps; GPU busy {enc.gpu_s:.0f}s; decode thread-s {pf.dec_s:.0f}); '
                f'{MT.gpu_line()}')
    ftok.close()
    for k_, x_ in meta.items():
        np.save(loc / f'{k_}.npy', np.concatenate(x_))
    Fm = pd.DataFrame(recs)
    Fm.to_parquet(loc / 'frames.parquet')
    el = time.time() - t0
    info = {'dom': dom, 'kind': kind, 'shard': args.shard, 'nshard': args.nshard, 'S': S, 'G': G, 'h': h,
            'n_frames': len(Fm), 'n_stored_tokens': int(Fm.n_store.sum()), 'wall_s': el, 'gpu_s': enc.gpu_s,
            'fps_wall': len(Fm) / el, 'gpu': torch.cuda.get_device_name(0),
            **{f'tokens_per_frame_{a}': float(Fm[f'n_{a}'].mean()) for a in ('M0', 'N1', 'N2')}}
    if args.shard == 0:
        info['check_vs_v2_store'] = MT.identity_check(loc, K['ref'], Fm, G)
    log(json.dumps(info, default=float))
    od = OUT / dom / 'prep'
    od.mkdir(parents=True, exist_ok=True)
    tag = '_selftest' if args.max_runs else ''
    (od / f'{kind}_s{args.shard}{tag}.json').write_text(json.dumps(info, indent=1, default=float))
    if panels:
        mask_panels(panels, od / f'mask_panels_{kind}{tag}.jpg', G)
    R4.atomic_copy(loc, BEE / dom / kind / f's{args.shard}')


def mask_panels(items, path, G):
    """rows: M0 | N1 | N2 kept tokens (green), grown pixels magenta, appendage pixels cyan (N2 panel)."""
    from PIL import Image, ImageDraw
    tile = 512
    sh = Image.new('RGB', (3 * tile, len(items) * tile))
    for r, (x, kept, grown, app, title) in enumerate(items):
        base = x.transpose(1, 2, 0).astype(np.float32)
        S = base.shape[0]
        for c, arm in enumerate(('M0', 'N1', 'N2')):
            im = base.copy()
            Kp = np.kron(kept[arm].reshape(G, G), np.ones((16, 16), bool))
            im[Kp] = 0.65 * im[Kp] + 0.35 * np.array([0, 255, 0])
            if arm != 'M0':
                im[grown] = np.array([255, 0, 255])
            if arm == 'N2':
                im[app & ~grown] = np.array([0, 255, 255])
            t = Image.fromarray(im.clip(0, 255).astype(np.uint8)).resize((tile, tile), Image.BILINEAR)
            ImageDraw.Draw(t).text((4, 4), f'{arm} {int(kept[arm].sum())} tok | {title}', fill=(255, 255, 0))
            sh.paste(t, (c * tile, r * tile))
    sh.save(path, quality=82)


# ---------------------------------------------------------------------------------------------- stores
class Union:
    def __init__(self, dom, kind):
        self.dom, self.kind = dom, kind
        root = BEE / dom / kind
        shards = sorted([p for p in root.glob('s*') if not p.name.endswith('.tmp')], key=lambda p: int(p.name[1:]))
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
        self.raw, self.new, self.app, self.any = (cat(k, np.float32) for k in ('fr_raw', 'fr_new', 'fr_app', 'fr_any'))
        self.ng = cat('n_grown', np.float32)
        self.G = series_cfg(dom, KINDS[dom][kind]['series'])['S'] // 16

    def arm(self, a):
        return {'M0': self.raw >= 0.25, 'N1': self.new >= 0.25,
                'N2': (self.new >= 0.25) | (self.ng >= M2.MIN_THIN_PX)}[a]

    def tokens(self, m):
        out, o = [], 0
        for p in self.parts:
            mm = np.memmap(p['root'] / 'tok.f16', np.float16, 'r', shape=(p['n'], D))
            out.append(np.asarray(mm[m[o:o + p['n']]]))
            o += p['n']
        return np.concatenate(out)


def arm_store(U, a, dev, on_gpu):
    import torch
    m = U.arm(a)
    St = object.__new__(FZ.TokStore)
    St.F = U.F
    St.tf, St.pos = U.tf[m], U.pos[m]
    St.fd = U.new[m]          # animal fraction (new cue) for sheets' locality numbers
    St.app, St.any, St.raw = U.app[m], U.any[m], U.raw[m]
    St.tok = U.tokens(m)
    St.nf, St.G, St.dev = len(U.F), U.G, dev
    St.starts = np.r_[0, np.cumsum(np.bincount(St.tf, minlength=St.nf))]
    St.dist = np.zeros(len(St.tf), np.int8)
    St.T = R2.tokens_to(torch, St.tok, dev) if on_gpu else None
    return St


# ---------------------------------------------------------------------------------------------- train / score
def sae_dir(dom, a, s):
    return BEE / dom / 'sae' / f'{a}_s{s}'


def load_sae(dom, a, s, dev):
    from src.eci.sae import load_sae as ls
    sae, norm, _ = ls(sae_dir(dom, a, s) / 'sae_theta.pt', dev)
    return sae, norm


def cmd_train(args):
    import torch
    dev = torch.device('cuda')
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
    log(f'{dom} {a}: ' + ', '.join(f'{r} {len(x):,}' for r, x in X.items()) + f' tokens; {MT.gpu_line()}')
    for s in todo:
        cfg = FZ.cfg_of('F0', 2048) | {'arm_mask2': a}
        if args.steps:
            cfg['steps'] = args.steps
        ti = R3.train_arm(torch, cfg, s, X['train'], X['calib'], X['eval'], sae_dir(dom, a, s), dev)
        rec = {k: ti[k] for k in ('train_time_s', 'n_train_tokens', 'theta_star', 'eval_fve', 'eval_l0',
                                  'dead_at_end_train')}
        log(f'{dom} {a}_s{s}: {ti["train_time_s"]:.0f}s, eval FVE {ti["eval_fve"]:.4f}, L0 {ti["eval_l0"]:.2f}, '
            f'dead {ti["dead_at_end_train"]}; {MT.gpu_line()}')
        res_f = OUT / dom / 'train' / f'{a}_s{s}.json'
        res_f.parent.mkdir(parents=True, exist_ok=True)
        res_f.write_text(json.dumps(rec, indent=1, default=float))


def fve_of(dom, a, s):
    f = OUT / dom / 'train' / f'{a}_s{s}.json'
    return json.loads(f.read_text()).get('eval_fve', np.nan) if f.exists() else np.nan


def cmd_score(args):
    import torch
    dev = torch.device('cuda')
    dom, a = args.dom, args.arm
    seeds = FZ.parse_seeds(args.seeds, SEEDS[dom])
    od = OUT / dom / 'score'
    od.mkdir(parents=True, exist_ok=True)
    seeds = [s for s in seeds if not (od / f'{a}_s{s}.json').exists() and (sae_dir(dom, a, s) / 'sae_theta.pt').exists()]
    if not seeds:
        return
    if dom == 'mice':
        St = arm_store(Union('mice', 'diag'), a, dev, on_gpu=True)
        Y, sets, half, _, _ = MT.mice_sets(St)
        Yg = {k: torch.from_numpy(Y[ix]).to(dev) for k, ix in sets.items()}
        ixg = {k: torch.from_numpy(ix).to(dev) for k, ix in sets.items()}
    else:
        sts = {ser: arm_store(Union('ants', f'scored_{ser}'), a, dev, on_gpu=True) for ser in ('v2', 'v3')}
        lab = {}
        for ser, St_ in sts.items():
            Y, hh, labs = MT.ants_set(St_, ser)
            lab[ser] = (torch.from_numpy(Y).to(dev), hh, labs)
    for s in seeds:
        sae, norm = load_sae(dom, a, s, dev)
        out = {'eval_fve': fve_of(dom, a, s)}
        if dom == 'mice':
            X = St.codes(sae, norm, 0, 'all')
            for k in sets:
                o = crossfit(X if k == 'all' else X.index_select(0, ixg[k]), Yg[k], half[k], FZ.MLAB)
                o['primary'] = float(np.mean([o['mean']['nose_nose']['R'], o['mean']['nose_tail']['R']]))
                out[k] = FZ.compact(o) | {'primary': o['primary']}
            m = out['all']['mean']
            log(f'mice {a}_s{s}: primary R~ {out["all"]["primary"]:.4f} | nn R~ {m["nose_nose"]["R"]:.4f} | nt R~ '
                f'{m["nose_tail"]["R"]:.4f} | consistent primary R~ {out["consistent"]["primary"]:.4f}; {MT.gpu_line()}')
            del X
        else:
            for ser, St_ in sts.items():
                Yg_, hh, labs = lab[ser]
                out[ser] = FZ.compact(crossfit(St_.codes(sae, norm, 0, 'all'), Yg_, hh, labs))
            out['primary'] = float(np.mean([out[x]['mean']['groom_any']['R'] for x in ('v2', 'v3')]))
            out['lid_any'] = out['v3']['mean']['on_lid_any']['R']
            log(f'ants {a}_s{s}: primary R~ {out["primary"]:.4f} | v2 groom_any R~ {out["v2"]["mean"]["groom_any"]["R"]:.4f}'
                f' | v3 groom_any R~ {out["v3"]["mean"]["groom_any"]["R"]:.4f} | v3 on_lid_any R~ {out["lid_any"]:.4f}')
        (od / f'{a}_s{s}.json').write_text(json.dumps(out, indent=1, default=float))
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------------------------- neurons
def token_stats(St, sae, norm, fsel, odor):
    """selection frames fsel -> per neuron accumulators (firing frames, argmax off-animal / in odour zone, appendage
    numerator / denominator) + frame codes (nf, W) fp16."""
    import torch
    W = sae.n_latents
    dev = St.dev
    acc = {k: torch.zeros(W, dtype=torch.float64, device=dev) for k in ('fire', 'hit', 'off', 'odor', 'num', 'den')}
    X = torch.zeros((St.nf, W), dtype=torch.float16, device=dev)
    fs = torch.from_numpy(fsel).to(dev)
    G = St.G
    tokc = None
    if odor is not None:  # token centre inside the frame's odour zone
        oy, ox, orad = (np.array([odor[o][i] for o in St.F.obs.values]) for i in range(3))
        pr, pc = St.pos // G, St.pos % G
        fr = St.tf
        tokc = ((pr * 16 + 8 - oy[fr]) ** 2 + (pc * 16 + 8 - ox[fr]) ** 2) <= orad[fr] ** 2
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
            hit = (c > 0) & (c == fm[fl]) & tok_ok[:, None]
            acc['fire'] += ((fm > 0) & ok[:, None]).sum(0).double()
            acc['hit'] += hit.sum(0).double()
            acc['off'] += hit[torch.from_numpy(St.any[a:b] < LEAK).to(dev)].sum(0).double()
            if tokc is not None:
                acc['odor'] += hit[torch.from_numpy(tokc[a:b]).to(dev)].sum(0).double()
            ct = c * tok_ok[:, None]
            acc['num'] += (ct * torch.from_numpy(St.app[a:b]).to(dev)[:, None]).sum(0).double()
            acc['den'] += (ct * torch.from_numpy(St.fd[a:b]).to(dev)[:, None]).sum(0).double()
    return {k: v.cpu().numpy() for k, v in acc.items()}, X


def cmd_neurons(args):
    import torch
    dev = torch.device('cuda')
    dom, a = args.dom, args.arm
    if dom == 'mice':
        sts = {'v1': arm_store(Union('mice', 'diag'), a, dev, on_gpu=True)}
        Y, sets, half, uidx, SPL = MT.mice_sets(sts['v1'])
        hA = {'v1': SPL['rnd0'][uidx] == 0}
        odor = odour_corner(sts['v1'].G * 16)
    else:
        sts = {ser: arm_store(Union('ants', f'scored_{ser}'), a, dev, on_gpu=True) for ser in ('v2', 'v3')}
        hA = {ser: MT.ants_set(St, ser)[1]['rnd0'] == 0 for ser, St in sts.items()}
        odor = None
    hB = {k: ~v for k, v in hA.items()}
    fv = {s: fve_of(dom, a, s) for s in DEPLOY_FROM}
    dep = max(fv, key=lambda s: fv[s] if np.isfinite(fv[s]) else -1)
    od = OUT / dom / 'neurons'
    od.mkdir(parents=True, exist_ok=True)
    seeds = FZ.parse_seeds(args.seeds, SEEDS[dom])
    for s in seeds:
        f = od / f'{a}_s{s}.json'
        if f.exists() or not (sae_dir(dom, a, s) / 'sae_theta.pt').exists():
            continue
        sae, norm = load_sae(dom, a, s, dev)
        accs, Xs = {}, {}
        for ser, St in sts.items():
            accs[ser], Xs[ser] = token_stats(St, sae, norm, hA[ser], odor)
        acc = {k: sum(accs[ser][k] for ser in sts) for k in accs[next(iter(sts))]}
        ok = acc['fire'] >= MIN_FIRE
        off = np.where(ok, acc['off'] / np.maximum(acc['hit'], 1), np.nan)
        odr = np.where(ok, acc['odor'] / np.maximum(acc['hit'], 1), np.nan)
        app = np.where(ok & (acc['den'] > 0), acc['num'] / np.maximum(acc['den'], 1e-12), np.nan)
        rank = lambda sel: [[int(j), float(app[j]), float(odr[j]), float(acc['fire'][j])]  # noqa: E731
                            for j in sel[np.argsort(-acc['fire'][sel])][:10]]
        an = np.flatnonzero(np.nan_to_num(app, nan=-1) > APP_NEURON)
        on = np.flatnonzero(np.nan_to_num(odr, nan=-1) > ODOR_FRAC)
        r = {'deploy_seed': dep, 'n_scored': int(ok.sum()), 'n_background': int(np.nansum(off > 0.5)),
             'share_background': float(np.nansum(off > 0.5) / max(ok.sum(), 1)),
             'n_app': int(len(an)), 'n_app_gt_0.3': int(np.nansum(app > 0.3)),
             'app_share_median': float(np.nanmedian(app)), 'app_share_p99': float(np.nanquantile(app, 0.99)),
             'top_app': rank(an), 'n_odor': int(len(on)), 'top_odor': rank(on)}
        log(f'{dom} {a}_s{s}: background {r["n_background"]}/{r["n_scored"]}, appendage neurons {r["n_app"]}, '
            f'odour {r["n_odor"]}, top app {r["top_app"][:3]}, top odour {r["top_odor"][:3]}; {MT.gpu_line()}')
        if s == dep:
            r['sheets'] = {'app': sheets(dom, a, s, sae, norm, sts, Xs, hB, r['top_app'][:3], 'app'),
                           'odor': sheets(dom, a, s, sae, norm, sts, Xs, hB, r['top_odor'][:3], 'odor')}
        f.write_text(json.dumps(r, indent=1, default=float))
        del Xs
        torch.cuda.empty_cache()


def sheets(dom, a, s, sae, norm, sts, Xs, hB, top, kind):
    import torch
    from src.eci.v2.pipeline import Series
    out = {}
    (OUT / 'sheets').mkdir(parents=True, exist_ok=True)
    ser_cfg = {'v1': ('mice', 'v1'), 'v2': ('ants', 'v2'), 'v3': ('ants', 'v3')}
    SE = {ser: Series(series_cfg(*ser_cfg[ser]), St.dev, load_encoder=False) for ser, St in sts.items()}
    labf = {}
    if dom == 'mice':
        ann = pd.read_csv(REPO / N.C['ann'], usecols=['Y_nn', 'Y_np', 'Y_nt']).values
        labf['v1'] = lambda f, St=sts['v1']: ''.join(c for c, y in zip(('nn', 'np', 'nt'), ann[int(St.F.row.values[f])])
                                                    if y > 0) or '-'
    else:
        for ser, St in sts.items():
            _, _, Y, _ = FZ.ants_labels(ser)
            labf[ser] = lambda f, St=St, Y=Y: ''.join(c for c, y in zip(('G', 'gY', 'gB', 'L'),
                                                                          Y[int(St.F.row.values[f])][:4]) if y) or '-'
    want = {'mice': ('nn', 'np', 'nt'), 'ants': ('G', 'L')}[dom]
    for j, ap, odr, nfire in top:
        cand = []
        for ser, St in sts.items():
            v = Xs[ser][:, j].float().cpu().numpy()
            c = pd.DataFrame({'f': np.arange(St.nf), 'v': v, 'obs': St.F.obs.values})[hB[ser]]
            c = c[c.v > 0].sort_values('v', ascending=False).drop_duplicates('obs').head(24)
            cand += [(r.v, ser, int(r.f)) for r in c.itertuples()]
        cand = sorted(cand, reverse=True)[:24]
        summ_all = []
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
            fn = OUT / 'sheets' / f'{dom}_{a}_s{s}_{kind}_n{j}_{ser}'
            summ, _ = FZ.frame_sheet(St, fr, sae, norm, 0, 'all', j, labf[ser], want, dec, fn,
                                     f'v2 thin-mask test {dom} {a} s{s} {kind} neuron n{j} (appendage share {ap:.2f}, '
                                     f'argmax in odour zone {odr:.2f}; selected on half A; top held-out half-B frames, '
                                     f'<= 1 per video) [{ser}]')
            pk = []
            for f_ in fr:
                a0, b0 = int(St.starts[f_]), int(St.starts[f_ + 1])
                zt = torch.from_numpy(np.ascontiguousarray(St.tok[a0:b0])).to(St.dev)
                with torch.no_grad():
                    act = sae.encode(norm(zt.float()), mode='threshold')[:, j].float().cpu().numpy()
                q = a0 + int(np.argmax(act))
                pk.append([float(St.app[q]), float(St.fd[q]), float(St.raw[q])])
            summ['peak_token_app_frac_mean'] = float(np.mean([p[0] for p in pk]))
            summ['peak_token_animal_frac_mean'] = float(np.mean([p[1] for p in pk]))
            summ['peak_token_frozen_cue_frac_mean'] = float(np.mean([p[2] for p in pk]))
            summ['peak_token_outside_M0_share'] = float(np.mean([p[2] < 0.25 for p in pk]))
            summ_all.append({'series': ser, 'n_frames': len(fr), **summ, 'sheet': str(fn.relative_to(REPO))})
        out[str(j)] = {'app_share': ap, 'odor_share': odr, 'fire_frames_A': nfire, 'per_series': summ_all}
        log(f'sheet {dom} {a} s{s} {kind} n{j}: ' + json.dumps(summ_all, default=float)[:600])
    return out


# ---------------------------------------------------------------------------------------------- decide
def coverage(dom):
    kinds = {'mice': ('diag', 'pool'), 'ants': ('scored_v2', 'pool_v2', 'scored_v3', 'pool_v3')}
    out = {}
    for kind in kinds[dom]:
        Fs = [pd.read_parquet(p / 'frames.parquet') for p in sorted((BEE / dom / kind).glob('s*'))
              if not p.name.endswith('.tmp')]
        if not Fs:
            continue
        Fm = pd.concat(Fs, ignore_index=True)
        r = {'n_frames': len(Fm), 'app_px_share_of_animal_px': float(Fm.app_px.sum() / Fm.animal_px.sum()),
             'grown_px_share_of_animal_px': float(Fm.grown_px.sum() / Fm.animal_px.sum())}
        for a in ('M0', 'N1', 'N2'):
            r[a] = {'app_coverage': float(Fm[f'app_{a}'].sum() / max(Fm.app_px.sum(), 1)),
                    'animal_px_recall': float(Fm[f'anim_{a}'].sum() / Fm.animal_px.sum()),
                    'tokens_per_frame': float(Fm[f'n_{a}'].mean()),
                    'leak_share': float(Fm[f'leak_{a}'].sum() / max(Fm[f'n_{a}'].sum(), 1))}
        out[kind] = r
    return out


def load_dir(d, a, seeds):
    out = {}
    for s in seeds:
        f = d / f'{a}_s{s}.json'
        if f.exists():
            out[s] = json.loads(f.read_text())
    return out


def cmd_decide(args):
    st1 = json.loads((OUT / 'validate' / 'step1.json').read_text())
    A_ = st1['carried_arm']
    arms = ('M0', A_)
    dec = {'rule': 'scripts/eci/v2_mask2/mask2_test.py docstring (pre-registered 2026-10-22)', 'step1': st1,
           'arm': A_, 'coverage': {d: coverage(d) for d in ('mice', 'ants')}}
    sc = {d: {a: load_dir(OUT / d / 'score', a, SEEDS[d]) for a in arms} for d in ('mice', 'ants')}
    nr = {d: {a: load_dir(OUT / d / 'neurons', a, SEEDS[d]) for a in arms} for d in ('mice', 'ants')}
    mq = lambda a, q, st='all': [r[st]['primary'] if q == 'primary' else r[st]['mean'][q[0]][q[1]]  # noqa: E731
                                 for r in sc['mice'][a].values()]
    aq = lambda a, q: [r['primary'] if q == 'primary' else r[q[0]]['mean'][q[1]][q[2]]  # noqa: E731
                       for r in sc['ants'][a].values()]
    tab = {}
    for a in arms:
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
        for d in ('mice', 'ants'):
            rr = list(nr[d][a].values())
            if rr:
                t[f'{d} background-neuron share'] = FZ.mean_sd([r['share_background'] for r in rr])
                t[f'{d} appendage (tail / antenna / leg) neurons'] = FZ.mean_sd([r['n_app'] for r in rr])
                t[f'{d} appendage share p99'] = FZ.mean_sd([r['app_share_p99'] for r in rr])
                if d == 'mice':
                    t['mice odour-zone neurons'] = FZ.mean_sd([r['n_odor'] for r in rr])
        tab[a] = t
    dec['table'] = tab
    cov = dec['coverage']
    mw, aw = welch(mq(A_, 'primary'), mq('M0', 'primary')), welch(aq(A_, 'primary'), aq('M0', 'primary'))
    leak = {f'{d}/{k}': cov[d][k][A_]['leak_share'] for d in cov for k in cov[d] if 'pool' not in k}
    crit = {'i_step1': bool(st1['pass'][A_]), 'mice_primary_minus_M0': mw, 'ants_primary_minus_M0': aw,
            'ii': bool(mw[1] >= -0.01 and aw[1] >= -0.01), 'leak': leak,
            'bg_share': {d: tab[A_][f'{d} background-neuron share'][0] for d in ('mice', 'ants')},
            'app_neurons': {d: tab[A_][f'{d} appendage (tail / antenna / leg) neurons'][0] for d in ('mice', 'ants')}}
    crit['iii'] = bool(max(leak.values()) <= 0.05 and max(crit['bg_share'].values()) <= 0.01)
    crit['iv_count'] = bool(min(crit['app_neurons'].values()) >= 5)
    crit['iv_visual'] = 'see report: per-sheet visual judgement'
    dec['criteria'] = crit
    dec['DECISION_numeric'] = f'adopt {A_}' if (crit['i_step1'] and crit['ii'] and crit['iii'] and crit['iv_count']) \
        else 'keep M0'
    (OUT / 'decision.json').write_text(json.dumps(dec, indent=1, default=float))
    L = ['| metric | ' + ' | '.join(arms) + ' |', '|---|' + '---|' * len(arms)]
    for k in tab['M0']:
        L.append(f'| {k} | ' + ' | '.join(f'{tab[a][k][0]:.4f} ± {tab[a][k][1]:.4f} (n={tab[a][k][2]})'
                                          if k in tab[a] else '-' for a in arms) + ' |')
    L += ['', '| coverage (domain/frames) | arm | appendage coverage | animal-px recall (new cue) | tokens / frame | '
          'leak share |', '|---|---|---|---|---|---|']
    for d in cov:
        for k in cov[d]:
            for a in ('M0', 'N1', 'N2'):
                r = cov[d][k][a]
                L.append(f'| {d}/{k} | {a} | {r["app_coverage"]:.3f} | {r["animal_px_recall"]:.3f} | '
                         f'{r["tokens_per_frame"]:.1f} | {r["leak_share"]:.3f} |')
    L += ['', f'DECISION (numeric criteria; (iv) visual in the report): {dec["DECISION_numeric"]}', '', '```',
          json.dumps(crit, indent=1, default=float), '```']
    (OUT / 'table.md').write_text('\n'.join(L))
    print('\n'.join(L))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=('bgs', 'prep', 'train', 'score', 'neurons', 'decide'))
    p.add_argument('--dom', default='mice')
    p.add_argument('--kind', default='diag')
    p.add_argument('--arm', default='M0')
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--nshard', type=int, default=1)
    p.add_argument('--max-runs', type=int, default=0)
    p.add_argument('--seeds', default='')
    p.add_argument('--steps', type=int, default=0)
    args = p.parse_args()
    if args.shard < 0:
        args.shard = int(os.environ['SLURM_ARRAY_TASK_ID'])
    {'bgs': cmd_bgs, 'prep': cmd_prep, 'train': cmd_train, 'score': cmd_score, 'neurons': cmd_neurons,
     'decide': cmd_decide}[args.step](args)


if __name__ == '__main__':
    main()
