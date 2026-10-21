"""
ECI v2 CANDIDATE P (pair-centred crops). v2 is NOT delivered; this is a candidate for it, never "v2.1" / "v3".
Question: does adding label-free, tracking-free, egocentrically aligned pair crops to the frame's token set give a
behaviour representation in which nose-nose / nose-tail atoms emerge (and generalise to ants)? Labels are used ONLY
to score / select among the pre-registered arms, never in any representation step.

==================================================================================================================
PRE-REGISTRATION (written 2026-10-21, BEFORE any crop token, SAE or score of this round existed; only label-free
selftest contact sheets of the contact detector were looked at). Numbers known beforehand: the F0 (frozen v2) seeds
6-25 / 30-35 scores (mice primary R~ 0.119 / 0.114, ants primary R~ ~0.59) and the round-2 token ceiling (one
supervised direction, nn R~ 0.356 / nt 0.296, best SAE atom |cos| <= 0.08).

FIXED PARTS = the frozen v2 spec F0 (configs/eci_v2/representation.json): DINOv3-L/16 block 20 (revision ea8dc28,
bf16 + SDPA), native-capped frames (mice 1024 px of the 2064 px source, 64 x 64 grid; ants v2 768 / v3 832 px),
canonical rotation (mice: odour corner top right), TIGHT mask (token kept iff >= 25% of its 16 x 16 patch are animal
pixels), no context, Matryoshka BatchTopK k 16, 30,000 steps x 4096 tokens, theta* on calibration videos, frame value
= max over the frame's readout tokens, scored on the raw frame max (as every round: no smoothing in the scorer).

CANDIDATE P (src/eci/v2/pair.py, full definitions there):
  contacts   per frame, from the native pixel mask only: analysis grid (mice 1/8, ants 1/2 of native; ~32-38 cells per
             body), 3 x 3 opening, blobs, A1 = median over the decode run of (animal area / N) (N = 4 mice, 3 ants),
             n = round(area / A1) animals per blob, k-means split of merged blobs, contacts = adjacent k-means parts
             inside a blob + blob pairs <= 0.15 body lengths apart; pair axis = centroid to centroid.
  crops      side 1.5 body lengths (body = r4 prior: mice 303, ants v2 64, v3 76 native px), centred on the contact
             point, rotated so the pair axis is horizontal; 180-degree ambiguity SYMMETRISED (crops at theta and theta
             + 180 deg, both added). Encoder side S_c: mice 448 native px -> 256 px (16 x 16 patches = 10.7 patches per
             body length; prior evidence: CalMS21 48 = 64 = 80 grid, mice 2048 zoom not kept, so ~10 per body is
             enough); ants at native spacing (no upsampling): v2 96 px (6 x 6, 4.0 patches per body), v3 112 px (7 x 7,
             4.4 patches per body). Same DINOv3-L block 20; tight mask on the crop (same pixel rule, same background).
             Frames without a contact have no crop tokens.
ARMS (identical frames, splits and seeds)
  P0  frozen F0 reference, retrained with this round's seeds: frame tokens only, Matryoshka 2048 (128/512/2048).
  P1  F0 + pair crops (variant 'rot') in the SAME 2048 dictionary, trained on the pooled frame + crop tokens of the
      training pool; frame value = max over frame tokens AND the frame's crop tokens.
  P2  P0 dictionary (same seed) + a SEPARATE crop dictionary (Matryoshka 512, prefixes 128/256/512, trained on crop
      tokens only), codes concatenated (2048 + 512 columns; crop columns = max over the frame's crop tokens, 0 without
      a contact).
  P3  control = P1 with UNROTATED crops (variant 'can': same centre and side, angles 0 and 180 deg in the canonical
      frame): isolates the egocentric alignment from the crop zoom / framing. Reported, never adopted.
  Pool mixing (P1 / P3): all frame tokens; crop tokens of each role (train / calib / eval) subsampled uniformly (rng 0)
  to at most the number of frame tokens of that role, so neither view supplies more than half of the tokens.
SEEDS: round 40-45 for every arm (never used before: rounds 1-4 0-25, freeze 6-25, confirmation 30-35). Confirmation:
  fresh seeds 50-55 for the round winner and for P0.
DATA (= freeze.py / confirm.py): MICE v1 train pool = round-2 pool (62,208 frames, all 432 videos, video roles);
  scored = round-2 diagnosis set (86,400 frames, 144 annotated videos). Labels nose_nose = Y_nn | Y_np (primary),
  nose_tail = Y_nt (primary), nn_mutual, np_directional; sets 'all' and 'consistent' (annotators SG / SD / MR / CP).
  Units = 24 pools, the 10 random 12 / 12 splits of round 1. ANTS v2 + v3: freeze pools (one SAE on both series) and
  scored frames (v2 13,200, v3 32,100); groom_any (primary), groom_yellow, groom_blue, on_lid_any (v3).
METRICS (src/eci/v2/score.py): R~ = cross-fitted normalised R-precision (neuron picked on one half, scored on the
  other, both directions, mean over 10 random splits); cf AUROC. mice primary R~ = mean(R~ nose_nose, R~ nose_tail);
  ants primary R~ = mean over v2 / v3 of R~ groom_any.
INTERPRETABILITY: per arm the deployment seed = highest held-out FVE among seeds 40-42 (label-free; confirmation
  50-52); nose_nose / nose_tail neuron picked on pools half 0 of split rnd0; its 24 top held-out frames (<= 1 per video):
  share labelled positive; heat map = per-token activation of the neuron over the frame tokens AND the crop tokens
  (crop tokens mapped to frame coordinates through the crop transform); spread = number of DISTINCT 64-grid frame cells
  (16 px at 1024) holding a token at >= half the frame's max activation (a crop token counts in the cell of its mapped
  centre); peak = the token with the max activation: its animal-pixel fraction and whether it is a crop token.
RULE (decided on the CONFIRMATION seeds 50-55 only). Round winner = P1 or P2, whichever has the higher mean mice primary
  R~ over seeds 40-45. Adopt candidate P (= the winner's design) iff ALL of
    (a) mice primary R~: mean(winner) - mean(P0) >= +0.03 AND the Welch 95% CI excludes 0 (6 vs 6 fresh seeds);
    (b) ants primary R~: Welch 95% CI lower bound of mean(winner) - mean(P0) >= -0.01;
    (c) interpretability not worse, deployment seed, for nose_nose AND nose_tail: share of the 24 top held-out frames
        labelled positive >= P0's; mean distinct half-max cells <= max(12, P0's) (r3 / r4 diffuse maps had ~22-25,
        local ones 2-4); mean peak animal-pixel fraction >= P0's - 0.10.
  Reported, never deciding: the 40-45 round, nose_nose / nose_tail / nn_mutual / np_directional separately, AUROC, the
  consistent-annotator set, ants per series and label, P3 vs P1 (alignment effect), contact detection numbers (recall =
  share of nn / nt-labelled frames with >= 1 contact; contact rate on frames with no nn / np / nt label), crops per
  frame, cost, and the |cos| of every arm's deployment-seed atoms with the supervised token-ceiling direction (refit
  as r3 tmax: one direction on standardised frame tokens, max over tokens, pools half 0 of rnd0) and with the same
  supervised direction fitted on the crop tokens (pair-view ceiling).
==================================================================================================================

Steps (scripts/eci/v2/pair/pair.sh runs each in Slurm; inputs staged in $LOCAL_DIR)
    prep      (GPU) --dom mice|ants --store pool|diag|scored [--series v2|v3] --shard i --nshards n: decode native frames
              once, contacts, crops (both variants), crop tokens -> BEE/<dom>/<store>[_<series>]/shard<i>/
    contacts  (CPU) contact detection numbers -> OUT/contacts.json
    train     (GPU) --dom --arms --seeds -> BEE/<dom>/sae/<arm>_s<seed>/ (P2 = P0 + P2c, the crop dictionary)
    score     (GPU) --dom --arms --seeds -> OUT/<dom>/score_<arm>.json
    sheets    (GPU) --dom mice --arms --seeds (deployment among them) -> OUT/mice/sheets/, OUT/mice/locality.json
    cos       (GPU) supervised direction alignment -> OUT/mice/cos.json
    decide    (CPU) -> OUT/decision.json, OUT/table.md
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from math import gcd
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts/eci'))
sys.path.insert(0, str(REPO / 'scripts/eci/nnnt'))
sys.path.insert(0, str(REPO / 'scripts/eci/v2'))
import freeze as FZ  # noqa: E402  (read-only: TokStore, labels, scorer helpers)

N, R2, R3, R4 = FZ.N, FZ.R2, FZ.R3, FZ.R4
from src.eci.v2 import pair as PR  # noqa: E402
from src.eci.v2.score import crossfit, welch  # noqa: E402

OUT = Path(os.environ.get('PAIR_OUT') or REPO / 'results/vision/eci_v2/pair')
BEE = Path(os.environ.get('PAIR_BEE') or '/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2/pair')
log = N.log
D = 1024
SEEDS = tuple(range(40, 46))
SEEDS_C = tuple(range(50, 56))
VARIANTS = ('rot', 'can')
ARMS = {'P0': dict(crop=None), 'P1': dict(crop='rot'), 'P2': dict(crop='rot', sep=True), 'P3': dict(crop='can')}
W_MAIN, PREF_MAIN = 2048, (128, 512, 2048)
W_CROP, PREF_CROP = 512, (128, 256, 512)
MLAB = FZ.MLAB
CONSISTENT = FZ.CONSISTENT

DOM = {
    'mice': dict(n_animals=4, f=8, body=303.0, dark_abs=60, dark_rel=40, S=1024),
    'v2': dict(n_animals=3, f=2, body=64.0, dark_abs=256, dark_rel=40, S=768, dom='ants_v2p'),
    'v3': dict(n_animals=3, f=2, body=76.0, dark_abs=256, dark_rel=40, S=832, dom='ants_v3p'),
}


def ser_cfg(ser):
    c = DOM[ser]
    side, S_c = PR.crop_side(c['body'], S_cap=256)
    return {**c, 'side': side, 'S_c': S_c}


# ---------------------------------------------------------------------------------------------- store / video access
def store_src(dom, store, ser=None):
    """Frame-token store (BeeGFS) of round 2 (mice) / freeze (ants)."""
    if dom == 'mice':
        return R2.BEE2 / store
    return FZ.BEE / 'ants' / f'{store}_{ser}'


def store_tag(dom, store, ser=None):
    return store if dom == 'mice' else f'{store}_{ser}'


def videos_of(dom, ser=None):
    """{obs: dict(path, s_off, step, fps, w, h, rot, src_lo, role)}"""
    out = {}
    if dom == 'mice':
        for v in N.load_plan()['videos']:
            out[v['obs']] = dict(path=v['path'], s_off=int(v['offset']), step=N.C['step'], fps=N.C['fps'], w=2064,
                                 h=2064, rot=int(v['rot']), src_lo=int(v['src_lo']), role=v['role'])
        return out
    c = R4.DOMS[DOM[ser]['dom']]
    for v in R4.load_plan(DOM[ser]['dom'])['videos']:
        out[v['obs']] = dict(path=v['path'], s_off=R4.voff(c, v), step=c['step'], fps=c['fps'], w=v['probe']['w'],
                             h=v['probe']['h'], rot=0, src_lo=int(v['src_lo']), role=v['role'])
    return out


def tight_mask(dom, ser, dev):
    from src.eci.v2.mask import TightMask
    from src.eci.v2.pipeline import obs_ranges
    if dom == 'mice':
        return TightMask(REPO / 'dataset/mice/v1/eci/fg448al/background', obs_ranges(REPO / N.C['ann']), 'dark', dev,
                         align='odor', dark_abs=60, dark_rel=40)
    c = R4.DOMS[DOM[ser]['dom']]
    _, Rr = R4.obs_ranges_of(REPO / c['ann'])
    return TightMask(REPO / c['bg'], Rr, 'dark', dev, dark_abs=DOM[ser]['dark_abs'], dark_rel=DOM[ser]['dark_rel'])


def runs_of(ks, max_n=60, max_gap=10):
    """Sorted 5 fps frame indices -> decode runs [(k0, dk, n, [kept ks])]."""
    ks = np.sort(np.asarray(ks))
    out, cur = [], [int(ks[0])]
    for k in ks[1:]:
        if k - cur[-1] <= max_gap and (k - cur[0]) // max(1, reduce(gcd, np.diff(cur + [int(k)]).tolist())) < max_n:
            cur.append(int(k))
        else:
            out.append(cur)
            cur = [int(k)]
    out.append(cur)
    res = []
    for g in out:
        dk = reduce(gcd, np.diff(g).tolist()) if len(g) > 1 else 1
        res.append((g[0], dk, (g[-1] - g[0]) // dk + 1, g))
    return res


# ---------------------------------------------------------------------------------------------- prep
def cmd_prep(args):
    import torch
    from src.eci.v2.encode import Encoder
    from src.eci.v2.mask import THR, grey_of
    from src.eci.v2.pipeline import decode_native
    from src.eci.v2.video import Prefetch
    dev = torch.device('cuda')
    dom, store = args.dom, args.store
    ser = 'mice' if dom == 'mice' else args.series
    cfg = ser_cfg(ser)
    tag = store_tag(dom, store, None if dom == 'mice' else ser)
    Fs = pd.read_parquet(store_src(dom, store, None if dom == 'mice' else ser) / 'frames.parquet')
    V = videos_of(dom, None if dom == 'mice' else ser)
    obs_all = sorted(Fs.obs.unique())
    mine = [o for i, o in enumerate(obs_all) if i % args.nshards == args.shard]
    if args.max_videos:
        mine = mine[:args.max_videos]
    fidx = {o: g.index.values for o, g in Fs.groupby('obs', sort=False)}
    items = []
    for o in mine:
        ix = fidx[o]
        kk = Fs.k.values[ix]
        order = np.argsort(kk)
        ix, kk = ix[order], kk[order]
        pos_of = dict(zip(kk.tolist(), ix.tolist()))
        for k0, dk, n, g in runs_of(kk):
            items.append((o, k0, dk, n, g, [pos_of[k] for k in g]))
    if args.max_frames:
        items = items[:args.max_frames]
    jobs = []
    for i, (o, k0, dk, n, g, _) in enumerate(items):
        v = V[o]
        jobs.append(((i,), dict(path=v['path'], s0=v['s_off'] + v['step'] * k0, n=n, step=v['step'] * dk, w=v['w'],
                                h=v['h'], fps=v['fps'], threads=2)))
    enc = Encoder(dev)
    tm = tight_mask(dom, None if dom == 'mice' else ser, dev)
    f, side, S_c, nA = cfg['f'], cfg['side'], cfg['S_c'], cfg['n_animals']
    body_cells = cfg['body'] / f
    G_c = S_c // 16
    loc = R4.local_dir() / f'pair_{tag}_s{args.shard}'
    for vn in VARIANTS:
        (loc / vn).mkdir(parents=True, exist_ok=True)
    ftok = {vn: open(loc / vn / 'tok.f16', 'wb') for vn in VARIANTS}
    meta = {vn: {'tok_crop': [], 'tok_pos': [], 'tok_fdark': []} for vn in VARIANTS}
    crecs = {vn: [] for vn in VARIANTS}
    frecs = []
    sheet = []
    cpu = ThreadPoolExecutor(8)
    t0 = time.time()
    n_crops_enc = 0
    pf = Prefetch(jobs, max(4, int(os.environ.get('SLURM_CPUS_PER_TASK', 16)) - 6), fn=decode_native)
    for (i,), fr in pf:
        o, k0, dk, n, g, fids = items[i]
        v = V[o]
        sel = (np.asarray(g) - k0) // dk
        x = torch.from_numpy(np.ascontiguousarray(fr[sel])).to(dev).permute(0, 3, 1, 2)
        if v['rot']:
            x = torch.rot90(x, v['rot'], dims=(2, 3))
        x = x.contiguous()
        rows = v['src_lo'] + np.asarray(g)
        with torch.no_grad():
            px = tm.pixels(x, rows)
            am = PR.analysis_mask(px, f).cpu().numpy()
        del px
        bl = list(cpu.map(PR.blobs, list(am)))
        A1 = PR.run_A1([b[2] for b in bl], nA)
        cts = list(cpu.map(lambda b: PR.contacts_of(b[0], b[1], A1, nA, body_cells), bl))
        planes = {kk: (sel_, pb) for sel_, pb, _ in tm.planes(rows, x.shape[-1]) for kk in sel_}
        # crops of every contact of the run, both variants, both orientations
        todo = {vn: [] for vn in VARIANTS}  # (frame j, contact c, o, centre, theta)
        for j, (cl, inf) in enumerate(cts):
            frecs.append({'f': int(fids[j]), 'obs': o, 'k': int(g[j]), **inf,
                          'n_crops': 2 * len(cl)})
            for c_i, c in enumerate(cl):
                ctr = (c['x'] * f, c['y'] * f)
                for o_ in (0, 1):
                    todo['rot'].append((j, c_i, o_, ctr, c['theta'] + np.pi * o_, c))
                    todo['can'].append((j, c_i, o_, ctr, np.pi * o_, c))
        for vn in VARIANTS:
            T = todo[vn]
            for a in range(0, len(T), 96):
                part = T[a:a + 96]
                crops, bgs = [], []
                for j in sorted(set(t[0] for t in part)):
                    tj = [t for t in part if t[0] == j]
                    img = x[j].float()
                    pb = planes[j][1][None]
                    cr, bg = PR.sample_crops(img, pb, [t[3] for t in tj], [t[4] for t in tj], side, S_c)
                    crops.append(cr)
                    bgs.append(bg[:, 0])
                    if vn == 'rot' and len(sheet) < 48 and (j == 0 or args.selftest):
                        for q, t in enumerate(tj[:2]):
                            sheet.append((cr[q].cpu().numpy(), f'{o} k{g[j]} {t[5]["kind"]} o{t[2]}'))
                order = [t for j in sorted(set(t[0] for t in part)) for t in part if t[0] == j]
                cr = torch.cat(crops)
                bg = torch.cat(bgs)
                with torch.no_grad():
                    gr = grey_of(cr).float()
                    pxc = (gr < cfg['dark_abs']) & ((bg - gr) > cfg['dark_rel'])
                    fd = torch.nn.functional.avg_pool2d(pxc.float()[:, None], 16).flatten(1)
                    keep = fd >= THR
                    z = enc(cr)
                n_crops_enc += len(cr)
                qi, pi = torch.nonzero(keep, as_tuple=True)  # crop-major order
                ftok[vn].write(z[qi, pi].cpu().numpy().tobytes())
                qi_n = qi.cpu().numpy()
                cid0 = len(crecs[vn])
                meta[vn]['tok_crop'].append((qi_n + cid0).astype(np.int32))
                meta[vn]['tok_pos'].append(pi.cpu().numpy().astype(np.int16))
                meta[vn]['tok_fdark'].append(fd[qi, pi].cpu().numpy().astype(np.float16))
                nk = np.bincount(qi_n, minlength=len(order))
                for q, t in enumerate(order):
                    c = t[5]
                    crecs[vn].append({'crop': cid0 + q, 'f': int(fids[t[0]]), 'contact': int(t[1]), 'o': int(t[2]),
                                      'cx': float(t[3][0]), 'cy': float(t[3][1]), 'theta': float(t[4]),
                                      'kind': c['kind'], 'n_kept': int(nk[q])})
        if args.selftest and i < 6:
            selftest_sheet(x, am, cts, f, OUT / 'selftest' / f'{tag}_contacts_{i}.jpg', o, g)
        if i % 200 == 0:
            el = time.time() - t0
            log(f'prep {tag} s{args.shard}: {i}/{len(items)} runs, {len(frecs):,} frames, {len(crecs["rot"]):,} crops, '
                f'{el:.0f}s ({len(frecs) / max(el, 1):.1f} frames/s, {n_crops_enc / max(el, 1):.1f} crops/s; GPU busy '
                f'{enc.gpu_s:.0f}s; decode thread-s {pf.dec_s:.0f})')
    for vn in VARIANTS:
        ftok[vn].close()
        for k_, xs in meta[vn].items():
            np.save(loc / vn / f'{k_}.npy', np.concatenate(xs) if xs else np.zeros(0, np.int32))
        pd.DataFrame(crecs[vn]).to_parquet(loc / vn / 'crops.parquet')
    Fr = pd.DataFrame(frecs)
    Fr.to_parquet(loc / 'frames.parquet')
    el = time.time() - t0
    info = {'tag': tag, 'shard': args.shard, 'nshards': args.nshards, 'n_videos': len(mine), 'n_frames': len(Fr),
            'side_native': side, 'S_c': S_c, 'G_c': G_c, 'patches_per_body': G_c / PR.SIDE_BODY,
            'contact_frame_share': float((Fr.n_contacts > 0).mean()), 'contacts_per_frame': float(Fr.n_contacts.mean()),
            'crops_per_frame': float(Fr.n_crops.mean()),
            'merge_contact_share': float(Fr.n_merge_contacts.sum() / max(Fr.n_contacts.sum(), 1)),
            'tokens_per_crop': {vn: float(np.mean([r['n_kept'] for r in crecs[vn]])) if crecs[vn] else 0.0
                                for vn in VARIANTS},
            'n_crop_tokens': {vn: int(sum(r['n_kept'] for r in crecs[vn])) for vn in VARIANTS},
            'wall_s': el, 'gpu_s': enc.gpu_s, 'frames_per_s': len(Fr) / el, 'crops_per_s_wall': n_crops_enc / el,
            'gpu': torch.cuda.get_device_name(0)}
    log(json.dumps(info))
    (OUT / 'prep').mkdir(parents=True, exist_ok=True)
    st = '_selftest' if args.selftest else ''
    (OUT / 'prep' / f'{tag}_s{args.shard}{st}.json').write_text(json.dumps(info, indent=1))
    if sheet:
        crop_sheet(sheet, OUT / 'prep' / f'crops_{tag}_s{args.shard}{st}.jpg')
    if not args.selftest:
        R4.atomic_copy(loc, BEE / dom / tag / f'shard{args.shard}')
        log(f'-> {BEE / dom / tag / f"shard{args.shard}"}')


def selftest_sheet(x, am, cts, f, path, o, g):
    """Label-free check of the contact detector: frame (downscaled), mask cells, contact points and axes."""
    from PIL import Image, ImageDraw
    path.parent.mkdir(parents=True, exist_ok=True)
    tiles = []
    for j in range(min(8, len(x))):
        im = x[j].permute(1, 2, 0).cpu().numpy()
        h = 512
        t = Image.fromarray(im).resize((h, h), Image.BILINEAR)
        a = np.asarray(t).astype(np.float32)
        M = np.array(Image.fromarray(am[j].astype(np.uint8) * 255).resize((h, h), Image.NEAREST)) > 0
        a[M] = 0.6 * a[M] + 0.4 * np.array([0, 255, 0])
        t = Image.fromarray(a.astype(np.uint8))
        dr = ImageDraw.Draw(t)
        s = h / x.shape[-1] * f
        cl, inf = cts[j]
        for c in cl:
            cx, cy = c['x'] * s, c['y'] * s
            L = 30
            dx, dy = np.cos(c['theta']) * L, np.sin(c['theta']) * L
            col = (255, 0, 0) if c['kind'] == 'merge' else (0, 128, 255)
            dr.line([cx - dx, cy - dy, cx + dx, cy + dy], fill=col, width=2)
            dr.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], outline=col, width=2)
        dr.text((4, 4), f'{o} k{g[j]} blobs {inf["n_blobs"]} parts {inf["n_parts"]} contacts {inf["n_contacts"]}',
                fill=(255, 255, 0))
        tiles.append(t)
    sh = Image.new('RGB', (4 * 512, 512 * int(np.ceil(len(tiles) / 4))))
    for i, t in enumerate(tiles):
        sh.paste(t, ((i % 4) * 512, (i // 4) * 512))
    sh.save(path, quality=80)


def crop_sheet(items, path):
    from PIL import Image, ImageDraw
    tiles = []
    for cr, title in items:
        t = Image.fromarray(cr.transpose(1, 2, 0)).resize((192, 192), Image.BILINEAR)
        ImageDraw.Draw(t).text((3, 3), title[:40], fill=(255, 255, 0))
        tiles.append(t)
    cols = 8
    sh = Image.new('RGB', (cols * 192, 192 * int(np.ceil(len(tiles) / cols))))
    for i, t in enumerate(tiles):
        sh.paste(t, ((i % cols) * 192, (i // cols) * 192))
    sh.save(path, quality=80)


# ---------------------------------------------------------------------------------------------- crop token stores
class CropStore:
    """All shards of one (store, variant): crop tokens with their frame index (in the frame store's order)."""

    def __init__(self, dom, tag, vn, dev, stage=True, on_gpu=False):
        import torch
        src = BEE / dom / tag
        shards = sorted(src.glob('shard*'), key=lambda p: int(p.name[5:]))
        assert shards, f'no shards in {src}'
        parts, toks = [], []
        crop0 = 0
        Cs, frs = [], []
        for sh in shards:
            d = sh / vn
            if stage:
                dst = R4.local_dir() / f'crops_{dom}_{tag}_{vn}_{sh.name}'
                if not dst.exists():
                    R4.atomic_copy(d, dst)
                d = dst
            C = pd.read_parquet(d / 'crops.parquet')
            tc = np.load(d / 'tok_crop.npy')
            parts.append({'tok_frame': C.f.values[tc] if len(tc) else np.zeros(0, np.int64),
                          'tok_crop': tc + crop0, 'tok_pos': np.load(d / 'tok_pos.npy'),
                          'tok_fdark': np.load(d / 'tok_fdark.npy')})
            toks.append(np.memmap(d / 'tok.f16', np.float16, 'r', shape=(len(tc), D)) if len(tc) else None)
            C['crop'] = C['crop'] + crop0
            Cs.append(C)
            crop0 += len(C)
            frs.append(pd.read_parquet(sh / 'frames.parquet'))
        self.C = pd.concat(Cs, ignore_index=True)
        self.Fr = pd.concat(frs, ignore_index=True).sort_values('f').reset_index(drop=True)
        self.tf = np.concatenate([p['tok_frame'] for p in parts]).astype(np.int64)
        self.tc = np.concatenate([p['tok_crop'] for p in parts]).astype(np.int64)
        self.pos = np.concatenate([p['tok_pos'] for p in parts]).astype(np.int64)
        self.fd = np.concatenate([p['tok_fdark'] for p in parts]).astype(np.float32)
        n = len(self.tf)
        nb = n * D * 2
        on_gpu = on_gpu and nb <= 0.45 * torch.cuda.get_device_properties(dev).total_memory
        self.T = torch.empty((n, D), dtype=torch.float16, device=dev if on_gpu else 'cpu',
                             pin_memory=not on_gpu)
        o = 0
        for t in toks:
            if t is None:
                continue
            for a in range(0, len(t), 1_000_000):
                b = min(len(t), a + 1_000_000)
                self.T[o + a:o + b] = torch.from_numpy(np.ascontiguousarray(t[a:b]))
            o += len(t)
        self.dev = dev
        log(f'crop store {dom}/{tag}/{vn}: {len(self.C):,} crops, {n:,} tokens ({nb / 2**30:.1f} GiB) on '
            f'{"GPU" if on_gpu else "pinned host RAM"}')

    def chunks(self, n=2_000_000):
        for a in range(0, len(self.tf), n):
            b = min(len(self.tf), a + n)
            yield a, b, self.T[a:b].to(self.dev, non_blocking=True)

    def codes(self, sae, norm, nf):
        """(nf, W) fp16 cuda: max over each frame's crop tokens (0 for frames without crop tokens)."""
        import torch
        W = sae.n_latents
        X = torch.zeros((nf, W), dtype=torch.float32, device=self.dev)
        with torch.no_grad():
            for a, b, z in self.chunks(500_000):
                c = sae.encode(norm(z.float()), mode='threshold')
                fl = torch.from_numpy(self.tf[a:b]).to(self.dev)
                X.index_reduce_(0, fl, c, 'amax', include_self=True)
        return X.half()


# ---------------------------------------------------------------------------------------------- training
def sae_dir(dom, arm, s):
    return BEE / dom / 'sae' / f'{arm}_s{s}'


def load_sae(dom, arm, s, dev):
    from src.eci.sae import load_sae as ls
    sae, norm, _ = ls(sae_dir(dom, arm, s) / 'sae_theta.pt', dev)
    return sae, norm


def pool_stores(dom, dev, variants):
    """Frame-token pools (mice round-2 pool; ants v2 + v3 freeze pools) + crop pools per variant."""
    if dom == 'mice':
        St = {'mice': FZ.mice_pool(dev)}
        tags = {'mice': 'pool'}
    else:
        St = FZ.ants_stores('pool', dev, on_gpu=False)
        tags = {s: f'pool_{s}' for s in St}
    Cr = {vn: {s: CropStore(dom, tags[s], vn, dev) for s in St} for vn in variants}
    return St, Cr


def role_of_frames(dom, St_s):
    return St_s.F.role.values


def build_inputs(dom, St, Cr, arm, seed_mix=0):
    """{role: (n, 1024) fp16} SAE inputs of one arm ('P0' frames, 'P1'/'P3' frames + mixed crops, 'P2c' crops)."""
    import torch
    dev = FZ.dev_of()
    out = {}
    for r in ('train', 'calib', 'eval'):
        fsel = {}
        nfr = 0
        if arm in ('P0', 'P1', 'P3'):
            for s, st in St.items():
                fsel[s] = st.F.role.values[st.tf] == r
                nfr += int(fsel[s].sum())
        csel = {}
        if arm in ('P1', 'P3', 'P2c'):
            vn = 'can' if arm == 'P3' else 'rot'
            ix = {s: np.flatnonzero(st.F.role.values[Cr[vn][s].tf] == r) for s, st in St.items()}
            ntot = sum(len(v) for v in ix.values())
            cap = nfr if arm != 'P2c' else ntot
            g = np.random.default_rng(seed_mix)
            keep_all = np.sort(g.choice(ntot, cap, replace=False)) if ntot > cap else np.arange(ntot)
            o = 0
            for s in St:
                kk = keep_all[(keep_all >= o) & (keep_all < o + len(ix[s]))] - o
                o += len(ix[s])
                csel[s] = ix[s][kk]
        ncr = sum(len(v) for v in csel.values())
        X = torch.empty((nfr + ncr, D), dtype=torch.float16, device=dev if r == 'train' else 'cpu')
        o = 0
        for s, st in St.items():
            if s not in fsel:
                continue
            for f0, f1, a, b, z in st.chunks(256):
                m = fsel[s][a:b]
                if m.any():
                    x = z[torch.from_numpy(m).to(z.device)]
                    X[o:o + len(x)] = x.to(X.device)
                    o += len(x)
        for s, ixs in csel.items():
            cs = Cr['can' if arm == 'P3' else 'rot'][s]
            for a in range(0, len(ixs), 2_000_000):
                sel = torch.from_numpy(ixs[a:a + 2_000_000])
                x = cs.T[sel] if cs.T.device.type == 'cpu' else cs.T[sel.to(cs.T.device)]
                X[o:o + len(x)] = x.to(X.device)
                o += len(x)
        assert o == len(X)
        out[r] = X
        log(f'{dom} {arm} {r}: {len(X):,} tokens ({nfr:,} frame tokens, {ncr:,} crop tokens)')
    return out


def cmd_train(args):
    import torch
    dev = FZ.dev_of()
    dom = args.dom
    arms = args.arms.split(',')
    seeds = FZ.parse_seeds(args.seeds, SEEDS)
    need = sorted({('can' if a == 'P3' else 'rot') for a in arms if a in ('P1', 'P2', 'P3', 'P2c')})
    St, Cr = pool_stores(dom, dev, need)
    res_f = OUT / dom / f'train_{args.tag or "_".join(arms)}_{args.seeds.replace(",", "_")}.json'
    res = json.loads(res_f.read_text()) if res_f.exists() else {}
    for arm in arms:
        sub = ['P2c'] if arm == 'P2' else [arm]
        for am in sub:
            todo = [s for s in seeds if args.force or not (sae_dir(dom, am, s) / 'sae_theta.pt').exists()]
            if not todo:
                continue
            t0 = time.time()
            X = build_inputs(dom, St, Cr, am)
            W, pref = (W_CROP, PREF_CROP) if am == 'P2c' else (W_MAIN, PREF_MAIN)
            log(f'{dom} {am}: inputs built in {time.time() - t0:.0f}s; {FZ.gpu_line()}')
            for s in todo:
                cfg = {**R3.BASE, 'W': W, 'prefixes': pref, 'prep': 'id', 'sel': 'all', 'sampler': 'uniform',
                       'arm_pair': am}
                ti = R3.train_arm(torch, cfg, s, X['train'], X['calib'], X['eval'], sae_dir(dom, am, s), dev)
                res[f'{am}_s{s}'] = {k: ti[k] for k in ('train_time_s', 'steps_per_s', 'n_train_tokens', 'theta_star',
                                                        'eval_fve', 'eval_l0', 'dead_at_end_train')}
                res[f'{am}_s{s}']['n_calib_tokens'] = int(len(X['calib']))
                log(f'{dom} {am}_s{s}: {ti["train_time_s"]:.0f}s, eval FVE {ti["eval_fve"]:.4f}, L0 '
                    f'{ti["eval_l0"]:.2f}, dead {ti["dead_at_end_train"]}')
                res_f.parent.mkdir(parents=True, exist_ok=True)
                res_f.write_text(json.dumps(res, indent=1, default=float))
            del X
            torch.cuda.empty_cache()


def fve_of(dom, arm, s):
    for f in (OUT / dom).glob('train_*.json'):
        d = json.loads(f.read_text())
        if f'{arm}_s{s}' in d:
            return d[f'{arm}_s{s}']['eval_fve']
    return float('nan')


# ---------------------------------------------------------------------------------------------- scoring
def scored_stores(dom, dev, variants):
    if dom == 'mice':
        St = {'mice': FZ.mice_diag(dev)}
        tags = {'mice': 'diag'}
    else:
        St = FZ.ants_stores('scored', dev, on_gpu=True)
        tags = {s: f'scored_{s}' for s in St}
    Cr = {vn: {s: CropStore(dom, tags[s], vn, dev) for s in St} for vn in variants}
    return St, Cr


def arm_codes(dom, arm, s, St, Cr, dev):
    """{series: (nf, W) fp16 cuda} frame codes of one arm / seed."""
    import torch
    out = {}
    if arm == 'P2':
        sae, norm = load_sae(dom, 'P0', s, dev)
        sc, nc = load_sae(dom, 'P2c', s, dev)
        for k, st in St.items():
            out[k] = torch.cat([st.codes(sae, norm, 0, 'all'), Cr['rot'][k].codes(sc, nc, st.nf)], 1)
        return out
    sae, norm = load_sae(dom, arm, s, dev)
    for k, st in St.items():
        X = st.codes(sae, norm, 0, 'all')
        if ARMS[arm]['crop']:
            X = torch.maximum(X, Cr[ARMS[arm]['crop']][k].codes(sae, norm, st.nf))
        out[k] = X
    return out


def mice_setup(St, dev):
    import torch
    Y, unit, ann = FZ.mice_labels(St.F)
    sc = json.loads((N.OUT / 'score.json').read_text())
    units, SPL = sc['units'], {k_: np.array(v_) for k_, v_ in sc['splits'].items()}
    uidx = pd.Series(range(len(units)), index=units).loc[unit].values
    sets = {'all': np.arange(St.nf), 'consistent': np.flatnonzero(pd.Series(ann).isin(CONSISTENT).values)}
    half = {k: {s_: h[uidx[ix]] for s_, h in SPL.items()} for k, ix in sets.items()}
    Yg = {k: torch.from_numpy(Y[ix]).to(dev) for k, ix in sets.items()}
    ixg = {k: torch.from_numpy(ix).to(dev) for k, ix in sets.items()}
    return sets, half, Yg, ixg


def ants_setup(St, dev):
    import torch
    lab = {}
    for ser, s_ in St.items():
        ok, lid_ok, Y, _ = FZ.ants_labels(ser)
        rows = s_.F.row.values
        assert ok[rows].all()
        obs = s_.F.obs.values
        units, SPL = FZ.splits_of(set(obs))
        uidx = pd.Series(range(len(units)), index=units).loc[obs].values
        labs = list(FZ.ALAB) if lid_ok[rows].all() else list(FZ.ALAB[:3])
        lab[ser] = (torch.from_numpy(Y[rows][:, :len(labs)]).to(dev), {k: h[uidx] for k, h in SPL.items()}, labs)
    return lab


def cmd_score(args):
    import torch
    dev = FZ.dev_of()
    dom = args.dom
    arms = args.arms.split(',')
    seeds = FZ.parse_seeds(args.seeds, SEEDS)
    need = sorted({('can' if a == 'P3' else 'rot') for a in arms if a != 'P0'})
    St, Cr = scored_stores(dom, dev, need)
    if dom == 'mice':
        sets, half, Yg, ixg = mice_setup(St['mice'], dev)
    else:
        lab = ants_setup(St, dev)
    for arm in arms:
        res_f = OUT / dom / f'score_{arm}.json'
        res = json.loads(res_f.read_text()) if res_f.exists() else {}
        for s in seeds:
            name = f'{arm}_s{s}'
            if name in res and not args.force:
                continue
            t0 = time.time()
            Xs = arm_codes(dom, arm, s, St, Cr, dev)
            te = time.time() - t0
            if dom == 'mice':
                X = Xs['mice']
                out = {}
                for k in sets:
                    Xk = X if k == 'all' else X.index_select(0, ixg[k])
                    o = crossfit(Xk, Yg[k], half[k], MLAB)
                    o['primary'] = float(np.mean([o['mean']['nose_nose']['R'], o['mean']['nose_tail']['R']]))
                    out[k] = FZ.compact(o) | {'primary': o['primary']}
                out['eval_fve'] = fve_of(dom, 'P0' if arm == 'P2' else arm, s)
                out['codes_s'] = te
                out['frame_fire_rate'] = float((X > 0).float().mean())
                m = out['all']['mean']
                log(f'mice {name}: primary R~ {out["all"]["primary"]:.4f} | nn R~ {m["nose_nose"]["R"]:.4f} AUROC '
                    f'{m["nose_nose"]["A"]:.4f} | nt R~ {m["nose_tail"]["R"]:.4f} AUROC {m["nose_tail"]["A"]:.4f} | '
                    f'consistent primary R~ {out["consistent"]["primary"]:.4f}; codes {te:.0f}s; {FZ.gpu_line()}')
            else:
                out = {}
                for ser, X in Xs.items():
                    Yg_, h, labs = lab[ser]
                    out[ser] = FZ.compact(crossfit(X, Yg_, h, labs))
                out['primary'] = float(np.mean([out[x]['mean']['groom_any']['R'] for x in FZ.ASER]))
                out['lid_any'] = out['v3']['mean']['on_lid_any']['R']
                out['eval_fve'] = fve_of(dom, 'P0' if arm == 'P2' else arm, s)
                log(f'ants {name}: primary R~ (groom_any, mean v2 v3) {out["primary"]:.4f} | v2 groom_any R~ '
                    f'{out["v2"]["mean"]["groom_any"]["R"]:.4f} | v3 {out["v3"]["mean"]["groom_any"]["R"]:.4f} | v3 '
                    f'on_lid_any R~ {out["lid_any"]:.4f}; codes {te:.0f}s')
            res[name] = out
            res_f.parent.mkdir(parents=True, exist_ok=True)
            res_f.write_text(json.dumps(res, indent=1, default=float))
            del Xs
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------------------------- contacts
def cmd_contacts(args):
    """Contact detection numbers (labels used only to describe the detector)."""
    out = {}
    Fm = pd.read_parquet(R2.BEE2 / 'diag' / 'frames.parquet')
    for tag, Fs in (('diag', Fm), ('pool', None)):
        Fr = pd.concat([pd.read_parquet(p / 'frames.parquet') for p in sorted((BEE / 'mice' / tag).glob('shard*'))])
        Fr = Fr.sort_values('f').reset_index(drop=True)
        r = {'n_frames': len(Fr), 'contact_frame_share': float((Fr.n_contacts > 0).mean()),
             'contacts_per_frame': float(Fr.n_contacts.mean()), 'crops_per_frame': float(Fr.n_crops.mean()),
             'crops_per_contact_frame': float(Fr.n_crops[Fr.n_contacts > 0].mean()),
             'merge_contact_share': float(Fr.n_merge_contacts.sum() / max(Fr.n_contacts.sum(), 1)),
             'blobs_per_frame': float(Fr.n_blobs.mean()), 'parts_per_frame': float(Fr.n_parts.mean()),
             'parts_eq_4_share': float((Fr.n_parts == 4).mean()),
             'n_contacts_hist': Fr.n_contacts.value_counts().sort_index().to_dict()}
        if tag == 'diag':
            assert len(Fr) == len(Fm) and (Fr.f.values == np.arange(len(Fm))).all()
            Y, _ = R2.labels_of(Fm.row.values)
            has = Fr.n_contacts.values > 0
            for li, lb in enumerate(MLAB):
                r[f'recall_{lb}'] = float(has[Y[:, li]].mean())
                r[f'n_{lb}'] = int(Y[:, li].sum())
            unl = ~Y[:, :2].any(1)
            r['contact_rate_unlabelled'] = float(has[unl].mean())
            r['contact_rate_all'] = float(has.mean())
        out[f'mice_{tag}'] = r
        log(f'mice {tag}: ' + json.dumps({k: v for k, v in r.items() if k != 'n_contacts_hist'}))
    for ser in ('v2', 'v3'):
        for st in ('scored', 'pool'):
            tag = f'{st}_{ser}'
            shards = sorted((BEE / 'ants' / tag).glob('shard*'))
            if not shards:
                continue
            Fr = pd.concat([pd.read_parquet(p / 'frames.parquet') for p in shards]).sort_values('f').reset_index(drop=True)
            r = {'n_frames': len(Fr), 'contact_frame_share': float((Fr.n_contacts > 0).mean()),
                 'contacts_per_frame': float(Fr.n_contacts.mean()), 'crops_per_frame': float(Fr.n_crops.mean()),
                 'merge_contact_share': float(Fr.n_merge_contacts.sum() / max(Fr.n_contacts.sum(), 1)),
                 'parts_eq_3_share': float((Fr.n_parts == 3).mean())}
            if st == 'scored':
                Fs = pd.read_parquet(FZ.BEE / 'ants' / tag / 'frames.parquet')
                ok, lid_ok, Y, _ = FZ.ants_labels(ser)
                Yr = Y[Fs.row.values]
                has = Fr.n_contacts.values > 0
                for li, lb in enumerate(FZ.ALAB[:4]):
                    if lb == 'on_lid_any' and ser == 'v2':
                        continue
                    r[f'recall_{lb}'] = float(has[Yr[:, li]].mean())
                r['contact_rate_no_groom'] = float(has[~Yr[:, 0]].mean())
            out[f'ants_{tag}'] = r
            log(f'ants {tag}: ' + json.dumps(r))
    (OUT / 'contacts.json').write_text(json.dumps(out, indent=1, default=float))


# ---------------------------------------------------------------------------------------------- sheets (mice + ants)
def deploy_seed(dom, arm, seeds):
    fa = 'P0' if arm == 'P2' else arm
    fv = {s: fve_of(dom, fa, s) for s in seeds[:3]}
    return int(max(fv, key=lambda s: fv[s] if np.isfinite(fv[s]) else -1)), fv


def cmd_sheets(args):
    import torch
    dev = FZ.dev_of()
    dom = args.dom
    arms = args.arms.split(',')
    seeds = FZ.parse_seeds(args.seeds, SEEDS)
    need = sorted({('can' if a == 'P3' else 'rot') for a in arms if a != 'P0'})
    St, Cr = scored_stores(dom, dev, need)
    out_f = OUT / dom / 'locality.json'
    out = json.loads(out_f.read_text()) if out_f.exists() else {}
    (OUT / dom / 'sheets').mkdir(parents=True, exist_ok=True)
    if dom == 'mice':
        V = {v['obs']: v for v in N.load_plan()['videos']}
        ann = pd.read_csv(REPO / N.C['ann'], usecols=['Y_nn', 'Y_np', 'Y_nt'])
        sc = json.loads((N.OUT / 'score.json').read_text())
        hmap = dict(zip(sc['units'], sc['splits']['rnd0']))
        jobs = [('mice', lb, want) for lb, want in (('nose_nose', ('nn', 'np')), ('nose_tail', ('nt',)))]
    else:
        jobs = [('v2', 'groom_any', ('Y2F', 'B2F')), ('v3', 'groom_any', ('Y2F', 'B2F')),
                ('v3', 'on_lid_any', ('YOL', 'BOL', 'FOL'))]
    for arm in arms:
        scores = json.loads((OUT / dom / f'score_{arm}.json').read_text())
        s, fv = deploy_seed(dom, arm, seeds)
        Xs = arm_codes(dom, arm, s, St, Cr, dev)
        for ser, lb, want in jobs:
            key = f'{arm}_s{s}|{ser}|{lb}' if dom != 'mice' else f'{arm}_s{s}|{lb}'
            if key in out and not args.force:
                continue
            st = St[ser]
            if dom == 'mice':
                r0 = scores[f'{arm}_s{s}']['all']['splits']['rnd0'][lb]['neurons'][0]
                hold = np.array([hmap[o.rsplit('_', 2)[0]] == 1 for o in st.F.obs.values])

                def labf(f, st=st):
                    y = ann.iloc[int(st.F.row.values[f])]
                    return ''.join(c for c, col in (('nn', 'Y_nn'), ('np', 'Y_np'), ('nt', 'Y_nt')) if y[col] > 0) or '-'

                def dec(f, st=st):
                    o, k = st.F.obs.values[f], int(st.F.k.values[f])
                    v = V[o]
                    fr = R4.decode(v['path'], v['offset'] + N.C['step'] * k, 1, N.C['step'], 1024, v['hflip'],
                                   N.C['fps'], threads=4)[0]
                    return np.rot90(fr, v['rot'], axes=(0, 1)).transpose(2, 0, 1).copy()
                scale = 1024 / 2064
            else:
                r0 = scores[f'{arm}_s{s}'][ser]['splits']['rnd0'][lb]['neurons'][0]
                units, SPL = FZ.splits_of(set(st.F.obs.values))
                hm = dict(zip(units, SPL['rnd0']))
                hold = np.array([hm[o] == 1 for o in st.F.obs.values])
                c = R4.DOMS[DOM[ser]['dom']]
                A_ = pd.read_csv(REPO / c['eci'] / 'annotations.csv', usecols=['Y_Y2F', 'Y_B2F', 'Y_YOL', 'Y_BOL', 'Y_FOL'])
                Vp = {v['obs']: v for v in R4.load_plan(DOM[ser]['dom'])['videos']}

                def labf(f, st=st, A_=A_):
                    y = A_.iloc[int(st.F.row.values[f])]
                    return '+'.join(c_[2:] for c_ in ('Y_Y2F', 'Y_B2F', 'Y_YOL', 'Y_BOL', 'Y_FOL') if y[c_] > 0) or '-'

                def dec(f, st=st, Vp=Vp, c=c, ser=ser):
                    v = Vp[st.F.obs.values[f]]
                    k = int(st.F.k.values[f])
                    return R4.decode(v['path'], R4.voff(c, v) + c['step'] * k, 1, c['step'], DOM[ser]['S'], v['hflip'],
                                     c['fps'], threads=4)[0].transpose(2, 0, 1).copy()
                scale = DOM[ser]['S'] / Vp[st.F.obs.values[0]]['probe']['w']
            Xj = Xs[ser][:, r0].float().cpu().numpy()
            frames = FZ.top_heldout(Xj, hold, st.F.obs.values)
            summ, recs = pair_sheet(dom, ser, arm, s, r0, st, Cr, frames, labf, want, dec, scale, dev,
                                    OUT / dom / 'sheets' / f'{arm}_s{s}_{ser}_{lb}_n{r0}',
                                    f'candidate P {arm} s{s} {ser} {lb} neuron {r0} (picked on half 0 of rnd0; top 24 '
                                    f'held-out frames, <= 1 per video; crop tokens painted at their frame position)')
            out[key] = {'seed': s, 'fve': fv, 'neuron': int(r0), 'summary': summ, 'frames': recs}
            log(f'{key}: ' + json.dumps(summ))
            out_f.write_text(json.dumps(out, indent=1, default=float))
        del Xs
        torch.cuda.empty_cache()


def pair_sheet(dom, ser, arm, s, jg, st, Cr, frames, labf, want, dec, scale, dev, fn, title):
    """Per top frame: activations of neuron jg over the frame tokens and the frame's crop tokens -> spread / peak
    numbers and a contact sheet (128 x 128 paint grid over the displayed frame)."""
    import torch
    G = st.G
    S = G * 16
    sep = arm == 'P2'
    if sep:
        sae_f, norm_f = load_sae(dom, 'P0', s, dev)
        sae_c, norm_c = load_sae(dom, 'P2c', s, dev)
        crop_side = jg >= W_MAIN
        j = jg - W_MAIN if crop_side else jg
    else:
        sae_f, norm_f = load_sae(dom, arm, s, dev)
        sae_c, norm_c = sae_f, norm_f
        crop_side = None
        j = jg
    vn = 'can' if arm == 'P3' else 'rot'
    cs = Cr[vn][ser] if arm != 'P0' else None
    cfg = ser_cfg(ser)
    GP = 2 * G
    tiles, recs = [], []
    for f in frames:
        a0, b0 = int(st.starts[f]), int(st.starts[f + 1])
        z = torch.from_numpy(np.ascontiguousarray(st.tok[a0:b0])).to(dev)
        with torch.no_grad():
            af = sae_f.encode(norm_f(z.float()), mode='threshold')[:, j].cpu().numpy() if crop_side in (None, False) \
                else np.zeros(b0 - a0, np.float32)
        posf = st.pos[a0:b0]
        cellf = posf  # 64-grid (or G-grid) cell index of frame tokens
        fdf = st.fd[a0:b0]
        ac, cellc, fdc, xyc = np.zeros(0, np.float32), np.zeros(0, np.int64), np.zeros(0, np.float32), np.zeros((0, 2))
        if cs is not None and crop_side in (None, True):
            ti = np.flatnonzero(cs.tf == f)
            if len(ti):
                zc = cs.T[torch.from_numpy(ti)].to(dev)
                with torch.no_grad():
                    ac = sae_c.encode(norm_c(zc.float()), mode='threshold')[:, j].cpu().numpy()
                C = cs.C.set_index('crop').loc[cs.tc[ti]]
                xyc = np.zeros((len(ti), 2))
                for q, (cid, row) in enumerate(C.iterrows()):
                    xyc[q] = PR.crop_token_to_frame([cs.pos[ti[q]]], cfg['S_c'] // 16, (row.cx, row.cy), row.theta,
                                                    cfg['side'])[0]
                xyc = xyc * scale  # displayed-frame px
                gx = np.clip((xyc[:, 0] // 16).astype(np.int64), 0, G - 1)
                gy = np.clip((xyc[:, 1] // 16).astype(np.int64), 0, G - 1)
                cellc = gy * G + gx
                fdc = cs.fd[ti]
        allv = np.r_[af, ac]
        mx = float(allv.max()) if len(allv) else 0.0
        if mx > 0:
            hi_f, hi_c = af >= 0.5 * mx, ac >= 0.5 * mx
            cells = np.unique(np.r_[cellf[hi_f], cellc[hi_c]])
            ip = int(np.argmax(allv))
            peak_crop = ip >= len(af)
            peak_fd = float(fdc[ip - len(af)] if peak_crop else fdf[ip])
        else:
            hi_f, hi_c, cells, peak_crop, peak_fd = af > 0, ac > 0, np.zeros(0), False, float('nan')
        lab = labf(f)
        r = {'frame': int(f), 'obs': st.F.obs.values[f], 'k': int(st.F.k.values[f]), 'value': mx, 'label': lab,
             'positive': bool(any(x in lab for x in want)), 'n_half_max_cells': int(len(cells)),
             'n_half_max_frame_tokens': int(hi_f.sum()), 'n_half_max_crop_tokens': int(hi_c.sum()),
             'peak_is_crop': bool(peak_crop), 'peak_fdark': peak_fd, 'n_crop_tokens': int(len(ac)),
             'n_frame_tokens': int(b0 - a0)}
        recs.append(r)
        # paint grid (GP x GP over the S x S displayed frame): frame tokens fill their 16 px cell, crop tokens a 16 px
        # axis-aligned square at their mapped centre (crop token side ~ 14 px at 1024; rotation ignored in the drawing)
        H = np.zeros((GP, GP), np.float32)
        k_ = GP // G
        for p, v in zip(posf, af):
            if v > 0:
                y, x = divmod(int(p), G)
                H[y * k_:(y + 1) * k_, x * k_:(x + 1) * k_] = np.maximum(H[y * k_:(y + 1) * k_, x * k_:(x + 1) * k_], v)
        cp = S / GP
        for (x, y), v in zip(xyc, ac):
            if v > 0:
                y0, y1 = int(np.clip((y - 8) // cp, 0, GP - 1)), int(np.clip((y + 8) // cp, 0, GP - 1))
                x0, x1 = int(np.clip((x - 8) // cp, 0, GP - 1)), int(np.clip((x + 8) // cp, 0, GP - 1))
                H[y0:y1 + 1, x0:x1 + 1] = np.maximum(H[y0:y1 + 1, x0:x1 + 1], v)
        tiles.append((dec(f), H, {'obs': r['obs'], 'k': r['k'], 'label': lab, 'stored': mx,
                                  'n_half_max': r['n_half_max_cells']}))
    N.contact_sheet(tiles, Path(str(fn) + '.jpg'), title, GP)
    R = pd.DataFrame(recs)
    summ = {'share_label_pos': float(R.positive.mean()), 'label_counts': R.label.value_counts().to_dict(),
            'n_half_max_cells': float(R.n_half_max_cells.mean()),
            'n_half_max_frame_tokens': float(R.n_half_max_frame_tokens.mean()),
            'n_half_max_crop_tokens': float(R.n_half_max_crop_tokens.mean()),
            'peak_is_crop_share': float(R.peak_is_crop.mean()), 'peak_fdark': float(R.peak_fdark.mean()),
            'single_cell_frames': int((R.n_half_max_cells == 1).sum()),
            'frames_with_crops': int((R.n_crop_tokens > 0).sum()), 'sheet': str(fn) + '.jpg'}
    return summ, recs


# ---------------------------------------------------------------------------------------------- cos (supervised diagnosis)
def cmd_cos(args):
    """DIAGNOSIS (supervised; nothing enters any SAE): the T-max direction w (r3 cmd_tmax recipe: one linear direction
    on standardised tokens, frame score = max over the frame's tokens, fitted on pools half 0 of rnd0, 1500 steps,
    seed 0) on (i) the diag FRAME tokens and (ii) the diag CROP tokens ('rot'; frames without crops score -1e4), its
    held-out R~ / AUROC (half 1), and the best |cos| of each listed dictionary's decoder atoms with w (raw token
    space: W_dec / TokenNorm scale)."""
    import torch
    dev = FZ.dev_of()
    names = args.names.split(',')
    St = FZ.mice_diag(dev)
    Y, unit = R2.labels_of(St.F.row.values)
    sc = json.loads((N.OUT / 'score.json').read_text())
    hmap = dict(zip(sc['units'], sc['splits']['rnd0']))
    hold = np.array([hmap[o.rsplit('_', 2)[0]] == 1 for o in St.F.obs.values])
    tr = torch.from_numpy(np.flatnonzero(~hold)).to(dev)
    te = np.flatnonzero(hold)
    T = St.T
    ii = torch.from_numpy(np.random.default_rng(0).choice(len(St.tf), 1_000_000, replace=False))
    sub = T[ii.to(T.device)].to(dev).float()
    mu, sd = sub.mean(0), sub.std(0) + 1e-3
    del sub
    out_f = OUT / 'mice' / 'cos.json'
    out = json.loads(out_f.read_text()) if out_f.exists() else {}
    atoms = {}
    for nm in names:
        arm, s = nm.split('_s')
        sae, norm = load_sae('mice', arm, int(s), dev)
        Wd = sae.W_dec.detach().float() / norm.scale.float()
        atoms[nm] = torch.nn.functional.normalize(Wd, dim=1)

    def fit(view, Tsrc, tf, nf):
        Tn = torch.empty((len(tf), D), dtype=torch.float16, device=dev)
        for a in range(0, len(tf), 1_000_000):
            Tn[a:a + 1_000_000] = ((Tsrc[a:a + 1_000_000].to(dev).float() - mu) / sd).half()
        bags = R2.Bags(torch, Tn, tf, nf, dev)
        res = {}
        for li, lb in enumerate(('nose_nose', 'nose_tail')):
            torch.manual_seed(0)
            f = R2.fit_probe(torch, 'max', bags, None, Y[:, li], tr, dev, steps=1500)
            m = R2.metrics(f(torch.from_numpy(te).to(dev)), Y[te, li])
            w = f.net.weight[0].detach()
            wraw = torch.nn.functional.normalize(w / sd, dim=0)
            cos = {}
            for nm, A in atoms.items():
                c_ = A @ wraw
                cos[nm] = {'max_abs_cos': float(c_.abs().max()), 'argmax': int(c_.abs().argmax()),
                           'n_atoms_cos_gt_0.3': int((c_.abs() > 0.3).sum())}
            res[lb] = {'heldout_rnd0_half1': m, 'dict_alignment': cos, 'w_raw': wraw.cpu().numpy().tolist()}
            log(f'T-max [{view}] {lb}: held-out R~ {m["R"]:.4f} AUROC {m["A"]:.4f}; ' +
                json.dumps({k: round(v['max_abs_cos'], 4) for k, v in cos.items()}))
        del Tn, bags
        torch.cuda.empty_cache()
        return res
    out['frame_tokens'] = fit('frame', St.T, St.tf, St.nf)
    St.T = None
    torch.cuda.empty_cache()
    cs = CropStore('mice', 'diag', 'rot', dev)
    out['crop_tokens'] = fit('crop', cs.T, cs.tf, St.nf)
    for lb in ('nose_nose', 'nose_tail'):
        a, b = (torch.tensor(out[v][lb]['w_raw']) for v in ('frame_tokens', 'crop_tokens'))
        out[f'cos_frame_vs_crop_direction_{lb}'] = float(a @ b)
    for v in ('frame_tokens', 'crop_tokens'):
        for lb in ('nose_nose', 'nose_tail'):
            out[v][lb].pop('w_raw')
    out_f.parent.mkdir(parents=True, exist_ok=True)
    out_f.write_text(json.dumps(out, indent=1, default=float))


# ---------------------------------------------------------------------------------------------- decide
def cmd_decide(args):
    mean_sd = FZ.mean_sd
    M, A = {}, {}
    for arm in ARMS:
        for d, D_ in (('mice', M), ('ants', A)):
            f = OUT / d / f'score_{arm}.json'
            if f.exists():
                D_.update(json.loads(f.read_text()))

    def mp(arm, seeds, q='primary', set_='all'):
        return [M[f'{arm}_s{s}'][set_]['primary'] if q == 'primary' else M[f'{arm}_s{s}'][set_]['mean'][q]['R']
                for s in seeds if f'{arm}_s{s}' in M]

    def ap(arm, seeds, key='primary'):
        return [A[f'{arm}_s{s}'][key] for s in seeds if f'{arm}_s{s}' in A]
    dec = {'rule': 'scripts/eci/v2/pair/pair.py docstring (pre-registered 2026-10-21)'}
    rnd = {arm: mean_sd(mp(arm, SEEDS)) for arm in ARMS}
    dec['round_40_45'] = {'mice_primary_R': rnd, 'ants_primary_R': {arm: mean_sd(ap(arm, SEEDS)) for arm in ARMS},
                          'mice_vs_P0': {arm: welch(mp(arm, SEEDS), mp('P0', SEEDS)) for arm in ARMS
                                         if arm != 'P0' and len(mp(arm, SEEDS)) > 1 and len(mp('P0', SEEDS)) > 1},
                          'ants_vs_P0': {arm: welch(ap(arm, SEEDS), ap('P0', SEEDS)) for arm in ARMS
                                         if arm != 'P0' and len(ap(arm, SEEDS)) > 1 and len(ap('P0', SEEDS)) > 1},
                          'P1_minus_P3_mice': welch(mp('P1', SEEDS), mp('P3', SEEDS))
                          if len(mp('P1', SEEDS)) > 1 and len(mp('P3', SEEDS)) > 1 else None,
                          'P1_minus_P3_ants': welch(ap('P1', SEEDS), ap('P3', SEEDS))
                          if len(ap('P1', SEEDS)) > 1 and len(ap('P3', SEEDS)) > 1 else None}
    win = max(('P1', 'P2'), key=lambda a: rnd[a][0] if rnd[a][2] else -9)
    dec['winner'] = win
    c_m, c_a = mp(win, SEEDS_C), ap(win, SEEDS_C)
    r_m, r_a = mp('P0', SEEDS_C), ap('P0', SEEDS_C)
    conf = {'n_seeds': {'winner_mice': len(c_m), 'P0_mice': len(r_m), 'winner_ants': len(c_a), 'P0_ants': len(r_a)}}
    if len(c_m) > 1 and len(r_m) > 1:
        conf['mice_primary'] = {'winner': mean_sd(c_m), 'P0': mean_sd(r_m), 'diff': welch(c_m, r_m)}
        for q in ('nose_nose', 'nose_tail', 'nn_mutual', 'np_directional'):
            conf[f'mice_{q}'] = {'winner': mean_sd(mp(win, SEEDS_C, q)), 'P0': mean_sd(mp('P0', SEEDS_C, q))}
        conf['mice_consistent_primary'] = {'winner': mean_sd(mp(win, SEEDS_C, set_='consistent')),
                                           'P0': mean_sd(mp('P0', SEEDS_C, set_='consistent'))}
    if len(c_a) > 1 and len(r_a) > 1:
        conf['ants_primary'] = {'winner': mean_sd(c_a), 'P0': mean_sd(r_a), 'diff': welch(c_a, r_a)}
    L = json.loads((OUT / 'mice' / 'locality.json').read_text()) if (OUT / 'mice' / 'locality.json').exists() else {}
    sw, _ = deploy_seed('mice', win, SEEDS_C)
    s0, _ = deploy_seed('mice', 'P0', SEEDS_C)
    sheets = {}
    c_ok = True
    for lb in ('nose_nose', 'nose_tail'):
        kw, k0 = f'{win}_s{sw}|{lb}', f'P0_s{s0}|{lb}'
        if kw not in L or k0 not in L:
            c_ok = None
            continue
        a_, b_ = L[kw]['summary'], L[k0]['summary']
        ok = (a_['share_label_pos'] >= b_['share_label_pos'] and a_['n_half_max_cells'] <= max(12, b_['n_half_max_cells'])
              and a_['peak_fdark'] >= b_['peak_fdark'] - 0.10)
        sheets[lb] = {'winner': {k: a_[k] for k in ('share_label_pos', 'n_half_max_cells', 'peak_fdark',
                                                    'peak_is_crop_share')},
                      'P0': {k: b_[k] for k in ('share_label_pos', 'n_half_max_cells', 'peak_fdark')}, 'ok': bool(ok)}
        c_ok = c_ok and ok if c_ok is not None else None
    conf['interpretability'] = sheets
    crit = {'a': bool('mice_primary' in conf and conf['mice_primary']['diff'][0] >= 0.03 and
                      conf['mice_primary']['diff'][1] > 0),
            'b': bool('ants_primary' in conf and conf['ants_primary']['diff'][1] >= -0.01),
            'c': c_ok}
    conf['criteria'] = crit
    dec['confirmation_50_55'] = conf
    dec['ADOPT_candidate_P'] = bool(crit['a'] and crit['b'] and crit['c'])
    (OUT / 'decision.json').write_text(json.dumps(dec, indent=1, default=float))
    table(M, A, dec, L)
    print(json.dumps(dec, indent=1, default=float))


def table(M, A, dec, L):
    f = lambda x: f'{x[0]:.3f} ± {x[1]:.3f}' if x[2] > 1 else (f'{x[0]:.3f}' if x[2] else '-')  # noqa: E731
    ms = FZ.mean_sd
    rows = ['# ECI v2 candidate P (pair-centred crops) — results', '',
            'Rule: scripts/eci/v2/pair/pair.py docstring (pre-registered). R~ = cross-fitted normalised R-precision; '
            'AUROC = cf AUROC; mean ± sd over SAE seeds. mice primary R~ = mean(R~ nose_nose, R~ nose_tail); ants '
            'primary R~ = mean over v2 / v3 of R~ groom_any.', '', '## Mice v1 (86,400 diagnosis frames)', '',
            '| arm | seeds | n | primary R~ | nose_nose R~ | nose_nose AUROC | nose_tail R~ | nose_tail AUROC | '
            'nn_mutual R~ | np_directional R~ | consistent primary R~ | held-out FVE |', '|' + '---|' * 12]
    for seeds, sn in ((SEEDS, '40-45'), (SEEDS_C, '50-55')):
        for arm in ARMS:
            rs = [M[f'{arm}_s{s}'] for s in seeds if f'{arm}_s{s}' in M]
            if not rs:
                continue
            c = lambda g: f(ms([g(r) for r in rs]))  # noqa: E731
            rows.append(f'| {arm} | {sn} | {len(rs)} | {c(lambda r: r["all"]["primary"])} | '
                        f'{c(lambda r: r["all"]["mean"]["nose_nose"]["R"])} | {c(lambda r: r["all"]["mean"]["nose_nose"]["A"])} | '
                        f'{c(lambda r: r["all"]["mean"]["nose_tail"]["R"])} | {c(lambda r: r["all"]["mean"]["nose_tail"]["A"])} | '
                        f'{c(lambda r: r["all"]["mean"]["nn_mutual"]["R"])} | {c(lambda r: r["all"]["mean"]["np_directional"]["R"])} | '
                        f'{c(lambda r: r["consistent"]["primary"])} | {c(lambda r: r.get("eval_fve"))} |')
    rows += ['', '## Ants v2 + v3', '', '| arm | seeds | n | primary R~ | v2 groom_any R~ | v2 AUROC | v3 groom_any R~ | '
             'v3 AUROC | v2 yellow / blue R~ | v3 yellow / blue R~ | v3 on_lid_any R~ | held-out FVE |', '|' + '---|' * 12]
    for seeds, sn in ((SEEDS, '40-45'), (SEEDS_C, '50-55')):
        for arm in ARMS:
            rs = [A[f'{arm}_s{s}'] for s in seeds if f'{arm}_s{s}' in A]
            if not rs:
                continue
            c = lambda g: f(ms([g(r) for r in rs]))  # noqa: E731
            rows.append(f'| {arm} | {sn} | {len(rs)} | {c(lambda r: r["primary"])} | {c(lambda r: r["v2"]["mean"]["groom_any"]["R"])} | '
                        f'{c(lambda r: r["v2"]["mean"]["groom_any"]["A"])} | {c(lambda r: r["v3"]["mean"]["groom_any"]["R"])} | '
                        f'{c(lambda r: r["v3"]["mean"]["groom_any"]["A"])} | {c(lambda r: r["v2"]["mean"]["groom_yellow"]["R"])} / '
                        f'{c(lambda r: r["v2"]["mean"]["groom_blue"]["R"])} | {c(lambda r: r["v3"]["mean"]["groom_yellow"]["R"])} / '
                        f'{c(lambda r: r["v3"]["mean"]["groom_blue"]["R"])} | {c(lambda r: r["lid_any"])} | {c(lambda r: r.get("eval_fve"))} |')
    for dom in ('mice', 'ants'):
        lf = OUT / dom / 'locality.json'
        if not lf.exists():
            continue
        rows += ['', f'## Interpretability ({dom}; deployment seed; 24 top held-out frames of the neuron picked on half 0 '
                 f'of rnd0)', '', '| set | neuron | share labelled | distinct half-max cells | half-max frame / crop tokens '
                 '| peak is a crop token | peak animal-pixel fraction | sheet |', '|---|---|---|---|---|---|---|---|']
        for k, v in json.loads(lf.read_text()).items():
            s_ = v['summary']
            rows.append(f'| {k} | n{v["neuron"]} | {s_["share_label_pos"]:.2f} | {s_["n_half_max_cells"]:.1f} | '
                        f'{s_["n_half_max_frame_tokens"]:.1f} / {s_["n_half_max_crop_tokens"]:.1f} | '
                        f'{s_["peak_is_crop_share"]:.2f} | {s_["peak_fdark"]:.2f} | {Path(s_["sheet"]).name} |')
    rows += ['', '## Decision', '', '```', json.dumps(dec, indent=1, default=float), '```']
    (OUT / 'table.md').write_text('\n'.join(rows) + '\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=('prep', 'contacts', 'train', 'score', 'sheets', 'cos', 'decide'))
    p.add_argument('--dom', default='mice')
    p.add_argument('--store', default='diag')
    p.add_argument('--series', default='v2')
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--nshards', type=int, default=1)
    p.add_argument('--arms', default='P0')
    p.add_argument('--seeds', default='')
    p.add_argument('--names', default='')
    p.add_argument('--tag', default='')
    p.add_argument('--max-videos', type=int, default=0)
    p.add_argument('--max-frames', type=int, default=0)
    p.add_argument('--force', action='store_true')
    p.add_argument('--selftest', action='store_true')
    args = p.parse_args()
    {'prep': cmd_prep, 'contacts': cmd_contacts, 'train': cmd_train, 'score': cmd_score, 'sheets': cmd_sheets,
     'cos': cmd_cos, 'decide': cmd_decide}[args.step](args)


if __name__ == '__main__':
    main()
