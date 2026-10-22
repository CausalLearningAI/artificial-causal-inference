"""
ECI v2 release check, PART 2 (DISCOVERY): does the N2 thin-part mask give the representation a TAIL neuron and an
ODOUR-BOX neuron that the deployed v1 representation lacks? v1 vs M0-v2 vs N2-v2 on the same frames, scored the same
way, cross-fitted the same way. Labels are never used (all targets below are label-free pixel / geometry facts); the
behaviour labels appear only in an identity check of the v2 codes (nose_nose R~ must equal the step-2 score files).

==================================================================================================================
PRE-REGISTRATION (written 2026-10-22, after the step-2 amendment commit e41d3be and BEFORE any score of this file or
of step 2 existed: no SAE of step 2 had finished training when this file was committed).

REPRESENTATIONS
    v1      the deployed v1 ECI representation = what the atlas calls v1 ('online'): matryoshka_btk_1024_k16_fg448al_s0
            (DINOv2-B at 448 px on the whole odour-rotated 512 px frame, 32 x 32 patches, its own foreground mask
            fg448 with the fg448al backgrounds, Matryoshka BatchTopK 1024 / k16 SAE, frame max). Its frame codes are
            re-computed here from the frames (identity check: max |recomputed - deployed codes_max.npy| per frame,
            reported) so that per-patch codes exist. One SAE (the deployed one): its score is a single number.
    M0-v2   step-2 M0 SAEs (frozen v2 tight mask), seeds 70-89 (mice SAE).
    N2-v2   step-2 N2 SAEs (new cue, >= 25% OR >= 16 appendage px), seeds 70-89.
    v2 codes are the unsmoothed frame max over the arm's tokens (as the step-2 alignment scores); v1 likewise.
FRAMES  mice v1 diagnosis set of the mask tests (43,200 frames, 144 annotated videos, every second round-2 diagnosis
    frame; union token store BEE/mice/diag). Cross-fitting units = pools (24), halves = the 10 random pool halves
    rnd0-rnd9 of the alignment scorer (scripts/eci/v2_mask/mask_test.py mice_sets). MABe22 = the step-1 held-out set
    (120 mouse_submission clips x 8 frames, x4 upsampled to 896 px; validate.py), units = clips, halves = 10 random
    clip halves (src/eci/v2/score.splits_of).
NEURON SETS  PRIMARY = the first 256 neurons of each dictionary (the user's discovery search, 2026-10-22). Secondary =
    the whole dictionary (2048 v2, 1024 v1), reported.

TARGETS (label-free)
  ODOUR BOX. The bag sits outside the arena at one corner; its corner is known per video from the spout / bag geometry
    (dataset/mice/v1/eci/odor_corner.csv, scripts/eci/mice_odor_corner.py; after the odour rotation it is the top
    right corner for every video). Zone = disk of radius 0.5 x near_radius_px (= 0.25 x arena side) around the
    rotated arena corner (mask2_test.odour_corner, the step-2 odour-zone definition), in 1024 px canonical frame
    coordinates; a token / patch is in the zone iff its centre is.
    BOX-F (frame target, PRIMARY for the box): Y_box = 1 iff >= 4 frozen-cue body tokens (frozen dark-cue pixel
      fraction >= 0.25, i.e. the M0 rule, so the target does not favour N2) have their centre in the zone.
      Score = cross-fitted R~ (src/eci/v2/score.crossfit: the neuron with the best R~ on one pool half, scored on the
      other, both directions, mean over rnd0-9).
    BOX-T (token, reported): for each neuron, over the frames where it fires, the share whose argmax token lies in the
      zone; neurons with >= 20 firing frames in the selection half; cross-fitted (best share on one half, its share on
      the other half). Chance reference = share of the representation's tokens that lie in the zone.
  TAIL. Tail pixels = appendage pixels of the mask2 cue (src/eci/v2/mask2.appendage: new-cue pixels outside the opened
    body, within 6 body half-widths of it = thin attached parts; in mice almost all tails), label-free.
    TAIL-T (token, PRIMARY lab tail score): appendage share of a neuron = sum_t a_t app_t / sum_t a_t anim_t over all
      tokens of the half's frames (a_t its code on token t, app_t / anim_t the token's appendage / new-cue animal pixel
      fraction; neurons with >= 20 firing frames in the selection half); cross-fitted as BOX-T. For v1 patches (32
      grid = 2 x 2 v2 tokens of the 1024 frame, both whole canonical frames) app / anim = the mean of the 4 v2
      tokens' fractions in the union store (tokens not stored count 0; N2 stores 98% of appendage pixels).
    TAIL-MAB (held-out benchmark, PRIMARY confirmatory tail score): MABe22 tail share = sum_t a_t tail_t / sum_t a_t
      reg_t, tail_t = token pixel fraction inside the tail capsules (radius 0.75 h along tail_base -> tail_mid ->
      tail_tip keypoints), reg_t = fraction inside the whole keypoint animal region (validate.region); neurons with
      >= 10 firing frames in the selection clip half; cross-fitted over clip halves. Tokens: M0 / N2 = the arm's own
      kept tokens; v1 = 32-grid patches (28 px of 896) holding any N2-kept pixel (v1's own fg448 mask needs per-video
      DINOv2 backgrounds that do not exist for MABe22; this generous token set can only help v1, so it is
      conservative for an 'N2 better than v1' claim). v1 frames: the 896 px frame resized to 512 (bicubic), then the
      v1 processor (448).
    TAIL-F (frame target, reported): Y_tail = top 20% of the video's frames by appendage pixel count (within video).
      Caveat stated now: confounded with how spread out the mice are.
    COLLECTIVE TAIL (reported): Y_tail2 = >= 2 separate 8-connected groups of tokens with >= 16 appendage pixels
      (>= 2 tails visible apart). Tail ALIGNMENT is NOT definable label-free from these data (no tail orientation
      without keypoints) and is not scored. 'Tail moving' is NOT definable on this frame set (frames 4 s apart).

RULE ("N2-v2 better than v1" on a score): Delta_s = score(N2, seed s) - score(v1) on identical halves, s = 70-89;
    PASS iff mean Delta >= margin AND the 95% t-CI lower bound of mean Delta over the 20 seeds > 0. Margins: R~ 0.02;
    shares 0.05. v1 is one deployed SAE and is treated as fixed (its split-to-split sd is reported).
    TAIL verdict = PASS iff TAIL-T AND TAIL-MAB pass (primary neuron set). BOX verdict = PASS iff BOX-F passes
    (primary neuron set). M0-v2 vs v1 (same rule) and N2 vs M0 (Welch 95% CI) are reported, never decide.
    'A tail neuron exists' (reported): held-out appendage share > 0.5 (the step-2 appendage-neuron threshold).
EXAMPLES: the deployment seed (highest held-out FVE among 70-72) of M0 / N2 and v1: the modal cross-fitted neuron of
    TAIL-T and of BOX-F (primary set); contact sheets of its 24 top frames of pool half B of rnd0 (<= 1 per video),
    real per-token activation maps -> results/vision/eci_v2_mask2/discovery/sheets/.
==================================================================================================================

Steps (job.sh, SCRIPT=discovery.py):
    lab --arm M0|N2 [--seeds]   (GPU) mice diag scores per seed -> OUT/discovery/lab/<arm>_s<seed>.json (+ sheets)
    v1lab                        (GPU) v1 re-encode of the diag frames -> OUT/discovery/lab/v1.json (+ sheets)
    mab --arm M0|N2|v1 [--seeds] (GPU) MABe22 tail shares -> OUT/discovery/mab/<arm>[_s<seed>].json
    decide                       (CPU) -> OUT/discovery/decision.json, table.md
"""
import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mask2_test as T  # noqa: E402  (stores, SAEs, odour zone, deployment seed)

REPO, OUT, BEE, log, FZ, MT, N = T.REPO, T.OUT / 'discovery', T.BEE, T.log, T.FZ, T.MT, T.N
from src.eci.v2.score import crossfit, splits_of, welch  # noqa: E402
from src.eci.v2 import mask2 as M2  # noqa: E402

V1 = 'matryoshka_btk_1024_k16_fg448al_s0'
V1_DIR = REPO / 'dataset/mice/v1/eci'
SEEDS = T.SEEDS['mice']
PRIMARY = 256
MIN_FIRE, MIN_FIRE_MAB = 20, 10
BOX_MIN_TOK = 4
MARGIN = {'R': 0.02, 'share': 0.05}
FLAB = ['box', 'tail_vis', 'tail2']
RND = tuple(f'rnd{i}' for i in range(10))


# ---------------------------------------------------------------------------------------------- targets
def lab_targets(U):
    """Union('mice', 'diag') -> per-token zone flag (U row space), per-frame Y (nf, 3) bool, extra facts."""
    from scipy import ndimage
    G = U.G
    od = T.odour_corner(G * 16)
    oy, ox, orad = (np.array([od[o][i] for o in U.F.obs.values]) for i in range(3))
    pr, pc, fr = U.pos // G, U.pos % G, U.tf
    tokc = ((pr * 16 + 8 - oy[fr]) ** 2 + (pc * 16 + 8 - ox[fr]) ** 2) <= orad[fr] ** 2
    body = U.raw >= 0.25
    nbox = np.bincount(fr[body & tokc], minlength=len(U.F))
    y_box = nbox >= BOX_MIN_TOK
    app = U.F.app_px.values
    rk = U.F.assign(a=app).groupby('obs').a.rank(pct=True, method='average').values
    y_tail = rk > 0.8
    thin = U.ng >= M2.MIN_THIN_PX
    ncomp = np.zeros(len(U.F), int)
    st = np.r_[0, np.cumsum(np.bincount(fr, minlength=len(U.F)))]
    for f in range(len(U.F)):
        a, b = st[f], st[f + 1]
        m = thin[a:b]
        if m.sum() < 2:
            ncomp[f] = int(m.sum())
            continue
        g = np.zeros(G * G, bool)
        g[U.pos[a:b][m]] = True
        ncomp[f] = ndimage.label(g.reshape(G, G), structure=np.ones((3, 3)))[1]
    Y = np.stack([y_box, y_tail, ncomp >= 2], 1)
    facts = {'n_frames': int(len(U.F)), 'base_rate': {k: float(Y[:, i].mean()) for i, k in enumerate(FLAB)},
             'box_zone_radius_px_1024_median': float(np.median(orad)),
             'body_tokens_in_zone_per_frame_mean': float(nbox.mean())}
    return tokc, Y, facts, od


def unit_halves(St):
    Y, sets, half, uidx, SPL = MT.mice_sets(St)
    return Y, half['all'], uidx, SPL


# ---------------------------------------------------------------------------------------------- cross-fitted shares
def cf_share(num, den, fire, SPL, P, min_fire):
    """Per-unit accumulators (n_units, W) -> cross-fitted best ratio num / den (eligible: fire >= min_fire on the
    selection half); {'share', 'share_sd_splits', 'modal_neuron', 'modal_share', 'n_eligible_mean'}."""
    num, den, fire = num[:, :P], den[:, :P], fire[:, :P]
    vals, sel, nel = [], [], []
    for s in RND:
        h = np.asarray(SPL[s])
        H = {k: (num[h == k].sum(0), den[h == k].sum(0), fire[h == k].sum(0)) for k in (0, 1)}
        d = []
        for a_, b_ in ((0, 1), (1, 0)):
            na, da, fa = H[a_]
            r = np.where((fa >= min_fire) & (da > 0), na / np.maximum(da, 1e-12), -np.inf)
            j = int(np.argmax(r))
            nb, db, _ = H[b_]
            d.append(float(nb[j] / db[j]) if db[j] > 0 else 0.0)
            sel.append(j)
            nel.append(int(np.isfinite(r).sum()))
        vals.append(float(np.mean(d)))
    v, c = np.unique(sel, return_counts=True)
    return {'share': float(np.mean(vals)), 'share_sd_splits': float(np.std(vals)), 'modal_neuron': int(v[np.argmax(c)]),
            'modal_share': float(c.max() / len(sel)), 'n_eligible_mean': float(np.mean(nel))}


def frame_scores(X, Yg, half, P):
    import torch
    o = crossfit(X[:, :P].contiguous(), Yg, half, FLAB)
    return {lb: {k: o['mean'][lb][k] for k in ('R', 'R_sd_splits', 'A_at_R', 'rate_at_R', 'modal_neuron',
                                                 'modal_share')} for lb in FLAB}


def all_scores(X, Yg, half, acc, SPL, W):
    out = {}
    for P, nm in ((PRIMARY, 'first256'), (W, 'all')):
        out[nm] = {'frame': frame_scores(X, Yg, half, P),
                   'tail_token': cf_share(acc['num'], acc['den'], acc['fire'], SPL, P, MIN_FIRE),
                   'box_token': cf_share(acc['zone'], acc['fire'], acc['fire'], SPL, P, MIN_FIRE)}
    return out


# ---------------------------------------------------------------------------------------------- v2 arms (lab)
def cmd_lab(args):
    import torch
    dev = torch.device('cuda')
    a = args.arm
    od_ = OUT / 'lab'
    od_.mkdir(parents=True, exist_ok=True)
    seeds = [s for s in FZ.parse_seeds(args.seeds, SEEDS) if not (od_ / f'{a}_s{s}.json').exists()
             and (T.sae_dir('mice', a, s) / 'sae_theta.pt').exists()]
    if not seeds:
        return
    U = T.Union('mice', 'diag')
    tokc_all, Y, facts, od = lab_targets(U)
    (od_ / 'targets.json').write_text(json.dumps(facts, indent=1))
    m = U.arm(a)
    St = T.arm_store(U, a, dev, on_gpu=True)
    tokc = tokc_all[m]
    Ylab, half, uidx, SPL = unit_halves(St)
    nU = int(uidx.max()) + 1
    Yg = torch.from_numpy(Y).to(dev)
    sc0 = OUT.parent / 'mice' / 'score'
    fv = {s: T.fve_of('mice', a, s) for s in T.DEPLOY_FROM}
    dep = max(fv, key=lambda s: fv[s] if np.isfinite(fv[s]) else -1)
    zone_tok_share = float(tokc.mean())
    uf = torch.from_numpy(uidx).to(dev)
    ut = torch.from_numpy(uidx[St.tf]).to(dev)
    app_t, anim_t = torch.from_numpy(St.app).to(dev), torch.from_numpy(St.fd).to(dev)
    zc = torch.from_numpy(tokc).to(dev)
    for s in seeds:
        t0 = time.time()
        sae, norm = T.load_sae('mice', a, s, dev)
        W = sae.n_latents
        X = torch.zeros((St.nf, W), dtype=torch.float16, device=dev)
        acc = {k: torch.zeros((nU, W), dtype=torch.float64, device=dev) for k in ('num', 'den', 'fire', 'zone')}
        fp = max(4, min(48, int(2**27 // W // 160)))
        with torch.no_grad():
            for f0, f1, a0, b0, z in St.chunks(fp):
                fl = torch.from_numpy(St.tf[a0:b0] - f0).to(dev)
                c = sae.encode(norm(z.float()), mode='threshold').float()
                fm = torch.zeros((f1 - f0, W), device=dev).index_reduce_(0, fl, c, 'amax', include_self=True)
                X[f0:f1] = fm.half()
                hit = (c > 0) & (c == fm[fl]) & zc[a0:b0, None]
                zf = torch.zeros((f1 - f0, W), device=dev).index_reduce_(0, fl, hit.float(), 'amax', include_self=True)
                acc['fire'].index_add_(0, uf[f0:f1], (fm > 0).double())
                acc['zone'].index_add_(0, uf[f0:f1], zf.double())
                acc['num'].index_add_(0, ut[a0:b0], (c * app_t[a0:b0, None]).double())
                acc['den'].index_add_(0, ut[a0:b0], (c * anim_t[a0:b0, None]).double())
        accn = {k: v.cpu().numpy() for k, v in acc.items()}
        res = {'arm': a, 'seed': s, 'deploy_seed': dep, 'eval_fve': fv.get(s, T.fve_of('mice', a, s)),
               'zone_token_share': zone_tok_share, **all_scores(X, Yg, half, accn, SPL, W)}
        # identity check vs the step-2 alignment score (same codes, same scorer)
        f0_ = sc0 / f'{a}_s{s}.json'
        Yl = torch.from_numpy(Ylab).to(dev)
        o = crossfit(X, Yl, half, FZ.MLAB)
        res['check_nose_nose_R'] = o['mean']['nose_nose']['R']
        if f0_.exists():
            ref = json.loads(f0_.read_text())['all']['mean']['nose_nose']['R']
            res['check_vs_step2_score'] = float(abs(ref - res['check_nose_nose_R']))
        log(f'lab {a}_s{s}: box R~ {res["first256"]["frame"]["box"]["R"]:.4f} | tail-T '
            f'{res["first256"]["tail_token"]["share"]:.4f} | box-T {res["first256"]["box_token"]["share"]:.4f} | '
            f'tail_vis R~ {res["first256"]["frame"]["tail_vis"]["R"]:.4f} | nn check {res["check_nose_nose_R"]:.4f} '
            f'({time.time() - t0:.0f}s; {MT.gpu_line()})')
        if s == dep:
            res['sheets'] = v2_sheets(a, s, sae, norm, St, X, Y, half, res, tokc_all, U, od)
        (od_ / f'{a}_s{s}.json').write_text(json.dumps(res, indent=1, default=float))
        del X
        torch.cuda.empty_cache()


def heldout_frames(xj, half, obs, n=24):
    hold = np.asarray(half['rnd0']) == 1
    c = pd.DataFrame({'f': np.arange(len(xj)), 'v': xj, 'obs': obs})[hold]
    c = c[c.v > 0].sort_values('v', ascending=False).drop_duplicates('obs').head(n)
    return c.f.values.astype(int), c.v.values


def decoder():
    import torch
    from src.eci.v2.pipeline import Series
    se = Series(T.series_cfg('mice', 'v1'), torch.device('cuda'), load_encoder=False)

    def dec(o, k):
        v = se.V[o]
        x, _ = se.frames(v, se.run_job(**se.job(v, int(k), 1)), [int(k)])
        return x[0].cpu().numpy()
    return dec, se


def draw_zone(x, zone):
    """x (3, S, S) uint8 -> copy with the odour-zone circle outline (cyan)."""
    y0, x0, r = zone
    S = x.shape[-1]
    yy, xx = np.mgrid[0:S, 0:S]
    d = np.sqrt((yy - y0) ** 2 + (xx - x0) ** 2)
    ring = np.abs(d - r) < 2.5
    x = x.copy()
    x[:, ring] = np.array([0, 255, 255], np.uint8)[:, None]
    return x


def v2_sheets(a, s, sae, norm, St, X, Y, half, res, tokc_all, U, od):
    import torch
    dec, _ = decoder()
    out = {}
    sd = OUT / 'sheets'
    sd.mkdir(parents=True, exist_ok=True)
    for tgt, j in (('tail', res['first256']['tail_token']['modal_neuron']),
                   ('box', res['first256']['frame']['box']['modal_neuron'])):
        fr, vals = heldout_frames(X[:, j].float().cpu().numpy(), half, St.F.obs.values)
        tiles, apf, inz = [], [], []
        for f, v in zip(fr, vals):
            a0, b0 = int(St.starts[f]), int(St.starts[f + 1])
            zt = torch.from_numpy(np.ascontiguousarray(St.tok[a0:b0])).to(St.dev)
            with torch.no_grad():
                act = sae.encode(norm(zt.float()), mode='threshold')[:, j].float().cpu().numpy()
            h = np.zeros(St.G * St.G, np.float32)
            h[St.pos[a0:b0]] = act
            q = a0 + int(np.argmax(act))
            apf.append(float(St.app[q]))
            o, k = St.F.obs.values[f], int(St.F.k.values[f])
            pr, pc = St.pos[q] // St.G, St.pos[q] % St.G
            inz.append(bool((pr * 16 + 8 - od[o][0]) ** 2 + (pc * 16 + 8 - od[o][1]) ** 2 <= od[o][2] ** 2))
            lab = ''.join(c for c, y in zip(('B', 'T', 'TT'), Y[f]) if y) or '-'
            x = draw_zone(dec(o, k), od[o]) if tgt == 'box' else dec(o, k)
            tiles.append((x, h.reshape(St.G, St.G), {'obs': o, 'k': k, 'label': lab, 'stored': float(v),
                                                     'n_half_max': int((h >= 0.5 * h.max()).sum())}))
        fn = sd / f'{a}_s{s}_{tgt}_n{j}.jpg'
        N.contact_sheet(tiles, fn, f'discovery {tgt}: {a}-v2 seed {s} neuron n{j} (modal cross-fitted, first 256); '
                                   f'top half-B frames rnd0, <= 1 per video; labels B=box zone occupied, T=tail top 20%, '
                                   f'TT=2+ tails; cyan circle = odour zone', St.G)
        out[tgt] = {'neuron': j, 'sheet': str(fn.relative_to(REPO)), 'n_frames': len(fr),
                    'peak_token_app_frac_mean': float(np.mean(apf)) if apf else None,
                    'peak_token_in_zone_share': float(np.mean(inz)) if inz else None,
                    'frames_box_target_share': float(np.mean([Y[f, 0] for f in fr])) if len(fr) else None}
        log(f'sheet {a}_s{s} {tgt} n{j}: {out[tgt]}')
    return out


# ---------------------------------------------------------------------------------------------- v1 (lab)
def v1_tokens_fracs(U, frames_idx, dev):
    """(B,) frame indices -> (B, 1024) app / anim fractions of the v1 32-grid patches (mean of 2 x 2 v2 tokens)."""
    import torch
    G = U.G
    st = U.starts
    B = len(frames_idx)
    app = torch.zeros((B, G * G), device=dev)
    anim = torch.zeros((B, G * G), device=dev)
    for i, f in enumerate(frames_idx):
        a, b = int(st[f]), int(st[f + 1])
        p = torch.from_numpy(U.pos[a:b]).to(dev)
        app[i, p] = torch.from_numpy(U.app[a:b]).to(dev)
        anim[i, p] = torch.from_numpy(U.new[a:b]).to(dev)
    pool = lambda t: t.view(B, G // 2, 2, G // 2, 2).mean((2, 4)).flatten(1)  # noqa: E731
    return pool(app), pool(anim)


def cmd_v1lab(args):
    import torch
    from src.eci.foreground import FrameDatasetFG, encode_batch
    from src.eci.viz import make_patch_encoder
    dev = torch.device('cuda')
    od_ = OUT / 'lab'
    od_.mkdir(parents=True, exist_ok=True)
    U = T.Union('mice', 'diag')
    U.starts = np.r_[0, np.cumsum(np.bincount(U.tf, minlength=len(U.F)))]
    tokc_all, Y, facts, od = lab_targets(U)
    from types import SimpleNamespace
    Ylab, half, uidx, SPL = unit_halves(SimpleNamespace(F=U.F, nf=len(U.F)))  # halves / labels only
    nU = int(uidx.max()) + 1
    enc = make_patch_encoder(V1, dataset_dir=str(REPO / 'dataset'))
    log(f'v1 SAE hparams: {json.dumps({k: str(v) for k, v in torch.load(V1_DIR / "sae" / V1 / "sae.pt", map_location="cpu", weights_only=False)["hparams"].items()})}')
    rows = U.F.row.values.astype(np.int64)
    _, se = decoder()
    rp = np.array([se.rot.get(o, 0) for o in U.F.obs.values])
    assert (enc.rot[rows] % 4 == rp % 4).all(), 'v1 / v2 rotations differ'
    fp = pd.read_csv(REPO / N.C['ann'], usecols=['frame_path'])['frame_path'].values[rows]
    loc = N.local_dir() / 'v1frames'
    loc.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    dst = [loc / f'{i:06d}.jpg' for i in range(len(fp))]
    with ThreadPoolExecutor(16) as ex:
        list(ex.map(lambda t: shutil.copyfile(REPO / 'dataset' / t[0], t[1]), zip(fp, dst)))
    log(f'v1: staged {len(dst)} frames in {time.time() - t0:.0f}s')
    # deployed frame codes for the identity check (rows sorted: one pass)
    cm = np.load(V1_DIR / 'codes' / V1 / 'codes_max.npy', mmap_mode='r')
    oi = np.argsort(rows)
    ref = np.empty((len(rows), cm.shape[1]), np.float16)
    ref[oi] = cm[rows[oi]]
    W = enc.sae.n_latents
    X = torch.zeros((len(rows), W), dtype=torch.float16, device=dev)
    acc = {k: torch.zeros((nU, W), dtype=torch.float64, device=dev) for k in ('num', 'den', 'fire', 'zone')}
    G1 = 32
    od1 = np.array([od[o] for o in U.F.obs.values])  # (nf, 3) 1024 px
    cy = (np.arange(G1 * G1) // G1) * 32 + 16
    cx = (np.arange(G1 * G1) % G1) * 32 + 16
    jac0, jac2, ntok = [], [], []
    m0 = U.arm('M0')
    n2 = U.arm('N2')
    dl = torch.utils.data.DataLoader(FrameDatasetFG([str(p) for p in dst], enc.processor, rows, rot=enc.rot[rows]),
                                     batch_size=64, num_workers=12, shuffle=False)
    uf_all = torch.from_numpy(uidx).to(dev)
    o = 0
    t0 = time.time()
    with torch.no_grad():
        for pix, grey, r in dl:
            B = len(r)
            fidx = np.arange(o, o + B)
            tok = encode_batch(enc.model, pix, dev)
            mask, _ = enc.bgs.mask(tok, grey.to(dev), r.numpy())
            fi, pi = torch.nonzero(mask, as_tuple=True)
            c = enc.sae.encode(enc.norm(tok[fi, pi]), mode='threshold').float()
            fm = torch.zeros((B, W), device=dev).index_reduce_(0, fi, c, 'amax', include_self=True)
            X[o:o + B] = fm.half()
            app, anim = v1_tokens_fracs(U, fidx, dev)
            zon = torch.from_numpy(((cy[None] - od1[fidx, 0:1]) ** 2 + (cx[None] - od1[fidx, 1:2]) ** 2
                                    <= od1[fidx, 2:3] ** 2)).to(dev)
            hit = (c > 0) & (c == fm[fi]) & zon[fi, pi][:, None]
            zf = torch.zeros((B, W), device=dev).index_reduce_(0, fi, hit.float(), 'amax', include_self=True)
            uf = uf_all[o:o + B]
            acc['fire'].index_add_(0, uf, (fm > 0).double())
            acc['zone'].index_add_(0, uf, zf.double())
            acc['num'].index_add_(0, uf[fi], (c * app[fi, pi][:, None]).double())
            acc['den'].index_add_(0, uf[fi], (c * anim[fi, pi][:, None]).double())
            # mask agreement with the v2 masks mapped to the 32 grid (geometry check)
            mk = mask.cpu().numpy()
            for i, f in enumerate(fidx):
                a, b = int(U.starts[f]), int(U.starts[f + 1])
                for arr, lst in ((m0, jac0), (n2, jac2)):
                    g = np.zeros(64 * 64, bool)
                    g[U.pos[a:b][arr[a:b]]] = True
                    g = g.reshape(32, 2, 32, 2).any((1, 3)).ravel()
                    lst.append((mk[i] & g).sum() / max(1, (mk[i] | g).sum()))
                ntok.append(int(mk[i].sum()))
            o += B
            if (o // B) % 100 == 0:
                el = time.time() - t0
                log(f'v1: {o}/{len(rows)} frames, {o / el:.0f} fps; {MT.gpu_line()}')
    Xn = X.float().cpu().numpy()
    diff = np.abs(Xn - ref.astype(np.float32)).max(1)
    accn = {k: v.cpu().numpy() for k, v in acc.items()}
    Yg = torch.from_numpy(Y).to(dev)
    zone_share = float(np.mean([((cy - od1[f, 0]) ** 2 + (cx - od1[f, 1]) ** 2 <= od1[f, 2] ** 2).mean()
                                for f in range(0, len(rows), 97)]))
    res = {'arm': 'v1', 'sae': V1, 'n_frames': len(rows), 'tokens_per_frame': float(np.mean(ntok)),
           'check_vs_deployed_codes_max': {'max_abs_diff': float(diff.max()), 'mean_abs_diff_per_frame_max':
                                           float(diff.mean()), 'share_frames_diff_gt_0.01': float((diff > 0.01).mean())},
           'mask_jaccard_vs_M0_32grid': float(np.mean(jac0)), 'mask_jaccard_vs_N2_32grid': float(np.mean(jac2)),
           'zone_patch_share_of_frame_approx': zone_share,
           **all_scores(X, Yg, half, accn, SPL, W)}
    Yl = torch.from_numpy(Ylab).to(dev)
    o_ = crossfit(X, Yl, half, FZ.MLAB)
    res['check_nose_nose_R'] = o_['mean']['nose_nose']['R']
    res['check_nose_tail_R'] = o_['mean']['nose_tail']['R']
    log('v1 lab: ' + json.dumps({k: res[k] for k in res if k not in ('first256', 'all')}, default=float))
    log('v1 lab first256: ' + json.dumps(res['first256'], default=float))
    res['sheets'] = v1_sheets(enc, U, X, Y, half, res, od, dst, rows)
    (od_ / 'v1.json').write_text(json.dumps(res, indent=1, default=float))


def v1_sheets(enc, U, X, Y, half, res, od, dst, rows):
    dec, _ = decoder()
    out = {}
    sd = OUT / 'sheets'
    sd.mkdir(parents=True, exist_ok=True)
    for tgt, j in (('tail', res['first256']['tail_token']['modal_neuron']),
                   ('box', res['first256']['frame']['box']['modal_neuron'])):
        fr, vals = heldout_frames(X[:, j].float().cpu().numpy(), half, U.F.obs.values)
        if not len(fr):
            continue
        pc, mk = enc.patch_codes([str(dst[f]) for f in fr], [j], rows[fr], return_mask=True)
        tiles = []
        for i, (f, v) in enumerate(zip(fr, vals)):
            o, k = U.F.obs.values[f], int(U.F.k.values[f])
            h = pc[i, :, :, 0]
            lab = ''.join(c for c, y in zip(('B', 'T', 'TT'), Y[f]) if y) or '-'
            x = draw_zone(dec(o, k), od[o]) if tgt == 'box' else dec(o, k)
            tiles.append((x, h, {'obs': o, 'k': k, 'label': lab, 'stored': float(v),
                                 'n_half_max': int((h >= 0.5 * h.max()).sum()) if h.max() > 0 else 0}))
        fn = sd / f'v1_{tgt}_n{j}.jpg'
        N.contact_sheet(tiles, fn, f'discovery {tgt}: v1 ({V1}) neuron n{j} (modal cross-fitted, first 256); top '
                                   f'half-B frames rnd0, <= 1 per video; B=box zone occupied, T=tail top 20%, TT=2+ '
                                   f'tails; cyan circle = odour zone', 32)
        out[tgt] = {'neuron': j, 'sheet': str(fn.relative_to(REPO)), 'n_frames': len(fr),
                    'frames_box_target_share': float(np.mean([Y[f, 0] for f in fr]))}
        log(f'sheet v1 {tgt} n{j}: {out[tgt]}')
    return out


# ---------------------------------------------------------------------------------------------- MABe22
def cmd_mab(args):
    import torch
    import torch.nn.functional as F
    from PIL import Image
    import validate as VA
    from common import N_BG as NB_, mab_decode
    mab_clips = VA.mab_clips
    dev = torch.device('cuda')
    od_ = OUT / 'mab'
    od_.mkdir(parents=True, exist_ok=True)
    arm = args.arm
    seeds = [None] if arm == 'v1' else [s for s in FZ.parse_seeds(args.seeds, SEEDS)
                                        if not (od_ / f'{arm}_s{s}.json').exists()]
    if arm == 'v1' and (od_ / 'v1.json').exists() or not seeds:
        return
    items = mab_clips(120, 8, 0)
    vids = []
    for c, ks, kps in items:
        a_ = mab_decode('mouse_submission', c)
        ib = np.linspace(0, len(a_) - 1, NB_).round().astype(int)
        vids.append((c, VA.up4(a_[ib], dev), VA.up4(a_[ks + 1], dev), ks, kps))
    packs = [M2.background_pack(xb) for _, xb, _, _, _ in vids]
    h = VA.body_h([cc.cpu().numpy() for (_, xb, _, _, _), pk in zip(vids, packs)
                   for cc in VA.dark_cue(xb[::6], pk['q98'])])
    S, G = 896, 56
    G1, P1 = 32, 28
    log(f'mab: {len(vids)} clips, h {h:.3f}')
    if arm != 'v1':
        from src.eci.v2.encode import Encoder
        enc = Encoder(dev)
    else:
        from src.eci.foreground import encode_batch, load_encoder_fg
        from src.eci.sae import load_sae as lsae
        _, proc, model = load_encoder_fg(device=dev)
        sae1, norm1, _ = lsae(V1_DIR / 'sae' / V1 / 'sae.pt', dev)
    toks, tailf, regf, unit, nfr, ntok = [], [], [], [], [], []
    for ci, ((c, xb, x, ks, kps), pk) in enumerate(zip(vids, packs)):
        for b0 in range(0, len(x), 8):
            xb_ = x[b0:b0 + 8]
            ms, _ = VA.masks_all(xb_, pk, h)
            if arm == 'v1':
                pil = [Image.fromarray(xi.permute(1, 2, 0).cpu().numpy()).resize((512, 512), Image.BICUBIC)
                       for xi in xb_]
                z = encode_batch(model, proc(images=pil, return_tensors='pt')['pixel_values'], dev)
                kp = F.avg_pool2d(F.interpolate(ms['N2'][0].view(-1, 1, G, G).float(), scale_factor=16,
                                                mode='nearest'), P1).flatten(1) > 0
            else:
                z = enc(xb_)
                kp = ms[arm][0]
            for i in range(len(xb_)):
                kk = kps[b0 + i]
                tail = VA.region(S, kk, [], VA.MAB_TAIL, h, False)
                reg = VA.region(S, kk, VA.MAB_SKEL, VA.MAB_TAIL, h, False)
                ps = P1 if arm == 'v1' else 16
                tf_ = F.avg_pool2d(torch.from_numpy(tail).float()[None, None], ps).flatten()
                rf_ = F.avg_pool2d(torch.from_numpy(reg).float()[None, None], ps).flatten()
                sel = torch.nonzero(kp[i]).flatten()
                toks.append(z[i, sel].cpu())
                tailf.append(tf_[sel.cpu()])
                regf.append(rf_[sel.cpu()])
                unit.append(np.full(len(sel), ci))
                nfr.append(np.full(len(sel), len(ntok)))
                ntok.append(len(sel))
    Z = torch.cat(toks).to(dev)
    tl, rg = torch.cat(tailf).to(dev).double(), torch.cat(regf).to(dev).double()
    ut = torch.from_numpy(np.concatenate(unit)).to(dev)
    ft = torch.from_numpy(np.concatenate(nfr)).to(dev)
    nf = len(ntok)
    fu = torch.from_numpy(np.repeat(np.arange(len(vids)), [len(v[3]) for v in vids])).to(dev)
    units, SPL = splits_of(range(len(vids)))
    base = {'n_clips': len(vids), 'n_frames': nf, 'h': h, 'tokens_per_frame': float(np.mean(ntok)),
            'tail_px_share_of_region_px_in_tokens': float((tl.sum() / rg.sum()).item())}
    log(f'mab {arm}: {json.dumps(base)}')
    for s in seeds:
        if arm == 'v1':
            sae, norm = sae1, norm1
        else:
            sae, norm = T.load_sae('mice', arm, s, dev)
        W = sae.n_latents
        acc = {k: torch.zeros((len(vids), W), dtype=torch.float64, device=dev) for k in ('num', 'den', 'fire')}
        with torch.no_grad():
            for a0 in range(0, len(Z), 8192):
                c = sae.encode(norm(Z[a0:a0 + 8192].float()), mode='threshold').double()
                acc['num'].index_add_(0, ut[a0:a0 + 8192], c * tl[a0:a0 + 8192, None])
                acc['den'].index_add_(0, ut[a0:a0 + 8192], c * rg[a0:a0 + 8192, None])
            fm = torch.zeros((nf, W), device=dev)
            for a0 in range(0, len(Z), 8192):
                c = sae.encode(norm(Z[a0:a0 + 8192].float()), mode='threshold').float()
                fm.index_reduce_(0, ft[a0:a0 + 8192], c, 'amax', include_self=True)
            acc['fire'].index_add_(0, fu, (fm > 0).double())
        accn = {k: v.cpu().numpy() for k, v in acc.items()}
        res = {'arm': arm, 'seed': s, **base}
        for P, nm in ((PRIMARY, 'first256'), (W, 'all')):
            res[nm] = {'tail_mab': cf_share(accn['num'], accn['den'], accn['fire'], SPL, P, MIN_FIRE_MAB)}
        log(f'mab {arm}_s{s}: tail share first256 {res["first256"]["tail_mab"]} | all {res["all"]["tail_mab"]}')
        (od_ / (f'{arm}_s{s}.json' if s is not None else 'v1.json')).write_text(json.dumps(res, indent=1, default=float))


# ---------------------------------------------------------------------------------------------- decide
def tci(d):
    from scipy import stats
    d = np.asarray(d, float)
    m, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
    hw = stats.t.ppf(0.975, len(d) - 1) * se
    return float(m), float(m - hw), float(m + hw)


def cmd_decide(args):
    L = {a: {s: json.loads((OUT / 'lab' / f'{a}_s{s}.json').read_text()) for s in SEEDS
             if (OUT / 'lab' / f'{a}_s{s}.json').exists()} for a in ('M0', 'N2')}
    Mb = {a: {s: json.loads((OUT / 'mab' / f'{a}_s{s}.json').read_text()) for s in SEEDS
              if (OUT / 'mab' / f'{a}_s{s}.json').exists()} for a in ('M0', 'N2')}
    v1 = json.loads((OUT / 'lab' / 'v1.json').read_text())
    v1m = json.loads((OUT / 'mab' / 'v1.json').read_text())
    get = {'BOX-F R~': (lambda r, P: r[P]['frame']['box']['R'], 'R', 'lab'),
           'TAIL-T share': (lambda r, P: r[P]['tail_token']['share'], 'share', 'lab'),
           'TAIL-MAB share': (lambda r, P: r[P]['tail_mab']['share'], 'share', 'mab'),
           'BOX-T share': (lambda r, P: r[P]['box_token']['share'], 'share', 'lab'),
           'TAIL-F R~': (lambda r, P: r[P]['frame']['tail_vis']['R'], 'R', 'lab'),
           'TAIL2-F R~ (2+ tails)': (lambda r, P: r[P]['frame']['tail2']['R'], 'R', 'lab')}
    tab = {}
    for P in ('first256', 'all'):
        for nm, (fn, kind, src) in get.items():
            D = L if src == 'lab' else Mb
            ref = v1 if src == 'lab' else v1m
            v = fn(ref, P)
            row = {'v1': v}
            for a in ('M0', 'N2'):
                xs = [fn(r, P) for r in D[a].values()]
                row[a] = FZ.mean_sd(xs)
                if len(xs) >= 2:
                    row[f'{a}_minus_v1'] = tci(np.array(xs) - v)
                    m, lo, _ = row[f'{a}_minus_v1']
                    row[f'{a}_better_than_v1'] = bool(m >= MARGIN[kind] and lo > 0)
            if all(len(D[a]) >= 2 for a in ('M0', 'N2')):
                row['N2_minus_M0'] = welch([fn(r, P) for r in D['N2'].values()], [fn(r, P) for r in D['M0'].values()])
            tab[f'{P} | {nm}'] = row
    p = 'first256'
    verdict = {'TAIL (N2 better than v1)': bool(tab[f'{p} | TAIL-T share'].get('N2_better_than_v1') and
                                                tab[f'{p} | TAIL-MAB share'].get('N2_better_than_v1')),
               'BOX (N2 better than v1)': bool(tab[f'{p} | BOX-F R~'].get('N2_better_than_v1')),
               'reported: TAIL (M0 better than v1)': bool(tab[f'{p} | TAIL-T share'].get('M0_better_than_v1') and
                                                          tab[f'{p} | TAIL-MAB share'].get('M0_better_than_v1')),
               'reported: BOX (M0 better than v1)': bool(tab[f'{p} | BOX-F R~'].get('M0_better_than_v1'))}
    exists = {a: FZ.mean_sd([float(r['first256']['tail_token']['share'] > 0.5) for r in L[a].values()])
              for a in ('M0', 'N2')} | {'v1': float(v1['first256']['tail_token']['share'] > 0.5)}
    dep = {a: next((r for r in L[a].values() if r['seed'] == r['deploy_seed']), {}).get('sheets') for a in L}
    dec = {'rule': 'scripts/eci/v2_mask2/discovery.py docstring (pre-registered 2026-10-22)', 'table': tab,
           'verdict': verdict, 'tail_neuron_exists_share_of_seeds (held-out appendage share > 0.5, first 256)': exists,
           'targets': json.loads((OUT / 'lab' / 'targets.json').read_text()),
           'v1_checks': {k: v1[k] for k in v1 if k.startswith('check') or k.startswith('mask') or k == 'tokens_per_frame'},
           'v2_identity_checks': {a: [r.get('check_vs_step2_score') for r in L[a].values()] for a in L},
           'examples': {'v1': v1.get('sheets'), **dep}, 'n_seeds': {a: len(L[a]) for a in L},
           'n_seeds_mab': {a: len(Mb[a]) for a in Mb}}
    (OUT / 'decision.json').write_text(json.dumps(dec, indent=1, default=float))
    f3 = lambda x: f'{x[0]:.4f} ± {x[1]:.4f} (n={x[2]})' if isinstance(x, (list, tuple)) and len(x) == 3 and \
        isinstance(x[2], int) else (f'{x[0]:+.4f} [{x[1]:+.4f}, {x[2]:+.4f}]' if isinstance(x, (list, tuple))
                                    else (f'{x:.4f}' if isinstance(x, float) else str(x)))  # noqa: E731
    Ls = ['| neurons | score | v1 | M0-v2 | N2-v2 | M0 - v1 (95% t-CI) | N2 - v1 (95% t-CI) | N2 better? | '
          'N2 - M0 (Welch) |', '|---|' + '---|' * 8]
    for k, r in tab.items():
        P, nm = k.split(' | ')
        Ls.append(f'| {P} | {nm} | {f3(r["v1"])} | {f3(r.get("M0", "-"))} | {f3(r.get("N2", "-"))} | '
                  f'{f3(r.get("M0_minus_v1", "-"))} | {f3(r.get("N2_minus_v1", "-"))} | '
                  f'{r.get("N2_better_than_v1", "-")} | {f3(r.get("N2_minus_M0", "-"))} |')
    Ls += ['', 'VERDICT: ' + json.dumps(verdict), '', '```', json.dumps({k: dec[k] for k in
           ('tail_neuron_exists_share_of_seeds (held-out appendage share > 0.5, first 256)', 'targets', 'v1_checks',
            'v2_identity_checks', 'examples', 'n_seeds', 'n_seeds_mab')}, indent=1, default=float), '```']
    (OUT / 'table.md').write_text('\n'.join(Ls))
    print('\n'.join(Ls))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=('lab', 'v1lab', 'mab', 'decide'))
    p.add_argument('--arm', default='N2')
    p.add_argument('--seeds', default='')
    args = p.parse_args()
    {'lab': cmd_lab, 'v1lab': cmd_v1lab, 'mab': cmd_mab, 'decide': cmd_decide}[args.step](args)


if __name__ == '__main__':
    main()
