"""
BEHAVIOURAL NEURON MAP v2: the post-hoc map of the ECI SAE neurons (social level self / pair / collective, place-bound
= environment interaction, halo / spill, plain background), rebuilt on INSTANCE masks instead of merged dark blobs, and
validated (visual sheets, annotation anchors, collective test, old-vs-new confusion). Instance masks, tracking and arena
geometry are ANALYSIS TOOLS ONLY; nothing here changes the representation.

Dictionaries (sae.pt, input exactly as deployed)
    fg448m5   mice  C1 DEPLOYMENT SAE (rollout_c1): dataset/mice/v1/eci/sae/matryoshka_btk_4096_k16_fg448m5_s0, input
                    M5 = [token_t, token_t - token_{t-5}] (base fg448 store + d5 prev, combined.MotionTokens)
    antsfgm5  ants  C1 DEPLOYMENT SAE: dataset/ants/eci/sae/matryoshka_btk_4096_k16_antsfgm5_s0, same input path
    fg448al   mice  old deployed 1024 (odour-corner-aligned store, static tokens; positions rotated back to the raw frame)
    antsfg    ants  old deployed 1024 (static tokens of the ants base store)

Frames (fixed before any analysis)
    ants  all 76,800 C1 evaluation frames (the levers eval split: 128 videos x 600 1-fps frames), weight 1.
    mice  the 20,000-frame B subset (results/vision/eci_repr_diag/mice/b: ALL 5,394 contact frames of the 172,800 eval
          frames, weight 1, + 14,606 random non-contact frames, weight 11.46 = inverse sampling probability), because
          SAM 2 instance masks already exist for every one of them (results/vision/eci_mice_pairs/bsub, the
          pre-registered splitter of commit ec82118). Weighted, the subset represents all 172,800 eval frames.
          Mask quality = that splitter's automatic rule ('correct': 4 masks, areas and darkness in range); frames
          failing it are EXCLUDED from every instance statistic (the share excluded is reported per neuron).
          Strata (descriptive): contact (labelled nose-nose / nose-tail), huddle (>= 3 SAM mice in one touching cluster,
          masks dilated 3 px), spread (no two mice touching), partial (otherwise).

Instance labels per 16 px patch (32 x 32 grid of the 512 px frame)
    mice  patch overlaps mouse k iff >= 26 px (10% of the patch) of SAM mask k fall in it (a patch can overlap 2 mice);
          ON-ANIMAL patch = overlaps >= 1 mouse.
    ants  ON-ANIMAL patch = >= 26 px (10% of the patch, the same share as mice) pass the T2b noise-scaled PIXEL test
          (|grey - the video's per-time-block median background| > 30 grey levels, the T2 delta; the T2b delta of 6 -
          ants' noise sigma is 0, floored to 1 - sits inside the JPEG ringing around the dark ants: 10% at delta 6
          still keeps 43 patches per frame; backgrounds results/vision/eci_t2t4/ants/pixbg), on frames decoded
          locally from the standardised mkv.
          Changed from the T2b PATCH mask (>= 5% = 13 px, 47 patches per frame) before any sheet was looked at, after
          its counts showed it holds the ants plus a ~1-patch ring (dark-blob patches ~9 per frame), which would turn
          halo neurons into on-animal ones; the T2b-patch version is kept as the sensitivity variant 't2b'. Instances from the
          tracker: ant k (focal / yellow / blue) has the points {body centroid (gap-filled), raw colour dot (true
          detection, yellow / blue only)}; ant k belongs to a T2b component (8-connected patches) iff one of its points
          lies within 1 patch (Chebyshev) of the component; the component's patches go to the nearest split point among
          its ants (Voronoi). The tracker gives touching ants the IDENTICAL blob centroid (measured: 19-77% of frames
          per pair in the videos checked), so inside such a group the marked ants' split point is their raw dot and
          the ant without a dot gets n x C - sum(other dots) (neuron_map_v2.voronoi_points): the split of merged ants
          rests on the colour dots and is approximate. A component with no ant = 'untracked animal' (counted as 1
          animal only when the footprint region touches no tracked ant).

Per-neuron statistics, over its TOP frames = the highest frame values (max over the SAE mask's patches) until 1% of the
total frame weight is reached (ants: exactly the K = 768 top frames of tight_mask_sae.level_map), value > 0.
    active patches     the frame's patches where the neuron is > 0 (threshold inference)
    footprint regions  8-connected components of the active patches in that frame
    n_region           max over the frame's footprint regions of the number of distinct animals that region touches
                       (mice: union of the overlapped SAM masks; ants: union of the instance labels of its on-animal
                       patches). n_frame = the same count without the region split (all active patches together),
                       reported to show 'two separate animals active at once'; n_sites = regions touching an animal.
    on_share           share of active patches that are on-animal (weight = frame weight)
    off-animal patches split by Chebyshev distance to the nearest on-animal patch: near = 1, mid = 2, far >= 3
    zones              mice: BAG (bag core dilated 2 patches, per session) and WALL (inside the floor box < 2 patches
                       from its edge, plus OUTSIDE the floor box = the cage walls), per video, raw frame
                       (results/vision/eci_repr_diag/mice/switch_regions/prep/regions.npz reg_m2);
                       ants: EDGE = patch centre farther than R - 32 px from the dish centre (dish circle per video from all
                       tracked positions, neuron_map v1 dish_circles.csv; R < 205 px -> median circle). The lid is NOT
                       a place in this top view (it covers the whole dish; 'on lid' is a depth state), so ants have no
                       lid zone. share_Z = share of the neuron's ON-animal active patches in zone Z; base_Z = the same
                       share over every on-animal SAE-mask patch of every frame (weighted).
CLASSES (rules fixed here, before any label, anchor or sheet was looked at)
    location     on-animal         on_share >= 0.5
                 halo / spill      on_share < 0.5 and near >= 0.5 of the off-animal active patches
                 plain background  on_share < 0.5 and far >= 0.5 of the off-animal active patches
                 off-other         on_share < 0.5, neither
    social       weighted median of n_region over the top frames with valid instances: 0 -> none, 1 self, 2 pair,
                 >= 3 collective (>= 3 = 75% of a mouse group of 4 = all 3 ants)
    place-bound  (environment interaction) location on-animal AND some zone Z with share_Z >= 0.5 AND
                 share_Z >= min(2 x base_Z, (1 + base_Z) / 2)   (the high spatial concentration: at least twice the
                 animals' own use of that place, or halfway from it to 1 when the base rate is above 1/3)
    reported, not used: conc (1 - H / H_ref of the argmax patches on the 8 x 8 cell grid, as v1), n_frame, n_sites,
                 the shares of top frames with n_region 1 / 2 / >= 3, place_off (off-animal at a zone, same rule)
OLD (blob) map on the SAME frames and top frames: tight_mask_sae.level_map's rules re-implemented with frame weights
(T2 site maps: dark blobs, animal count = round(area / A1)): social from the medians NA~ / E~ (off-animal NA~ = 0, self
NA~ <= 1 and E~ <= 1.5, collective NA~ >= 3 or E~ >= 2.5, else pair), place_bound conc > 0.3 and argmax_off <= 0.5,
plain_background active_off > 0.5. Checked against tight_mask_sae.level_map itself (run in the encode step, unweighted).

Validation (analyse / sheets): visual sheets of 15 random neurons per new class; annotation anchors (in-sample AUROC
top-3 per label: ants groom_* -> expect pair, onlid_* -> expect self + place-bound; mice nose_nose / nose_tail -> pair;
bag neuron = highest BAG share of active patches -> place-bound); collective candidates with Spearman rho vs foreground
count and vs dispersion, partial on the count; old-vs-new confusion matrices.

Steps (scripts/eci/neuron_map_v2.sh)
    encode   GPU   per domain: stage tokens (+ d5 prev), encode every dictionary -> sparse patch codes (CSR), FVE / L0,
                   frame-max checks vs the deployed codes_max, tight_mask_sae.level_map reference
    prep     CPU   per domain: instance labels per patch, distance maps, zones, blob sites, tracking, dispersion, strata
    analyse  CPU   per dictionary: statistics, old / new classes, anchors, collective, confusion -> OUT/<dict>/
    sheets   CPU   contact sheets of the hand-check neurons (mice: bsub targets.tar; ants: the standardised mkv decoded
                   locally) -> OUT/sheets/<dict>/
Outputs results/vision/eci_neuron_map_v2/ (gitignored).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts/eci'))

OUT = REPO / 'results/vision/eci_neuron_map_v2'
PAIRS = REPO / 'results/vision/eci_mice_pairs/bsub'
T2 = REPO / 'results/vision/eci_t2t4'
COMB = REPO / 'results/vision/eci_combined'
LEV = REPO / 'results/vision/eci_sae_levers'
V1 = REPO / 'results/vision/eci_neuron_map'
REGIONS = REPO / 'results/vision/eci_repr_diag/mice/switch_regions/prep/regions.npz'
ODOR = REPO / 'dataset/mice/v1/eci/odor_corner.csv'
AL_STORE = 'dataset/mice/v1/eci/train_tokens/dinov2_base_l-1_fg448al_fps1'
DICTS = {
    'fg448m5': dict(domain='mice', motion=True, al=False,
                    ckpt='dataset/mice/v1/eci/sae/matryoshka_btk_4096_k16_fg448m5_s0/sae.pt', codes=None),
    'fg448al': dict(domain='mice', motion=False, al=True,
                    ckpt='dataset/mice/v1/eci/sae/matryoshka_btk_1024_k16_fg448al_s0/sae.pt',
                    codes='dataset/mice/v1/eci/codes/matryoshka_btk_1024_k16_fg448al_s0/codes_max.npy'),
    'antsfgm5': dict(domain='ants', motion=True, al=False,
                     ckpt='dataset/ants/eci/sae/matryoshka_btk_4096_k16_antsfgm5_s0/sae.pt', codes=None),
    'antsfg': dict(domain='ants', motion=False, al=False,
                   ckpt='dataset/ants/eci/sae/matryoshka_btk_1024_k16_antsfg_s0/sae.pt',
                   codes='dataset/ants/eci/codes/matryoshka_btk_1024_k16_antsfg_s0/codes_max.npy')}
DOM_DICTS = {'mice': ('fg448m5', 'fg448al'), 'ants': ('antsfgm5', 'antsfg')}
DEPLOY = ('fg448m5', 'antsfgm5')
OLD_ANCHORS = {'antsfg': {90: 'groom_any/Y2F/B2F (v1)', 246: 'onlid_yellow (v1)', 786: 'onlid_yellow (v1)',
                          860: 'onlid_yellow AP (v1)', 374: 'groom top1 (v1)'},
               'fg448al': {611: 'nose_nose (v1)', 414: 'nose_tail (v1)', 49: 'BAG (v1)'}}
ODOR_ROT90 = {'TR': 0, 'BR': 1, 'BL': 2, 'TL': 3}
GRID, PX = 32, 16
# fixed thresholds (docstring)
TOP_FRAC, MIN_FIRE = 0.01, 0.01
MICE_OVERLAP_PX = 26
ANTS_ON_PX = 26  # 10% of a patch's pixels pass the pixel test
ANTS_DELTA = 30  # grey levels (the T2 delta); the T2b delta 6 sits inside the JPEG ringing of ants (sigma 0)
ON_MIN, NEAR_MIN, FAR_MIN, ZONE_MIN = 0.5, 0.5, 0.5, 0.5
HUDDLE_DIL = 3
EDGE_BAND, DISH_R_MIN = 32.0, 205.0
# old rule (tight_mask_sae)
SELF_E, COLL_E, COLL_NA = 1.5, 2.5, 3
PLACE_CONC, PLACE_OFF, PLAIN_OFF = 0.3, 0.5, 0.5
N_CHECK = 15
log = lambda s: print(time.strftime('%H:%M:%S'), s, flush=True)  # noqa: E731


def local_dir():
    d = os.environ.get('LOCAL_DIR')
    if not d:
        raise SystemExit('LOCAL_DIR is not set (the job staging dir /localhome/$USER/$SLURM_JOB_ID)')
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    return d


def rot_lut():
    g = np.arange(GRID * GRID).reshape(GRID, GRID)
    return np.stack([np.rot90(g, k).ravel() for k in range(4)])


# ---------------------------------------------------------------------------------------------- encode (GPU)
def encode_csr(sae, norm, X, lens, dev):
    import torch

    import sae_levers as sl
    m = sae.n_latents
    st = np.r_[0, np.cumsum(lens)]
    I, V, C = [], [], []
    with torch.no_grad():
        for f0, f1 in sl.frame_chunks(lens, max(4096, 2 ** 27 // m)):
            lo, hi = st[f0], st[f1]
            if hi == lo:
                continue
            z = sae.encode(norm(torch.from_numpy(np.ascontiguousarray(X[lo:hi])).to(dev)), mode='threshold')
            nz = torch.nonzero(z)
            C.append(torch.bincount(nz[:, 0], minlength=hi - lo).cpu().numpy())
            I.append(nz[:, 1].to(torch.int16).cpu().numpy())
            V.append(z[nz[:, 0], nz[:, 1]].half().cpu().numpy())
    cnt = np.concatenate(C)
    assert len(cnt) == st[-1]
    return np.r_[0, np.cumsum(cnt)].astype(np.int64), np.concatenate(I), np.concatenate(V)


def frame_max_csr(indptr, I, V, lens, M):
    """-> (F, M) float32 frame max per latent (0 = not firing)."""
    F = len(lens)
    fid = np.repeat(np.arange(F), lens)
    tok_of = np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))
    out = np.zeros(F * M, np.float32)
    np.maximum.at(out, fid[tok_of] * M + I.astype(np.int64), V.astype(np.float32))
    return out.reshape(F, M)


def cmd_encode(args):
    import torch

    import combined as cb
    import motion_t3 as mt
    import multiscale_sae as ms
    import spatial_sae_pilot as ssp
    import tight_mask_sae as tms
    from src.eci.sae import load_sae
    d = args.domain
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False
    loc = local_dir()
    t0 = time.time()
    idx = ssp.StoreIndex(d)
    _, ev, _, _, _, _ = idx.split(d, ms.N_TRAIN[d], 0)
    ev = np.sort(ev)
    lab, beh, valid = ms.eval_labels(d, idx, ev)
    for b in beh:
        lab[f'valid_{b}'] = valid[b]
    if d == 'ants':
        k = np.arange(len(ev))
        Fr = lab.copy()
        Fr['w'] = 1.0
    else:
        B = pd.read_parquet(PAIRS / 'scores.parquet')
        assert (B.tid.values == np.arange(len(B))).all()
        r2k = pd.Series(np.arange(len(ev)), index=idx.rows[ev])
        kb = r2k.loc[B.row.values].values
        o = np.argsort(kb)
        k = kb[o]
        Fr = lab.iloc[k].reset_index(drop=True)
        Bo = B.iloc[o].reset_index(drop=True)
        assert (Bo.row.values == Fr.row.values).all()
        for c in ('tid', 'w', 'set', 'correct', 'n_masks', 'n_dark_blobs', 'merged_blob', 'branch', 'anchor_d'):
            Fr[c] = Bo[c].values
    frames = ev[k]
    Fr.insert(0, 'store_frame', frames)
    Fr.insert(1, 'ev_pos', k)
    Fr['n_fg'] = idx.nfg[frames]
    od = OUT / d
    od.mkdir(parents=True, exist_ok=True)
    Fr.to_parquet(od / 'frames.parquet')
    log(f'{d}: {len(frames):,} frames, {idx.nfg[frames].sum():,} tokens; labels {beh}')
    S = np.load(T2 / d / 'mask' / 'sites.npz')
    assert (S['frames'] == ev).all()
    site, bcount = S['site'][k], S['bcount'][k]
    a1 = json.loads((LEV / d / 'train_meta_w4096_w16384.json').read_text())['median_single_animal_area']
    tok, pos, lens = ms.stage(idx, frames, loc / 'tok')
    prev = None
    checks = {'n_frames': int(len(frames)), 'a1': a1}
    for key in DOM_DICTS[d]:
        cfg = DICTS[key]
        t1 = time.time()
        pos_raw = pos
        if cfg['al']:
            ssp.STORES['mice'] = AL_STORE
            ia = ssp.StoreIndex('mice')
            r2a = pd.Series(np.arange(len(ia.rows)), index=ia.rows)
            fa = r2a.loc[Fr.row.values].values
            assert (np.diff(fa) > 0).all(), 'aligned store frame order differs'
            X, pa, la = ms.stage(ia, fa, loc / 'tok_al')
            oc = pd.read_csv(ODOR).set_index('observation_id')
            rot = np.array([ODOR_ROT90[c] for c in oc.loc[Fr.obs.values, 'odor_corner']])
            pos_raw = rot_lut()[np.repeat(rot, la), pa.astype(np.int64)].astype(np.int16)
            ln = la
        elif cfg['motion']:
            if prev is None:
                prev = mt.stage_prev5(idx, d, frames, loc / 'prev.npy')
            X, ln = cb.MotionTokens(tok, prev), lens
        else:
            X, ln = tok, lens
        sae, norm, ck = load_sae(REPO / cfg['ckpt'], dev)
        indptr, I, V = encode_csr(sae, norm, X, ln, dev)
        cd = od / 'codes' / key
        cd.mkdir(parents=True, exist_ok=True)
        np.save(cd / 'indptr.npy', indptr)
        np.save(cd / 'idx.npy', I)
        np.save(cd / 'val.npy', V)
        np.save(cd / 'pos.npy', pos_raw.astype(np.int16))
        np.save(cd / 'lens.npy', ln.astype(np.int32))
        pres, ext, stt = cb.encode_pe(sae, norm, X, ln, dev)
        ch = {'n_latents': int(sae.n_latents), 'n_tokens': int(len(indptr) - 1), 'nnz': int(len(I)),
              'l0_per_token': float(len(I) / max(len(indptr) - 1, 1)),
              'stats': {k_: v for k_, v in stt.items() if np.isscalar(v)}}
        fm = frame_max_csr(indptr, I, V, ln, sae.n_latents)
        pm = pres.float().cpu().numpy()
        ch['csr_frame_max_vs_encode_pe'] = float(np.abs(fm - pm).max())
        if cfg['codes']:
            sel = np.sort(np.random.default_rng(0).choice(len(Fr), min(3000, len(Fr)), replace=False))
            ref = np.asarray(np.load(REPO / cfg['codes'], mmap_mode='r')[Fr.row.values[sel]], np.float32)
            ch['vs_deployed_codes_max'] = {
                'n_frames': int(len(sel)), 'firing_agree': float(((fm[sel] > 0) == (ref > 0)).mean()),
                'within_1pct': float((np.abs(fm[sel] - ref) <= 0.01 * np.abs(ref) + 1e-3).mean()),
                'max_abs_diff': float(np.abs(fm[sel] - ref).max())}
        with torch.no_grad():
            T, _, _, chk = tms.level_map(sae, norm, X, pos_raw, ln, pres, ext, site, bcount, a1, dev, bcount.shape[1])
        T.to_csv(cd / 'levels_tms_unweighted.csv', index=False)
        ch['level_map'] = chk
        ch['level_counts_unweighted'] = tms.level_counts(T)
        ch['elapsed_s'] = time.time() - t1
        checks[key] = ch
        log(f'{key}: {json.dumps(ch, default=float)}')
        del pres, ext, sae
        torch.cuda.empty_cache()
        if cfg['al']:
            del X
            shutil.rmtree(loc / 'tok_al')
    checks['stage'] = dict(mt.STAGE_INFO)
    checks['elapsed_s'] = time.time() - t0
    (od / 'encode_checks.json').write_text(json.dumps(checks, indent=1, default=float))
    log(f'{d}: encode done ({time.time() - t0:.0f}s)')



# ---------------------------------------------------------------------------------------------- prep (CPU)
def cheb_dist(on):
    """(32, 32) bool on-animal -> (1024,) uint8 Chebyshev distance (patches) to the nearest on-animal patch (99 if none)."""
    from scipy import ndimage as ndi
    if not on.any():
        return np.full(GRID * GRID, 99, np.uint8)
    return np.minimum(ndi.distance_transform_cdt(~on, metric='chessboard'), 99).astype(np.uint8).ravel()


def _mice_shard(path):
    """One SAM label shard -> per target id: patch overlap bits, distance map, instance centroids / areas, max touching
    cluster size."""
    from scipy import ndimage as ndi
    z = np.load(path)
    L, tids = z['labels'], z['idx']
    n = len(tids)
    bits = np.zeros((n, GRID * GRID), np.uint8)
    dist = np.zeros((n, GRID * GRID), np.uint8)
    cen = np.full((n, 4, 2), np.nan, np.float32)
    area = np.zeros((n, 4), np.int32)
    clus = np.zeros(n, np.int8)
    yy, xx = np.mgrid[0:512, 0:512]
    st = np.ones((2 * HUDDLE_DIL + 1, 2 * HUDDLE_DIL + 1), bool)
    for i in range(n):
        Li = L[i]
        cnt = np.stack([(Li == k).reshape(GRID, PX, GRID, PX).sum((1, 3)) for k in range(1, 5)], -1).reshape(-1, 4)
        b = np.zeros(GRID * GRID, np.uint8)
        for k in range(4):
            b |= ((cnt[:, k] >= MICE_OVERLAP_PX).astype(np.uint8) << k)
        bits[i] = b
        dist[i] = cheb_dist((b > 0).reshape(GRID, GRID))
        masks = [Li == k for k in range(1, 5)]
        dil = []
        for k, m in enumerate(masks):
            a = int(m.sum())
            area[i, k] = a
            if a:
                cen[i, k] = (yy[m].mean(), xx[m].mean())
            dil.append(ndi.binary_dilation(m, st) if a else m)
        par = list(range(4))

        def f(a):
            while par[a] != a:
                a = par[a]
            return a
        for a in range(4):
            for c in range(a + 1, 4):
                if area[i, a] and area[i, c] and (dil[a] & masks[c]).any():
                    par[f(a)] = f(c)
        roots = [f(a) for a in range(4) if area[i, a]]
        clus[i] = max(np.bincount(roots)) if roots else 0
    return tids, bits, dist, cen, area, clus


def voronoi_points(body, dots):
    """Split points per ant for one frame (patch units). The tracker gives MERGED ants the identical blob centroid, so
    within a group of coincident centroids (< 1 px apart) a marked ant uses its raw dot, and the one ant without a dot
    (focal, or a missed dot) uses n x C - sum(other dots) (the blob centroid C taken as the mean of the n ant centres)
    when every other member has a dot, else C. Ants not in such a group use their centroid and dot."""
    n = 3
    co = [[c for c in range(n) if np.isfinite(body[a]).all() and np.isfinite(body[c]).all()
           and np.hypot(*(body[a] - body[c])) < 1 / PX] for a in range(n)]
    out = []
    for a in range(n):
        if not np.isfinite(body[a]).all():
            if np.isfinite(dots[a]).all():
                out.append((a, dots[a][0], dots[a][1]))
            continue
        grp = co[a]
        if len(grp) <= 1:
            out.append((a, body[a][0], body[a][1]))
            if np.isfinite(dots[a]).all():
                out.append((a, dots[a][0], dots[a][1]))
        elif np.isfinite(dots[a]).all():
            out.append((a, dots[a][0], dots[a][1]))
        else:
            oth = [c for c in grp if c != a]
            if all(np.isfinite(dots[c]).all() for c in oth):
                q = len(grp) * body[a] - sum(dots[c] for c in oth)
                out.append((a, q[0], q[1]))
            else:
                out.append((a, body[a][0], body[a][1]))
    return out


def _ants_label(on_all, body, dots):
    """(n, 1024) on-animal patches + tracking points -> instance bits (1 focal, 2 yellow, 4 blue, 8 untracked),
    Chebyshev distance map, untracked component count."""
    from scipy import ndimage as ndi
    n = len(on_all)
    bits = np.zeros((n, GRID * GRID), np.uint8)
    dist = np.zeros((n, GRID * GRID), np.uint8)
    s8 = np.ones((3, 3), bool)
    cy, cx = np.divmod(np.arange(GRID * GRID), GRID)
    cy, cx = cy + 0.5, cx + 0.5
    n_untracked = np.zeros(n, np.int16)
    for i in range(n):
        on = on_all[i].reshape(GRID, GRID)
        dist[i] = cheb_dist(on)
        lab, nc = ndi.label(on, s8)
        if nc == 0:
            continue
        pts = []  # membership points (ant, x, y): body centroid + raw dot
        for a in range(3):
            for P in (body[i, a], dots[i, a]):
                if np.isfinite(P).all():
                    pts.append((a, P[0], P[1]))
        vor = voronoi_points(body[i], dots[i])
        b = np.zeros(GRID * GRID, np.uint8)
        labf = lab.ravel()
        for c in range(1, nc + 1):
            near = ndi.binary_dilation(lab == c, s8)
            mine = [p for p in pts if 0 <= int(p[2]) < GRID and 0 <= int(p[1]) < GRID and near[int(p[2]), int(p[1])]]
            ix = np.flatnonzero(labf == c)
            if not mine:
                b[ix] = 8
                n_untracked[i] += 1
                continue
            ants_in = sorted({p[0] for p in mine})
            sel_ = [q for q in vor if q[0] in ants_in]
            P = np.array([[q[1], q[2]] for q in sel_])
            A = np.array([q[0] for q in sel_])
            d2 = (cx[ix, None] - P[None, :, 0]) ** 2 + (cy[ix, None] - P[None, :, 1]) ** 2
            b[ix] = (1 << A[d2.argmin(1)]).astype(np.uint8)
        bits[i] = b
    return bits, dist, n_untracked


def _ants_video(job):
    """One ants video: decode the standardised mkv locally, per-pixel noise-scaled animal test (|grey - block median
    background| > delta), on-animal patches (>= 10% of pixels) + T2b patches -> instance labels; dispersion."""
    from PIL import Image

    from src.eci.foreground import segment_of
    obs, ex, fidx, rows, fpaths, on_t2b, delta, dst = job
    loc = Path(dst) / obs
    loc.mkdir(parents=True, exist_ok=True)
    mkv = REPO / f'data/ants/{ex}/observations/full/{obs}.mkv'
    subprocess.run(['ffmpeg', '-nostdin', '-loglevel', 'error', '-y', '-i', str(mkv), '-vf', 'fps=5,format=rgb24',
                    '-start_number', '0', '-q:v', '2', str(loc / 'frame_%06d.jpg')], check=True)
    pb = np.load(T2 / 'ants' / 'pixbg' / f'{obs}.npz')
    seg = segment_of(rows, pb['rows'], pb['seg_bounds'])
    cnt = np.zeros((len(fidx), GRID * GRID), np.uint16)
    cnt6 = np.zeros((len(fidx), GRID * GRID), np.uint16)
    mid = None
    for a in range(0, len(fidx), 64):
        g = np.stack([np.asarray(Image.open(loc / f'frame_{int(f):06d}.jpg').convert('RGB').convert('L'), np.uint8)
                      for f in fidx[a:a + 64]])
        ad = np.abs(g.astype(np.int16) - pb['bg'][seg[a:a + 64]].astype(np.int16))
        cnt[a:a + 64] = (ad > ANTS_DELTA).reshape(-1, GRID, PX, GRID, PX).sum((2, 4)).reshape(len(g), -1)
        cnt6[a:a + 64] = (ad > delta).reshape(-1, GRID, PX, GRID, PX).sum((2, 4)).reshape(len(g), -1)
        if a <= len(fidx) // 2 < a + 64:
            mid = (len(fidx) // 2, g[len(fidx) // 2 - a].copy())
    chk_bytes = int((REPO / 'dataset' / fpaths[0]).read_bytes() == (loc / f'frame_{int(fidx[0]):06d}.jpg').read_bytes())
    shutil.rmtree(loc)
    t = pd.read_csv(REPO / f'dataset/ants/{ex}/tracking/{obs}.csv').set_index('frame_idx').reindex(fidx)
    body = np.stack([t[[f'{a}_x', f'{a}_y']].values for a in ('focal', 'yellow', 'blue')], 1) / PX  # (n, 3, 2) x, y
    dots = np.stack([np.full((len(t), 2), np.nan), t[['raw_yellow_x', 'raw_yellow_y']].values,
                     t[['raw_blue_x', 'raw_blue_y']].values], 1) / PX
    on10 = cnt >= ANTS_ON_PX
    bits, dist, nun = _ants_label(on10, body, dots)
    bits_t, dist_t, _ = _ants_label(on_t2b, body, dots)
    t2b_rep = float(((cnt6 / 256.0 >= 0.05) == on_t2b).mean())
    bp = body * PX
    pd_ = [np.hypot(*(bp[:, a] - bp[:, c]).T) for a, c in ((0, 1), (0, 2), (1, 2))]
    disp = np.nanmean(np.stack(pd_, 1), 1)
    amb = np.nanmin(np.stack(pd_, 1), 1) < PX
    return (obs, bits, dist, bits_t, dist_t, disp.astype(np.float32), amb, nun,
            np.minimum(cnt, 255).astype(np.uint8), {'obs': obs, 'jpg_bytes_equal': chk_bytes, 't2b_reproduced': t2b_rep},
            np.minimum(cnt6, 255).astype(np.uint8), mid)


def ants_mask_sheet(F, mids, t2b, on6, on30, bits, path):
    """8 videos' middle frames: T2b patch mask (blue), 10% at delta 6 (yellow), 10% at delta 30 (green, tinted by
    instance)."""
    from PIL import Image, ImageDraw
    obs = sorted(mids)[::max(1, len(mids) // 8)][:8]
    sheet = Image.new('RGB', (4 * 512, 2 * 530), (255, 255, 255))
    for i, o in enumerate(obs):
        f, g = mids[o]
        im = Image.fromarray(np.stack([g] * 3, -1))
        dr = ImageDraw.Draw(im)
        for q in range(GRID * GRID):
            r, c = divmod(q, GRID)
            box = [c * PX, r * PX, c * PX + PX - 1, r * PX + PX - 1]
            if on30[f, q]:
                col = {1: (230, 25, 75), 2: (255, 200, 0), 4: (0, 130, 255), 8: (128, 128, 128)}.get(int(bits[f, q]), (0, 200, 0))
                dr.rectangle(box, outline=col, width=2)
            elif on6[f, q]:
                dr.rectangle([b + 3 if k < 2 else b - 3 for k, b in enumerate(box)], outline=(0, 220, 0), width=1)
            elif t2b[f, q]:
                dr.rectangle([b + 5 if k < 2 else b - 5 for k, b in enumerate(box)], outline=(120, 120, 255), width=1)
        sheet.paste(im, ((i % 4) * 512, (i // 4) * 530))
        ImageDraw.Draw(sheet).text(((i % 4) * 512 + 4, (i // 4) * 530 + 514),
                                   f'{o} f{F.frame_idx.values[f]}: delta30 on {on30[f].sum()} (instance colour), '
                                   f'+delta6 {on6[f].sum()} (green), +T2b {t2b[f].sum()} (blue)', fill=(0, 0, 0))
    sheet.save(path, quality=88)


def cmd_prep(args):
    from multiprocessing import get_context
    d = args.domain
    t0 = time.time()
    F = pd.read_parquet(OUT / d / 'frames.parquet')
    nF = len(F)
    S = np.load(T2 / d / 'mask' / 'sites.npz')
    site, bcount = S['site'][F.ev_pos.values], S['bcount'][F.ev_pos.values]
    assert (S['frames'][F.ev_pos.values] == F.store_frame.values).all()
    vids = sorted(F.obs.unique())
    vid = pd.Series(np.arange(len(vids)), index=vids).loc[F.obs.values].values
    meta = {'n_frames': nF, 'n_videos': len(vids)}
    if d == 'mice':
        pos_of_tid = np.full(int(F.tid.max()) + 1, -1)
        pos_of_tid[F.tid.values] = np.arange(nF)
        bits = np.zeros((nF, GRID * GRID), np.uint8)
        dist = np.zeros((nF, GRID * GRID), np.uint8)
        cen = np.full((nF, 4, 2), np.nan, np.float32)
        area = np.zeros((nF, 4), np.int32)
        clus = np.zeros(nF, np.int8)
        seen = np.zeros(nF, bool)
        paths = sorted(Path(os.environ['LOCAL_DIR']).glob('labels_s*.npz'))
        assert len(paths) == 5, paths
        with get_context('fork').Pool(min(args.workers, len(paths))) as pool:
            for tids, b, dd, c, a, cl in pool.imap_unordered(_mice_shard, paths):
                k = pos_of_tid[tids]
                assert (k >= 0).all()
                bits[k], dist[k], cen[k], area[k], clus[k] = b, dd, c, a, cl
                seen[k] = True
                log(f'  shard of {len(tids)} frames ({time.time() - t0:.0f}s)')
        assert seen.all(), f'{(~seen).sum()} frames without SAM labels'
        n_inst = (area > 0).sum(1)
        valid = F.correct.values.astype(bool) & (n_inst == 4)
        cc = cen.copy()
        pdist = [np.hypot(*(cc[:, a] - cc[:, b]).T) for a in range(4) for b in range(a + 1, 4)]
        disp = np.nanmean(np.stack(pdist, 1), 1).astype(np.float32)
        contact = F.nose_nose.values | F.nose_tail.values
        stratum = np.where(contact, 'contact', np.where(clus >= 3, 'huddle', np.where(clus <= 1, 'spread', 'partial')))
        R = np.load(REGIONS)
        rmap = pd.Series(np.arange(len(R['obs'])), index=R['obs'])
        zone_v = R['reg_m2'][rmap.loc[vids].values].astype(np.int8)  # 0 BAG, 1 WALL, 2 CENTRE, 3 OUTSIDE
        zone_names = ['BAG', 'WALL', 'CENTRE', 'OUTSIDE']
        w = F.w.values
        for s_ in ('contact', 'huddle', 'spread', 'partial'):
            m = stratum == s_
            meta[f'stratum_{s_}'] = {'n_frames': int(m.sum()), 'weighted_share': float(w[m].sum() / w.sum()),
                                     'valid_share': float(valid[m].mean()) if m.any() else None}
        meta['valid_frames'] = int(valid.sum())
        meta['valid_weighted_share'] = float(w[valid].sum() / w.sum())
        np.savez_compressed(OUT / d / 'prep.npz', bits=bits, dist=dist, site=site, bcount=bcount, vid=vid,
                            zone_v=zone_v, zone_names=zone_names, valid=valid, disp=disp, stratum=stratum,
                            clus=clus, cen=cen, area=area)
    else:
        Z = np.load(COMB / 'ants' / 'mask' / 'masks.npz')
        kz = np.searchsorted(Z['frames'], F.store_frame.values)
        assert (Z['frames'][kz] == F.store_frame.values).all() and Z['is_eval'][kz].all()
        on = np.unpackbits(Z['bits'][kz], axis=1).astype(bool)
        assert (on.sum(1) == Z['n_new'][kz]).all()
        delta = json.loads((COMB / 'rule.json').read_text())['ants']['delta']
        groups = F.groupby('obs', sort=False).indices
        dst = local_dir() / 'dec'
        jobs = [(o, F.experiment.values[ix[0]], F.frame_idx.values[ix], F.row.values[ix], F.frame_path.values[ix],
                 on[ix], delta, str(dst)) for o, ix in groups.items()]
        bits = np.zeros((nF, GRID * GRID), np.uint8)
        dist = np.zeros((nF, GRID * GRID), np.uint8)
        bits_t = np.zeros((nF, GRID * GRID), np.uint8)
        dist_t = np.zeros((nF, GRID * GRID), np.uint8)
        pxc = np.zeros((nF, GRID * GRID), np.uint8)
        pxc6 = np.zeros((nF, GRID * GRID), np.uint8)
        mids = {}
        disp = np.zeros(nF, np.float32)
        amb = np.zeros(nF, bool)
        nun = np.zeros(nF, np.int16)
        chk = []
        with get_context('fork').Pool(args.workers) as pool:
            for k_, (o, b, dd, bt, dt, ds, am, nu, pc, ck, pc6, mid) in enumerate(pool.imap_unordered(_ants_video, jobs)):
                ix = groups[o]
                bits[ix], dist[ix], bits_t[ix], dist_t[ix], disp[ix], amb[ix], nun[ix], pxc[ix] = b, dd, bt, dt, ds, am, nu, pc
                pxc6[ix] = pc6
                mids[o] = (ix[mid[0]], mid[1])
                chk.append(ck)
                if k_ % 16 == 0:
                    log(f'  ants {k_ + 1}/{len(jobs)} videos ({time.time() - t0:.0f}s) {ck}')
        C = pd.read_csv(V1 / 'ants' / 'dish_circles.csv').set_index('obs')
        small = C.R < DISH_R_MIN
        med = C[~small][['cx', 'cy', 'R']].median()
        C.loc[small, ['cx', 'cy', 'R']] = med.values
        C = C.loc[vids]
        cy, cx = np.divmod(np.arange(GRID * GRID), GRID)
        cy, cx = cy * PX + PX / 2, cx * PX + PX / 2
        r = np.hypot(cx[None] - C.cx.values[:, None], cy[None] - C.cy.values[:, None])
        zone_v = np.where(r > (C.R.values[:, None] - EDGE_BAND), 0, 1).astype(np.int8)  # 0 EDGE, 1 INNER
        zone_names = ['EDGE', 'INNER']
        valid = np.ones(nF, bool)
        on10 = pxc >= ANTS_ON_PX
        dark = S['dark'][F.ev_pos.values]
        meta.update({'dish_videos_fallback': int(small.loc[vids].sum()), 'dish_R_px': [float(C.R.min()),
                     float(C.R.median()), float(C.R.max())], 'pixel_delta': delta, 'on_min_px': ANTS_ON_PX,
                     'pixel_delta_used': ANTS_DELTA,
                     'on10_patches_per_frame': float(on10.sum(1).mean()),
                     'on10_delta6_patches_per_frame': float((pxc6 >= ANTS_ON_PX).sum(1).mean()),
                     'on10_off_dark_blob_share': float((on10 & (site == 0)).sum() / on10.sum()),
                     'dark_blob_patches_covered_by_on10': float((on10 & (site > 0) & (dark > 0)).sum() / max(((site > 0) & (dark > 0)).sum(), 1)),
                     't2b_patches_per_frame': float(on.sum(1).mean()),
                     'dark_cue_ge26px_patches_per_frame': float((dark >= ANTS_ON_PX).sum(1).mean()),
                     'site_blob_patches_per_frame': float((site > 0).sum(1).mean()),
                     'on10_label_share': {nm: float(((bits == v) & on10).sum() / on10.sum())
                                          for nm, v in (('focal', 1), ('yellow', 2), ('blue', 4), ('untracked', 8))},
                     'frames_with_untracked_component_on10': float((nun > 0).mean()),
                     'frames_two_ants_within_1_patch': float(amb.mean()),
                     't2b_reproduced_from_decoded_frames': float(np.mean([c['t2b_reproduced'] for c in chk])),
                     't2b_reproduced_min_video': float(np.min([c['t2b_reproduced'] for c in chk])),
                     'decoded_jpg_bytes_equal_videos': int(sum(c['jpg_bytes_equal'] for c in chk)),
                     'n_videos_checked': len(chk)})
        np.savez_compressed(OUT / d / 'prep.npz', bits=bits, dist=dist, site=site, bcount=bcount, vid=vid,
                            zone_v=zone_v, zone_names=zone_names, valid=valid, disp=disp, amb=amb, n_untracked=nun,
                            pix_count=pxc, pix_count6=pxc6)
        np.savez_compressed(OUT / d / 'prep_t2b.npz', bits=bits_t, dist=dist_t)
        ants_mask_sheet(F, mids, on, pxc6 >= ANTS_ON_PX, on10, bits, OUT / d / 'mask_sheet.jpg')
    meta['elapsed_s'] = time.time() - t0
    (OUT / d / 'prep.json').write_text(json.dumps(meta, indent=1, default=float))
    log(f'{d}: prep {json.dumps(meta, default=float)}')



# ---------------------------------------------------------------------------------------------- analyse (CPU)
POP = np.array([bin(i).count('1') for i in range(256)], np.int64)


def n_animals(bits, domain):
    """OR-ed instance bits -> number of animals (ants: untracked bit 8 counts 1 only without a tracked ant)."""
    bits = bits.astype(np.int64)
    if domain == 'mice':
        return POP[bits & 15]
    k = POP[bits & 7]
    return np.where(k > 0, k, (bits & 8) > 0).astype(np.int64)


def wmedian(vals, w, grp, n_grp):
    """Weighted lower median of vals per group (smallest x with cumulative weight >= half) -> (n_grp,) float, NaN if
    the group is empty. Unit weights = torch.nanmedian (lower median)."""
    out = np.full(n_grp, np.nan)
    ok = np.isfinite(vals) & (w > 0)
    vals, w, grp = vals[ok], w[ok], grp[ok]
    if not len(vals):
        return out
    o = np.lexsort((vals, grp))
    vals, w, grp = vals[o], w[o], grp[o]
    st = np.r_[0, np.flatnonzero(np.diff(grp)) + 1]
    cw = np.cumsum(w)
    base = np.repeat(np.r_[0, cw[st[1:] - 1]], np.diff(np.r_[st, len(vals)]))
    tot = np.repeat(np.add.reduceat(w, st), np.diff(np.r_[st, len(vals)]))
    hit = (cw - base) >= tot / 2 - 1e-9
    first = np.r_[True, grp[1:] != grp[:-1]]
    # first index per group where hit
    idx = np.flatnonzero(hit)
    g_idx = grp[idx]
    keep = np.r_[True, g_idx[1:] != g_idx[:-1]]
    out[g_idx[keep]] = vals[idx[keep]]
    del first
    return out


def entropy64(h):
    h = np.asarray(h, np.float64)
    p = h[h > 0] / h.sum() if h.sum() > 0 else np.array([1.0])
    return float(-(p * np.log(p)).sum())


def wauroc(score, y, w):
    """Weighted AUROC with ties (score (F,), y bool (F,), w (F,))."""
    o = np.argsort(score, kind='stable')
    s, yy, ww = score[o], y[o], w[o]
    st = np.r_[0, np.flatnonzero(np.diff(s)) + 1]
    wp = np.add.reduceat(np.where(yy, ww, 0.0), st)
    wn = np.add.reduceat(np.where(yy, 0.0, ww), st)
    below = np.r_[0, np.cumsum(wn)[:-1]]
    Wp, Wn = wp.sum(), wn.sum()
    return float((wp * (below + 0.5 * wn)).sum() / max(Wp * Wn, 1e-12))


class Data:
    def __init__(self, key, variant=''):
        self.key = key
        self.domain = d = DICTS[key]['domain']
        self.F = pd.read_parquet(OUT / d / 'frames.parquet')
        P = np.load(OUT / d / 'prep.npz')
        self.P = {k: P[k] for k in P.files}
        if variant == 't2b':
            Pt = np.load(OUT / d / 'prep_t2b.npz')
            self.P['bits'], self.P['dist'] = Pt['bits'], Pt['dist']
        cd = OUT / d / 'codes' / key
        self.indptr = np.load(cd / 'indptr.npy')
        self.I = np.load(cd / 'idx.npy').astype(np.int64)
        self.V = np.load(cd / 'val.npy').astype(np.float32)
        self.pos = np.load(cd / 'pos.npy').astype(np.int64)
        self.lens = np.load(cd / 'lens.npy').astype(np.int64)
        self.M = json.loads((OUT / d / 'encode_checks.json').read_text())[key]['n_latents']
        self.nF = len(self.F)
        assert self.lens.sum() == len(self.pos) == len(self.indptr) - 1 and len(self.lens) == self.nF
        self.fid_tok = np.repeat(np.arange(self.nF), self.lens)
        self.tok_of = np.repeat(np.arange(len(self.pos)), np.diff(self.indptr))
        self.w = self.F.w.values.astype(np.float64)
        self.a1 = json.loads((OUT / d / 'encode_checks.json').read_text())['a1']

    def frame_max(self):
        FM = np.zeros(self.M * self.nF, np.float32)
        np.maximum.at(FM, self.I * self.nF + self.fid_tok[self.tok_of], self.V)
        return FM.reshape(self.M, self.nF)


def top_sets(FMT, w, frac=TOP_FRAC):
    """Per neuron: the highest-value frames until frac of the total weight (value > 0). -> list of index arrays."""
    Kw = frac * w.sum()
    kmax = int(min(len(w), np.ceil(Kw / w.min()) + 1))
    out = []
    for j in range(len(FMT)):
        v = FMT[j]
        c = np.argpartition(-v, kmax - 1)[:kmax] if kmax < len(v) else np.arange(len(v))
        c = c[np.lexsort((c, -v[c]))]
        cw = np.cumsum(w[c])
        n = int(np.searchsorted(cw, Kw - 1e-9, 'left')) + 1
        c = c[:n]
        out.append(c[v[c] > 0])
    return out


def group_stats(D, tops, w_used):
    """Per (neuron, top frame) group statistics of the active patches (new instance and old blob)."""
    from scipy import ndimage as ndi
    nF, M = D.nF, D.M
    P = D.P
    gl = np.concatenate([np.full(len(t), j, np.int64) for j, t in enumerate(tops)])
    gf = np.concatenate(tops).astype(np.int64)
    G = len(gl)
    gkey = gl * nF + gf
    gid_of = np.full(M * nF, -1, np.int64)
    gid_of[gkey] = np.arange(G)
    rk = D.I * nF + D.fid_tok[D.tok_of]
    g = gid_of[rk]
    sel = np.flatnonzero(g >= 0)
    g = g[sel]
    o = np.argsort(g, kind='stable')
    sel, g = sel[o], g[o]
    tok = D.tok_of[sel]
    f = D.fid_tok[tok]
    p = D.pos[tok]
    val = D.V[sel]
    bits = P['bits'][f, p]
    dist = P['dist'][f, p].astype(np.int64)
    zone = P['zone_v'][P['vid'][f], p].astype(np.int64)
    site = P['site'][f, p].astype(np.int64)
    on = bits > 0
    off = ~on
    R = {'gl': gl, 'gf': gf, 'w': w_used[gf]}
    cnt = lambda m: np.bincount(g, weights=m.astype(np.float64), minlength=G)  # noqa: E731
    R['n_act'] = np.bincount(g, minlength=G).astype(np.float64)
    R['n_on'] = cnt(on)
    R['n_near'] = cnt(off & (dist == 1))
    R['n_mid'] = cnt(off & (dist == 2))
    R['n_far'] = cnt(off & (dist >= 3))
    nz = int(P['zone_v'].max()) + 1
    R['n_on_zone'] = np.stack([cnt(on & (zone == z)) for z in range(nz)], 1)
    R['n_off_zone'] = np.stack([cnt(off & (zone == z)) for z in range(nz)], 1)
    R['n_all_zone'] = np.stack([cnt(zone == z) for z in range(nz)], 1)
    # argmax patch per group (first of the highest value)
    st = np.r_[0, np.flatnonzero(np.diff(g)) + 1]
    oo = np.lexsort((-val, g))
    am = oo[np.r_[0, np.flatnonzero(np.diff(g[oo])) + 1]]
    assert len(am) == G == len(st), (len(am), G, len(st))
    R['am_pos'] = p[am]
    R['am_site_off'] = site[am] == 0
    R['am_on'] = on[am]
    # old: distinct site blobs and their animal counts
    ok = site > 0
    u = np.unique(g[ok] * 128 + site[ok])
    ug, us = u // 128, u % 128
    bc = P['bcount'][gf[ug], us - 1].astype(np.float64)
    R['NB'] = np.bincount(ug, minlength=G).astype(np.float64)
    R['NA'] = np.bincount(ug, weights=bc, minlength=G)
    R['act_off_old'] = cnt(site == 0)
    # new: footprint regions (8-connected components of the active patches per group), animals per region
    s3 = np.zeros((3, 3, 3), bool)
    s3[1] = True
    comp = np.zeros(len(g), np.int64)
    base = 0
    CH = 200_000
    for g0 in range(0, G, CH):
        g1 = min(G, g0 + CH)
        a, b = st[g0], (st[g1] if g1 < G else len(g))
        img = np.zeros((g1 - g0, GRID, GRID), bool)
        img[g[a:b] - g0, p[a:b] // GRID, p[a:b] % GRID] = True
        lab, n = ndi.label(img, s3)
        comp[a:b] = lab[g[a:b] - g0, p[a:b] // GRID, p[a:b] % GRID] + base - 1
        base += n
    cb_ = np.zeros(base, np.int64)
    np.bitwise_or.at(cb_, comp, bits.astype(np.int64))
    cg = np.zeros(base, np.int64)
    cg[comp] = g
    na_c = n_animals(cb_, D.domain)
    R['n_region'] = np.zeros(G)
    np.maximum.at(R['n_region'], cg, na_c.astype(np.float64))
    R['n_sites'] = np.bincount(cg, weights=(na_c > 0).astype(np.float64), minlength=G)
    R['n_regions_all'] = np.bincount(cg, minlength=G).astype(np.float64)
    gb = np.zeros(G, np.int64)
    np.bitwise_or.at(gb, g, bits.astype(np.int64))
    R['n_frame'] = n_animals(gb, D.domain).astype(np.float64)
    R['valid'] = P['valid'][gf]
    return R


def neuron_table(D, FMT, tops, R, Href_fn):
    M, d = D.M, D.domain
    gl, wg, valid = R['gl'], R['w'], R['valid']
    wv = np.where(valid, wg, 0.0)
    sw = lambda x, ww=wg: np.bincount(gl, weights=ww * x, minlength=M)  # noqa: E731
    T = pd.DataFrame({'neuron': np.arange(M)})
    W = D.w.sum()
    T['fire_frac'] = (FMT > 0).astype(np.float32) @ D.w.astype(np.float32) / W
    T['eligible'] = T.fire_frac >= MIN_FIRE
    T['n_top'] = [len(t) for t in tops]
    T['w_top'] = sw(np.ones(len(gl)))
    T['valid_top_share'] = sw(valid.astype(float)) / np.maximum(T.w_top, 1e-9)
    # ---- new
    act_v = sw(R['n_act'], wv)
    T['on_share'] = sw(R['n_on'], wv) / np.maximum(act_v, 1e-9)
    offv = sw(R['n_act'] - R['n_on'], wv)
    for k in ('near', 'mid', 'far'):
        T[f'off_{k}'] = sw(R[f'n_{k}'], wv) / np.maximum(offv, 1e-9)
    on_v = sw(R['n_on'], wv)
    zn = list(D.P['zone_names'])
    for z, nm in enumerate(zn):
        T[f'zone_on_{nm}'] = sw(R['n_on_zone'][:, z], wv) / np.maximum(on_v, 1e-9)
        T[f'zone_off_{nm}'] = sw(R['n_off_zone'][:, z], wv) / np.maximum(offv, 1e-9)
        T[f'zone_all_{nm}'] = sw(R['n_all_zone'][:, z], wv) / np.maximum(act_v, 1e-9)
    nreg = np.where(valid, R['n_region'], np.nan)
    T['n_region_med'] = wmedian(nreg, wg, gl, M)
    T['n_frame_med'] = wmedian(np.where(valid, R['n_frame'], np.nan), wg, gl, M)
    T['n_sites_med'] = wmedian(np.where(valid, R['n_sites'], np.nan), wg, gl, M)
    vt = np.maximum(sw(valid.astype(float)), 1e-9)
    for k, m in (('p0', R['n_region'] == 0), ('p1', R['n_region'] == 1), ('p2', R['n_region'] == 2),
                 ('p3', R['n_region'] >= 3)):
        T[f'nreg_{k}'] = sw((m & valid).astype(float)) / vt
    T['nframe_p2plus'] = sw(((R['n_frame'] >= 2) & valid).astype(float)) / vt
    # base zone rates over every on / off SAE-mask patch of every valid frame
    bits_all = D.P['bits'][D.fid_tok, D.pos]
    zone_all = D.P['zone_v'][D.P['vid'][D.fid_tok], D.pos]
    wt = (D.w * D.P['valid'])[D.fid_tok]
    on_all = bits_all > 0
    base_on = {nm: float((wt * (on_all & (zone_all == z))).sum() / (wt * on_all).sum()) for z, nm in enumerate(zn)}
    base_off = {nm: float((wt * (~on_all & (zone_all == z))).sum() / (wt * ~on_all).sum()) for z, nm in enumerate(zn)}
    place_zones = ['BAG', 'WALL'] if d == 'mice' else ['EDGE']
    if d == 'mice':  # WALL zone = WALL band + OUTSIDE the floor box
        for pre, base in (('zone_on_', base_on), ('zone_off_', base_off)):
            T[f'{pre}WALLX'] = T[f'{pre}WALL'] + T[f'{pre}OUTSIDE']
        base_on['WALLX'] = base_on['WALL'] + base_on['OUTSIDE']
        base_off['WALLX'] = base_off['WALL'] + base_off['OUTSIDE']
        place_zones = ['BAG', 'WALLX']

    def place_rule(pre, base):
        hit = np.zeros(M, bool)
        which = np.full(M, '', object)
        for nm in place_zones:
            b = base[nm]
            ok = (T[f'{pre}{nm}'] >= ZONE_MIN) & (T[f'{pre}{nm}'] >= min(2 * b, (1 + b) / 2))
            which[ok.values & ~hit] = nm
            hit |= ok.values
        return hit, which
    loc = np.where(T.on_share >= ON_MIN, 'on-animal', np.where(T.off_near >= NEAR_MIN, 'halo',
                   np.where(T.off_far >= FAR_MIN, 'plain-background', 'off-other')))
    T['location'] = np.where(np.isfinite(T.n_region_med), loc, 'no-valid-top')
    soc = np.where(T.n_region_med >= 3, 'collective', np.where(T.n_region_med == 2, 'pair',
                   np.where(T.n_region_med == 1, 'self', 'none')))
    T['social'] = np.where(np.isfinite(T.n_region_med), soc, 'no-valid-top')
    pb, pz = place_rule('zone_on_', base_on)
    T['place_bound'] = pb & (T.location == 'on-animal').values
    T['place_zone'] = np.where(T.place_bound, pz, '')
    po, poz = place_rule('zone_off_', base_off)
    T['place_off'] = po & (T.location != 'on-animal').values
    T['place_off_zone'] = np.where(T.place_off, poz, '')
    # new concentration (argmax patches on the 8 x 8 cell grid, as v1 / tms), weighted
    cell = (R['am_pos'] // GRID) // 4 * 8 + (R['am_pos'] % GRID) // 4
    H = np.zeros((M, 64))
    np.add.at(H, (gl, cell), wg)
    hr = {k: Href_fn(k) for k in np.unique(T.n_top.values) if k > 1}
    T['conc'] = [1 - entropy64(H[j]) / hr[T.n_top[j]] if T.n_top[j] > 1 and hr[T.n_top[j]] > 0 else np.nan
                 for j in range(M)]
    # ---- old (blob) rule, all top frames
    E = R['n_act'] / D.a1
    T['old_E_med'] = wmedian(E, wg, gl, M)
    T['old_NA_med'] = wmedian(R['NA'], wg, gl, M)
    T['old_NB_med'] = wmedian(R['NB'], wg, gl, M)
    T['old_active_off'] = sw(R['act_off_old']) / np.maximum(sw(R['n_act']), 1e-9)
    T['old_argmax_off'] = sw(R['am_site_off'].astype(float)) / np.maximum(T.w_top, 1e-9)
    T['old_plain_background'] = T.old_active_off > PLAIN_OFF
    T['old_place_bound'] = (T.conc > PLACE_CONC) & (T.old_argmax_off <= PLACE_OFF)
    T['old_social'] = np.where(T.n_top == 0, 'none', np.where(T.old_NA_med == 0, 'off-animal', np.where(
        (T.old_NA_med <= 1) & (T.old_E_med <= SELF_E), 'self', np.where(
            (T.old_NA_med >= COLL_NA) | (T.old_E_med >= COLL_E), 'collective', 'pair'))))
    return T, base_on, base_off


def href_factory(D, w):
    """H_ref(K): mean entropy of K SAE-mask patches drawn from the frames (frame drawn by weight, tms: uniform over
    tokens when the weights are equal), 200 draws, rng 0; memoised."""
    cells = (D.pos // GRID) // 4 * 8 + (D.pos % GRID) // 4
    pt = (w[D.fid_tok] / w[D.fid_tok].sum())
    unit = np.allclose(w, w[0])
    cdf = np.cumsum(pt)
    memo = {}

    def f(K):
        if K not in memo:
            rng = np.random.default_rng(0)
            if unit:
                memo[K] = float(np.mean([entropy64(np.bincount(cells[rng.choice(len(cells), K, replace=False)],
                                                               minlength=64)) for _ in range(200)]))
            else:  # weighted frames: draws with replacement from the token distribution (cdf sampling)
                memo[K] = float(np.mean([entropy64(np.bincount(cells[np.minimum(np.searchsorted(cdf, rng.random(K)),
                                                                                len(cells) - 1)], minlength=64))
                                         for _ in range(200)]))
        return memo[K]
    return f


def anchors(D, FMT, T):
    """In-sample AUROC (weighted) of every eligible neuron's frame value for each annotation label; top 3 per label."""
    F = D.F
    labels = (['groom_any', 'groom_yellow', 'groom_blue', 'onlid_yellow', 'onlid_blue', 'onlid_focal']
              if D.domain == 'ants' else ['nose_nose', 'nose_tail', 'nn_mutual'])
    out = {}
    elig = np.flatnonzero(T.eligible.values)
    for b in labels:
        y = F[b].values.astype(bool)
        v = F[f'valid_{b}'].values.astype(bool) if f'valid_{b}' in F else np.ones(len(F), bool)
        au = np.array([wauroc(FMT[j][v], y[v], D.w[v]) for j in elig])
        top = elig[np.argsort(-au)[:3]]
        out[b] = [{'neuron': int(j), 'auroc': float(au[list(elig).index(j)])} for j in top]
        log(f'  anchor {b}: ' + ', '.join(f'{e["neuron"]} ({e["auroc"]:.3f})' for e in out[b]))
    return out


def spearman_w(x, y, w):
    from scipy.stats import rankdata
    rx, ry = rankdata(x), rankdata(y)
    mx, my = np.average(rx, weights=w), np.average(ry, weights=w)
    c = np.average((rx - mx) * (ry - my), weights=w)
    return float(c / np.sqrt(np.average((rx - mx) ** 2, weights=w) * np.average((ry - my) ** 2, weights=w)))


def partial_w(x, y, z, w):
    rxy, rxz, ryz = spearman_w(x, y, w), spearman_w(x, z, w), spearman_w(y, z, w)
    return float((rxy - rxz * ryz) / np.sqrt(max((1 - rxz ** 2) * (1 - ryz ** 2), 1e-12)))


def collective(D, FMT, T):
    v = D.P['valid'] & np.isfinite(D.P['disp'])
    w, cnt, disp = D.w[v], D.lens[v].astype(float), D.P['disp'][v].astype(float)
    rows = []
    cand = set(np.flatnonzero((T.social == 'collective').values & T.eligible.values)) | set(
        np.flatnonzero((T.old_social == 'collective').values & T.eligible.values)) | set(
        np.flatnonzero((T.nreg_p3 >= 0.25).values & T.eligible.values))
    for j in sorted(cand):
        x = FMT[j][v]
        rows.append({'neuron': int(j), 'new_social': T.social[j], 'old_social': T.old_social[j],
                     'nreg_p3': float(T.nreg_p3[j]), 'n_region_med': float(T.n_region_med[j]),
                     'rho_count': spearman_w(x, cnt, w), 'rho_disp': spearman_w(x, disp, w),
                     'partial_disp_given_count': partial_w(x, disp, cnt, w)})
    # every eligible neuron: the strongest dispersion followers after the count (is any neuron collective at all?)
    elig = np.flatnonzero(T.eligible.values)
    rng = np.random.default_rng(0)
    sub = np.sort(rng.choice(int(v.sum()), min(int(v.sum()), 20000), replace=False))
    pr = np.array([partial_w(FMT[j][v][sub], disp[sub], cnt[sub], w[sub]) for j in elig])
    o = np.argsort(-np.abs(pr))[:10]
    top = [{'neuron': int(elig[i]), 'partial_disp_given_count_20k': float(pr[i]), 'new_social': T.social[elig[i]],
            'location': T.location[elig[i]], 'nreg_p3': float(T.nreg_p3[elig[i]])} for i in o]
    return {'candidates': rows, 'top_partial_any_eligible': top,
            'abs_partial_quantiles_eligible': {q: float(np.quantile(np.abs(pr), q)) for q in (0.5, 0.9, 0.99, 1.0)},
            'n_eligible': int(len(elig))}


def crosstab(a, b):
    t = pd.crosstab(pd.Series(a, name='old'), pd.Series(b, name='new'))
    return {str(i): {str(c): int(t.loc[i, c]) for c in t.columns} for i in t.index}


def cmd_analyse(args):
    key = args.dict
    t0 = time.time()
    D = Data(key, args.variant)
    od = OUT / (key + (f'_{args.variant}' if args.variant else ''))
    od.mkdir(parents=True, exist_ok=True)
    FMT = D.frame_max()
    log(f'{key}: frame max {FMT.shape} ({time.time() - t0:.0f}s)')
    tops = top_sets(FMT, D.w)
    log(f'  top sets: median {np.median([len(t) for t in tops]):.0f} frames ({time.time() - t0:.0f}s)')
    R = group_stats(D, tops, D.w)
    log(f'  group stats: {len(R["gl"]):,} groups ({time.time() - t0:.0f}s)')
    T, base_on, base_off = neuron_table(D, FMT, tops, R, href_factory(D, D.w))
    S = {'dict': key, 'domain': D.domain, 'n_frames': D.nF, 'n_latents': D.M, 'base_zone_on': base_on,
         'base_zone_off': base_off}
    # fidelity of the old-rule re-implementation: unit weights vs tight_mask_sae.level_map (same frames)
    ref = pd.read_csv(OUT / D.domain / 'codes' / key / 'levels_tms_unweighted.csv')
    if np.allclose(D.w, D.w[0]):
        Tu = T
    else:
        u = np.ones(D.nF)
        tu = top_sets(FMT, u)
        Ru = group_stats(D, tu, u)
        Tu, _, _ = neuron_table(D, FMT, tu, Ru, href_factory(D, u))
    e = ref.eligible.values
    S['old_rule_reimplementation_check'] = {
        'n_eligible_ref': int(e.sum()), 'eligible_agree': float((ref.eligible.values == Tu.eligible.values).mean()),
        'social_agree_eligible': float((ref.social.values[e] == Tu.old_social.values[e]).mean()),
        'place_bound_agree_eligible': float((ref.place_bound.values[e] == Tu.old_place_bound.values[e]).mean()),
        'plain_background_agree_eligible': float((ref.plain_background.values[e] == Tu.old_plain_background.values[e]).mean()),
        'NA_med_max_abs_diff': float(np.nanmax(np.abs(ref.NA_med.values[e] - Tu.old_NA_med.values[e]))),
        'E_med_max_abs_diff': float(np.nanmax(np.abs(ref.E_med.values[e] - Tu.old_E_med.values[e]))),
        'conc_max_abs_diff': float(np.nanmax(np.abs(ref.conc.values[e] - Tu.conc.values[e])))}
    log(f'  old-rule check: {json.dumps(S["old_rule_reimplementation_check"])}')
    del R
    E = T[T.eligible]
    S['n_eligible'] = int(len(E))
    S['counts'] = {
        'new_location': E.location.value_counts().to_dict(),
        'new_social_all_eligible': E.social.value_counts().to_dict(),
        'new_social_on_animal': E[E.location == 'on-animal'].social.value_counts().to_dict(),
        'new_place_bound': int(E.place_bound.sum()),
        'new_place_bound_by_zone': E[E.place_bound].place_zone.value_counts().to_dict(),
        'new_place_off': int(E.place_off.sum()),
        'new_place_off_by_zone': E[E.place_off].place_off_zone.value_counts().to_dict(),
        'old_social': E.old_social.value_counts().to_dict(),
        'old_place_bound': int(E.old_place_bound.sum()), 'old_plain_background': int(E.old_plain_background.sum())}
    S['confusion'] = {'social_old_vs_new': crosstab(E.old_social, E.social),
                      'social_old_vs_new_on_animal': crosstab(E[E.location == 'on-animal'].old_social,
                                                              E[E.location == 'on-animal'].social),
                      'place_bound_old_vs_new': crosstab(E.old_place_bound, E.place_bound),
                      'old_place_bound_vs_new_location': crosstab(E.old_place_bound, E.location),
                      'old_plain_background_vs_new_location': crosstab(E.old_plain_background, E.location)}
    S['agreement_social'] = float((E.old_social == E.social).mean())
    S['valid_top_share_median'] = float(E.valid_top_share.median())
    log(f'  counts {json.dumps(S["counts"], default=int)}')
    S['anchors'] = anchors(D, FMT, T)
    if key in OLD_ANCHORS:
        S['anchors']['v1_marked'] = [{'neuron': j, 'what': w_} for j, w_ in OLD_ANCHORS[key].items()]
    if D.domain == 'mice':
        b = E.sort_values('zone_all_BAG', ascending=False).head(5)
        S['anchors']['bag_like'] = [{'neuron': int(j), 'bag_share_active': float(x)}
                                    for j, x in zip(b.neuron, b.zone_all_BAG)]
    cols = ['neuron', 'fire_frac', 'n_top', 'valid_top_share', 'location', 'social', 'place_bound', 'place_zone',
            'place_off', 'on_share', 'off_near', 'off_far', 'n_region_med', 'n_frame_med', 'nreg_p1', 'nreg_p2',
            'nreg_p3', 'conc', 'old_social', 'old_place_bound', 'old_plain_background', 'old_NA_med', 'old_E_med']
    for grp in S['anchors'].values():
        for e_ in grp:
            j = e_['neuron']
            e_.update({c: (T[c][j].item() if hasattr(T[c][j], 'item') else T[c][j]) for c in cols[1:]})
            e_['eligible'] = bool(T.eligible[j])
    S['collective'] = collective(D, FMT, T)
    log(f'  collective: {len(S["collective"]["candidates"])} candidates; top partial '
        f'{S["collective"]["top_partial_any_eligible"][:3]}')
    # hand-check picks: 15 random eligible neurons per new class (fixed seed)
    rng = np.random.default_rng(0)
    on = E[E.location == 'on-animal']
    classes = {'self': on[on.social == 'self'], 'pair': on[on.social == 'pair'],
               'collective': on[on.social == 'collective'], 'place-bound': E[E.place_bound],
               'halo': E[E.location == 'halo'], 'plain-background': E[E.location == 'plain-background']}
    S['check_picks'] = {c: sorted(int(x) for x in rng.choice(t.neuron.values, min(N_CHECK, len(t)), replace=False))
                        for c, t in classes.items() if len(t)}
    T.to_csv(od / 'neurons.csv', index=False)
    np.savez(od / 'top_frames.npz', flat=np.concatenate(tops), off=np.r_[0, np.cumsum([len(t) for t in tops])])
    S['elapsed_s'] = time.time() - t0
    (od / 'summary.json').write_text(json.dumps(S, indent=1, default=lambda x: x.item() if hasattr(x, 'item') else str(x)))
    log(f'{key}: analyse done ({time.time() - t0:.0f}s)')



# ---------------------------------------------------------------------------------------------- sheets (CPU)
N_SHEET = 12
INST_RGB = np.array([[230, 25, 75], [255, 215, 0], [0, 130, 255], [60, 200, 75]], np.uint8)  # mice 1-4 / ants f,y,b


def _decode_ants(job):
    obs, ex, fidx, dst = job
    loc = Path(dst) / obs
    loc.mkdir(parents=True, exist_ok=True)
    mkv = REPO / f'data/ants/{ex}/observations/full/{obs}.mkv'
    subprocess.run(['ffmpeg', '-nostdin', '-loglevel', 'error', '-y', '-i', str(mkv), '-vf', 'fps=5,format=rgb24',
                    '-start_number', '0', '-q:v', '2', str(loc / 'frame_%06d.jpg')], check=True)
    from PIL import Image
    out = {int(f): np.asarray(Image.open(loc / f'frame_{int(f):06d}.jpg').convert('RGB')) for f in fidx}
    shutil.rmtree(loc)
    return obs, out


def _decode_mice(job):
    """Selected 5 fps frames of one standardised mice mp4 (fps=5 then select by output index), decoded locally."""
    from PIL import Image
    v, fidx, fpaths, dst = job
    loc = Path(dst) / v
    loc.mkdir(parents=True, exist_ok=True)
    o = np.argsort(fidx)
    fidx, fpaths = np.asarray(fidx)[o], np.asarray(fpaths)[o]
    sel = '+'.join(f'eq(n\\,{int(f)})' for f in fidx)
    subprocess.run(['ffmpeg', '-nostdin', '-loglevel', 'error', '-y', '-i',
                    str(REPO / f'data/mice/v1/observations/full/{v}.mp4'), '-vf', f'fps=5,select={sel},format=rgb24',
                    '-vsync', '0', '-start_number', '0', str(loc / 'out_%06d.png')], check=True)
    files = sorted(loc.glob('out_*.png'))
    assert len(files) == len(fidx), (v, len(files), len(fidx))
    out = {p: np.asarray(Image.open(f).convert('RGB')) for p, f in zip(fpaths, files)}
    shutil.rmtree(loc)
    return out


def sheet_frames(D, picks, tops):
    """-> {neuron: [frame indices (12, evenly spaced in rank over the valid top frames)]}."""
    out = {}
    FMv = {}
    for j in picks:
        t = tops[j]
        t = t[D.P['valid'][t]]
        if not len(t):
            continue
        k = np.unique(np.linspace(0, len(t) - 1, min(N_SHEET, len(t))).round().astype(int))
        out[j] = t[k]
        FMv[j] = None
    return out


def render_tile(img, act, val, bits, zone, zone_names, domain, inst_px=None, pts=None, dish=None, title=''):
    """One frame: left = full frame (256 px), right = 2x zoom around the footprint (256 px)."""
    from PIL import Image, ImageDraw
    im = img.astype(np.float32).copy()
    if inst_px is not None:  # mice: SAM pixel masks tinted
        for k in range(1, 5):
            m = inst_px == k
            im[m] = 0.55 * im[m] + 0.45 * INST_RGB[k - 1]
    else:  # ants: patch-level instance labels tinted
        for b_, col in ((1, INST_RGB[0]), (2, INST_RGB[1]), (4, INST_RGB[2]), (8, np.array([150, 150, 150]))):
            for q in np.flatnonzero(bits == b_):
                r, c = divmod(int(q), GRID)
                sl = im[r * PX:(r + 1) * PX, c * PX:(c + 1) * PX]
                sl[:] = 0.65 * sl + 0.35 * col
    pil = Image.fromarray(im.clip(0, 255).astype(np.uint8))
    dr = ImageDraw.Draw(pil)
    if domain == 'mice':  # zone outlines: BAG cyan, WALL band + outside thin magenta
        for q in range(GRID * GRID):
            r, c = divmod(q, GRID)
            if zone[q] == 0:
                dr.rectangle([c * PX, r * PX, c * PX + PX - 1, r * PX + PX - 1], outline=(0, 255, 255), width=1)
    elif dish is not None:
        cx, cy, R = dish
        for rad, col in ((R, (255, 0, 255)), (R - EDGE_BAND, (255, 120, 255))):
            dr.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], outline=col, width=1)
    vmax = max(float(val.max()), 1e-6)
    am = act[int(np.argmax(val))]
    for q, v in zip(act, val):
        r, c = divmod(int(q), GRID)
        g = int(80 + 175 * v / vmax)
        dr.rectangle([c * PX, r * PX, c * PX + PX - 1, r * PX + PX - 1], outline=(g, 255 - g // 2, 0), width=2)
    r, c = divmod(int(am), GRID)
    dr.rectangle([c * PX - 2, r * PX - 2, c * PX + PX + 1, r * PX + PX + 1], outline=(255, 255, 255), width=2)
    if pts is not None:
        for (x, y, col) in pts:
            if np.isfinite(x) and np.isfinite(y):
                dr.ellipse([x - 4, y - 4, x + 4, y + 4], outline=col, width=2)
    rr, cc = np.divmod(act.astype(int), GRID)
    y0, y1 = rr.min() * PX, (rr.max() + 1) * PX
    x0, x1 = cc.min() * PX, (cc.max() + 1) * PX
    side = max(128, y1 - y0, x1 - x0) + 32
    yc, xc = (y0 + y1) / 2, (x0 + x1) / 2
    y0, x0 = int(np.clip(yc - side / 2, 0, 512 - side)), int(np.clip(xc - side / 2, 0, 512 - side))
    zoom = pil.crop((x0, y0, x0 + side, y0 + side)).resize((256, 256), Image.NEAREST)
    full = pil.resize((256, 256), Image.BILINEAR)
    ImageDraw.Draw(full).rectangle([x0 / 2, y0 / 2, (x0 + side) / 2, (y0 + side) / 2], outline=(255, 255, 255))
    tile = Image.new('RGB', (512, 276), (255, 255, 255))
    tile.paste(full, (0, 0))
    tile.paste(zoom, (256, 0))
    ImageDraw.Draw(tile).text((4, 259), title, fill=(0, 0, 0))
    return tile


def cmd_sheets(args):
    from PIL import Image, ImageDraw
    key = args.dict
    D = Data(key)
    d = D.domain
    S = json.loads((OUT / key / 'summary.json').read_text())
    T = pd.read_csv(OUT / key / 'neurons.csv')
    z = np.load(OUT / key / 'top_frames.npz')
    tops = [z['flat'][a:b] for a, b in zip(z['off'][:-1], z['off'][1:])]
    want = {}
    for c, js in S['check_picks'].items():
        for j in js:
            want.setdefault(int(j), []).append(c)
    for b, lst in S['anchors'].items():
        for e in lst:
            want.setdefault(int(e['neuron']), []).append(f'anchor:{b}')
    SF = sheet_frames(D, list(want), tops)
    need = sorted({int(f) for v in SF.values() for f in v})
    log(f'{key}: {len(want)} neurons, {len(need)} frames')
    loc = local_dir()
    imgs, inst = {}, {}
    if d == 'mice':
        from multiprocessing import get_context
        sub = D.F.iloc[need]
        vname = sub.frame_path.str.split('/').str[-2].values
        jobs = [(v, sub.frame_idx.values[vname == v], sub.frame_path.values[vname == v], str(loc / 'dec'))
                for v in sorted(set(vname))]
        with get_context('fork').Pool(args.workers) as pool:
            for out in pool.imap_unordered(_decode_mice, jobs):
                imgs.update(out)
        chk = []
        for f in need[:3]:
            a = np.asarray(Image.open(REPO / 'dataset' / D.F.frame_path.values[f]).convert('RGB'), np.int16)
            chk.append(int(np.abs(a - imgs[D.F.frame_path.values[f]].astype(np.int16)).max()))
        log(f'  decode check vs dataset JPG (max abs px of 3 frames): {chk}')
        tid_need = {int(D.F.tid.values[f]): f for f in need}
        for pth in sorted(PAIRS.glob('labels_s*.npz')):
            shutil.copyfile(pth, loc / pth.name)
            zz = np.load(loc / pth.name)
            ids = zz['idx']
            hit = [i for i, t_ in enumerate(ids) if int(t_) in tid_need]
            if hit:
                L = zz['labels']
                for i in hit:
                    inst[tid_need[int(ids[i])]] = L[i].copy()
                del L
            (loc / pth.name).unlink()
        get_img = lambda f: imgs[D.F.frame_path.values[f]]  # noqa: E731
    else:
        from multiprocessing import get_context
        jobs = []
        for o, g in D.F.iloc[need].groupby('obs'):
            jobs.append((o, g.experiment.values[0], g.frame_idx.values, str(loc / 'dec')))
        with get_context('fork').Pool(args.workers) as pool:
            for o, out in pool.imap_unordered(_decode_ants, jobs):
                imgs[o] = out
        chk = []
        for f in need[:3]:
            a = np.asarray(Image.open(REPO / 'dataset' / D.F.frame_path.values[f]).convert('RGB'), np.int16)
            b = imgs[D.F.obs.values[f]][int(D.F.frame_idx.values[f])].astype(np.int16)
            chk.append(int(np.abs(a - b).max()))
        log(f'  decode check vs dataset JPG (max abs px of 3 frames): {chk}')
        get_img = lambda f: imgs[D.F.obs.values[f]][int(D.F.frame_idx.values[f])]  # noqa: E731
        C = pd.read_csv(V1 / 'ants' / 'dish_circles.csv').set_index('obs')
        small = C.R < DISH_R_MIN
        C.loc[small, ['cx', 'cy', 'R']] = C[~small][['cx', 'cy', 'R']].median().values
        trk = {}
        for o, g in D.F.iloc[need].groupby('obs'):
            ex = g.experiment.values[0]
            trk[o] = pd.read_csv(REPO / f'dataset/ants/{ex}/tracking/{o}.csv').set_index('frame_idx')
    od = OUT / 'sheets' / key
    od.mkdir(parents=True, exist_ok=True)
    zn = list(D.P['zone_names'])
    index = []
    tst = np.r_[0, np.cumsum(D.lens)]
    for j, frs in SF.items():
        tiles = []
        for f in frs:
            t0_, t1_ = tst[f], tst[f + 1]
            act, val = [], []
            for tk in range(t0_, t1_):
                lo, hi = D.indptr[tk], D.indptr[tk + 1]
                m = D.I[lo:hi] == j
                if m.any():
                    act.append(D.pos[tk])
                    val.append(D.V[lo:hi][m][0])
            act, val = np.array(act), np.array(val)
            bits = D.P['bits'][f]
            zone = D.P['zone_v'][D.P['vid'][f]]
            on = bits[act] > 0
            nreg_title = f'w{D.w[f]:.0f} act {len(act)} on {on.sum()} | n_frame {int(n_animals(np.bitwise_or.reduce(bits[act].astype(np.int64))[None], d)[0])}'
            if d == 'mice':
                tile = render_tile(get_img(f), act, val, bits, zone, zn, d, inst_px=inst.get(int(f)),
                                   title=f'{D.F.obs.values[f]} f{D.F.frame_idx.values[f]} {D.P["stratum"][f]} ' + nreg_title)
            else:
                o = D.F.obs.values[f]
                r_ = trk[o].loc[int(D.F.frame_idx.values[f])]
                pts = [(r_.focal_x, r_.focal_y, (230, 25, 75)), (r_.yellow_x, r_.yellow_y, (255, 215, 0)),
                       (r_.blue_x, r_.blue_y, (0, 130, 255)), (r_.raw_yellow_x, r_.raw_yellow_y, (255, 255, 255)),
                       (r_.raw_blue_x, r_.raw_blue_y, (255, 255, 255))]
                c_ = C.loc[o]
                tile = render_tile(get_img(f), act, val, bits, zone, zn, d, pts=pts, dish=(c_.cx, c_.cy, c_.R),
                                   title=f'{o} f{D.F.frame_idx.values[f]} ' + nreg_title)
            tiles.append(tile)
        cols = 3
        rows = int(np.ceil(len(tiles) / cols))
        sheet = Image.new('RGB', (cols * 512, rows * 276 + 40), (255, 255, 255))
        for i, t_ in enumerate(tiles):
            sheet.paste(t_, ((i % cols) * 512, 40 + (i // cols) * 276))
        r = T.loc[j]
        head = (f'{key} n{j} [{", ".join(want[j])}] NEW: {r.location}, {r.social}, place_bound {r.place_bound} '
                f'{r.place_zone if isinstance(r.place_zone, str) else ""} | on {r.on_share:.2f} near {r.off_near:.2f} '
                f'far {r.off_far:.2f} nreg p1/p2/p3 {r.nreg_p1:.2f}/{r.nreg_p2:.2f}/{r.nreg_p3:.2f} | OLD: {r.old_social}, '
                f'place {r.old_place_bound}, plain {r.old_plain_background}')
        ImageDraw.Draw(sheet).text((6, 6), head, fill=(0, 0, 0))
        ImageDraw.Draw(sheet).text((6, 22), 'left: full frame (white box = zoom); right: 2x zoom. green-orange boxes = active '
                                   'patches (brighter = higher), white box = argmax; tint = instances', fill=(80, 80, 80))
        name = f'n{j:04d}.jpg'
        sheet.save(od / name, quality=85)
        index.append({'neuron': int(j), 'why': want[j], 'sheet': str((od / name).relative_to(REPO)),
                      'frames': [int(x) for x in frs]})
    (od / 'index.json').write_text(json.dumps(index, indent=1))
    log(f'{key}: {len(index)} sheets -> {od}')



# ---------------------------------------------------------------------------------------------- extras (CPU, post hoc)
def cmd_extras(args):
    """POST-HOC descriptive additions (decided after the hand-check started, not used by the fixed classes):
    top-video share per neuron; multi-site flag (n_sites_med >= 2: the neuron fires at >= 2 separate places in its
    median top frame - the generic 'every animal' neurons of the ants pair class); the animals' own cluster structure
    (largest number of animals in one touching cluster per frame) overall and in annotated frames."""
    from scipy import ndimage as ndi
    key = args.dict
    D = Data(key)
    d = D.domain
    T = pd.read_csv(OUT / key / 'neurons.csv')
    z = np.load(OUT / key / 'top_frames.npz')
    tops = [z['flat'][a:b] for a, b in zip(z['off'][:-1], z['off'][1:])]
    vid = D.P['vid']
    share = np.zeros(D.M)
    for j, t in enumerate(tops):
        if len(t):
            share[j] = np.bincount(vid[t], weights=D.w[t]).max() / D.w[t].sum()
    T['top_video_share'] = share
    T['multi_site'] = T.n_sites_med >= 2
    E = T[T.eligible]
    on = E[E.location == 'on-animal']
    R = {'dict': key, 'note': 'post hoc, descriptive',
         'video_specific_share_eligible (top_video_share > 0.5)': float((E.top_video_share > 0.5).mean()),
         'by_class': {}}
    for nm, sub in (('self', on[on.social == 'self']), ('pair', on[on.social == 'pair']),
                    ('collective', on[on.social == 'collective']), ('place-bound', E[E.place_bound]),
                    ('halo', E[E.location == 'halo']), ('plain-background', E[E.location == 'plain-background'])):
        R['by_class'][nm] = {'n': int(len(sub)), 'multi_site': int(sub.multi_site.sum()),
                             'video_specific': int((sub.top_video_share > 0.5).sum()),
                             'median_top_video_share': float(sub.top_video_share.median()) if len(sub) else None}
    pr = on[on.social == 'pair']
    R['pair_single_site'] = int((~pr.multi_site).sum())
    # the animals' own cluster structure
    if d == 'ants':
        s8 = np.ones((3, 3), bool)
        mx = np.zeros(D.nF, np.int8)
        for i in range(D.nF):
            b = D.P['bits'][i].reshape(GRID, GRID)
            lab, n = ndi.label(b > 0, s8)
            if n == 0:
                continue
            acc = np.zeros(n + 1, np.int64)
            np.bitwise_or.at(acc, lab.ravel(), (b.ravel() & 7).astype(np.int64))
            mx[i] = POP[acc[1:]].max()
        F = D.F
        rows = {'all': np.ones(D.nF, bool), 'groom_any': F.groom_any.values.astype(bool),
                'onlid_any': (F.onlid_yellow.values | F.onlid_blue.values | F.onlid_focal.values).astype(bool)}
        R['ant_cluster_size'] = {k: {str(c): float((mx[m] == c).mean()) for c in range(4)} | {'n': int(m.sum())}
                                 for k, m in rows.items()}
    else:
        F = D.F
        cl = D.P['clus']
        v = D.P['valid']
        rows = {'all': v, 'nose_nose': v & F.nose_nose.values.astype(bool), 'nose_tail': v & F.nose_tail.values.astype(bool)}
        R['mouse_cluster_size_weighted'] = {k: {str(c): float(D.w[m & (cl == c)].sum() / D.w[m].sum()) for c in range(5)}
                                            | {'n': int(m.sum())} for k, m in rows.items()}
        R['strata_weighted'] = {s_: float(D.w[D.P['stratum'] == s_].sum() / D.w.sum())
                                for s_ in ('contact', 'huddle', 'spread', 'partial')}
    T.to_csv(OUT / key / 'neurons.csv', index=False)
    (OUT / key / 'extras.json').write_text(json.dumps(R, indent=1))
    log(f'{key}: extras {json.dumps(R)}')


# ---------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['encode', 'prep', 'analyse', 'sheets', 'extras'])
    ap.add_argument('--domain', choices=['mice', 'ants'])
    ap.add_argument('--dict', choices=list(DICTS))
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--variant', default='', choices=['', 't2b'], help='ants: on-animal = the T2b patch mask (5%%)')
    args = ap.parse_args()
    globals()[f'cmd_{args.step}'](args)


if __name__ == '__main__':
    main()
