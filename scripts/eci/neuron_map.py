"""
Post-hoc BEHAVIOURAL MAP of the neurons of the existing ECI SAE dictionaries (no new model, no label used to define
anything). For every neuron: where does it fire (environment or animal), how many animals sit at its firing site,
does it follow group dispersion, and (ants) does it prefer one colour-marked ant. Tracking, dark blobs, the bag zone
and the dish circle are ANALYSIS TOOLS only.

Dictionaries
    ants  antsfg      deployed Matryoshka BatchTopK 1024, k 16 (dataset/ants/eci/sae/matryoshka_btk_1024_k16_antsfg_s0),
                      patch codes reused from results/vision/eci_repr_diag/ants/patch_codes (scripts/eci/diag_ants_pairs.py)
    mice  fg448al     deployed 1024, odour-corner-aligned frames (dataset/mice/v1/eci/sae/matryoshka_btk_1024_k16_fg448al_s0)
    mice  w4096cell   results/vision/eci_sae_levers/mice/sae/w4096+cell_s0 (4096 latents, per-(video, position) centring,
                      CellMeans over ALL 172,800 eval frames exactly as sae_levers / diag_switch_regions did)

Frames (stratified by video, evenly spaced in time within each video, fixed before any analysis)
    ants  150 of the 600 1-fps store frames of each of the 256 videos -> 38,400 frames (v2 44 videos, v3 212)
    mice  210 of the 900-1800 1-fps eval frames of each of the 144 annotated videos -> 30,240 frames

Analysis tools (per frame, raw camera frame, 512 px, patch = 16 px)
    dark cue        grey < 60 and pix_bg - grey > 40 (the foreground rule's dark cue, src/eci/foreground.py)
    dark blobs      the split test's dark-blob detector (scripts/eci/split_test_common.py dark_clean / blob_labels:
                    opening, hole filling, 8-connected components >= min_area px). Single-animal dark area A1 from
                    its label-free calibrations (results/vision/eci_split_test/<d>/calib.json): ants 302 px, mice 1985.5 px.
    site blob of a patch   the blob holding most of the patch's blob pixels; if the patch holds none, the blob nearest
                    to the patch centre if within 16 px (one patch); else none (= off-animal)
    animal count    clip(round(site blob area / A1), 1, 4); 0 = no site blob
    ants tracking   dataset/ants/<v>/tracking/<obs>.csv (5 fps): raw_yellow/raw_blue = true colour-dot detections (NaN if
                    missed), focal_x/y and yellow_x/blue_x = gap-filled body centroids
    ants dish       per video, from all tracked positions (body centroids + raw dots): centre = midpoint of the 0.5% /
                    99.5% quantiles in x and y, radius R = 99.5% quantile of the distance to it. EDGE = patch centre
                    farther than R - 32 px (2 patches) from the centre. The repo has NO lid geometry (searched), so the
                    ants environment cue (c) is edge only. Videos whose tracked extent gives R < 205 px (the ants never
                    explored the whole dish) use the median circle of the other videos.
    mice regions    results/vision/eci_repr_diag/mice/switch_regions/prep/regions.npz reg_m2 (raw frame): BAG (label-free
                    bag core, dilated 2 patches), WALL (inside the floor box, < 2 patches from its edge), OUTSIDE
                    (outside the floor box = the cage wall surfaces), CENTRE. (c) = BAG or OUTSIDE (off the floor).
                    fg448al positions are rotated back to the raw frame (np.rot90 inverse) for every pixel-based tool;
                    the spatial concentration (a) uses the SAE's own frame.

Per-neuron statistics (DEFINITIONS AND THRESHOLDS FIXED HERE, BEFORE ANY LABEL ALIGNMENT IS LOOKED AT)
    frame value     max over the frame's foreground patches (as codes_max / codes_best)
    eligible        fires (frame value > 0) on >= 1% of frames; others are 'rare' (> 0 and < 1%) or 'dead' (never)
    top patches     the argmax patch of each of the neuron's K = round(1% of frames) highest frames (one patch per frame,
                    so a single large blob cannot fill the set)
    (a) conc        1 - H(top) / H_ref(K): H = entropy of the top patches' positions on an 8 x 8 grid of 4 x 4-patch
                    cells; H_ref(K) = mean entropy of K foreground patches drawn uniformly (200 draws), same K, so the
                    small-sample bias cancels. 0 = spread like the foreground, 1 = one cell.
    (b) off         fraction of top patches with NO site blob (not on / next to a dark animal: arena or object)
    (c) env_zone    ants: fraction on the dish EDGE; mice: fraction in BAG or OUTSIDE
    count           animal-count distribution at the on-animal top patches; p1 = share with count 1, p2 = share >= 2,
                    median; ants also n_tracked = raw dots + focal centroid within 2 patches of the patch centre
    collective      Spearman rho(frame value, dispersion) over all frames; dispersion = ants: mean pairwise distance of
                    the 3 tracked body centroids; mice: spatial sd of the foreground mask = sqrt(var row + var col) of
                    the frame's foreground patch positions. Also rho with the foreground patch count and the partial
                    rho(value, dispersion | count) (partial correlation of the ranks).
    identity (ants) top patches in frames where BOTH raw dots are detected; anchor = nearest of {raw yellow dot, raw blue
                    dot, focal body centroid}; nearest distance > 3 patches -> 'neither'; focal also counts as 'neither'.
                    shares yellow / blue / neither, selectivity = max share - 1/3. IDENTITY NEURON iff n >= 50 and
                    (share_yellow or share_blue) >= 0.6 and that share >= 2 x its share among all foreground patches.
                    Mice: no identity cue visible at 512 px -> skipped.
    off_dist / far  descriptive (not used by the classes): median distance (patches) of the off-animal top patches to
                    the nearest animal site patch, the share of top patches > 3 patches from any animal, and for the
                    off-animal top patches the count of the NEAREST animal (off_near_p1 = 1 animal, off_near_p2 = >= 2)
    video flag      top-video share = largest fraction of top patches from one video (> 0.5 flagged 'video-specific')
Level class (fixed rules, applied in this order)
    environment     off > 0.5, OR (conc > 0.3 AND env_zone > 0.5 AND off > 0.25)
    collective      |partial rho(value, dispersion | fg count)| > 0.4
    individual      p1 >= 0.6
    pair-group      p2 >= 0.6
    mixed           otherwise

Validation (only after the map is built): where the annotated-behaviour neurons and the controls land
    ants antsfg     90 (groom_any / Y2F / B2F cross-fitted pick), on-lid-yellow picks 246, 786, 860, 374 (a top-1% pick),
                    74 (recording-day neuron)
    mice fg448al    611 (nose-nose), 414 (nose-tail), 49 (odour bag, ENVIRONMENT positive control), 123 (NES round-1 pick)
    mice w4096cell  351 (nose-nose and nose-tail AUROC pick on both halves), 529, 535 (nose-tail AP / top-1% picks)
    Expected: grooming -> pair-group; on-lid -> individual (environment: the lid); bag 49 -> environment.

Steps (scripts/eci/neuron_map.sh)
    encode   GPU   mice: stage the eval tokens, CellMeans (raw store, all eval frames), encode the subset with both mice
                   SAEs -> sparse patch codes (CSR) + checks against codes_best / codes_max.npy
    prep     CPU   both domains: frame tables, dark-cue counts, dark blobs and site maps, ants dish circles
    analyse  CPU   per-neuron statistics, classes, figures, contact sheets -> results/vision/eci_neuron_map/{ants,mice}/
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts/eci'))

OUT = REPO / 'results/vision/eci_neuron_map'
ANTS_PC = REPO / 'results/vision/eci_repr_diag/ants/patch_codes'
ANTS_ANN = REPO / 'dataset/ants/eci/annotations.csv'
ANTS_BG = REPO / 'dataset/ants/eci/fg448/background'
ANTS_SAE = REPO / 'dataset/ants/eci/sae/matryoshka_btk_1024_k16_antsfg_s0/sae.pt'
MICE_ANN = REPO / 'dataset/mice/v1/annotations.csv'
MICE_BG = REPO / 'dataset/mice/v1/eci/fg448/background'
LEV = REPO / 'results/vision/eci_sae_levers/mice'
REGIONS = REPO / 'results/vision/eci_repr_diag/mice/switch_regions/prep/regions.npz'
ODOR = REPO / 'dataset/mice/v1/eci/odor_corner.csv'
STORES = {'raw': 'dataset/mice/v1/eci/train_tokens/dinov2_base_l-1_fg448_fps1',
          'al': 'dataset/mice/v1/eci/train_tokens/dinov2_base_l-1_fg448al_fps1'}
DEP_MAX = REPO / 'dataset/mice/v1/eci/codes/matryoshka_btk_1024_k16_fg448al_s0/codes_max.npy'
MICE_DICTS = {  # key -> (store, checkpoint, centring, n latents)
    'w4096cell': ('raw', LEV / 'sae/w4096+cell_s0/sae.pt', 'cell', 4096),
    'fg448al': ('al', REPO / 'dataset/mice/v1/eci/sae/matryoshka_btk_1024_k16_fg448al_s0/sae.pt', None, 1024)}
DICTS = {'antsfg': 'ants', 'fg448al': 'mice', 'w4096cell': 'mice'}
MARKED = {'antsfg': {90: 'groom_any/Y2F/B2F', 246: 'onlid_yellow', 786: 'onlid_yellow', 860: 'onlid_yellow (AP)',
                     374: 'groom top1 pick', 74: 'recording day'},
          'fg448al': {611: 'nose_nose', 414: 'nose_tail', 49: 'BAG control', 123: 'NES round-1'},
          'w4096cell': {351: 'nose_nose + nose_tail', 529: 'nose_tail (top1)', 535: 'nose_tail (AP)'}}
N_PER_VIDEO = {'ants': 150, 'mice': 210}
BLOB = {'ants': dict(dark_abs=60, dark_rel=40, open_r=1, min_area=40),
        'mice': dict(dark_abs=60, dark_rel=40, open_r=2, min_area=150)}
A1 = {'ants': 302.0, 'mice': 1985.5}
ODOR_ROT90 = {'TR': 0, 'BR': 1, 'BL': 2, 'TL': 3}
GRID, PX = 32, 16
MAX_BLOBS = 64
# fixed thresholds (module docstring)
MIN_FIRE = 0.01
TOP_FRAC = 0.01
ENV_OFF, ENV_CONC, ENV_ZONE, ENV_OFF_MIN = 0.5, 0.3, 0.5, 0.25
COLL_RHO = 0.4
P_DOM = 0.6
ID_SHARE, ID_LIFT, ID_MIN_N, ID_FAR = 0.6, 2.0, 50, 3.0
EDGE_BAND = 32.0  # px
FAR_PATCHES = 3.0  # descriptive: a top patch > 3 patches from any animal site is 'far' (true arena / object)
DISH_R_MIN = 205.0  # px: a tracked-extent radius below this falls back to the median circle of the other videos
CLASSES = ('environment', 'collective', 'individual', 'pair-group', 'mixed')


def log(m):
    print(time.strftime('%H:%M:%S'), m, flush=True)


# ---------------------------------------------------------------------------------------------- frame subsets
def even_pick(n, k):
    return np.unique(np.linspace(0, n - 1, k).round().astype(np.int64))


def ants_subset():
    """-> sorted store-frame indices of the ants patch-code store (150 per video)."""
    rows = np.load(ANTS_PC / 'frame_rows.npy')
    obs = pd.read_csv(ANTS_ANN, usecols=['observation_id'])['observation_id'].values[rows]
    sel = [ix[even_pick(len(ix), N_PER_VIDEO['ants'])] for ix in pd.Series(np.arange(len(obs))).groupby(obs).groups.values()]
    sel = np.sort(np.concatenate([np.asarray(s) for s in sel]))
    assert len(sel) == 256 * N_PER_VIDEO['ants'], len(sel)
    return sel


def mice_subset(lab):
    """lab = sae_levers labels.parquet (172,800 eval frames, store order) -> sorted positions into lab (210 per video)."""
    sel = [np.asarray(ix)[even_pick(len(ix), N_PER_VIDEO['mice'])] for ix in lab.groupby('obs').indices.values()]
    sel = np.sort(np.concatenate(sel))
    assert len(sel) == 144 * N_PER_VIDEO['mice'], len(sel)
    return sel


def video_index(obs):
    vids = sorted(set(obs))
    vmap = {v: i for i, v in enumerate(vids)}
    return vids, np.array([vmap[v] for v in obs], np.int64)


# ---------------------------------------------------------------------------------------------- encode (GPU, mice)
def cmd_encode(args):
    import torch

    import multiscale_sae as ms
    import spatial_sae_pilot as ssp
    from src.eci.levers import CellMeans
    from src.eci.sae import load_sae
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cuda.matmul.allow_tf32 = False
    loc = Path(args.stage)
    t0 = time.time()
    lab = pd.read_parquet(LEV / 'labels.parquet')
    sub = mice_subset(lab)
    vids, fvid = video_index(lab.obs.values)
    checks = {'n_subset_frames': int(len(sub))}
    for key in ('w4096cell', 'fg448al'):
        store, ckpt, mode, M = MICE_DICTS[key]
        ssp.STORES['mice'] = STORES[store]
        idx = ssp.StoreIndex('mice')
        _, ev, _, _, _, _ = idx.split('mice', ms.N_TRAIN['mice'], 0)
        ev = np.sort(ev)
        assert len(ev) == len(lab) and (idx.rows[ev] == lab.row.values).all(), f'{store}: eval frames differ from labels'
        cm = None
        if mode == 'cell':  # CellMeans need every eval token (as sae_levers)
            tok, pos, lens = ms.stage(idx, ev, loc / 'tok')
            tvid = np.repeat(fvid, lens)
            cm = CellMeans(len(vids), tok.shape[1], dev)
            for a in range(0, len(tok), 2_000_000):
                b = min(len(tok), a + 2_000_000)
                cm.add(torch.from_numpy(np.ascontiguousarray(tok[a:b])).to(dev), torch.from_numpy(tvid[a:b]).to(dev),
                       torch.from_numpy(pos[a:b].astype(np.int64)).to(dev))
            cm.finalize(20)
            ref = json.loads((LEV / 'eval_meta_w4096+cell.json').read_text())['variance']['cell_means']
            checks['cell_means'] = {'mine': cm.stats, 'sae_levers': ref, 'equal': cm.stats == ref}
            log(f'{key}: CellMeans {cm.stats} equal to sae_levers: {cm.stats == ref} ({time.time() - t0:.0f}s)')
            st = np.r_[0, np.cumsum(lens)]
            fr = sub
        else:
            tok, pos, lens = ms.stage(idx, ev[sub], loc / 'tok')
            st = np.r_[0, np.cumsum(lens)]
            fr = np.arange(len(sub))
        sae, norm, ck = load_sae(ckpt, dev)
        assert sae.n_latents == M
        log(f'{key}: SAE {ckpt} threshold {float(sae.threshold):.4f}, align {ck.get("align")}')
        indptr, I, V, P, L = [np.zeros(1, np.int64)], [], [], [], []
        tot = 0
        fmax = np.zeros((len(sub), M), np.float32)
        CH = 600
        with torch.no_grad():
            for c0 in range(0, len(fr), CH):
                ff = fr[c0:c0 + CH]
                sl = [np.arange(st[f], st[f + 1]) for f in ff]
                ti = np.concatenate(sl)
                ln = np.array([len(s) for s in sl])
                T = torch.from_numpy(np.ascontiguousarray(tok[ti] if mode else tok[ti[0]:ti[-1] + 1])).to(dev)
                if mode:
                    T = cm.center(T, torch.from_numpy(np.repeat(fvid[sub[c0:c0 + CH]], ln)).to(dev),
                                  torch.from_numpy(pos[ti].astype(np.int64)).to(dev), mode)
                z = sae.encode(norm(T.float()), mode='threshold')
                fid = torch.from_numpy(np.repeat(np.arange(len(ff)), ln)).to(dev)
                fm = torch.zeros(len(ff), M, device=dev)
                fm.index_reduce_(0, fid, z, 'amax', include_self=True)
                fmax[c0:c0 + len(ff)] = fm.cpu().numpy()
                nz = torch.nonzero(z)
                cnt = torch.bincount(nz[:, 0], minlength=len(z))
                indptr.append((tot + torch.cumsum(cnt, 0)).cpu().numpy())
                I.append(nz[:, 1].to(torch.int16).cpu().numpy())
                V.append(z[nz[:, 0], nz[:, 1]].half().cpu().numpy())
                P.append(pos[ti].astype(np.int16))
                L.append(ln)
                tot += len(nz)
        od = loc / 'out' / key
        od.mkdir(parents=True, exist_ok=True)
        np.save(od / 'indptr.npy', np.concatenate(indptr))
        np.save(od / 'idx.npy', np.concatenate(I))
        np.save(od / 'val.npy', np.concatenate(V))
        np.save(od / 'pos.npy', np.concatenate(P))
        np.save(od / 'lens.npy', np.concatenate(L).astype(np.int32))
        np.save(od / 'subset.npy', sub)
        n_tok = int(np.concatenate(L).sum())
        ch = {'n_tokens': n_tok, 'nnz': int(tot), 'l0_per_token': tot / max(n_tok, 1)}
        # check the per-frame max against the stored codes
        if mode == 'cell':
            zb = np.load(LEV / 'codes_best' / 'w4096+cell_s0.npz')
            for c, j in enumerate(zb['neurons']):
                r = zb['codes'][sub, c].astype(np.float32)
                ch[f'n{int(j)}_vs_codes_best'] = {'max_abs_diff': float(np.abs(fmax[:, j] - r).max()),
                                                  'pearson': float(np.corrcoef(fmax[:, j], r)[0, 1]),
                                                  'firing_agree': float(((fmax[:, j] > 0) == (r > 0)).mean())}
        else:
            shutil.copyfile(DEP_MAX, loc / 'codes_max.npy')
            X = np.load(loc / 'codes_max.npy', mmap_mode='r')
            ref = np.asarray(X[lab.row.values[sub]], np.float32)
            (loc / 'codes_max.npy').unlink()
            ch['all_latents_vs_codes_max_5fps'] = {
                'frac_frame_latent_firing_agrees': float(((fmax > 0) == (ref > 0)).mean()),
                'frac_within_1pct': float((np.abs(fmax - ref) <= 0.01 * np.abs(ref) + 1e-3).mean()),
                'pearson_n611_n414_n49': [float(np.corrcoef(fmax[:, j], ref[:, j])[0, 1]) for j in (611, 414, 49)]}
        checks[key] = ch
        log(f'{key}: {json.dumps(ch)} ({time.time() - t0:.0f}s)')
        del tok
        shutil.rmtree(loc / 'tok')
    dst = OUT / 'mice' / 'codes'
    dst.mkdir(parents=True, exist_ok=True)
    (loc / 'out' / 'encode_checks.json').write_text(json.dumps(checks, indent=1, default=str))
    shutil.copytree(loc / 'out', dst, dirs_exist_ok=True)
    log(f'encode done -> {dst} ({time.time() - t0:.0f}s)')


# ---------------------------------------------------------------------------------------------- prep (CPU)
def _grey(path):
    from PIL import Image
    with Image.open(path) as im:
        return np.asarray(im.convert('RGB').convert('L'), dtype=np.uint8)


def site_frame(grey, pix_bg, p):
    """-> dark-cue count per patch (1024,) uint8, site blob id per patch (1024,) int8 (0 = none), blob areas (n,)."""
    from scipy import ndimage as ndi

    from split_test_common import blob_labels, dark_clean
    g = grey.astype(np.int16)
    d = (g < p['dark_abs']) & ((pix_bg.astype(np.int16) - g) > p['dark_rel'])
    dcnt = np.minimum(d.reshape(GRID, PX, GRID, PX).sum((1, 3)), 255).astype(np.uint8).ravel()
    lab, areas = blob_labels(dark_clean(grey, pix_bg, p), p)
    n = len(areas)
    site = np.zeros(GRID * GRID, np.int8)
    if n == 0:
        return dcnt, site, areas
    if n > MAX_BLOBS:  # keep the largest MAX_BLOBS blobs (never reached in practice; logged by the caller)
        keep = np.argsort(-areas)[:MAX_BLOBS] + 1
        remap = np.zeros(n + 1, np.int32)
        remap[keep] = np.arange(1, MAX_BLOBS + 1)
        lab, areas, n = remap[lab], areas[keep - 1], MAX_BLOBS
    Lp = lab.reshape(GRID, PX, GRID, PX).transpose(0, 2, 1, 3).reshape(GRID * GRID, PX * PX)
    has = (Lp > 0).any(1)
    if has.any():
        cnt = np.stack([(Lp[has] == k).sum(1) for k in range(1, n + 1)], 1)
        site[has] = cnt.argmax(1) + 1
    dist, ind = ndi.distance_transform_edt(lab == 0, return_indices=True)
    c = np.arange(GRID) * PX + PX // 2
    cy, cx = np.meshgrid(c, c, indexing='ij')
    near = lab[ind[0][cy, cx], ind[1][cy, cx]].ravel()
    ok = (~has) & (dist[cy, cx].ravel() <= PX)
    site[ok] = near[ok]
    return dcnt, site, areas


def _prep_video(job):
    domain, obs, bgpath, paths = job
    pb = np.load(bgpath)['pix_bg']
    p = BLOB[domain]
    D, S, A, NB = [], [], [], []
    for pth in paths:
        dc, si, ar = site_frame(_grey(REPO / 'dataset' / pth), pb, p)
        D.append(dc); S.append(si)
        a = np.zeros(MAX_BLOBS, np.int32)
        a[:min(len(ar), MAX_BLOBS)] = ar[:MAX_BLOBS]
        A.append(a); NB.append(len(ar))
    return obs, np.stack(D), np.stack(S), np.stack(A), np.array(NB, np.int32)


def frames_table(domain):
    """Frame table of the domain's subset, in subset (store) order."""
    if domain == 'ants':
        sel = ants_subset()
        rows = np.load(ANTS_PC / 'frame_rows.npy')[sel]
        ann = pd.read_csv(ANTS_ANN, usecols=['observation_id', 'frame_idx', 'frame_path', 'experiment', 'T', 'Y_Y2F',
                                             'Y_B2F', 'Y_YOL', 'Y_BOL', 'Y_FOL'])
        f = ann.iloc[rows].reset_index(drop=True).rename(columns={'observation_id': 'obs'})
        f.insert(0, 'store_frame', sel)
        f.insert(1, 'row', rows)
        return f
    lab = pd.read_parquet(LEV / 'labels.parquet')
    sel = mice_subset(lab)
    f = lab.iloc[sel].reset_index(drop=True)
    f.insert(0, 'lab_pos', sel)
    ann = pd.read_csv(MICE_ANN, usecols=['frame_path'])
    f['frame_path'] = ann.frame_path.values[f.row.values]
    oc = pd.read_csv(ODOR).set_index('observation_id')
    f['rot'] = [ODOR_ROT90[c] for c in oc.loc[f.obs.values, 'odor_corner']]
    return f


def ants_tracking(f):
    cols = ['blue_x', 'blue_y', 'yellow_x', 'yellow_y', 'focal_x', 'focal_y', 'raw_blue_x', 'raw_blue_y',
            'raw_yellow_x', 'raw_yellow_y', 'n_blobs']
    out, circ = [], []
    for (obs, ex), g in f.groupby(['obs', 'experiment'], sort=False):
        t = pd.read_csv(REPO / f'dataset/ants/{ex}/tracking/{obs}.csv').set_index('frame_idx')
        out.append(t.reindex(g.frame_idx.values)[cols].set_index(g.index))
        xs = np.concatenate([t[c].values for c in ('blue_x', 'yellow_x', 'focal_x', 'raw_blue_x', 'raw_yellow_x')])
        ys = np.concatenate([t[c].values for c in ('blue_y', 'yellow_y', 'focal_y', 'raw_blue_y', 'raw_yellow_y')])
        ok = np.isfinite(xs) & np.isfinite(ys)
        xs, ys = xs[ok], ys[ok]
        cx = 0.5 * (np.quantile(xs, 0.005) + np.quantile(xs, 0.995))
        cy = 0.5 * (np.quantile(ys, 0.005) + np.quantile(ys, 0.995))
        R = float(np.quantile(np.hypot(xs - cx, ys - cy), 0.995))
        circ.append({'obs': obs, 'cx': cx, 'cy': cy, 'R': R})
    return pd.concat(out).loc[f.index], pd.DataFrame(circ)


def cmd_prep(args):
    from multiprocessing import get_context
    t0 = time.time()
    for domain in args.domains.split(','):
        out = OUT / domain
        out.mkdir(parents=True, exist_ok=True)
        f = frames_table(domain)
        log(f'{domain}: {len(f):,} frames, {f.obs.nunique()} videos ({time.time() - t0:.0f}s)')
        bgd = ANTS_BG if domain == 'ants' else MICE_BG
        groups = f.groupby('obs', sort=False).indices
        jobs = [(domain, o, bgd / f'{o}.npz', f.frame_path.values[ix]) for o, ix in groups.items()]
        D = np.zeros((len(f), GRID * GRID), np.uint8)
        S = np.zeros((len(f), GRID * GRID), np.int8)
        A = np.zeros((len(f), MAX_BLOBS), np.int32)
        NB = np.zeros(len(f), np.int32)
        with get_context('fork').Pool(args.workers) as pool:
            for k, (o, d, s, a, nb) in enumerate(pool.imap_unordered(_prep_video, jobs)):
                ix = groups[o]
                D[ix], S[ix], A[ix], NB[ix] = d, s, a, nb
                if k % 40 == 0:
                    log(f'  {domain} blobs {k + 1}/{len(jobs)} videos ({time.time() - t0:.0f}s)')
        meta = {'n_frames': len(f), 'frames_with_more_than_max_blobs': int((NB > MAX_BLOBS).sum()),
                'n_blobs_hist': np.bincount(np.minimum(NB, 10), minlength=11).tolist(),
                'blob_params': BLOB[domain], 'A1_dark_px': A1[domain]}
        # single-animal area check: frames with exactly N blobs, median area / A1
        N = 3 if domain == 'ants' else 4
        m = NB == N
        meta['median_blob_area_in_N_blob_frames_over_A1'] = float(np.median(A[m][:, :N]) / A1[domain]) if m.any() else None
        meta['frac_frames_with_N_blobs'] = float(m.mean())
        if domain == 'ants':
            tr, circ = ants_tracking(f)
            for c in tr.columns:
                f[c] = tr[c].values
            circ.to_csv(out / 'dish_circles.csv', index=False)
            meta['dish_R_px'] = [float(circ.R.min()), float(circ.R.median()), float(circ.R.max())]
            sanity_dish(f, circ, out)
        f.to_parquet(out / 'frames.parquet')
        np.savez_compressed(out / 'blobs.npz', dark=D, site=S, areas=A, n_blobs=NB)
        (out / 'prep.json').write_text(json.dumps(meta, indent=1))
        log(f'{domain}: prep {json.dumps(meta)} ({time.time() - t0:.0f}s)')


def sanity_dish(f, circ, out):
    from PIL import Image, ImageDraw
    pick = circ.iloc[np.linspace(0, len(circ) - 1, 6).round().astype(int)]
    sheet = Image.new('RGB', (6 * 256, 256), 'white')
    for i, (_, r) in enumerate(pick.iterrows()):
        pth = f.frame_path[f.obs == r.obs].values[0]
        im = Image.open(REPO / 'dataset' / pth).convert('RGB').resize((512, 512))
        dr = ImageDraw.Draw(im)
        for rad, col in ((r.R, (255, 0, 0)), (r.R - EDGE_BAND, (0, 120, 255))):
            dr.ellipse([r.cx - rad, r.cy - rad, r.cx + rad, r.cy + rad], outline=col, width=3)
        sheet.paste(im.resize((256, 256)), (i * 256, 0))
        ImageDraw.Draw(sheet).text((i * 256 + 4, 2), f'{r.obs} R={r.R:.0f}', fill=(255, 0, 0))
    sheet.save(out / 'sanity_dish.jpg', quality=88)


# ---------------------------------------------------------------------------------------------- analyse (CPU)
class Codes:
    """Sparse patch codes of one dictionary on its domain's subset frames: CSR + per-token frame / SAE position /
    raw-frame position."""

    def __init__(self, key, f, stage):
        dom = DICTS[key]
        if dom == 'ants':
            src = Path(stage) / 'ants_pc' if stage else ANTS_PC
            nfg = np.load(src / 'frame_nfg.npy')
            sel = f.store_frame.values
            infr = np.zeros(len(nfg), bool)
            infr[sel] = True
            ind_all = np.load(src / 'indptr.npy')
            tmask = np.repeat(infr, nfg)
            cnt = np.diff(ind_all)
            nmask = np.repeat(tmask, cnt)
            self.idx = np.load(src / 'idx.npy', mmap_mode='r')[nmask]
            self.val = np.load(src / 'val.npy', mmap_mode='r')[nmask].astype(np.float32)
            self.indptr = np.r_[0, np.cumsum(cnt[tmask])]
            self.pos = np.load(src / 'pos.npy')[tmask].astype(np.int64)
            self.lens = nfg[sel].astype(np.int64)
            self.M = 1024
            self.praw = self.pos
        else:
            src = OUT / 'mice' / 'codes' / key
            assert (np.load(src / 'subset.npy') == f.lab_pos.values).all()
            self.indptr = np.load(src / 'indptr.npy')
            self.idx = np.load(src / 'idx.npy')
            self.val = np.load(src / 'val.npy').astype(np.float32)
            self.pos = np.load(src / 'pos.npy').astype(np.int64)
            self.lens = np.load(src / 'lens.npy').astype(np.int64)
            self.M = MICE_DICTS[key][3]
            self.praw = self.pos.copy()
            if key == 'fg448al':  # aligned-frame position -> raw-frame position (inverse of np.rot90(raw, k))
                g = np.arange(GRID * GRID).reshape(GRID, GRID)
                lut = np.stack([np.rot90(g, k).ravel() for k in range(4)])
                rot = np.repeat(f.rot.values, self.lens)
                self.praw = lut[rot, self.pos]
        self.F = len(f)
        assert self.lens.sum() == len(self.pos) == len(self.indptr) - 1
        self.fid = np.repeat(np.arange(self.F), self.lens)
        self.ptok = np.repeat(np.arange(len(self.pos)), np.diff(self.indptr))

    def frame_max(self):
        """-> FM (M, F) float32 frame max per latent, AT (M, F) int64 argmax token (-1 = not firing)."""
        key = self.idx.astype(np.int64) * self.F + self.fid[self.ptok]
        o = np.lexsort((-self.val, key))
        ks = key[o]
        first = np.r_[True, ks[1:] != ks[:-1]]
        s = o[first]
        FM = np.zeros(self.M * self.F, np.float32)
        AT = np.full(self.M * self.F, -1, np.int64)
        FM[key[s]] = self.val[s]
        AT[key[s]] = self.ptok[s]
        return FM.reshape(self.M, self.F), AT.reshape(self.M, self.F)


def entropy_cells(pos):
    cell = (pos // GRID) // 4 * 8 + (pos % GRID) // 4
    p = np.bincount(cell, minlength=64) / len(cell)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def rank_rows(X):
    from scipy.stats import rankdata
    R = rankdata(X, axis=1).astype(np.float32)
    R -= R.mean(1, keepdims=True)
    R /= np.linalg.norm(R, axis=1, keepdims=True).clip(1e-9)
    return R


def rank_vec(x):
    from scipy.stats import rankdata
    r = rankdata(x).astype(np.float32)
    r -= r.mean()
    return r / max(np.linalg.norm(r), 1e-9)


def anchors_px(f):
    """ants: (F, 3, 2) raw yellow dot, raw blue dot, focal body centroid in patch units (NaN where missing)."""
    return np.stack([np.c_[f.raw_yellow_x, f.raw_yellow_y], np.c_[f.raw_blue_x, f.raw_blue_y],
                     np.c_[f.focal_x, f.focal_y]], 1) / PX


def animal_distance(site):
    """(F, 1024) site map -> (F, 1024) float16 distance (patches, patch centre to patch centre) to the nearest patch
    with a site blob (0 on such patches; 99 when the frame has no animal at all). Descriptive only."""
    from scipy import ndimage as ndi
    out = np.full(site.shape, 99, np.float16)
    near = np.full(site.shape, -1, np.int16)  # nearest patch with a site blob (row-major), -1 = none in the frame
    for i in range(len(site)):
        m = site[i].reshape(GRID, GRID) == 0
        if not m.all():
            d, ind = ndi.distance_transform_edt(m, return_indices=True)
            out[i] = d.ravel()
            near[i] = (ind[0] * GRID + ind[1]).ravel()
    return out, near


def site_stats(dom, fr, praw, B, f, ctx):
    """Statistics of a set of (frame, raw position) sites -> dict (shared by top patches and the baselines)."""
    site = B['site'][fr, praw].astype(np.int64)
    area = np.where(site > 0, B['areas'][fr, np.maximum(site - 1, 0)], 0)
    count = np.where(site > 0, np.clip(np.round(area / A1[dom]), 1, 4), 0).astype(int)
    on = count > 0
    r = {'n': int(len(fr)), 'off': float((~on).mean()),
         'count_hist': [float((count[on] == c).mean()) if on.any() else 0.0 for c in (1, 2, 3, 4)],
         'median_count': float(np.median(count[on])) if on.any() else 0.0,
         'mean_count': float(count[on].mean()) if on.any() else 0.0,
         'dark_core': float((B['dark'][fr, praw] >= 39).mean())}
    ad = B['adist'][fr, praw].astype(np.float32)
    r['off_dist_median'] = float(np.median(ad[~on])) if (~on).any() else 0.0
    r['far'] = float((ad > FAR_PATCHES).mean())
    # off-animal top patches: animal count of the NEAREST animal site (does the halo surround one animal or a group)
    nb = B['anear'][fr, praw].astype(np.int64)
    k = (~on) & (nb >= 0)
    ns = B['site'][fr[k], nb[k]].astype(np.int64)
    nc = np.clip(np.round(B['areas'][fr[k], np.maximum(ns - 1, 0)] / A1[dom]), 1, 4)
    r['off_near_p1'] = float((nc == 1).mean()) if k.any() else float('nan')
    r['off_near_p2'] = float((nc >= 2).mean()) if k.any() else float('nan')
    r['p1'] = r['count_hist'][0]
    r['p2'] = float(sum(r['count_hist'][1:]))
    pyx = np.c_[praw // GRID + 0.5, praw % GRID + 0.5]  # (row, col) patch centre, patch units
    if dom == 'mice':
        reg = ctx['reg'][f.vid.values[fr], praw]
        for k, nme in enumerate(('BAG', 'WALL', 'CENTRE', 'OUTSIDE')):
            r[f'frac_{nme}'] = float((reg == k).mean())
        r['env_zone'] = r['frac_BAG'] + r['frac_OUTSIDE']
    else:
        c = ctx['circ']
        d = np.hypot(pyx[:, 1] * PX - c[fr, 0], pyx[:, 0] * PX - c[fr, 1])
        r['env_zone'] = float((d > c[fr, 2] - EDGE_BAND).mean())
        r['frac_outside_dish'] = float((d > c[fr, 2]).mean())
        an = ctx['anchors'][fr]  # (n, 3, 2) x, y
        dd = np.hypot(an[..., 0] - pyx[:, 1:2], an[..., 1] - pyx[:, 0:1])  # (n, 3)
        ntr = (np.nan_to_num(dd, nan=1e9) <= 2.0).sum(1)
        r['n_tracked_hist'] = [float((ntr == c).mean()) for c in (0, 1, 2, 3)]
        r['median_n_tracked'] = float(np.median(ntr))
        both = np.isfinite(dd[:, 0]) & np.isfinite(dd[:, 1]) & np.isfinite(dd[:, 2])
        r['id_n'] = int(both.sum())
        if both.any():
            nn = np.argmin(dd[both], 1)
            far = dd[both].min(1) > ID_FAR
            cls = np.where(far, 2, np.where(nn == 2, 2, nn))
            r['share_yellow'], r['share_blue'], r['share_neither'] = (float((cls == k).mean()) for k in range(3))
            r['id_selectivity'] = max(r['share_yellow'], r['share_blue'], r['share_neither']) - 1 / 3
        else:
            r['share_yellow'] = r['share_blue'] = r['share_neither'] = r['id_selectivity'] = float('nan')
    return r


def classify(r):
    if r['off'] > ENV_OFF or (r['conc'] > ENV_CONC and r['env_zone'] > ENV_ZONE and r['off'] > ENV_OFF_MIN):
        return 'environment'
    if abs(r['rho_disp_partial']) > COLL_RHO:
        return 'collective'
    if r['p1'] >= P_DOM:
        return 'individual'
    if r['p2'] >= P_DOM:
        return 'pair-group'
    return 'mixed'


def label_auroc(x, y):
    from scipy.stats import rankdata
    y = np.asarray(y, bool)
    if y.all() or not y.any():
        return None
    R = rankdata(x)
    P, N = y.sum(), (~y).sum()
    return round(float((R[y].sum() - P * (P + 1) / 2) / (P * N)), 4)


def cmd_analyse(args):
    t0 = time.time()
    rng = np.random.default_rng(0)
    allres = {}
    for key in args.dicts.split(','):
        dom = DICTS[key]
        out = OUT / dom
        f = pd.read_parquet(out / 'frames.parquet')
        z = np.load(out / 'blobs.npz')
        B = {k: z[k] for k in ('dark', 'site', 'areas')}
        B['adist'], B['anear'] = animal_distance(B['site'])
        C = Codes(key, f, args.stage if dom == 'ants' else None)
        F = C.F
        log(f'[{key}] {F:,} frames, {len(C.pos):,} tokens, {len(C.idx):,} nonzeros ({time.time() - t0:.0f}s)')
        ctx = {}
        res = {'dict': key, 'domain': dom, 'n_frames': F, 'n_tokens': int(len(C.pos)), 'n_videos': int(f.obs.nunique())}
        # geometry check: foreground tokens (raw frame) should sit on dark-cue patches far more than chance
        on_dark = float((B['dark'][C.fid, C.praw] > 0).mean())
        res['geometry_check'] = {'frac_tokens_with_dark_pixels_raw_pos': on_dark}
        if key == 'fg448al':
            res['geometry_check']['frac_tokens_with_dark_pixels_unrotated_pos'] = float((B['dark'][C.fid, C.pos] > 0).mean())
        if dom == 'mice':
            rz = np.load(REGIONS)
            vmap = {o: i for i, o in enumerate(rz['obs'])}
            f['vid'] = [vmap[o] for o in f.obs]
            ctx['reg'] = rz['reg_m2']
            r_ = C.pos // GRID if key != 'fg448al' else C.praw // GRID
            c_ = C.pos % GRID if key != 'fg448al' else C.praw % GRID
            disp = np.full(F, np.nan)
            st = np.r_[0, np.cumsum(C.lens)]
            for i in range(F):
                if C.lens[i] >= 2:
                    disp[i] = np.sqrt(r_[st[i]:st[i + 1]].var() + c_[st[i]:st[i + 1]].var())
        else:
            circ = pd.read_csv(out / 'dish_circles.csv').set_index('obs')
            bad = circ.R < DISH_R_MIN  # ants stayed in one part of the dish: tracked extent < the dish
            circ.loc[bad, ['cx', 'cy', 'R']] = circ.loc[~bad, ['cx', 'cy', 'R']].median().values
            res['dish_circle_fallback_videos'] = int(bad.sum())
            ctx['circ'] = circ.loc[f.obs.values, ['cx', 'cy', 'R']].values
            ctx['anchors'] = anchors_px(f)
            bc = np.stack([np.c_[f.yellow_x, f.yellow_y], np.c_[f.blue_x, f.blue_y], np.c_[f.focal_x, f.focal_y]], 1) / PX
            disp = np.mean([np.hypot(*(bc[:, a] - bc[:, b]).T) for a, b in ((0, 1), (0, 2), (1, 2))], 0)
        fgc = C.lens.astype(np.float64)
        okd = np.isfinite(disp)
        res['frames_with_dispersion'] = int(okd.sum())
        rd, rc = rank_vec(disp[okd]), rank_vec(fgc[okd])
        res['rho_dispersion_vs_fgcount'] = float(rd @ rc)
        # baseline: uniform foreground patches
        bs = rng.choice(len(C.pos), 20000, replace=False)
        base = site_stats(dom, C.fid[bs], C.praw[bs], B, f, ctx)
        res['baseline_all_fg_patches'] = base
        log(f'[{key}] geometry {res["geometry_check"]}; baseline fg patches: ' + json.dumps(
            {k: (round(v, 3) if isinstance(v, float) else v) for k, v in base.items()}))
        # frame max
        FM, AT = C.frame_max()
        fire = (FM > 0).mean(1)
        elig = np.flatnonzero(fire >= MIN_FIRE)
        K = int(round(TOP_FRAC * F))
        Href = float(np.mean([entropy_cells(C.pos[rng.choice(len(C.pos), K, replace=False)]) for _ in range(200)]))
        res.update({'n_latents': C.M, 'n_eligible': int(len(elig)), 'n_rare': int(((fire > 0) & (fire < MIN_FIRE)).sum()),
                    'n_dead': int((fire == 0).sum()), 'K_top': K, 'H_ref': Href})
        log(f'[{key}] eligible {len(elig)}, rare {res["n_rare"]}, dead {res["n_dead"]}, K {K}, H_ref {Href:.3f} '
            f'({time.time() - t0:.0f}s)')
        R = rank_rows(FM[elig][:, okd])
        rho_d, rho_c = R @ rd, R @ rc
        rdc = res['rho_dispersion_vs_fgcount']
        partial = (rho_d - rho_c * rdc) / np.sqrt(np.clip((1 - rho_c ** 2) * (1 - rdc ** 2), 1e-12, None))
        del R
        rows = []
        for n_, j in enumerate(elig):
            v = FM[j]
            top = np.argpartition(-v, K - 1)[:K]
            top = top[np.argsort(-v[top], kind='stable')]
            top = top[v[top] > 0]
            tk = AT[j, top]
            r = {'neuron': int(j), 'fire_frac': float(fire[j]), 'mean_value': float(v.mean())}
            r.update(site_stats(dom, top, C.praw[tk], B, f, ctx))
            r['conc'] = 1 - entropy_cells(C.pos[tk]) / Href
            r['rho_disp'], r['rho_fgcount'], r['rho_disp_partial'] = float(rho_d[n_]), float(rho_c[n_]), float(partial[n_])
            vc = f.obs.values[top]
            r['n_top_videos'] = int(len(set(vc)))
            r['top_video_share'] = float(pd.Series(vc).value_counts().iloc[0] / len(vc))
            r['class'] = classify(r)
            if dom == 'ants':
                r['identity'] = bool(r['id_n'] >= ID_MIN_N and any(
                    r[f'share_{c}'] >= ID_SHARE and r[f'share_{c}'] >= ID_LIFT * base[f'share_{c}'] for c in ('yellow', 'blue')))
            rows.append(r)
        T = pd.DataFrame(rows)
        for c in ('count_hist', 'n_tracked_hist'):
            if c in T:
                h_ = np.stack(T.pop(c).values)
                T[[f'{c}_{i}' for i in range(h_.shape[1])]] = h_
        T['video_specific'] = T.top_video_share > 0.5
        T.to_csv(out / f'neurons_{key}.csv', index=False)
        rare = pd.DataFrame({'neuron': np.flatnonzero((fire > 0) & (fire < MIN_FIRE)),
                             'fire_frac': fire[(fire > 0) & (fire < MIN_FIRE)]})
        rare.to_csv(out / f'rare_{key}.csv', index=False)
        res['class_counts'] = {c: int((T['class'] == c).sum()) for c in CLASSES}
        res['class_summary'] = {c: {q: round(float(T.loc[T['class'] == c, q].median()), 3) for q in
                                    ('off', 'off_dist_median', 'far', 'off_near_p1', 'off_near_p2', 'conc', 'env_zone', 'p1', 'p2', 'mean_count', 'rho_disp', 'rho_fgcount',
                                     'rho_disp_partial', 'fire_frac', 'top_video_share')}
                                for c in CLASSES if (T['class'] == c).any()}
        res['abs_rho_fgcount_by_class'] = {c: {'median': round(float(T.loc[T['class'] == c, 'rho_fgcount'].abs().median()), 3),
                                               'frac_above_0.4': round(float((T.loc[T['class'] == c, 'rho_fgcount'].abs() > 0.4).mean()), 3)}
                                           for c in CLASSES if (T['class'] == c).any()}
        res['video_specific_by_class'] = {c: int(T.loc[T['class'] == c, 'video_specific'].sum()) for c in CLASSES}
        if dom == 'ants':
            res['identity_neurons'] = T.loc[T.identity, ['neuron', 'class', 'id_n', 'share_yellow', 'share_blue',
                                                         'share_neither', 'id_selectivity', 'fire_frac']].round(3).to_dict('records')
            res['id_selectivity_quantiles'] = {q: round(float(T.id_selectivity.quantile(q)), 3) for q in (0.1, 0.5, 0.9, 0.99)}
        log(f'[{key}] class counts {res["class_counts"]} ({time.time() - t0:.0f}s)')
        # ---- validation: marked neurons (labels are looked at only here)
        labs = (({'Y2F': f.Y_Y2F.fillna(0) > 0, 'B2F': f.Y_B2F.fillna(0) > 0},
                 {c: f[f'Y_{c}'].fillna(0) > 0 for c in ('YOL', 'BOL', 'FOL')}) if dom == 'ants' else
                ({'nose_nose': f.nose_nose.values, 'nose_tail': f.nose_tail.values}, {}))
        v3 = (f.experiment == 'v3').values if dom == 'ants' else None
        mk = {}
        for j, why in MARKED[key].items():
            e = {'why': why, 'fire_frac': float(fire[j])}
            row = T[T.neuron == j]
            if len(row):
                e.update({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.iloc[0].to_dict().items()
                          if k in ('class', 'off', 'off_dist_median', 'far', 'off_near_p1', 'off_near_p2', 'conc', 'env_zone', 'p1', 'p2', 'median_count', 'mean_count',
                                   'rho_disp', 'rho_fgcount', 'rho_disp_partial', 'share_yellow', 'share_blue',
                                   'share_neither', 'id_selectivity', 'identity', 'median_n_tracked', 'frac_BAG',
                                   'frac_OUTSIDE', 'frac_WALL', 'top_video_share', 'n_top_videos')})
            else:
                e['class'] = 'rare' if fire[j] > 0 else 'dead'
            e['auroc'] = {k: label_auroc(FM[j], y) for k, y in labs[0].items()}
            if dom == 'ants':
                e['auroc'].update({f'{k} (v3)': label_auroc(FM[j][v3], y.values[v3]) for k, y in labs[1].items()})
            mk[int(j)] = e
            log(f'[{key}] marked n{j} ({why}): {json.dumps(e)}')
        res['marked'] = mk
        allres[key] = res
        sheets(key, T, f, C, FM, AT, out)
        (out / f'results_{key}.json').write_text(json.dumps(res, indent=1, default=float))
        del FM, AT, C
    figures(args.dicts.split(','))
    log(f'analyse done ({time.time() - t0:.0f}s)')


# ---------------------------------------------------------------------------------------------- figures / sheets
BLUE, RED, GREY = '#2a78d6', '#e34948', '#f0efec'
CLS_COL = {'environment': '#eda100', 'collective': '#4a3aa7', 'individual': '#2a78d6', 'pair-group': '#eb6834',
           'mixed': '#8a8a85'}


def figures(keys):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('div', [BLUE, GREY, RED])
    fig, ax = plt.subplots(1, len(keys), figsize=(7.2 * len(keys), 6.4), squeeze=False)
    for a, key in zip(ax[0], keys):
        dom = DICTS[key]
        T = pd.read_csv(OUT / dom / f'neurons_{key}.csv')
        jit = np.random.default_rng(0).normal(0, 0.02, len(T))
        sc = a.scatter(T.mean_count + jit, T.off, c=T.rho_disp_partial, cmap=cmap, vmin=-0.6, vmax=0.6, s=10,
                       linewidths=0.3, edgecolors='white')
        for i_, (j, why) in enumerate(MARKED[key].items()):
            r = T[T.neuron == j]
            if len(r):
                a.scatter(r.mean_count, r.off, s=90, facecolors='none', edgecolors='black', linewidths=1.5)
                a.annotate(f'n{j} {why}', (r.mean_count.iloc[0], r.off.iloc[0]), xytext=(8, (6 + 11 * (i_ % 3)) * (-1.6 if r.off.iloc[0] > 0.85 else 1)),
                           textcoords='offset points', fontsize=7.5,
                           arrowprops=dict(arrowstyle='-', lw=0.5, color='#555555'))
        a.axhline(ENV_OFF, color='#8a8a85', lw=0.8, ls='--')
        a.set_xlabel('mean animal count at on-animal top patches')
        a.set_ylabel('environment score (b) = share of top patches off any animal')
        cc = pd.Series(T['class']).value_counts().to_dict()
        a.set_title(f'{key} ({dom}): {len(T)} eligible neurons\n' + ', '.join(f'{c} {cc.get(c, 0)}' for c in CLASSES[:3])
                    + '\n' + ', '.join(f'{c} {cc.get(c, 0)}' for c in CLASSES[3:]), fontsize=9)
        a.set_xlim(-0.1, 4.1)
        a.set_ylim(-0.03, 1.03)
        for s in ('top', 'right'):
            a.spines[s].set_visible(False)
    cb = fig.colorbar(sc, ax=ax[0].tolist(), shrink=0.8)
    cb.set_label('collective score: partial Spearman rho(value, dispersion | fg count)')
    fig.savefig(OUT / 'scatter_env_vs_count.png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    if 'antsfg' in keys:
        T = pd.read_csv(OUT / 'ants' / 'neurons_antsfg.csv')
        fig, a = plt.subplots(1, 2, figsize=(11, 3.8))
        a[0].hist(T.id_selectivity.dropna(), bins=40, color=BLUE, edgecolor='white', linewidth=0.5)
        a[0].axvline(ID_SHARE - 1 / 3, color=RED, lw=1, ls='--')
        a[0].set_xlabel('identity selectivity = max(share yellow, blue, neither) - 1/3')
        a[0].set_ylabel('neurons')
        a[1].scatter(T.share_yellow, T.share_blue, s=8, c=np.where(T.identity, RED, '#8a8a85'))
        a[1].set_xlabel('share of top patches nearest the YELLOW dot')
        a[1].set_ylabel('share nearest the BLUE dot')
        a[1].set_title(f'identity neurons (red): {int(T.identity.sum())}', fontsize=9)
        for x in a:
            for s in ('top', 'right'):
                x.spines[s].set_visible(False)
        fig.tight_layout()
        fig.savefig(OUT / 'ants' / 'identity_hist.png', dpi=130)
        plt.close(fig)


def sheets(key, T, f, C, FM, AT, out, n_tiles=12, crop=192):
    """Top-12 patches (one per frame) of 2 example neurons per class (the most typical by the class score, and one random
    member, seed 0), plus the marked neurons and the ants identity neurons."""
    from PIL import Image, ImageDraw
    rng = np.random.default_rng(0)
    score = {'environment': T.off, 'collective': T.rho_disp_partial.abs(), 'individual': T.p1, 'pair-group': T.p2,
             'mixed': -(T.p1 - 0.5).abs()}
    picks = []
    for c in CLASSES:
        m = (T['class'] == c).values
        if not m.any():
            continue
        i1 = int(T.neuron[m].values[np.argmax(score[c][m].values)])
        rest = [n for n in T.neuron[m].values if n != i1]
        picks.append((c, [i1] + ([int(rng.choice(rest))] if rest else [])))
    picks.append(('marked', [j for j in MARKED[key] if (T.neuron == j).any()]))
    if 'identity' in T:
        ids = T[T.identity].sort_values('id_selectivity', ascending=False).neuron.values[:6]
        if len(ids):
            picks.append(('identity', [int(x) for x in ids]))
    sd = out / 'sheets'
    sd.mkdir(exist_ok=True)
    h = 30
    manifest = []
    for grp, neurons in picks:
        sheet = Image.new('RGB', (n_tiles * crop, len(neurons) * (crop + h)), 'white')
        dr = ImageDraw.Draw(sheet)
        for r_, j in enumerate(neurons):
            row = T[T.neuron == j].iloc[0]
            v = FM[j]
            top = np.argsort(-v, kind='stable')[:n_tiles]
            y0 = r_ * (crop + h)
            txt = (f'n{j} [{row["class"]}] fire {row.fire_frac:.3f} off {row.off:.2f} conc {row.conc:.2f} zone '
                   f'{row.env_zone:.2f} p1 {row.p1:.2f} p2 {row.p2:.2f} rho_disp {row.rho_disp:.2f} rho_fg '
                   f'{row.rho_fgcount:.2f} partial {row.rho_disp_partial:.2f}')
            if 'share_yellow' in row:
                txt += f' | Y {row.share_yellow:.2f} B {row.share_blue:.2f} N {row.share_neither:.2f}'
            if j in MARKED[key]:
                txt += f' | {MARKED[key][j]}'
            dr.text((4, y0 + 2), txt, fill=(0, 0, 0))
            for c_, fr in enumerate(top):
                if v[fr] <= 0:
                    continue
                tk = AT[j, fr]
                p = int(C.praw[tk])
                py, px = (p // GRID) * PX, (p % GRID) * PX
                im = Image.open(REPO / 'dataset' / f.frame_path.values[fr]).convert('RGB').resize((512, 512))
                x0 = int(np.clip(px + PX // 2 - crop // 2, 0, 512 - crop))
                yy0 = int(np.clip(py + PX // 2 - crop // 2, 0, 512 - crop))
                ImageDraw.Draw(im).rectangle([px - 1, py - 1, px + PX, py + PX], outline=(255, 0, 0), width=2)
                sheet.paste(im.crop((x0, yy0, x0 + crop, yy0 + crop)), (c_ * crop, y0 + h))
                dr.text((c_ * crop + 3, y0 + 16), f'{f.obs.values[fr]} {v[fr]:.1f}', fill=(80, 80, 80))
                manifest.append({'group': grp, 'neuron': int(j), 'tile': c_, 'obs': f.obs.values[fr],
                                 'frame_idx': int(f.frame_idx.values[fr]), 'value': float(v[fr]), 'raw_patch': p})
        sheet.save(sd / f'{key}_{grp}.jpg', quality=85)
    pd.DataFrame(manifest).to_csv(sd / f'{key}_manifest.csv', index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['encode', 'prep', 'analyse', 'figures'])
    p.add_argument('--stage', default=None)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--domains', default='ants,mice')
    p.add_argument('--dicts', default='antsfg,fg448al,w4096cell')
    a = p.parse_args()
    {'encode': cmd_encode, 'prep': cmd_prep, 'analyse': cmd_analyse,
     'figures': lambda a_: figures(a_.dicts.split(','))}[a.cmd](a)


if __name__ == '__main__':
    main()
