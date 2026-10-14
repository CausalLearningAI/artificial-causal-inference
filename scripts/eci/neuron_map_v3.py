"""
BEHAVIOURAL NEURON MAP v3, MICE ONLY: the v2 map (scripts/eci/neuron_map_v2.py, commits 2ce93eb / d5fc1d0) with three
fixes of the mice instance geometry, diagnosed by the v2 visual hand-check of fg448m5 (correct / wrong / unclear of 15:
self 8/0/7, pair 2/12/1, collective 11/1/3, place-bound 3/2/10, halo 2/10/3, plain background 6/9/0). Analysis only:
the SAE codes, the frames (the 20,000-frame B subset with SAM 2 masks, 16,775 valid by the v2 rule) and the top frames
are exactly v2's (read from results/vision/eci_neuron_map_v2/); nothing changes the representation. Ants: unchanged (v2).

Diagnosed v2 causes -> fixes (all rules below were written before any v3 number or sheet was looked at)
  1 SAM masks leave out TAILS (all 9 wrong plain-background, 4 wrong halo)
    -> TAILS. Per frame, the cue pixels outside every SAM mask (cue = the foreground dark cue of src/eci/foreground.py:
       grey < 60 AND > 40 grey levels below the video's pixel background pix_bg, dataset/mice/v1/eci/fg448/background) are
       split into 8-connected components; a component whose nearest pixel is <= TAIL_R = 4 px (Euclidean) from an
       instance joins THE NEAREST instance (whole component, the instance of its closest pixel); repeated up to 3 rounds
       on the grown masks, so a tail broken into pieces chains on. Components farther away stay unassigned.
       Frames are decoded locally from the standardised 5 fps mp4 (v2's decode matches the dataset JPG within 12 grey
       levels, JPEG noise). Two looser cues are computed alongside as SENSITIVITY variants, because tails are grey, not
       black: 'mid' (grey < 120 AND > 50 below pix_bg) and 'rel' (> 40 below pix_bg only). Pre-registered choice: the
       spec cue 'dark' is used unless the 20-frame tail sheet shows tails left out in most frames; then 'mid', then
       'rel' (the choice is made on the tail sheet alone, before any neuron statistic or sheet of v3 is looked at).
       REVISION (2026-10-14, after the first tail sheet, before any v3 neuron statistic or sheet was looked at): all
       three cues FAILED the tail sheet (visual judgement, 5 frames shown per sheet): 'dark' attached no tail in any
       frame and joined a wall shadow to a mouse; 'mid' still left most tails out and attached wall shadows; 'rel'
       flooded the bedding (198 on-animal patches per frame vs 56 for v2). Tails are thin, grey, start at a body end
       and are about one body long, so a 4th cue 'line' was added (extend_line): thin dark pixels (black top-hat,
       grey closing 9 x 9 minus grey, > 30, AND > 40 below pix_bg), grown 8-connected outside every mask from the cue
       pixels within 6 px of a body end, geodesically up to 120 px; the grown set is a tail iff it reaches >= 20 px with
       mean width (pixels / geodesic length) <= 7 px; of the two ends the longer one is kept (one tail per mouse). The
       choice among the 4 cues is again made on the (re-rendered) tail sheet alone.
       REVISION 2 (2nd tail sheet): the line cue traced the tails well, but most were not attached because the thin
       part starts > 6 px from the body end (the tail base is thick); seed radius 6 -> 20 px, and candidates are
       assigned closest end first (a set > 50% taken by a closer mouse is dropped).
       REVISION 3 (per-end diagnostic on the 20 sheet frames, step taildiag): the width test divided ALL pixels,
       including the up-to-180-px seed disk, by the length, rejecting real tails (e.g. 527 px / 57 steps = 9.2); the
       width is now (grown pixels - seed pixels) / steps. Final tail rule; no further change after this sheet.
       Patch labels: a patch overlaps mouse k iff >= 26 px (10%, as v2) of the extended mask k fall in it, OR >= 10 px
       of k's tail pixels (a tail is 3-5 px wide; a 16 px tail segment gives 50-80 px, a tip clip ~10). 'Tail-only'
       patch = overlaps k only through the second clause or only with the help of tail pixels (< 26 SAM px of k).
  2 Huddle neurons split into separate active regions, so 3-4 mice read as 2 (most wrong 'pair')
    -> ANIMAL COUNT PER CONTACT CLUSTER. Touching instances (extended masks, one dilated by 3 px overlaps the other, as
       v2's huddle stratum) form clusters (union-find). In a top frame the neuron's footprint = ALL its active patches;
       n = the number of distinct instances the footprint overlaps (union over all regions, not per connected region).
       Frame type: n = 0 none, 1 self; n >= 2: all overlapped instances in one touching cluster -> 'pair' (n = 2) or
       'huddle' (n >= 3); otherwise 'apart' (co-active, apart). Neuron: weighted lower median of n over its valid top
       frames (frame weights as v2): 0 none, 1 self; >= 2: 'apart' if the 'apart' frames hold >= 50% of the weight of
       its n >= 2 frames, else 'pair' (median 2) or 'huddle' (median >= 3; v2's 'collective'). Reported, not used:
       pair_in_huddle = share of the 'pair' frames whose touching cluster holds >= 3 mice.
  3 Mice 'place-bound' was mostly huddles resting at walls; 4 object interactions put the active patches ON the object
    next to the head, so they were filed as halo
    -> CAGE-OBJECT MAP + ENVIRONMENT INTERACTION.
       Object map (step objects, label-free, per recording session = pool x odour: its H, O, P videos, one camera
       rotation; raw camera frame). Inputs: the per-video pixel background pix_bg (0.98 quantile of grey over 200
       frames: the empty arena), the floor box (arena_r0..c1 of dataset/mice/v1/eci/odor_corner.csv, per video), the
       spout side and odour corner of that table, the SAM masks of the session's B-subset frames.
         static image   S = pixelwise MAX of the session's pix_bg (a pixel is dark in S only if dark in all 3 videos)
         SAM cover      share of the session's B frames in which the pixel lies in a SAM mask (mice resting there)
         dark static    S < 72 (the spout rule of mice_odor_corner.py) AND SAM cover <= 0.3
         edges          Sobel magnitude of S (gaussian sigma 1); 'edgy' = > 3 x the median magnitude outside the box
         wall           every pixel outside the floor box (the cage walls seen from above)
         structure      outside the box: dark static or edgy (wall grid / vent slots, corner fixtures, the spout's
                        mount); inside the box: dark static components (8-conn, >= 30 px) within 48 px of the box edge
                        (things that reach into the arena; components farther in are not counted)
         spout          dark static components (>= 30 px) inside the band from 45 px outside to 48 px inside the box
                        wall on the video's camera_spout_side
         bag            per VIDEO (the bag is not a fixed structure): white pixels (pix_bg > max(0.95 x centre_grey,
                        170), the bag rule of mice_odor_corner.py) outside the floor box grown by 12 px, in the image
                        quadrant of the odour corner, components >= 200 px. Compared with the v2 BAG zones (regions.npz
                        reg_m2, bag core dilated 2 patches; 2 of the 48 cores lie inside the arena, i.e. not a bag).
       Patch object type (per video): bag if >= 26 bag px; else spout if >= 26 spout px; else structure if >= 26
       structure px (outside-box structure counts only for patches whose centre is outside the box); else wall if the
       patch centre is outside the floor box; else floor (no object).
       REVISION (after the first object sheet, before any neuron statistic): (a) the floor box is the per-SESSION
       median of the 3 videos' boxes (single-video boxes were off by up to 4 patch rows, e.g. het_ash1l_f_1_F_O, which
       turned real floor into wall); (b) outside-box structure no longer labels patches whose centre is on the floor
       (it put a 1-patch object band inside the whole floor edge, giving 14 interaction patches per frame and an
       interaction patch in 98.6% of frames). The wall / structure split is descriptive only: the edge cue marks most
       of the wall region (median 338 structure vs 282 wall patches per video), so the two are reported together.
       Extremities: per instance, from its SAM body mask (no tail): e1 = the pixel farthest from the centroid, e2 = the
       pixel farthest from the centroid on the other side (negative projection on e1 - centroid): the two body ends,
       nose or tail base (no keypoints exist; head-or-tail proxy, both ends used).
       Interaction patch (per frame): an object patch q such that, for some instance k and one of its extremities e,
       q is within 1 patch (Chebyshev) of e's patch, and q is either off every animal or is e's own patch (so the body
       of a mouse lying on a wall patch does not count, the touching end does).
CLASSES (per neuron, its valid top frames, active-patch weights = frame weights; mutually exclusive, in this order)
    env_share = share of active patches that are interaction patches; base_env = the same share over every SAE-mask
    patch of every valid frame (weighted).
    1 environment interaction  env_share >= 0.5 AND env_share >= min(2 x base_env, (1 + base_env) / 2) (as v2's zone
                               concentration); reported with the object type holding most of its interaction patches
    2 on-animal                on_share >= 0.5 (extended masks) -> social: self / pair / huddle / apart / none (fix 2)
    3 off-animal               of the off-animal active patches: halo = not on an object AND 1 patch from an animal;
                               object-near = on an object AND <= 2 patches from an animal; mid = not on an object AND 2
                               patches away; far = >= 3 patches away (object or floor).
                               halo if halo >= 0.5; plain background if far >= 0.5; object-near if object-near >= 0.5;
                               else off-other.
    Eligible neurons, top frames and frame weights: v2 (fire share >= 1%, top 1% of the frame weight).

Validation: class counts (fg448m5, fg448al for comparison), v2 -> v3 confusion (v2 exclusive class = place-bound if
flagged, else its social class when on-animal, else its location), anchors (915 nose-nose / nose-tail -> pair; the
bag-like 180 -> environment interaction; the 3 eligible neurons with the highest tail-only share -> self), the v3 class
of every v2 hand-checked neuron, and visual sheets of 15 random neurons per v3 class (seed 0) with the extended
instances (tails in full colour), objects, extremities and active patches.

Steps (scripts/eci/neuron_map_v3.sh; CPU)
    look     design renders (pix_bg of 5 sessions + edges; 8 frames with SAM masks + dark-cue pixels)
    objects  per-video object map + sanity sheet of 4 sessions -> OUT/mice/objects.npz, objects.json, objects_sheet.jpg
    prep     per frame (all 3 cues): extended masks -> patch bits, tail-only bits, distance map, clusters, extremities,
             interaction map -> OUT/mice/prep_<cue>.npz; tail stats; tail sheet of 20 valid frames (seed 0)
    analyse  per dictionary (--cue) -> OUT/<dict>[_<cue>]/neurons.csv, summary.json
    sheets   hand-check sheets -> OUT/sheets/<dict>[_<cue>]/
Outputs results/vision/eci_neuron_map_v3/ (gitignored).
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
import neuron_map_v2 as nm2  # noqa: E402

OUT = REPO / 'results/vision/eci_neuron_map_v3'
V2 = nm2.OUT
BGDIR = REPO / 'dataset/mice/v1/eci/fg448/background'
VIDDIR = REPO / 'data/mice/v1/observations/full'
GRID, PX = nm2.GRID, nm2.PX
log = nm2.log
# fix 1 (tails)
CUES = {'dark': (60, 40), 'mid': (120, 50), 'rel': (256, 40), 'line': None}  # (grey < abs) & (pix_bg - grey > rel)
TAIL_R, TAIL_ROUNDS, TAIL_PX, OVERLAP_PX = 4.0, 3, 10, 26
HUDDLE_DIL = 3
# fix 3 (objects)
OBJ_NAMES = ['floor', 'wall', 'structure', 'spout', 'bag']
DARK_STATIC, SAM_COVER_MAX, EDGE_K = 72, 0.3, 3.0
BAND_IN, BAND_OUT, MIN_COMP_PX = 48, 45, 30
BAG_GROW, BAG_MIN_PX, OBJ_PX = 12, 200, 26
BOX = ['arena_r0', 'arena_r1', 'arena_c0', 'arena_c1']
# fix 1, revised after the tail sheet (see docstring): thin dark line grown from a body end
LINE_TOPHAT, LINE_REL, LINE_SEED_R, LINE_LEN, LINE_MAX_WIDTH, LINE_MIN_LEN, LINE_CLOSE = 30, 40, 20, 120, 7.0, 20, 9
# classes
ON_MIN, ENV_MIN, HALO_MIN, FAR_MIN, OBJ_MIN, APART_MIN = 0.5, 0.5, 0.5, 0.5, 0.5, 0.5
N_CHECK, N_SHEET = 15, 12
CLASSES = ['self', 'pair', 'huddle', 'apart', 'env-interaction', 'halo', 'plain-background', 'object-near']
# the v2 visual hand-check of fg448m5 (results/vision/eci_neuron_map_v2/handcheck/fg448m5_notes.txt, REVIEW lines):
# neuron -> (v2 class, verdict, cause noted in v2). Used only to report where v3 files them.
V2_JUDGED = {
    **{j: ('self', 'C', '') for j in (55, 181, 778, 2094, 2213, 3101, 3217, 3934)},
    **{j: ('self', 'U', '') for j in (97, 442, 906, 1712, 1715, 2318, 3631)},
    **{j: ('pair', 'C', 'pair contact') for j in (378, 4046)},
    **{j: ('pair', 'W', 'huddle 3-4') for j in (209, 834, 835, 903, 1191, 1642, 2993)},
    **{j: ('pair', 'W', 'generic whole-body, every mouse') for j in (14, 76, 677, 935, 1944)},
    117: ('pair', 'U', ''),
    **{j: ('collective', 'C', 'huddle') for j in (79, 225, 322, 330, 333, 447, 1012, 1039, 1353, 1893, 2313)},
    275: ('collective', 'W', 'generic'), **{j: ('collective', 'U', '') for j in (222, 433, 594)},
    **{j: ('place-bound', 'C', 'object interaction') for j in (1259, 1308)},
    **{j: ('place-bound', 'W', 'generic spot') for j in (2080, 2117)},
    **{j: ('place-bound', 'U', 'huddle at wall') for j in (779, 800, 1093, 3443)},
    **{j: ('place-bound', 'U', '') for j in (2879, 3106, 4089)},
    **{j: ('halo', 'C', '') for j in (177, 3698)},
    **{j: ('halo', 'W', 'tails') for j in (109, 843, 1843, 3413)},
    **{j: ('halo', 'W', 'object interaction') for j in (436, 2521, 3432, 4026)},
    **{j: ('halo', 'W', 'wall surface') for j in (586, 1616)},
    **{j: ('halo', 'U', '') for j in (369, 1530, 2152)},
    **{j: ('plain-background', 'C', 'walls') for j in (6, 348, 453, 480, 907, 922)},
    **{j: ('plain-background', 'W', 'tails') for j in (178, 440, 464, 807, 1365, 1992, 2765, 2852, 2926)}}


def vname_of(fpath):
    return fpath.split('/')[-2]


def load_frames():
    F = pd.read_parquet(V2 / 'mice' / 'frames.parquet')
    F['vname'] = F.frame_path.map(vname_of)
    F['sess'] = F.obs.str[:-2]
    return F


def stage_video(v, loc):
    dst = Path(loc) / 'vid' / f'{v}.mp4'
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(VIDDIR / f'{v}.mp4', dst)
    return dst


def decode_frames(v, fidx, loc, keep=False):
    """Frames fidx (indices of the 5 fps standardised mp4 = the dataset frame_idx) of video v, decoded locally from a
    staged copy. -> {frame_idx: (512, 512, 3) uint8}."""
    from PIL import Image
    src = stage_video(v, loc)
    d = Path(loc) / 'dec' / v
    d.mkdir(parents=True, exist_ok=True)
    fidx = np.unique(np.asarray(fidx, np.int64))
    sel = '+'.join(f'eq(n\\,{int(f)})' for f in fidx)
    subprocess.run(['ffmpeg', '-nostdin', '-loglevel', 'error', '-y', '-i', str(src), '-vf', f'select={sel},format=rgb24',
                    '-vsync', '0', '-start_number', '0', str(d / 'out_%06d.png')], check=True)
    files = sorted(d.glob('out_*.png'))
    assert len(files) == len(fidx), (v, len(files), len(fidx))
    out = {int(f): np.asarray(Image.open(p).convert('RGB')) for f, p in zip(fidx, files)}
    shutil.rmtree(d)
    if not keep:
        src.unlink()
    return out


def grey_of(rgb):
    from PIL import Image
    return np.asarray(Image.fromarray(rgb).convert('L'), np.uint8)


def pix_bg(obs):
    return np.load(BGDIR / f'{obs}.npz')['pix_bg']


def shard_paths(loc):
    out = []
    for pth in sorted(nm2.PAIRS.glob('labels_s*.npz')):
        lp = Path(loc) / pth.name
        if not lp.exists():
            shutil.copyfile(pth, lp)
        out.append(lp)
    assert len(out) == 5, out
    return out


def load_sam(loc, tids):
    """SAM 2 labels (512, 512) uint8 (0 bg, 1-4 mice) of the given target ids."""
    want = set(int(t) for t in tids)
    out = {}
    for lp in shard_paths(loc):
        z = np.load(lp)
        ids = z['idx']
        hit = [i for i, t in enumerate(ids) if int(t) in want]
        if hit:
            L = z['labels']
            for i in hit:
                out[int(ids[i])] = L[i].copy()
            del L
    return out


# ---------------------------------------------------------------------------------------------- geometry per frame
S8 = np.ones((3, 3), bool)


def cue_mask(g, pb, cue):
    a, r = CUES[cue]
    return (g.astype(np.int16) < a) & ((pb.astype(np.int16) - g.astype(np.int16)) > r)


DIAG = None


def line_cue(g, pb):
    """Thin dark pixels: black top-hat (grey closing 9 x 9 - grey) > 30 AND > 40 below pix_bg."""
    from scipy import ndimage as ndi
    gi = g.astype(np.int16)
    th = ndi.grey_closing(g, size=(LINE_CLOSE, LINE_CLOSE)).astype(np.int16) - gi
    return (th > LINE_TOPHAT) & ((pb.astype(np.int16) - gi) > LINE_REL)


def extend_line(L, g, pb, ext):
    """Fix 1 (revised, 'line'): per instance and body end, grow the line cue (8-connected, outside every mask) from the
    cue pixels within 20 px of the end, geodesically up to 120 px; a grown set is a tail candidate iff it reaches >= 20
    px with mean width (pixels / geodesic length) <= 7 px; per mouse the longer of its two ends' candidates. Candidates
    are assigned greedily in order of the end-to-set distance (closest first): a candidate whose pixels are > 50%
    already taken is dropped, one tail per mouse. -> as extend()."""
    from scipy import ndimage as ndi
    c = line_cue(g, pb)
    free = c & (L == 0)
    W = LINE_LEN + LINE_SEED_R + 4
    cands = []
    for k in range(4):
        best = None
        for e in range(2):
            y, x = ext[k, e]
            if not np.isfinite(y):
                continue
            y, x = int(y), int(x)
            y0, y1, x0, x1 = max(y - W, 0), min(y + W + 1, 512), max(x - W, 0), min(x + W + 1, 512)
            fw = free[y0:y1, x0:x1]
            yy, xx = np.mgrid[y0:y1, x0:x1]
            d2 = (yy - y) ** 2 + (xx - x) ** 2
            seed = fw & (d2 <= LINE_SEED_R ** 2)
            if not seed.any():
                if DIAG is not None:
                    DIAG.append((k, e, 0, 0, 0, 0.0))
                continue
            cur = seed.copy()
            steps = 0
            for _ in range(LINE_LEN):
                nxt = ndi.binary_dilation(cur, S8) & fw & ~cur
                if not nxt.any():
                    break
                cur |= nxt
                steps += 1
            if DIAG is not None:
                DIAG.append((k, e, int(seed.sum()), steps, int(cur.sum()), float(cur.sum() / max(steps, 1))))
            if steps >= LINE_MIN_LEN and (cur.sum() - seed.sum()) / steps <= LINE_MAX_WIDTH:
                if best is None or steps > best[0]:
                    best = (steps, (y0, y1, x0, x1), cur, float(np.sqrt(d2[cur].min())))
        if best is not None:
            cands.append((best[3], k, best[1], best[2]))
    Le = L.copy()
    for _, k, (y0, y1, x0, x1), cur in sorted(cands, key=lambda t: t[0]):
        sub = Le[y0:y1, x0:x1]
        if (sub[cur] > 0).mean() > 0.5:
            continue
        sub[cur & (sub == 0)] = k + 1
    tail = np.array([int(((Le == k) & (L == 0)).sum()) for k in range(1, 5)])
    st = np.array([int(c.sum()), int((c & (L > 0)).sum()), int(tail.sum()), int((c & (Le == 0)).sum())])
    return Le, tail, st


def extend_any(L, g, pb, ext, cue):
    return extend_line(L, g, pb, ext) if cue == 'line' else extend(L, cue_mask(g, pb, cue))


def extend(L, cue):
    """Fix 1. L (512, 512) SAM labels, cue (512, 512) bool -> extended labels Le, tail-pixel count per instance (4,),
    cue pixels: total, inside SAM, attached, unattached."""
    from scipy import ndimage as ndi
    ex = cue & (L == 0)
    lab, n = ndi.label(ex, S8)
    Le = L.copy()
    if n and (L > 0).any():
        att = np.zeros(n + 1, bool)
        att[0] = True
        lf = lab.ravel()
        for _ in range(TAIL_ROUNDS):
            D, ind = ndi.distance_transform_edt(Le == 0, return_indices=True)
            NL = Le[ind[0], ind[1]].ravel()
            Df = D.ravel()
            idx = np.flatnonzero(~att[lf])
            if not len(idx):
                break
            o = np.lexsort((Df[idx], lf[idx]))
            idx = idx[o]
            c = lf[idx]
            first = np.r_[True, c[1:] != c[:-1]]
            cid, pix = c[first], idx[first]
            ok = Df[pix] <= TAIL_R
            if not ok.any():
                break
            new = np.zeros(n + 1, np.uint8)
            new[cid[ok]] = NL[pix[ok]]
            add = new[lab]
            m = add > 0
            Le[m] = add[m]
            att[cid[ok]] = True
    tail = np.array([int(((Le == k) & (L == 0)).sum()) for k in range(1, 5)])
    st = np.array([int(cue.sum()), int((cue & (L > 0)).sum()), int(tail.sum()), int((ex & (Le == 0)).sum())])
    return Le, tail, st


def patch_counts(M):
    """(512, 512) labels -> (1024, 4) pixel count per patch of labels 1-4."""
    return np.stack([(M == k).reshape(GRID, PX, GRID, PX).sum((1, 3)).ravel() for k in range(1, 5)], -1)


def patch_bits(L, Le):
    sam = patch_counts(L)
    tl = patch_counts(np.where(L == 0, Le, 0))
    on = ((sam + tl) >= OVERLAP_PX) | (tl >= TAIL_PX)
    tonly = on & (sam < OVERLAP_PX)
    w = (1 << np.arange(4))[None]
    return (on * w).sum(1).astype(np.uint8), (tonly * w).sum(1).astype(np.uint8)


def clusters(Le):
    """Fix 2: union-find over touching extended instances (dilated by 3 px) -> root per instance (4,) int8 (-1 absent),
    largest cluster size."""
    from scipy import ndimage as ndi
    st = np.ones((2 * HUDDLE_DIL + 1, 2 * HUDDLE_DIL + 1), bool)
    masks = [Le == k for k in range(1, 5)]
    pres = [m.any() for m in masks]
    dil = []
    for m, p in zip(masks, pres):
        if p:
            ys, xs = np.nonzero(m)
            y0, y1 = max(ys.min() - HUDDLE_DIL, 0), min(ys.max() + HUDDLE_DIL + 1, 512)
            x0, x1 = max(xs.min() - HUDDLE_DIL, 0), min(xs.max() + HUDDLE_DIL + 1, 512)
            d = np.zeros_like(m)
            d[y0:y1, x0:x1] = ndi.binary_dilation(m[y0:y1, x0:x1], st)
            dil.append(d)
        else:
            dil.append(m)
    par = list(range(4))

    def f(a):
        while par[a] != a:
            a = par[a]
        return a
    for a in range(4):
        for c in range(a + 1, 4):
            if pres[a] and pres[c] and (dil[a] & masks[c]).any():
                par[f(a)] = f(c)
    root = np.array([f(a) if pres[a] else -1 for a in range(4)], np.int8)
    r = root[root >= 0]
    return root, int(np.bincount(r).max()) if len(r) else 0


def extremities(L):
    """Per SAM body mask: the two body ends (e1 farthest from the centroid, e2 farthest on the other side) -> (4, 2, 2)
    float32 (instance, end, (y, x)), NaN if absent."""
    out = np.full((4, 2, 2), np.nan, np.float32)
    for k in range(4):
        ys, xs = np.nonzero(L == k + 1)
        if len(ys) < 5:
            continue
        cy, cx = ys.mean(), xs.mean()
        d2 = (ys - cy) ** 2 + (xs - cx) ** 2
        i1 = int(np.argmax(d2))
        out[k, 0] = (ys[i1], xs[i1])
        proj = (ys - cy) * (ys[i1] - cy) + (xs - cx) * (xs[i1] - cx)
        neg = proj < 0
        if neg.any():
            j = np.flatnonzero(neg)[int(np.argmax(d2[neg]))]
            out[k, 1] = (ys[j], xs[j])
    return out


def interaction_map(bits, ext, otype):
    """Fix 3: (1024,) bool interaction patches (module docstring) for one frame. bits (1024,) uint8, ext (4, 2, 2),
    otype (1024,) int8 object type of the video."""
    obj = (otype > 0).reshape(GRID, GRID)
    on = (bits > 0).reshape(GRID, GRID)
    out = np.zeros((GRID, GRID), bool)
    for k in range(4):
        for e in range(2):
            y, x = ext[k, e]
            if not np.isfinite(y):
                continue
            py, px = int(y) // PX, int(x) // PX
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    qy, qx = py + dy, px + dx
                    if 0 <= qy < GRID and 0 <= qx < GRID and obj[qy, qx] and (not on[qy, qx] or (dy == 0 and dx == 0)):
                        out[qy, qx] = True
    return out.ravel()


# ---------------------------------------------------------------------------------------------- look (CPU, design)
def cmd_look(args):
    """Design renders (label-free, no neuron): pix_bg of the H / O / P videos of 5 sessions with the arena box, the v2
    bag core and the edge map; 8 valid frames with SAM masks and the dark-cue pixels outside them (magenta)."""
    from PIL import Image, ImageDraw
    from scipy import ndimage as ndi
    loc = nm2.local_dir()
    od = OUT / 'look'
    od.mkdir(parents=True, exist_ok=True)
    F = load_frames()
    oc = pd.read_csv(nm2.ODOR).set_index('observation_id')
    R = np.load(nm2.REGIONS)
    sess = list(R['sessions'])
    rng = np.random.default_rng(0)
    corners = oc.loc[[s + '_H' for s in sess], 'odor_corner'].values
    pick = []
    for c, k in (('BL', 2), ('TR', 2), ('BR', 1)):
        ix = [i for i, x in enumerate(corners) if x == c]
        pick += list(rng.choice(ix, min(k, len(ix)), replace=False))
    tiles = []
    for si in pick:
        s = sess[si]
        row = []
        for ph in 'HOP':
            o = f'{s}_{ph}'
            pb = pix_bg(o)
            im = Image.fromarray(np.stack([pb] * 3, -1))
            dr = ImageDraw.Draw(im)
            r = oc.loc[o]
            dr.rectangle([r.arena_c0, r.arena_r0, r.arena_c1, r.arena_r1], outline=(255, 0, 0))
            for q in np.flatnonzero(R['core'][si].ravel()):
                a, b = divmod(int(q), GRID)
                dr.rectangle([b * PX, a * PX, b * PX + PX - 1, a * PX + PX - 1], outline=(0, 200, 255))
            dr.text((4, 4), f'{o} corner {r.odor_corner} spout {r.camera_spout_side}', fill=(255, 0, 0))
            row.append(im)
        S = np.max(np.stack([pix_bg(f'{s}_{ph}') for ph in 'HOP']), 0).astype(np.float32)
        g = ndi.gaussian_filter(S, 1.0)
        e = np.hypot(ndi.sobel(g, 0), ndi.sobel(g, 1))
        row.append(Image.fromarray(np.clip(e / np.quantile(e, 0.99) * 255, 0, 255).astype(np.uint8)).convert('RGB'))
        sheet = Image.new('RGB', (4 * 512, 512))
        for i, t in enumerate(row):
            sheet.paste(t, (i * 512, 0))
        tiles.append(sheet)
    big = Image.new('RGB', (4 * 512, 512 * len(tiles)))
    for i, t in enumerate(tiles):
        big.paste(t, (0, i * 512))
    big.save(od / 'pixbg_sessions.jpg', quality=85)
    P = np.load(V2 / 'mice' / 'prep.npz')
    fr = np.sort(rng.choice(np.flatnonzero(P['valid']), 8, replace=False))
    sam = load_sam(loc, F.tid.values[fr])
    tl = []
    for f in fr:
        rgb = decode_frames(F.vname.values[f], [F.frame_idx.values[f]], loc)[int(F.frame_idx.values[f])]
        g = grey_of(rgb)
        pb = pix_bg(F.obs.values[f])
        L = sam[int(F.tid.values[f])]
        im = rgb.astype(np.float32).copy()
        for k in range(1, 5):
            m = L == k
            im[m] = 0.55 * im[m] + 0.45 * nm2.INST_RGB[k - 1]
        for cue, col in (('rel', (120, 255, 120)), ('mid', (255, 160, 0)), ('dark', (255, 0, 255))):
            im[cue_mask(g, pb, cue) & (L == 0)] = col
        t = Image.fromarray(im.clip(0, 255).astype(np.uint8))
        ImageDraw.Draw(t).text((4, 4), f'{F.obs.values[f]} f{F.frame_idx.values[f]}: outside SAM, magenta dark / '
                               f'orange +mid / green +rel', fill=(255, 0, 0))
        tl.append(t)
    sh = Image.new('RGB', (4 * 512, 2 * 512))
    for i, t in enumerate(tl):
        sh.paste(t, ((i % 4) * 512, (i // 4) * 512))
    sh.save(od / 'frames_cues.jpg', quality=90)
    log(f'look -> {od}')


# ---------------------------------------------------------------------------------------------- objects (CPU)
def comps_min(mask, min_px):
    from scipy import ndimage as ndi
    lab, n = ndi.label(mask, S8)
    if not n:
        return mask & False
    sz = np.bincount(lab.ravel())
    keep = sz >= min_px
    keep[0] = False
    return keep[lab]


def object_pixels(S, cover, pb, r, edge_thr=None):
    """One video's object pixel maps (dict of (512, 512) bool) from the session static image S, the session SAM cover,
    the video's pix_bg and its odor_corner.csv row r (module docstring)."""
    from scipy import ndimage as ndi
    yy, xx = np.mgrid[0:512, 0:512]
    r0, r1, c0, c1 = int(r.arena_r0), int(r.arena_r1), int(r.arena_c0), int(r.arena_c1)
    inside = (yy >= r0) & (yy <= r1) & (xx >= c0) & (xx <= c1)
    din = np.minimum.reduce([yy - r0, r1 - yy, xx - c0, c1 - xx])  # > 0 inside: px to the nearest box edge
    g = ndi.gaussian_filter(S.astype(np.float32), 1.0)
    e = np.hypot(ndi.sobel(g, 0), ndi.sobel(g, 1))
    if edge_thr is None:
        edge_thr = EDGE_K * float(np.median(e[~inside]))
    dark_static = (S < DARK_STATIC) & (cover <= SAM_COVER_MAX)
    struct_out = ~inside & (dark_static | (e > edge_thr))
    struct_in = comps_min(dark_static & inside & (din <= BAND_IN), MIN_COMP_PX)
    side = r.camera_spout_side
    if side == 'right':
        band = (xx >= c1 - BAND_IN) & (xx <= c1 + BAND_OUT) & (yy >= r0) & (yy <= r1)
    elif side == 'left':
        band = (xx <= c0 + BAND_IN) & (xx >= c0 - BAND_OUT) & (yy >= r0) & (yy <= r1)
    elif side == 'top':
        band = (yy <= r0 + BAND_IN) & (yy >= r0 - BAND_OUT) & (xx >= c0) & (xx <= c1)
    else:
        band = (yy >= r1 - BAND_IN) & (yy <= r1 + BAND_OUT) & (xx >= c0) & (xx <= c1)
    spout = comps_min(dark_static & band, MIN_COMP_PX)
    white = pb.astype(np.float32) > max(0.95 * float(r.centre_grey), 170.0)
    grown = (yy >= r0 - BAG_GROW) & (yy <= r1 + BAG_GROW) & (xx >= c0 - BAG_GROW) & (xx <= c1 + BAG_GROW)
    q = r.odor_corner
    quad = ((yy < 256) if q[0] == 'T' else (yy >= 256)) & ((xx < 256) if q[1] == 'L' else (xx >= 256))
    bag = comps_min(white & ~grown & quad, BAG_MIN_PX)
    return {'inside': inside, 'structure': struct_out | struct_in, 'struct_out': struct_out, 'struct_in': struct_in,
            'spout': spout, 'bag': bag,
            'edge_thr': edge_thr}


def patch_types(op):
    cnt = lambda m: m.reshape(GRID, PX, GRID, PX).sum((1, 3)).ravel()  # noqa: E731
    cy, cx = np.divmod(np.arange(GRID * GRID), GRID)
    centre_in = op['inside'][cy * PX + PX // 2, cx * PX + PX // 2]
    t = np.where(~centre_in, 1, 0)
    t = np.where(((cnt(op['struct_out']) >= OBJ_PX) & ~centre_in) | (cnt(op['struct_in']) >= OBJ_PX), 2, t)
    t = np.where(cnt(op['spout']) >= OBJ_PX, 3, t)
    t = np.where(cnt(op['bag']) >= OBJ_PX, 4, t)
    return t.astype(np.int8)


def cmd_objects(args):
    from PIL import Image, ImageDraw
    loc = nm2.local_dir()
    t0 = time.time()
    F = load_frames()
    oc = pd.read_csv(nm2.ODOR).set_index('observation_id')
    vids = sorted(F.obs.unique())
    sessions = sorted({v[:-2] for v in vids})
    sess_of_tid = pd.Series(F.sess.values, index=F.tid.values)
    cov = {s: np.zeros((512, 512), np.float32) for s in sessions}
    ncov = {s: 0 for s in sessions}
    for lp in shard_paths(loc):
        z = np.load(lp)
        ids, L = z['idx'], z['labels']
        for i, t in enumerate(ids):
            s = sess_of_tid[int(t)]
            cov[s] += L[i] > 0
            ncov[s] += 1
        del L
        log(f'  cover {lp.name} ({time.time() - t0:.0f}s)')
    R = np.load(nm2.REGIONS)
    rmap = pd.Series(np.arange(len(R['obs'])), index=R['obs'])
    otype = np.zeros((len(vids), GRID * GRID), np.int8)
    opx = np.zeros((len(vids), 512, 512), np.uint8)  # pixel types (sheets): 1 wall .. 4 bag, 5 inside structure
    box = np.zeros((len(vids), 4), np.int32)  # session-median floor box r0, r1, c0, c1
    rows = []
    for s in sessions:
        vs = [v for v in vids if v[:-2] == s]
        S = np.max(np.stack([pix_bg(f'{s}_{ph}') for ph in 'HOP' if (BGDIR / f'{s}_{ph}.npz').exists()]), 0)
        cover = cov[s] / max(ncov[s], 1)
        sbox = oc.loc[vs, BOX].median().round().astype(int)
        for v in vs:
            i = vids.index(v)
            r = oc.loc[v].copy()
            r[BOX] = sbox.values
            box[i] = sbox.values
            op = object_pixels(S, cover, pix_bg(v), r)
            t = patch_types(op)
            otype[i] = t
            px = np.where(~op['inside'], 1, 0)
            px = np.where(op['structure'], 2, px)
            px = np.where(op['struct_in'], 5, px)
            px = np.where(op['spout'], 3, px)
            px = np.where(op['bag'], 4, px)
            opx[i] = px
            bagz = R['reg_m2'][rmap[v]] == 0
            rows.append({'obs': v, 'session': s, 'phase': v[-1], 'spout_side': r.camera_spout_side,
                         'corner': r.odor_corner, 'edge_thr': op['edge_thr'], 'sam_cover_frames': ncov[s],
                         **{f'n_{nm}': int((t == k).sum()) for k, nm in enumerate(OBJ_NAMES)},
                         'n_struct_inside_px': int(op['struct_in'].sum()), 'n_spout_px': int(op['spout'].sum()),
                         'n_bag_px': int(op['bag'].sum()), 'bag_patches_in_v2_BAG_zone': int(((t == 4) & bagz).sum()),
                         'v2_BAG_zone_patches': int(bagz.sum()),
                         'v2_core_inside_arena': bool(R['core'][list(R['sessions']).index(s)].ravel()[
                             (op['inside'][np.arange(GRID)[:, None] * PX + 8, np.arange(GRID)[None] * PX + 8]).ravel()].any())})
    T = pd.DataFrame(rows)
    od = OUT / 'mice'
    od.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(od / 'objects.npz', obs=np.array(vids), otype=otype, opx=opx, names=np.array(OBJ_NAMES), box=box)
    T.to_csv(od / 'objects.csv', index=False)
    summ = {'n_videos': len(vids), 'n_sessions': len(sessions),
            'patches_per_video_median': {nm: float(T[f'n_{nm}'].median()) for nm in OBJ_NAMES},
            'videos_with_spout_patch': int((T.n_spout > 0).sum()),
            'videos_with_bag_patch_by_phase': T.assign(b=T.n_bag > 0).groupby('phase').b.sum().astype(int).to_dict(),
            'videos_with_inside_structure': int((T.n_struct_inside_px > 0).sum()),
            'bag_patches_in_v2_zone_share': float(T.bag_patches_in_v2_BAG_zone.sum() / max(T.n_bag.sum(), 1)),
            'elapsed_s': time.time() - t0}
    (od / 'objects.json').write_text(json.dumps(summ, indent=1, default=float))
    log(f'objects {json.dumps(summ, default=float)}')
    # sanity sheet: 4 sessions (seed 0; one per spout side present, then random), H / O / P
    rng = np.random.default_rng(0)
    first = T.groupby('spout_side').session.first().tolist()
    rest = [s for s in rng.permutation(sessions) if s not in first]
    pick = (first + rest)[:4]
    col = {1: None, 2: (255, 0, 255), 3: (255, 0, 0), 4: (0, 230, 255), 5: (255, 140, 0)}
    pcol = {2: (255, 0, 255), 3: (255, 0, 0), 4: (0, 230, 255)}
    sheet = Image.new('RGB', (3 * 512, 4 * 530), (255, 255, 255))
    for a, s in enumerate(pick):
        for b, ph in enumerate('HOP'):
            v = f'{s}_{ph}'
            if v not in vids:
                continue
            i = vids.index(v)
            im = np.stack([pix_bg(v)] * 3, -1).astype(np.float32)
            for k, c in col.items():
                if c is not None:
                    m = opx[i] == k
                    im[m] = 0.4 * im[m] + 0.6 * np.array(c)
            pil = Image.fromarray(im.clip(0, 255).astype(np.uint8))
            dr = ImageDraw.Draw(pil)
            b0 = box[i]
            dr.rectangle([b0[2], b0[0], b0[3], b0[1]], outline=(0, 255, 0))
            for q in range(GRID * GRID):
                tq = int(otype[i, q])
                if tq in pcol:
                    y, x = divmod(q, GRID)
                    dr.rectangle([x * PX, y * PX, x * PX + PX - 1, y * PX + PX - 1], outline=pcol[tq], width=1)
            sheet.paste(pil, (b * 512, a * 530))
            rr = T[T.obs == v].iloc[0]
            ImageDraw.Draw(sheet).text((b * 512 + 4, a * 530 + 514),
                                       f'{v} spout {rr.spout_side} corner {rr.corner} | patches wall {rr.n_wall} struct '
                                       f'{rr.n_structure} spout {rr.n_spout} bag {rr.n_bag}', fill=(0, 0, 0))
    sheet.save(od / 'objects_sheet.jpg', quality=88)
    log(f'objects sheet -> {od / "objects_sheet.jpg"} sessions {pick}')


# ---------------------------------------------------------------------------------------------- prep (CPU)
_SH = {}


def _prep_video(job):
    """One video: decode its B frames, extend the SAM masks with each cue, patch bits, clusters, extremities,
    interaction map. Uses the shard labels of the parent (_SH, fork)."""
    v, obs, pos, fidx, li, vi, render = job
    loc = os.environ['LOCAL_DIR']
    imgs = decode_frames(v, fidx, loc)
    pb = pix_bg(obs)
    L_all = _SH['L']
    otype = _SH['otype'][vi]
    n = len(pos)
    res = {c: {'bits': np.zeros((n, 1024), np.uint8), 'tonly': np.zeros((n, 1024), np.uint8),
               'dist': np.zeros((n, 1024), np.uint8), 'root': np.zeros((n, 4), np.int8), 'clus': np.zeros(n, np.int8),
               'intr': np.zeros((n, 1024), bool), 'tail': np.zeros((n, 4), np.int32), 'st': np.zeros((n, 4), np.int64)}
           for c in CUES}
    ext = np.zeros((n, 4, 2, 2), np.float32)
    tiles = {}
    for i in range(n):
        L = L_all[li[i]]
        g = grey_of(imgs[int(fidx[i])])
        ext[i] = extremities(L)
        for c in CUES:
            Le, tail, st = extend_any(L, g, pb, ext[i], c)
            b, to = patch_bits(L, Le)
            r = res[c]
            r['bits'][i], r['tonly'][i], r['tail'][i], r['st'][i] = b, to, tail, st
            r['dist'][i] = nm2.cheb_dist((b > 0).reshape(GRID, GRID))
            r['root'][i], r['clus'][i] = clusters(Le)
            r['intr'][i] = interaction_map(b, ext[i], otype)
            if render[i]:
                tiles.setdefault(int(pos[i]), {})[c] = tail_tile(imgs[int(fidx[i])], L, Le,
                                                                 line_cue(g, pb) if c == 'line' else cue_mask(g, pb, c), ext[i],
                                                                 f'{obs} f{fidx[i]} cue {c}: tail px {tail.tolist()}, '
                                                                 f'unattached {st[3]}', r['intr'][i], otype)
    return pos, res, ext, tiles


def tail_tile(rgb, L, Le, cue, ext, title, intr=None, otype=None):
    """SAM masks tinted, attached tail pixels in full instance colour, unattached cue pixels magenta, body ends white."""
    from PIL import Image, ImageDraw
    im = rgb.astype(np.float32).copy()
    for k in range(1, 5):
        m = L == k
        im[m] = 0.6 * im[m] + 0.4 * nm2.INST_RGB[k - 1]
        im[(Le == k) & (L == 0)] = nm2.INST_RGB[k - 1]
    im[cue & (Le == 0)] = (255, 0, 255)
    pil = Image.fromarray(im.clip(0, 255).astype(np.uint8))
    dr = ImageDraw.Draw(pil)
    for k in range(4):
        for e in range(2):
            y, x = ext[k, e]
            if np.isfinite(y):
                dr.ellipse([x - 3, y - 3, x + 3, y + 3], outline=(255, 255, 255), width=2)
    for q in range(GRID * GRID):
        r_, c_ = divmod(q, GRID)
        if otype is not None and otype[q] >= 2:
            dr.rectangle([c_ * PX + 2, r_ * PX + 2, c_ * PX + PX - 3, r_ * PX + PX - 3], outline=OBJ_RGB[int(otype[q])])
        if intr is not None and intr[q]:
            dr.rectangle([c_ * PX, r_ * PX, c_ * PX + PX - 1, r_ * PX + PX - 1], outline=(0, 255, 255), width=2)
    dr.text((4, 4), title, fill=(255, 0, 0))
    return np.asarray(pil)


def cmd_prep(args):
    from multiprocessing import get_context

    from PIL import Image
    loc = nm2.local_dir()
    t0 = time.time()
    F = load_frames()
    nF = len(F)
    P2 = np.load(V2 / 'mice' / 'prep.npz')
    valid = P2['valid']
    O = np.load(OUT / 'mice' / 'objects.npz')
    ovids = list(O['obs'])
    vids = sorted(F.obs.unique())
    assert ovids == vids
    _SH['otype'] = O['otype']
    rng = np.random.default_rng(0)
    render = np.zeros(nF, bool)
    rpick = np.sort(rng.choice(np.flatnonzero(valid), 20, replace=False))
    render[rpick] = True
    out = {c: {'bits': np.zeros((nF, 1024), np.uint8), 'tonly': np.zeros((nF, 1024), np.uint8),
               'dist': np.zeros((nF, 1024), np.uint8), 'root': np.full((nF, 4), -1, np.int8),
               'clus': np.zeros(nF, np.int8), 'intr': np.zeros((nF, 1024), bool), 'tail': np.zeros((nF, 4), np.int32),
               'st': np.zeros((nF, 4), np.int64)} for c in CUES}
    ext = np.full((nF, 4, 2, 2), np.nan, np.float32)
    seen = np.zeros(nF, bool)
    tiles = {}
    pos_of_tid = pd.Series(np.arange(nF), index=F.tid.values)
    for lp in shard_paths(loc):
        z = np.load(lp)
        ids = z['idx']
        _SH['L'] = z['labels']
        pos = pos_of_tid.loc[ids].values
        li_of = pd.Series(np.arange(len(ids)), index=pos)
        jobs = []
        for o, g in F.iloc[np.sort(pos)].groupby('obs'):
            p = g.index.values
            jobs.append((g.vname.values[0], o, p, g.frame_idx.values, li_of.loc[p].values, vids.index(o), render[p]))
        with get_context('fork').Pool(args.workers) as pool:
            for p, res, ex, tl in pool.imap_unordered(_prep_video, jobs):
                for c in CUES:
                    for k_, a in res[c].items():
                        out[c][k_][p] = a
                ext[p] = ex
                seen[p] = True
                tiles.update(tl)
        del _SH['L'], z
        log(f'  {lp.name}: {len(ids)} frames, {len(jobs)} videos ({time.time() - t0:.0f}s)')
        (Path(loc) / lp.name).unlink()
    assert seen.all(), f'{(~seen).sum()} frames not processed'
    od = OUT / 'mice'
    meta = {'n_frames': nF, 'n_valid': int(valid.sum())}
    w = F.w.values
    v2bits = P2['bits']
    for c in CUES:
        r = out[c]
        st = r['st'][valid].astype(np.float64)
        wv = w[valid][:, None]
        tot, insam, att, un = (st * wv).sum(0)
        b = r['bits']
        meta[c] = {
            'cue_px_per_frame_weighted': float(tot / w[valid].sum()),
            'share_cue_px_inside_SAM': float(insam / tot), 'share_cue_px_attached': float(att / tot),
            'share_cue_px_unattached': float(un / tot),
            'share_outside_SAM_cue_px_attached': float(att / max(att + un, 1)),
            'tail_px_per_mouse_median_valid': float(np.median(r['tail'][valid])),
            'on_patches_per_frame_v2': float((v2bits[valid] > 0).sum(1).mean()),
            'on_patches_per_frame_v3': float((b[valid] > 0).sum(1).mean()),
            'tail_only_patches_per_frame': float((r['tonly'][valid] > 0).sum(1).mean()),
            'patches_v2_on_now_off': int(((v2bits[valid] > 0) & (b[valid] == 0)).sum()),
            'v3_bits_superset_of_v2': bool(((v2bits[valid] & ~b[valid]) == 0).all()),
            'cluster_size_weighted': {str(k): float(w[valid & (r['clus'] == k)].sum() / w[valid].sum()) for k in range(5)},
            'cluster_size_weighted_v2': {str(k): float(w[valid & (P2['clus'] == k)].sum() / w[valid].sum()) for k in range(5)},
            'interaction_patches_per_frame': float(r['intr'][valid].sum(1).mean()),
            'frames_with_interaction_patch': float((w[valid] * r['intr'][valid].any(1)).sum() / w[valid].sum())}
        np.savez_compressed(od / f'prep_{c}.npz', bits=b, tonly=r['tonly'], dist=r['dist'], root=r['root'],
                            clus=r['clus'], intr=r['intr'], tail=r['tail'], st=r['st'], ext=ext)
        log(f'  {c}: {json.dumps(meta[c])}')
    meta['elapsed_s'] = time.time() - t0
    (od / 'prep.json').write_text(json.dumps(meta, indent=1, default=float))
    # tail sheets: 20 frames x 3 cues, 4 sheets of 5 frames (rows) x 3 cues (columns)
    ks = sorted(tiles)
    for s in range(4):
        sub = ks[5 * s:5 * s + 5]
        sh = Image.new('RGB', (len(CUES) * 512, 512 * len(sub)))
        for a, f in enumerate(sub):
            for b_, c in enumerate(CUES):
                sh.paste(Image.fromarray(tiles[f][c]), (b_ * 512, a * 512))
        sh.save(od / f'tail_sheet_{s}.jpg', quality=90)
    (od / 'tail_sheet_frames.json').write_text(json.dumps([int(x) for x in ks]))
    log(f'prep done ({time.time() - t0:.0f}s)')


def cmd_taildiag(args):
    """Diagnostic on the 20 tail-sheet frames: per mouse and body end, seed px, geodesic steps, grown px, width."""
    global DIAG
    loc = nm2.local_dir()
    F = load_frames()
    fr = json.loads((OUT / 'mice' / 'tail_sheet_frames.json').read_text())
    sam = load_sam(loc, F.tid.values[fr])
    Q = np.load(OUT / 'mice' / 'prep_line.npz')
    for f in fr:
        rgb = decode_frames(F.vname.values[f], [F.frame_idx.values[f]], loc)[int(F.frame_idx.values[f])]
        DIAG = []
        Le, tail, st = extend_line(sam[int(F.tid.values[f])], grey_of(rgb), pix_bg(F.obs.values[f]), Q['ext'][f])
        log(f'{F.obs.values[f]} f{F.frame_idx.values[f]} tail {tail.tolist()} | ' +
            ' '.join(f'm{k}e{e}:seed{a} st{b} px{c} w{d:.1f}' for k, e, a, b, c, d in DIAG))
    DIAG = None


# ---------------------------------------------------------------------------------------------- analyse (CPU)
POP = nm2.POP


class Data3(nm2.Data):
    """v2 codes / frames / weights, with the v3 mice geometry of one cue."""

    def __init__(self, key, cue):
        super().__init__(key)
        assert self.domain == 'mice'
        Q = np.load(OUT / 'mice' / f'prep_{cue}.npz')
        for k in Q.files:
            self.P[k] = Q[k]
        O = np.load(OUT / 'mice' / 'objects.npz')
        vids = sorted(self.F.obs.unique())
        assert list(O['obs']) == vids
        self.P['otype'] = O['otype']
        self.P['box'] = O['box']
        self.cue = cue


def frame_types(gb, roots):
    """gb (G,) OR-ed instance bits, roots (G, 4) cluster root per instance -> n (G,), n_clusters touched (G,), size of
    the touched cluster when one (G,)."""
    gb = gb.astype(np.int64) & 15
    n = POP[gb]
    r = np.where((gb[:, None] >> np.arange(4)[None]) & 1, roots.astype(np.int64), -1)
    rs = np.sort(r, 1)
    distinct = ((rs[:, 1:] != rs[:, :-1]) & (rs[:, 1:] >= 0)).sum(1) + (rs[:, 0] >= 0)
    first = np.where(rs >= 0, rs, 99).min(1)
    size = np.where(distinct == 1, (roots.astype(np.int64) == first[:, None]).sum(1), 0)
    return n, distinct, size


def group_stats3(D, tops):
    nF = D.nF
    P = D.P
    gl = np.concatenate([np.full(len(t), j, np.int64) for j, t in enumerate(tops)])
    gf = np.concatenate(tops).astype(np.int64)
    G = len(gl)
    gid_of = np.full(D.M * nF, -1, np.int64)
    gid_of[gl * nF + gf] = np.arange(G)
    g = gid_of[D.I * nF + D.fid_tok[D.tok_of]]
    sel = np.flatnonzero(g >= 0)
    g = g[sel]
    tok = D.tok_of[sel]
    f = D.fid_tok[tok]
    p = D.pos[tok]
    bits = P['bits'][f, p]
    on = bits > 0
    off = ~on
    dist = P['dist'][f, p].astype(np.int64)
    ot = P['otype'][P['vid'][f], p].astype(np.int64)
    obj = ot > 0
    it = P['intr'][f, p]
    tonly = P['tonly'][f, p]
    cnt = lambda m: np.bincount(g, weights=m.astype(np.float64), minlength=G)  # noqa: E731
    R = {'gl': gl, 'gf': gf, 'w': D.w[gf], 'valid': P['valid'][gf]}
    R['n_act'] = np.bincount(g, minlength=G).astype(np.float64)
    R['n_on'] = cnt(on)
    R['n_int'] = cnt(it)
    R['n_tailonly'] = cnt(on & (tonly == bits))
    R['n_halo'] = cnt(off & ~obj & (dist == 1))
    R['n_objnear'] = cnt(off & obj & (dist <= 2))
    R['n_mid'] = cnt(off & ~obj & (dist == 2))
    R['n_far'] = cnt(off & (dist >= 3))
    R['n_int_type'] = np.stack([cnt(it & (ot == k)) for k in range(len(OBJ_NAMES))], 1)
    R['n_obj_type'] = np.stack([cnt(ot == k) for k in range(len(OBJ_NAMES))], 1)
    gb = np.zeros(G, np.int64)
    np.bitwise_or.at(gb, g, bits.astype(np.int64))
    n, nclu, size = frame_types(gb, P['root'][gf])
    R['n'] = n.astype(np.float64)
    R['nclu'] = nclu
    R['csize'] = size
    R['ftype'] = np.where(n == 0, 'none', np.where(n == 1, 'self', np.where(nclu > 1, 'apart',
                          np.where(n == 2, 'pair', 'huddle'))))
    return R


def neuron_table3(D, FMT, tops, R, base_env, T2):
    M = D.M
    gl, wg, valid = R['gl'], R['w'], R['valid']
    wv = np.where(valid, wg, 0.0)
    sw = lambda x, ww=wv: np.bincount(gl, weights=ww * x, minlength=M)  # noqa: E731
    T = pd.DataFrame({'neuron': np.arange(M)})
    T['fire_frac'] = T2.fire_frac.values
    T['eligible'] = T2.eligible.values
    T['n_top'] = [len(t) for t in tops]
    act = sw(R['n_act'])
    T['on_share'] = sw(R['n_on']) / np.maximum(act, 1e-9)
    T['env_share'] = sw(R['n_int']) / np.maximum(act, 1e-9)
    T['tail_only_share'] = sw(R['n_tailonly']) / np.maximum(act, 1e-9)
    offv = sw(R['n_act'] - R['n_on'])
    for k in ('halo', 'objnear', 'mid', 'far'):
        T[f'off_{k}'] = sw(R[f'n_{k}']) / np.maximum(offv, 1e-9)
    for k, nm in enumerate(OBJ_NAMES):
        T[f'int_{nm}'] = sw(R['n_int_type'][:, k]) / np.maximum(sw(R['n_int']), 1e-9)
        T[f'act_on_{nm}'] = sw(R['n_obj_type'][:, k]) / np.maximum(act, 1e-9)
    T['env_object'] = np.array(OBJ_NAMES)[np.argmax(T[[f'int_{nm}' for nm in OBJ_NAMES]].values, 1)]
    T['n_med'] = nm2.wmedian(np.where(valid, R['n'], np.nan), wg, gl, M)
    ft = R['ftype']
    m2 = np.maximum(sw((R['n'] >= 2).astype(float)), 1e-9)
    for k in ('pair', 'huddle', 'apart'):
        T[f'{k}_share_n2'] = sw((ft == k).astype(float)) / m2
    vt = np.maximum(sw(np.ones(len(gl))), 1e-9)
    for k in ('none', 'self', 'pair', 'huddle', 'apart'):
        T[f'ft_{k}'] = sw((ft == k).astype(float)) / vt
    T['pair_in_huddle'] = sw(((ft == 'pair') & (R['csize'] >= 3)).astype(float)) / np.maximum(sw((ft == 'pair').astype(float)), 1e-9)
    has = np.isfinite(T.n_med.values)
    soc = np.where(T.n_med == 0, 'none', np.where(T.n_med == 1, 'self', np.where(
        T.apart_share_n2 >= APART_MIN, 'apart', np.where(T.n_med == 2, 'pair', 'huddle'))))
    T['social'] = np.where(has, soc, 'no-valid-top')
    thr = max(ENV_MIN, min(2 * base_env, (1 + base_env) / 2))
    T['env'] = has & (T.env_share >= thr).values
    offc = np.where(T.off_halo >= HALO_MIN, 'halo', np.where(T.off_far >= FAR_MIN, 'plain-background',
                    np.where(T.off_objnear >= OBJ_MIN, 'object-near', 'off-other')))
    cls = np.where(T.env, 'env-interaction', np.where(T.on_share >= ON_MIN, T.social, offc))
    T['cls'] = np.where(has, cls, 'no-valid-top')
    T['location'] = np.where(~has, 'no-valid-top', np.where(T.on_share >= ON_MIN, 'on-animal', 'off-animal'))
    # v2 (exclusive)
    v2 = np.where(T2.place_bound.values, 'place-bound', np.where(T2.location.values == 'on-animal', T2.social.values,
                                                               T2.location.values))
    T['v2_cls'] = v2
    T['v2_location'] = T2.location.values
    T['v2_social'] = T2.social.values
    T['v2_place_bound'] = T2.place_bound.values
    return T, thr


def cmd_analyse(args):
    key, cue = args.dict, args.cue
    t0 = time.time()
    D = Data3(key, cue)
    od = OUT / f'{key}_{cue}'
    od.mkdir(parents=True, exist_ok=True)
    FMT = D.frame_max()
    tops = nm2.top_sets(FMT, D.w)
    z = np.load(V2 / key / 'top_frames.npz')
    assert np.array_equal(np.concatenate(tops), z['flat']), 'top frames differ from v2'
    T2 = pd.read_csv(V2 / key / 'neurons.csv')
    R = group_stats3(D, tops)
    log(f'{key} {cue}: {len(R["gl"]):,} groups ({time.time() - t0:.0f}s)')
    # base rate of interaction patches over every SAE-mask patch of every valid frame
    wt = (D.w * D.P['valid'])[D.fid_tok]
    it_all = D.P['intr'][D.fid_tok, D.pos]
    base_env = float((wt * it_all).sum() / wt.sum())
    on_all = D.P['bits'][D.fid_tok, D.pos] > 0
    T, thr = neuron_table3(D, FMT, tops, R, base_env, T2)
    E = T[T.eligible]
    S = {'dict': key, 'cue': cue, 'n_eligible': int(len(E)), 'base_env': base_env, 'env_threshold': thr,
         'base_on_share_sae_patches': float((wt * on_all).sum() / wt.sum()),
         'counts_v3': E.cls.value_counts().to_dict(), 'counts_v2': E.v2_cls.value_counts().to_dict(),
         'env_by_object': E[E.cls == 'env-interaction'].env_object.value_counts().to_dict(),
         'social_on_animal_v3': E[E.location == 'on-animal'].social.value_counts().to_dict(),
         'confusion_v2_rows_v3_cols': nm2.crosstab(E.v2_cls, E.cls)}
    log(f'  counts v3 {S["counts_v3"]}; v2 {S["counts_v2"]}; base_env {base_env:.4f} thr {thr:.3f}')
    # anchors
    cols = ['cls', 'v2_cls', 'on_share', 'env_share', 'env_object', 'tail_only_share', 'n_med', 'apart_share_n2',
            'ft_self', 'ft_pair', 'ft_huddle', 'ft_apart', 'pair_in_huddle', 'off_halo', 'off_objnear', 'off_far',
            'fire_frac', 'eligible']
    rec = lambda j: {'neuron': int(j), **{c: (T[c][j].item() if hasattr(T[c][j], 'item') else T[c][j]) for c in cols}}  # noqa: E731
    S2 = json.loads((V2 / key / 'summary.json').read_text())
    anc = {'nose_contact (v2 AUROC picks)': [rec(e['neuron']) for e in S2['anchors']['nose_nose'][:2]] +
           [rec(e['neuron']) for e in S2['anchors']['nose_tail'][1:2]],
           'bag_like (v2: highest BAG share)': [rec(e['neuron']) for e in S2['anchors']['bag_like']],
           'tail (v3: highest tail-only share, eligible)': [rec(j) for j in
                                                           E.sort_values('tail_only_share', ascending=False).neuron.head(3)]}
    if key == 'fg448m5':
        anc['expected'] = {'915': 'pair', '180': 'env-interaction', 'tail': 'self'}
        rows = []
        for j, (c2, vd, why) in V2_JUDGED.items():
            rows.append({'neuron': j, 'v2_judged_cls': c2, 'v2_verdict': vd, 'v2_note': why, **rec(j)})
        S['v2_judged'] = rows
        J = pd.DataFrame(rows)
        S['v2_judged_crosstab'] = {f'{c2}|{vd}|{why}': g.cls.value_counts().to_dict()
                                   for (c2, vd, why), g in J.groupby(['v2_judged_cls', 'v2_verdict', 'v2_note'])}
    S['anchors'] = anc
    for nm, lst in anc.items():
        if isinstance(lst, list):
            log(f'  anchor {nm}: ' + ', '.join(f'{e["neuron"]} {e["cls"]} (v2 {e["v2_cls"]})' for e in lst))
    rng = np.random.default_rng(0)
    S['check_picks'] = {c: sorted(int(x) for x in rng.choice(E[E.cls == c].neuron.values,
                                                             min(N_CHECK, int((E.cls == c).sum())), replace=False))
                        for c in CLASSES if (E.cls == c).any()}
    T.to_csv(od / 'neurons.csv', index=False)
    S['elapsed_s'] = time.time() - t0
    (od / 'summary.json').write_text(json.dumps(S, indent=1, default=lambda x: x.item() if hasattr(x, 'item') else str(x)))
    log(f'{key} {cue}: analyse done ({time.time() - t0:.0f}s)')


# ---------------------------------------------------------------------------------------------- sheets (CPU)
OBJ_RGB = {2: (255, 0, 255), 3: (255, 0, 0), 4: (0, 230, 255)}


def _dec_job(job):
    v, ff, fidx, loc = job
    dec = decode_frames(v, fidx, loc)
    return ff, {int(f): dec[int(fi)] for f, fi in zip(ff, fidx)}


def render_tile3(img, act, val, L, Le, ext, otype, box, intr, title):
    """Left: full frame (256 px); right: 2x zoom on the footprint. SAM masks tinted, tails in full instance colour,
    object patches outlined (structure magenta, spout red, bag cyan; floor box green), body ends white dots, active
    patches green-orange boxes (white = argmax), interaction active patches with a cyan centre dot."""
    from PIL import Image, ImageDraw
    im = img.astype(np.float32).copy()
    for k in range(1, 5):
        m = L == k
        im[m] = 0.55 * im[m] + 0.45 * nm2.INST_RGB[k - 1]
        im[(Le == k) & (L == 0)] = nm2.INST_RGB[k - 1]
    pil = Image.fromarray(im.clip(0, 255).astype(np.uint8))
    dr = ImageDraw.Draw(pil)
    dr.rectangle(box, outline=(0, 200, 0), width=1)
    for q in np.flatnonzero(otype >= 2):
        r, c = divmod(int(q), GRID)
        dr.rectangle([c * PX + 1, r * PX + 1, c * PX + PX - 2, r * PX + PX - 2], outline=OBJ_RGB[int(otype[q])], width=1)
    vmax = max(float(val.max()), 1e-6)
    for q, v in zip(act, val):
        r, c = divmod(int(q), GRID)
        g = int(80 + 175 * v / vmax)
        dr.rectangle([c * PX, r * PX, c * PX + PX - 1, r * PX + PX - 1], outline=(g, 255 - g // 2, 0), width=2)
        if intr[q]:
            dr.ellipse([c * PX + 5, r * PX + 5, c * PX + 10, r * PX + 10], fill=(0, 255, 255))
    am = act[int(np.argmax(val))]
    r, c = divmod(int(am), GRID)
    dr.rectangle([c * PX - 2, r * PX - 2, c * PX + PX + 1, r * PX + PX + 1], outline=(255, 255, 255), width=2)
    for k in range(4):
        for e in range(2):
            y, x = ext[k, e]
            if np.isfinite(y):
                dr.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(255, 255, 255))
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
    key, cue = args.dict, args.cue
    D = Data3(key, cue)
    od_in = OUT / f'{key}_{cue}'
    S = json.loads((od_in / 'summary.json').read_text())
    T = pd.read_csv(od_in / 'neurons.csv')
    z = np.load(V2 / key / 'top_frames.npz')
    tops = [z['flat'][a:b] for a, b in zip(z['off'][:-1], z['off'][1:])]
    want = {}
    for c, js in S['check_picks'].items():
        for j in js:
            want.setdefault(int(j), []).append(c)
    for nm, lst in S['anchors'].items():
        if isinstance(lst, list):
            for e in lst:
                want.setdefault(int(e['neuron']), []).append(f'anchor:{nm.split(" ")[0]}')
    SF = nm2.sheet_frames(D, list(want), tops)
    need = sorted({int(f) for v in SF.values() for f in v})
    log(f'{key} {cue}: {len(want)} neurons, {len(need)} frames')
    loc = nm2.local_dir()
    F = load_frames()
    from multiprocessing import get_context
    imgs = {}
    jobs = [(v, g.index.values, g.frame_idx.values, str(loc)) for v, g in F.iloc[need].groupby('vname')]
    with get_context('fork').Pool(args.workers) as pool:
        for ff, dec in pool.imap_unordered(_dec_job, jobs):
            imgs.update(dec)
    sam = load_sam(loc, F.tid.values[need])
    oc = pd.read_csv(nm2.ODOR).set_index('observation_id')
    od = OUT / 'sheets' / f'{key}_{cue}'
    od.mkdir(parents=True, exist_ok=True)
    tst = np.r_[0, np.cumsum(D.lens)]
    pbc = {}
    geo = {}
    for f in need:
        o = F.obs.values[f]
        if o not in pbc:
            pbc[o] = pix_bg(o)
        L = sam[int(F.tid.values[f])]
        Le, _, _ = extend_any(L, grey_of(imgs[f]), pbc[o], D.P['ext'][f], cue)
        geo[f] = (L, Le)
        b, _ = patch_bits(L, Le)
        assert np.array_equal(b, D.P['bits'][f]), f'bits of frame {f} not reproduced'
    index = []
    for j, frs in SF.items():
        tiles = []
        for f in frs:
            act, val = [], []
            for tk in range(tst[f], tst[f + 1]):
                lo, hi = D.indptr[tk], D.indptr[tk + 1]
                m = D.I[lo:hi] == j
                if m.any():
                    act.append(D.pos[tk])
                    val.append(D.V[lo:hi][m][0])
            act, val = np.array(act), np.array(val)
            bits = D.P['bits'][f]
            gb = np.bitwise_or.reduce(bits[act].astype(np.int64))
            n, ncl, size = frame_types(np.array([gb]), D.P['root'][f][None])
            ft = 'none' if n[0] == 0 else 'self' if n[0] == 1 else 'apart' if ncl[0] > 1 else 'pair' if n[0] == 2 else 'huddle'
            o = F.obs.values[f]
            bx = D.P['box'][D.P['vid'][f]]
            L, Le = geo[f]
            tiles.append(render_tile3(imgs[f], act, val, L, Le, D.P['ext'][f], D.P['otype'][D.P['vid'][f]],
                                      [bx[2], bx[0], bx[3], bx[1]], D.P['intr'][f],
                                      f'{o} f{F.frame_idx.values[f]} w{D.w[f]:.0f} act {len(act)} on '
                                      f'{int((bits[act] > 0).sum())} int {int(D.P["intr"][f][act].sum())} | n {n[0]} {ft}'))
        cols = 3
        rows = int(np.ceil(len(tiles) / cols))
        sheet = Image.new('RGB', (cols * 512, rows * 276 + 40), (255, 255, 255))
        for i, t_ in enumerate(tiles):
            sheet.paste(t_, ((i % cols) * 512, 40 + (i // cols) * 276))
        t = T.loc[j]
        head = (f'{key} n{j} [{", ".join(want[j])}] V3: {t.cls} | on {t.on_share:.2f} env {t.env_share:.2f} '
                f'({t.env_object}) tail-only {t.tail_only_share:.2f} n_med {t.n_med} pair/huddle/apart(n>=2) '
                f'{t.pair_share_n2:.2f}/{t.huddle_share_n2:.2f}/{t.apart_share_n2:.2f} off halo/obj/far '
                f'{t.off_halo:.2f}/{t.off_objnear:.2f}/{t.off_far:.2f} | V2: {t.v2_cls}')
        ImageDraw.Draw(sheet).text((6, 6), head, fill=(0, 0, 0))
        ImageDraw.Draw(sheet).text((6, 22), 'tint = SAM masks, full colour = attached tails, white dots = body ends; '
                                   'outlines: magenta structure, red spout, cyan bag, green floor box; boxes = active '
                                   'patches (white = argmax), cyan dot = interaction patch', fill=(80, 80, 80))
        name = f'n{j:04d}.jpg'
        sheet.save(od / name, quality=85)
        index.append({'neuron': int(j), 'why': want[j], 'cls': t.cls, 'v2_cls': t.v2_cls,
                      'sheet': str((od / name).relative_to(REPO)), 'frames': [int(x) for x in frs]})
    (od / 'index.json').write_text(json.dumps(index, indent=1))
    log(f'{key} {cue}: {len(index)} sheets -> {od}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['look', 'objects', 'prep', 'taildiag', 'analyse', 'sheets'])
    ap.add_argument('--dict', choices=['fg448m5', 'fg448al'])
    ap.add_argument('--cue', default='dark', choices=list(CUES))
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()
    globals()[f'cmd_{args.step}'](args)


if __name__ == '__main__':
    main()
