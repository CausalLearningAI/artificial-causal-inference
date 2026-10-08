"""
Where in the frame do the mice contact neurons' phase-switch reactions come from? And does masking that region make
the neurons track behaviour at the video level? (ECI diagnostic, mice v1; follow-up of scripts/eci/diag_video_level.py,
commit 448a8bd.)

Units, frames, windows: the 172,800 eval frames (1 fps) of the 144 annotated videos (24 pools x 6 stages) that
scripts/eci/sae_levers.py scored (results/vision/eci_sae_levers/mice/labels.parquet, store order). Every per-video
number is a mean over the frames inside the NES window (MiceDomain.window_start: habituation = its last 15 min, odour /
post whole; 900 eval frames per video). Switches: social H->O (1->2), O->P (2->3), fear H->O (4->5), O->P (5->6).

Neurons (per-frame value = max over the frame's foreground tokens, as codes_best / codes_max):
    new SAEs (fg448 store = unaligned frames; results/vision/eci_sae_levers/mice/sae/): w4096+cell s0 n351, s1 n3187,
        s2 n629 and w4096 s0 n104 (nose-nose AUROC picks, the same neuron on both cross-fit halves), plus s1 n425 (the
        s1 nose-tail pick). 'cell' centring = token - mean token of the same (video, grid position) over the video's
        eval frames (src/eci/levers.py CellMeans, min 20 tokens, else the per-video mean), exactly as sae_levers did.
    deployed fg448al SAE (odour-corner-aligned store): n611 (nose-nose), n414 (nose-tail), n49 (the odour-bag neuron,
        POSITIVE CONTROL for the region analysis).

Steps
  prep (CPU)    BAG location per recording session (pool x odour = its H, O, P videos, one camera rotation): from the
                per-video empty-arena backgrounds (dataset/mice/v1/eci/fg448/background/<obs>.npz, bg_masked = per-position
                median DINOv2 token over 200 frames, mice masked out), D_O = cosine distance between the O video's and
                the H video's background token at each of the 32 x 32 positions. Label-free: no behaviour label, only the
                repo's odour corner (dataset/mice/v1/eci/odor_corner.csv, from the water-spout position) to pick the image
                quadrant. Bag core = the 8-connected component, inside the odour-corner quadrant, of cells with
                D_O >= 0.5 x the quadrant max that contains the quadrant argmax. BAG zone = core dilated by BAG_MARGIN
                (2) patches (also 1 and 0 as sensitivity); the same zone is used for the session's H, O and P videos.
                Specificity check: the same max in the diagonally opposite quadrant.
                Regions per video (raw camera frame, arena box of odor_corner.csv): BAG; WALL = inside the arena box,
                patch centre < 2 patches from its edge; OUTSIDE = patch centre outside the arena box; CENTRE = the rest.
                Mouse-like patch: the foreground rule's own dark cue (src/eci/foreground.py dark_fraction: > 15% of the
                patch's 256 pixels have grey < 60 and are > 40 darker than the video's pixel background pix_bg), from
                the frame jpg; computed for every eval frame. Sanity image of 4 sessions.
  encode (GPU)  per-token codes of the candidate neurons on the eval tokens of both stores; checks that the per-frame max
                reproduces codes_best (new SAEs) / codes_max.npy (deployed) and that CellMeans reproduces sae_levers.
  analyse (CPU) for each neuron: (1) decomposition of the per-video mean of the frame max into the region of the
                frame's argmax token (exactly additive), per switch, in dz units: the change d (later - earlier, per
                pool; each half's per-video values divided by their sd over that half's 72 videos, as
                diag_video_level.py) is split into d_BAG + d_WALL + d_CENTRE + d_OUTSIDE, so
                dz = mean(d) / sd(d) = sum_r mean(d_r) / sd(d); share_r = mean(d_r) / mean(d). Also split by mouse-like.
                (2) foreground patch count per region per frame, its change per switch. (3) masked re-reads: frame max
                over tokens outside BAG (noBAG, margin 2; noBAG1, margin 1), over mouse-like tokens only (mouse), and
                both (mouse_noBAG), re-scored with the diag_video_level.py measurements (held-out halves; here the
                neuron is the same on both halves). (4) correlates of the per-pool change: annotated rate / bouts, fg
                counts, number of foreground components.
Pre-registered readings (per candidate): 'bag-driven' iff BAG share >= 50% at social O->P AND social H->O; 'masking
fixes it' iff for a masked read the social O->P dz is within +-0.5 of the annotated rate's dz AND the held-out video
r (rate) rises by >= 0.15 over the unmasked read.

Output: results/vision/eci_repr_diag/mice/switch_regions/ (prep/, tokens/, results.json, table.md, sanity_bag.png)
Usage: scripts/eci/diag_switch_regions.sh (STEP=prep | encode | analyse)
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
from diag_video_level import SWITCHES, corr, fmt_p, ols_intercept, ttest  # noqa: E402

OUT = REPO / 'results/vision/eci_repr_diag/mice/switch_regions'
LEV = REPO / 'results/vision/eci_sae_levers/mice'
VL = REPO / 'results/vision/eci_repr_diag/mice/video_level'
BG = REPO / 'dataset/mice/v1/eci/fg448/background'
ODOR = REPO / 'dataset/mice/v1/eci/odor_corner.csv'
ANN = REPO / 'dataset/mice/v1/annotations.csv'
STORES = {'raw': 'dataset/mice/v1/eci/train_tokens/dinov2_base_l-1_fg448_fps1',
          'al': 'dataset/mice/v1/eci/train_tokens/dinov2_base_l-1_fg448al_fps1'}
DEP_SAE = REPO / 'dataset/mice/v1/eci/sae/matryoshka_btk_1024_k16_fg448al_s0/sae.pt'
DEP_MAX = REPO / 'dataset/mice/v1/eci/codes/matryoshka_btk_1024_k16_fg448al_s0/codes_max.npy'
GRID, PX = 32, 16
REG = ('BAG', 'WALL', 'CENTRE', 'OUTSIDE')
WALL_W = 2  # patches
BAG_MARGIN = 2
MARGINS = (0, 1, 2)
DARK_ABS, DARK_REL = 60, 40
DARK_MIN = 39  # dark pixels of 256 per patch: fraction > 0.15 <=> count >= 39
ODOR_ROT90 = {'TR': 0, 'BR': 1, 'BL': 2, 'TL': 3}  # src/eci/foreground.py
# model key -> (store, checkpoint, centring, neurons)
MODELS = {'w4096+cell_s0': ('raw', LEV / 'sae/w4096+cell_s0/sae.pt', 'cell', [351]),
          'w4096+cell_s1': ('raw', LEV / 'sae/w4096+cell_s1/sae.pt', 'cell', [3187, 425]),
          'w4096+cell_s2': ('raw', LEV / 'sae/w4096+cell_s2/sae.pt', 'cell', [629]),
          'w4096_s0': ('raw', LEV / 'sae/w4096_s0/sae.pt', None, [104]),
          'deployed': ('al', DEP_SAE, None, [611, 414, 49])}
# candidate name -> (model, neuron, behaviour, per_video.parquet reference column (unmasked per-video mean))
CANDS = {'cell_s0 n351': ('w4096+cell_s0', 351, 'nose_nose', 'w4096+cell_s0|auc|nose_nose|test0'),
         'cell_s1 n3187': ('w4096+cell_s1', 3187, 'nose_nose', 'w4096+cell_s1|auc|nose_nose|test0'),
         'cell_s2 n629': ('w4096+cell_s2', 629, 'nose_nose', 'w4096+cell_s2|auc|nose_nose|test0'),
         'w4096_s0 n104': ('w4096_s0', 104, 'nose_nose', 'w4096_s0|auc|nose_nose|test0'),
         'cell_s1 n425 (nose-tail)': ('w4096+cell_s1', 425, 'nose_tail', 'w4096+cell_s1|auc|nose_tail|test0'),
         'deployed n611': ('deployed', 611, 'nose_nose', 'deployed_max_1fps|nose_nose'),
         'deployed n414 (nose-tail)': ('deployed', 414, 'nose_tail', 'deployed_max_1fps|nose_tail'),
         'deployed n49 (bag control)': ('deployed', 49, 'nose_nose', None)}
NEW = ('cell_s0 n351', 'cell_s1 n3187', 'cell_s2 n629', 'w4096_s0 n104')
VARIANTS = ('all', 'noBAG', 'noBAG1', 'mouse', 'mouse_noBAG', 'noBAGG')
SW = list(SWITCHES)


def log(m):
    print(time.strftime('%H:%M:%S'), m, flush=True)


def design():
    """The 144 annotated videos in per_video.parquet order (pool, stage, half, annotated rates / bouts per minute)."""
    pv = pd.read_parquet(VL / 'per_video.parquet')
    A = pv[pv.annotated].reset_index(drop=True)
    assert len(A) == 144 and (A.groupby('pool').size() == 6).all()
    A['session'] = A.observation_id.str[:-2]
    return A


def cosdist(a, b):
    a, b = a.astype(np.float32), b.astype(np.float32)
    return 1 - (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def quadrant(corner):
    q = np.zeros((GRID, GRID), bool)
    rs = slice(0, 16) if corner[0] == 'T' else slice(16, 32)
    cs = slice(0, 16) if corner[1] == 'L' else slice(16, 32)
    q[rs, cs] = True
    return q


OPPOSITE = {'TR': 'BL', 'BL': 'TR', 'TL': 'BR', 'BR': 'TL'}


def region_map(bag, box):
    """bag (32, 32) bool, box (r0, r1, c0, c1) px -> (1024,) int8 region index into REG."""
    r0, r1, c0, c1 = box
    c = (np.arange(GRID) + 0.5) * PX
    yy, xx = np.meshgrid(c, c, indexing='ij')
    inside = (yy >= r0) & (yy <= r1) & (xx >= c0) & (xx <= c1)
    edge = np.minimum.reduce([yy - r0, r1 - yy, xx - c0, c1 - xx])
    reg = np.full((GRID, GRID), 2, np.int8)
    reg[~inside] = 3
    reg[inside & (edge < WALL_W * PX)] = 1
    reg[bag] = 0
    return reg.ravel()


def dark_counts(grey, pix_bg):
    """grey (512, 512) uint8, pix_bg (512, 512) uint8 -> (1024,) uint8 number of dark pixels per patch (clipped 255)."""
    g = grey.astype(np.int16)
    d = (g < DARK_ABS) & ((pix_bg.astype(np.int16) - g) > DARK_REL)
    return np.minimum(d.reshape(GRID, PX, GRID, PX).sum((1, 3)), 255).astype(np.uint8).ravel()


def _grey(path):
    from PIL import Image
    with Image.open(path) as im:
        return np.asarray(im.convert('RGB').convert('L'), dtype=np.uint8)


def _dark_video(job):
    obs, paths = job
    pb = np.load(BG / f'{obs}.npz')['pix_bg']
    return obs, np.stack([dark_counts(_grey(REPO / 'dataset' / p), pb) for p in paths])


# ---------------------------------------------------------------------------------------------- prep (CPU)
def cmd_prep(args):
    from multiprocessing import get_context

    from scipy import ndimage
    stg = Path(args.stage)
    stg.mkdir(parents=True, exist_ok=True)
    out = OUT / 'prep'
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    A = design()
    lab = pd.read_parquet(LEV / 'labels.parquet')
    shutil.copyfile(ANN, stg / 'ann.csv')
    ann = pd.read_csv(stg / 'ann.csv', usecols=['observation_id', 'frame_path'])
    assert (ann.observation_id.values[lab.row.values] == lab.obs.values).all()
    paths = ann.frame_path.values[lab.row.values]
    del ann
    oc = pd.read_csv(ODOR).set_index('observation_id')
    log(f'design {len(A)} videos, {A.session.nunique()} sessions; {len(lab):,} eval frames ({time.time() - t0:.0f}s)')
    # ------------------------------------------------ bag zone per session
    sessions = sorted(A.session.unique())
    S, Dmaps, cores = [], {}, {}
    pix = {}
    for s in sessions:
        z = {ph: np.load(BG / f'{s}_{ph}.npz') for ph in 'HOP'}
        bgm = {ph: z[ph]['bg_masked'] for ph in 'HOP'}
        pix[s] = {ph: z[ph]['pix_bg'] for ph in 'HOP'}
        corners = {ph: oc.loc[f'{s}_{ph}', 'odor_corner'] for ph in 'HOP'}
        assert len(set(corners.values())) == 1, (s, corners)
        corner = corners['O']
        DO = cosdist(bgm['O'], bgm['H']).reshape(GRID, GRID)
        DP = cosdist(bgm['P'], bgm['H']).reshape(GRID, GRID)
        DPO = cosdist(bgm['P'], bgm['O']).reshape(GRID, GRID)
        q, qo = quadrant(corner), quadrant(OPPOSITE[corner])
        m = np.where(q, DO, -1)
        amax = np.unravel_index(int(np.argmax(m)), m.shape)
        cand = q & (DO >= 0.5 * m.max())
        lab_, _ = ndimage.label(cand, structure=np.ones((3, 3)))
        core = lab_ == lab_[amax]
        Dmaps[s] = np.stack([DO, DP, DPO])
        cores[s] = core
        r = oc.loc[f'{s}_O']
        box = (r.arena_r0, r.arena_r1, r.arena_c0, r.arena_c1)
        c = (np.arange(GRID) + 0.5) * PX
        yy, xx = np.meshgrid(c, c, indexing='ij')
        inside = (yy >= box[0]) & (yy <= box[1]) & (xx >= box[2]) & (xx <= box[3])
        cy, cx = np.argwhere(core).mean(0) + 0.5
        S.append({'session': s, 'corner': corner, 'confidence_O': r.confidence, 'bag_visible_O': bool(r.bag_visible),
                  'DO_max_corner_quadrant': float(m.max()), 'DO_max_opposite_quadrant': float(DO[qo].max()),
                  'DO_q90_frame': float(np.quantile(DO, 0.9)), 'core_cells': int(core.sum()),
                  'core_frac_outside_arena': float((core & ~inside).sum() / core.sum()),
                  'core_centroid_patch': [float(cy), float(cx)],
                  'core_dist_to_arena_corner_patch': float(np.hypot(cy - r.corner_y_patch, cx - r.corner_x_patch)),
                  'DO_core_mean': float(DO[core].mean()), 'DP_core_mean': float(DP[core].mean()),
                  'DPO_core_mean': float(DPO[core].mean()),
                  'pix_core_mean_HOP': [float(pix[s][ph].reshape(GRID, PX, GRID, PX).mean((1, 3))[core].mean())
                                        for ph in 'HOP']})
    Sdf = pd.DataFrame(S)
    log(f'bag cores: cells median {Sdf.core_cells.median():.0f} [{Sdf.core_cells.min()}..{Sdf.core_cells.max()}], '
        f'corner-quadrant max D_O median {Sdf.DO_max_corner_quadrant.median():.3f} vs opposite '
        f'{Sdf.DO_max_opposite_quadrant.median():.3f} (corner > opposite in '
        f'{(Sdf.DO_max_corner_quadrant > Sdf.DO_max_opposite_quadrant).sum()}/{len(Sdf)}), dist to arena corner median '
        f'{Sdf.core_dist_to_arena_corner_patch.median():.2f} patches, outside arena {Sdf.core_frac_outside_arena.mean():.2f}')
    # ------------------------------------------------ region maps per video
    regs = {mg: np.zeros((len(A), GRID * GRID), np.int8) for mg in MARGINS}
    rot = np.zeros(len(A), np.int8)
    for i, (o, s) in enumerate(zip(A.observation_id, A.session)):
        r = oc.loc[o]
        box = (r.arena_r0, r.arena_r1, r.arena_c0, r.arena_c1)
        for mg in MARGINS:
            bag = ndimage.binary_dilation(cores[s], structure=np.ones((2 * mg + 1,) * 2)) if mg else cores[s]
            regs[mg][i] = region_map(bag, box)
        rot[i] = ODOR_ROT90[r.odor_corner]
    cnt = {mg: {REG[k]: float((regs[mg] == k).sum(1).mean()) for k in range(4)} for mg in MARGINS}
    log(f'mean cells per region: {json.dumps(cnt)}')
    # ------------------------------------------------ dark counts per eval frame
    jobs = [(o, paths[(lab.obs == o).values]) for o in A.observation_id]
    order = {o: np.flatnonzero((lab.obs == o).values) for o in A.observation_id}
    dark = np.zeros((len(lab), GRID * GRID), np.uint8)
    with get_context('fork').Pool(args.workers) as p:
        for k, (o, d) in enumerate(p.imap_unordered(_dark_video, jobs)):
            dark[order[o]] = d
            if k % 24 == 0:
                log(f'  dark counts {k + 1}/{len(jobs)} videos ({time.time() - t0:.0f}s)')
    # check against the background files' own dark fractions (their 200 sample frames) for 4 videos
    ann = pd.read_csv(stg / 'ann.csv', usecols=['frame_path'])['frame_path'].values
    chk = {}
    for o in A.observation_id.iloc[[0, 37, 75, 140]]:
        z = np.load(BG / f'{o}.npz')
        mine = np.stack([dark_counts(_grey(REPO / 'dataset' / ann[rw]), z['pix_bg']) for rw in z['rows'][:40]])
        ref = z['dark'][:40].astype(np.float32)
        chk[o] = {'max_abs_frac_diff': float(np.abs(np.minimum(mine, 255) / 256 - ref).max()),
                  'core_agree': float(((mine >= DARK_MIN) == (ref > 0.15)).mean())}
    log(f'dark recompute vs background files (40 sample frames each): {json.dumps(chk)}')
    np.save(stg / 'dark_count.npy', dark)
    np.savez(stg / 'regions.npz', obs=A.observation_id.values.astype(str), rot=rot,
             **{f'reg_m{mg}': regs[mg] for mg in MARGINS}, sessions=np.array(sessions),
             D=np.stack([Dmaps[s] for s in sessions]), core=np.stack([cores[s] for s in sessions]))
    meta = {'sessions': S, 'cells_per_region': cnt, 'dark_check': chk,
            'mouse_like_patch_frac_per_frame': float((dark >= DARK_MIN).sum(1).mean()),
            'rule': {'bag_core': 'component of D_O >= 0.5 x corner-quadrant max containing the argmax',
                     'D_O': 'cosine distance of bg_masked tokens, O vs H video of the session', 'wall_width_patches': WALL_W,
                     'dark': f'grey < {DARK_ABS} and pix_bg - grey > {DARK_REL}; mouse-like = count >= {DARK_MIN}/256'}}
    (stg / 'prep.json').write_text(json.dumps(meta, indent=1, default=float))
    # ------------------------------------------------ sanity image (4 sessions)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pick = []
    for cn in ('TR', 'BL'):
        sub = Sdf[Sdf.corner == cn]
        for vis in (True, False):
            v = sub[sub.bag_visible_O == vis]
            if len(v):
                pick.append(v.session.iloc[0])
    pick = (pick + [s for s in sessions if s not in pick])[:4]
    fig, ax = plt.subplots(4, 5, figsize=(20, 16.5))
    for i, s in enumerate(pick):
        o = f'{s}_O'
        vi = int(np.flatnonzero(A.observation_id.values == o)[0])
        reg = regs[BAG_MARGIN][vi].reshape(GRID, GRID)
        fr = order[o][len(order[o]) // 2]
        im = _grey(REPO / 'dataset' / paths[fr])
        panels = [(pix[s]['H'], 'pix_bg H'), (pix[s]['O'], 'pix_bg O'), (pix[s]['P'], 'pix_bg P'),
                  (Dmaps[s][0], 'D_O = cos dist bg tokens O vs H'), (im, f'O frame {lab.frame_idx.iat[fr]}')]
        for j, (img, t) in enumerate(panels):
            a = ax[i, j]
            if j == 3:
                a.imshow(img, cmap='magma', vmin=0, vmax=0.8, extent=(0, 512, 512, 0))
            else:
                a.imshow(img, cmap='gray', vmin=0, vmax=255)
            a.contour(np.kron(reg == 0, np.ones((PX, PX))), levels=[0.5], colors='red', linewidths=1.5)
            a.contour(np.kron(cores[s], np.ones((PX, PX))), levels=[0.5], colors='cyan', linewidths=1)
            a.contour(np.kron(reg == 1, np.ones((PX, PX))), levels=[0.5], colors='dodgerblue', linewidths=0.8)
            if j == 4:
                dk = (dark[fr] >= DARK_MIN).reshape(GRID, GRID)
                a.contour(np.kron(dk, np.ones((PX, PX))), levels=[0.5], colors='lime', linewidths=1)
            r = Sdf.set_index('session').loc[s]
            a.set_title(f'{s} {t}' + (f'\ncorner {r.corner}, bag_visible {r.bag_visible_O}, core {r.core_cells} cells'
                                       if j == 0 else ''), fontsize=8)
            a.axis('off')
    fig.suptitle('BAG zone (red: core dilated 2 patches; cyan: core), WALL band (blue), mouse-like dark patches (green)')
    fig.tight_layout()
    fig.savefig(stg / 'sanity_bag.png', dpi=70)
    for f in ('dark_count.npy', 'regions.npz', 'prep.json'):
        shutil.copyfile(stg / f, out / f)
    shutil.copyfile(stg / 'sanity_bag.png', OUT / 'sanity_bag.png')
    log(f'prep done -> {out} ({time.time() - t0:.0f}s)')


# ---------------------------------------------------------------------------------------------- encode (GPU)
def cmd_encode(args):
    import torch

    import multiscale_sae as ms
    import spatial_sae_pilot as ssp
    from src.eci.levers import CellMeans
    from src.eci.sae import load_sae
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False
    loc = Path(args.stage)
    lout = loc / 'out'
    lout.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    lab = pd.read_parquet(LEV / 'labels.parquet')
    checks = {}
    for store in ('raw', 'al'):
        ssp.STORES['mice'] = STORES[store]
        idx = ssp.StoreIndex('mice')
        _, ev, _, _, _, _ = idx.split('mice', ms.N_TRAIN['mice'], 0)
        ev = np.sort(ev)
        assert len(ev) == len(lab) and (idx.rows[ev] == lab.row.values).all(), f'{store}: eval frames differ from labels'
        tok_m, pos, lens = ms.stage(idx, ev, loc / 'tok')
        del tok_m
        tok = np.load(loc / 'tok' / 'tok.npy')
        shutil.rmtree(loc / 'tok')
        log(f'{store}: {len(ev):,} frames, {len(tok):,} tokens in RAM ({time.time() - t0:.0f}s)')
        vids, fvid = ms_video_index(lab.obs.values)
        tvid = np.repeat(fvid, lens)
        keys = [k for k, m in MODELS.items() if m[0] == store]
        cm = None
        if any(MODELS[k][2] == 'cell' for k in keys):
            cm = CellMeans(len(vids), tok.shape[1], dev)
            for a in range(0, len(tok), 2_000_000):
                b = min(len(tok), a + 2_000_000)
                cm.add(torch.from_numpy(tok[a:b]).to(dev), torch.from_numpy(tvid[a:b]).to(dev),
                       torch.from_numpy(pos[a:b].astype(np.int64)).to(dev))
            cm.finalize(20)
            ref = json.loads((LEV / 'eval_meta_w4096+cell.json').read_text())['variance']['cell_means']
            checks['cell_means'] = {'mine': cm.stats, 'sae_levers': ref, 'equal': cm.stats == ref}
            log(f'  CellMeans {cm.stats} (sae_levers {ref})')
        cols, Z = [], []
        for k in keys:
            _, pth, mode, J = MODELS[k]
            sae, norm, ck = load_sae(pth, dev)
            Jt = torch.as_tensor(J, device=dev)
            We, be, th = sae.W_enc[:, Jt], sae.b_enc[Jt], sae.threshold
            z_ = np.empty((len(tok), len(J)), np.float16)
            with torch.no_grad():
                for a in range(0, len(tok), 500_000):
                    b = min(len(tok), a + 500_000)
                    T = torch.from_numpy(tok[a:b]).to(dev)
                    if mode:
                        T = cm.center(T, torch.from_numpy(tvid[a:b]).to(dev),
                                      torch.from_numpy(pos[a:b].astype(np.int64)).to(dev), mode)
                    pre = torch.relu((norm(T) - sae.b_dec) @ We + be)
                    z_[a:b] = (pre * (pre > th)).half().cpu().numpy()
            cols += [(k, j) for j in J]
            Z.append(z_)
            log(f'  {k} neurons {J} encoded ({time.time() - t0:.0f}s)')
        Z = np.concatenate(Z, 1)
        # per-frame max vs the stored frame codes
        nz = lens > 0
        st = np.r_[0, np.cumsum(lens)][:-1]
        fm = np.zeros((len(lens), Z.shape[1]), np.float32)
        fm[nz] = np.maximum.reduceat(Z.astype(np.float32), st[nz], axis=0)
        if store == 'al':
            shutil.copyfile(DEP_MAX, loc / 'codes_max.npy')
            X = np.load(loc / 'codes_max.npy', mmap_mode='r')
            rows_, jj = lab.row.values, [j for _, j in cols]
            refm = np.zeros_like(fm)
            for a in range(0, len(rows_), 20_000):
                refm[a:a + 20_000] = np.asarray(X[rows_[a:a + 20_000]], np.float32)[:, jj]
            del X
            (loc / 'codes_max.npy').unlink()
        else:
            refm = np.zeros_like(fm)
            for c, (k, j) in enumerate(cols):
                zb = np.load(LEV / 'codes_best' / f'{k}.npz')
                refm[:, c] = zb['codes'][:, list(zb['neurons']).index(j)].astype(np.float32)
        for c, (k, j) in enumerate(cols):
            d = np.abs(fm[:, c] - refm[:, c])
            checks[f'{k}|{j}'] = {'store': store, 'reference': 'codes_max.npy (5 fps encode)' if store == 'al' else
                                  'codes_best', 'max_abs_diff': float(d.max()),
                                  'frac_equal_fp16': float((fm[:, c].astype(np.float16) == refm[:, c].astype(np.float16)).mean()),
                                  'frac_within_1pct': float((d <= 0.01 * np.maximum(np.abs(refm[:, c]), 1e-3) + 1e-6).mean()),
                                  'pearson': float(np.corrcoef(fm[:, c], refm[:, c])[0, 1]),
                                  'nonzero_frac_mine_ref': [float((fm[:, c] > 0).mean()), float((refm[:, c] > 0).mean())],
                                  'mean_mine_ref': [float(fm[:, c].mean()), float(refm[:, c].mean())]}
            log(f'  frame-max check {k} n{j}: {json.dumps(checks[f"{k}|{j}"])}')
        np.save(lout / f'{store}_codes.npy', Z)
        np.save(lout / f'{store}_pos.npy', pos.astype(np.int16))
        np.save(lout / f'{store}_lens.npy', lens.astype(np.int32))
        (lout / f'{store}_cols.json').write_text(json.dumps(cols))
        del tok, Z
    (lout / 'encode_checks.json').write_text(json.dumps(checks, indent=1, default=str))
    (OUT / 'tokens').mkdir(parents=True, exist_ok=True)
    for f in lout.iterdir():
        shutil.copyfile(f, OUT / 'tokens' / f.name)
    log(f'encode done -> {OUT / "tokens"} ({time.time() - t0:.0f}s)')


def ms_video_index(obs):
    vids = sorted(set(obs))
    vmap = {v: i for i, v in enumerate(vids)}
    return vids, np.array([vmap[v] for v in obs], np.int64)


# ---------------------------------------------------------------------------------------------- analyse (CPU)
def video_metrics(x, A, b):
    """diag_video_level.py measurements for a per-video array x (A order): held-out halves video r, pooled d-r,
    per-switch effect tests (each half's values / their sd over that half's 72 videos), plus the standardised
    per-pool changes (for the decomposition)."""
    pools = sorted(A.pool.unique())
    ph = A.groupby('pool').half.first()
    ix = {(p, s): i for i, (p, s) in enumerate(zip(A.pool, A.stage))}
    rate, bpm = A[f'{b}_rate'].values, A[f'{b}_bpm'].values

    def deltas(v, pl):
        return {sw: np.array([v[ix[(p, bb)]] - v[ix[(p, aa)]] for p in pl]) for sw, (aa, bb) in SWITCHES.items()}
    H, dn_all, dr_all, db_all = {}, {sw: [] for sw in SW}, {sw: [] for sw in SW}, {sw: [] for sw in SW}
    cat = lambda d_: np.concatenate([d_[sw] for sw in SW])  # noqa: E731
    sds = {}
    for h in (0, 1):
        m = (A.half == h).values
        pl = [p for p in pools if ph[p] == h]
        dn, dr, db = deltas(x, pl), deltas(rate, pl), deltas(bpm, pl)
        H[h] = {'video_rate': corr(x[m], rate[m])['pearson'], 'video_bpm': corr(x[m], bpm[m])['pearson'],
                'delta_rate_pooled': corr(cat(dn), cat(dr))['pearson'], 'delta_bpm_pooled': corr(cat(dn), cat(db))['pearson']}
        sds[h] = np.std(x[m], ddof=1)
        for sw in SW:
            dn_all[sw].append(dn[sw] / sds[h])
            dr_all[sw].append(dr[sw])
            db_all[sw].append(db[sw])
    mh = lambda k: float(np.mean([H[h][k] for h in (0, 1)]))  # noqa: E731
    out = {'video_r_rate': mh('video_rate'), 'video_r_bpm': mh('video_bpm'), 'delta_r_pooled': mh('delta_rate_pooled'),
           'delta_r_pooled_bpm': mh('delta_bpm_pooled'), 'halves': H, 'effects': {}}
    for sw in SW:
        n_, r_, b_ = ttest(np.concatenate(dn_all[sw])), ttest(np.concatenate(dr_all[sw])), ttest(np.concatenate(db_all[sw]))
        out['effects'][sw] = {'neuron': n_, 'rate': r_, 'bpm': b_, 'sign_agree_rate': bool(np.sign(n_['mean']) == np.sign(r_['mean'])),
                              'sign_agree_bpm': bool(np.sign(n_['mean']) == np.sign(b_['mean']))}
    return out, sds, (pools, ph, ix)


def std_deltas(v, A, sds, pools, ph, ix):
    """per switch: the per-pool change of v (A order), each half divided by sds[half] -> {sw: (24,)} (half 0 pools first)."""
    o = {}
    for sw, (aa, bb) in SWITCHES.items():
        o[sw] = np.concatenate([np.array([v[ix[(p, bb)]] - v[ix[(p, aa)]] for p in pools if ph[p] == h]) / sds[h]
                                for h in (0, 1)])
    return o


def plain_deltas(v, pools, ix):
    return {sw: np.array([v[ix[(p, bb)]] - v[ix[(p, aa)]] for p in pools]) for sw, (aa, bb) in SWITCHES.items()}


def cmd_analyse(args):
    from src.eci.domain import MiceDomain
    from src.eci.levers import frame_components
    t0 = time.time()
    stg = Path(args.stage)
    stg.mkdir(parents=True, exist_ok=True)
    A = design()
    lab = pd.read_parquet(LEV / 'labels.parquet')
    F = len(lab)
    vi = {o: i for i, o in enumerate(A.observation_id)}
    fv = lab.obs.map(vi).values.astype(np.int64)
    # windows
    shutil.copyfile(ANN, stg / 'ann.csv')
    obs_all = pd.read_csv(stg / 'ann.csv', usecols=['observation_id'])['observation_id']
    nfr = obs_all.value_counts()
    wstart = MiceDomain().window_start(nfr.loc[A.observation_id].values, A.stage.values)
    inwin = lab.frame_idx.values >= wstart[fv]
    nwin = np.bincount(fv[inwin], minlength=len(A))
    assert (nwin == 900).all(), np.unique(nwin)
    for f in ('dark_count.npy', 'regions.npz'):
        shutil.copyfile(OUT / 'prep' / f, stg / f)
    dark = np.load(stg / 'dark_count.npy')
    R = np.load(stg / 'regions.npz')
    assert (R['obs'] == A.observation_id.values).all()
    regs = {mg: R[f'reg_m{mg}'] for mg in MARGINS}
    # generous sensitivity zone 'G': the margin-2 zone plus every odour-corner-quadrant patch outside the arena box
    # plus arena patches whose centre is within 3 patches of the arena corner on the odour side
    oc = pd.read_csv(ODOR).set_index('observation_id')
    cc = (np.arange(GRID) + 0.5) * PX
    yy, xx = np.meshgrid(cc, cc, indexing='ij')
    gz = regs[2].copy()
    for i, o in enumerate(A.observation_id):
        r = oc.loc[o]
        inside = (yy >= r.arena_r0) & (yy <= r.arena_r1) & (xx >= r.arena_c0) & (xx <= r.arena_c1)
        near = np.hypot(yy / PX - r.corner_y_patch, xx / PX - r.corner_x_patch) <= 3
        z_ = (regs[2][i] == 0) | (quadrant(r.odor_corner) & ~inside).ravel() | near.ravel()
        gz[i][z_] = 0
    regs['G'] = gz
    rot = R['rot'].astype(np.int64)
    G = np.arange(GRID * GRID).reshape(GRID, GRID)
    ROT = np.stack([np.rot90(G, k).ravel() for k in range(4)])  # aligned position -> raw position, per turn count
    prep = json.loads((OUT / 'prep' / 'prep.json').read_text())
    enc = json.loads((OUT / 'tokens' / 'encode_checks.json').read_text())
    res = {'prep': prep, 'encode_checks': enc, 'stores': {}, 'candidates': {}}
    win_f = inwin.astype(np.float64)
    per_store = {}
    PV = A[['observation_id', 'pool', 'stage', 'half', 'nose_nose_rate', 'nose_nose_bpm', 'nose_tail_rate',
            'nose_tail_bpm']].copy()
    for store in ('raw', 'al'):
        for f in (f'{store}_codes.npy', f'{store}_pos.npy', f'{store}_lens.npy', f'{store}_cols.json'):
            shutil.copyfile(OUT / 'tokens' / f, stg / f)
        Z = np.load(stg / f'{store}_codes.npy')
        pos = np.load(stg / f'{store}_pos.npy').astype(np.int64)
        lens = np.load(stg / f'{store}_lens.npy').astype(np.int64)
        cols = [tuple(c) for c in json.loads((stg / f'{store}_cols.json').read_text())]
        assert len(lens) == F and lens.sum() == len(pos) == len(Z)
        tf = np.repeat(np.arange(F), lens)
        tv = fv[tf]
        rawpos = pos if store == 'raw' else ROT[rot[tv], pos]
        treg = {mg: regs[mg][tv, rawpos] for mg in regs}
        tmouse = dark[tf, rawpos] >= DARK_MIN
        # the dark cue is part of the foreground rule: every dark-core patch should be a stored foreground token
        core_n = (dark >= DARK_MIN).sum(1)
        cover = float(tmouse.sum() / max(core_n.sum(), 1))
        # foreground counts per region per frame -> per-video means in the window
        fg = {}
        for name, msk in (('all', np.ones(len(tf), bool)), ('mouse', tmouse)):
            c = np.bincount(tf[msk] * 4 + treg[BAG_MARGIN][msk], minlength=F * 4).reshape(F, 4).astype(np.float64)
            fg[name] = np.stack([np.bincount(fv, weights=c[:, k] * win_f, minlength=len(A)) / 900 for k in range(4)], 1)
        per_store[store] = {'fg': fg}
        st_info = {'tokens': int(len(tf)), 'tokens_per_frame': float(len(tf) / F), 'dark_core_covered_by_fg': cover,
                   'mouse_like_token_frac': float(tmouse.mean()),
                   'token_region_frac': {REG[k]: float((treg[BAG_MARGIN] == k).mean()) for k in range(4)},
                   'token_frac_in_zone_G': float((treg['G'] == 0).mean()),
                   'mouse_like_frac_by_region': {REG[k]: float(tmouse[treg[BAG_MARGIN] == k].mean())
                                                 for k in range(4) if (treg[BAG_MARGIN] == k).any()}}
        if store == 'raw':
            nc, largest, _, _ = frame_components(pos, lens)
            per_store['raw']['ncomp'] = np.bincount(fv, weights=nc * win_f, minlength=len(A)) / 900
            per_store['raw']['largest'] = np.bincount(fv, weights=largest * win_f, minlength=len(A)) / 900
        res['stores'][store] = st_info
        log(f'{store}: {json.dumps(st_info)} ({time.time() - t0:.0f}s)')
        nzf = lens > 0
        st = np.r_[0, np.cumsum(lens)][:-1]

        def fmax(z):
            o = np.zeros(F, np.float64)
            o[nzf] = np.maximum.reduceat(z, st[nzf])
            return o

        def vmean(fr):
            return np.bincount(fv, weights=fr * win_f, minlength=len(A)) / 900
        for cname, (mk, j, b, refcol) in CANDS.items():
            if MODELS[mk][0] != store:
                continue
            z = Z[:, cols.index((mk, j))].astype(np.float32)
            m_all = fmax(z)
            # region (and mouse-likeness) of the frame's argmax token
            ismax = np.flatnonzero((z == m_all[tf].astype(np.float32)) & (z > 0))
            fr_, first = np.unique(tf[ismax], return_index=True)
            tok_ = ismax[first]
            freg = {mg: np.full(F, -1, np.int64) for mg in regs}
            for mg in regs:
                freg[mg][fr_] = treg[mg][tok_]
            fmouse = np.zeros(F, bool)
            fmouse[fr_] = tmouse[tok_]
            contrib = {mg: np.stack([vmean(np.where(freg[mg] == k, m_all, 0)) for k in range(4)], 1) for mg in regs}
            contrib_mouse = np.stack([vmean(np.where(fmouse, m_all, 0)), vmean(np.where(~fmouse, m_all, 0))], 1)
            reads = {'all': m_all, 'noBAG': fmax(np.where(treg[2] != 0, z, 0)), 'noBAG1': fmax(np.where(treg[1] != 0, z, 0)),
                     'mouse': fmax(np.where(tmouse, z, 0)), 'mouse_noBAG': fmax(np.where(tmouse & (treg[2] != 0), z, 0)),
                     'noBAGG': fmax(np.where(treg['G'] != 0, z, 0))}
            x_all = vmean(m_all)
            E = {'model': mk, 'neuron': j, 'store': store, 'behaviour': b,
                 'argmax_region_frac_frames': {REG[k]: float((freg[BAG_MARGIN][inwin & (m_all > 0)] == k).mean())
                                               for k in range(4)},
                 'argmax_mouse_like_frac': float(fmouse[inwin & (m_all > 0)].mean()),
                 'firing_frame_frac': float((m_all[inwin] > 0).mean())}
            if refcol:
                ref = A[refcol].values
                E['reproduce_per_video_mean'] = {'max_abs_diff': float(np.abs(x_all - ref).max()),
                                                 'pearson': float(np.corrcoef(x_all, ref)[0, 1])}
            # video-level reads
            E['reads'] = {}
            for vname, fr in reads.items():
                xv = vmean(fr)
                vm, sds, (pools, ph, ix) = video_metrics(xv, A, b)
                E['reads'][vname] = vm
                PV[f'{cname}|{vname}'] = xv
                if vname == 'all':
                    base_sds, geo = sds, (pools, ph, ix)
            # decomposition of the unmasked switch change (dz units; additive)
            pools, ph, ix = geo
            dtot = std_deltas(x_all, A, base_sds, pools, ph, ix)
            dec = {}
            for sw in SW:
                sd_ = np.std(dtot[sw], ddof=1)
                mt = dtot[sw].mean()
                e = {'dz_total': float(mt / sd_), 'mean_d_total_sdunits': float(mt)}
                for mg in regs:
                    for k in range(4):
                        dr_ = std_deltas(contrib[mg][:, k], A, base_sds, pools, ph, ix)[sw]
                        e[f'm{mg}|{REG[k]}'] = {'dz_contrib': float(dr_.mean() / sd_), 'share': float(dr_.mean() / mt),
                                                'own': ttest(dr_)}
                for k, nm in enumerate(('mouse_like', 'not_mouse_like')):
                    dr_ = std_deltas(contrib_mouse[:, k], A, base_sds, pools, ph, ix)[sw]
                    e[nm] = {'dz_contrib': float(dr_.mean() / sd_), 'share': float(dr_.mean() / mt)}
                dec[sw] = e
            E['decomposition'] = dec
            # correlates of the per-pool change (24 pools; plain units)
            dn = plain_deltas(x_all, pools, ix)
            cov = {'rate': A[f'{b}_rate'].values, 'bpm': A[f'{b}_bpm'].values,
                   'fg_total': per_store[store]['fg']['all'].sum(1), 'fg_BAG': per_store[store]['fg']['all'][:, 0],
                   'fg_CENTRE': per_store[store]['fg']['all'][:, 2], 'fg_WALL': per_store[store]['fg']['all'][:, 1],
                   'mouse_total': per_store[store]['fg']['mouse'].sum(1),
                   'n_components': per_store['raw']['ncomp'], 'largest_component': per_store['raw']['largest']}
            E['change_correlates'] = {}
            for cn, cv in cov.items():
                dc = plain_deltas(cv, pools, ix)
                cen = lambda d_: np.concatenate([d_[sw] - d_[sw].mean() for sw in SW])  # noqa: E731
                E['change_correlates'][cn] = {'per_switch': {sw: corr(dn[sw], dc[sw])['pearson'] for sw in SW},
                                              'pooled_centred': corr(cen(dn), cen(dc))['pearson'],
                                              'video_r_144': corr(x_all, cv)['pearson']}
            # does the foreground patch count explain the neuron? (a) video level: residual of the per-video mean on
            # fg_total, its r with the annotated rate (per half, mean); (b) per switch: d_neuron = a + b d_fg_total
            # (d_fg standardised), a / sd(resid) = the change not explained by the count change (diag_video_level
            # ols_intercept); the same for the annotated rate as reference.
            fgt = cov['fg_total']
            rr = []
            for h in (0, 1):
                m = (A.half == h).values
                X_ = np.c_[np.ones(m.sum()), fgt[m]]
                res_ = x_all[m] - X_ @ np.linalg.lstsq(X_, x_all[m], rcond=None)[0]
                rr.append(corr(res_, A[f'{b}_rate'].values[m])['pearson'])
            dfg = std_deltas(fgt, A, {0: 1.0, 1: 1.0}, pools, ph, ix)
            drt = std_deltas(A[f'{b}_rate'].values, A, {0: 1.0, 1: 1.0}, pools, ph, ix)
            E['fg_count_control'] = {'video_r_rate_resid_on_fg_total': float(np.mean(rr)), 'per_half': rr,
                                     'switch': {sw: {'neuron': ols_intercept(dtot[sw], dfg[sw] / dfg[sw].std(ddof=1)),
                                                     'rate': ols_intercept(drt[sw], dfg[sw] / dfg[sw].std(ddof=1)),
                                                     'r_dneuron_dfg': corr(dtot[sw], dfg[sw])['pearson']} for sw in SW}}
            res['candidates'][cname] = E
            log(f'{cname}: done ({time.time() - t0:.0f}s)')
    # foreground count changes per region (raw store; the aligned store as a check)
    pools = sorted(A.pool.unique())
    ix = {(p, s): i for i, (p, s) in enumerate(zip(A.pool, A.stage))}
    fgc = {}
    for store in ('raw', 'al'):
        fgc[store] = {}
        for name in ('all', 'mouse'):
            M = per_store[store]['fg'][name]
            fgc[store][name] = {}
            for k in range(5):
                v = M.sum(1) if k == 4 else M[:, k]
                rn = 'TOTAL' if k == 4 else REG[k]
                d = plain_deltas(v, pools, ix)
                fgc[store][name][rn] = {'mean_per_frame': float(v.mean()),
                                        'by_stage': {int(s): float(v[(A.stage == s).values].mean()) for s in range(1, 7)},
                                        **{sw: ttest(d[sw]) for sw in SW}}
    for nm in ('ncomp', 'largest'):
        v = per_store['raw'][nm]
        d = plain_deltas(v, pools, ix)
        fgc['raw'][nm] = {'by_stage': {int(s): float(v[(A.stage == s).values].mean()) for s in range(1, 7)},
                          **{sw: ttest(d[sw]) for sw in SW}}
    res['fg_counts'] = fgc
    res['truth'] = {b: {q: {sw: ttest(plain_deltas(A[f'{b}_{q}'].values, pools, ix)[sw]) for sw in SW}
                        for q in ('rate', 'bpm')} for b in ('nose_nose', 'nose_tail')}
    # verdicts
    V = {}
    for cname, E in res['candidates'].items():
        b = E['behaviour']
        tz = res['truth'][b]['rate']['social_O>P']['dz']
        dec = E['decomposition']
        base_r = E['reads']['all']['video_r_rate']
        v = {'bag_share_social_O>P': dec['social_O>P'][f'm{BAG_MARGIN}|BAG']['share'],
             'bag_share_social_H>O': dec['social_H>O'][f'm{BAG_MARGIN}|BAG']['share'],
             'bagG_share_social_O>P': dec['social_O>P']['mG|BAG']['share'],
             'bagG_share_social_H>O': dec['social_H>O']['mG|BAG']['share']}
        v['bag_driven_G'] = bool(v['bagG_share_social_O>P'] >= 0.5 and v['bagG_share_social_H>O'] >= 0.5)
        v['bag_driven'] = bool(v['bag_share_social_O>P'] >= 0.5 and v['bag_share_social_H>O'] >= 0.5)
        v['masking'] = {}
        for vn in VARIANTS[1:]:
            rd = E['reads'][vn]
            dz = rd['effects']['social_O>P']['neuron']['dz']
            v['masking'][vn] = {'social_O>P_dz': dz, 'annotated_rate_dz': tz, 'dz_within_0.5': bool(abs(dz - tz) <= 0.5),
                                'video_r_rate': rd['video_r_rate'], 'video_r_gain': rd['video_r_rate'] - base_r,
                                'r_gain_ok': bool(rd['video_r_rate'] - base_r >= 0.15)}
            v['masking'][vn]['fixes'] = bool(v['masking'][vn]['dz_within_0.5'] and v['masking'][vn]['r_gain_ok'])
        v['masking_fixes_any'] = any(m['fixes'] for m in v['masking'].values())
        V[cname] = v
    res['verdicts'] = V
    res['notes'] = {'bag_margin_primary': BAG_MARGIN, 'wall_width_patches': WALL_W, 'wall_s': round(time.time() - t0, 1),
                    'decomposition': 'frame max attributed to the region of its argmax token (first token on ties); '
                                     'dz_contrib sums over regions to dz_total; share = mean(d_region)/mean(d_total)'}
    for store in ('raw', 'al'):
        for k in range(4):
            PV[f'fg_{store}_{REG[k]}'] = per_store[store]['fg']['all'][:, k]
            PV[f'fgmouse_{store}_{REG[k]}'] = per_store[store]['fg']['mouse'][:, k]
    PV['n_components'], PV['largest_component'] = per_store['raw']['ncomp'], per_store['raw']['largest']
    PV.to_parquet(OUT / 'per_video.parquet')
    (OUT / 'results.json').write_text(json.dumps(res, indent=1, default=float))
    shutil.rmtree(stg, ignore_errors=True)
    log(f'wrote {OUT / "results.json"} ({time.time() - t0:.0f}s)')
    summary()


def summary():
    res = json.loads((OUT / 'results.json').read_text())
    L = ['# Where do the contact neurons\' switch reactions come from? (scripts/eci/diag_switch_regions.py)\n']
    S = pd.DataFrame(res['prep']['sessions'])
    L.append(f"## Bag location ({len(S)} sessions)\n")
    L.append(f"D_O (cos distance of background tokens, O vs H) max in the odour-corner quadrant: median "
             f"{S.DO_max_corner_quadrant.median():.3f} [{S.DO_max_corner_quadrant.min():.3f}..{S.DO_max_corner_quadrant.max():.3f}]"
             f"; in the opposite quadrant {S.DO_max_opposite_quadrant.median():.3f}; corner > opposite in "
             f"{(S.DO_max_corner_quadrant > S.DO_max_opposite_quadrant).sum()}/{len(S)}. Core cells median "
             f"{S.core_cells.median():.0f} [{S.core_cells.min()}..{S.core_cells.max()}], share outside the arena box "
             f"{S.core_frac_outside_arena.mean():.2f}, centroid distance to the arena corner median "
             f"{S.core_dist_to_arena_corner_patch.median():.1f} patches. In the core: D(O,H) {S.DO_core_mean.mean():.3f}, "
             f"D(P,H) {S.DP_core_mean.mean():.3f}, D(P,O) {S.DPO_core_mean.mean():.3f}; mean grey H/O/P "
             + '/'.join(f'{np.mean([s[k] for s in S.pix_core_mean_HOP]):.0f}' for k in range(3))
             + f". Cells per region (margin 2): {json.dumps({k: round(v, 1) for k, v in res['prep']['cells_per_region']['2'].items()})}.\n")
    L.append(f"Dark-cue recompute vs background files: {json.dumps(res['prep']['dark_check'])}\n")
    for st, e in res['stores'].items():
        L.append(f"Store {st}: {e['tokens_per_frame']:.1f} fg tokens/frame, dark-core patches covered by the fg mask "
                 f"{e['dark_core_covered_by_fg']:.4f}, mouse-like token share {e['mouse_like_token_frac']:.3f}, token share by "
                 f"region {json.dumps({k: round(v, 3) for k, v in e['token_region_frac'].items()})}, mouse-like share by region "
                 f"{json.dumps({k: round(v, 3) for k, v in e['mouse_like_frac_by_region'].items()})}\n")
    L.append('## Encode checks (per-frame max of the re-encoded tokens vs the stored frame codes)\n')
    for k, e in res['encode_checks'].items():
        if k == 'cell_means':
            L.append(f"- CellMeans equal to sae_levers: {e['equal']}")
        else:
            L.append(f"- {k} ({e['reference']}): max |diff| {e['max_abs_diff']:.4g}, equal in fp16 {e['frac_equal_fp16']:.4f}, "
                     f"within 1% {e['frac_within_1pct']:.4f}, r {e['pearson']:.5f}")
    L.append('')
    for c, E in res['candidates'].items():
        if 'reproduce_per_video_mean' in E:
            L.append(f"- {c}: per-video mean vs per_video.parquet max |diff| {E['reproduce_per_video_mean']['max_abs_diff']:.2e}")
    tr = res['truth']
    L.append('\n## Decomposition of the unmasked switch change (BAG margin 2; dz contributions sum to dz; share %)\n')
    L.append('| neuron | switch | dz (p) | BAG | WALL | CENTRE | OUTSIDE | BAG m1 / m0 / G share | mouse-like share | annot. rate dz | annot. bouts dz |')
    L.append('|---|---|---|---|---|---|---|---|---|---|---|')
    for c, E in res['candidates'].items():
        b = E['behaviour']
        for sw in SW:
            d = E['decomposition'][sw]
            n_ = E['reads']['all']['effects'][sw]['neuron']
            cells = [f"{d[f'm2|{r}']['dz_contrib']:+.2f} ({100 * d[f'm2|{r}']['share']:.0f}%)" for r in REG]
            L.append(f"| {c} | {sw} | {n_['dz']:+.2f} ({fmt_p(n_['p'])}) | " + ' | '.join(cells)
                     + f" | {100 * d['m1|BAG']['share']:.0f}% / {100 * d['m0|BAG']['share']:.0f}% / {100 * d['mG|BAG']['share']:.0f}% | "
                     f"{100 * d['mouse_like']['share']:.0f}% | {tr[b]['rate'][sw]['dz']:+.2f} | {tr[b]['bpm'][sw]['dz']:+.2f} |")
    L.append('\nArgmax-token region share of firing frames (window), mouse-like argmax share:\n')
    for c, E in res['candidates'].items():
        L.append(f"- {c}: {json.dumps({k: round(v, 3) for k, v in E['argmax_region_frac_frames'].items()})}, mouse-like "
                 f"{E['argmax_mouse_like_frac']:.3f}, firing frames {E['firing_frame_frac']:.3f}")
    L.append('\n## Foreground patches per frame by region (raw store): stage means and per-switch change dz (p)\n')
    L.append('| count | ' + ' | '.join(f'stage {s}' for s in range(1, 7)) + ' | ' + ' | '.join(SW) + ' |')
    L.append('|---|' + '---|' * (6 + len(SW)))
    fg = res['fg_counts']['raw']
    rows = [(f'{nm} {rn}', fg[nm][rn]) for nm in ('all', 'mouse') for rn in REG + ('TOTAL',)] + \
           [('n components', fg['ncomp']), ('largest component', fg['largest'])]
    for nm, e in rows:
        L.append(f'| {nm} | ' + ' | '.join(f"{e['by_stage'][str(s)]:.2f}" for s in range(1, 7)) + ' | '
                 + ' | '.join(f"{e[sw]['dz']:+.2f} ({fmt_p(e[sw]['p'])})" for sw in SW) + ' |')
    L.append('\n## Masked re-reads (video level, diag_video_level.py measurements)\n')
    L.append('| neuron | read | video r rate (h0/h1) | r bouts | pooled d-r | ' + ' | '.join(f'{s} dz (p)' for s in SW) + ' |')
    L.append('|---|---|---|---|---|' + '---|' * len(SW))
    for c, E in res['candidates'].items():
        b = E['behaviour']
        L.append(f"| {c} | ANNOTATED rate / bouts | | | | " + ' | '.join(
            f"{tr[b]['rate'][s]['dz']:+.2f} / {tr[b]['bpm'][s]['dz']:+.2f}" for s in SW) + ' |')
        for vn in VARIANTS:
            r = E['reads'][vn]
            cells = []
            for s in SW:
                f = r['effects'][s]
                cells.append(f"{f['neuron']['dz']:+.2f} ({fmt_p(f['neuron']['p'])}) {'=' if f['sign_agree_rate'] else 'X'}"
                             f"{'=' if f['sign_agree_bpm'] else 'X'}")
            L.append(f"| | {vn} | {r['video_r_rate']:.2f} ({r['halves']['0']['video_rate']:.2f}/{r['halves']['1']['video_rate']:.2f})"
                     f" | {r['video_r_bpm']:.2f} | {r['delta_r_pooled']:.2f} | " + ' | '.join(cells) + ' |')
    L.append('\nSign marks: first = agrees with the annotated rate change, second = with the annotated bouts/min change.\n')
    L.append('## Correlates of the per-pool change (unmasked; Pearson over pools, pooled = centred within switch, 96 pairs)\n')
    keys = list(next(iter(res['candidates'].values()))['change_correlates'])
    L.append('| neuron | ' + ' | '.join(keys) + ' |')
    L.append('|---|' + '---|' * len(keys))
    for c, E in res['candidates'].items():
        L.append(f'| {c} | ' + ' | '.join(f"{E['change_correlates'][k]['pooled_centred']:+.2f} (v {E['change_correlates'][k]['video_r_144']:+.2f})"
                                          for k in keys) + ' |')
    L.append('\n(v = Pearson over the 144 videos of the per-video means.)\n')
    L.append('## Foreground-count control: per-switch change not explained by the change in fg patches/frame '
             '(intercept a/sd (p) of d ~ a + b d_fg_total), and video r(rate) of the residual on fg_total\n')
    L.append('| neuron | resid video r rate | ' + ' | '.join(f'{s} neuron / annot. rate (r d-neuron,d-fg)' for s in SW) + ' |')
    L.append('|---|---|' + '---|' * len(SW))
    for c, E in res['candidates'].items():
        f = E['fg_count_control']
        L.append(f"| {c} | {f['video_r_rate_resid_on_fg_total']:.2f} | " + ' | '.join(
            f"{f['switch'][s]['neuron']['a_over_sd']:+.2f} ({fmt_p(f['switch'][s]['neuron']['p_a'])}) / "
            f"{f['switch'][s]['rate']['a_over_sd']:+.2f} ({fmt_p(f['switch'][s]['rate']['p_a'])}) ({f['switch'][s]['r_dneuron_dfg']:+.2f})"
            for s in SW) + ' |')
    L.append('')
    L.append('## Pre-registered readings\n')
    for c, v in res['verdicts'].items():
        mm = '; '.join(f"{vn}: dz {m['social_O>P_dz']:+.2f} vs {m['annotated_rate_dz']:+.2f} "
                       f"({'in' if m['dz_within_0.5'] else 'out'}), r {m['video_r_rate']:.2f} ({m['video_r_gain']:+.2f})"
                       f"{' FIXES' if m['fixes'] else ''}" for vn, m in v['masking'].items())
        L.append(f"- {c}: BAG share social O>P {100 * v['bag_share_social_O>P']:.0f}%, H>O {100 * v['bag_share_social_H>O']:.0f}% "
                 f"-> {'BAG-DRIVEN' if v['bag_driven'] else 'not bag-driven'} (generous zone G: O>P {100 * v['bagG_share_social_O>P']:.0f}%, "
                 f"H>O {100 * v['bagG_share_social_H>O']:.0f}% -> {'BAG-DRIVEN' if v['bag_driven_G'] else 'not'}); masking fixes it: "
                 f"{'YES' if v['masking_fixes_any'] else 'no'} ({mm})")
    (OUT / 'table.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['prep', 'encode', 'analyse', 'summary'])
    ap.add_argument('--stage', default=None, help='local staging dir (/localhome/$USER/$SLURM_JOB_ID)')
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    if a.step != 'summary' and not a.stage:
        raise SystemExit('--stage is required')
    OUT.mkdir(parents=True, exist_ok=True)
    {'prep': cmd_prep, 'encode': cmd_encode, 'analyse': cmd_analyse, 'summary': lambda _: summary()}[a.step](a)
